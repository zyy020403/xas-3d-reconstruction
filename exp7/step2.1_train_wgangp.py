"""
step2.1_train_wgangp.py — Exp7 WGAN-GP Full Training
纯 PyTorch 版（移除 pytorch_lightning，MA1 决议 2026-05-12）

超参全部锁定，不许改：
  device         : cuda:0
  max_epochs     : 500
  batch_size     : 32
  n_critic       : 5（Phase 1+ 标准值；Phase 0 由 MA1 决议降为 1，见 get_n_critic）
  G optimizer    : AdamW(lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-4)
  D optimizer    : AdamW(lr=4e-4, betas=(0.0, 0.9), weight_decay=1e-4)
  gradient_clip G: 1.0 (D 不 clip)
  lambda_gp      : 10.0
  lambda_pmin    : 1.0
  lambda_type    : 1.0
  mixed_precision: fp32

val_composite_ckpt_score (锁定):
  = val_cps_mean + 0.3 * val_pv_pass_rate

MA1 决议变更 (2026-05-13, mode collapse epoch=5 RAISE 后):
  1. get_n_critic: Phase 0 (epoch<50) n_critic=1, Phase 1+ n_critic=5
     依据: Proposal §附录B 第9条 风险1应对授权
  2. G_adv clamp min=-50.0: 防止 D 极度压制期梯度爆炸
     仅数值稳定保护，不改变训练目标
  3. mode_diversity epoch=5 阈值放宽至 0.002
     (Phase 0 n_critic=1 时 D/G 力量均衡，diversity 可维持)
  4. epoch=10 强制上报 MA1（确认模式稳定）
  5. VAL_EVERY=5, ES_PATIENCE=6 (≡ 30 epoch)
  6. collate_fn: None-safe（过滤 cache miss 样本）
  7. dual_delta raise 只在 epoch>=150 生效
"""

import os
import sys
import json
import pickle
import math
import logging
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

ROOT   = Path('/home/tcat/experiment7')
SHARED = ROOT / 'shared'
sys.path.insert(0, str(SHARED))

from cond_wgan_gp import (
    LocalStructureGenerator,
    LocalStructureDiscriminator,
    compute_gradient_penalty,
    LAMBDA_GP, N_CRITIC, NOISE_DIM, SPECTRUM_DIM, CENTER_EMB_DIM,
    N_ATOMS, N_NEIGHBOR_TYPES, NO_OBJECT_IDX, N_TYPES_WITH_NO_OBJ,
)
from spectrum_encoder import SpectrumEncoder
from curriculum_callbacks import get_curriculum_min_pdist
from xas_local_dataset_v2 import XasLocalDatasetV2
from torch.utils.data import DataLoader

# ── 绝对路径常量 ────────────────────────────────────────────────────────────
SHELL_BOUNDARIES_PATH = '/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl'
VOCAB_PATH            = str(SHARED / 'exp7_element_vocab.json')
BASELINE_DUAL_PATH    = str(SHARED / 'baseline_dual.json')
DATA_DIR              = '/home/tcat/diffcsp_exp5_prime/data'
LOG_DIR               = str(ROOT / 'logs')
CKPT_DIR              = str(ROOT / 'checkpoints' / 'step2')

# ── 锁定超参 ────────────────────────────────────────────────────────────────
MIN_PDIST_CALIBRATED  = 1.5075718402862548
BOUNDARIES            = [50, 100, 150]
FRACTIONS             = [0.33, 0.53, 0.73, 1.00]
LAMBDA_PMIN   = 1.0
LAMBDA_TYPE   = 1.0
MAX_EPOCHS    = 500
BATCH_SIZE    = 32
GRAD_CLIP_G   = 1.0
DEVICE        = torch.device('cuda:0')
G_ADV_CLAMP_MIN = -50.0   # MA1 决议: 数值稳定保护，不改变训练目标

# VAL_EVERY=5, ES_PATIENCE=6 × 5 = 30 epoch (规格等价)
VAL_EVERY        = 5
ES_PATIENCE      = 6
ES_START_EPOCH   = 150
CKPT_START_EPOCH = 150

# ── raise 阈值 ──────────────────────────────────────────────────────────────
RAISE_DUAL_DELTA    = 0.50
RAISE_COLLAPSE_RATE = 0.20
RAISE_GP_HIGH       = 5.0
RAISE_MODE_DIV_EP5  = 0.002   # MA1 决议: epoch=5 放宽至 0.002
RAISE_MODE_DIV_EP30 = 0.005   # 原规格: epoch<=30 阈值
RAISE_MODE_DIV_EP50 = 0.003   # 原规格: epoch<=50 阈值

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── logging ─────────────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(LOG_DIR, f'train_{ts}.log')
log_f    = open(log_path, 'w', buffering=1)

def tprint(s):
    print(s, flush=True)
    log_f.write(s + '\n')

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


# ═══════════════════════════════════════════════════════════════════════════
# MA1 决议: n_critic 按 phase 动态取
# Phase 0 (epoch < 50): n_critic=1，防止 D 过强压制 G
# Phase 1+ (epoch >= 50): n_critic=5，恢复标准值
# ═══════════════════════════════════════════════════════════════════════════
def get_n_critic(epoch: int) -> int:
    return 1 if epoch < BOUNDARIES[0] else N_CRITIC


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise min-distance penalty — min-image fold (锁定公式 §5.5.3)
# ═══════════════════════════════════════════════════════════════════════════
def pairwise_min_distance_penalty_minimage(pred_frac_coords, min_pdist, L=20.0):
    B, N, _ = pred_frac_coords.shape
    diff  = pred_frac_coords.unsqueeze(2) - pred_frac_coords.unsqueeze(1)
    diff  = diff - torch.round(diff)
    cart  = diff * L
    dists = cart.norm(dim=-1)
    eye   = torch.eye(N, dtype=torch.bool, device=pred_frac_coords.device)
    dists = dists.masked_fill(eye.unsqueeze(0), float('inf'))
    violation = F.relu(min_pdist - dists)
    return (violation ** 2).mean()


def pairwise_violation_rate(pred_frac_coords, threshold, L=20.0):
    B, N, _ = pred_frac_coords.shape
    diff  = pred_frac_coords.unsqueeze(2) - pred_frac_coords.unsqueeze(1)
    diff  = diff - torch.round(diff)
    cart  = diff * L
    dists = cart.norm(dim=-1)
    eye   = torch.eye(N, dtype=torch.bool, device=pred_frac_coords.device)
    dists = dists.masked_fill(eye.unsqueeze(0), float('inf'))
    min_d = dists.flatten(1).min(dim=1).values
    return (min_d < threshold).float().mean().item()


def compute_distance_matrix(frac_coords, L=20.0):
    diff  = frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)
    diff  = diff - torch.round(diff)
    cart  = diff * L
    return cart.norm(dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Vocab helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_z_to_idx(vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {int(k): int(v) for k, v in vocab['neighbor']['Z_to_idx'].items()}


def load_neighbor_idx_to_Z(vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {int(v): int(k) for k, v in vocab['neighbor']['Z_to_idx'].items()}


def atom_types_Z_to_idx(atom_types_Z, z_to_idx, no_object_idx=88):
    idx = torch.full_like(atom_types_Z, no_object_idx)
    for b in range(atom_types_Z.shape[0]):
        for n in range(atom_types_Z.shape[1]):
            z = int(atom_types_Z[b, n].item())
            idx[b, n] = z_to_idx.get(z, no_object_idx)
    return idx.clamp(0, no_object_idx)


# ═══════════════════════════════════════════════════════════════════════════
# D input builders
# ═══════════════════════════════════════════════════════════════════════════
def build_d_input_real(frac_coords, atom_types_Z, z_to_idx, device):
    dist_mat    = compute_distance_matrix(frac_coords)
    cart        = frac_coords * 20.0
    dist_center = cart.norm(dim=-1)
    idx      = atom_types_Z_to_idx(atom_types_Z, z_to_idx)
    idx_d    = idx.clamp(0, N_NEIGHBOR_TYPES - 1)
    types_oh = torch.zeros(frac_coords.shape[0], N_ATOMS, N_NEIGHBOR_TYPES, device=device)
    types_oh.scatter_(2, idx_d.unsqueeze(-1), 1.0)
    return dist_mat, dist_center, types_oh


def build_d_input_fake(pred_coords, pred_logits):
    dist_mat    = compute_distance_matrix(pred_coords)
    cart        = pred_coords * 20.0
    dist_center = cart.norm(dim=-1)
    fake_idx = pred_logits.argmax(dim=-1).clamp(0, N_NEIGHBOR_TYPES - 1)
    types_oh = torch.zeros(*fake_idx.shape, N_NEIGHBOR_TYPES, device=pred_coords.device)
    types_oh.scatter_(2, fake_idx.unsqueeze(-1), 1.0)
    return dist_mat, dist_center, types_oh


# ═══════════════════════════════════════════════════════════════════════════
# collate_fn — None-safe
# ═══════════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    result = {}
    for k in batch[0].keys():
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            result[k] = torch.stack([s[k] for s in batch])
        elif isinstance(v0, bool):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.bool)
        elif isinstance(v0, int):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.long)
        elif isinstance(v0, float):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
        else:
            result[k] = [s[k] for s in batch]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# EarlyStopping + CheckpointManager
# ═══════════════════════════════════════════════════════════════════════════
class EarlyStoppingManual:
    def __init__(self, patience, start_epoch):
        self.patience    = patience
        self.start_epoch = start_epoch
        self.best        = -math.inf
        self.wait        = 0

    def step(self, epoch, value):
        if epoch < self.start_epoch:
            self.wait = 0
            return False
        if value > self.best:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


class CheckpointManager:
    def __init__(self, dirpath, start_epoch, save_top_k=3):
        self.dirpath      = dirpath
        self.start_epoch  = start_epoch
        self.save_top_k   = save_top_k
        self.best_k       = []

    def step(self, epoch, score, state):
        torch.save(state, os.path.join(self.dirpath, 'last.pt'))
        if epoch < self.start_epoch:
            return
        if (len(self.best_k) >= self.save_top_k and
                score <= min(s for s, _ in self.best_k)):
            return
        path = os.path.join(
            self.dirpath,
            f'exp7_epoch{epoch:04d}_score{score:.4f}.pt')
        torch.save(state, path)
        self.best_k.append((score, path))
        self.best_k.sort(key=lambda x: x[0], reverse=True)
        while len(self.best_k) > self.save_top_k:
            _, old = self.best_k.pop()
            if os.path.exists(old):
                os.remove(old)
        tprint(f"[Ckpt] saved epoch={epoch} score={score:.4f} → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    tprint("=" * 70)
    tprint("Exp7 WGAN-GP Full Training — 纯 PyTorch 版")
    tprint(f"MIN_PDIST_CALIBRATED={MIN_PDIST_CALIBRATED}")
    tprint(f"VAL_EVERY={VAL_EVERY}  ES_PATIENCE={ES_PATIENCE} "
           f"(≡ {ES_PATIENCE*VAL_EVERY} epoch)")
    tprint(f"[MA1决议] get_n_critic: Phase0={get_n_critic(0)}, Phase1+={get_n_critic(50)}")
    tprint(f"[MA1决议] G_adv clamp min={G_ADV_CLAMP_MIN}")
    tprint("=" * 70)

    # ── baseline gate ────────────────────────────────────────────────────
    with open(BASELINE_DUAL_PATH) as f:
        bdual = json.load(f)
    cps_val = bdual.get('exp5_prime', {}).get('CPS_val', None)
    if cps_val is None or isinstance(cps_val, str):
        raise RuntimeError(f"[RAISE] baseline_dual.json CPS_val 未填 ({cps_val})")
    tprint(f"Baseline CPS_val (Exp5') = {cps_val:.4f}  ✓ gate passed")

    # ── vocab + shell boundaries ─────────────────────────────────────────
    z_to_idx          = load_z_to_idx(VOCAB_PATH)
    neighbor_idx_to_Z = load_neighbor_idx_to_Z(VOCAB_PATH)
    with open(SHELL_BOUNDARIES_PATH, 'rb') as f:
        shell_boundaries = pickle.load(f)

    # ── dataset ──────────────────────────────────────────────────────────
    ds_train = XasLocalDatasetV2(split='train', data_dir=DATA_DIR,
                                  verbose_init_benchmark=False)
    ds_val   = XasLocalDatasetV2(split='val',   data_dir=DATA_DIR,
                                  verbose_init_benchmark=False)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn,
                              persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn,
                              persistent_workers=True)

    # ── models ───────────────────────────────────────────────────────────
    encoder       = SpectrumEncoder().to(DEVICE)
    generator     = LocalStructureGenerator().to(DEVICE)
    discriminator = LocalStructureDiscriminator().to(DEVICE)

    opt_G = torch.optim.AdamW(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-4)
    opt_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=4e-4, betas=(0.0, 0.9), weight_decay=1e-4)

    n_G = sum(p.numel() for p in generator.parameters())
    n_E = sum(p.numel() for p in encoder.parameters())
    n_D = sum(p.numel() for p in discriminator.parameters())
    tprint(f"Generator params:     {n_G:,}")
    tprint(f"Encoder params:       {n_E:,}")
    tprint(f"Discriminator params: {n_D:,}")
    tprint(f"D/(G+E) ratio: {n_D/(n_G+n_E):.3f}")

    # ── eval imports ─────────────────────────────────────────────────────
    from eval_cps import composite_physical_score, physical_validity, init_constants
    from eval_step5_3 import compute_one_sample
    init_constants()

    # ── checkpoint + early stopping ──────────────────────────────────────
    es   = EarlyStoppingManual(ES_PATIENCE, ES_START_EPOCH)
    ckpt = CheckpointManager(CKPT_DIR, CKPT_START_EPOCH)

    def get_state(epoch):
        return {
            'epoch':         epoch,
            'encoder':       encoder.state_dict(),
            'generator':     generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'opt_G':         opt_G.state_dict(),
            'opt_D':         opt_D.state_dict(),
        }

    fixed_spectra       = None
    collapse_high_count = 0
    gp_high_count       = 0
    last_val = dict(val_cps_mean=0.0, val_pv_pass_rate=0.0,
                    val_s53=0.0, val_collapse=0.0,
                    ckpt_score=0.0, dual_delta=0.0)

    # ═════════════════════════════════════════════════════════════════════
    # Training loop
    # ═════════════════════════════════════════════════════════════════════
    for epoch in range(MAX_EPOCHS):
        cur_pdist    = get_curriculum_min_pdist(epoch, MIN_PDIST_CALIBRATED)
        n_critic_now = get_n_critic(epoch)   # MA1 决议: Phase 0=1, Phase 1+=5

        if epoch in [0] + BOUNDARIES:
            phase = sum(1 for b in BOUNDARIES if epoch >= b)
            tprint(f"\n[Curriculum] epoch={epoch} Phase {phase}: "
                   f"min_pdist={cur_pdist:.4f} Å "
                   f"({FRACTIONS[phase]:.2f} × {MIN_PDIST_CALIBRATED:.4f}) "
                   f"n_critic={n_critic_now}")

        # ── train ────────────────────────────────────────────────────────
        encoder.train(); generator.train(); discriminator.train()
        metrics = dict(D_critic=[], D_gp=[], G_adv=[], G_pmin=[], G_type=[],
                       viol_curr=[], viol_full=[])

        for batch in train_loader:
            if batch is None:
                continue
            xmu  = batch['xmu'].to(DEVICE)
            chi1 = batch['chi1'].to(DEVICE)
            feff = batch['feff'].to(DEVICE)
            fc   = batch['frac_coords'].to(DEVICE)
            at   = batch['atom_types'].to(DEVICE)
            cz   = batch['center_element_Z']
            if isinstance(cz, list):
                cz = torch.tensor(cz, dtype=torch.long)
            cz = cz.to(DEVICE)
            B  = fc.shape[0]

            with torch.no_grad():
                enc_out = encoder(xmu, chi1, feff, cz)
            spec_lat = enc_out[:, :SPECTRUM_DIM]
            cen_emb  = enc_out[:, SPECTRUM_DIM:]

            # D steps × n_critic_now (MA1 决议: Phase 0=1, Phase 1+=5)
            d_critic_sum = 0.0
            gp_sum       = 0.0
            for _ in range(n_critic_now):
                z = torch.randn(B, NOISE_DIM, device=DEVICE)
                with torch.no_grad():
                    fake_coords, fake_logits = generator(z, spec_lat, cen_emb)
                real_dm, real_dc, real_to = build_d_input_real(fc, at, z_to_idx, DEVICE)
                fake_dm, fake_dc, fake_to = build_d_input_fake(fake_coords, fake_logits)
                d_real = discriminator(real_dm, real_dc, real_to, spec_lat.detach())
                d_fake = discriminator(fake_dm, fake_dc, fake_to, spec_lat.detach())
                gp = compute_gradient_penalty(
                    discriminator, real_dm, real_dc, real_to,
                    fake_dm, fake_dc, fake_to, spec_lat.detach(), DEVICE)
                loss_D = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                d_critic_sum += (d_fake.mean() - d_real.mean()).item()
                gp_sum       += gp.item()

            metrics['D_critic'].append(d_critic_sum / n_critic_now)
            metrics['D_gp'].append(gp_sum / n_critic_now)

            # G step × 1
            enc_out2  = encoder(xmu, chi1, feff, cz)
            spec_lat2 = enc_out2[:, :SPECTRUM_DIM]
            cen_emb2  = enc_out2[:, SPECTRUM_DIM:]
            z = torch.randn(B, NOISE_DIM, device=DEVICE)
            fake_coords, fake_logits = generator(z, spec_lat2, cen_emb2)
            fake_dm, fake_dc, fake_to = build_d_input_fake(fake_coords, fake_logits)
            d_fake_g = discriminator(fake_dm, fake_dc, fake_to, spec_lat2)

            # MA1 决议: G_adv clamp min=-50.0，防止 D 极度压制期梯度爆炸
            loss_adv  = torch.clamp(-d_fake_g.mean(), min=G_ADV_CLAMP_MIN)
            loss_pmin = LAMBDA_PMIN * pairwise_min_distance_penalty_minimage(
                fake_coords, cur_pdist)
            gt_idx    = atom_types_Z_to_idx(at, z_to_idx).to(DEVICE)
            loss_type = LAMBDA_TYPE * F.cross_entropy(
                fake_logits.reshape(-1, N_TYPES_WITH_NO_OBJ),
                gt_idx.reshape(-1))
            loss_G = loss_adv + loss_pmin + loss_type

            opt_G.zero_grad(); loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()),
                GRAD_CLIP_G)
            opt_G.step()

            for name, val in [('loss_G', loss_G), ('loss_adv', loss_adv),
                               ('loss_pmin', loss_pmin), ('loss_type', loss_type)]:
                if not math.isfinite(val.item()):
                    raise RuntimeError(
                        f"[RAISE] NaN/Inf: {name}={val.item()} epoch={epoch}")

            metrics['G_adv'].append(loss_adv.item())
            metrics['G_pmin'].append(loss_pmin.item())
            metrics['G_type'].append(loss_type.item())
            with torch.no_grad():
                metrics['viol_curr'].append(
                    pairwise_violation_rate(fake_coords, cur_pdist))
                metrics['viol_full'].append(
                    pairwise_violation_rate(fake_coords, MIN_PDIST_CALIBRATED))

        gp_avg = sum(metrics['D_gp']) / len(metrics['D_gp'])
        if gp_avg > RAISE_GP_HIGH:
            gp_high_count += 1
            if gp_high_count >= 5:
                raise RuntimeError(
                    f"[RAISE] D_gp={gp_avg:.4f} > {RAISE_GP_HIGH} "
                    f"连续 {gp_high_count} epoch — 上报 MA1")
        else:
            gp_high_count = 0

        m = {k: sum(v)/len(v) for k, v in metrics.items()}
        do_val = (epoch % VAL_EVERY == 0)

        # ── val (每 VAL_EVERY epoch) ──────────────────────────────────────
        if do_val:
            encoder.eval(); generator.eval()
            cps_scores_all   = []
            pv_pass_all      = []
            s53_scores_all   = []
            s53_collapse_all = []

            with torch.no_grad():
                for bidx, batch in enumerate(val_loader):
                    if batch is None:
                        continue
                    xmu  = batch['xmu'].to(DEVICE)
                    chi1 = batch['chi1'].to(DEVICE)
                    feff = batch['feff'].to(DEVICE)
                    fc   = batch['frac_coords'].to(DEVICE)
                    at   = batch['atom_types'].to(DEVICE)
                    cz   = batch['center_element_Z']
                    if isinstance(cz, list):
                        cz = torch.tensor(cz, dtype=torch.long)
                    cz     = cz.to(DEVICE)
                    snames = batch['sample_name']
                    B = fc.shape[0]

                    enc_out  = encoder(xmu, chi1, feff, cz)
                    spec_lat = enc_out[:, :SPECTRUM_DIM]
                    cen_emb  = enc_out[:, SPECTRUM_DIM:]
                    z        = torch.randn(B, NOISE_DIM, device=DEVICE)
                    pred_coords, pred_logits = generator(z, spec_lat, cen_emb)
                    pred_types = pred_logits.argmax(dim=-1)

                    if fixed_spectra is None and bidx == 0:
                        fixed_spectra = []
                        for i in range(min(5, B)):
                            fixed_spectra.append((
                                spec_lat[i:i+1].cpu(),
                                cen_emb[i:i+1].cpu(),
                            ))

                    for i in range(B):
                        sname = snames[i]
                        try:
                            score, bd = composite_physical_score(
                                pred_coords[i].cpu(), pred_types[i].cpu(),
                                sname, 20.0, neighbor_idx_to_Z)
                            pv = physical_validity(pred_coords[i].cpu(), 20.0)
                            cps_scores_all.append(score)
                            pv_pass_all.append(float(pv))
                        except Exception:
                            cps_scores_all.append(0.0)
                            pv_pass_all.append(0.0)

                        try:
                            sb_i = shell_boundaries.get(sname, None)
                            if sb_i is not None:
                                eval_cutoff = float(sb_i.get('eval_cutoff', 6.0))
                                row = compute_one_sample(
                                    sname, pred_coords[i].cpu(), pred_types[i].cpu(),
                                    eval_cutoff, 20.0, sb_i)
                                s53_scores_all.append(float(row.get('total_score', 0.0)))
                                s53_collapse_all.append(float(row.get('collapse', 0)))
                        except Exception:
                            s53_scores_all.append(0.0)
                            s53_collapse_all.append(0.0)

            val_cps_mean     = sum(cps_scores_all) / max(len(cps_scores_all), 1)
            val_pv_pass_rate = sum(pv_pass_all)    / max(len(pv_pass_all), 1)
            val_s53          = sum(s53_scores_all) / max(len(s53_scores_all), 1)
            val_collapse     = sum(s53_collapse_all) / max(len(s53_collapse_all), 1)
            ckpt_score       = val_cps_mean + 0.3 * val_pv_pass_rate
            denom            = max(val_cps_mean, val_s53, 1e-8)
            dual_delta       = abs(val_cps_mean - val_s53) / denom

            if val_collapse > RAISE_COLLAPSE_RATE:
                collapse_high_count += 1
                if collapse_high_count >= 3:
                    raise RuntimeError(
                        f"[RAISE] collapse_rate={val_collapse:.3f} > "
                        f"{RAISE_COLLAPSE_RATE} 连续 {collapse_high_count} val — 上报 MA1")
            else:
                collapse_high_count = 0

            # dual_delta raise 只在 epoch >= 150 生效
            if dual_delta > RAISE_DUAL_DELTA and epoch >= 150:
                raise RuntimeError(
                    f"[RAISE] dual_delta={dual_delta:.1%} > {RAISE_DUAL_DELTA:.0%} "
                    f"CPS={val_cps_mean:.4f} s53={val_s53:.4f} epoch={epoch} — 上报 MA1")

            # mode diversity（与 val 同步）
            mean_div = None
            if fixed_spectra is not None:
                divs = []
                with torch.no_grad():
                    for sl_cpu, ce_cpu in fixed_spectra:
                        sl = sl_cpu.to(DEVICE); ce = ce_cpu.to(DEVICE)
                        preds = []
                        for _ in range(10):
                            z = torch.randn(1, NOISE_DIM, device=DEVICE)
                            coords, _ = generator(z, sl, ce)
                            preds.append(coords)
                        preds_cat = torch.cat(preds, dim=0)
                        divs.append(preds_cat.std(dim=0).mean().item())
                mean_div = sum(divs) / len(divs)
                tprint(f"[Mode Diversity] epoch={epoch}: mean={mean_div:.5f} "
                       f"per={[f'{d:.5f}' for d in divs]}")

                # MA1 决议: epoch=5 阈值 0.002，其余原规格
                if epoch == 5 and mean_div < RAISE_MODE_DIV_EP5:
                    raise RuntimeError(
                        f"[RAISE] mode_diversity={mean_div:.5f} < "
                        f"{RAISE_MODE_DIV_EP5} epoch={epoch} — 上报 MA1")
                elif epoch <= 30 and epoch != 5 and mean_div < RAISE_MODE_DIV_EP30:
                    raise RuntimeError(
                        f"[RAISE] mode_diversity={mean_div:.5f} < "
                        f"{RAISE_MODE_DIV_EP30} epoch={epoch} — 上报 MA1")
                elif epoch <= 50 and mean_div < RAISE_MODE_DIV_EP50:
                    raise RuntimeError(
                        f"[RAISE] mode_diversity={mean_div:.5f} < "
                        f"{RAISE_MODE_DIV_EP50} epoch={epoch} — 上报 MA1")

            last_val = dict(val_cps_mean=val_cps_mean, val_pv_pass_rate=val_pv_pass_rate,
                            val_s53=val_s53, val_collapse=val_collapse,
                            ckpt_score=ckpt_score, dual_delta=dual_delta)

            ckpt.step(epoch, ckpt_score, get_state(epoch))
            if es.step(epoch, ckpt_score):
                tprint(f"[EarlyStopping] epoch={epoch} patience exhausted "
                       f"({ES_PATIENCE} val × {VAL_EVERY} = {ES_PATIENCE*VAL_EVERY} epoch), stop.")
                break

            # MA1 决议: epoch=10 强制上报（确认模式稳定）
            if epoch == 10:
                tprint(f"\n{'='*60}")
                tprint(f"[MA1 REPORT POINT — 模式稳定确认] epoch={epoch}")
                tprint(f"  n_critic_now          = {n_critic_now}")
                tprint(f"  G_adv (epoch mean)    = {m['G_adv']:+.4f}")
                tprint(f"  D_critic              = {m['D_critic']:+.4f}")
                tprint(f"  D_gp                  = {m['D_gp']:.4f}")
                tprint(f"  val_cps_mean          = {val_cps_mean:.4f}")
                tprint(f"  val_pv_pass_rate      = {val_pv_pass_rate:.1%}")
                tprint(f"  mode_diversity_mean   = {mean_div:.5f}" if mean_div else "  mode_diversity: N/A")
                tprint(f"  G_adv clamp 是否触发  = {'YES' if m['G_adv'] <= G_ADV_CLAMP_MIN + 1 else 'NO'}")
                tprint(f"{'='*60}\n")

            # phase milestone report
            if epoch in [50, 100, 150]:
                tprint(f"\n{'='*60}")
                tprint(f"[MA1 REPORT POINT] epoch={epoch}")
                tprint(f"  val_cps_mean          = {val_cps_mean:.4f}")
                tprint(f"  val_pv_pass_rate      = {val_pv_pass_rate:.1%}")
                tprint(f"  val_s53_composite     = {val_s53:.4f}")
                tprint(f"  val_collapse_rate     = {val_collapse:.1%}")
                tprint(f"  dual_delta_relative   = {dual_delta:.1%}")
                if mean_div is not None:
                    tprint(f"  mode_diversity_mean   = {mean_div:.5f}")
                if epoch == 150:
                    tprint(f"  [确认] epoch>=150: Ckpt + EarlyStop 已 enable")
                tprint(f"{'='*60}\n")

        # ── 每 epoch 打印 ─────────────────────────────────────────────────
        v   = last_val
        tag = "[VAL]" if do_val else "[trn]"
        tprint(
            f"epoch={epoch:03d} {tag} n_critic={n_critic_now} | "
            f"D_critic={m['D_critic']:+.3f} D_gp={m['D_gp']:.4f} | "
            f"G_adv={m['G_adv']:+.3f} G_pmin={m['G_pmin']:.5f} "
            f"G_type={m['G_type']:.4f} | "
            f"viol_curr={m['viol_curr']:.3f} viol_full={m['viol_full']:.3f} | "
            f"val_CPS={v['val_cps_mean']:.4f} val_PV={v['val_pv_pass_rate']:.1%} "
            f"val_s53={v['val_s53']:.4f} Δ={v['dual_delta']:.1%} | "
            f"ckpt={v['ckpt_score']:.4f} pdist={cur_pdist:.4f}"
        )

    log_f.close()
    tprint("Training complete.")


if __name__ == '__main__':
    main()
