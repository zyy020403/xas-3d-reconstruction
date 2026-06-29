"""
diffusion_w_type_xas.py  (v3 — Step4c, [-0.5, 0.5] coordinate space)
                         (Exp5 SA1 patch — TypeClassifierHead + center conditioning)
======================================================================
Changes from Exp4 v3:
  Exp5 SA1 — three structural additions (EXP5_STEP1_HANDOFF §2):
    1. SpectrumEncoder.forward() now takes batch.center_element_Z and outputs
       (B, 272) = latent (256) ⊕ center_emb (16). Decoder input thus widens
       from 512 to 528 (time_emb 256 + spectrum 272). Decoder yaml latent_dim
       must be 528 — Exp4 ckpt's first decoder Linear (in=512) is shape-
       mismatched with Exp5 (in=528) and will be SKIPPED on strict=False load
       (random-init, intentional; SA2 phased training Notes §8.1 handles).
    2. New TypeClassifierHead: input (B, 272), output (B, 20, 100). Uses
       slot-aligned CE loss against batch.atom_types (pre-sorted by distance
       in dataset_v2). NO Hungarian during training (handoff §2.3).
    3. yaml-driven type_loss_mode flag with three modes (handoff §2.4):
         "diffusion_only": use Exp4 inner type MSE only
         "head_only":      use new head CE only (diffusion type still computed
                           but weight 0 — backbone gradient stays alive)
         "both":           sum both (default for SA2 first run)
       Default head/diffusion type weights: head_type_weight=0.5,
       diffusion_type_weight=1.0 (overrides Exp4 cost_type when set).

Changes from Exp2 v2 (preserved Exp4 behavior, unchanged here):
  1. forward(): removed % 1. from noisy_frac_coords computation
     → noisy coords stay in [-0.5, 0.5] neighbourhood, matching Dataset v5 output
  2. sample(): x_T prior changed from Uniform[0,1] to Uniform[-0.5, 0.5]
               internal % 1. fold removed from traj steps
               final min-image fold retained (clips values slightly outside [-0.5, 0.5])
  3. _density_loss(): unchanged — its internal % 1. fold is intentional (maps arbitrary
     range to [-0.5, 0.5]) and must NOT be removed.

Coordinate contract (v3, unchanged):
  Dataset v5      → frac_coords ∈ [-0.5, 0.5]
  forward()       → input_frac_coords ∈ [-0.5, 0.5] (no % 1. fold)
  sample() prior  → x_T ∈ [-0.5, 0.5]  (torch.rand - 0.5)
  sample() output → frac_coords ∈ [-0.5, 0.5] (min-image clip at final step)
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step2'))
from spectrum_encoder import SpectrumEncoder  # noqa: E402

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

import hydra
import pytorch_lightning as pl
from torch_scatter import scatter
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard,
    lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM = 100


# ── 基类 ─────────────────────────────────────────────────────────────────────

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(), _convert_="partial")
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": scheduler,
                "monitor": "val_loss"}


# ── 时间嵌入 ──────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ── Exp5 SA1: TypeClassifierHead ─────────────────────────────────────────────

class TypeClassifierHead(nn.Module):
    """
    Decoupled type classification head (EXP5_STEP1_HANDOFF §2.2 row 3).

    Takes the spectrum-with-center latent (B, 272) and predicts atom-type
    logits for all 20 neighbor slots simultaneously:
        Linear(272 → 512) → SiLU → Linear(512 → 20*100) → reshape (B, 20, 100)

    Slot-aligned with dataset_v2 output: logits[:, k, :] ↔ ground-truth slot k
    (atoms pre-sorted by distance to center). Training uses simple CE per slot;
    no Hungarian in the training loop (Hungarian only in eval/sample path,
    SA3's responsibility — see handoff §2.3).

    Output classes = MAX_ATOMIC_NUM = 100 (matches Exp4 inner one_hot dim;
    handoff §3.1 OPEN QUESTION about [1, 109] forward_test bound vs 100 inner —
    SA1 chooses 100 to align with diffusion inner; SA2 must verify dataset
    has no Z>100 neighbor atom — Exp4 trained successfully so this should hold).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512,
                 n_atoms: int = 20, n_elements: int = 100):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_elements = n_elements
        self.in_dim = in_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_atoms * n_elements),
        )

    def forward(self, latent_with_center: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_with_center : Tensor (B, in_dim)   — typically (B, 272)

        Returns
        -------
        Tensor (B, n_atoms, n_elements)  — typically (B, 20, 100)
        """
        B = latent_with_center.shape[0]
        flat = self.fc(latent_with_center)            # (B, n_atoms * n_elements)
        return flat.view(B, self.n_atoms, self.n_elements)


# ── 主模型 ────────────────────────────────────────────────────────────────────

class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim = self.hparams.latent_dim + self.hparams.time_dim,
            pred_type  = True,
            smooth     = True)
        self.beta_scheduler  = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim        = self.hparams.time_dim
        self.time_embedding  = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice    = self.hparams.cost_lattice < 1e-5
        self.keep_coords     = self.hparams.cost_coord  < 1e-5

        # ★ SpectrumEncoder (Exp5 SA1: now outputs (B, 272) = latent ⊕ center_emb)
        self.spectrum_encoder = SpectrumEncoder(
            xmu_dim    = self.hparams.get('xmu_dim',   150),
            chi_dim    = self.hparams.get('chi_dim',   200),
            feat_dim   = self.hparams.get('feat_dim',  74),
            latent_dim = self.hparams.get('spectrum_latent_dim', 256),
            n_center_elements = self.hparams.get('n_center_elements', 95),
            center_emb_dim    = self.hparams.get('center_emb_dim', 16),
        )
        # spectrum encoder full output dim (latent + center_emb)
        self._spectrum_out_dim = self.spectrum_encoder.output_dim   # 272

        # ★ density loss 权重（从 YAML 读取，默认 0.5）
        self.cost_density = float(self.hparams.get('cost_density', 0.5))

        # ── Exp5 SA1: TypeClassifierHead (decoupled) ──────────
        # Input dim = spectrum_encoder.output_dim (272 by default).
        # Output = (B, n_atoms=20, n_elements=100) logits.
        self.type_head = TypeClassifierHead(
            in_dim     = self._spectrum_out_dim,
            hidden_dim = int(self.hparams.get('type_head_hidden_dim', 512)),
            n_atoms    = int(self.hparams.get('n_atoms', 20)),
            n_elements = int(self.hparams.get('n_elements', MAX_ATOMIC_NUM)),
        )

        # ── Exp5 SA1: three-mode type-loss aggregation flags ──
        # diffusion_only: use Exp4 inner type MSE only (Exp4 baseline behavior)
        # head_only:      use new head CE only (diffusion type still computed for
        #                 backbone gradient continuity but weight-zeroed in total)
        # both:           sum both — default for SA2 first run
        self.type_loss_mode = str(self.hparams.get('type_loss_mode', 'both')).lower()
        if self.type_loss_mode not in ('diffusion_only', 'head_only', 'both'):
            raise ValueError(
                f"type_loss_mode must be one of "
                f"'diffusion_only'/'head_only'/'both', got {self.type_loss_mode!r}"
            )
        # Weights:
        #   diffusion_type_weight overrides legacy cost_type when explicitly set
        #   head_type_weight is new (no Exp4 analog)
        self.diffusion_type_weight = float(
            self.hparams.get('diffusion_type_weight',
                             self.hparams.get('cost_type', 1.0))
        )
        self.head_type_weight = float(self.hparams.get('head_type_weight', 0.5))

    # ─────────────────────────────────────────────────────────────────────────
    # ★ 密度正则：Tweedie 估算 x0，最小镜像后 L2 惩罚
    #   注意：此处的 % 1.0 折叠是"将任意范围的 x0_hat 映射到 [-0.5, 0.5]"
    #         与 forward() 中已删除的 % 1. 用途不同，必须保留。
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _density_loss(input_frac_coords, pred_x, sigmas_per_atom, sigmas_norm_per_atom):
        """
        用 Tweedie 公式从 (x_t, score) 估算去噪后的 x0_hat，
        再用最小镜像将其映射到 [-0.5, 0.5]，计算 L2 均值。
        这迫使模型预测的"干净位置"集中于 Fe 原点。
        """
        with torch.no_grad():
            sigma2     = sigmas_per_atom ** 2           # (N, 1)
            sqrt_norm  = torch.sqrt(sigmas_norm_per_atom)  # (N, 1)

        x0_hat = input_frac_coords + sigma2 * pred_x.detach() * sqrt_norm
        # 最小镜像：将 x0_hat 折叠到 [-0.5, 0.5]（保留，此处折叠是有意为之）
        x0_hat_mi = x0_hat % 1.0
        x0_hat_mi = x0_hat_mi - (x0_hat_mi > 0.5).float()
        # L2 惩罚（鼓励 x0_hat 集中于原点）
        return (x0_hat_mi ** 2).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # forward（训练）
    # ★ v3 改动：去掉 % 1.，noisy_frac_coords 保持在 [-0.5, 0.5] 附近
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch, _return_noisy_frac=False):
        """
        _return_noisy_frac : 调试用，True 时额外返回 input_frac_coords（用于开训前检查 2）
        """
        batch_size = batch.num_graphs
        times      = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb   = self.time_embedding(times)

        # Exp5 SA1: SpectrumEncoder now returns (B, 272) = latent ⊕ center_emb
        spectrum_cond = self.spectrum_encoder(
            batch.xmu_xanes,
            batch.chi1,
            batch.feff_features,
            batch.center_element_Z,   # Exp5 SA1
        )
        # condition dim = time_dim (256) + spectrum_out_dim (272) = 528
        condition = torch.cat([time_emb, spectrum_cond], dim=-1)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta           = self.beta_scheduler.betas[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas      = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l = torch.randn_like(lattices)
        rand_x = torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        sigmas_per_atom      = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        # ★★★ v3 核心改动：去掉 % 1.，保持坐标在 [-0.5, 0.5] 附近自然扩散 ★★★
        # v2（错误）：input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.
        # v3（正确）：
        input_frac_coords = frac_coords + sigmas_per_atom * rand_x

        # Exp5 SA1 fix (phase 6.5): cast onehot to model's working dtype, not fp32.
        # F.one_hot returns int64; .float() forces fp32 even when model is bf16,
        # which then causes "mat1 Float vs mat2 BFloat16" in cspnet's first
        # F.linear(atom_type_probs, weight). Use c0.dtype to follow the model's
        # current precision (fp32 in CPU phase 6.4, bf16 in GPU phase 6.5).
        gt_atom_types_onehot = F.one_hot(
            batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).to(c0.dtype)
        rand_t = torch.randn_like(gt_atom_types_onehot)
        atom_type_probs = (
            c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot
            + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, pred_t = self.decoder(
            condition,
            atom_type_probs, input_frac_coords, input_lattice,
            batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord   = F.mse_loss(pred_x, tar_x)
        loss_type    = F.mse_loss(pred_t, rand_t)   # diffusion-internal type MSE

        # ★ 密度正则损失
        loss_density = self._density_loss(
            input_frac_coords, pred_x, sigmas_per_atom, sigmas_norm_per_atom)

        # ── Exp5 SA1: head CE loss (slot-aligned, no Hungarian in training) ──
        # spectrum_cond is per-graph (B, 272); head outputs (B, 20, 100).
        # batch.atom_types is (B*20,) flat; reshape to (B, 20) for slot CE.
        head_logits = self.type_head(spectrum_cond)                      # (B, 20, n_elem)
        true_types_per_slot = batch.atom_types.view(batch_size, -1)      # (B, 20)
        # CE expects logits in (N, C) and targets (N,) with class indices in [0, C-1].
        # atom_types are 1-indexed Z values; we use Z-1 to align with one_hot logic above.
        loss_type_ce_head = F.cross_entropy(
            head_logits.reshape(-1, self.type_head.n_elements),          # (B*20, n_elem)
            (true_types_per_slot.reshape(-1) - 1).long(),                # (B*20,) class idx
        )

        # ── Exp5 SA1: three-mode type-loss aggregation ──
        if self.type_loss_mode == 'diffusion_only':
            loss_type_total = self.diffusion_type_weight * loss_type
        elif self.type_loss_mode == 'head_only':
            # head only enters total; diffusion-inner type still computed above
            # (kept alive for backbone gradient continuity per handoff §2.4 note)
            loss_type_total = self.head_type_weight * loss_type_ce_head
        else:  # 'both'
            loss_type_total = (self.diffusion_type_weight * loss_type
                               + self.head_type_weight * loss_type_ce_head)

        loss = (self.hparams.cost_lattice * loss_lattice
                + self.hparams.cost_coord  * loss_coord
                + loss_type_total                          # Exp5 SA1: replaces cost_type * loss_type
                + self.cost_density        * loss_density)

        output = {
            'loss':              loss,
            'loss_lattice':      loss_lattice,
            'loss_coord':        loss_coord,
            'loss_type':         loss_type,                # diffusion-internal MSE (back-compat name)
            'loss_density':      loss_density,
            # ── Exp5 SA1 additions ──
            'loss_diffusion_type': loss_type,              # alias for clarity
            'loss_type_ce_head':   loss_type_ce_head,      # head CE
            'loss_type_total':     loss_type_total,        # what enters total per current mode
        }
        if _return_noisy_frac:
            output['_input_frac_coords'] = input_frac_coords.detach()
        return output

    # ─────────────────────────────────────────────────────────────────────────
    # sample（推断）
    # ★ v3 改动：
    #   - x_T 先验改为 Uniform[-0.5, 0.5]（与训练分布一致）
    #   - 内部步骤去掉 % 1. 折叠
    #   - 最终最小镜像 clip 保留（处理轻微超界）
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-5):
        batch_size = batch.num_graphs

        # Exp5 SA1: spectrum_encoder now takes center_Z and outputs (B, 272)
        spectrum_cond = self.spectrum_encoder(
            batch.xmu_xanes,
            batch.chi1,
            batch.feff_features,
            batch.center_element_Z,   # Exp5 SA1
        )

        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        # ★★★ v3 改动：先验从 [0,1] 改为 [-0.5, 0.5]，与训练坐标系一致 ★★★
        # v2（错误）：x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        # v3（正确）：
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device) - 0.5
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        # ★★★ v3 改动：初始 traj 不再做 % 1. 折叠 ★★★
        traj = {self.beta_scheduler.timesteps: {
            'num_atoms':   batch.num_atoms,
            'atom_types':  t_T,
            'frac_coords': x_T,   # v2 此处有 x_T % 1.，v3 删除
            'lattices':    l_T,
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            times    = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            condition = torch.cat([time_emb, spectrum_cond], dim=-1)

            alphas         = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas         = self.beta_scheduler.sigmas[t]
            sigma_x        = self.sigma_scheduler.sigmas[t]
            sigma_norm     = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x     = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_t = self.decoder(
                condition, t_t, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = (x_t - step_size * pred_x + std_x * rand_x
                            if not self.keep_coords else x_t)
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt(
                (adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2))
                / (sigma_x ** 2))

            pred_l, pred_x, pred_t = self.decoder(
                condition, t_t_minus_05, x_t_minus_05, l_t_minus_05,
                batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = (x_t_minus_05 - step_size * pred_x + std_x * rand_x
                           if not self.keep_coords else x_t)
            l_t_minus_1 = (c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                           if not self.keep_lattice else l_t)
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            # ★★★ v3 改动：用最小镜像折叠替代旧的 % 1. 折叠 ★★★
            # v2（错误）：x_t_minus_1 % 1.     → 强制到 [0,1]，与训练坐标系不符
            # v3（正确）：x - round(x)          → 折叠到 [-0.5, 0.5]，与训练一致
            # 注意：此折叠是数值稳定性必须项，无约束时坐标会在 1000 步内漂移爆炸
            x_t_folded = x_t_minus_1 - torch.round(x_t_minus_1)
            traj[t - 1] = {
                'num_atoms':   batch.num_atoms,
                'atom_types':  t_t_minus_1,
                'frac_coords': x_t_folded,
                'lattices':    l_t_minus_1,
            }

        # ★ 最终输出：最小镜像 clip，确保轻微超界的坐标也落在 [-0.5, 0.5]
        # （此处保留：正常情况下 final_frac 已在 [-0.5, 0.5]，clip 只处理极少数边界值）
        final_frac    = traj[0]['frac_coords']
        final_frac_mi = final_frac.clone()
        final_frac_mi[final_frac_mi >  0.5] -= 1.0
        final_frac_mi[final_frac_mi < -0.5] += 1.0

        traj_stack = {
            'num_atoms':  batch.num_atoms,
            'atom_types': torch.stack([
                traj[i]['atom_types']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]).argmax(dim=-1) + 1,
            'all_frac_coords': torch.stack([
                traj[i]['frac_coords']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]),
            'all_lattices': torch.stack([
                traj[i]['lattices']
                for i in range(self.beta_scheduler.timesteps, -1, -1)
            ]),
        }

        traj[0] = {**traj[0], 'frac_coords': final_frac_mi}

        return traj[0], traj_stack

    # ─────────────────────────────────────────────────────────────────────────
    # Lightning hooks
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        # Exp5 SA1: log all four losses + total
        self.log_dict({
            'train_loss':              output_dict['loss'],
            'lattice_loss':            output_dict['loss_lattice'],
            'coord_loss':              output_dict['loss_coord'],
            'type_loss':               output_dict['loss_type'],            # diffusion-internal MSE
            'density_loss':            output_dict['loss_density'],
            # Exp5 SA1 additions
            'loss_diffusion_type':     output_dict['loss_diffusion_type'],
            'loss_type_ce_head':       output_dict['loss_type_ce_head'],
            'loss_type_total':         output_dict['loss_type_total'],
        }, on_step=True, on_epoch=True, prog_bar=True)
        if output_dict['loss'].isnan():
            return None
        return output_dict['loss']

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='test')
        self.log_dict(log_dict)
        return loss

    def compute_stats(self, output_dict, prefix):
        log_dict = {
            f'{prefix}_loss':              output_dict['loss'],
            f'{prefix}_lattice_loss':      output_dict['loss_lattice'],
            f'{prefix}_coord_loss':        output_dict['loss_coord'],
            f'{prefix}_type_loss':         output_dict['loss_type'],
            f'{prefix}_density_loss':      output_dict['loss_density'],
            # Exp5 SA1 additions
            f'{prefix}_loss_diffusion_type': output_dict['loss_diffusion_type'],
            f'{prefix}_loss_type_ce_head':   output_dict['loss_type_ce_head'],
            f'{prefix}_loss_type_total':     output_dict['loss_type_total'],
        }
        return log_dict, output_dict['loss']

    # ─────────────────────────────────────────────────────────────────────────
    # Exp5 SA1: head_predict_types — eval-time inference helper for SA3
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def head_predict_types(self, batch) -> torch.Tensor:
        """
        Eval-time helper: predict atom types from head only (no diffusion).

        SA3 uses this in evaluation/sample path to obtain per-slot Z predictions
        independent of the diffusion atom-type score. The slot ordering matches
        dataset_v2 (sorted by distance to center); SA3 must do its own
        Hungarian assignment against pred_coords for true TypeAcc reporting
        (handoff §2.3).

        Parameters
        ----------
        batch : PyG Batch with xmu_xanes / chi1 / feff_features / center_element_Z

        Returns
        -------
        LongTensor (B, n_atoms)  — argmax-decoded Z values, ∈ [1, n_elements]
                                   (we add 1 because head is trained on Z-1 indices)
        """
        was_training = self.training
        self.eval()
        try:
            spectrum_cond = self.spectrum_encoder(
                batch.xmu_xanes,
                batch.chi1,
                batch.feff_features,
                batch.center_element_Z,
            )
            head_logits = self.type_head(spectrum_cond)        # (B, n_atoms, n_elements)
            pred_classes = head_logits.argmax(dim=-1)          # (B, n_atoms), in [0, n_elem-1]
            return (pred_classes + 1).long()                   # back to 1-indexed Z
        finally:
            if was_training:
                self.train()