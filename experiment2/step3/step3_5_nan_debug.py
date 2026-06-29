"""
Step 3.5 — NaN 诊断脚本
========================
逐步检查 forward() 内部每个计算步骤，找到 NaN 首次出现的位置。
"""

import os
os.environ.setdefault("PROJECT_ROOT", r"C:\Users\T-Cat\Desktop\DiffCSP-main")

import sys
DIFFCSP_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
sys.path.insert(0, os.path.join(DIFFCSP_ROOT, "experiment2", "step3"))
sys.path.insert(0, os.path.join(DIFFCSP_ROOT, "experiment2", "step2"))
sys.path.insert(0, DIFFCSP_ROOT)

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from xas_local_datamodule import XASDataModule
from diffusion_w_type_xas import CSPDiffusion, SinusoidalTimeEmbeddings, MAX_ATOMIC_NUM
from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
import torch.nn.functional as F

import hydra

STEP1_DIR = os.path.join(DIFFCSP_ROOT, "experiment2", "step1")
DATA_ROOT = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site")
FEFF_CSV  = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv")


def chk(name, t):
    """检查 tensor 是否含 NaN/Inf，打印统计。"""
    if isinstance(t, torch.Tensor):
        has_nan = t.isnan().any().item()
        has_inf = t.isinf().any().item()
        tag = "❌ NaN" if has_nan else ("❌ Inf" if has_inf else "✓")
        print(f"  {tag:8s}  {name:35s}  shape={tuple(t.shape)}  "
              f"min={t.float().min().item():.4f}  max={t.float().max().item():.4f}")
        return has_nan or has_inf
    else:
        print(f"  ✓         {name:35s}  value={t}")
        return False


# ── 加载 batch ────────────────────────────────────────────────────────────────
dm = XASDataModule(data_root=DATA_ROOT, step1_dir=STEP1_DIR,
                   feff_feat_csv=FEFF_CSV, batch_size=8, num_workers=0)
dm.setup('fit')
batch = next(iter(dm.train_dataloader()))
print(f"batch: {batch.num_graphs} samples, {batch.num_atoms.sum().item()} nodes\n")

# ── 初始化模型 ────────────────────────────────────────────────────────────────
cfg = OmegaConf.create({
    'cost_lattice': 0.0, 'cost_coord': 1.0, 'cost_type': 1.0,
    'time_dim': 256, 'spectrum_latent_dim': 256, 'latent_dim': 256,
    'xmu_dim': 150, 'chi_dim': 200, 'feat_dim': 73,
    'decoder': {
        '_target_': 'diffcsp.pl_modules.cspnet.CSPNet',
        'hidden_dim': 256, 'latent_dim': 512, 'num_layers': 6,
        'max_atoms': 100, 'act_fn': 'silu', 'dis_emb': 'sin',
        'num_freqs': 10, 'edge_style': 'fc', 'cutoff': 6.0,
        'max_neighbors': 20, 'ln': True, 'ip': True,
        'smooth': True, 'pred_type': True,
    },
    'beta_scheduler': {
        '_target_': 'diffcsp.pl_modules.diff_utils.BetaScheduler',
        'timesteps': 1000, 'scheduler_mode': 'cosine',
    },
    'sigma_scheduler': {
        '_target_': 'diffcsp.pl_modules.diff_utils.SigmaScheduler',
        'timesteps': 1000, 'sigma_begin': 0.005, 'sigma_end': 0.5,
    },
    'optim': {
        'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 1e-4},
        'use_lr_scheduler': False,
    },
})

model = CSPDiffusion(**cfg)
model.train()

print("=" * 60)
print("逐步 NaN 诊断")
print("=" * 60)

# ── 1. 输入字段 ───────────────────────────────────────────────────────────────
print("\n[1] 输入字段")
chk("batch.frac_coords",   batch.frac_coords)
chk("batch.atom_types",    batch.atom_types.float())
chk("batch.lengths",       batch.lengths)
chk("batch.angles",        batch.angles)
chk("batch.xmu_xanes",     batch.xmu_xanes)
chk("batch.chi1",          batch.chi1)
chk("batch.feff_features", batch.feff_features)
print(f"  batch.num_atoms = {batch.num_atoms}  (type={type(batch.num_atoms)})")
print(f"  batch.batch     = {batch.batch.shape}")

# ── 2. 时间嵌入 ───────────────────────────────────────────────────────────────
print("\n[2] 时间嵌入")
batch_size = batch.num_graphs
times    = model.beta_scheduler.uniform_sample_t(batch_size, 'cpu')
time_emb = model.time_embedding(times)
chk("times",    times.float())
chk("time_emb", time_emb)

# ── 3. 谱编码 ─────────────────────────────────────────────────────────────────
print("\n[3] SpectrumEncoder")
spectrum_cond = model.spectrum_encoder(
    batch.xmu_xanes, batch.chi1, batch.feff_features)
chk("spectrum_cond", spectrum_cond)

condition = torch.cat([time_emb, spectrum_cond], dim=-1)
chk("condition", condition)

# ── 4. 噪声调度器 ─────────────────────────────────────────────────────────────
print("\n[4] 噪声调度器")
alphas_cumprod = model.beta_scheduler.alphas_cumprod[times]
sigmas         = model.sigma_scheduler.sigmas[times]
sigmas_norm    = model.sigma_scheduler.sigmas_norm[times]
chk("alphas_cumprod", alphas_cumprod)
chk("sigmas",         sigmas)
chk("sigmas_norm",    sigmas_norm)

c0 = torch.sqrt(alphas_cumprod)
c1 = torch.sqrt(1. - alphas_cumprod)
chk("c0", c0)
chk("c1", c1)

# ── 5. 加噪 ───────────────────────────────────────────────────────────────────
print("\n[5] 加噪过程")
lattices    = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
frac_coords = batch.frac_coords
chk("lattices",    lattices)
chk("frac_coords", frac_coords)

rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
chk("rand_l", rand_l)
chk("rand_x", rand_x)

sigmas_per_atom      = sigmas.repeat_interleave(batch.num_atoms)[:, None]
sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
chk("sigmas_per_atom",      sigmas_per_atom)
chk("sigmas_norm_per_atom", sigmas_norm_per_atom)

input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.
chk("input_frac_coords", input_frac_coords)

gt_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
rand_t    = torch.randn_like(gt_onehot)
atom_type_probs = (
    c0.repeat_interleave(batch.num_atoms)[:, None] * gt_onehot
    + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
chk("gt_onehot",       gt_onehot)
chk("atom_type_probs", atom_type_probs)

input_lattice = lattices  # keep_lattice=True

# ── 6. Decoder ────────────────────────────────────────────────────────────────
print("\n[6] Decoder")
try:
    pred_l, pred_x, pred_t = model.decoder(
        condition, atom_type_probs, input_frac_coords, input_lattice,
        batch.num_atoms, batch.batch)
    chk("pred_l", pred_l)
    chk("pred_x", pred_x)
    chk("pred_t", pred_t)
except Exception as e:
    print(f"  ❌ Decoder 抛出异常: {e}")
    import traceback; traceback.print_exc()

# ── 7. 目标值 ─────────────────────────────────────────────────────────────────
print("\n[7] 目标值计算")
tar_x = d_log_p_wrapped_normal(
    sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
chk("tar_x", tar_x)