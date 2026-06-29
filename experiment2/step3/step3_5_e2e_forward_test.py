r"""
Step 3.5 — 端到端前向测试  (v2)
=================================
修改记录（v2）：
  - 在最顶部设置 PROJECT_ROOT 环境变量，避免 hydra 未初始化时的 import 报错
  - 改为用 OmegaConf.create() 在脚本内直接构建配置，完全不依赖 yaml 文件路径
    （前版本 cfg.cost_lattice 报 Missing key，根因是 yaml 路径未命中或结构异常）

运行方式：
  cd C:\Users\T-Cat\Desktop\DiffCSP-main
  python experiment2/step3/step3_5_e2e_forward_test.py
"""

# ── 必须在所有 import 之前设置 PROJECT_ROOT ──────────────────────────────────
import os
os.environ.setdefault(
    "PROJECT_ROOT",
    r"C:\Users\T-Cat\Desktop\DiffCSP-main")

import sys

DIFFCSP_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
STEP2_DIR    = os.path.join(DIFFCSP_ROOT, "experiment2", "step2")
STEP3_DIR    = os.path.join(DIFFCSP_ROOT, "experiment2", "step3")

sys.path.insert(0, STEP3_DIR)   # diffusion_w_type_xas.py
sys.path.insert(0, STEP2_DIR)   # spectrum_encoder.py
sys.path.insert(0, DIFFCSP_ROOT)

import torch
from omegaconf import OmegaConf, DictConfig
from torch_geometric.data import Batch

from xas_local_datamodule import XASDataModule
from diffusion_w_type_xas import CSPDiffusion
from diffcsp.common.data_utils import lattice_params_to_matrix_torch

STEP1_DIR = os.path.join(DIFFCSP_ROOT, "experiment2", "step1")
DATA_ROOT = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site")
FEFF_CSV  = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv")

# ── Step 1：加载一个 batch ────────────────────────────────────────────────────

print("=" * 60)
print("Step 3.5 — 端到端前向测试")
print("=" * 60)

dm = XASDataModule(
    data_root     = DATA_ROOT,
    step1_dir     = STEP1_DIR,
    feff_feat_csv = FEFF_CSV,
    batch_size    = 8,
    num_workers   = 0,
)
dm.setup('fit')

loader = dm.train_dataloader()
batch  = next(iter(loader))
assert batch is not None, "第一个 batch 为 None，请检查数据路径"

print(f"\n✓ batch 加载成功：{batch.num_graphs} 个样本，"
      f"{batch.num_atoms.sum().item()} 个节点")
print(f"  frac_coords : {batch.frac_coords.shape}")
print(f"  xmu_xanes   : {batch.xmu_xanes.shape}")
print(f"  lengths[0]  : {batch.lengths[0].tolist()}")


# ── Step 2：用 OmegaConf.create() 直接构建配置（不读文件）─────────────────────
# 这样完全规避 yaml 路径问题和 cfg.cost_lattice Missing key 问题

cfg = OmegaConf.create({
    # 晶格固定
    'cost_lattice': 0.0,
    'cost_coord':   1.0,
    'cost_type':    1.0,

    # 时间嵌入
    'time_dim': 256,

    # 谱编码器
    'spectrum_latent_dim': 256,
    'xmu_dim':  150,
    'chi_dim':  200,
    'feat_dim': 73,

    # CSPNet 实际接收 latent_dim = time_dim + spectrum_latent_dim = 512
    'latent_dim': 256,

    # decoder
    'decoder': {
        '_target_': 'diffcsp.pl_modules.cspnet.CSPNet',
        'hidden_dim':    256,
        'latent_dim':    512,
        'num_layers':    6,
        'max_atoms':     100,
        'act_fn':        'silu',
        'dis_emb':       'sin',
        'num_freqs':     10,
        'edge_style':    'fc',
        'cutoff':        6.0,
        'max_neighbors': 20,
        'ln':            True,
        'ip':            True,
        'smooth':        True,
        'pred_type':     True,
    },

    # 噪声调度器
    'beta_scheduler': {
        '_target_': 'diffcsp.pl_modules.diff_utils.BetaScheduler',
        'timesteps':      1000,
        'scheduler_mode': 'cosine',
    },
    'sigma_scheduler': {
        '_target_': 'diffcsp.pl_modules.diff_utils.SigmaScheduler',
        'timesteps':   1000,
        'sigma_begin': 0.005,
        'sigma_end':   0.5,
    },

    # optimizer（BaseModule.configure_optimizers 需要）
    'optim': {
        'optimizer': {
            '_target_': 'torch.optim.Adam',
            'lr': 1e-4,
        },
        'use_lr_scheduler': False,
    },
})

print(f"\n✓ 配置构建成功：cost_lattice={cfg.cost_lattice}")


# ── Step 3：初始化模型 ─────────────────────────────────────────────────────────

model = CSPDiffusion(**cfg)
model.eval()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ 模型初始化成功：{n_params:,} 个可训练参数")
print(f"  keep_lattice = {model.keep_lattice}  （应为 True）")
assert model.keep_lattice, "❌ keep_lattice 应为 True（cost_lattice=0），请检查配置"


# ── Step 4：前向传播 ──────────────────────────────────────────────────────────

model.train()
output_dict  = model(batch)

loss         = output_dict['loss']
loss_coord   = output_dict['loss_coord']
loss_type    = output_dict['loss_type']
loss_lattice = output_dict['loss_lattice']

print(f"\n=== Forward Pass 结果 ===")
print(f"  loss         = {loss.item():.4f}")
print(f"  loss_coord   = {loss_coord.item():.4f}")
print(f"  loss_type    = {loss_type.item():.4f}")
print(f"  loss_lattice = {loss_lattice.item():.6f}  （keep_lattice=True 时不参与 loss）")

assert not loss.isnan(),      "❌ loss 为 NaN"
assert loss.item() > 0,       "❌ loss 为 0，不正常"
assert loss_coord.item() > 0, "❌ loss_coord 为 0"
assert loss_type.item()  > 0, "❌ loss_type 为 0"

print("\n✅ forward pass 通过，loss 不为 NaN")


# ── Step 5：反向传播 ──────────────────────────────────────────────────────────

loss.backward()

grad_norms = [p.grad.norm().item()
              for p in model.parameters() if p.grad is not None]
assert len(grad_norms) > 0, "❌ 没有梯度"
max_grad = max(grad_norms)
has_nan  = any(g != g for g in grad_norms)

print(f"✅ backward 通过：{len(grad_norms)} 个参数有梯度，"
      f"max_grad={max_grad:.4f}，nan={has_nan}")
assert not has_nan, "❌ 存在 NaN 梯度"


# ── Step 6：晶格固定检查（防御 Exp1 覆辙）────────────────────────────────────

print("\n=== 晶格固定检查 ===")

# 取前 2 个非 None 样本
samples = []
for i in range(len(dm.train_dataset)):
    s = dm.train_dataset[i]
    if s is not None:
        samples.append(s)
    if len(samples) == 2:
        break

mini_batch = Batch.from_data_list(samples)

# 验证输入晶格对角元素 = 12
input_lattice = lattice_params_to_matrix_torch(
    mini_batch.lengths, mini_batch.angles)
diag = torch.stack([input_lattice[:, i, i] for i in range(3)], dim=1)
print(f"  输入晶格对角元素：{diag.tolist()}")
assert (diag - 12.0).abs().max() < 0.01, \
    f"❌ 输入晶格不是 diag(12,12,12)！diag={diag}"

# keep_lattice=True → sample() 全程用 l_T = input_lattice，不更新
assert model.keep_lattice, "❌ keep_lattice 不为 True"
print(f"  keep_lattice = {model.keep_lattice}  → 晶格固定，不参与扩散")
print(f"✅ 晶格固定检查通过")


# ── 总结 ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("✅ Step 3.5 全部检查通过")
print(f"   loss         = {loss.item():.4f}")
print(f"   loss_coord   = {loss_coord.item():.4f}")
print(f"   loss_type    = {loss_type.item():.4f}")
print(f"   keep_lattice = {model.keep_lattice}")
print(f"   n_params     = {n_params:,}")
print("=" * 60)
print("\nStep3 完成，可进入 Step4 训练。")