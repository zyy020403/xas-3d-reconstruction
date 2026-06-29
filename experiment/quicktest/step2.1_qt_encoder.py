# =============================================================================
# 脚本编号: step2.1_qt
# 脚本名称: step2.1_qt_encoder.py
# 输入:
#   - 各位点文件夹下的 chi.dat 文件
# 输出:
#   - 本文件作为模块被 step3_qt_train.py 导入
#   - 提供: preprocess_chi(), SpectrumEncoder
# 说明:
#   基于正式服 step2.1_spectrum_encoder.py，QuickTest 唯一改动：
#   preprocess_chi 中的 k 权重从 k²χ(k) 改为 k¹χ(k)（第124行等价位置）。
#   变量名 k2chi 保持不变，避免牵连其他代码。
#   其余代码（归一化、边界情况处理、SpectrumEncoder 类结构）完全不变。
#
#   与正式服的差异（仅此一处）：
#     正式服: k2chi = k_uniform ** 2 * chi_interp
#     QuickTest: k2chi = k_uniform ** 1 * chi_interp
# =============================================================================

import os
import sys
import warnings
import logging

import numpy as np
import torch
import torch.nn as nn

# 项目根目录
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP1_DIR = os.path.join(EXPERIMENT_DIR, "step1")
STEP2_DIR = os.path.join(EXPERIMENT_DIR, "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

# 数据根目录
SITE_DATASET_DIR  = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 预处理函数
# ---------------------------------------------------------------------------

def preprocess_chi(
    chi_path: str,
    k_grid_points: int = 512,
    k_min: float = 0.0,
    k_max: float = 20.0,
    min_valid_points: int = 50,
) -> torch.Tensor:
    """
    读取 chi.dat，返回归一化后的 k¹χ(k) 信号，shape = [1, k_grid_points]

    QuickTest 改动：k 权重为 1（正式服为 2），其余逻辑完全相同。

    处理步骤：
      1. 读取文件，跳过 '#' 开头的注释行
      2. 解析第一列（k，Å⁻¹）和第二列（χ(k)）
         ─ 若只有一列，视为 χ(k)，自动生成等间距 k 网格
      3. 丢弃 k < 0 的点（仪器 artifact）
      4. 检查有效点数；不足 min_valid_points 则返回全零向量并记录警告
      5. 在均匀 k 网格 [0, 20] 插值
      6. 计算 k¹χ(k)   ← QuickTest 唯一改动
      7. 归一化：除以 max(|k¹χ(k)|)；全零则原样返回
      8. 转为 float32 tensor，shape = [1, k_grid_points]
    """
    k_uniform = np.linspace(k_min, k_max, k_grid_points, dtype=np.float32)
    zero_tensor = torch.zeros(1, k_grid_points, dtype=torch.float32)

    # ── 读文件 ──────────────────────────────────────────────────────────────
    if not os.path.isfile(chi_path):
        logger.warning(f"preprocess_chi: 文件不存在: {chi_path}")
        return zero_tensor

    rows = []
    try:
        with open(chi_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue  # 跳过无法解析的行
    except Exception as e:
        logger.warning(f"preprocess_chi: 读取失败 {chi_path}: {e}")
        return zero_tensor

    if len(rows) == 0:
        logger.warning(f"preprocess_chi: 空文件或全注释: {chi_path}")
        return zero_tensor

    rows = np.array(rows, dtype=np.float32)

    # ── 解析列 ───────────────────────────────────────────────────────────────
    if rows.ndim == 1 or rows.shape[1] == 1:
        # 只有一列：χ(k)，自动生成 k
        chi_raw = rows.ravel()
        n = len(chi_raw)
        k_raw = np.linspace(k_min, k_max, n, dtype=np.float32)
    else:
        k_raw   = rows[:, 0]
        chi_raw = rows[:, 1]

    # ── 丢弃 k < 0 ───────────────────────────────────────────────────────────
    mask = k_raw >= 0.0
    k_raw   = k_raw[mask]
    chi_raw = chi_raw[mask]

    if len(k_raw) < min_valid_points:
        logger.warning(
            f"preprocess_chi: 有效点数不足 {min_valid_points}，"
            f"实际 {len(k_raw)} 点，返回全零: {chi_path}"
        )
        return zero_tensor

    # ── 插值到均匀网格 ────────────────────────────────────────────────────────
    # np.interp 始终返回 float64，需显式转回 float32
    chi_interp = np.interp(
        k_uniform, k_raw, chi_raw,
        left=chi_raw[0], right=chi_raw[-1]
    ).astype(np.float32)

    # ── k¹χ(k)  ← QuickTest 唯一改动（正式服为 k_uniform ** 2）────────────
    k2chi = k_uniform ** 1 * chi_interp

    # ── 归一化 ────────────────────────────────────────────────────────────────
    max_abs = np.max(np.abs(k2chi))
    if max_abs > 1e-10:
        k2chi = k2chi / max_abs
    # else: 保持全零（信号本身为零，直接使用）

    tensor = torch.from_numpy(k2chi).unsqueeze(0)   # [1, k_grid_points]
    return tensor


# ---------------------------------------------------------------------------
# 单谱编码器（与正式服完全相同）
# ---------------------------------------------------------------------------

class SpectrumEncoder(nn.Module):
    """
    将单条 k¹χ(k) 信号 + 元素类型 + ionic 标记 → 局部结构 embedding

    输入:
        spectrum      : [batch, 1, k_grid_points]   归一化的 k¹χ(k)
        atomic_number : [batch]                      int，原子序数 1-94
        is_ionic      : [batch]                      int，0 或 1

    输出:
        site_embedding: [batch, d_site]

    网络结构:
        1D CNN (4层，逐步下采样) → cnn_feat [batch, 256]
        nn.Embedding(95, 64)   → elem_emb  [batch, 64]
        nn.Embedding(2, 16)    → ionic_emb [batch, 16]
        concat → [batch, 336]
        MLP(336 → d_site → d_site)
    """

    def __init__(self, d_site: int = 256, k_grid_points: int = 512):
        super().__init__()
        self.d_site = d_site
        self.k_grid_points = k_grid_points

        # ── 1D CNN ────────────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # [B, 1, 512] → [B, 32, 512]
            nn.Conv1d(1,   32,  kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            # [B, 32, 512] → [B, 64, 256]
            nn.Conv1d(32,  64,  kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # [B, 64, 256] → [B, 128, 128]
            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # [B, 128, 128] → [B, 256, 64]
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # [B, 256, 64] → [B, 256, 1]
            nn.AdaptiveAvgPool1d(1),
        )
        # 展平后 cnn_feat: [B, 256]

        # ── 元素 & ionic embedding ─────────────────────────────────────────────
        # atomic_number 范围 1-94；index 0 留作 padding/未知
        self.element_emb = nn.Embedding(95, 64, padding_idx=0)
        self.ionic_emb   = nn.Embedding(2,  16)

        # ── 融合 MLP ──────────────────────────────────────────────────────────
        cnn_out_dim = 256
        elem_dim    = 64
        ionic_dim   = 16
        concat_dim  = cnn_out_dim + elem_dim + ionic_dim  # 336

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, d_site),
            nn.ReLU(inplace=True),
            nn.Linear(d_site, d_site),
        )

    def forward(
        self,
        spectrum: torch.Tensor,       # [B, 1, k_grid_points]
        atomic_number: torch.Tensor,  # [B]  int
        is_ionic: torch.Tensor,       # [B]  int (0 or 1)
    ) -> torch.Tensor:                # [B, d_site]

        # ── CNN 特征 ─────────────────────────────────────────────────────────
        cnn_out  = self.cnn(spectrum)           # [B, 256, 1]
        cnn_feat = cnn_out.squeeze(-1)          # [B, 256]

        # ── 元素 & ionic embedding ────────────────────────────────────────────
        elem_emb  = self.element_emb(atomic_number.long())   # [B, 64]
        ionic_emb = self.ionic_emb(is_ionic.long())          # [B, 16]

        # ── 融合 ──────────────────────────────────────────────────────────────
        fused = torch.cat([cnn_feat, elem_emb, ionic_emb], dim=-1)  # [B, 336]
        return self.mlp(fused)                                        # [B, d_site]


# ---------------------------------------------------------------------------
# 工具：统计模型参数量
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 快速验证（__main__）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("Step 2.1_qt 快速验证（k¹ 权重）")
    print("=" * 60)

    # ── 1. 测试 preprocess_chi：合成信号 ─────────────────────────────────────
    print("\n[1] preprocess_chi 合成信号测试")

    import tempfile
    k_vals = np.linspace(0.5, 18.0, 300)
    chi_vals = np.sin(k_vals) * np.exp(-0.05 * k_vals)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    )
    tmp.write("# 测试文件\n")
    for k, c in zip(k_vals, chi_vals):
        tmp.write(f"{k:.4f}  {c:.6f}\n")
    tmp.close()

    spec = preprocess_chi(tmp.name)
    print(f"  spectrum shape : {spec.shape}")
    print(f"  dtype          : {spec.dtype}")
    print(f"  值域           : [{spec.min():.4f}, {spec.max():.4f}]")
    assert spec.shape == (1, 512), "shape 错误！"
    assert spec.dtype == torch.float32, "dtype 不是 float32！"
    assert spec.max().abs() <= 1.0 + 1e-5, "归一化溢出！"
    os.unlink(tmp.name)
    print("  ✓ 合成信号测试通过")

    # ── 2. 边界情况：空文件 ───────────────────────────────────────────────────
    print("\n[2] preprocess_chi 边界情况：空文件")
    tmp2 = tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    )
    tmp2.write("# 只有注释\n")
    tmp2.close()
    spec2 = preprocess_chi(tmp2.name)
    assert spec2.shape == (1, 512)
    assert spec2.sum() == 0.0
    os.unlink(tmp2.name)
    print("  ✓ 空文件返回全零")

    # ── 3. 边界情况：单列文件 ────────────────────────────────────────────────
    print("\n[3] preprocess_chi 边界情况：单列文件")
    tmp3 = tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    )
    for c in chi_vals:
        tmp3.write(f"{c:.6f}\n")
    tmp3.close()
    spec3 = preprocess_chi(tmp3.name)
    assert spec3.shape == (1, 512)
    assert spec3.dtype == torch.float32
    print(f"  spectrum shape : {spec3.shape}")
    os.unlink(tmp3.name)
    print("  ✓ 单列文件处理通过")

    # ── 4. SpectrumEncoder 前向传播 ─────────────────────────────────────────
    print("\n[4] SpectrumEncoder forward pass（batch=4）")
    encoder = SpectrumEncoder(d_site=256)
    encoder.eval()

    batch_spec = torch.randn(4, 1, 512)
    batch_z    = torch.tensor([26, 14, 3, 38])   # Fe, Si, Li, Sr
    batch_ion  = torch.tensor([0, 0, 1, 1])

    with torch.no_grad():
        out = encoder(batch_spec, batch_z, batch_ion)

    print(f"  output shape : {out.shape}")
    assert out.shape == (4, 256), "输出维度错误！"
    assert not torch.isnan(out).any(),  "输出包含 NaN！"
    assert not torch.isinf(out).any(),  "输出包含 Inf！"
    print("  ✓ 正常输入通过")

    # ── 5. 对全零输入不崩溃 ──────────────────────────────────────────────────
    print("\n[5] SpectrumEncoder 全零输入（无效谱）")
    zero_spec = torch.zeros(2, 1, 512)
    with torch.no_grad():
        out_zero = encoder(zero_spec, torch.tensor([26, 3]), torch.tensor([0, 1]))
    assert not torch.isnan(out_zero).any()
    print("  ✓ 全零输入不崩溃")

    # ── 6. 真实数据（如果清单存在）──────────────────────────────────────────
    inventory_path = os.path.join(STEP1_DIR, "data_inventory.csv")
    if os.path.isfile(inventory_path):
        print("\n[6] 真实数据测试")
        df = pd.read_csv(inventory_path)
        valid = df[df.get("files_complete", df.get("chi_exists", True)) == True]
        if len(valid) > 0:
            row = valid.iloc[0]
            source_col = "source_path" if "source_path" in df.columns else "folder_path"
            chi_path = os.path.join(row[source_col], "chi.dat")
            spec_real = preprocess_chi(chi_path)
            print(f"  chi.dat        : {chi_path}")
            print(f"  spectrum shape : {spec_real.shape}")
            print(f"  dtype          : {spec_real.dtype}")
            assert spec_real.dtype == torch.float32
            with torch.no_grad():
                atomic_num = torch.tensor([int(row.get("atomic_number", 26))])
                is_ion     = torch.tensor([int(row.get("is_ionic", 0))])
                out_real   = encoder(spec_real.unsqueeze(0), atomic_num, is_ion)
            print(f"  site embedding shape : {out_real.shape}")
            print("  ✓ 真实数据测试通过")
    else:
        print("\n[6] data_inventory.csv 未找到，跳过真实数据测试")

    # ── 7. 参数量 ─────────────────────────────────────────────────────────────
    n_params = count_parameters(encoder)
    print(f"\n[参数量] SpectrumEncoder: {n_params:,} 个可训练参数 "
          f"（约 {n_params/1e6:.2f} 万）")

    print("\n" + "=" * 60)
    print("Step 2.1_qt 所有验证通过 ✓")
    print("=" * 60)