# STEP2_HANDOFF.md（最终版）
# Step2 Agent 交接文档：谱预处理验证 + SpectrumEncoder 开发

> **你的角色**：Step2 Agent
> **你的任务**：
>   1. 实现并验证三路谱预处理函数（xmu_xanes / chi1 / feff_features）
>   2. 实现 SpectrumEncoder，做端到端前向测试
>   3. 额外：统计确定虚拟晶格边长 L
> **前置文档**：先读 SHARED_00_v2.md、SHARED_01_DATA_MANIFEST.md
> **输出目录**：`C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step2\`

---

## Step1 交接的关键信息（与共享文档有出入之处，以此为准）

| 项目 | 实际情况 | 对你的影响 |
|------|----------|------------|
| xmu.dat 列顺序 | `omega / e / k / mu / mu0 / chi`，能量在**第1列**（0-indexed），μ(E) 在**第3列**（0-indexed） | load_xmu_xanes 用 `E=data[:,1], mu=data[:,3]` |
| chi1.dat | 全部 18,384 条有效，无需 std 过滤 | 直接读取即可 |
| feff_features 路径 | `C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv` | 所有脚本统一使用此路径 |
| feff_features NaN | 8,714 行含 NaN，需填充 | 用 feff_feature_stats.csv 的训练集均值填充 |
| 有效化合物数 | 11,636 个 | 无影响 |
| 局部结构方案 | 固定取最近 **20 个邻居**（Fe自身不计），共 21 个原子 | 影响 Step2.5 的虚拟晶格 L 统计 |

---

## 你需要的文件（向用户索取）

```
共享文档（已有）：
  SHARED_00_v2.md
  SHARED_01_DATA_MANIFEST.md

Step1 输出（需要访问路径，不需要发送文件内容）：
  experiment2/step1/data_inventory.csv
  experiment2/step1/feff_feature_scaler.pkl
  experiment2/step1/feff_feature_stats.csv

参考脚本（可访问，仅参考逻辑）：
  experiment/quicktest/step2.1_qt_encoder.py
```

---

## 工作内容

### Step 2.1：实现谱预处理函数库

**文件名**：`spectrum_preprocessor.py`（Step3 会直接 import 这个文件）

实现以下三个函数，**文件只含函数定义，不含执行代码**：

#### 函数1：load_xmu_xanes

```python
import numpy as np

def load_xmu_xanes(xmu_path, E0, n_points=150, pre_eV=50, post_eV=150):
    """
    从 xmu.dat 截取 XANES 窗口 [E0-50, E0+150] eV，插值到 150 点

    xmu.dat 列顺序（Step1 实测确认）：omega / e / k / mu / mu0 / chi
      能量列：data[:,1]（第2列，0-indexed=1）
      μ(E)列：data[:,3]（第4列，0-indexed=3）

    Args:
        xmu_path : str
        E0       : float，来自 feff_features 表的 E0 列（列索引6，列名'E0'）
        n_points : int = 150
        pre_eV   : float = 50（边沿前窗口）
        post_eV  : float = 150（边沿后窗口）

    Returns:
        np.ndarray shape (150,)，逐样本 z-score 标准化
    """
    data = np.loadtxt(xmu_path, comments='#')
    E  = data[:, 1]
    mu = data[:, 3]

    E_lo, E_hi = E0 - pre_eV, E0 + post_eV
    mask = (E >= E_lo) & (E <= E_hi)

    if mask.sum() < 5:
        # 容错：窗口内点数不足，用全局范围插值
        E_win, mu_win = E, mu
    else:
        E_win, mu_win = E[mask], mu[mask]

    E_uniform = np.linspace(E_lo, E_hi, n_points)
    mu_interp = np.interp(E_uniform, E_win, mu_win,
                          left=mu_win[0], right=mu_win[-1])

    mu_norm = (mu_interp - mu_interp.mean()) / (mu_interp.std() + 1e-8)
    return mu_norm.astype(np.float32)
```

#### 函数2：load_chi1

```python
def load_chi1(chi1_path, n_points=200):
    """
    读取 chi1.dat（已是 k¹χ(k)），插值到 200 点，逐样本归一化
    ★ 不做额外 k 加权

    Returns:
        np.ndarray shape (200,)
    """
    data = np.loadtxt(chi1_path, comments='#')
    k    = data[:, 0]
    chi1 = data[:, 1]

    k_uniform  = np.linspace(k.min(), k.max(), n_points)
    chi_interp = np.interp(k_uniform, k, chi1)

    chi_norm = (chi_interp - chi_interp.mean()) / (chi_interp.std() + 1e-8)
    return chi_norm.astype(np.float32)
```

#### 函数3：load_feff_features

```python
import pickle
import pandas as pd

def load_feff_features(features_row, scaler, col_means_for_nan):
    """
    从 feff_features 表的一行提取 73 个数值列，NaN 填充 + 标准化

    Args:
        features_row      : pd.Series，已按 (mp_id, site_nn) 匹配的一行
        scaler            : 已加载的 StandardScaler（来自 feff_feature_scaler.pkl）
        col_means_for_nan : np.ndarray (73,)，训练集各列均值（来自 feff_feature_stats.csv）

    Returns:
        np.ndarray shape (73,)，已标准化

    数值列：iloc[3:76]，共 73 列
    """
    vals = features_row.iloc[3:76].values.astype(np.float32)

    nan_mask = np.isnan(vals)
    if nan_mask.any():
        vals[nan_mask] = col_means_for_nan[nan_mask]

    vals_scaled = scaler.transform(vals.reshape(1, -1)).flatten()
    return vals_scaled.astype(np.float32)
```

---

### Step 2.2：预处理验证与可视化

**文件名**：`step2_2_preprocess_validation.py`

从 `data_inventory.csv` 随机抽取 20 个样本，对每个样本运行三个预处理函数，检查 shape / NaN / 数值范围，生成可视化。

**验证检查**：
```python
assert xmu_out.shape  == (150,)
assert chi1_out.shape == (200,)
assert feats_out.shape == (73,)
assert not np.isnan(xmu_out).any()
assert not np.isnan(chi1_out).any()
assert not np.isnan(feats_out).any()
assert np.abs(xmu_out).max()  < 20
assert np.abs(chi1_out).max() < 20
```

**可视化**（`step2_visualization.png`）：
- 4行×5列，20个样本
- 每格：上半=xmu XANES（150点），下半=chi1（200点）
- 标题标注 mp_id
- 目的：肉眼确认 xmu 有清晰吸收边和白线，chi1 有振荡

**控制台打印**：
```
xmu_xanes : mean_of_means=X, mean_of_stds=X（应接近 0 和 1）
chi1      : mean_of_means=X, mean_of_stds=X
feats     : mean_of_means=X, mean_of_stds=X（StandardScaler 后精确为 0 和 1）
```

---

### Step 2.3：SpectrumEncoder 实现

**文件名**：`spectrum_encoder.py`（Step3 直接 import，文件只含类定义）

```python
import torch
import torch.nn as nn

class SpectrumEncoder(nn.Module):
    """
    三路编码器：xmu_xanes(150) + chi1(200) + feff_features(73) → (B, 256)

    输出的 (B, 256) 将在 diffusion_w_type.py 中与 time_emb(256) 拼接，
    得到 condition(512) 传入 CSPNet（CSPNet 本身不修改任何一行代码）。
    """
    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=73, latent_dim=256):
        super().__init__()

        # xmu 分支：E空间 XANES
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.SiLU(),
        )

        # chi1 分支：k空间 EXAFS
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 128), nn.SiLU(),
        )

        # feff_features 分支：MLP
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )

        # 融合：256+128+64=448 → latent_dim
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, xmu_xanes, chi1, feff_feats):
        """
        Args:
            xmu_xanes : (B, 150)
            chi1      : (B, 200)
            feff_feats: (B, 73)
        Returns:
            (B, 256)
        """
        xmu_out  = self.xmu_encoder(xmu_xanes.unsqueeze(1))
        chi_out  = self.chi_encoder(chi1.unsqueeze(1))
        feat_out = self.feat_encoder(feff_feats)
        return self.fusion(
            torch.cat([xmu_out, chi_out, feat_out], dim=-1))
```

---

### Step 2.4：SpectrumEncoder 前向测试

**文件名**：`step2_4_encoder_test.py`

```python
if __name__ == "__main__":
    import torch
    from spectrum_encoder import SpectrumEncoder

    encoder = SpectrumEncoder()
    encoder.eval()
    B = 4

    xmu   = torch.randn(B, 150)
    chi1  = torch.randn(B, 200)
    feats = torch.randn(B, 73)

    with torch.no_grad():
        out = encoder(xmu, chi1, feats)

    assert out.shape == (B, 256), f"shape error: {out.shape}"
    assert not out.isnan().any(), "NaN in output"
    print(f"✅ 输出 shape: {out.shape}")
    print(f"✅ 无 NaN")
    print(f"   数值范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 测试 condition 拼接（模拟 diffusion_w_type.py 中的操作）
    time_emb  = torch.randn(B, 256)
    condition = torch.cat([time_emb, out], dim=-1)
    assert condition.shape == (B, 512)
    print(f"✅ condition shape: {condition.shape}（time_emb(256) + spectrum(256) = 512）")
```

---

### Step 2.5：统计虚拟晶格边长 L

**文件名**：`step2_5_determine_L.py`

**这是 Step2 的重要额外任务**：确定虚拟晶格边长 L，供 Step3 Dataset 使用。

**方法**：
```python
# 对 data_inventory.csv 中随机抽取 500 个样本
# 对每个样本：
#   1. 读 POSCAR → 原胞转换（symprec=0.1）
#   2. 定位 LVSI 对应的 Fe 原子（取第一个 Fe 位点）
#   3. 找最近 20 个邻居，记录第20个邻居的笛卡尔距离 d20
#   4. 记录 20 个邻居中最远的距离 d_max

# 统计 d20 和 d_max 的分布
# L 的确定原则：
#   L 应 > 2 × max(d_max 的 99th percentile)
#   （保证所有邻居的分数坐标都在 [-0.5, 0.5] 范围内，不溢出）
#   同时 L 不要太大（避免浪费坐标空间）

# 输出：
#   d20_distribution.png
#   console 打印：d20 mean/median/99th pct，d_max mean/median/99th pct
#   推荐 L 值（取 2 × d_max_99th，向上取整到整数）
```

**注意**：
- Fe 定位方式：`[i for i, s in enumerate(primitive) if s.specie.symbol == 'Fe'][0]`
- 用 `primitive.get_neighbors(primitive[fe_idx], r=10.0)` 取全部邻居，按距离排序取前20
- 若某结构 Fe 邻居不足 20 个（在 10Å 内），记录并跳过，报告比例

**输出文件**：
```
experiment2/step2/
├── d20_distribution.png
└── L_recommendation.txt（写入推荐 L 值和统计依据）
```

---

## 最终输出文件清单

```
experiment2/step2/
├── spectrum_preprocessor.py    ★ 预处理函数库（Step3 import）
├── spectrum_encoder.py         ★ SpectrumEncoder 类（Step3 import）
├── step2_2_preprocess_validation.py
├── step2_visualization.png
├── step2_4_encoder_test.py
├── step2_5_determine_L.py
├── d20_distribution.png
└── L_recommendation.txt        ★ Step3 必须读取这个来确定 L 值
```

标 ★ 的是 Step3 必须用到的文件。

---

## 注意事项

1. `spectrum_preprocessor.py` 和 `spectrum_encoder.py` 是纯库文件，**不含任何执行代码**，Step3 直接 import

2. **xmu.dat 列索引务必用 Step1 实测的结果**：E=data[:,1]，mu=data[:,3]

3. **feff_features 路径统一用**：`C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv`

4. Step 2.5 的原胞转换代码直接复用 Exp1：`experiment/step3/xas_dataset.py` 中的 `get_primitive_structure`

5. 一个脚本跑完，汇报结果，Main Agent 确认无误后再写下一个

---

## 完成后向 Main Agent 汇报

重点汇报：
- Step2.2：20个样本验证是否全部通过，可视化谱形态是否正常
- Step2.4：shape 和 NaN 检查是否通过
- Step2.5：d20 和 d_max 的 99th percentile，推荐 L 值
- 任何异常或需要决策的问题
