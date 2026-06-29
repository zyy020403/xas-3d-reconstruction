# SHARED_02_SPECTRAL_AND_MODEL.md
# 谱处理策略与模型架构说明

> **适用范围**：Step2、Step3 Agent 重点阅读；其他 Agent 可参考
> **状态**：LOCKED

---

## 1. 三路谱输入设计（最终版）

### 1.1 分工

```
xmu_xanes  [E₀-50, E₀+150 eV, 150点]
  → 氧化态、局部对称性（XANES 独有信息）
  → chi1 中不包含此信息

chi1       [k_min, k_max, 200点]
  → 键长（振荡频率）、配位数（振荡幅度）
  → 已是 k¹χ(k)，不再额外加权

feff_features [73维标量]
  → 从双谱提炼的物理描述子
  → 包含从 xmu 提取的 XANES 特征和从 chi 提取的 EXAFS 特征
  → 补充模型的先验，加速收敛
```

### 1.2 为什么不只用 chi1

chi1 是从 xmu 减去原子背景后变换到 k 空间得到的，因此：
- xmu 的前 ~100 eV（XANES 区域）的信息**不存在于** chi1 中
- XANES 包含：Fe 的 d 轨道占据情况（氧化态）、近邻原子的几何排列（点群对称性）、配位几何（八面体/四面体）
- 这些信息对晶体结构预测至关重要，不可丢弃

### 1.3 为什么 xmu 只截 XANES 不用全谱

- xmu 在 E₀+150 eV 以后进入 EXAFS 能量区域，与 chi1 信息高度重叠（只是表示空间不同）
- 截取 [E₀-50, E₀+150 eV] 保留 XANES 独有信息，避免冗余
- chi1 负责 EXAFS 部分，两路信息边界清晰

---

## 2. 谱预处理代码规范

### 2.1 xmu → XANES 窗口（150点）

```python
def load_xmu_xanes(xmu_path, E0, n_points=150, pre_eV=50, post_eV=150):
    """
    从 xmu.dat 截取 XANES 窗口 [E0-50, E0+150] eV，插值到 150 点
    
    Args:
        xmu_path: xmu.dat 文件路径
        E0:       吸收边能量（来自 feff_features 的 E0 列）
        n_points: 输出点数，固定 150
        pre_eV:   边沿前窗口（50 eV）
        post_eV:  边沿后窗口（150 eV）
    
    Returns:
        np.ndarray (150,)，已逐样本标准化
    """
    data = np.loadtxt(xmu_path, comments='#')
    E, mu = data[:, 0], data[:, 1]
    
    E_lo, E_hi = E0 - pre_eV, E0 + post_eV
    mask = (E >= E_lo) & (E <= E_hi)
    
    # 安全检查：若截取区间内点数不足，用全局截取后线性插值
    E_win = E[mask] if mask.sum() >= 5 else E
    mu_win = mu[mask] if mask.sum() >= 5 else mu
    
    E_uniform = np.linspace(E_lo, E_hi, n_points)
    mu_interp = np.interp(E_uniform, E_win, mu_win,
                          left=mu_win[0], right=mu_win[-1])
    
    mu_norm = (mu_interp - mu_interp.mean()) / (mu_interp.std() + 1e-8)
    return mu_norm.astype(np.float32)
```

### 2.2 chi1 → EXAFS（200点）

```python
def load_chi1(chi1_path, n_points=200):
    """
    读取 chi1.dat，插值到 200 点均匀 k 网格
    
    ★ 重要：chi1.dat 已是 k¹χ(k)，不再额外 k 加权
    ★ 只做插值 + 逐样本标准化
    """
    data = np.loadtxt(chi1_path, comments='#')
    k, chi1 = data[:, 0], data[:, 1]
    
    k_uniform  = np.linspace(k.min(), k.max(), n_points)
    chi_interp = np.interp(k_uniform, k, chi1)
    
    chi_norm = (chi_interp - chi_interp.mean()) / (chi_interp.std() + 1e-8)
    return chi_norm.astype(np.float32)
```

### 2.3 feff_features（73维）

```python
def load_feff_features(features_row, scaler=None):
    """
    从 feff_features_all_site_v2.csv 的一行中提取 73 个数值列
    
    Args:
        features_row: DataFrame 的一行（已按 (mp_id, site_nn) 匹配）
        scaler:       sklearn StandardScaler（训练集拟合，测试集复用）
    
    Returns:
        np.ndarray (73,)
    
    数值列：列索引 3-75（共 73 列）
    flag 列（27-29）：视为普通 float
    NaN 处理：用训练集均值填充
    """
    numeric_cols = features_row.iloc[3:76].values.astype(np.float32)
    if scaler is not None:
        numeric_cols = scaler.transform(numeric_cols.reshape(1, -1)).flatten()
    return numeric_cols
```

---

## 3. SpectrumEncoder 架构

```python
class SpectrumEncoder(nn.Module):
    """
    三路编码器：xmu_xanes + chi1 + feff_features → (B, 256) 条件向量
    
    这个向量将拼接到 time_emb 上，作为 CSPNet 的条件输入。
    CSPNet 本身不修改。
    """
    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=73, latent_dim=256):
        super().__init__()

        # xmu 分支：E空间 XANES，1D CNN
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 256), nn.SiLU(),
        )

        # chi1 分支：k空间 EXAFS，1D CNN
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

        # 融合：256+128+64=448 → 256
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, xmu_xanes, chi1, feff_feats):
        # xmu_xanes: (B, 150), chi1: (B, 200), feff_feats: (B, 73)
        xmu_out  = self.xmu_encoder(xmu_xanes.unsqueeze(1))   # (B, 256)
        chi_out  = self.chi_encoder(chi1.unsqueeze(1))         # (B, 128)
        feat_out = self.feat_encoder(feff_feats)               # (B, 64)
        return self.fusion(torch.cat([xmu_out, chi_out, feat_out], dim=-1))
```

---

## 4. 如何注入 CSPDiffusion（diffusion_w_type.py 改动要点）

### 4.1 __init__ 改动

```python
# 新增（在原有初始化之后添加）：
self.spectrum_encoder = SpectrumEncoder(
    xmu_dim=150, chi_dim=200, feat_dim=73,
    latent_dim=self.hparams.spectrum_latent_dim  # 256
)
# yaml 中 latent_dim = time_dim(256) + spectrum_latent_dim(256) = 512
# decoder 的 latent_dim 参数已在原代码中设为 hparams.latent_dim + hparams.time_dim
# 将 hparams.latent_dim 设为 spectrum_latent_dim=256 即可
```

### 4.2 forward() 改动

```python
# 原代码（保留）：
time_emb = self.time_embedding(times)  # (B, 256)

# 新增（在 time_emb 之后）：
spectrum_cond = self.spectrum_encoder(
    batch.xmu_xanes,       # (B, 150)
    batch.chi1,            # (B, 200)
    batch.feff_features)   # (B, 73)
# → (B, 256)

# 修改（把 time_emb 换成 condition）：
condition = torch.cat([time_emb, spectrum_cond], dim=-1)  # (B, 512)

# 原调用改为：
pred_l, pred_x, pred_t = self.decoder(
    condition,   # ← 原来是 time_emb
    atom_type_probs, input_frac_coords, input_lattice,
    batch.num_atoms, batch.batch)
```

### 4.3 sample() 改动

```python
# sample() 接收 batch，从中取谱：
spectrum_cond = self.spectrum_encoder(
    batch.xmu_xanes, batch.chi1, batch.feff_features)
# 其余采样逻辑不变，只是每次调用 self.decoder 时把 time_emb 换成 condition
```

### 4.4 batch 新字段（dataset 负责填充）

```python
# dataset.__getitem__ 返回的字典中新增：
{
    'xmu_xanes':      torch.FloatTensor(150,),
    'chi1':           torch.FloatTensor(200,),
    'feff_features':  torch.FloatTensor(73,),
    # 原有字段保留：frac_coords, atom_types, lengths, angles, num_atoms
}
```

---

## 5. 配置文件改动要点

### conf/model/diffusion_xas.yaml（新建）

```yaml
# 在原 diffusion_w_type.yaml 基础上改动：
_target_: diffcsp.pl_modules.diffusion_w_type.CSPDiffusion

latent_dim: 256         # spectrum_latent_dim，拼到 time_dim=256 上
time_dim: 256           # 不变
spectrum_latent_dim: 256

# decoder 的 latent_dim 由代码计算：latent_dim + time_dim = 512
# cspnet.yaml 中 latent_dim 会被覆盖为 512，不需要手动改
```
