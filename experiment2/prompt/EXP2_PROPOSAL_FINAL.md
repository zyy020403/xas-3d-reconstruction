# Experiment 2 Proposal：基于 DiffCSP 的 XAS → 晶体结构预测
# ★ 最终定稿版（v4）★

> **状态**：LOCKED — 不再修改
> **日期**：2026-04-09

---

## 0. 一句话核心任务

用 Fe K-edge XAS 谱（xmu XANES 区间 + chi1 EXAFS + 物理先验特征）替换化学组成，作为 DiffCSP 的条件输入，预测**原胞**的完整晶体结构（分数坐标 + 晶格参数 + 原子类型）。

```
原始 DiffCSP：
  条件输入  →  化学组成（原子类型列表，已知）
  预测目标  →  frac_coords + lengths/angles + atom_types

Experiment 2：
  条件输入  →  xmu XANES 窗口（150点）
             + chi1 EXAFS（200点）
             + feff_features（73维）
  预测目标  →  frac_coords + lengths/angles + atom_types  ← 完全不变

坐标空间、晶格预测、扩散机制、图构建 → 全部原版保留
唯一改动 → 把"已知组成"换成"双谱+物理特征"的三路条件
cspnet.py → 零改动
```

---

## 1. Experiment 1 核心教训

| 错误 | Exp2 对策 |
|------|-----------|
| 超胞作为训练标签（原子数 64-70，坐标 RMSE ≈ 随机） | 全程原胞（get_primitive_standard_structure） |
| 局部球形截取（6Å球）导致原子数爆炸 | 彻底放弃，恢复原版分数坐标+晶格 |
| 多位点多谱输入混淆模型 | 每化合物选唯一代表谱（LVSI） |
| k²χ(k) 额外加权（后段几乎直线） | chi1.dat 已是 k¹χ(k)，不再额外加权 |
| 只用 chi1，丢失 XANES 氧化/对称信息 | xmu XANES + chi1 EXAFS 双谱同时输入 |

---

## 2. 数据集

### 2.1 规模

| 属性 | 数值 |
|------|------|
| 来源 | Materials Project，at least include Fe |
| 化合物预期总数 | ~12,956 |
| 文件夹总数 | 18,385（多位点展开） |
| 数据根目录 | `C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site\` |

### 2.2 文件夹命名解析

```
mp_{id}_{formula}_feff_Fe_site_{nn}
  mp_id   → Materials Project ID，仅作数据管理键，禁止入模型
  formula → 化学式，禁止入模型
  nn      → 位点序号，从 01 起始
```

### 2.3 使用文件

```
chi1.dat              ★ EXAFS（k¹χ(k)，k空间）
xmu.dat               ★ 全谱（μ(E)，能量空间，截取 XANES 窗口）
POSCAR_supercell_fixed ★ 结构文件（提取标签，内部原胞转换）
chi.dat / chi2.dat / feff.inp → 不使用
```

### 2.4 有效性过滤标准（Step1 脚本执行）

1. `chi1.dat`：存在 + 行数 ≥ 30 + chi 列 std > 自动阈值（扫全库取 5th percentile）
2. `xmu.dat`：存在 + 行数 ≥ 50 + mu 列无全零/NaN
3. `POSCAR_supercell_fixed`：存在且 pymatgen 可解析
4. 原胞转换后原子数 N ∈ [2, 100]

### 2.5 多位点代表谱选取（LVSI）

```
对每个 mp_id：
  收集所有有效文件夹，按 site_nn 升序
  选序号最小的有效文件夹作为代表
  其余位点不参与训练，但在 data_inventory 中记录总位点数
```

### 2.6 feff_features 匹配键

```
feff_features_all_site_v2.csv 的 sample_name 列：mp_1047285_FeO2__feff_site_01
文件夹名：                                        mp_1047285_FeO2__feff_Fe_site_01
差异："feff_site" vs "feff_Fe_site"

匹配方式：提取 (mp_id, site_nn) 复合键做 JOIN，不做字符串直接匹配
```

---

## 3. 谱与特征预处理（最终版）

### 3.1 xmu.dat → XANES 窗口（150点）

```
截取范围：[E₀ - 50 eV,  E₀ + 150 eV]
总宽度：  200 eV
点数：    150 点（约 1.33 eV/点）
E₀ 来源：feff_features 表第 6 列（E0）

物理含义：
  E₀ - 50  捕获前沿峰（pre-edge，~7110-7117 eV）
  E₀       吸收边（~7120 eV）
  E₀ + 150 覆盖白线 + 近边振荡（~7270 eV 以内）
  EXAFS 部分（>E₀+150）由 chi1 负责，不重复
```

```python
def load_xmu_xanes(xmu_path, E0, n_points=150, pre=50, post=150):
    data = np.loadtxt(xmu_path, comments='#')
    E, mu = data[:, 0], data[:, 1]
    mask = (E >= E0 - pre) & (E <= E0 + post)
    E_win, mu_win = E[mask], mu[mask]
    E_uniform = np.linspace(E0 - pre, E0 + post, n_points)
    mu_interp = np.interp(E_uniform, E_win, mu_win)
    mu_norm = (mu_interp - mu_interp.mean()) / (mu_interp.std() + 1e-8)
    return mu_norm.astype(np.float32)  # (150,)
```

### 3.2 chi1.dat → EXAFS（200点）

```python
def load_chi1(chi1_path, n_points=200):
    data = np.loadtxt(chi1_path)
    k, chi1 = data[:, 0], data[:, 1]   # chi1.dat 已是 k¹χ(k)，不再加权
    k_uniform = np.linspace(k.min(), k.max(), n_points)
    chi_interp = np.interp(k_uniform, k, chi1)
    chi_norm = (chi_interp - chi_interp.mean()) / (chi_interp.std() + 1e-8)
    return chi_norm.astype(np.float32)  # (200,)
```

### 3.3 feff_features → 73维物理先验

```
来源：feff_features_all_site_v2.csv，列 3-75（共 73 列）
排除：sample_dir（0）、sample_name（1）、feature_version（2）
flag 列（27-29）：视为普通 float（0.0/1.0）
标准化：全局 z-score，用训练集统计量（测试/holdout 沿用训练集均值方差）
```

---

## 4. 模型架构

### 4.1 条件注入（读代码后确认的最小改动）

```python
# diffusion_w_type.py — forward() 改动位置
spectrum_cond = self.spectrum_encoder(
    batch.xmu_xanes,       # (B, 150)
    batch.chi1,            # (B, 200)
    batch.feff_features)   # (B, 73)
# → (B, 256)

condition = torch.cat([time_emb, spectrum_cond], dim=-1)  # (B, 512)

pred_l, pred_x, pred_t = self.decoder(
    condition,   # 原来是 time_emb(256)，现在是 condition(512)
    atom_type_probs, input_frac_coords, input_lattice,
    batch.num_atoms, batch.batch)

# conf yaml 中：latent_dim = 512（time_dim=256 + spectrum_latent=256）
# cspnet.py：零改动
```

### 4.2 SpectrumEncoder（三路）

```python
class SpectrumEncoder(nn.Module):
    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=73, latent_dim=256):
        super().__init__()

        # xmu 分支：E空间，XANES，1D CNN
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),    # → (B, 64, 16)
            nn.Flatten(),                # → (B, 1024)
            nn.Linear(1024, 256), nn.SiLU(),
        )

        # chi1 分支：k空间，EXAFS，1D CNN
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),    # → (B, 64, 16)
            nn.Flatten(),                # → (B, 1024)
            nn.Linear(1024, 128), nn.SiLU(),
        )

        # feff_features 分支：MLP
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )

        # 融合：256+128+64=448 → latent_dim=256
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, xmu_xanes, chi1, feff_feats):
        xmu_out  = self.xmu_encoder(xmu_xanes.unsqueeze(1))  # (B, 256)
        chi_out  = self.chi_encoder(chi1.unsqueeze(1))        # (B, 128)
        feat_out = self.feat_encoder(feff_feats)              # (B, 64)
        fused = torch.cat([xmu_out, chi_out, feat_out], dim=-1)
        return self.fusion(fused)                             # (B, 256)
```

### 4.3 改动文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `diffcsp/pl_modules/diffusion_w_type.py` | 手术式添加 | SpectrumEncoder 类；__init__ 实例化；forward/sample 改条件拼接 |
| `diffcsp/pl_data/dataset.py` | 重写 | 读 xmu+chi1+POSCAR+feff_features；原胞转换 |
| `diffcsp/pl_data/datamodule.py` | 小改 | 按 experiment2/step1 id 文件划分 |
| `conf/data/xas_fe_only.yaml` | 新建 | 数据路径 |
| `conf/model/diffusion_xas.yaml` | 新建 | latent_dim=512 |
| `diffcsp/pl_modules/cspnet.py` | **零改动** | — |

---

## 5. 样本标签：原胞转换

```python
def get_primitive(poscar_path, tolerance=0.1):
    structure = Structure.from_file(poscar_path)
    analyzer  = SpacegroupAnalyzer(structure, symprec=tolerance)
    return analyzer.get_primitive_standard_structure()
# 目标：原胞平均原子数 < 20
```

---

## 6. Holdout 集（1000个）

```
策略：用 feff_features 的 (E0, white_line_I, R1_peak_pos, chi_kmax) 做 K-Means(K=100)
      每簇抽 ~10%（size≥5 才抽），确保训练集有"双胞胎"
结果：holdout_1000_ids.txt — 训练期间绝对禁止触碰
划分：剩余有效化合物 → train:val:test = 70:15:15
```

---

## 7. 训练策略

```yaml
硬件：RTX A4000 16GB，Windows，num_workers=0，bf16
batch_size: 32
lr: 1e-4
gradient_clip: 1.0
max_epochs: 500
early_stop_patience: 30
scheduler: CosineAnnealingLR(T_max=500)
cost_lattice/coord/type: 各 1.0
spectrum_latent_dim: 256  # → CSPNet latent_dim = 256+256 = 512
```

**开训前强制检查**：5 样本 sample() → pred_lengths ∈ [2, 15] Å 才继续

---

## 8. Pipeline 步骤

```
Step 1：数据清洗与清单构建（Step1 Agent）
Step 2：谱预处理验证（Step2 Agent）
Step 3：数据集 + 模型改造（Step3 Agent）
Step 4：训练、采样、评估（Step4 Agent）
Step 5：Holdout 检验（Step5 Agent，最后执行）
```

---

## 9. 文件存储规范

```
根目录：C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\
子目录：step1/ step2/ step3/ step4/ step5/ shared/
脚本命名：step{N}.{M}_{描述}.py
```

---

*LOCKED — Main Agent 定稿 2026-04-09*
