# EXP4_STEP3_SUBAGENT_HANDOFF.md
# DiffCSP-Experiment4 Step 3 Sub-Agent 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 3
> **接收者**：DiffCSP-Exp4-Step3-SubAgent
> **日期**：2026-04-25
> **前置阶段**：Step 1 / Step 2 / Step 2.5 已全部完成（v2 数据集 75,637 样本就绪）
> **目标阶段**：服务器端 Dataset + SpectrumEncoder + 前向测试（**不训练，不评估**）

---

## 1. 任务全景

### 1.1 你（Step 3 Sub-Agent）要做的

| # | 子任务 | 关键产出 |
|---|--------|----------|
| Phase 0 | 环境验证（5 项 sanity check） | 一份 `step3_phase0_report.txt` |
| Phase 1 | 数据上传 + Exp2 仓库 fork + 目录结构搭建 | `/home/tcat/diffcsp_exp4/` 目录就绪 |
| Phase 2 | Exp2 代码审计（哪些必改、哪些不动） | 审计清单 |
| Phase 3 | Dataset 改造 → `xas_local_dataset_v2.py` | 重写后的文件 |
| Phase 4 | SpectrumEncoder 一行改 → `spectrum_encoder.py` | 改后的文件 |
| Phase 5 | diffusion 路径常量 → `diffusion_w_type_xas.py` | 改后的文件 |
| Phase 6 | 前向测试（batch_size=4，CPU + GPU bf16） | `step3_forward_test_log.txt` |
| Phase 7 | 训练前 checklist 自查 | 完成后汇报模板填写 |

### 1.2 你**不要**做的

- ❌ 不实际训练（那是 Step 4）
- ❌ 不接触 holdout（Step 4 训练只用 train + val，holdout 留 Step 5）
- ❌ 不接触 `incompat_pool.csv`（Exp4 全程封存，留给 Exp5）
- ❌ 不混用 v1 split 文件（`*_ids.txt`、v1 `*_samples.csv`），只用 `*_samples_v2.csv`
- ❌ 不改不可变量：L=6，坐标系 [-0.5, 0.5]，`frac -= np.round(frac)`，N_NEIGHBORS=20，batch=16，lr=1e-4，bf16，三路 SpectrumEncoder（xmu 150 + chi1 200 + feff 74 → latent 256），cost_lattice=0
- ❌ 不加 TypeClassifier（Exp3 已证伪）
- ❌ 不为"完整性"跑 incompat 样本（即使你觉得"反正多 40% 数据"——拒绝这个想法，它是错的）

### 1.3 与 Exp2 的关系

继承 Exp2 的 DiffCSP + 三路 SpectrumEncoder 架构，扩展到全元素。**模型框架本身不动**（gnn.py / cspnet.py / diff_utils.py / 扩散数学），只改：
- Dataset（最大改动，重写）
- SpectrumEncoder 中 feff 分支 `nn.Linear(73, ...)` → `nn.Linear(74, ...)`（一行）
- diffusion_w_type_xas.py 路径常量

---

## 2. 环境基线（已确认）

### 2.1 服务器

- Host：`scsmlnprd02.its.auckland.ac.nz`，SSH 用户 `tcat`，**密码登录**（用户没有 ssh key 权限）
- OS：Ubuntu 22.04.4 LTS
- Python env：`jhub_env`（登录默认激活）
- GPU：2× RTX 4090 24 GB，CUDA 12.2，driver 535.183.01
- 根盘：1.72 TB，已用 94.3% → 剩 ~98 GB
- Swap：使用率 80%（系统 RAM 历史压力大；训练前 cache 到 `/tmp` 时务必先 `free -h`）

### 2.2 关键包版本（用户 pip freeze 已确认）

```
pymatgen==2025.10.7         ← 比本地 Windows 2024.8.9 新一年（Phase 0.1 验证 Cython bug 是否仍在）
numpy==2.2.6                ← NumPy 2.x！np.float/np.int/np.bool 已删（Phase 0.2 grep 修补）
torch==2.4.1+cu124
torch-geometric==2.7.0
lightning==2.6.0.dev20250810      ← 注意：双版本共存
pytorch-lightning==2.5.5          ← Exp2 大概率用这个
hydra-core==1.3.2
omegaconf==2.3.0
pandas==2.3.3
scikit-learn==1.7.2         ← Phase 0.3 验证 RobustScaler joblib unpickle 是否有 warning
scipy==1.15.3
joblib==1.5.2
```

### 2.3 已识别的 4 个版本兼容性风险（Phase 0 必须验证）

| # | 风险 | 现象 | 应对 |
|---|------|------|------|
| R1 | Pytorch-Lightning 1.x → 2.5.5 API breakage | `Trainer(gpus=N)` 旧参数名报错；`on_train_epoch_end(outputs=...)` outputs 参数被移除；`LightningDataModule` setup() 签名调整 | grep Exp2 PL import 模式 + 已知 breakage 列表对照修 |
| R2 | NumPy 2.x 删除别名 | `AttributeError: module 'numpy' has no attribute 'float'` | grep 替换：`np.float`→`np.float64`、`np.int`→`np.int64`、`np.bool`→`bool`、`np.long`→`np.int64`、`np.complex`→`np.complex128`、`np.object`→`object` |
| R3 | Pymatgen 2025.10.7 Cython buffer bug 状态 | 本地 Phase D v1 在 multi-site `find_points_in_spheres` 上 100% 失败 | Phase 0.1 用 5 个 multi-site 样本对比 Phase A 已存盘 distances；通过则用 pymatgen，失败则启用 brute-force fallback |
| R4 | sklearn 跨版本 unpickle | `RobustScaler` joblib 在 sklearn 1.7.2 unpickle 可能 FutureWarning 或属性缺失 | Phase 0.3 try-load + transform sanity；失败则用 `feff_feature_stats.csv` 现场重建 |

---

## 3. Phase 0：环境验证（任何代码改动之前必跑）

> **闸门规则**：Phase 0 任何一项失败 → 停下来汇报 Main Agent 3，**不要自行决策推进**。

### 3.1 Phase 0.1 — Pymatgen Cython sanity（5 multi-site 样本对比）

**目的**：判断服务器 pymatgen 2025.10.7 + numpy 2.2.6 是否复现 Phase D v1 的 buffer dtype 失败。

**前置**：`shell_boundaries.pkl` 已上传（含 Phase A 算的 distances 真值）。

**意图**：
1. 从 `shell_boundaries.pkl` 选 5 个 `n_center_sites >= 2` 的样本
2. 对每个样本：load POSCAR → SpacegroupAnalyzer → primitive → 找 `species == center_element` 的第一个 site → `prim.get_neighbors(prim[center_idx], r=10.0)`
3. 排序 distances，取前 20 个，与 `shells[sname]["distances"][:20]` 比对
4. 期望 `max|diff| < 1e-3 Å`（float32 量级误差）

**失败处理**：
- 如果 `get_neighbors` 抛 `ValueError: Buffer dtype mismatch` 或类似 → 启用 brute-force fallback（见 §3.1.1）
- 如果 distances 偏差 > 1e-3 Å → 报告 Main Agent，可能是 SpacegroupAnalyzer 的 primitive 化与 Phase A 不一致（symprec 不同？）

#### 3.1.1 Brute-force fallback（如果 Phase 0.1 失败才启用）

新建 `/home/tcat/diffcsp_exp4/code/exp4_utils/neighbors.py`，封装一个纯 numpy 周期镜像枚举的邻居查找函数：

**函数签名意图**：
- 输入：`Structure` 对象、`center_idx`（int）、`cutoff`（float, Å）
- 输出：`(coords (M, 3), distances (M,), species_Z (M,))`，其中 M = cutoff 内邻居数（不含中心自身）
- 算法：
  1. `lattice.matrix` (3, 3) → 三个方向 perpendicular distance（用 `lattice.matrix` 与法向量点积）
  2. 每方向需要的镜像层数 `n_i = ceil(cutoff / d_perp_i)`
  3. `meshgrid(range(-n_i, n_i+1), ...)` 生成 (Nx*Ny*Nz, 3) 镜像偏移
  4. 所有原子坐标 × 所有镜像 → (n_atoms × n_images, 3) 候选位置
  5. 与中心位置距离过滤 ≤ cutoff，去掉中心自身
  6. 返回 (coords, dists, Z)

**参考来源**：用户本地 `step2_5d_full_multisite_tag_v2.py` 已有同算法实现（已 sanity check 5/5 max diff = 0.00000 Å）。可让用户 scp 该脚本到服务器作为参考；或直接根据上述算法重写。

### 3.2 Phase 0.2 — NumPy 2.x 兼容性 grep

**意图**：在写任何新代码之前，先扫描 Exp2 仓库已知的 numpy 别名：

```bash
cd /home/tcat/diffcsp_exp4/code
grep -rn -E 'np\.(float|int|bool|object|long|complex)\b' --include="*.py" . > /tmp/numpy_legacy.txt
wc -l /tmp/numpy_legacy.txt
```

**处理**：
- 0 行 → 通过，进 Phase 0.3
- > 0 行 → 逐行替换：

| 旧 | 新 |
|----|----|
| `np.float` | `np.float64` |
| `np.int` | `np.int64` |
| `np.bool` | `bool` |
| `np.long` | `np.int64` |
| `np.complex` | `np.complex128` |
| `np.object` | `object` |

注意：`np.float32` / `np.int64` 等带宽度的别名**没动**，不要替换。

### 3.3 Phase 0.3 — sklearn RobustScaler unpickle 测试

**意图**：用服务器 sklearn 1.7.2 加载本地 fit 的 scaler，确认 transform 输出形状正确且数值合理。

**步骤**：
1. `joblib.load("feff_feature_scaler.pkl")` 用 `warnings.catch_warnings(record=True)` 包裹
2. 检查 `sk.center_.shape == (74,)` 且 `sk.scale_.shape == (74,)`
3. `sk.transform(np.zeros((3, 74), dtype=np.float32))` 输出形状 `(3, 74)`，无异常
4. 期望对零向量输入：输出 `≈ -center_/scale_`（每列 robust standardized）

**失败处理**：
- `InconsistentVersionWarning` / `FutureWarning` → 输出 sanity check 通过即可，记录 warning 文本到 phase0_report
- `AttributeError` 或属性缺失（如新 sklearn 加了 `n_features_in_` 但 pkl 里没有）→ 现场重建：

  ```python
  from sklearn.preprocessing import RobustScaler
  import numpy as np, pandas as pd, joblib
  stats = pd.read_csv("feff_feature_stats.csv")
  # stats 列：median, iqr, q1, q3, min, max, n_nan_before_impute（顺序与 feff_feature_names.txt 对齐）
  sk = RobustScaler()
  sk.center_ = stats["median"].values.astype(np.float64)
  sk.scale_  = stats["iqr"].values.astype(np.float64)
  sk.n_features_in_ = 74
  joblib.dump(sk, "feff_feature_scaler_regenerated.pkl")
  ```
  然后用 regenerated 版本继续。

### 3.4 Phase 0.4 — Lightning 双版本冲突排查

**意图**：判断 Exp2 实际 import 哪个 lightning，并诊断已知 API breakage。

**步骤**：
1. ```bash
   grep -rn -E "import (pytorch_lightning|lightning)\b|from (pytorch_lightning|lightning)\.?" \
     --include="*.py" /home/tcat/diffcsp_exp4/code/ > /tmp/lightning_imports.txt
   ```
2. 分类：
   - 全部 `pytorch_lightning` → 用 2.5.5（推荐，迁移成本最小）
   - 混用 `lightning` 和 `pytorch_lightning` → 统一改成 `pytorch_lightning`
   - 全部 `lightning` → 评估是否切换到 2.5.5，或保留 2.6.0.dev（开发版，更激进，更可能踩坑）

3. PL 1.x → 2.5 已知 breakage 速查（grep + 修补）：

| 旧 (1.x) | 新 (2.5.5) |
|---------|------------|
| `Trainer(gpus=N)` | `Trainer(devices=N, accelerator='gpu')` |
| `Trainer(tpu_cores=...)` | `Trainer(devices=..., accelerator='tpu')` |
| `Trainer(progress_bar_refresh_rate=...)` | 移除，用 `RichProgressBar` callback |
| `on_train_epoch_end(self, trainer, pl_module, outputs)` | outputs 参数移除 |
| `pl.Callback` 仍可用 | 检查 hook 签名 |
| `LightningModule.training_epoch_end()` | 移除，改用 `on_train_epoch_end()` + 手动 accumulator |
| `Trainer.test()` 自动 load best ckpt | 不再自动，需显式 `ckpt_path="best"` |

**失败处理**：发现非 trivial breakage（如自定义 LightningModule 的 training_epoch_end 用法）→ 汇报 Main Agent 3，**不要自行重构 Exp2 训练循环**，等 Step 4 一起修。

### 3.5 Phase 0.5 — 4 数据源 key 对齐 sanity（最关键的一项）

**意图**：在写任何 Dataset 代码之前，确认 `data_inventory_v2.csv`、`spectra_*.pkl`、`feff_features_imputed.pkl`、`shell_boundaries.pkl` 用同一个 `sample_name` 作 key 且全部能 lookup 到。

**步骤**：
```python
import pickle, pandas as pd, random
DATA = "/home/tcat/diffcsp_exp4/data"

inv    = pd.read_csv(f"{DATA}/data_inventory_v2.csv")            # 75,637 rows
feff   = pd.read_pickle(f"{DATA}/feff_features_imputed.pkl")     # 128,382 rows (v1 全集)
shells = pickle.load(open(f"{DATA}/shell_boundaries.pkl", "rb")) # 128,382 keys (v1 全集)

assert len(inv) == 75637
assert len(feff) == 128382
assert len(shells) == 128382

random.seed(42)
report = {}
for split in ["train", "val", "test", "holdout"]:
    samples = pd.read_csv(f"{DATA}/{split}_samples_v2.csv")
    spec    = pickle.load(open(f"{DATA}/spectra_{split}.pkl", "rb"))
    pick = samples.sample(min(100, len(samples)), random_state=42)
    miss = []
    for sn in pick.sample_name:
        if sn not in spec["name_to_idx"]: miss.append(("spec",  sn))
        if sn not in feff.index:           miss.append(("feff",  sn))
        if sn not in shells:               miss.append(("shell", sn))
    report[split] = (len(pick), len(miss), miss[:5])
    print(f"{split}: 100/{len(pick)} sampled, {len(miss)} miss, first 5: {miss[:5]}")
```

**期望**：4 个 split 全部 0 miss。

**失败处理**：任何 miss → 立刻停，汇报 Main Agent 3。这意味着 Step 1/2/2.5 的 sample_name 命名口径在某处分裂了，必须查清不能猜。

### 3.6 Phase 0 完成汇报模板

写入 `/home/tcat/diffcsp_exp4/logs/step3_phase0_report.txt`：

```
Phase 0.1 pymatgen sanity:    [PASS / FAIL]  max diff = ___ Å, fallback enabled = [yes/no]
Phase 0.2 numpy 2.x grep:     [PASS / FAIL]  legacy hits = ___, fixes applied = ___
Phase 0.3 sklearn unpickle:   [PASS / FAIL / REBUILT]  warnings = ___
Phase 0.4 lightning conflict: [PASS / FAIL]  Exp2 imports = pytorch_lightning|lightning, breakages = ___
Phase 0.5 key alignment:      [PASS / FAIL]  miss per split: train=__, val=__, test=__, holdout=__
```

全部 PASS / REBUILT 才进 Phase 1。

---

## 4. Phase 1：数据上传 + 仓库搭建 + 目录结构

### 4.1 服务器目标目录结构（最终态）

```
/home/tcat/diffcsp_exp4/
├── data/                                    (~700 MB)
│   ├── data_inventory_v2.csv                33.5 MB ★主索引
│   ├── train_samples_v2.csv                 3.3  MB ★
│   ├── val_samples_v2.csv                   0.42 MB ★
│   ├── test_samples_v2.csv                  0.24 MB ★
│   ├── holdout_samples_v2.csv               0.17 MB ★ (Step 4 训练时不读)
│   ├── feff_features_imputed.pkl            40.3 MB ★
│   ├── feff_feature_scaler.pkl              1.6  KB ★
│   ├── feff_feature_names.txt               1.0  KB
│   ├── feff_feature_stats.csv               6.8  KB (Phase 0.3 备用)
│   ├── spectra_train.pkl                    148.4 MB ★
│   ├── spectra_val.pkl                      18.7  MB ★
│   ├── spectra_test.pkl                     11.1  MB ★
│   ├── spectra_holdout.pkl                  7.4   MB ★ (Step 4 训练时不读)
│   ├── shell_boundaries.pkl                 369.5 MB ★
│   ├── site_equivalence_tag.csv             9.5  MB (归档)
│   ├── incompat_pool.csv                    3.3  MB (封存,绝对不 load)
│   └── MP_all_POSCAR_flat/                  (从 /home/tcat/mp-9_POSCAR mv 过来)
│
├── code/                                    (Exp2 仓库 fork + Exp4 改动)
│   ├── exp4_utils/
│   │   ├── __init__.py
│   │   └── neighbors.py                     (brute-force fallback, 仅 Phase 0.1 失败启用)
│   ├── xas_local_dataset_v2.py              (新建,改自 xas_local_dataset.py)
│   ├── spectrum_encoder.py                  (Exp2 拷贝 + 一行改)
│   ├── diffusion_w_type_xas.py              (Exp2 拷贝 + 路径常量改)
│   ├── (其余 Exp2 文件原样保留：gnn.py, cspnet.py, diff_utils.py, run.py, ...)
│   └── conf/                                (hydra config,路径常量改)
│
├── checkpoints/                             (Step 4 训练后产出,目前为空)
└── logs/
    ├── step3_phase0_report.txt
    └── step3_forward_test_log.txt
```

### 4.2 数据上传命令（用户在 Windows PowerShell 执行）

**前置**：用户密码登录，无 ssh key。每次 scp 会要求输入密码。建议**一次性 scp 整个 data 目录**，只输一次密码：

```powershell
# 在 Windows 本地，准备一个临时上传目录
$UPLOAD = "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\_upload_to_server"
New-Item -ItemType Directory -Force -Path $UPLOAD

# 拷贝所有要传的文件到临时目录(避免 scp 多个不同位置的文件要输多次密码)
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\data_inventory.csv" $UPLOAD\ -ErrorAction SilentlyContinue
# ↑ 注意：data_inventory.csv (v1) 是 56.8 MB,Step 3+ 不用,可不传

Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\feff_features_imputed.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\feff_feature_scaler.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\step1_3_feff_feature_names.txt" "$UPLOAD\feff_feature_names.txt"
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\feff_feature_stats.csv" $UPLOAD\

Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2\spectra_train.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2\spectra_val.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2\spectra_test.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2\spectra_holdout.pkl" $UPLOAD\

Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\data_inventory_v2.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\train_samples_v2.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\val_samples_v2.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\test_samples_v2.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\holdout_samples_v2.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\shell_boundaries.pkl" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\site_equivalence_tag.csv" $UPLOAD\
Copy-Item "C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2_5\incompat_pool.csv" $UPLOAD\

# 一次性 scp 整个目录(只输一次密码)
ssh tcat@scsmlnprd02.its.auckland.ac.nz "mkdir -p /home/tcat/diffcsp_exp4/data /home/tcat/diffcsp_exp4/code /home/tcat/diffcsp_exp4/checkpoints /home/tcat/diffcsp_exp4/logs"
scp $UPLOAD\* tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/data/

# Exp2 仓库整体上传
scp -r C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\* tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/
```

**预计耗时**：~700 MB + Exp2 仓库（取决于校园网速）= 5-15 min。

### 4.3 服务器端整理命令

用户上传完后，在服务器上跑：

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
conda activate jhub_env
cd /home/tcat/diffcsp_exp4

# 1. POSCAR 路径整合(从用户原上传位置 mv 到 data/)
ls /home/tcat/mp-9_POSCAR | head -5    # 确认存在
ls /home/tcat/mp-9_POSCAR | wc -l      # 确认数量(应 ~41,431)
mv /home/tcat/mp-9_POSCAR /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat
ls /home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat | wc -l   # 再次确认

# 2. 检查 data 目录
du -sh /home/tcat/diffcsp_exp4/data/                          # 期望 ~700 MB + POSCAR 体积
ls -la /home/tcat/diffcsp_exp4/data/ | head -30

# 3. 创建 exp4_utils 包
mkdir -p /home/tcat/diffcsp_exp4/code/exp4_utils
touch /home/tcat/diffcsp_exp4/code/exp4_utils/__init__.py

# 4. 空间核实
df -h ~                                                       # 确认 /home/tcat 还有 ≥30 GB
free -h                                                       # 确认 RAM 可用 ≥10 GB(训练前 cache 用)
```

### 4.4 训练前数据 cache（仅 Step 4 启动时执行，不是现在）

```bash
# Step 4 启动时执行(此处只是说明,Step 3 不跑)
mkdir -p /tmp/diffcsp_cache
cp -r /home/tcat/diffcsp_exp4/data/* /tmp/diffcsp_cache/
# Step 4 训练读 /tmp/diffcsp_cache/(tmpfs RAM,快很多)
```

**注意**：POSCAR 目录 ~600 MB（41K 文件），cache 时间较长但只跑一次。

---

## 5. Phase 2：Exp2 代码仓库审计与改动定位

### 5.1 必改文件清单

| 文件 | 改动级别 | 改什么 |
|------|---------|--------|
| `xas_local_dataset.py` | **重写**（产出 `xas_local_dataset_v2.py`，原文件保留） | 见 §6 |
| `spectrum_encoder.py` | **一行** | feff 分支 `nn.Linear(73, ...)` → `nn.Linear(74, ...)` |
| `diffusion_w_type_xas.py` | **路径常量** | 数据路径指向 `/tmp/diffcsp_cache/`（Step 4 启动时）；`cost_lattice = 0` 必须保持；不加 TypeClassifier |
| `conf/*.yaml`（Hydra config） | **路径常量** | data_dir、poscar_dir、log_dir 等指向新路径 |
| `run.py` | **路径常量 / Trainer 参数** | 视 PL 版本调整（Phase 0.4 输出决定）|

### 5.2 不要碰的文件（直接复用 Exp2）

- `gnn.py` / `cspnet.py` / `diff_utils.py`：扩散模型框架
- 数学层、采样器、loss 函数：完全继承
- 任何与 Fe-only 无关的工具函数

### 5.3 新建文件清单

| 文件 | 内容 |
|------|------|
| `exp4_utils/__init__.py` | 空 |
| `exp4_utils/neighbors.py` | brute-force 邻居查找（**仅 Phase 0.1 失败时启用**）|

### 5.4 审计输出

完成审计后写 `/home/tcat/diffcsp_exp4/logs/step3_audit.txt`，记录：
- Exp2 仓库实际包含哪些文件（`find . -name "*.py" | sort`）
- 上面"必改文件"是否都存在
- 是否发现额外文件需要改（例如 dataset 注册表、import path）

---

## 6. Phase 3：Dataset 改造（最大改动）

> **重要**：以下章节描述**改动意图**，不写代码。Sub-Agent（你）负责实现细节，但所有"约束"必须满足。

### 6.1 文件命名

- 源文件：`xas_local_dataset.py`（Exp2 原版，**不删**）
- 新文件：`xas_local_dataset_v2.py`（Exp4 v2 版本）
- 类名：`XasLocalDatasetV2`（与 Exp2 的 `XasLocalDataset` 区分）
- import 入口：训练脚本里改成 `from xas_local_dataset_v2 import XasLocalDatasetV2`

### 6.2 路径常量（在文件顶部定义，不要散落）

```python
DATA_DIR    = "/tmp/diffcsp_cache"          # Step 4 启动时 cache 后的路径
POSCAR_DIR  = f"{DATA_DIR}/MP_all_POSCAR_flat"
L_VIRTUAL   = 6.0                           # 虚拟晶格边长 (Å) — 不可变
N_NEIGHBORS = 20                            # — 不可变
CUTOFF_R    = 10.0                          # 邻居搜索半径 (Å),与 Phase A 一致
SYMPREC     = 0.1                           # SpacegroupAnalyzer 的 symprec — 不可变
```

> **Step 3 forward 测试时**：DATA_DIR 暂用 `/home/tcat/diffcsp_exp4/data`（还没 cache）。Step 4 启动时改 `/tmp/diffcsp_cache`。建议从环境变量读：`DATA_DIR = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")`。

### 6.3 `__init__(self, split, data_dir=DATA_DIR, ...)` 加载内容

在 `__init__` 一次性加载（避免 `__getitem__` 反复 load）：

| 属性 | 来源 | 形态 | 备注 |
|------|------|------|------|
| `self.samples` | `pd.read_csv(f"{data_dir}/{split}_samples_v2.csv")` | DataFrame, 4 列 `[mp_id, center_element, sample_name, site_equivalence_tag]` | **训练时只用 train + val，holdout/test 由调用方决定** |
| `self.spectra` | `pickle.load(f"{data_dir}/spectra_{split}.pkl")` | dict（含 `xmu`, `chi1`, `name_to_idx`, `E0`, ...）| 注意：spectra pkl 是 **v1 全集**，v2 sample_name 是其子集 |
| `self.feff` | `pd.read_pickle(f"{data_dir}/feff_features_imputed.pkl")` | DataFrame index=sample_name, (128382, 74) float32 | v1 全集 |
| `self.scaler` | `joblib.load(f"{data_dir}/feff_feature_scaler.pkl")` | RobustScaler | Phase 0.3 通过的版本 |
| `self.shells` | `pickle.load(f"{data_dir}/shell_boundaries.pkl")` | dict[sample_name → 9 字段] | v1 全集；Step 5 评估用，Step 3 forward 仅做 metadata 透传 |
| `self.poscar_dir` | 参数 `POSCAR_DIR` | str | 用 `f"{poscar_dir}/{mp_id}_POSCAR"` 拼路径 |

**断言**（`__init__` 末尾，启动期一次性验证）：
```python
assert len(self.samples) > 0
assert "name_to_idx" in self.spectra and "xmu" in self.spectra and "chi1" in self.spectra
assert self.feff.shape[1] == 74
# 抽 5 个样本验证 4 数据源 lookup（防御性，Phase 0.5 已做但启动期再过一遍）
for sn in self.samples.sample_name.head(5):
    assert sn in self.spectra["name_to_idx"], sn
    assert sn in self.feff.index, sn
    assert sn in self.shells, sn
```

### 6.4 `__getitem__(self, idx)` 实现意图

```
row = self.samples.iloc[idx]
sname, mp_id, center_elem = row.sample_name, row.mp_id, row.center_element

# 1. 谱（已 Step 2 预处理，直接索引）
spec_idx = self.spectra["name_to_idx"][sname]
xmu  = torch.from_numpy(self.spectra["xmu"][spec_idx]).float()    # (150,)
chi1 = torch.from_numpy(self.spectra["chi1"][spec_idx]).float()   # (200,)

# 2. FEFF + RobustScaler transform（注意：Step 1 只 fit 没 transform，Step 3 在这里做）
feff_raw    = self.feff.loc[sname].values.reshape(1, -1).astype(np.float32)   # (1, 74)
feff_scaled = self.scaler.transform(feff_raw).astype(np.float32).squeeze()    # (74,)
feff = torch.from_numpy(feff_scaled).float()

# 3. POSCAR → primitive → 中心位点 → 20 邻居
poscar_path = os.path.join(self.poscar_dir, f"{mp_id}_POSCAR")
structure = Structure.from_file(poscar_path)
prim = SpacegroupAnalyzer(structure, symprec=SYMPREC).get_primitive_standard_structure()

# 中心位点选择：第一个 species == center_elem 的 site
# （Option D 已剔除 incompat，剩下样本第一位点 ≈ site-averaged 在容差内，无需多位点分支）
center_idx = next(i for i, site in enumerate(prim) if site.specie.symbol == center_elem)
center_cart = prim[center_idx].coords    # (3,)

# 邻居（Phase 0.1 通过则用 pymatgen，否则 fallback）
try:
    nbrs = prim.get_neighbors(prim[center_idx], r=CUTOFF_R)
    coords  = np.array([n.coords for n in nbrs])         # (M, 3) Cartesian
    dists   = np.array([n.nn_distance for n in nbrs])    # (M,)
    species = np.array([n.specie.Z for n in nbrs])       # (M,)
except Exception:
    from exp4_utils.neighbors import find_neighbors_brute
    coords, dists, species = find_neighbors_brute(prim, center_idx, CUTOFF_R)

# 距离排序，取前 20
order   = np.argsort(dists)[:N_NEIGHBORS]
coords  = coords[order]                                  # (20, 3)
species = species[order]                                 # (20,)

# 4. Cartesian → 虚拟晶格分数坐标
frac_coords = (coords - center_cart) / L_VIRTUAL         # (20, 3)
frac_coords -= np.round(frac_coords)                     # min-image，∈ [-0.5, 0.5]
# 注意：forward() 不做 % 1.，由 Dataset 这里保证 [-0.5, 0.5]

frac_coords_t = torch.from_numpy(frac_coords).float()    # (20, 3)
atom_types    = torch.from_numpy(species).long()         # (20,)

# 5. 评估元数据透传（Step 5 用）
shell_info = self.shells[sname]
eval_cutoff          = float(shell_info["eval_cutoff"])
eval_cutoff_fallback = bool(eval_cutoff > 9.5)   # 启发式：> 9.5 Å 视为 fallback
n_center_sites       = int(shell_info["n_center_sites"])

return {
    "xmu":           xmu,                # (150,)
    "chi1":          chi1,               # (200,)
    "feff":          feff,               # (74,)
    "frac_coords":   frac_coords_t,      # (20, 3)
    "atom_types":    atom_types,         # (20,)
    "sample_name":   sname,              # str
    "mp_id":         mp_id,              # str
    "center_element": center_elem,       # str
    "eval_cutoff":   eval_cutoff,        # float
    "eval_cutoff_fallback": eval_cutoff_fallback,
    "n_center_sites": n_center_sites,
    "site_equivalence_tag": row.site_equivalence_tag,
}
```

### 6.5 关键约束（不能违反）

1. **L=6 不变**，`frac -= np.round(frac)` 必须执行（Exp2 数学一致）
2. **N_NEIGHBORS=20 固定**，少于 20 邻居的样本：理论上不应出现（CUTOFF=10 Å 内 ~30-100 邻居），如真出现报错而非 padding
3. **center_idx 取第一个匹配 species 的 site**，Option D 已剔除 incompat，无需多位点分支或 site-averaging
4. **forward() 内不做 `% 1.`**：分数坐标在 Dataset 出口已 ∈ [-0.5, 0.5]，模型 forward 信任这个不变量
5. **TypeClassifier 不加**（Exp3 已证伪）
6. **不要 `if center_element == "Fe"` 这种硬编码**，center_element 永远从 `row.center_element` 读

### 6.6 `__getitem__` 的物理注意事项（Sub-Agent 知道即可，不改）

虚拟晶格 L=6 + min-image 折叠后，理论上 d_max(Cartesian) > L/2 = 3 Å 的邻居会被错误折叠。Step 2.5 的 eval_cutoff 中 5-10% 样本 fallback 到 ~10 Å，意味着这些样本 d20 可能 > 3 Å。这是**继承 Exp2 的设计妥协**，不在 Step 3 修复。Step 5 评估时通过 `eval_cutoff_fallback` 标记单独统计。

---

## 7. Phase 4：SpectrumEncoder 改动（一行）

### 7.1 改动位置

文件：`spectrum_encoder.py`（Exp2 原版拷贝过来）

### 7.2 改动内容

找到 feff 分支的第一层 Linear，将输入维度从 73 改为 74：

```python
# Exp2 原版（保留作对比）：
# self.feff_proj = nn.Sequential(
#     nn.Linear(73, hidden_dim),
#     ...
# )

# Exp4 修改后：
self.feff_proj = nn.Sequential(
    nn.Linear(74, hidden_dim),
    ...
)
```

**只改这一处**。其他所有维度（xmu 150 → conv → ..., chi1 200 → conv → ..., 三路 concat → 256）**保持不变**。

### 7.3 验证

```python
import torch
from spectrum_encoder import SpectrumEncoder
enc = SpectrumEncoder()
xmu  = torch.randn(4, 150)
chi1 = torch.randn(4, 200)
feff = torch.randn(4, 74)
out = enc(xmu, chi1, feff)
assert out.shape == (4, 256), out.shape
```

如果 SpectrumEncoder 接口与上面不同（例如 forward 接 dict）→ 按实际接口调用，但**维度合约不变**：xmu 150 + chi1 200 + feff 74 → latent 256。

---

## 8. Phase 5：diffusion_w_type_xas.py 改动（路径常量）

### 8.1 改动位置

文件：`diffusion_w_type_xas.py`（Exp2 拷贝）

### 8.2 改动内容

**只改两类东西**：

1. **数据路径常量**（顶部硬编码 / Hydra config 读取均可）：
   - 旧：Exp2 的 Fe-oxide 路径
   - 新：`/tmp/diffcsp_cache/`（Step 4 cache 后）；Step 3 forward 测试期可用 `/home/tcat/diffcsp_exp4/data/`

2. **Dataset 类切换**：
   - 旧：`from xas_local_dataset import XasLocalDataset`
   - 新：`from xas_local_dataset_v2 import XasLocalDatasetV2`

### 8.3 不改的东西

- `cost_lattice = 0`（必须保持）
- 扩散数学（β schedule、DDPM/DDIM 参数）
- forward / training_step / validation_step 主体逻辑（**除非 PL 版本不兼容触发 Phase 0.4 的修补**）
- **不加 TypeClassifier**（Exp3 已证伪）

### 8.4 Hydra config

如果 Exp2 用 hydra config 管理路径（`conf/*.yaml`），相应字段也要改：

```yaml
# 旧
data:
  data_dir: /path/to/exp2/fe_oxide
  ...

# 新
data:
  data_dir: ${oc.env:EXP4_DATA_DIR,/home/tcat/diffcsp_exp4/data}
  poscar_dir: ${data.data_dir}/MP_all_POSCAR_flat
  ...
```

---

## 9. Phase 6：前向测试协议

### 9.1 测试目的

在不实际训练的前提下验证：
- Dataset 能稳定 `__getitem__`，无形状错位
- DataLoader collate 通过
- forward pass 数值范围合理（无 NaN/Inf）
- backward pass 梯度可计算
- bf16 路径在 GPU 上跑通

### 9.2 测试脚本（建议放 `code/forward_test.py`）

**Phase 6.1 — Dataset 单点测试**

```python
from xas_local_dataset_v2 import XasLocalDatasetV2
ds = XasLocalDatasetV2(split="train", data_dir="/home/tcat/diffcsp_exp4/data")
print(f"Dataset size: {len(ds)}")  # 期望 60,507

sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")
# 期望:
#   xmu: (150,) float32
#   chi1: (200,) float32
#   feff: (74,) float32
#   frac_coords: (20, 3) float32, all values in [-0.5, 0.5]
#   atom_types: (20,) int64, all in [1, 109]

# Sanity: frac 范围
import torch
fc = sample["frac_coords"]
assert fc.min() >= -0.5 - 1e-6 and fc.max() <= 0.5 + 1e-6, (fc.min(), fc.max())
```

**Phase 6.2 — DataLoader collate 测试**

```python
from torch.utils.data import DataLoader

# 自定义 collate（因为 dict 含 string 字段）
def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float, bool)):
            out[k] = torch.tensor(vals)
        else:  # string list
            out[k] = vals
    return out

dl = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=collate)
batch = next(iter(dl))
print(batch["xmu"].shape)        # (4, 150)
print(batch["frac_coords"].shape) # (4, 20, 3)
print(batch["sample_name"])       # list of 4 strings
```

**Phase 6.3 — SpectrumEncoder 前向**

```python
from spectrum_encoder import SpectrumEncoder
enc = SpectrumEncoder().eval()
with torch.no_grad():
    z = enc(batch["xmu"], batch["chi1"], batch["feff"])
print(z.shape)         # (4, 256)
print(z.mean(), z.std())  # 期望 |mean| < 5, std ∈ [0.1, 5]
assert not torch.isnan(z).any()
```

**Phase 6.4 — 完整 model forward + loss + backward（CPU）**

```python
from diffusion_w_type_xas import DiffusionWithTypeXAS  # 实际类名按 Exp2 对齐
model = DiffusionWithTypeXAS(...).train()

# 模拟一个 training step
loss = model.training_step(batch, batch_idx=0)
print(f"Loss: {loss.item():.4f}")
# 期望范围: 2-6（DiffCSP 随机初始化典型值），不是 NaN/Inf

loss.backward()
# 检查 grad
total_grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Total grad norm: {total_grad_norm:.4f}")
assert total_grad_norm > 0 and total_grad_norm < 1e4
```

**Phase 6.5 — GPU bf16 测试**

```python
model = model.to("cuda:0").to(torch.bfloat16)
batch_gpu = {k: (v.to("cuda:0").to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                 else v.to("cuda:0") if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
loss = model.training_step(batch_gpu, batch_idx=0)
print(f"GPU bf16 loss: {loss.item():.4f}")
loss.backward()
# 检查无 NaN
for n, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"NaN grad in {n}")
        break
else:
    print("✓ No NaN gradients")
```

### 9.3 期望结果汇总

| 测试 | 期望 |
|------|------|
| Dataset[0] | dict with all expected keys, frac ∈ [-0.5, 0.5] |
| DataLoader collate | 4 samples, no error |
| SpectrumEncoder forward | (4, 256), no NaN |
| CPU forward+backward | loss ∈ [2, 6], grad_norm ∈ (0, 1e4), no NaN |
| GPU bf16 forward+backward | loss 类似 CPU 范围（bf16 精度允许 ±10% 漂移），no NaN |

### 9.4 写日志到 `logs/step3_forward_test_log.txt`

```
=== Phase 6.1 Dataset[0] ===
shapes: xmu=(150,) chi1=(200,) feff=(74,) frac_coords=(20,3) atom_types=(20,)
frac range: [-0.4823, 0.4912]
[PASS]

=== Phase 6.2 DataLoader collate (bs=4) ===
[PASS]

=== Phase 6.3 SpectrumEncoder forward ===
output: (4, 256), mean=0.0231, std=0.8743
[PASS]

=== Phase 6.4 CPU full forward+backward ===
loss = 4.2317
grad_norm = 12.45
[PASS]

=== Phase 6.5 GPU bf16 forward+backward ===
loss = 4.1029 (bf16)
grad_norm = 12.78
[PASS]

Total wall-clock: __ s
```

---

## 10. Phase 7：训练前 checklist（Step 4 启动闸门）

完成 Phase 0-6 后，自查以下 10 项，全部 ✓ 才汇报 Main Agent 3：

```
[ ]  1. 服务器 /home/tcat/diffcsp_exp4/ 目录结构与 §4.1 一致
[ ]  2. POSCAR 在 data/MP_all_POSCAR_flat/ 下，文件数 ~41,431
[ ]  3. /home/tcat 剩余空间 ≥ 30 GB，free -h RAM ≥ 10 GB
[ ]  4. Phase 0 五项 sanity 全 PASS（含 fallback / rebuild 标记）
[ ]  5. xas_local_dataset_v2.py 写完，类名 XasLocalDatasetV2，路径常量从环境变量读
[ ]  6. spectrum_encoder.py feff Linear 是 Linear(74, ...)（git diff Exp2 应为单行）
[ ]  7. diffusion_w_type_xas.py 路径常量更新，cost_lattice=0 保持，未加 TypeClassifier
[ ]  8. forward_test.py 五个 phase 全跑过，loss/grad/no-NaN 全 PASS
[ ]  9. 没有任何代码触碰 incompat_pool.csv
[ ] 10. 没有任何代码触碰 holdout_samples_v2.csv 或 spectra_holdout.pkl
       （除了 Phase 0.5 的 4-source key alignment sanity 之外）
```

---

## 11. 完成后汇报模板

完成全部 Phase 后，按以下格式汇报给 Main Agent 3：

```
# Step 3 Sub-Agent 完成汇报

## Phase 0 环境验证
0.1 pymatgen sanity: [PASS / FAIL]
    - 5 multi-site 样本 max diff: ___ Å
    - fallback enabled: [yes / no]
0.2 numpy 2.x grep: [PASS / FAIL]
    - legacy hits: ___ 个
    - 修复列表: [文件:行号 旧→新]
0.3 sklearn unpickle: [PASS / REBUILT / FAIL]
    - warnings: ___
    - rebuilt scaler: [yes / no]
0.4 lightning conflict: [PASS / FAIL]
    - Exp2 imports: pytorch_lightning / lightning
    - 已知 breakage 命中: ___
0.5 4-source key alignment: [PASS / FAIL]
    - 各 split miss: train=__ val=__ test=__ holdout=__

## Phase 1 数据上传
- /home/tcat/diffcsp_exp4/data 总大小: ___ GB
- POSCAR 文件数: ___
- /home/tcat 剩余空间: ___ GB
- free -h RAM 可用: ___ GB

## Phase 2 Exp2 仓库审计
- Exp2 文件总数: ___
- 必改文件全部存在: [yes / no]
- 额外发现需改的文件: ___

## Phase 3 Dataset 改造
- xas_local_dataset_v2.py: 行数 ___
- 路径常量是否从环境变量读: [yes / no]
- __getitem__ 5 个测试样本 frac ∈ [-0.5, 0.5]: [yes / no]
- 无多位点分支逻辑（无 site-averaging / 第二位点处理）: [yes / no]

## Phase 4 SpectrumEncoder
- feff Linear 输入维度: ___ (期望 74)
- git diff Exp2 行数: ___ (期望 1)

## Phase 5 diffusion 改动
- cost_lattice 值: ___ (期望 0)
- TypeClassifier 是否加: [no]
- 路径常量更新行数: ___

## Phase 6 前向测试
- Dataset[0] PASS: [yes / no]
- DataLoader collate (bs=4) PASS: [yes / no]
- SpectrumEncoder forward PASS: [yes / no]
- CPU full forward+backward: loss=___, grad_norm=___, [PASS/FAIL]
- GPU bf16 forward+backward: loss=___, grad_norm=___, [PASS/FAIL]
- Total wall-clock: ___ s

## Phase 7 Checklist
10 项全 ✓: [yes / no]，未 ✓ 的列出: ___

## Open Questions / Anomalies
（任何不确定、未达预期、需要 Main Agent 决策的事项）

## 资源占用
- Sub-Agent 已耗用 token 估算: ___ k
- 是否接近上下文窗口限制: [yes / no]
```

---

## 12. 你不要做的事（再次强调）

1. ❌ 不要 import `incompat_pool.csv`，不要 load 这 52,745 样本中的任何一个
2. ❌ 不要为"完整性"在 v1 全集上跑（v1 含 incompat，已被 MA2 决策 Option D 剔除）
3. ❌ 不要训练（Step 3 只到 forward+backward 1 step，不进 epoch）
4. ❌ 不要在 Step 3 forward 测试期使用 holdout 数据（除了 Phase 0.5 key alignment sanity）
5. ❌ 不要修改虚拟晶格 L=6、坐标系 [-0.5, 0.5]、N_NEIGHBORS=20、SYMPREC=0.1
6. ❌ 不要加 TypeClassifier（Exp3 已证伪）
7. ❌ 不要在 forward() 里加 `% 1.`（Dataset 已保证 frac ∈ [-0.5, 0.5]）
8. ❌ 不要在 Sub-Agent 范围内做 site-averaging（Option D 路径上不需要）
9. ❌ 不要忽略 Phase 0 任一项失败而硬推
10. ❌ 不要主动安装新包（pip install / conda install），所有依赖必须在 `jhub_env` 现有版本内解决，确实需要新包先汇报 Main Agent

---

## 13. 关键原则（继承 Main Agent 1/2/3 工作哲学）

如果你（Step 3 Sub-Agent）在执行中发现：
- "Main Agent 3 的某个假设是错的"
- "某个改动会触发我没预料的副作用"
- "现有数据有结构问题，按现状跑会得到错误结果"

**不要硬推**。停下来，按以下格式汇报：

1. **承认观察**："我发现 X 与文档假设 Y 不一致"
2. **解释影响**："如果按文档继续，会导致 Z"
3. **给选项**：列 A/B/C 选项，写各自代价
4. **不替 Main Agent 做决定**，让 Main Agent 拍板

参考：Main Agent 2 在 Step 2.5 Phase D 时承认"我之前理解错了 site-specific vs site-averaged"，给用户 4 个选项让她拍板（最终选 Option D），反而比硬推 Option B 省时间。

**诚实 > 流畅**。你对 Main Agent 3 的信任值由"汇报真实"建立，不是由"流程顺畅"建立。

---

*Main Agent 3 撰写，2026-04-25*
