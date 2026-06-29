# EXP4_FILE_INVENTORY.md
# Exp4 全部文件清单与位置

> **撰写者**：Main Agent 2
> **日期**：2026-04-25
> **目的**：Main Agent 3 知道每个文件在哪、是干什么的、什么时候用

---

## 1. 本地 Windows 文件树

```
C:\Users\T-Cat\Desktop\DiffCSP-main\
├── experiment4\
│   ├── data\                                      <- 原始数据
│   │   ├── MP_all_EXAFS_only_chi_csv\
│   │   │   └── MP_all_EXAFS_only_chi_csv\         <- chi.csv 文件
│   │   ├── MP_all_EXAFS_only_csv\
│   │   │   └── MP_all_EXAFS_only_csv\             <- xmu.csv 文件
│   │   ├── POSCAR_zip\
│   │   │   ├── MP_all_POSCAR_flat\                <- {mp_id}_POSCAR 文件
│   │   │   └── missing_poscar_list.csv            <- 失效，不用
│   │   └── feff_features_all_csv_75cols(in).csv
│   │
│   ├── step1\                                     <- Step 1 产出 ✓
│   ├── step2\                                     <- Step 2 产出 ✓
│   └── step2_5\                                   <- Step 2.5 产出 ✓
│
└── experiment2\                                   <- Exp2 代码仓库（继承用）
    └── (Exp2 全部脚本)
```

---

## 2. Step 1 产出（`experiment4\step1\`）

### 2.1 给下游用的（必须保留）

| 文件 | 大小 | 用途 |
|------|------|------|
| `data_inventory.csv` | 56.8 MB | v1 主索引，128,382 行 × 14 列。**Step 3+ 不直接用**，被 v2 取代 |
| `feff_features_imputed.pkl` | 40.3 MB | DataFrame, index=sample_name, (128382, 74) float32。**Step 3 Dataset 直接 load** |
| `feff_feature_scaler.pkl` | 1.6 KB | RobustScaler 对象（fit on v1 train 102,660）。**Step 3 Dataset transform 用** |
| `feff_feature_stats.csv` | 6.8 KB | 74 列 median/iqr/q1/q3/min/max/n_nan_before_impute |
| `step1_3_feff_feature_names.txt` | 1.0 KB | 74 个特征名有序列表，与 pkl 列对齐 |

### 2.2 Split 文件（v1，被 Step 2.5 取代）

| 文件 | 大小 | 用途 |
|------|------|------|
| `train_ids.txt` / `val_ids.txt` / `test_ids.txt` / `holdout_ids.txt` | 共 ~430 KB | mp_id 级 v1 split。**Step 3+ 用 v2 替代** |
| `train_samples.csv` / `val_samples.csv` / `test_samples.csv` / `holdout_samples.csv` | 共 ~5.5 MB | sample 级 v1 split。**Step 3+ 用 v2 替代** |

### 2.3 诊断 / 日志（本地保留即可）

| 文件 | 用途 |
|------|------|
| `step1_excluded_log.csv` (292 KB) | 5,336 条剔除明细 |
| `element_distribution.csv` (2.5 KB) | 88 元素 × split 计数 |
| `step1_summary.txt` | 人类可读汇总 |
| `step1_1_raw_inventory.pkl` (135 MB) | Phase 1 原始 inventory（保留作历史归档）|
| `step1_2/3/4_*.pkl` (各 ~130 MB) | 中间产物，可清理（用户已确认） |

---

## 3. Step 2 产出（`experiment4\step2\`）

| 文件 | 大小 | 用途 |
|------|------|------|
| **`spectra_train.pkl`** | 148.4 MB | **Step 3 训练 DataLoader** |
| **`spectra_val.pkl`** | 18.7 MB | **Step 3 验证** |
| **`spectra_test.pkl`** | 11.1 MB | **Step 4 / 5 测试** |
| **`spectra_holdout.pkl`** | 7.4 MB | **Step 5 holdout 评估**（训练期禁访） |
| `step2_spectra_stats.csv` | <10 KB | 4 split × 8 指标 sanity stats |
| `step2_extrapolation_log.csv` | <5 KB | 外推计数 |
| `step2_summary.txt` | <5 KB | 人类可读汇总 |
| `step2_qc_samples.png` | ~500 KB | 5 样本视觉 QC |

每个 spectra pkl schema：
```python
{
    "sample_names": list[str],         # len = N
    "xmu":          (N, 150) float32,
    "chi1":         (N, 200) float32,
    "name_to_idx":  dict[str, int],
    "E0":           (N,) float32,
    "meta":         {...}              # 处理参数记录
}
```

---

## 4. Step 2.5 产出（`experiment4\step2_5\`）

### 4.1 Step 3+ 必读（v2 数据集核心）

| 文件 | 大小 | 用途 |
|------|------|------|
| **`data_inventory_v2.csv`** | 33.5 MB | **Step 3 主索引**，75,637 行 × 15 列（v1 + site_equivalence_tag） |
| **`train_samples_v2.csv`** | 3.3 MB | 60,507 rows，Step 3 Dataset (train) |
| **`val_samples_v2.csv`** | 0.42 MB | 7,624 rows |
| **`test_samples_v2.csv`** | 0.24 MB | 4,481 rows |
| **`holdout_samples_v2.csv`** | 0.17 MB | 3,025 rows，Step 5 holdout 评估 |
| **`shell_boundaries.pkl`** | 369.5 MB | Step 5 分层 RMSD 评估，每样本 9 字段 schema |

`*_samples_v2.csv` schema：
```
[mp_id, center_element, sample_name, site_equivalence_tag]
```

`shell_boundaries.pkl` schema（dict[sample_name] → 9 字段）：
```python
{
    "threshold":       0.1563,                    # gap threshold (Å)
    "distances":       (N_neighbors,) float32,    # 全部邻居距离，截至 10 Å
    "species_Z":       (N_neighbors,) int8,
    "shell_starts":    (N_shells,) float32,
    "shell_ends":      (N_shells,) float32,
    "shell_n_atoms":   (N_shells,) int32,
    "shell_of_atom":   (N_neighbors,) int32,      # 每个邻居属于第几壳层
    "eval_cutoff":     float,                     # 含 d20 的最小壳层外缘
    "n_center_sites":  int,
}
```

注意：shell_boundaries.pkl 含全 128,382 样本（包括 incompat），Step 3/5 用 `data_inventory_v2.csv` 过滤后再访问。

### 4.2 文档归档（备查/Exp5 用）

| 文件 | 大小 | 用途 |
|------|------|------|
| `site_equivalence_tag.csv` | 9.5 MB | 全 128K 样本的 tag 详情，可追溯 |
| `incompat_pool.csv` | 3.3 MB | 52,745 封存样本，**Exp4 不动**，Step 5 final report 引述 |
| `step2_5g_summary.txt` | 1 KB | Exp4 数据集 name card |

### 4.3 中间产物（可清理）

| 文件 | 大小 | 是否清理 |
|------|------|---------|
| `step2_5_neighbor_distances.pkl` | 196 MB | Phase A 中间产物，可清理 |
| `step2_5_gap_histogram.png` | ~200 KB | 决策时的可视化，归档保留 |
| `step2_5_gap_stats.csv` | ~1 KB | gap 分布统计 |
| `step2_5_candidate_thresholds.csv` | ~1 KB | 5 候选阈值的效果模拟表 |
| `step2_5c_multisite_diagnostic.csv` | <5 KB | Phase C 20 样本诊断 |
| `step2_5f_filter_diagnosis.txt` | <5 KB | Phase F 剔除诊断 |
| 各 phase 的 `*_summary.txt` 和 `*.log` | <50 KB | 报告归档 |

---

## 5. 服务器上传清单（Step 3 启动前）

scp 到 `/home/tcat/diffcsp_exp4/data/`：

### 5.1 必传（Step 3 直接用）

| 文件 | 大小 |
|------|------|
| `data_inventory_v2.csv` | 33.5 MB |
| `train_samples_v2.csv` | 3.3 MB |
| `val_samples_v2.csv` | 0.42 MB |
| `test_samples_v2.csv` | 0.24 MB |
| `holdout_samples_v2.csv` | 0.17 MB |
| `feff_features_imputed.pkl` | 40.3 MB |
| `feff_feature_scaler.pkl` | 1.6 KB |
| `feff_feature_names.txt` | 1.0 KB |
| `spectra_train.pkl` | 148.4 MB |
| `spectra_val.pkl` | 18.7 MB |
| `spectra_test.pkl` | 11.1 MB |
| `spectra_holdout.pkl` | 7.4 MB |
| `shell_boundaries.pkl` | 369.5 MB |

**小计 ~640 MB**

### 5.2 备传（归档用）

| 文件 | 大小 |
|------|------|
| `site_equivalence_tag.csv` | 9.5 MB |
| `incompat_pool.csv` | 3.3 MB |
| `feff_feature_stats.csv` | 6.8 KB |

**小计 ~13 MB**

### 5.3 已上传

`MP_all_POSCAR_flat/`（POSCAR 目录）→ 已上传到 `/home/tcat/mp-9_POSCAR`

⚠️ **注意**：路径需要 mv 到 `/home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/`，否则 Step 3 Dataset 找不到。

### 5.4 不要传

- 所有 v1 split 文件（`*_ids.txt`、`*_samples.csv`）—— 用 v2 替代
- v1 `data_inventory.csv`（v2 已含全部信息）
- 中间产物 pkl（`step1_2/3/4_*.pkl`、`step2_5_neighbor_distances.pkl`）
- raw chi/xmu CSV 目录（已 Step 2 预处理为 pkl）
- raw feff_features csv（已 Step 1 处理为 pkl）

**总上传量** ≈ 640 + 13 = ~650 MB，预计 5-10 分钟（取决于网速）。

---

## 6. 服务器目标目录结构（Step 3 完成后）

```
/home/tcat/diffcsp_exp4/
├── data/                                          (~650 MB)
│   ├── data_inventory_v2.csv
│   ├── train_samples_v2.csv
│   ├── val_samples_v2.csv
│   ├── test_samples_v2.csv
│   ├── holdout_samples_v2.csv
│   ├── feff_features_imputed.pkl
│   ├── feff_feature_scaler.pkl
│   ├── feff_feature_names.txt
│   ├── spectra_train.pkl
│   ├── spectra_val.pkl
│   ├── spectra_test.pkl
│   ├── spectra_holdout.pkl
│   ├── shell_boundaries.pkl
│   ├── site_equivalence_tag.csv          (归档)
│   ├── incompat_pool.csv                 (封存)
│   └── MP_all_POSCAR_flat/               (从 /home/tcat/mp-9_POSCAR mv 过来)
│
├── code/                                          (Step 3 改造)
│   ├── (Exp2 仓库 fork)
│   ├── exp4_utils/                       (新建)
│   │   ├── __init__.py
│   │   └── neighbors.py                  (brute-force fallback if needed)
│   ├── xas_local_dataset_v2.py           (改造自 xas_local_dataset.py)
│   ├── spectrum_encoder.py               (一行 73→74)
│   └── diffusion_w_type_xas.py           (路径常量更新)
│
├── checkpoints/                                   (Step 4 训练产出)
│   ├── best.ckpt
│   └── last.ckpt
│
└── logs/                                          (训练日志)
```

**训练前 cache**（Step 4 启动时）：
```bash
mkdir -p /tmp/diffcsp_cache
cp -r /home/tcat/diffcsp_exp4/data/* /tmp/diffcsp_cache/
# 训练读 /tmp/diffcsp_cache/ (tmpfs RAM, 快很多)
```

---

## 7. 文件依赖图（Step 3 角度）

```
Step 3 Dataset 启动时 load:
                                                                    
┌──────────────────────────┐                                       
│ data_inventory_v2.csv    │ ← 主索引                              
│ {split}_samples_v2.csv   │ ← 当前 split 的 sample 列表          
└──────────────────────────┘                                       
              │                                                     
              ↓                                                     
┌──────────────────────────────┐                                   
│ For each sample_name:        │                                   
│   spectra_{split}.pkl        │ ← xmu, chi1 (preprocessed)       
│   feff_features_imputed.pkl  │ ← raw 74-dim feff               
│   feff_feature_scaler.pkl    │ ← apply scaler in __getitem__   
│   data_inventory_v2.poscar_path → POSCAR file                   
│   shell_boundaries.pkl       │ ← Step 5 evaluation only        
└──────────────────────────────┘                                   
              │                                                     
              ↓                                                     
┌──────────────────────────────┐                                   
│ Output dict (per sample):    │                                   
│   xmu (150,) chi1 (200,)     │                                   
│   feff (74,) (scaled)        │                                   
│   frac_coords (20, 3)        │                                   
│   atom_types (20,)           │                                   
│   sample_name, center_element│                                   
└──────────────────────────────┘                                   
```

---

## 8. 数据 key 对齐 sanity check 清单（Step 3 第一件事）

Step 3 Sub-Agent 写完 Dataset 后，**第一个 forward pass 之前**必须验证：

```python
# 取随机 100 个 v2 sample
samples = pd.read_csv(".../train_samples_v2.csv").sample(100, random_state=0)
spectra = pickle.load(open(".../spectra_train.pkl", "rb"))
feff = pd.read_pickle(".../feff_features_imputed.pkl")
shells = pickle.load(open(".../shell_boundaries.pkl", "rb"))

for sname in samples.sample_name:
    assert sname in spectra["name_to_idx"], f"{sname} not in spectra"
    assert sname in feff.index, f"{sname} not in feff"
    assert sname in shells, f"{sname} not in shells"
print("✓ All 4 sources aligned")
```

如果任一报错 → 数据 inconsistency，回头排查（不要继续训练）。

---

*Main Agent 2 撰写，2026-04-25*
