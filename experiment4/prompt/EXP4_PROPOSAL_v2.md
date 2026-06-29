# EXP4_PROPOSAL_v2.md
# DiffCSP-Experiment4 完整方案（更新版）

> **版本**：v2，覆盖原 EXP4_MAINAGENT_HANDOFF.md（v1）
> **撰写者**：Main Agent 2
> **日期**：2026-04-25
> **状态**：Step 1/2/2.5 已完成。下面是基于实际执行结果更新过的方案。

---

## 1. 任务定义

### 1.1 与 Exp2 的关系

继承 Exp2（Fe 氧化物 XAS → 局部结构）的架构，扩展到 **MP 全元素 EXAFS 数据集**。

| 项目 | Exp2 | **Exp4（更新后）** |
|------|------|------------------|
| 数据集范围 | Fe 氧化物 | MP 全元素 EXAFS |
| 中心元素 | 固定 Fe | 任意元素，从文件名读 |
| 原始样本数 | 18,385 文件夹 → 11,636 化合物 | 133,718 raw → **75,637 final** |
| 元素种数 | 1（Fe） | **88** |
| 训练数据量 | ~7,595 train | **60,507 train** |
| 模型架构 | DiffCSP + 三路 SpectrumEncoder | **完全继承（仅 feff Linear 改 73→74）** |

### 1.2 核心目标

**验证 Exp2 架构在全元素数据集上是否仍能成立，为 Exp5（多视角 attention 聚合）打 baseline**。

不是刷 RMSD。Exp4 vs Exp2 的可比性是核心。

### 1.3 不变量（继承 Exp2，绝对不能改）

- L = 6 Å（虚拟晶格）
- 坐标系 [-0.5, 0.5]，`frac -= np.round(frac)` min-image 折叠
- forward() 无 `% 1.`
- N_NEIGHBORS = 20
- batch_size = 16，lr = 1e-4，bf16，num_workers = 0
- 三路 SpectrumEncoder（xmu 150 + chi1 200 + feff **74** → latent 256）
- DiffCSP 扩散框架，cost_lattice = 0
- **不加 TypeClassifier**（Exp3 已证伪）

---

## 2. 数据集（v2，剔除 incompat 后）

### 2.1 数据规模

```
Exp4 Final Dataset (v2):
  Total samples:    75,637
  Total mp_ids:     35,445
  Element coverage: 88

Splits:
  train      60,507  (80.00%)   28,297 mp_ids
  val         7,624  (10.08%)    3,580 mp_ids
  test        4,481  ( 5.92%)    2,139 mp_ids
  holdout     3,025  ( 4.00%)    1,429 mp_ids

Tag distribution:
  single_site         13,018  (17.21%)  原胞唯一中心位点
  equivalent          53,877  (71.23%)  多位点全对称等价
  near_equivalent      8,742  (11.56%)  小数值漂移（< 0.1 Å MAE）
  
Sealed (incompat_pool.csv, reserved for Exp5):
  52,745 samples (40.31% of original 128,382)
```

### 2.2 切分规则（Step 1 完成）

- 粒度 = mp_id（同一 mp_id 所有样本同进同出）
- 含稀有元素（全局 count < 20）的 mp_id 全进 train
- 非稀有 mp_id 按 primary_element（该 mp_id 元素中全局 count 最小的非稀有元素）做 4-way stratified split
- 比例 train:val:test:holdout = 0.80:0.10:0.06:0.04
- 4 split mp_id 零交集（已 assert）
- v2 = v1 过滤掉 incompat 样本，**比例自动保持**

### 2.3 数据路径（本地 Windows）

```python
EXP4_DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
CHI_DIR        = EXP4_DATA_ROOT + r"\MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv"
XMU_DIR        = EXP4_DATA_ROOT + r"\MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv"
POSCAR_DIR     = EXP4_DATA_ROOT + r"\POSCAR_zip\MP_all_POSCAR_flat"
FEFF_CSV       = EXP4_DATA_ROOT + r"\feff_features_all_csv_75cols(in).csv"
EXP4_ROOT      = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
EXP2_ROOT      = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
```

### 2.4 服务器路径（Step 3+ 用）

```
/home/tcat/diffcsp_exp4/
├── data/                       <- scp 上传的所有数据
│   ├── data_inventory_v2.csv   <- 主索引
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
│   ├── site_equivalence_tag.csv  (归档)
│   ├── incompat_pool.csv         (封存,不用)
│   └── MP_all_POSCAR_flat/       <- 已上传 (用户传到 /home/tcat/mp-9_POSCAR,需 mv)
├── code/                       <- Exp2 仓库 fork + Exp4 改动
├── checkpoints/                <- 训练产出
└── logs/
```

---

## 3. 已确认的文件格式

### 3.1 chi.csv

- 表头：`k,chi,chi1,chi2`，逗号分隔，401 行（1 表头 + 400 数据点）
- **模型用 `chi1` 列**（k¹χ(k) 加权信号）—— 不要用 `chi` 或 `chi2`
- Step 2 已预处理为 (200,) np.float32，k ∈ [0, 12] Å⁻¹

### 3.2 xmu.csv

- 表头：`x,y`，401 行
- `x` 列 = 能量（eV），`y` 列 = 吸收强度
- Step 2 已预处理为 (150,) np.float32，窗口 [E0-50, E0+150] eV
- E0 从 feff_features 的 E0 列读

### 3.3 POSCAR

- 文件名格式：`{mp_id}_POSCAR`（无扩展名，如 `mp-12345_POSCAR`）
- 95% 已是 primitive，少数是 conventional
- **Step 3 Dataset 保留 `get_primitive_standard_structure(symprec=0.1)`** 防御
- 标准 VASP 格式，`Structure.from_file()` 直接读

### 3.4 feff_features_all_csv_75cols(in).csv

- 76 列（3 元数据 + 73 数值特征）
- 元数据列：`sample_dir`, `sample_name`, `feature_version`
- JOIN key：完整 `sample_name`（`mp-{id}__mp-{id}-EXAFS-{Element}-K`）
- E0 在列索引 6
- **Step 1 已处理为 74 维**（73 + has_pre_edge），存 `feff_features_imputed.pkl`

---

## 4. 数据处理决策（按 Step 顺序）

### 4.1 Step 1：清洗 + 切分（已完成）

**文件名解析**：`mp-{id}__mp-{id}-EXAFS-{Element}-K`，提取 mp_id 和 center_element

**异常剔除（5,336 个）**：
- chi_invalid: 1,911（chi1 std ≤ 1e-6 或行数 < 300 或读失败）
- missing_poscar: 790（按文件系统 isfile 判断，**不用** missing_poscar_list.csv，那个列表已失效）
- H_element: 479（走到这步的；raw H 总数 2,209，其余 1,730 在前序过滤器被剔）
- iqr_outlier: 2,156（按中心元素分组算 IQR，任一列 |值-median| > 50×IQR）
- chi/xmu/poscar_invalid 等: 0
- parse_fail: 0

**缺失值处理**：
- 强度/面积/比值类列：填 0
- 能量/位置类列（`*_E`）：按中心元素分组中位数填，全局中位数 fallback
- 新增 `has_pre_edge ∈ {0, 1}` 标志位
- 最终 feff = 73 + 1 = **74 维**

**RobustScaler**：在 train（102,660 v1，对应 v2 60,507）上 fit，存 `feff_feature_scaler.pkl`

**v1 切分**：mp_id 级 stratified split（80/10/6/4），共 41,431 mp_ids → 33,147 / 4,142 / 2,485 / 1,657

### 4.2 Step 2：谱预处理（已完成）

**xmu**：截 [E0-50, E0+150] eV，`np.interp` 线性插值到 150 点（常数外推）
**chi1**：取 chi.csv 第 3 列，截 k ∈ [0, 12] Å⁻¹，插值到 200 点（去 k<0 防御）
**输出**：4 个 split 各一个 `spectra_*.pkl`

每个 pkl schema：
```python
{
    "sample_names": list[str],
    "xmu":          np.ndarray (N, 150) float32,
    "chi1":         np.ndarray (N, 200) float32,
    "name_to_idx":  dict[str, int],
    "E0":           np.ndarray (N,) float32,
    "meta":         {...}
}
```

**关键观察（Step 5 评估时注意）**：90% 样本 xmu 前段 [E0-50, E0] 是常数平台填充（FEFF 输出从 E0 附近开始）。Conv1D 第一层会自动学到"前段低信息"。不影响主体，但 Exp5 优化时可考虑窗口移到 [E0, E0+200]。

### 4.3 Step 2.5：物理约束 + 多位点处理（已完成，Main Agent 1 原 proposal 没有这一步）

#### 4.3.1 壳层划分

- **算法**：相邻距离间隙 > gap_threshold 处切壳层
- **threshold 选择**：基于训练集间隙分布，`threshold_p10 = 0.1563 Å`（间隙分布 90 分位数）
- **eval_cutoff** = 包含第 20 个邻居的那个完整壳层的外缘（替代 Exp2 的 `min(d20, 4.0)`）
- **5-10% 样本** eval_cutoff fallback 到 ~10 Å（外层弥散，主要 O/Li/P/F），Step 5 加 `eval_cutoff_fallback: bool` 标记

#### 4.3.2 多位点处理（关键决策）

**背景**：MP EXAFS 数据是 **site-averaged over symmetrically unique sites**（Mathew 2018 Scientific Data 原文 + MP wiki 确认）。意思是每条谱是该元素所有不等价 Wyckoff 位点的物理加权平均，**不对应任何具体位点**。

**Phase D 全量诊断**：对 128,382 个样本算 site_equivalence_tag：

| Tag | 计数 | 占比 | 含义 |
|---|---:|---:|---|
| single_site | 13,018 | 10.14% | 原胞唯一中心位点 |
| equivalent | 53,877 | 41.97% | 多位点全对称等价（multiset 严格相等 + MAE < 0.01 Å）|
| near_equivalent | 9,741 | 7.59% | shell1 原子数等 + MAE < 0.1 Å（999 个 multiset 不等被重分类到 incompat）|
| **incompat** | **51,746** | **40.31%** | shell1 不等价 |

**Phase F 剔除诊断**：剔除 incompat + 999 multiset-mismatch 后：
- 75,637 样本保留（58.92%）
- 88 元素**全部保留**
- split 比例自动保持
- mp_id 损失 14.45%（5,986 / 41,431）

**最终决策（Option D）**：剔除 incompat，保留样本均直接用第一位点（incompat 已剔除，剩下样本第一位点 = site-averaged 在容差内）。

**52,745 incompat 封存**到 `incompat_pool.csv`，留给 Exp5。

#### 4.3.3 Pymatgen Cython bug

Phase D v1 在 Windows 上发现 pymatgen 2024.8.9 + numpy 1.26.4 的 `find_points_in_spheres` 有 buffer dtype mismatch（int64_t vs long），100% multi-site 调用失败。Phase D v2 用纯 numpy brute-force 邻居函数绕过。Sanity check 5/5 max diff = 0.00000 Å。

**Step 3 Sub-Agent 在服务器（Linux）上必须做一次 sanity check**，确认 pymatgen 是否可用。如果不可用，从 `step2_5d_full_multisite_tag_v2.py` 抽 brute-force 函数作为 utility module。

---

## 5. Step 3 / 4 / 5 改动清单（待 Main Agent 3 写交接文档）

### 5.1 必改

#### Step 3: xas_local_dataset.py（Exp4 重写）

- 主索引改成 `data_inventory_v2.csv` + `*_samples_v2.csv`
- 中心原子定位从硬编码 Fe 改成读 `center_element` 字段
- chi/xmu 文件读取改成从 `spectra_*.pkl` 直接 load（不再读 .dat 文件）
- POSCAR 读取保留，但**第一位点选 `species == center_element` 的第一个**
- 多位点处理：**无需任何分支逻辑**（incompat 已被 Option D 剔除）
- 邻居计算：服务器先尝试 pymatgen，失败 fallback brute-force
- feff RobustScaler.transform 在 `__getitem__` 里做
- shell_boundaries.pkl 在 Dataset 里 load（Step 5 评估时用）

#### Step 3: spectrum_encoder.py（一行改动）

```python
# Exp2:
nn.Linear(73, hidden_dim)
# Exp4:
nn.Linear(74, hidden_dim)
```

### 5.2 路径常量更改

#### Step 4: 训练脚本

- 数据路径改成服务器 `/home/tcat/diffcsp_exp4/data/`
- 训练前 `cp -r .../data /tmp/diffcsp_cache/`（tmpfs 加速）
- checkpoints 只留 best + last

#### Step 5: 评估脚本

- 用 `holdout_samples_v2.csv`（3,025 样本）
- 按中心元素分层报告 RMSD、Type Acc、pred_in_cutoff
- 加 `eval_cutoff_fallback` 标记的样本单独一栏
- final report 引述 incompat_pool 的存在（已知盲区声明）

### 5.3 不变（直接复用 Exp2）

- diffusion_w_type_xas.py（扩散框架本身）
- run.py
- gnn.py / cspnet.py / diff_utils.py

---

## 6. 预期指标

| 指标 | Exp2 Holdout | Exp4 目标 |
|------|-------------|-----------|
| RMSD | 1.47 Å | ≤ 1.8 Å |
| Type Accuracy | 0.241 | 0.15-0.25 |
| pred_in_cutoff | 17.5/20 | ≥ 15/20 |

---

## 7. 已知风险 / 须 Step 3+ 注意

### 7.1 spectra-shell key 对齐

`spectra_*.pkl`、`feff_features_imputed.pkl`、`shell_boundaries.pkl`、`data_inventory_v2.csv` 均用 `sample_name` 作 key。Step 3 Dataset 写完后**第一个 forward pass 之前**必须验证：随机取 100 个 v2 sample，确认 4 个数据源都能 lookup 到。

### 7.2 v1 vs v2 不要混用

Step 3 入口用 v2（`data_inventory_v2.csv` + `*_samples_v2.csv`）。**绝对不要**用 v1 的 `*_ids.txt` 或 v1 的 `*_samples.csv`（v1 含 incompat）。

### 7.3 holdout 全程封存

Step 4 训练只用 train + val。Step 5 评估前 holdout 不能被 load 过任何一次。

### 7.4 holdout 稀有元素统计噪声

3,025 holdout 样本对 88 元素分布而言，某些稀有元素 holdout 样本可能 < 5 个。Step 5 报告分层 RMSD 时要 caveat 稀有元素的统计噪声。

### 7.5 服务器存储紧张

`/home/tcat/` 只剩 30 GB。训练前数据 cache 到 `/tmp/diffcsp_cache/`（256 GB tmpfs RAM）。checkpoints 只留 best + last。

### 7.6 Step 5 final report 必含的"已知盲区"声明

> "本工作 Exp4 训练了 75,637 样本（剔除 52,745 'incompat' 样本，详见 Step 2.5 Phase D 报告）。incompat 样本结构上含多个不等价 Wyckoff 中心位点，与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"

---

## 8. v1 Proposal 与 v2 的差异（变更日志）

| 项 | v1（Main Agent 1 原 proposal） | v2（Main Agent 2 修订后） |
|---|---|---|
| 训练样本数 | ~126,000（估计） | **75,637**（剔除 incompat 后） |
| 多位点策略 | 未明确 / 选第一位点 | **Option D：剔除 incompat 样本** |
| eval_cutoff | min(d20, 4.0)（继承 Exp2） | **基于壳层 + p10=0.1563 Å threshold** |
| 物理约束 | 未提及 | **新增 Step 2.5：壳层统计 + site_equivalence_tag** |
| missing_poscar 判断 | 用 missing_poscar_list.csv（8,003 个） | **改用文件系统 isfile**（实际 790 个，列表已失效）|
| H 元素剔除 | 估计 1,562 | 实际 raw 2,209，被分散到多个过滤器 |
| iqr_outlier 估计 | 5,729 | 实际 2,156（grouped IQR by element 比 global IQR 严）|

---

## 9. 给 Main Agent 3 的 Step 3 入口指南

### 9.1 第一步该读什么

按顺序：
1. 本 Proposal v2 §2 数据集 §3 文件格式 §5 改动清单 §7 风险
2. EXP4_PROGRESS_LOG.md（看 Step 1/2/2.5 完整 action+result）
3. EXP4_FILE_INVENTORY.md（确认所有产出文件位置）

### 9.2 Step 3 Sub-Agent Dataset 类骨架（参考实现意图）

```python
class XasLocalDatasetV2(Dataset):
    def __init__(self, split, data_dir):
        # Load v2 sample list (already filtered, no incompat)
        self.samples = pd.read_csv(f"{data_dir}/{split}_samples_v2.csv")
        # Load shell boundaries (full 128K, but we only access keys in samples)
        self.shells = pickle.load(open(f"{data_dir}/shell_boundaries.pkl", "rb"))
        # Load FEFF features
        self.feff = pd.read_pickle(f"{data_dir}/feff_features_imputed.pkl")
        self.scaler = joblib.load(f"{data_dir}/feff_feature_scaler.pkl")
        # Load spectra
        self.spectra = pickle.load(open(f"{data_dir}/spectra_{split}.pkl", "rb"))

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        sname = row["sample_name"]
        center_elem = row["center_element"]
        
        # Spectra (already preprocessed by Step 2)
        spec_idx = self.spectra["name_to_idx"][sname]
        xmu = self.spectra["xmu"][spec_idx]   # (150,)
        chi1 = self.spectra["chi1"][spec_idx] # (200,)
        
        # FEFF (apply RobustScaler now)
        feff_raw = self.feff.loc[sname].values.reshape(1, -1)  # (1, 74)
        feff = self.scaler.transform(feff_raw).squeeze()       # (74,)
        
        # POSCAR -> primitive -> first center site -> 20 neighbors
        # (multi-site dispatching NOT needed -- incompat already filtered)
        poscar_path = ... # from data_inventory_v2 lookup
        prim = SpacegroupAnalyzer(Structure.from_file(poscar_path), 
                                   symprec=0.1).get_primitive_standard_structure()
        center_idx = next(i for i, s in enumerate(prim) 
                          if s.specie.symbol == center_elem)
        neighbors = prim.get_neighbors(prim[center_idx], r=10.0)
        sorted_nbrs = sorted(neighbors, key=lambda n: n.nn_distance)[:20]
        
        # Cartesian -> fractional in virtual lattice (L=6)
        L = 6.0
        center_cart = prim[center_idx].coords
        frac_coords = np.array([(n.coords - center_cart) / L for n in sorted_nbrs])
        frac_coords -= np.round(frac_coords)  # min-image
        
        atom_types = np.array([n.specie.Z for n in sorted_nbrs])
        
        return {
            "xmu": xmu, "chi1": chi1, "feff": feff,
            "frac_coords": frac_coords, "atom_types": atom_types,
            "sample_name": sname, "center_element": center_elem,
        }
```

**注意**：上面只是意图示意，**不是给 Sub-Agent 抄的代码**。Main Agent 3 写交接文档时按"改动意图"描述，Sub-Agent 实现细节。

---

*Main Agent 2 撰写，2026-04-25*
