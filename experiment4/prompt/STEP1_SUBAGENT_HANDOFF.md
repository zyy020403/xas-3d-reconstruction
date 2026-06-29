# STEP 1 SUB-AGENT 交接文档
# Experiment 4 — 数据清单构建、异常剔除、分层切分、RobustScaler 拟合

> **发送对象**：DiffCSP-Exp4-Step1-SubAgent（新会话窗口）
> **撰写者**：DiffCSP-Exp4-Main-Agent 2
> **日期**：2026-04-23
> **执行环境**：本地 Windows（Python 3.9，无需 SSH）
> **前置依赖**：无（你是整条 pipeline 的第一步）

---

## 1. 你的角色

你是 Experiment 4 的 **Step 1 Sub-Agent**。Experiment 4 要把 Exp2 验证过的"从 XAS 谱预测局部原子结构"架构，扩展到 Materials Project 全元素 EXAFS 数据集（~132K 原始样本，89 种中心元素）。

**你的唯一任务**：扫描原始数据、清洗、切分、拟合 RobustScaler，产出一套干净的"数据清单 + 切分文件 + scaler"给下游 Step 2/3/4/5 使用。

**你不做的事**：不动模型、不动 Dataset/encoder/扩散框架代码（那是 Step 3 的事）、不做谱预处理（Step 2 的事）。

**你不改决策**：本文档里标 🔒 的是 Main Agent 已敲定的决策，**不要自由发挥**。如果执行中发现决策有问题，**停下来写汇报给用户**，让 Main Agent 决定是否调整。

**你在本地 Windows 跑**，不需要 SSH 服务器，不需要写服务器命令。

---

## 2. 动手前：必须先做的三件事（不能跳过）

Main Agent 1 已收齐过这些信息，但让你**自己再确认一次作为动手前的最后检查**。

### 事一：listdir 打印前 10 个文件名

分别对以下目录调用 `os.listdir()[:10]`，把结果打印给用户：

| 目录 | 期望看到什么 |
|------|------------|
| `CHI_DIR` | 每个文件形如 `mp-{id}__mp-{id}-EXAFS-{Element}-K.csv` 的 chi 文件 |
| `XMU_DIR` | 每个文件形如 `mp-{id}__mp-{id}-EXAFS-{Element}-K.csv` 的 xmu 文件 |
| `POSCAR_DIR` | POSCAR 文件（命名可能是 `mp-{id}` 或 `mp-{id}.poscar` 或 `mp-{id}.vasp`，你要自己验证） |

**目的**：确认文件命名格式。Main Agent 1 没明确 chi/xmu/POSCAR 的完整文件名模式，你要自己探明。

### 事二：读文件头部样本

打开并打印：

1. 任意一个 chi.csv 的前 5 行 —— **期望**：表头 `k,chi,chi1,chi2`，逗号分隔，共 401 行（1 表头 + 400 数据点）
2. 任意一个 xmu.csv 的前 5 行 —— **期望**：表头 `x,y`，逗号分隔，共 401 行
3. 任意一个 POSCAR 的前 10 行 —— **期望**：标准 VASP 格式，`pymatgen.core.Structure.from_file()` 可直接读
4. `FEFF_CSV` 的前 2 行 —— **期望**：76 列，元数据列 `sample_dir / sample_name / feature_version`，`E0` 在列索引 6，sample_name 形如 `mp-10003__mp-10003-EXAFS-Co-K`
5. `MISSING_POSCAR_CSV` 的前 5 行 —— **期望**：一列 mp_id，约 8003 行

### 事三：向用户复述 4 条关键规则等确认

把下面 4 条规则用你自己的话复述给用户，等用户 reply "confirmed" 或 "go" 后再开始编码：

1. **文件名解析规则**：sample_name = `mp-{id}__mp-{id}-EXAFS-{Element}-K`，从这里提取 `mp_id`（`mp-{id}`，带连字符）和 `center_element`（1-2 字母元素符号）
2. **异常剔除三条并集**：(a) 所有 H 中心元素（~1,562 个）(b) missing_poscar_list 里的 mp_id（8003 个对应 ~791 个样本）(c) 按中心元素分组算 median/IQR，任一列 `|值 - median| > 50 × IQR` 即剔（预计 ~5,729 个）
3. **缺失值处理两类**：(a) 强度/面积/比值类列（见第 6 节清单）填 0 (b) 能量/位置类列（`*_E` 结尾的那 7 个）按中心元素分组取中位数填；再新增 `has_pre_edge ∈ {0,1}` 标记原始 `pre_peak_I` 是否为 NaN → 最终 feff 维度 **74**
4. **切分规则**：粒度 = mp_id（同一 mp_id 所有样本同进同出）；含任一稀有元素（样本级 count < 20）的 mp_id 全部进 train；其余 mp_id 按 "primary_element"（该 mp_id 所含元素中全局 count 最小的那个非稀有元素）做 4-way stratified split，比例 **train:val:test:holdout = 0.80:0.10:0.06:0.04**

⚠️ 如果用户对任一条提出修正，按用户新指令执行，并在最终汇报里记录修正项。

---

## 3. 用户会发给你的文件

1. **STEP1_SUBAGENT_HANDOFF.md**（本文档）—— 最权威
2. **EXP4_MAINAGENT_HANDOFF.md** —— Main Agent 1 的 Exp4 方案背景（参考）
3. **SHARED_00_v2.md、SHARED_01_DATA_MANIFEST.md** —— **这俩是 Exp2 的**共享文档，仅参考"L=6/20 邻居/cost_lattice=0 等不变量设计理念"。⚠️ **文件格式、路径、命名规则全部以本文档为准，Exp4 全变了**：
   - SHARED_01 说 chi1.dat 是 FEFF 原始 .dat 空格分隔 → Exp4 是 `chi.csv` 逗号分隔 4 列
   - SHARED_01 说文件夹结构 `mp_{id}_{formula}_feff_Fe_site_{nn}` → Exp4 不是文件夹，是扁平 CSV 文件
   - SHARED_01 说中心元素固定 Fe → Exp4 是任意元素从文件名读
4. **EXPERIMENT2_FINAL_REPORT.md** —— Exp2 历史结果，仅了解背景

---

## 4. 路径常量（脚本顶部直接粘贴）

```python
import os

# ========== Exp4 数据路径 ==========
EXP4_DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
CHI_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv")
XMU_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv")
POSCAR_DIR     = os.path.join(EXP4_DATA_ROOT, r"POSCAR_zip\MP_all_POSCAR_flat")
FEFF_CSV       = os.path.join(EXP4_DATA_ROOT, "feff_features_all_csv_75cols(in).csv")
MISSING_POSCAR_CSV = os.path.join(POSCAR_DIR, "missing_poscar_list.csv")

# ========== 输出路径 ==========
EXP4_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR   = os.path.join(EXP4_ROOT, "step1")
os.makedirs(STEP1_DIR, exist_ok=True)
```

脚本和所有产出文件都放在 `STEP1_DIR`。

推荐拆成几个文件（也可合一，只要每步都有清晰打印）：
- `step1_1_scan_and_parse.py` — 扫描目录 + 解析文件名 + 三路 JOIN
- `step1_2_filter_outliers.py` — 剔除 H + missing POSCAR + IQR×50 异常
- `step1_3_feff_imputation.py` — 缺失值填充 + 新增 has_pre_edge（74 维）
- `step1_4_split_stratified.py` — mp_id 级 4-way stratified split
- `step1_5_fit_scaler.py` — train 上 fit RobustScaler + 存 stats
- `step1_6_summary.py` — 按元素统计分布、自查、输出报告

---

## 5. 核心决策（🔒 LOCKED，不得自由发挥）

### 5.1 文件名解析规则 🔒

```
sample_name 格式：mp-{digits}__mp-{digits}-EXAFS-{Element}-K
                   ^^^^^^^^^^^^^^^^^^^^^^^^
                   前后两个 mp-id 应完全相同（冗余命名）

Element 规则：首字母大写，第 2 字母（若有）小写，1-2 个字母
             例：H, He, Li, Be, O, Fe, Co, Cu, Zn, U, Np, Pu

建议 regex：
    pattern = r"^(mp-\d+)__\1-EXAFS-([A-Z][a-z]?)-K"
    m = re.match(pattern, sample_name)
    if m is None: skip/log
    mp_id, center_element = m.group(1), m.group(2)
```

**要求**：解析失败的样本打印 sample_name 并记录到日志，但不报错退出（后续作为 `parse_fail` 统计）。

### 5.2 JOIN 策略 🔒

三方 JOIN key = **完整的 `sample_name` 字符串**（不是 mp_id）：

- feff_features CSV 里的 `sample_name` 列
- CHI_DIR 下的文件名去扩展名
- XMU_DIR 下的文件名去扩展名
- POSCAR_DIR 下按 **mp_id**（不是 sample_name）匹配（一个 mp_id 对应一个 POSCAR，被该 mp_id 的所有中心元素样本共享）

保留条件：**sample_name 同时存在于 (chi, xmu, feff)** 且 mp_id 的 POSCAR 存在且不在 missing_poscar_list 中。

### 5.3 chi.csv / xmu.csv 的有效性判定 🔒

**Step 1 不做谱预处理**（那是 Step 2），但要做"有效性"筛查：

- `chi_valid`：文件存在 + 用 `pd.read_csv` 读成功 + 行数 ≥ 300 + `chi1` 列 std > 1e-6
- `xmu_valid`：文件存在 + 用 `pd.read_csv` 读成功 + 行数 ≥ 300 + `y` 列 std > 1e-6
- `poscar_valid`：尝试 `Structure.from_file(poscar_path)` 成功 + 能跑通 `SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()` + 原胞原子数 ≥ 1

任一无效 → 样本被剔除。统计各项失败数报告给用户。

### 5.4 异常剔除三条 🔒

**必须在"缺失值填充之前"做 IQR 判定**（否则填 0 / 填中位数会污染 IQR）。

#### (a) 剔 H 元素

```python
df = df[df["center_element"] != "H"]
```

#### (b) 剔 missing POSCAR

```python
missing = set(pd.read_csv(MISSING_POSCAR_CSV).iloc[:, 0].astype(str))  # 自己确认列结构
df = df[~df["mp_id"].isin(missing)]
```

#### (c) IQR × 50 极端异常（按中心元素分组）🔒

```
对每个 center_element 分组：
    对每一列（73 个 feff 数值特征，不含 sample_dir/sample_name/feature_version/has_pre_edge）：
        group_median = group[col].median()    # skipna=True（默认）
        group_q1     = group[col].quantile(0.25)
        group_q3     = group[col].quantile(0.75)
        group_iqr    = group_q3 - group_q1
        if group_iqr == 0 或 NaN 或 该 group 样本数 < 5:
            skip this column for this group   # 避免小样本伪异常
        threshold    = 50 * group_iqr
        outlier_mask = (group[col] - group_median).abs() > threshold
        mark those rows as outliers
任一列被标异常 → 整行标 `is_iqr_outlier=True`
```

剔除 `is_iqr_outlier=True` 的所有样本。

**预期剔除数**：~5,729（Main Agent 1 估算，仅参考）。如果你得到 < 3,000 或 > 10,000，**不要自作主张调阈值**，在汇报里说明并等 Main Agent 决策。

### 5.5 缺失值填充（分两类） 🔒

在 IQR 剔除**之后**做。

#### 填 0 类（14 + 若干列，共约 30-40 列）

```
pre_peak_I, white_line_I, post_peak1_I, d1_pre_I, d1_post_I,
area_pre, area_edge, area_white_line, area_post1,
pre_white_ratio, post_white_ratio,
所有 k2chi_* 列, 所有 k3chi_* 列, 所有 R1_* 列, 所有 R2_* 列
```

用 `df[col].fillna(0.0)`。

#### 分中心元素中位数类（7 列）

```
pre_peak_E, white_line_E, post_peak1_E, d1_pre_E, d1_post_E,
pre_centroid_E, white_centroid_E
```

```python
for col in energy_position_columns:
    df[col] = df.groupby("center_element")[col].transform(
        lambda x: x.fillna(x.median())
    )
```

⚠️ 如果某元素组在某列仍全 NaN（即中位数也是 NaN），回退到**全局**中位数填。

#### 新增标志位列

```python
df["has_pre_edge"] = df["pre_peak_I"].notna().astype(int)  # 注意：在填 0 之前判定！
```

**顺序要严格**：先存下 `has_pre_edge` 这一列（基于原始 NaN），**再**做 `pre_peak_I.fillna(0)`。

**最终 feff 维度**：73（原数值列全部填好）+ 1（has_pre_edge）= **74**。

### 5.6 按 mp_id 分层切分 🔒

#### 定义 primary_element

```
对每个 mp_id：
    elem_set = set of center_elements in this mp_id
    rare_elems = {E | 全局样本级 count(E) < 20}
    if elem_set ∩ rare_elems ≠ ∅:
        is_rare_mpid = True   # 整个 mp_id 进 train
    else:
        # 非稀有 mp_id 的 primary：取该 mp_id 所有元素中全局 count 最小的
        primary_element = argmin(count(E) for E in elem_set)
        is_rare_mpid = False
```

#### 切分比例 🔒

- 含稀有元素的 mp_id：**100% 进 train**
- 非稀有 mp_id：按 `primary_element` **stratified** 切分为
  `train:val:test:holdout = 0.80:0.10:0.06:0.04`

#### 推荐实现方式

```python
from sklearn.model_selection import train_test_split

# 非稀有 mp_id 表
df_nonrare = pd.DataFrame({"mp_id": ..., "primary_element": ...})

# 先拆 holdout（4%）
rest, holdout = train_test_split(df_nonrare, test_size=0.04,
                                  stratify=df_nonrare["primary_element"],
                                  random_state=42)
# 再拆 test（6% of total = 0.0625 of rest）
rest2, test = train_test_split(rest, test_size=0.0625,
                                stratify=rest["primary_element"],
                                random_state=42)
# 再拆 val（10% of total = 0.1111 of rest2）
train_nonrare, val = train_test_split(rest2, test_size=0.1111,
                                       stratify=rest2["primary_element"],
                                       random_state=42)

# 合并稀有 mp_id 到 train
train = pd.concat([train_nonrare, df_rare])
```

⚠️ 某些元素组 primary_element 样本太少，stratify 会报错。此时把这些 primary 的 mp_id 直接扔进 train 并在汇报里记录。

**零交集检查**（硬性要求）：
```python
assert set(train_ids) & set(val_ids) == set()
assert set(train_ids) & set(test_ids) == set()
assert set(train_ids) & set(holdout_ids) == set()
assert set(val_ids) & set(test_ids) == set()
assert set(val_ids) & set(holdout_ids) == set()
assert set(test_ids) & set(holdout_ids) == set()
```

### 5.7 RobustScaler 🔒

- 在 **train 集 74 维 feff 特征**上 `fit`
- 全局 fit（不分元素）
- 只 fit，不 transform（transform 由 Step 2/3 的 DataLoader 做）
- 存 pkl：`joblib.dump(scaler, os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"))`

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(train_feff_74dim_array)  # shape (N_train_samples, 74)
```

---

## 6. 输出文件清单（严格按此，不要改名）

全部放在 `STEP1_DIR` 下：

| 文件名 | 内容 |
|--------|------|
| `data_inventory.csv` | 所有最终保留样本（~126K 行），列见 6.1 |
| `train_ids.txt` | train 的 mp_id 列表，每行一个 |
| `val_ids.txt` | val 的 mp_id 列表 |
| `test_ids.txt` | test 的 mp_id 列表 |
| `holdout_ids.txt` | holdout 的 mp_id 列表 |
| `train_samples.csv` | 样本级 train 清单，列 `[mp_id, center_element, sample_name]` |
| `val_samples.csv` | 同上 |
| `test_samples.csv` | 同上 |
| `holdout_samples.csv` | 同上 |
| `feff_feature_scaler.pkl` | `joblib.dump` 出的 RobustScaler 对象 |
| `feff_feature_stats.csv` | 74 列的统计表，列：`feature_name, median, iqr, q1, q3, min, max, n_nan_before_impute` |
| `element_distribution.csv` | 按中心元素统计，列：`element, n_samples_total, n_train, n_val, n_test, n_holdout, is_rare` |
| `step1_excluded_log.csv` | 被剔除样本明细，列：`sample_name, mp_id, center_element, reason`（reason ∈ {parse_fail, H_element, missing_poscar, chi_invalid, xmu_invalid, poscar_invalid, iqr_outlier}） |
| `step1_summary.txt` | 人类可读的汇总报告（见第 8 节模板） |

### 6.1 `data_inventory.csv` 字段

| 列名 | 类型 | 含义 |
|------|------|------|
| `sample_name` | str | 完整 JOIN key（如 `mp-10003__mp-10003-EXAFS-Co-K`） |
| `mp_id` | str | `mp-10003`（带连字符） |
| `center_element` | str | 中心元素符号（如 `Co`） |
| `chi_path` | str | 绝对路径 |
| `xmu_path` | str | 绝对路径 |
| `poscar_path` | str | 绝对路径 |
| `prim_n_atoms` | int | 原胞转换后原子数 |
| `has_pre_edge` | int | {0, 1} |
| `chi_valid` | bool | 一般为 True（无效的已被剔除）；保留此列用于追溯 |
| `xmu_valid` | bool | 同上 |
| `poscar_valid` | bool | 同上 |
| `is_iqr_outlier` | bool | False（异常的已剔除）；保留用于追溯 |
| `split` | str | `train` / `val` / `test` / `holdout` |
| `quality_tier` | str | `A`（prim_n_atoms ≤ 30）/ `B`（31-80）/ `C`（> 80）；供 Step 5 分层评估备用 |

**注**：你产出的 `data_inventory.csv` **只包含最终保留样本**，被剔除的样本在 `step1_excluded_log.csv` 里独立记录。这样下游 Step 2/3 直接 `pd.read_csv(data_inventory.csv)` 就是干净的样本集。

---

## 7. 自查清单（你汇报之前必须跑一遍）

打印以下检查的输出到 console 和 `step1_summary.txt`：

1. **总样本数**：最终 `data_inventory.csv` 行数应在 **120,000 - 130,000** 区间（Main Agent 1 估算 ~126,000）。偏离较大汇报说明。
2. **各 split 样本数**：
   - train 样本数
   - val 样本数
   - test 样本数
   - **holdout 样本数应在 3,000 - 5,000 区间**（偏离汇报说明）
3. **mp_id 零交集**（硬性，assert 通过）
4. **feff 维度 = 74**（打印 shape 验证）
5. **has_pre_edge 只取 {0, 1}**（打印 `value_counts()`）
6. **scaler 只在 train 上 fit**：比较 `scaler.center_` 与 `train_feff.median()`，应基本相等（全局 median）
7. **holdout 元素分布健康**：
   - 打印 holdout 里每个 center_element 的样本占比
   - O 和 Li 占比不应 > 25%（若 > 30% 汇报说明，可能需重新 stratify）
   - holdout 里元素种类数 ≥ 30（证明不是被头部元素垄断）
8. **稀有元素全在 train**：对所有 `center_element` count < 20 的元素，验证它们在 val/test/holdout 的样本数都为 0
9. **RobustScaler pkl 可重新 load**：`joblib.load` 回来，能 `transform` 一个小样本（sanity check）
10. **剔除总数拆解报表**：

| 剔除原因 | 样本数 | Main Agent 1 预估 |
|---------|-------|-----------------|
| parse_fail | ? | ~0 |
| H_element | ? | ~1,562 |
| missing_poscar | ? | ~791 |
| chi_invalid | ? | 少量 |
| xmu_invalid | ? | 少量 |
| poscar_invalid | ? | 少量 |
| iqr_outlier | ? | ~5,729 |
| **合计剔除** | ? | ~8,100 |

数字显著偏离预估（> 30%）要在汇报里说明。

---

## 8. 汇报模板

跑完后按以下格式向用户汇报（用户会把这个报告回传给 Main Agent）：

```markdown
## Step 1 完成报告

### 8.1 执行总览
- 扫描原始样本数：?
- 最终保留样本数：?
- 最终保留 mp_id 数：?

### 8.2 剔除明细
| 原因 | 数量 |
|------|-----|
| parse_fail | ? |
| H_element | ? |
| missing_poscar | ? |
| chi_invalid | ? |
| xmu_invalid | ? |
| poscar_invalid | ? |
| iqr_outlier | ? |
| 合计 | ? |

### 8.3 切分结果
| split | mp_id 数 | 样本数 | 占比 |
|-------|---------|-------|------|
| train | ? | ? | ? |
| val | ? | ? | ? |
| test | ? | ? | ? |
| holdout | ? | ? | ? |

零交集检查：[ PASS / FAIL ]

### 8.4 元素分布（前 10 高 + 前 5 稀有）
[粘贴 element_distribution.csv 的关键行]

稀有元素列表（count < 20）：[元素 1, 元素 2, ...]
稀有元素是否全部只在 train：[YES / NO]

### 8.5 Holdout 元素健康度
- Holdout 里元素种类数：?
- O 占比：?%
- Li 占比：?%
- 前 5 多元素及占比：?

### 8.6 feff 特征
- 最终维度：? (期望 74)
- has_pre_edge=1 占比：?%
- RobustScaler center_[0:5]：[?, ?, ?, ?, ?]
- RobustScaler scale_[0:5]：[?, ?, ?, ?, ?]

### 8.7 产出文件列表
[列出 STEP1_DIR 下所有产出文件及 size]

### 8.8 异常与发现
[执行中遇到的任何偏离预期的情况，包括 IQR 异常数和预估差异、某些 primary 元素因样本太少被塞进 train、某些列全 NaN 回退到全局中位数等等]

### 8.9 需要 Main Agent 决策的问题
[如果没有就写"无"]
```

---

## 9. 不要做的事

1. ❌ 不要修改 Exp2 的任何脚本（我们这一步纯数据处理，不碰 Exp2 代码）
2. ❌ 不要做谱预处理（插值、截取窗口）—— 那是 Step 2 的事
3. ❌ 不要用 StandardScaler —— 必须 RobustScaler
4. ❌ 不要把化学式 / mp_id 作为数值特征 —— 仅作数据管理键
5. ❌ 不要擅自调整 IQR × 50 阈值、切分比例、稀有元素阈值 20 —— 这些都 🔒 LOCKED，偏离预估在汇报里说明
6. ❌ 不要用 `num_workers > 0` —— Windows 不稳定
7. ❌ 不要假设文件编码，pd.read_csv 显式指定 `encoding="utf-8"`（遇到中文路径报错时尝试 `encoding="utf-8-sig"`）
8. ❌ 不要在本地生成"预处理后的 tensor 数据" —— 这是 Step 2 的事

---

## 10. 依赖

```
pandas
numpy
scikit-learn  (for RobustScaler, train_test_split)
pymatgen      (for Structure.from_file, SpacegroupAnalyzer)
joblib        (for pickle RobustScaler)
tqdm          (optional but recommended for 40K+ POSCAR 解析进度条)
```

Windows 建议在一个干净的 conda env 里跑：

```powershell
conda create -n exp4_step1 python=3.9 -y
conda activate exp4_step1
pip install pandas numpy scikit-learn pymatgen joblib tqdm
```

---

*DiffCSP-Exp4-Main-Agent 2 撰写，2026-04-23*
