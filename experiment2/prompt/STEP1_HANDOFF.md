# STEP1_HANDOFF.md
# Step1 Agent 交接文档：数据清洗与清单构建

> **你的角色**：Step1 Agent
> **你的任务**：扫描全部数据，做质量过滤，选定代表谱，生成所有后续步骤需要的清单文件
> **前置文档**：先读 SHARED_00_PROJECT_OVERVIEW.md 和 SHARED_01_DATA_MANIFEST.md
> **输出目录**：`C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1\`

---

## 你需要的文件（向用户索取）

在开始写任何代码之前，告诉用户你需要以下文件（或确认可访问）：

```
1. 数据目录（可访问即可，不需要发送）：
   C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site\

2. feff_features 表（需要发给你读取）：
   C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv

3. 参考脚本（可访问即可）：
   C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
   （Exp1 的旧清单，仅参考字段设计，路径和数据均不可直接用）
```

---

## 工作内容（按顺序执行）

### Step 1.1：扫描文件夹，三文件完整性检查

**脚本**：`step1.1_scan_folders.py`

**任务**：
- 遍历 `DATA_ROOT` 下的所有 18,385 个文件夹
- 对每个文件夹检查：chi1.dat、xmu.dat、POSCAR_supercell_fixed 是否存在
- 解析文件夹名，提取 (mp_id, formula, site_nn)
- 输出初步清单 `step1_raw_scan.csv`

**输出字段**：
```
folder_name, folder_path, mp_id, formula, site_nn,
has_chi1, has_xmu, has_poscar
```

**注意**：
- 文件夹名解析见 SHARED_01 第2节
- formula 字段记录但后续脚本不传入模型

---

### Step 1.2：chi1 + xmu 质量过滤

**脚本**：`step1.2_spectral_quality_filter.py`

**任务**：
- 对 step1.1 中三文件完整的样本，做谱质量检查
- **chi1 过滤**：
  - 读取 chi1.dat，计算第2列（chi 值）的 std
  - 记录所有样本的 chi_std，统计分布
  - 取 5th percentile 作为有效下限阈值
  - chi_std < 阈值 → 标注 chi1_valid=False
- **xmu 过滤**：
  - 读取 xmu.dat 第2列
  - 检查：有无 NaN/Inf，行数 ≥ 50，值域是否在合理范围（mu > -1 且 mu < 10）
  - 不满足 → xmu_valid=False

**输出**：
- `step1_quality_filter.csv`（在 raw_scan 基础上添加 chi1_valid、xmu_valid、chi_std）
- `chi_std_distribution.png`（分布图，标注 5th percentile 线）
- 控制台打印：`过滤前 N，chi1 无效 X，xmu 无效 Y，过滤后 Z`

---

### Step 1.3：POSCAR 解析 + 原胞转换预检

**脚本**：`step1.3_poscar_check.py`

**任务**：
- 对 Step1.2 后有效的样本，逐一尝试 pymatgen 解析 POSCAR
- 对**抽样 300 个**（随机，涵盖不同化学体系）做原胞转换，统计转换后原子数分布
- 不做全量原胞转换（太慢），只统计样本，验证 symprec=0.1 是否合适

**原胞转换代码**（直接复用 Exp1 的 xas_dataset.py）：
```python
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def try_get_primitive(poscar_path, symprec=0.1):
    try:
        structure = Structure.from_file(poscar_path)
        analyzer  = SpacegroupAnalyzer(structure, symprec=symprec)
        primitive = analyzer.get_primitive_standard_structure()
        return primitive, primitive.num_sites, None
    except Exception as e:
        return None, -1, str(e)
```

**输出**：
- `step1_poscar_check.csv`（在前一步基础上添加 poscar_valid, prim_n_atoms_sample）
- 控制台打印抽样统计：mean/median/95th percentile 原子数，是否 < 20

**若 95th percentile > 40**：在报告中标注，建议 Main Agent 考虑是否调整 symprec

---

### Step 1.4：LVSI 位点选取

**脚本**：`step1.4_lvsi_site_selection.py`

**任务**：
- 对所有通过质量过滤（chi1_valid=True, xmu_valid=True, poscar_valid=True）的文件夹
- 按 mp_id 分组，在每组内按 site_nn 升序选第一个（即 Lowest Valid Site Index）
- 记录每个 mp_id 的总 site 数和有效 site 数

**输出**：
- `selected_site_map.csv`：
  ```
  mp_id, selected_folder_name, selected_site_nn, total_sites, valid_sites
  ```
- 控制台打印：
  - 最终有效化合物总数
  - 单位点 vs 多位点化合物数量分布

---

### Step 1.5：feff_features 匹配

**脚本**：`step1.5_match_feff_features.py`

**任务**：
- 读取 `feff_features_all_site_v2.csv`
- 从 sample_name 列提取 (mp_id, site_nn) 复合键
- 与 selected_site_map.csv 做 JOIN
- 标注每个化合物是否在 feff_features 表中有匹配行

**匹配键提取逻辑**：
```python
def parse_sample_name(sample_name):
    # 示例：mp_1047285_FeO2__feff_site_01
    parts = sample_name.split("_")
    mp_id   = parts[0] + "_" + parts[1]   # "mp_1047285"
    site_nn = parts[-1]                    # "01"
    return mp_id, site_nn
```

**输出**：
- `data_inventory.csv`（最终合并清单，字段见 SHARED_01 第8节）
- 控制台打印：匹配成功 N，缺失 M（缺失样本暂时排除）

**NaN 处理**：
- 若某行 feff_features 中有 NaN，记录 has_nan_features=True
- 暂时保留，Step3 Dataset 中用训练集均值填充

---

### Step 1.6：Holdout 划分 + train/val/test 分割

**脚本**：`step1.6_split_and_holdout.py`

**任务**：

**A. Holdout 1000 个**：
```python
# 1. 用 feff_features 的 4 列做特征：E0, white_line_I, R1_peak_pos, chi_kmax
#    对应列名（查表确认列索引）：列6=E0, 列12=white_line_I, 列69=R1_peak_pos, 列68=chi_kmax
# 2. 对有效化合物的上述特征做 StandardScaler 标准化
# 3. KMeans(n_clusters=100, random_state=42)
# 4. 每个簇内按 10% 比例抽取（size >= 5 才抽，至少 1 个，至多 20 个）
# 5. 控制总数在 1000 左右（不足补抽大簇，超出去掉小簇末尾）
# 6. 确保 holdout 中每个被抽的化合物，其所在簇仍有 ≥ 2 个样本留在训练集
```

**B. 剩余样本 train/val/test**：
```python
# 剩余样本（排除 holdout）按 mp_id 做 70:15:15 分层随机划分
# random_state=42，可复现
```

**输出**：
- `holdout_1000_ids.txt`（每行一个 mp_id）
- `train_ids.txt`
- `val_ids.txt`
- `test_ids.txt`
- 控制台打印：各集合大小

---

### Step 1.7：feff_features 标准化器

**脚本**：`step1.7_fit_feature_scaler.py`

**任务**：
- 读取 train_ids.txt 对应的样本的 73 列 feff_features
- 拟合 `sklearn.preprocessing.StandardScaler`
- 保存为 `feff_feature_scaler.pkl`（后续 Dataset 加载时使用）
- 统计训练集 feff_features 各列的均值方差，用于填充测试集 NaN

**输出**：
- `feff_feature_scaler.pkl`
- `feff_feature_stats.csv`（均值、方差、NaN 比例，供 Step3 参考）

---

## 最终输出文件清单

```
experiment2/step1/
├── step1_raw_scan.csv            Step1.1 输出
├── chi_std_distribution.png      Step1.2 输出
├── step1_quality_filter.csv      Step1.2 输出
├── step1_poscar_check.csv        Step1.3 输出
├── selected_site_map.csv         Step1.4 输出
├── data_inventory.csv            Step1.5 最终合并清单 ★
├── holdout_1000_ids.txt          Step1.6 输出 ★
├── train_ids.txt                 Step1.6 输出 ★
├── val_ids.txt                   Step1.6 输出 ★
├── test_ids.txt                  Step1.6 输出 ★
├── feff_feature_scaler.pkl       Step1.7 输出 ★
└── feff_feature_stats.csv        Step1.7 输出
```

标 ★ 的文件是后续 Step 必须读取的关键文件。

---

## 注意事项与常见陷阱

1. **Windows 路径**：所有路径用 `r"..."` 原始字符串或 `pathlib.Path`，不要用正斜杠

2. **编码问题**：读 CSV 时加 `encoding='utf-8-sig'`（Windows Excel 保存的 CSV 可能有 BOM）

3. **pymatgen 报错**：部分 POSCAR 格式异常，必须用 try/except 捕获，不要让整个脚本崩溃

4. **内存**：18,385 个文件夹全扫一遍，chi1/xmu 不需要全部读入内存，逐个处理即可

5. **进度显示**：每个脚本加 tqdm 进度条，方便用户监控

6. **单脚本独立可运行**：每个 step1.X 脚本独立运行，读上一步的 CSV 作为输入，输出新 CSV，不做跨脚本状态依赖

7. **Step1.3 原胞转换只抽样**：全量 POSCAR 解析（poscar_valid 检查）不做原胞转换，转换只在抽样 300 个中做，实际原胞转换在 Step3 的 Dataset 中完成

---

## 完成后向 Main Agent 汇报

使用 SHARED_00 中的报告模板，重点汇报：
- 最终有效化合物数量（过滤后）
- chi1 std 自动确定的阈值
- 原胞转换抽样统计：mean 原子数，95th percentile 是否 < 40
- feff_features 匹配成功率
- holdout/train/val/test 各集合大小
- 任何异常（如某化合物既没有 site_01 也没有 site_02）
