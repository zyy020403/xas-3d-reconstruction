# STEP1_HANDOFF.md
# Step 1 Agent 交接文档：数据预处理与清洗

> **任务编号**: Step 1（共 4 个子步骤：1.1 → 1.2 → 1.3 → 1.4）  
> **前置条件**: 读完所有 SHARED_00 ~ SHARED_04 文档  
> **输出目录**: `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\`  
> **完成标志**: 生成 `data_inventory.csv`、四个 ID 列表文件、键长约束字典文件

---

## ⚡ 开始工作前——必须向用户索取的文件

**在写任何脚本之前**，请向用户说明：你需要参考以下 DiffCSP 原始文件来了解数据格式要求，请用户将这些文件内容发给你：

```
请提供以下文件内容（用于理解 DiffCSP 数据格式，不修改这些文件）：

1. diffcsp/pl_data/dataset.py        ← 了解 DiffCSP 如何加载 crystal 数据
2. diffcsp/common/data_utils.py      ← 了解 POSCAR 解析和 crystal 工具函数
```

**理由**：Step 1 的输出格式（特别是 POSCAR 解析结果的存储方式）必须与 DiffCSP 原有数据结构兼容，否则后续 Step 3 改造时要重新做格式转换，浪费时间。

**你只需要阅读，不修改这两个文件。**

---

## 任务概览

Step 1 的目标是：将两个原始数据文件夹整理成一份干净的数据清单，完成去重、格式验证、质量分级、数据集划分，为后续 Step 2 的谱编码做好准备。

Step 1 不涉及模型训练，全部是数据工程脚本。

---

## Step 1.1：扫描文件夹 + 去重 + 建立数据清单

### 脚本名
```
step1.1_scan_and_inventory.py
```

### 输入
```
C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset\               （主数据集）
C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A\ （离子数据集）
```

### 任务描述

**1. 扫描两个文件夹，列出所有子文件夹**

**2. 解析每个文件夹名，提取以下字段：**

文件夹名格式：`mp_{mp_id}_{formula}__feff_{element}_site_{nn}` 或 `mp_{mp_id}_{formula}__feff_site_{nn}`

提取规则（用正则或字符串处理）：
- `mp_id`：第一个 `_` 和第二个 `_` 之间的数字
- `formula`：**不提取，不使用，不存入任何输出文件**（防止数据泄露）
- `element`：`__feff_` 后、`_site_` 前的字符串；若 `__feff_` 后直接跟 `site_`（无元素名），则：
  - 扫描文件夹名中介于第二个 `_` 和 `__feff_` 之间的字符串（即 formula 字段）
  - 从中提取所有大写字母开头的连续字母序列（元素符号）
  - 过滤掉非金属元素（O、N、C、H、S、P、F、Cl、Br、I、Se、Te），取剩余金属元素
  - 若只有一种金属元素，则该元素即为 `element`
  - 若仍有多种金属，标记为 `element=UNKNOWN`，记录警告日志
- `site_id`：`site_` 后的两位数字字符串（如 `01`、`02`）
- `is_ionic`：若来自 `test_missing_keep3_packed_A` → True，否则 → False
- `source_folder`：记录来源文件夹名（site_dataset 或 ionic），便于追踪

**3. 去重：**
- 若某 `folder_name`（完整文件夹名，不含路径）在两个数据集中均出现 → 保留 `site_dataset` 中的，删除 ionic 中的对应记录（**只从清单中删除，不删除磁盘文件**）
- 记录去重报告：`experiment\step1\dedup_report.txt`，列出所有被移除的重复条目数量

**4. 验证三文件完整性：**
- 对每个文件夹，检查 `chi.dat`、`xmu.dat`、`POSCAR_supercell_fixed` 是否存在
- 缺失任何一个文件的文件夹，在清单中标记 `files_complete=False`
- 记录缺失统计

**5. 输出 `data_inventory.csv`**（保存至 `experiment\step1\`），列包含：

```
folder_name, mp_id, element, site_id, is_ionic, source_path, 
files_complete, quality_tier(暂填NaN，Step1.3填充)
```

### 验收标准
- `data_inventory.csv` 存在且可被 pandas 读取
- 所有行的 `folder_name` 唯一（无重复）
- `dedup_report.txt` 存在，记录去重数量
- `element=UNKNOWN` 的条目数量 < 总数的 1%（否则说明解析逻辑有问题，需要检查）

---

## Step 1.2：解析谱文件和 POSCAR 文件格式验证

### 脚本名
```
step1.2_validate_file_formats.py
```

### 输入
```
experiment\step1\data_inventory.csv （Step 1.1 输出）
```

### 任务描述

**这个脚本只做验证，不做解析存储。** 目的是确认所有文件格式正常，发现异常文件。

**1. 抽样验证 chi.dat 格式：**
- 随机抽取 200 个 `files_complete=True` 的文件夹
- 读取其 `chi.dat`，验证：
  - 跳过 `#` 开头的注释行后，剩余行可被解析为至少两列浮点数
  - 第一列（k 值）范围大致在 -2 到 22 Å⁻¹ 之间
  - 数据点数 > 100
- 记录异常文件路径

**2. 抽样验证 xmu.dat 格式：**
- 同上，随机抽 200 个，验证能量列（第一列）范围合理（各元素 K 边位置不同，只验证数值可读）

**3. 抽样验证 POSCAR 格式：**
- 随机抽 100 个，使用 `pymatgen.core.Structure.from_file()` 尝试读取
- 记录读取失败的文件

**4. 输出 `format_validation_report.txt`**（保存至 `experiment\step1\`）：
```
总文件夹数: XXXX
chi.dat 格式异常数: XX （列出路径）
xmu.dat 格式异常数: XX （列出路径）
POSCAR 读取失败数: XX （列出路径）
```

**注意**：不需要把谱数据全部读进内存，这一步只做抽样验证。

### 验收标准
- `format_validation_report.txt` 存在
- POSCAR 读取失败率 < 5%（若超过，立即停止并报告给用户）
- chi.dat 格式异常率 < 2%

---

## Step 1.3：数据质量分级

### 脚本名
```
step1.3_quality_grading.py
```

### 输入
```
experiment\step1\data_inventory.csv
C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv
C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv
```

### 任务描述

**1. 加载两个特征表，合并去重：**
- 用 `sample_name` 列作为 key
- 若 `sample_name` 在两表中均出现 → 保留 site_v2 中的行，删除 ionic_v3 中的行
- 合并后记录总行数

**2. 为每个 `folder_name` 分配 `quality_tier`：**

```python
def assign_quality_tier(row):
    if row['flag_pre_valid'] == 1 and row['flag_white_valid'] == 1 and row['flag_post_valid'] == 1:
        return 'A'
    elif row['flag_white_valid'] == 1:
        return 'B'
    else:
        return 'C'
```

- 若某 `folder_name` 在特征表中找不到（未被提取特征）→ 标记为 `quality_tier='unknown'`

**3. 更新 `data_inventory.csv`，填充 `quality_tier` 列**

**4. 输出质量分布统计 `quality_summary.txt`**（保存至 `experiment\step1\`）：
```
总条目数: XXXXX
A 级（全有效）: XXXX (XX%)
B 级（白线有效）: XXXX (XX%)
C 级（异常）: XXXX (XX%)
unknown（未找到特征）: XXXX (XX%)

is_ionic=True 中的质量分布:
  A: XX%, B: XX%, C: XX%, unknown: XX%
is_ionic=False 中的质量分布:
  A: XX%, B: XX%, C: XX%, unknown: XX%
```

**5. 同时处理键长约束表：**
- 读取 `all_center_neighbors_summary.csv`
- 解析 `raw_range_A_minmax` 列（格式 `"2.511-2.996"`）为 `(float_min, float_max)`
- 构建字典 `{pair_str: (min_A, max_A)}`，例如 `{"Fe-O": (1.7, 2.8)}`
- 序列化保存为 `experiment\step1\bond_length_constraints.json`

### 验收标准
- `data_inventory.csv` 中 `quality_tier` 列无 NaN（只有 A/B/C/unknown）
- `quality_summary.txt` 存在
- `bond_length_constraints.json` 存在且可被 json.load 读取
- A+B 级合计 > 60%（否则说明数据有问题，报告给用户）

---

## Step 1.4：数据集划分（含 1000 结构保留集）

### 脚本名
```
step1.4_split_dataset.py
```

### 输入
```
experiment\step1\data_inventory.csv
```

### 任务描述

**1. 按 `mp_id` 聚合**  
所有划分操作均以 `mp_id` 为单位（一个 `mp_id` = 一个化合物 = 多个位点），确保同一化合物的所有位点都在同一个集合中。

**2. 获取所有有效 mp_id 列表**  
- 只考虑 `files_complete=True` 的条目
- 去除 `quality_tier='C'` 的条目所在的 mp_id（若某 mp_id 的所有位点均为 C 级，则排除该 mp_id）
- 注意：若某 mp_id 有部分 A/B 级位点、部分 C 级位点，该 mp_id 仍保留（C 级位点在训练时降权或忽略）

**3. 生成 1000 结构保留集（holdout set）：**

策略：按元素组合多样性采样，确保每类结构训练集仍有代表

```python
# 步骤：
# a. 为每个 mp_id 生成"元素组合标签"
#    = 该 mp_id 下所有位点的 element 集合，排序后连接
#    例：{Fe, Sc} → "Fe-Sc"，{Fe} → "Fe"
#
# b. 按元素组合标签分组
#
# c. 对每组按比例采样（目标 1000 / 总mp_id数 的比例）
#    约束：若某组只有 1 个 mp_id → 不放入 holdout
#         若某组只有 2-3 个 mp_id → 最多放 1 个入 holdout
#         若某组 > 3 个 mp_id → 按比例放入，但该组在训练集中至少保留 2 个
#
# d. 累计采样直到总数达到 1000
#
# e. 若采样结果 < 1000，从最大的组中补充
```

**4. 划分剩余数据为 train / val / test：**
- 比例：train 80% / val 10% / test 10%
- 按 `mp_id` 随机划分（固定 random_seed=42）
- 同样按元素组合分层采样（stratified split），确保分布均匀

**5. 输出 4 个 ID 文件**（保存至 `experiment\step1\`）：
```
holdout_1000_ids.txt    ← 每行一个 mp_id，共约 1000 个
train_ids.txt           ← 训练集 mp_id
val_ids.txt             ← 验证集 mp_id
test_ids.txt            ← 测试集 mp_id
```

**6. 输出划分统计 `split_summary.txt`**：
```
总有效 mp_id 数: XXXXX
保留集（holdout）: 1000
训练集: XXXX
验证集: XXXX
测试集: XXXX

保留集元素组合覆盖率: XX% （有多少种元素组合在保留集中有代表）
训练集中与保留集相邻的化合物数: XXXX （相同元素组合但不同 mp_id）
```

### 验收标准
- 4 个 ID 文件存在，格式为每行一个整数 mp_id
- `holdout_1000_ids.txt` 行数在 990-1010 之间（允许小误差）
- train + val + test + holdout = 总有效 mp_id 数（无遗漏）
- 每个 mp_id 只出现在一个集合中（无交叉）

---

## 完成后提交的总结报告格式

请按 SHARED_02 中规定的格式提交报告，额外需要包含：

```markdown
### 数据统计摘要
- 总文件夹数（去重后）: 
- is_ionic=True 条目数: 
- is_ionic=False 条目数: 
- files_complete=True 条目数: 
- 质量分级：A/B/C/unknown 各多少
- 有效 mp_id 总数: 
- 保留集 mp_id 数: 
- 训练集 / 验证集 / 测试集 mp_id 数: 

### 发现的异常（如有）
- 格式异常文件: （列出路径）
- element=UNKNOWN 的条目: （数量和示例）
- quality_tier=C 的 mp_id 数量: 
```

---

## 注意事项汇总

1. **绝对不读取 formula 字段**：文件夹名中的化学式（如 Fe2N）不能出现在任何输出文件中，也不能被解析为特征
2. **不删除磁盘文件**：去重只在清单中操作，不删除实际文件
3. **按 mp_id 而非 folder 划分集合**：同一化合物的所有位点必须在同一个集合中
4. **seed 固定为 42**：所有随机操作使用 `random.seed(42)` 和 `numpy.random.seed(42)`
5. **保留集一旦生成，严禁后续步骤读取其对应的谱文件或 POSCAR**
