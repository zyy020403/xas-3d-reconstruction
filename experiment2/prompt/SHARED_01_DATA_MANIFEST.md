# SHARED_01_DATA_MANIFEST.md
# 数据文件说明与格式规范

> **适用范围**：所有 Agent
> **状态**：LOCKED

---

## 1. 数据根目录结构

```
C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site\
│
├── mp_204_CeFe2_feff_Fe_site_02\         ← 示例：site 从 02 开始（说明 01 计算失败）
│   ├── chi1.dat        ★ 使用
│   ├── xmu.dat         ★ 使用
│   ├── POSCAR_supercell_fixed  ★ 使用
│   ├── chi.dat         × 不使用
│   ├── chi2.dat        × 不使用
│   └── feff.inp        × 不使用
│
├── mp_13494_Nd3Fe29_feff_Fe_site_04\     ← 示例：该化合物 site_01~03 均无效
│   └── ...（同上）
│
└── mp_999189_Fe3Ni_feff_Fe_site_01\      ← 示例：多 site 化合物之一
    └── ...
    mp_999189_Fe3Ni_feff_Fe_site_02\      ← 同一化合物另一个 site
    └── ...
```

**文件夹总数**：18,385
**化合物总数（去重后）**：~12,956（每个 mp_id 算一个化合物）

---

## 2. 文件夹命名解析

```
格式：mp_{id}_{formula}_feff_Fe_site_{nn}

字段说明：
  {id}      整数，Materials Project 内部 ID，仅作数据管理键
  {formula} 化学式（如 CeFe2, FeO, Nd3Fe29）★ 禁止入模型
  {nn}      两位数字，位点序号，从 01 起始

Python 解析示例：
  folder_name = "mp_13494_Nd3Fe29_feff_Fe_site_04"
  parts = folder_name.split("_")
  mp_id   = parts[0] + "_" + parts[1]   # "mp_13494"
  site_nn = parts[-1]                    # "04"
  # formula = 中间部分，提取但不入模型
```

---

## 3. chi1.dat 格式

```
# FEFF output: k¹χ(k)
# 列1: k (Å⁻¹)     ← 波矢
# 列2: chi1(k)     ← k¹χ(k)，已由 FEFF 完成 k¹ 加权，不再额外处理
# 列3+: 其他（不使用）
# 注释行以 # 开头

示例（前几行）：
  2.000  0.0312
  2.050 -0.0124
  2.100  0.0089
  ...

典型范围：k ∈ [0, 12] Å⁻¹，约 100-300 行
有效性判断：行数 ≥ 30，第2列 std > 自动阈值（Step1 确定）
```

---

## 4. xmu.dat 格式

```
# FEFF output: normalized absorption μ(E)
# 列1: E (eV)      ← 能量
# 列2: μ(E)        ← 归一化吸收系数（主要用这列）
# 列3: μ₀(E)       ← 原子背景（不单独使用）
# 列4+: 其他（不使用）
# 注释行以 # 开头

示例（前几行）：
  7116.381  0.0021  0.0019
  7116.531  0.0035  0.0020
  ...

典型范围：E ∈ [7100, 8700] eV（Fe K-edge），约 500-1000 行
有效性判断：行数 ≥ 50，第2列无全零/NaN

★ XANES 截取窗口（模型输入）：
  [E₀ - 50 eV,  E₀ + 150 eV]，共 200 eV，插值到 150 点
  E₀ 来源：feff_features_all_site_v2.csv 的 E0 列（列索引 6）
```

---

## 5. POSCAR_supercell_fixed 格式

```
标准 VASP POSCAR 格式，pymatgen 可直接解析。
注意：这是超胞（约 64-70 个原子），不能直接用作训练标签。
必须在代码内部转换为原胞：

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

structure = Structure.from_file(poscar_path)
analyzer  = SpacegroupAnalyzer(structure, symprec=0.1)
primitive = analyzer.get_primitive_standard_structure()
# primitive 才是训练标签的来源
```

---

## 6. feff_features_all_site_v2.csv

```
位置：C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv
行数：约 12,000+（每行一个 site）

列结构（共 76 列）：
  列0: sample_dir      ← 旧路径，不使用
  列1: sample_name     ← 匹配键来源，格式：mp_1047285_FeO2__feff_site_01
  列2: feature_version ← 字符串，不使用
  列3-75: 数值特征（73列）← 全部使用

重要列（用于 Holdout 聚类）：
  列6:  E0              吸收边能量 (eV)
  列12: white_line_I    白线强度
  列69: R1_peak_pos     第一壳层峰位置 (Å)
  列67: chi_kmin / 列68: chi_kmax  k 空间范围

匹配键逻辑：
  sample_name = "mp_1047285_FeO2__feff_site_01"
  提取：mp_id = "mp_1047285"，site_nn = "01"
  与文件夹名提取的 (mp_id, site_nn) 做 JOIN
```

---

## 7. all_center_neighbors_summary.csv（键长约束表）

```
位置：C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv
用途：Step4 评估时计算键长违规率

使用列：
  B 列：元素对（如 Fe-O, Fe-Fe）
  F 列：真实统计的 min/max 键长（Å）
  D 列：5-95 置信区间（备用，若 F 列效果不好再换）
```

---

## 8. data_inventory.csv（Step1 输出，后续 Step 读取）

Step1 生成，字段定义：

```
mp_id          str    Materials Project ID（如 mp_13494）
formula        str    化学式（仅记录，不入模型）
site_nn        str    代表 site 序号（LVSI 结果，如 "01"）
folder_name    str    完整文件夹名
folder_path    str    完整绝对路径
total_sites    int    该化合物总 site 数
chi1_valid     bool   chi1.dat 有效
xmu_valid      bool   xmu.dat 有效
poscar_valid   bool   POSCAR 可解析
prim_n_atoms   int    原胞转换后原子数（-1 表示转换失败）
has_feff_feat  bool   feff_features 表中有对应行
quality_tier   str    A/B/C（Step1 按原子数和文件质量分级）
split          str    train/val/test/holdout/excluded
```
