# SHARED_00_PROJECT_OVERVIEW.md（v2，含局部结构方案更新）
# 所有 Agent 必读背景文档

> **版本**：v2 — 更新输出目标为局部结构（Fe中心+最近20邻居）
> **日期**：2026-04-09，LOCKED

---

## 一句话任务

给定一条 Fe K-edge XAS 谱，预测以 Fe 为中心的**局部原子结构**：Fe 本身 + 最近 20 个邻居的原子类型和位置（共 20 个原子，不含 Fe 自身，即 Fe + 19邻居，或按下方定义见Step1.8）。

> **Step1.8 确认**：固定取最近 N=20 个邻居（Fe自身不计入20个），共 21 个原子输入模型（Fe + 20邻居）。
> 99% 样本的第20个邻居在 4Å 内，1% 缺失率可接受。

---

## 核心约束（所有 Agent 必须遵守）

1. **禁止元素/化学式信息入模型**：文件夹名中的 formula、mp_id 仅作数据管理键，绝对不能作为 tensor 输入。POSCAR 只用于提取坐标标签，不作为模型输入。

2. **全程使用原胞**：POSCAR 是超胞，必须先 get_primitive_standard_structure(symprec=0.1)，再从原胞中定位 Fe 位点和邻居。

3. **训练期间禁止接触 Holdout**：holdout_ids.txt 中的 mp_id 在 Step4 结束前禁止出现在任何流程中。

---

## 输出目标（已更新，与 Exp1 不同）

```
预测目标：以 LVSI Fe 位点为中心，最近 20 个邻居
  输出1：frac_coords  (20, 3)   分数坐标（虚拟晶格下）
  输出2：atom_types   (20,)     原子序数
  输出3：（晶格固定，不预测）

虚拟晶格：diag(L, L, L)，L 由 Step3 根据数据确定（预期 12-15 Å）
坐标系：以 Fe 为原点，笛卡尔坐标归零后除以 L 得到分数坐标
原子数：固定 20（极少数 <20 邻居的样本排除）

训练配置：cost_lattice = 0（或 1e-4 极小值），晶格不参与扩散
```

---

## 评估截断（动态，非固定 4Å）

```
每个样本的评估截断距离 = min(该样本第20邻居的实际距离, 4.0 Å)

含义：
  - 结构复杂（密堆积）：20个邻居都在3.5Å内 → 只评估3.5Å内（即全部20个）
  - 结构稀疏：第20邻居超出4Å → 只评估4Å内的子集

评估时：对预测结果中距 Fe 原点 ≤ eval_cutoff 的原子计算 RMSD 和 Type Accuracy
        不使用 StructureMatcher（不适合非周期局部簇）
```

---

## 模型框架（DiffCSP，最小改动）

```
零改动文件：cspnet.py, diff_utils.py, gnn.py, run.py

修改文件：
  diffusion_w_type.py  ← 加 SpectrumEncoder；cost_lattice=0；改 forward/sample
  dataset.py           ← 重写：Fe定位+20邻居截取+虚拟晶格
  datamodule.py        ← 小改：按 id 文件划分
  conf/data/xas_fe_only.yaml   ← 新建
  conf/model/diffusion_xas.yaml ← 新建，latent_dim=512
```

---

## 三路条件输入

| 输入 | 来源 | 形状 | 物理含义 |
|------|------|------|----------|
| xmu_xanes | xmu.dat，截取 [E₀-50, E₀+150 eV] | (150,) | 氧化态、对称性 |
| chi1 | chi1.dat 全 k 空间 | (200,) | 键长、配位数 |
| feff_features | tesst_feff_features_all_full_v4.csv | (73,) | 物理先验 |

---

## 路径常量

```python
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
DIFFCSP_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main"

FEFF_FEAT_CSV  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
BOND_CSV       = r"C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv"

STEP1_DIR      = EXP2_ROOT + r"\step1"
INVENTORY_CSV  = STEP1_DIR + r"\data_inventory.csv"
SITE_MAP_CSV   = STEP1_DIR + r"\selected_site_map.csv"
TRAIN_IDS      = STEP1_DIR + r"\train_ids.txt"
VAL_IDS        = STEP1_DIR + r"\val_ids.txt"
TEST_IDS       = STEP1_DIR + r"\test_ids.txt"
HOLDOUT_IDS    = STEP1_DIR + r"\holdout_1000_ids.txt"
FEAT_SCALER    = STEP1_DIR + r"\feff_feature_scaler.pkl"
FEAT_STATS     = STEP1_DIR + r"\feff_feature_stats.csv"
PRIM_NATOMS    = STEP1_DIR + r"\prim_natoms_all.csv"
```

---

## 硬件配置

```
GPU：RTX A4000（16GB），Windows，Python 3.9
num_workers=0（Windows多进程不稳定）
精度：bf16，gradient_clip=1.0
batch_size：建议 16（固定20邻居，每batch约320个原子节点）
```

---

## 脚本命名规范

```
格式：step{N}.{M}_{描述}.py（文件名不含点，用下划线）
例：spectrum_preprocessor.py（在文件头注释标注 "Step 2.1"）
输出目录：experiment2/step1/ step2/ step3/ step4/ step5/
```

---

## Sub-Agent 汇报模板

```
## Step X.Y 完成报告
**执行内容**：
**Actions**：
**Results**：
**输出文件**：
**异常/发现**：
**需要 Main Agent 决策的问题**：
```
