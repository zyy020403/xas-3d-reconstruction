# SHARED_00_PROJECT_OVERVIEW.md
# 所有 Agent 必读背景文档

> **适用范围**：Step1 ~ Step5 所有 Agent 窗口
> **状态**：LOCKED

---

## 一句话任务

给定一条 Fe K-edge XAS 谱，预测对应材料的**原胞晶体结构**（分数坐标 + 晶格参数 + 原子类型）。

---

## 核心约束（所有 Agent 必须遵守）

1. **禁止元素/化学式信息入模型**：文件夹名中的 `{formula}`（如 CeFe2）和 mp_id 仅作数据管理键，绝对不能作为 tensor 输入模型。POSCAR 只用于提取坐标标签，不作为模型输入。

2. **全程使用原胞**：POSCAR 文件是超胞（~64-70个原子），必须在 dataset 内部调用 `get_primitive_standard_structure(symprec=0.1)` 转换为原胞后再提取标签。这是 Experiment 1 的致命教训，不可省略。

3. **训练期间禁止接触 Holdout**：`experiment2/step1/holdout_1000_ids.txt` 中的 mp_id 在 Step4 结束前禁止出现在任何训练/验证/测试流程中。

---

## 模型框架：DiffCSP（最小改动）

**基础框架**：DiffCSP（NeurIPS 2023），扩散模型做晶体结构预测。

**原版任务**：化学组成 → 晶体结构

**Exp2 改动**：仅替换条件输入，其余全部复用。

```
改动的文件：
  diffcsp/pl_modules/diffusion_w_type.py  ← 加 SpectrumEncoder，改条件拼接
  diffcsp/pl_data/dataset.py              ← 重写，读谱+POSCAR+特征
  diffcsp/pl_data/datamodule.py           ← 小改，适配新路径

零改动的文件（直接复用）：
  diffcsp/pl_modules/cspnet.py
  diffcsp/pl_modules/diff_utils.py
  diffcsp/pl_modules/gnn.py
  diffcsp/run.py
```

---

## 三路条件输入

| 输入 | 来源 | 形状 | 物理含义 |
|------|------|------|----------|
| xmu_xanes | xmu.dat 截取 XANES 窗口 | (150,) | 氧化态、点群对称性 |
| chi1 | chi1.dat 全 k 空间 | (200,) | 键长、配位数 |
| feff_features | feff_features_all_site_v2.csv | (73,) | 提炼的物理先验 |

**注意**：chi1.dat 已经是 k¹χ(k)（FEFF 完成加权），不需要额外 k 加权操作，只做插值和归一化。

---

## 输出（原版 DiffCSP 格式，不变）

```
frac_coords  (N, 3)   分数坐标
atom_types   (N,)     原子序数（整数）
lengths      (3,)     晶格参数 a, b, c（Å）
angles       (3,)     晶格角 α, β, γ（度）
num_atoms    int      原胞原子数 N
```

---

## 项目文件路径常量

```python
# 所有脚本统一使用以下路径常量

DATA_ROOT    = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT    = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
DIFFCSP_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"

FEFF_FEATURES_CSV   = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv"
BOND_CONSTRAINTS_CSV= r"C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv"

# Step1 输出（后续 Step 读取）
STEP1_DIR      = EXP2_ROOT + r"\step1"
INVENTORY_CSV  = STEP1_DIR + r"\data_inventory.csv"
SITE_MAP_CSV   = STEP1_DIR + r"\selected_site_map.csv"
TRAIN_IDS      = STEP1_DIR + r"\train_ids.txt"
VAL_IDS        = STEP1_DIR + r"\val_ids.txt"
TEST_IDS       = STEP1_DIR + r"\test_ids.txt"
HOLDOUT_IDS    = STEP1_DIR + r"\holdout_1000_ids.txt"
FEAT_SCALER    = STEP1_DIR + r"\feff_feature_scaler.pkl"
```

---

## 硬件配置

```
GPU：NVIDIA RTX A4000（16GB 显存）
CPU：（含 Intel UHD 集成显卡，不参与训练）
OS：Windows 10/11
Python：3.9
PyTorch：已安装（含 torch-geometric、torch-scatter）
关键注意：num_workers=0（Windows 多进程不稳定）
精度：bf16（A4000 支持）
```

---

## 脚本命名规范

```
格式：step{大步}.{小步}_{描述}.py
示例：step1.1_scan_folders.py
      step1.3_primitive_cell_check.py
      step3.3_modify_diffusion.py

所有脚本输出存放至对应的 step 子目录：
  experiment2/step1/  step2/  step3/  step4/  step5/
```

---

## Sub-Agent 工作报告模板

每个 Sub-Agent 完成工作后，向 Main Agent 提交报告，格式如下：

```
## Step X.Y 完成报告

**执行内容**：[一句话描述]

**Actions**：
  - [具体操作1]
  - [具体操作2]

**Results**：
  - [关键数字/结果1]
  - [关键数字/结果2]

**输出文件**：
  - experiment2/stepX/filename.py
  - experiment2/stepX/filename.csv

**异常/发现**：
  - [任何意外情况或需要 Main Agent 决策的问题]

**下一步建议**：[可选]
```
