# EXP4_MAINAGENT_HANDOFF.md
# DiffCSP-Experiment4 Main Agent 交接文档

> **写给 DiffCSP-Experiment4-MainAgent**
> **由 Experiment 2 Main Agent 2 撰写**
> **日期**：2026-04-09

---

## 你是谁，你要做什么

你是 Experiment 4 的 Main Agent。你的工作是：

1. **读懂 Experiment 2 的完整代码和流程**（你会收到所有 Exp2 脚本）
2. **判断哪些脚本需要改动，哪些可以直接复用**
3. **为每一个需要改动的步骤，写一份 Sub-Agent 交接文档**，然后告诉用户要给那个 Sub-Agent 发哪些文件
4. **不写代码**，代码由各 Sub-Agent 实现

改动后的文件全部存入 `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\`（内含 step1/ step2/ step3/ step4/ step5/）。不需要改动的脚本，直接在代码里引用 experiment2 的路径，不复制。

---

## Experiment 2 做了什么（必须理解）

### 任务

给定一条 **Fe** K-edge XAS 谱，预测以 **Fe** 为中心的最近 20 个邻居原子的类型和坐标。

### 数据集

- 来源：Materials Project，仅限 **Fe 氧化物**，单中心原子（Fe）
- 文件夹总数：18,385，有效化合物：11,636
- 命名格式：`mp_{id}_{formula}_feff_Fe_site_{nn}`（中心元素固定为 Fe）
- 每个文件夹包含：`chi1.dat`、`xmu.dat`、`POSCAR_supercell_fixed`

### 模型核心设计（需要理解，部分会改）

**虚拟晶格**：`diag(6, 6, 6)`，L=6 Å  
**坐标系**：以中心原子为原点，frac = cart / L，再做 min-image 折叠 → [-0.5, 0.5]  
**条件输入**：三路 SpectrumEncoder（xmu XANES 150点 + chi1 EXAFS 200点 + feff_features 73维）→ latent (B, 256)  
**扩散框架**：DiffCSP，cost_lattice=0（晶格固定不预测）

### Experiment 2 最终结果

| 指标 | 值 |
|------|-----|
| RMSD（holdout） | 1.47 Å |
| Type Accuracy（holdout） | 0.241 |

### Experiment 3 为什么不用

Experiment 3 加了独立的原子类型分类 Head，结果没有改善。原因：训练集里超过 50% 的邻居原子是 O，分类器无脑猜 O 就能得到高 loss 下降，但实际没有学到有意义的区分能力。**Experiment 4 不加 TypeClassifier，直接在 Exp2 基础上修改。**

---

## Experiment 4 的改动

### 核心变化：更大、更通用的数据集

| 项目 | Experiment 2 | Experiment 4 |
|------|-------------|-------------|
| 数据集范围 | 仅 Fe 氧化物 | **所有包含 XANES 和 EXAFS 的 Materials Project 材料** |
| 文件夹总数 | 18,385 | **~40,000+** |
| 中心元素 | 固定为 Fe | **任意元素**（由文件夹名决定） |
| feff_features 文件 | tesst_feff_features_all_full_v4.csv | **需要确认新文件路径** |

### 已知的文件夹命名变化

用户说新数据集的命名规则与 Exp2 有所不同。**你的第一个任务（Step 0）就是向用户询问新的命名格式**，然后再开始后续步骤。具体见下方"你的第一步"。

### 不变的内容

以下设计决策已经过充分验证，**不得修改**：
- L=6 Å，虚拟晶格 diag(6,6,6)
- 坐标系 [-0.5, 0.5]，min-image 折叠（`frac -= np.round(frac)`）
- forward() 无 `% 1.`
- N_NEIGHBORS = 20
- batch_size=16，lr=1e-4，bf16，num_workers=0
- 三路 SpectrumEncoder 架构（不加 TypeClassifier）
- DiffCSP 扩散框架，cost_lattice=0

---

## 你的第一步：Step 0 信息收集（直接问用户，不开 Sub-Agent）

在写任何 Sub-Agent 交接文档之前，你需要先向用户确认以下信息。**直接以问题列表的形式问用户，等用户回答后再继续工作。**

**问题 1**：新数据集的文件夹命名格式是什么？请提供 3-5 个真实文件夹名作为示例。
- Exp2 的格式是：`mp_204_CeFe2_feff_Fe_site_02`
- Exp4 的格式是？（用于解析 mp_id、中心元素、site_nn）

**问题 2**：新数据集的根目录路径是什么？
- Exp2 是：`C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site`
- Exp4 是？

**问题 3**：新数据集的 feff_features CSV 文件路径是什么？（73维物理先验特征）
- Exp2 是：`C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv`
- Exp4 是否有对应的新文件，还是复用旧文件？

**问题 4**：每个文件夹内的文件名是否和 Exp2 相同？（chi1.dat、xmu.dat、POSCAR_supercell_fixed）

**问题 5**：E0（吸收边能量）的来源是否变化？
- Exp2 中 E0 来自 data_inventory.csv 的 E0 列（Step1 统计时从 feff_features 取）
- Exp4 是否相同？

收到用户回答后，你才开始规划哪些脚本需要改。

---

## 预判：哪些步骤大概率需要改（供你参考，不是定论）

收到用户对 Step 0 的回答后，你需要自己判断。以下是基于已知信息的预判：

### 肯定需要改：Step 1（数据清洗与清单构建）

- 文件夹命名格式变了 → 解析逻辑需要更新
- 中心元素不再固定是 Fe → 解析时需要提取中心元素字段
- 数据根目录变了 → 路径常量需要更新
- 数据量从 18k 增加到 40k+ → 可能需要调整 Holdout 策略（原来 787 个，新的可以更多）
- feff_features 文件可能变了 → 匹配键逻辑可能需要更新

**关键**：Step 1 输出的 `data_inventory.csv` 需要新增 `center_element` 列，记录每个样本的中心原子元素符号，供后续步骤使用。

### 肯定需要改：Step 3 Dataset

- 中心原子定位逻辑：Exp2 中用"找 Fe 位点"定位中心原子（硬编码了 Fe）
- Exp4 中需要改为"找 `center_element` 位点"，从 data_inventory 读取每个样本的中心元素

**具体改动位置**：`xas_local_dataset.py` 中寻找中心原子的逻辑（大约 5-10 行）

### 可能需要改：Step 2 谱预处理

- xmu.dat 的列格式在 Exp2 中实测是 `data[:,0]`（能量）和 `data[:,3]`（μ(E)）——注意是第 4 列不是第 2 列，这是 Exp2 踩过的坑
- 新数据集需要验证列格式是否相同，如果不同需要更新
- E0 来源如果变了也需要更新

### 可能需要改：Step 1 Holdout 策略

- Exp2 用 K-Means 对 feff_features 聚类抽取 787 个 Holdout
- Exp4 数据量是 4 倍，Holdout 可以考虑扩大到 2000-3000 个（比例维持约 5-8%）
- 这个由你决策，不需要用户确认

### 大概率不需要改：Step 2 SpectrumEncoder 架构

- 输入维度不变（xmu 150点，chi1 200点，feff_features 73维）
- 输出维度不变（latent 256维）
- 直接复用 `experiment2/step3/spectrum_encoder.py`

### 大概率不需要改：Step 4 训练脚本

- 超参不变
- 路径常量更新后可以复用
- 直接复用 `experiment2/step4/step4_2_train.py`（可能只需改路径）

### 大概率不需要改：diffusion_w_type_xas.py

- 扩散框架不变
- SpectrumEncoder 接入方式不变
- 直接复用 `experiment2/step3/diffusion_w_type_xas.py`（Step4c 版）

---

## 你收到 Step 0 回答后的工作流程

1. **规划改动清单**：根据用户回答，列出所有需要改的文件，告诉用户你的判断
2. **Step 1 Sub-Agent**：写 Step1 交接文档，告诉用户发哪些文件给它
3. 等 Step 1 完成汇报，确认 data_inventory 格式正确，有 center_element 列
4. **Step 2 验证 Sub-Agent**（如果需要）：验证新数据集的谱文件列格式
5. **Step 3 Sub-Agent**：写 Step3 Dataset 交接文档（主要改中心原子定位逻辑）
6. 等 Step 3 完成前向测试，确认 loss 正常
7. **Step 4 Sub-Agent**：写训练交接文档
8. 依次完成训练、评估、Holdout 检验

---

## 路径常量模板（Step 0 回答后填入）

```python
# Experiment 4 路径常量（待填入）
DATA_ROOT_EXP4  = r"[用户回答问题2后填入]"
EXP4_ROOT       = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
FEFF_FEAT_CSV   = r"[用户回答问题3后填入]"

# 继承 Experiment 2 的路径（不变）
EXP2_ROOT       = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
STEP1_DIR_EXP4  = EXP4_ROOT + r"\step1"

# 不变的模型参数
L               = 6.0
N_NEIGHBORS     = 20
BATCH_SIZE      = 16
```

---

## Sub-Agent 交接文档的格式要求

你为每个 Sub-Agent 写的交接文档，必须包含：

1. **背景**：这个步骤要做什么，与 Exp2 的区别在哪里
2. **需要用户提供的文件清单**：精确到文件名和路径
3. **改动内容**：精确描述每个文件需要改哪里、改成什么（不写代码，写意图）
4. **验证方法**：改完后如何确认改对了（打印什么，检查什么数值）
5. **输出文件清单**：改好的文件存在哪里
6. **汇报模板**：完成后按这个格式汇报

---

## 工作原则（继承 Experiment 2）

1. **不写代码**，只出交接文档，代码由 Sub-Agent 实现
2. **每步汇报确认后再继续**，不跳步
3. **化学式/mp_id 禁止入模型**（元素符号/mp_id 只作数据管理键）
4. **Holdout 全程封存**，训练结束前禁止接触
5. **num_workers=0**（Windows 多进程不稳定）
6. 所有关键数字（loss、RMSD、样本数）记录在工作文档里

---

## 你要给用户的第一条回复格式

读完本文档后，你的第一条回复应该是：

```
我已阅读完所有文档，理解了 Experiment 2 的完整流程和 Experiment 4 的改动目标。

在开始规划之前，我需要向你确认以下信息：

[直接列出 Step 0 的 5 个问题]

收到你的回答后，我会告诉你哪些脚本需要改，并开始为第一个 Sub-Agent 出交接文档。
```

---

*Experiment 2 Main Agent 2 撰写，2026-04-09*
