# MAINAGENT2_HANDOFF.md
# Main Agent 2 交接文档：项目完整历史 + 当前状态 + 下一步指令

> **写给 Main Agent 2**：你是本项目的第二任 Main Agent，接替 Main Agent 1 继续指挥。
> 请完整阅读本文档后再开始工作。
> **日期**：2026-04-09

---

## 项目一句话

给定 Fe K-edge XAS 谱（xmu XANES + chi1 EXAFS + 73维物理先验），用改造后的 DiffCSP 扩散模型预测以 Fe 为中心的局部原子结构（最近 20 个邻居的类型和坐标）。

---

## 已完成的所有 Steps 与关键结论

### Step 1：数据清洗与清单构建 ✅

| 项目 | 结果 |
|------|------|
| 原始文件夹数 | 18,385 |
| 有效化合物数（LVSI后） | 11,636 |
| holdout | 787 个 |
| train / val / test | 7,595 / 1,627 / 1,627 |
| feff_features 路径 | `C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv` |
| E0 来源 | data_inventory.csv 的 E0 列（不从 feff_features 查） |
| xmu.dat 列索引（实测） | 能量 = data[:,0]，μ(E) = data[:,3] |
| 虚拟晶格边长 L | 12 Å（Step2.5 统计，d20_99th=5.14Å，2×5.14≈12） |

### Step 2：谱预处理 + SpectrumEncoder ✅

- `spectrum_preprocessor.py`：load_xmu_xanes / load_chi1 / load_feff_features
- `spectrum_encoder.py`：三路 CNN+MLP，输出 (B,256)，与 time_emb(256) 拼接为 condition(512)
- 20/20 样本验证通过，编码器前向测试通过

### Step 3：Dataset + 模型改造 ✅

- `xas_local_dataset.py` v5：Fe中心+最近20邻居，frac=cart/12，abs>0.5的样本过滤
- `diffusion_w_type_xas.py`：加 SpectrumEncoder，cost_lattice=0，晶格固定 diag(12,12,12)
- 前向测试：loss=2.66，loss_coord=1.27，loss_type=1.38 ✅

### Step 4：训练与评估 ✅（有根本性 bug 待修复）

**训练结果**：
- 500 epoch，epoch 489 early stop，val_loss = 0.6178 ✅（收敛正常）

**评估结果（有 bug，结果无效）**：

| 指标 | 值 | 含义 |
|------|-----|------|
| RMSD | 4.51 Å | ≈ 随机基线 4.65 Å，坐标预测无效 |
| Type Accuracy | 0.28 | 远高于随机 ~0.01，类型预测有效 ✅ |
| pred_in_cutoff | 2.75/20 | 预测原子几乎全在 4Å 外（异常） |
| true_in_cutoff | 17.23/20 | 真实原子绝大多数在 4Å 内（正常） |

---

## 当前根本性 Bug（Step4b 需要修复）

### Bug 描述

`xas_local_dataset.py` 存储坐标为 `frac ∈ [-0.5, 0.5]`（以 Fe 原点为中心）。

但 `diffusion_w_type_xas.py` 的 `forward()` 加噪时执行了 `(frac_coords + noise) % 1.`，把坐标强制映射到 `[0,1]`。

结果：Fe 周围的原子（坐标集中在原点附近）经过 `% 1.` 后分裂成双峰：`[0, 0.33] ∪ [0.67, 1.0]`。模型学到的是这个人为制造的双峰分布，采样时原子均匀散布全空间，RMSD 等同随机。

**Type Accuracy=0.28 说明模型学到了原子类型**，坐标失效是纯粹的坐标系不一致问题，不是模型能力问题。

### 确定修复方案（方案 A，已决策）

**修改 `xas_local_dataset.py`**，存储前对 frac_coords 做 `% 1.`：

```python
# 改前（v5）
frac_coords = (neighbor_carts / self.L).copy()

# 改后（v6）
frac_coords = ((neighbor_carts / self.L) % 1.0).copy()
```

这样 Dataset 输出的坐标在 `[0,1]`，与 `forward()` 的 `% 1.` 加噪完全一致。

**同步修改评估脚本**：去掉最小镜像修正，直接用 `[0,1]` 坐标做匈牙利匹配。

**需要重新训练**（旧 checkpoint 基于错误坐标系，不可复用）。

---

## 当前需要新开的窗口：Step4b Agent

Step4b Agent 的任务是在 Step4 Agent 对话达到上限后，接替完成修复和重训工作。

---

## 共享文档同步清单（你需要持有的文件）

以下文件你需要在对话开始时向用户索取，或确认可访问：

| 文件 | 位置 | 状态 |
|------|------|------|
| `SHARED_00_v2.md` | outputs/ | ✅ 最新版，有效 |
| `SHARED_01_DATA_MANIFEST.md` | outputs/ | ✅ 有效 |
| `SHARED_02_SPECTRAL_AND_MODEL.md` | outputs/ | ⚠️ xmu列索引有误，以Step1实测为准（data[:,0]和data[:,3]） |
| `EXP2_PROPOSAL_FINAL.md` | outputs/ | ✅ 定稿，仅参考 |
| `STEP4b_HANDOFF.md` | 见下方，你来生成 | — |

---

## 你的第一个任务：生成 Step4b Agent 交接文档

Step4b Agent 需要做：

1. **修改 `xas_local_dataset.py` → v6**（加 `% 1.0`，一行改动）
2. **修改评估脚本**（去掉最小镜像，改用 `[0,1]` 坐标系）
3. **重新训练**（从头，epoch=0，相同超参，相同 train_ids.txt）
4. **重新采样 + 重新评估**
5. **若 RMSD 仍 ≈ 随机基线（> 3.5 Å）**：需要进一步诊断（不太可能，Type Acc=0.28 已证明模型在学习）

Step4b 需要向用户索取的文件：
- `SHARED_00_v2.md`
- `experiment2/step3/xas_local_dataset.py`（v5，用于改 v6）
- `experiment2/step4/step4_2_train.py`（直接复用，不改）
- `experiment2/step4/step4_3_sample.py`（直接复用，不改）
- `experiment2/step4/step4_4_compute_metrics.py`（修改评估坐标系）

Step4b checkpoint 存放：
- `experiment2/step4b/checkpoints/`
- `experiment2/step4b/predictions_val.pt`
- `experiment2/step4b/predictions_test.pt`
- `experiment2/step4b/metrics_report.txt`

---

## 项目当前状态一览

```
Step 1  ✅ 完成，输出文件全部有效
Step 2  ✅ 完成，两个库文件可直接 import
Step 3  ✅ 完成，需要 v6 dataset（一行改动）
Step 4  ✅ 训练完成（val_loss=0.6178），评估有 bug，结果无效
Step 4b ⏳ 待开始：修复坐标系 bug，重训，重评估
Step 5  ⏳ 待开始：Holdout 检验，等 Step4b 完成后
```

---

## 关键路径常量（所有 Agent 统一使用）

```python
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
BOND_CSV      = r"C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv"
STEP1_DIR     = EXP2_ROOT + r"\step1"
L             = 12.0   # 虚拟晶格边长（Å）
N_NEIGHBORS   = 20
BATCH_SIZE    = 16
```

---

## Main Agent 2 工作原则（继承自 Main Agent 1）

1. **不写代码，只出交接文档和共享文档**，代码由各 Sub-Agent 实现
2. **每个 Sub-Agent 完成一个小步骤后汇报，确认无误再继续**，不跳步
3. **每次开新 Sub-Agent 窗口前，明确告诉用户要发给那个窗口哪些文件**
4. **任何数字结论（loss、RMSD、样本数）都要记录在本交接文档对应位置**，方便下一任 Main Agent 接替
5. **化学式/mp_id 禁止入模型**，始终坚守

---

*Main Agent 1 撰写，2026-04-09，移交 Main Agent 2*
