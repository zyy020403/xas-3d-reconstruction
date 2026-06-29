# STEP4b_HANDOFF.md
# Step4b Agent 交接文档：坐标系修复 + 重训 + 重评估

> **写给 Step4b Agent**：Step4 训练已正常收敛，但评估结果因坐标系 bug 完全无效。
> 你的任务是修复 bug、重新训练、重新评估，并汇报结果。
> **日期**：2026-04-09
> **由 Main Agent 2 撰写**

---

## 背景：为什么需要 Step4b

Step4 训练结果（val_loss=0.6178，500 epoch）证明模型结构和训练流程本身正常。
但评估发现根本性 bug，导致坐标预测完全失效：

| 指标 | Step4 结果 | 含义 |
|------|-----------|------|
| RMSD | 4.51 Å | ≈ 随机基线 4.65 Å，坐标预测无效 |
| Type Accuracy | 0.28 | 远高于随机 ~0.01，类型预测有效 ✅ |
| pred_in_cutoff | 2.75 / 20 | 预测原子几乎全在 4Å 外（异常） |
| true_in_cutoff | 17.23 / 20 | 真实原子绝大多数在 4Å 内（正常） |

---

## Bug 根因（一句话）

`xas_local_dataset.py`（v5）存储的 frac_coords ∈ **[-0.5, 0.5]**（以 Fe 为原点中心化），
但 `diffusion_w_type_xas.py` 的 `forward()` 加噪后执行 `(frac_coords + noise) % 1.`，
强制把坐标映射到 **[0, 1]**。

两端坐标系不一致 → 原本聚集在原点附近的原子坐标被撕裂成双峰 `[0, 0.33] ∪ [0.67, 1.0]`
→ 模型学到的是人为双峰分布 → 采样时原子均匀散布全空间 → RMSD ≈ 随机基线。

Type Accuracy=0.28 证明模型的特征提取和学习能力正常，坐标失效是纯粹的坐标系不一致，
**不是模型能力问题**。

旧 checkpoint 基于错误坐标系训练，**不可复用，必须从头重训**。

---

## 你需要向用户索取的文件

请在开始任何工作前，要求用户提供以下文件：

| 文件 | 用途 |
|------|------|
| `experiment2/step3/xas_local_dataset.py` | v5 → 修改为 v6（一行改动） |
| `experiment2/step4/step4_2_train.py` | 直接复用，不改动 |
| `experiment2/step4/step4_3_sample.py` | 直接复用，不改动 |
| `experiment2/step4/step4_4_compute_metrics.py` | 修改评估坐标系 |

参考文档（本次交接文档包含的内容已足够，无需额外索取）：
- `SHARED_00_v2.md`（已含路径常量、虚拟晶格参数等）

---

## 四项子任务

### Task A：修改 xas_local_dataset.py（v5 → v6）

**改动位置**：找到 frac_coords 赋值的那一行，在整除之后加上 `% 1.0`。

**改动意图**：使 Dataset 输出的 frac_coords 统一为 `[0, 1]` 范围，与 `forward()` 中的 `% 1.` 加噪操作保持一致。

**改动内容描述**（不是代码，是意图）：
- 改前：`frac_coords = (neighbor_carts / self.L).copy()`
  → 结果范围 ∈ [-0.5, 0.5]（中心化坐标）
- 改后：`frac_coords = ((neighbor_carts / self.L) % 1.0).copy()`
  → 结果范围 ∈ [0, 1]（与 forward 一致）

**改后验证**：取 5 个样本，打印 frac_coords 的 min 和 max，确认全部在 [0, 1] 范围内，无负值。

**保存为**：`experiment2/step3/xas_local_dataset_v6.py`
（文件头注释标注版本号 v6 和修改说明）

---

### Task B：修改 step4_4_compute_metrics.py（评估脚本）

**改动意图**：评估脚本中原先对坐标做了"最小镜像修正"（把坐标折叠回 [-0.5, 0.5] 再计算距离），这与旧的 [-0.5, 0.5] 坐标系配套。现在 Dataset 输出 [0, 1]，预测结果也在 [0, 1]，需要去掉最小镜像修正。

**改动内容描述**：
1. 删除或注释掉所有形如 `coords - round(coords)` 或 `coords % 1 - 0.5` 的最小镜像操作。
2. 匈牙利匹配时直接使用 [0, 1] 范围的分数坐标计算距离矩阵（笛卡尔距离 = frac × L，L=12）。
3. eval_cutoff 逻辑不变：仍按 `min(该样本第20邻居实际距离, 4.0 Å)` 截断。

**改后保存为**：`experiment2/step4b/step4b_4_compute_metrics.py`

---

### Task C：重新训练

**训练脚本**：直接使用 `experiment2/step4/step4_2_train.py`，不做任何改动。

**唯一变化**：
- Dataset 改为导入 `xas_local_dataset_v6.py`（Task A 的输出）
- Checkpoint 输出目录改为 `experiment2/step4b/checkpoints/`

**超参与 Step4 完全一致**：

```
batch_size    = 16
lr            = 1e-4
max_epochs    = 500
early_stop    patience = 30
gradient_clip = 1.0
精度          = bf16
num_workers   = 0
L             = 12.0（虚拟晶格边长，Å）
N_NEIGHBORS   = 20
```

**开训前强制检查**（必须执行，不可跳过）：
用修改后的 Dataset v6 加载 5 个样本，确认：
1. frac_coords.min() ≥ 0，frac_coords.max() ≤ 1
2. 5 个样本的 frac_coords 均无 NaN/Inf
3. 只有确认通过后才开始训练

**训练过程记录**：记录 best val_loss 和对应 epoch，写入汇报。

---

### Task D：重新采样 + 重新评估

**采样脚本**：直接使用 `experiment2/step4/step4_3_sample.py`，不做任何改动。

采样输出保存至：
- `experiment2/step4b/predictions_val.pt`
- `experiment2/step4b/predictions_test.pt`

**评估脚本**：使用 Task B 修改后的 `experiment2/step4b/step4b_4_compute_metrics.py`。

评估结果保存至：`experiment2/step4b/metrics_report.txt`

---

## 验收标准

| 指标 | 目标 | 若未达标 |
|------|------|----------|
| RMSD（val set） | < 2.0 Å | 若 > 3.5 Å，需汇报 Main Agent 2 进一步诊断 |
| pred_in_cutoff | 接近 true_in_cutoff（约 17/20） | 若仍 ≤ 5/20，说明坐标系修复可能未生效 |
| Type Accuracy | ≥ 0.28（不低于 Step4） | 若显著下降，需检查 Dataset v6 是否破坏了类型标签 |
| best val_loss | 参考 Step4（0.6178），预期相近 | 若 > 1.5，说明训练异常，需检查 |

---

## 输出文件清单

所有输出统一写入 `experiment2/step4b/`：

```
experiment2/step4b/
├── checkpoints/
│   └── best_model.ckpt         ← 训练产出
├── predictions_val.pt           ← 采样产出（val set）
├── predictions_test.pt          ← 采样产出（test set）
├── step4b_4_compute_metrics.py  ← Task B 修改后的评估脚本
└── metrics_report.txt           ← 最终指标报告
```

---

## 汇报模板（完成后按此格式向 Main Agent 2 汇报）

```
## Step4b 完成报告

**执行内容**：坐标系 bug 修复 + 重训 + 重评估

**Task A（Dataset v6）**：
  - 改动确认：frac_coords 范围验证结果（min/max）
  - 5样本验证：通过 / 未通过

**Task B（评估脚本）**：
  - 最小镜像修正已删除：是 / 否
  - 修改说明：

**Task C（训练）**：
  - best val_loss：
  - best epoch：
  - 总训练时间：

**Task D（评估）**：
  - RMSD（val）：
  - RMSD（test）：
  - Type Accuracy（val）：
  - pred_in_cutoff（val）：
  - true_in_cutoff（val）：

**输出文件**：
  - experiment2/step4b/checkpoints/best_model.ckpt ✅/❌
  - experiment2/step4b/predictions_val.pt ✅/❌
  - experiment2/step4b/predictions_test.pt ✅/❌
  - experiment2/step4b/metrics_report.txt ✅/❌

**异常/发现**：

**需要 Main Agent 2 决策的问题**：
```

---

## 关键路径常量（供参考）

```python
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
STEP1_DIR     = EXP2_ROOT + r"\step1"
STEP4b_DIR    = EXP2_ROOT + r"\step4b"
L             = 12.0   # 虚拟晶格边长（Å）
N_NEIGHBORS   = 20
```

---

## 注意事项

1. **化学式/mp_id 禁止入模型**，Dataset v6 继承此约束，改动不涉及此部分，无需额外检查。
2. **num_workers=0**，Windows 多进程不稳定，不得修改。
3. **Holdout 集禁止接触**，训练和评估只用 train/val/test。
4. 若训练中途中断，可从 checkpoint 续训，但必须确认 Dataset 已切换为 v6。
5. Task A、B 完成后，**必须先通过开训前强制检查**，再启动 Task C。

---

*Main Agent 2 撰写，2026-04-09*
