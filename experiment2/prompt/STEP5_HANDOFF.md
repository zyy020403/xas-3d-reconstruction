# STEP5_HANDOFF.md
# Step5 Agent 交接文档：Holdout 检验

> **写给 Step5 Agent**
> **日期**：2026-04-09
> **由 Main Agent 2 撰写**

---

## 你的任务

用 Step4d 训练好的模型对 Holdout 集（787个样本）做推断和评估。

**严格禁止**：
- 任何形式的模型修改
- 任何形式的重训或 fine-tune
- 修改 Holdout 样本列表
- 用 Holdout 结果反向调整模型

这是最终的盲测检验，结果只能观察，不能干预。

---

## 背景：Step4d 最终结果

| 指标 | val | test |
|------|-----|------|
| RMSD | 1.47 Å | 1.47 Å |
| Type Accuracy | 0.249 | — |
| pred_in_cutoff | 17.47/20 | — |
| best val_loss | 0.8554（epoch 249） | — |

Holdout 检验的预期结果应与 val/test 接近（模型泛化性良好，val/test 完全一致）。
若 Holdout RMSD > 2.5 Å 或 Type Accuracy < 0.20，属于异常，需汇报 Main Agent 2。

---

## 你需要向用户索取的文件

| 文件 | 用途 |
|------|------|
| `experiment2/step3/xas_local_dataset.py`（Step4d 版，L=6+min-image） | 加载 Holdout 数据 |
| `experiment2/step4d/checkpoints/best_model.ckpt` | 推断用 checkpoint |
| `experiment2/step4/step4_3_sample.py` | 采样脚本（改路径后复用） |
| `experiment2/step4b/step4b_4_compute_metrics.py` | 评估脚本（直接复用） |
| `experiment2/step1/holdout_1000_ids.txt` | Holdout 样本 ID 列表（实际约 787 个） |

---

## 操作步骤

### Step 1：确认 Dataset 使用 Holdout ID

将 `step4_3_sample.py` 复制为 `step5_1_sample.py`，修改以下两处：
1. ID 文件路径：从 `val_ids.txt` / `test_ids.txt` 改为 `holdout_1000_ids.txt`
2. 输出路径：改为 `experiment2/step5/predictions_holdout.pt`

Dataset 本身不需要改动（直接使用 Step4d 的 L=6+min-image 版本）。

### Step 2：采样

```
python step5_1_sample.py
```

使用 `best_model.ckpt`，对 Holdout 集全部样本做推断，保存预测结果。

### Step 3：评估

将 `step4b_4_compute_metrics.py` 复制为 `step5_2_compute_metrics.py`，修改以下两处：
1. 输入路径：从 `predictions_val.pt` 改为 `predictions_holdout.pt`
2. 输出路径：改为 `experiment2/step5/metrics_holdout.txt`

```
python step5_2_compute_metrics.py
```

---

## 验收标准（Holdout）

| 指标 | 预期范围 | 异常阈值 |
|------|---------|---------|
| RMSD | 1.4 - 2.0 Å | > 2.5 Å 需汇报 |
| Type Accuracy | 0.22 - 0.28 | < 0.20 需汇报 |
| pred_in_cutoff | 15 - 20 / 20 | < 10/20 需汇报 |

---

## 输出文件清单

```
experiment2/step5/
├── step5_1_sample.py           ← 修改后的采样脚本（备份）
├── step5_2_compute_metrics.py  ← 修改后的评估脚本（备份）
├── predictions_holdout.pt      ← 采样结果
└── metrics_holdout.txt         ← 最终指标报告
```

---

## 汇报模板

```
## Step5 完成报告

**执行内容**：Holdout 盲测检验

**Holdout 样本数**：（实际处理的样本数，应 ≤ 787）

**评估结果**：
  RMSD（holdout）：
  Type Accuracy（holdout）：
  pred_in_cutoff（holdout）：
  true_in_cutoff（holdout）：

**与 val/test 对比**：
  RMSD：val=1.47Å，test=1.47Å，holdout=
  Type Acc：val=0.249，holdout=

**输出文件**：
  predictions_holdout.pt ✅/❌
  metrics_holdout.txt ✅/❌

**异常/发现**：

**需要 Main Agent 2 决策的问题**：
```

---

## 关键路径常量

```python
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
STEP1_DIR     = EXP2_ROOT + r"\step1"
STEP4d_DIR    = EXP2_ROOT + r"\step4d"
STEP5_DIR     = EXP2_ROOT + r"\step5"
HOLDOUT_IDS   = STEP1_DIR + r"\holdout_1000_ids.txt"
L             = 6.0
N_NEIGHBORS   = 20
```

---

## 注意事项

1. Holdout 集从项目第一天起就被封存，模型训练期间从未接触过这些样本，这是真正的盲测。
2. 若某个 Holdout 样本的数据文件缺失或损坏，跳过并记录，不影响整体评估。
3. **化学式/mp_id 禁止入模型**，评估时仅用谱特征作为条件输入。
4. num_workers=0，不得修改。
5. 不要提前看 metrics_holdout.txt 的结果再决定是否汇报——无论结果好坏，完整汇报所有数字。

---

*Main Agent 2 撰写，2026-04-09*
