# STEP4c_HANDOFF.md
# Step4c Agent 交接文档：[-0.5, 0.5] 坐标系统一 + 重训 + 重评估

> **写给 Step4c Agent**
> **日期**：2026-04-09
> **由 Main Agent 2 撰写**

---

## 背景：为什么需要 Step4c

Step4b 已修复"Dataset 与 forward() 坐标系不一致"的 bug（两端现在都是 [0,1]），
但评估结果仍接近随机基线（RMSD=4.17 Å，pred_in_cutoff=2.89/20）。

Step4b Agent 的诊断：**[0,1] 坐标表示对本任务天然不利**。

```
本任务的真实结构：20个邻居原子聚集在 Fe 原点附近（frac ≈ 0）
% 1.0 操作的后果：frac≈0 的原子一部分留在 0 附近，一部分被折叠到 1 附近
                  → 人为制造双峰分布（frac≈0 端 + frac≈1 端）

DiffCSP 扩散先验：[0,1] 均匀分布
模型需要从"均匀分布"去噪到"双峰"→ 学习信号极弱 → 500 epoch 后仍≈随机
```

**正确做法**：保持坐标在 [-0.5, 0.5] 空间（单峰，以 Fe 原点为中心），
同时修改 `diffusion_w_type_xas.py` 去掉 `% 1.`，使两端坐标系统一。

---

## 你需要向用户索取的文件

| 文件 | 用途 |
|------|------|
| `experiment2/step3/xas_local_dataset.py`（v5） | 直接复用，**不改动** |
| `experiment2/step3/diffusion_w_type_xas.py` | 需修改 forward() 和 sample() |
| `experiment2/step4/step4_2_train.py` | 直接复用，**不改动** |
| `experiment2/step4/step4_3_sample.py` | 直接复用，**不改动** |
| `experiment2/step4b/step4b_4_compute_metrics.py` | 直接复用，**不改动**（含最小镜像，与 [-0.5,0.5] 坐标系配套） |

---

## 三项改动（精确说明）

### 改动 1：`diffusion_w_type_xas.py` — forward() 去掉 `% 1.`

**改动位置**：`forward()` 函数中，加噪步骤对 frac_coords 执行 `% 1.` 的那一行。

**改动意图**：
- 改前：`noisy_frac_coords = (frac_coords + noise) % 1.`
  → 强制映射到 [0,1]，破坏 [-0.5, 0.5] 的单峰分布
- 改后：`noisy_frac_coords = frac_coords + noise`
  → 保持坐标在 [-0.5, 0.5] 附近自然扩散，不做折叠

**注意**：仅删除 `% 1.`，其余加噪逻辑（noise scale、time step 等）不变。

### 改动 2：`diffusion_w_type_xas.py` — sample() 检查并去掉 `% 1.`

**改动位置**：`sample()` 函数中，对采样坐标执行 `% 1.` 的所有位置。

**改动意图**：采样阶段如果仍有 `% 1.` 折叠，会把预测坐标从 [-0.5, 0.5] 强制映射到 [0,1]，导致评估时坐标系再次错乱。需要全部删除。

**操作**：搜索 `sample()` 函数中所有 `% 1` 字样，逐一判断是否是对预测坐标的折叠操作，若是，则删除。

### 改动 3：Dataset 确认使用 v5（无需改动）

`xas_local_dataset.py` v5 输出的 frac_coords = `neighbor_carts / self.L`，范围 ∈ [-0.5, 0.5]，**直接使用，不做任何改动**。

---

## 开训前强制检查（必须全部通过，不可跳过）

**检查 1 — Dataset 输出范围**：
取 5 个样本，打印每个样本的 `frac_coords.min()` 和 `frac_coords.max()`。
要求：min ≥ -0.5，max ≤ 0.5，无负值超出 -0.5 或正值超出 0.5。

**检查 2 — forward() 加噪后坐标范围**：
对 5 个样本跑一次 forward()，打印 `noisy_frac_coords` 的 min/max。
要求：范围在 [-1.5, 1.5] 以内（加噪后允许轻微超出 [-0.5, 0.5]，但不应出现全部集中在 [0,1] 的迹象）。

**检查 3 — forward() loss 数值**：
对同一 batch 打印 `loss`、`loss_coord`、`loss_type`。
参考值：Step3 前向测试结果为 loss≈2.66，loss_coord≈1.27，loss_type≈1.38。
要求：数值在同一量级（不应出现 loss > 100 或 loss = NaN/Inf）。

**检查 4 — sample() 输出坐标范围**：
对 5 个样本跑一次 sample()，打印预测 frac_coords 的 min/max。
要求：坐标集中在 [-0.5, 0.5] 附近，**不应出现大量原子的 frac > 0.8 或 frac < -0.8**。
（若仍出现大量原子散布全空间，说明 sample() 中仍有未删除的 `% 1.`）

只有四项检查全部通过，才能启动训练。

---

## 训练配置（与 Step4 完全一致）

```
Dataset       = xas_local_dataset.py v5（不改动）
训练脚本      = step4_2_train.py（不改动）
batch_size    = 16
lr            = 1e-4
max_epochs    = 500
early_stop patience = 30
gradient_clip = 1.0
精度          = bf16
num_workers   = 0
L             = 12.0（虚拟晶格边长，Å）
N_NEIGHBORS   = 20
Checkpoint 输出目录 = experiment2/step4c/checkpoints/
```

---

## 采样与评估

**采样脚本**：直接使用 `experiment2/step4/step4_3_sample.py`，不改动。

采样输出保存至：
- `experiment2/step4c/predictions_val.pt`
- `experiment2/step4c/predictions_test.pt`

**评估脚本**：直接使用 `experiment2/step4b/step4b_4_compute_metrics.py`（含最小镜像修正，与 [-0.5, 0.5] 坐标系配套）。

评估结果保存至：`experiment2/step4c/metrics_report.txt`

---

## 验收标准

| 指标 | 目标 | 若未达标 |
|------|------|----------|
| RMSD（val） | < 2.0 Å | 若 > 3.5 Å，需汇报 Main Agent 2 进一步诊断 |
| pred_in_cutoff（val） | > 10/20 | 若 ≤ 5/20，说明 sample() 中仍有 `% 1.` 未删除 |
| Type Accuracy（val） | ≥ 0.27 | 不应低于 Step4b 的 0.270 |
| best val_loss | 参考 Step4b（0.6504），预期相近 | — |

---

## 输出文件清单

```
experiment2/step4c/
├── checkpoints/
│   └── best_model.ckpt
├── predictions_val.pt
├── predictions_test.pt
├── step4c_diffusion_w_type_xas.py   ← 改动后的模型文件（备份留存）
└── metrics_report.txt
```

---

## 汇报模板

```
## Step4c 完成报告

**执行内容**：[-0.5,0.5] 坐标系统一 + 重训 + 重评估

**开训前检查**：
  检查1（Dataset 输出范围）：通过 / 未通过，实测 min= ，max=
  检查2（forward 加噪后范围）：通过 / 未通过，实测 min= ，max=
  检查3（forward loss 数值）：通过 / 未通过，loss= ，loss_coord= ，loss_type=
  检查4（sample 输出范围）：通过 / 未通过，实测 min= ，max=

**训练结果**：
  best val_loss：
  best epoch：
  总训练时间：

**评估结果**：
  RMSD（val）：        RMSD（test）：
  Type Accuracy（val）：  Type Accuracy（test）：
  pred_in_cutoff（val）：  true_in_cutoff（val）：

**输出文件**：
  checkpoints/best_model.ckpt ✅/❌
  predictions_val.pt ✅/❌
  predictions_test.pt ✅/❌
  metrics_report.txt ✅/❌

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
STEP4c_DIR    = EXP2_ROOT + r"\step4c"
L             = 12.0
N_NEIGHBORS   = 20
```

---

## 注意事项

1. `diffusion_w_type_xas.py` 中可能不止一处 `% 1.`，请在文件内全局搜索 `% 1` 和 `%1`，确认所有针对预测坐标的折叠操作均已删除。若某处 `% 1.` 是用于其他目的（如晶格归一化），请汇报 Main Agent 2 再做决策，不要擅自删除。
2. **化学式/mp_id 禁止入模型**，本次改动不涉及此部分。
3. **Holdout 集禁止接触**，训练和评估只用 train/val/test。
4. num_workers=0，不得修改。

---

*Main Agent 2 撰写，2026-04-09*
