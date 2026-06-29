# STEP4d_HANDOFF.md
# Step4d Agent 交接文档：L=12→6 缩小虚拟晶格 + 重训 + 重评估

> **写给 Step4d Agent**
> **日期**：2026-04-09
> **由 Main Agent 2 撰写**

---

## 背景：为什么需要 Step4d

Step4c 的诊断实验（reverse diffusion std 曲线）已确认：

```
t=1000→640：std ≈ 0.28-0.30，噪声主导，信号被淹没
t=640→420：std 下降到 0.25-0.26 ← condition 信号真实存在，架构没问题
t=420→1：  std 回弹到 0.27，步长太小，原子无法从随机位置收敛
```

**根因**：L=12Å 时，20 个邻居原子实际只占 box 体积的约 3%。
模型在中间阶段捕捉到 condition 信号时，原子已散布在离真实位置 4-5Å 之外，
剩余低噪声步骤步长不足以完成收敛。

**修复**：L=12Å → L=6Å。
原子占据体积从 3% 提升到约 25%，中间阶段信号驱动的相对位移翻倍，收敛可行。

**已确认没有问题的部分（不需要改动）**：
- condition 注入架构（SpectrumEncoder + 拼接方式）✅
- 坐标系（[-0.5, 0.5]，无 % 1.）✅
- 训练流程、超参 ✅
- 评估脚本（含最小镜像）✅

---

## 你需要向用户索取的文件

| 文件 | 用途 |
|------|------|
| `experiment2/step3/xas_local_dataset.py`（v5） | 修改 L=12→6 |
| `experiment2/step3/diffusion_w_type_xas.py`（Step4c 版本） | 直接复用，**不改动**（已去掉 % 1.） |
| `experiment2/step4/step4_2_train.py` | 直接复用或修改 L 常量 |
| `experiment2/step4/step4_3_sample.py` | 直接复用，**不改动** |
| `experiment2/step4b/step4b_4_compute_metrics.py` | 直接复用，**不改动**（含最小镜像，[-0.5,0.5] 配套） |

---

## 唯一改动：L=12 → L=6

### 需要修改的位置（全局搜索 `12` 和 `12.0`，逐一确认）

**`xas_local_dataset.py`**：
- `self.L = 12.0` → `self.L = 6.0`
- 若有过滤条件形如 `if dist > 12` 或 `cutoff=12`，一并改为 6
- 若有形如 `frac = cart / 12.0` 的硬编码，一并改为 6.0
- 虚拟晶格定义 `diag(12, 12, 12)` → `diag(6, 6, 6)`

**`step4_2_train.py` 或 yaml config**：
- 若有 `L=12` 或 `lattice_scale=12` 的硬编码或配置项，改为 6
- 若 L 是从 dataset 读取的（`dataset.L`），则无需改动此文件

**`diffusion_w_type_xas.py`**：
- 若模型内部有 `L=12` 的硬编码（如晶格矩阵初始化），改为 6
- 若晶格是从 batch 读取的，则无需改动

**改动原则**：只改 L 的数值（12→6），不改任何其他逻辑。

---

## 开训前强制检查（必须全部通过）

**检查 1 — Dataset 输出范围**：
取 5 个样本，打印 `frac_coords.min()` 和 `frac_coords.max()`。
要求：min ≥ -0.5，max ≤ 0.5。（L=6 时，4Å 邻居的 frac = 4/6 ≈ 0.67，但取 [-0.5, 0.5] 中心化后应在该范围内。）

**检查 2 — 有效样本数变化**：
统计 L=6 后被过滤掉（第20邻居 > 6Å）的样本数量。
预期：Step1 统计 d20_99th=5.14Å，L=6 应覆盖约 97-99% 的样本，丢失样本极少。
若丢失 > 5%，需汇报 Main Agent 2 再决策。

**检查 3 — forward() loss 数值**：
打印 loss、loss_coord、loss_type。
要求：数值在合理量级（loss < 10，无 NaN/Inf）。
注意：L 缩小后 frac_coords 的分布会更集中，loss 数值可能与之前略有不同，只要无 NaN/Inf 即可继续。

**检查 4 — 晶格矩阵确认**：
打印 batch 中的晶格矩阵（或 L 值），确认为 diag(6, 6, 6)，不是旧的 diag(12, 12, 12)。

---

## 训练配置

```
Dataset       = xas_local_dataset.py（L=6 修改版）
训练脚本      = step4_2_train.py（直接复用或仅改 L 常量）
batch_size    = 16
lr            = 1e-4
max_epochs    = 500
early_stop patience = 30
gradient_clip = 1.0
精度          = bf16
num_workers   = 0
L             = 6.0（本次唯一变化）
N_NEIGHBORS   = 20
Checkpoint 输出目录 = experiment2/step4d/checkpoints/
```

---

## 采样与评估

**采样脚本**：直接使用 `step4_3_sample.py`，不改动。

采样输出：
- `experiment2/step4d/predictions_val.pt`
- `experiment2/step4d/predictions_test.pt`

**评估脚本**：直接使用 `step4b_4_compute_metrics.py`（含最小镜像，[-0.5,0.5] 坐标系配套）。

**注意**：评估时的 eval_cutoff（`min(第20邻居实际距离, 4.0Å)`）不变，L 缩小不影响评估截断逻辑。

评估结果：`experiment2/step4d/metrics_report.txt`

---

## 验收标准

| 指标 | 目标 | 说明 |
|------|------|------|
| RMSD（val） | < 2.0 Å | 若仍 > 3.5 Å 需汇报 |
| pred_in_cutoff（val） | > 10/20 | L=6 后此指标应显著提升 |
| Type Accuracy（val） | ≥ 0.27 | 不低于 Step4c |
| 丢失样本率 | < 5% | 开训前检查 2 确认 |

---

## 输出文件清单

```
experiment2/step4d/
├── checkpoints/
│   └── best_model.ckpt
├── predictions_val.pt
├── predictions_test.pt
├── xas_local_dataset_L6.py     ← 改动后的 dataset 文件（备份留存）
└── metrics_report.txt
```

---

## 汇报模板

```
## Step4d 完成报告

**执行内容**：L=12→6 缩小虚拟晶格 + 重训 + 重评估

**开训前检查**：
  检查1（frac_coords 范围）：通过/未通过，min= ，max=
  检查2（有效样本丢失率）：丢失样本数= ，丢失率= %
  检查3（forward loss）：loss= ，loss_coord= ，loss_type=
  检查4（晶格矩阵）：diag(6,6,6) 确认 ✅/❌

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
STEP4d_DIR    = EXP2_ROOT + r"\step4d"
L             = 6.0    # ← 本次唯一变化
N_NEIGHBORS   = 20
```

---

## 注意事项

1. L=6 后，第20邻居距离 > 6Å 的样本在 Dataset 中会被排除（返回 None 或跳过）。训练脚本的 collate_fn 需要能处理 None 样本（Step4 原版应已处理，若报错请汇报）。
2. **化学式/mp_id 禁止入模型**，本次改动不涉及。
3. **Holdout 集禁止接触**。
4. num_workers=0，不得修改。
5. 若 early stop 在 epoch 200 以前触发（val_loss 不再下降），请立即汇报，不要等训练完成。

---

*Main Agent 2 撰写，2026-04-09*
