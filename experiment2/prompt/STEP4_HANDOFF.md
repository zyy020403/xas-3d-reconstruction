# STEP4_HANDOFF.md
# Step4 Agent 交接文档：训练、采样与评估

> **你的角色**：Step4 Agent
> **你的任务**：
>   1. 训练前健康检查（5 样本采样，晶格固定验证）
>   2. 正式训练，监控收敛
>   3. val/test 集采样 + 全套评估指标
>   4. 子群分析报告
> **前置文档**：先读 SHARED_00_v2.md
> **输出目录**：`C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step4\`

---

## 前序步骤关键参数（以此为准）

| 参数 | 值 |
|------|----|
| 虚拟晶格边长 L | 12 Å |
| 局部结构原子数 | 20 个邻居 |
| frac_coords 过滤 | abs > 0.5 的样本已在 Dataset 层返回 None 并过滤 |
| xmu.dat 列索引 | E = data[:,0]，μ = data[:,3] |
| feff_features 路径 | `C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv` |
| E0 来源 | data_inventory.csv 的 E0 列（不从 feff_features 查） |
| Step3 前向测试结果 | loss=2.66，loss_coord=1.27，loss_type=1.38，keep_lattice=True ✅ |

---

## 你需要的文件（向用户索取）

```
共享文档：
  SHARED_00_v2.md

Step3 输出（需要访问路径）：
  experiment2/step3/xas_local_dataset.py       ← v5（含 frac_coords 过滤）
  experiment2/step3/xas_local_datamodule.py
  experiment2/step3/diffusion_w_type_xas.py
  experiment2/step3/conf_xas/model/diffusion_xas.yaml
  experiment2/step3/conf_xas/data/xas_fe_local.yaml

Step1 输出（需要访问路径）：
  experiment2/step1/train_ids.txt
  experiment2/step1/val_ids.txt
  experiment2/step1/test_ids.txt
  experiment2/step1/data_inventory.csv
  experiment2/step1/feff_feature_scaler.pkl
  experiment2/step1/feff_feature_stats.csv

Step2 输出（需要访问路径）：
  experiment2/step2/spectrum_preprocessor.py
  experiment2/step2/spectrum_encoder.py

参考脚本（Exp1 经验）：
  experiment/step3/step3.3_train.py     ← 训练脚本结构参考
  experiment/step4/step4.2_compute_metrics.py ← 评估子群框架参考
```

---

## 工作内容

### Step 4.1：训练前健康检查

**文件名**：`step4_1_health_check.py`

**必须在任何训练开始前执行此检查。这是 Exp1 265 epoch 白跑的教训。**

```python
# 步骤：
# 1. 加载模型（随机初始化，不需要 checkpoint）
# 2. 从 val dataloader 取 5 个样本
# 3. 调用 model.sample(batch)
# 4. 检查采样结果

traj, _ = model.sample(mini_batch)
pred_lattices = traj['lattices']   # (5, 3, 3)

# 由于 cost_lattice=0，晶格应精确固定为 diag(12,12,12)
pred_lengths = torch.stack([
    torch.tensor([m[0,0].abs(), m[1,1].abs(), m[2,2].abs()])
    for m in pred_lattices])

print(f"pred_lengths（应全部约为 12.0）：{pred_lengths}")

# ✅ 通过：所有值在 [11, 13] 范围内
# ❌ 失败：任何值 > 30 或 < 1 → 停止，检查 cost_lattice 配置

# 同时检查采样出的 frac_coords 范围
pred_frac = traj['frac_coords']    # (sum_of_atoms, 3)
print(f"pred_frac 范围：[{pred_frac.min():.3f}, {pred_frac.max():.3f}]（期望 ≈ [-0.5, 0.5]）")
```

汇报此步骤结果后，Main Agent 确认才开始训练。

---

### Step 4.2：正式训练

**文件名**：`step4_2_train.py`

参考 `experiment/step3/step3.3_train.py`，关键配置：

```python
# 训练超参
MAX_EPOCHS        = 500
BATCH_SIZE        = 16
LR                = 1e-4
GRADIENT_CLIP     = 1.0
PRECISION         = 'bf16-mixed'    # A4000 支持
EARLY_STOP_PAT    = 50              # patience=50 epoch 无改善则停止
NUM_WORKERS       = 0               # Windows

# checkpoint 配置
checkpoint_callback = ModelCheckpoint(
    dirpath    = r'C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step4\checkpoints',
    filename   = 'epoch={epoch}-val_loss={val_loss:.4f}',
    monitor    = 'val_loss',
    save_top_k = 3,
    mode       = 'min',
)

# 学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
```

**Windows 环境必须的设置**（在所有 import 之前）：
```python
import os
os.environ['PROJECT_ROOT'] = r'C:\Users\T-Cat\Desktop\DiffCSP-main'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# num_workers=0 防止多进程问题
# 不要在 Windows 上用 spawn multiprocessing
```

**训练监控**：每 10 epoch 打印一次：
```
train_loss / val_loss / loss_coord / loss_type / loss_lattice（仅监控）
```

**收敛判断**：
- val_loss 稳定下降 → 继续
- val_loss 在 loss ≈ 1.0 附近震荡 50 epoch → early stop 触发
- val_loss 始终不下降（> 2.5，50 epoch 后）→ 停止，汇报 Main Agent

---

### Step 4.3：val/test 集采样

**文件名**：`step4_3_sample.py`

使用最优 checkpoint（val_loss 最低）对 val 集和 test 集采样：

```python
# 每个样本生成 1 个预测结构（可后续改为多个取最优）
# 保存格式：
predictions = {
    'mp_id':          [...],    # 样本 ID
    'pred_frac_coords': [...],  # list of (20, 3) tensors
    'pred_atom_types':  [...],  # list of (20,) tensors
    'true_frac_coords': [...],  # Ground truth
    'true_atom_types':  [...],
    'eval_cutoff':      [...],  # 每个样本的动态评估截断距离
}
torch.save(predictions, 'experiment2/step4/predictions_val.pt')
torch.save(predictions, 'experiment2/step4/predictions_test.pt')
```

---

### Step 4.4：评估指标计算

**文件名**：`step4_4_compute_metrics.py`

参考 `experiment/step4/step4.2_compute_metrics.py` 的子群框架，但评估逻辑完全不同（不用 StructureMatcher）。

#### 核心评估函数

```python
def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=12.0):
    """
    对单个样本计算评估指标

    eval_cutoff = min(第20邻居实际距离, 4.0)，来自 batch.eval_cutoff

    步骤：
    1. 将 frac_coords 转回笛卡尔坐标（乘以 L=12）
    2. 按 eval_cutoff 过滤：只保留距原点 <= eval_cutoff 的原子
       - 对 true 结构过滤 → true_subset
       - 对 pred 结构过滤 → pred_subset（按同样截断）
    3. 在 true_subset 和 pred_subset 之间做匈牙利匹配（最小化总距离）
    4. 计算匹配后的指标
    """
    # 转回笛卡尔
    pred_cart = pred_frac * L   # (20, 3)
    true_cart = true_frac * L   # (20, 3)

    # 按 eval_cutoff 过滤
    pred_dists = np.linalg.norm(pred_cart, axis=1)
    true_dists = np.linalg.norm(true_cart, axis=1)
    pred_mask  = pred_dists <= eval_cutoff
    true_mask  = true_dists <= eval_cutoff

    pred_sub_cart  = pred_cart[pred_mask]
    pred_sub_types = pred_types[pred_mask]
    true_sub_cart  = true_cart[true_mask]
    true_sub_types = true_types[true_mask]

    n_pred = len(pred_sub_cart)
    n_true = len(true_sub_cart)

    if n_true == 0:
        return None   # 不应发生，跳过

    # 匈牙利匹配（最小化空间距离，不考虑类型）
    from scipy.optimize import linear_sum_assignment
    n = max(n_pred, n_true)
    # padding 到相同大小（用大距离填充）
    cost_matrix = np.full((n, n), 999.0)
    if n_pred > 0 and n_true > 0:
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(pred_sub_cart, true_sub_cart)
        cost_matrix[:n_pred, :n_true] = dist_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 只考虑有效匹配（n_pred 和 n_true 的较小值以内的匹配）
    valid = (row_ind < n_pred) & (col_ind < n_true)
    row_ind, col_ind = row_ind[valid], col_ind[valid]

    # RMSD（Å）
    matched_pred = pred_sub_cart[row_ind]
    matched_true = true_sub_cart[col_ind]
    rmsd = np.sqrt(((matched_pred - matched_true) ** 2).sum(axis=1).mean())

    # Type Accuracy（匹配后的类型是否一致）
    matched_pred_types = pred_sub_types[row_ind]
    matched_true_types = true_sub_types[col_ind]
    type_acc = (matched_pred_types == matched_true_types).mean()

    # 原子数误差
    n_atoms_error = abs(n_pred - n_true)

    return {
        'rmsd':           rmsd,
        'type_acc':       type_acc,
        'n_true':         n_true,
        'n_pred':         n_pred,
        'n_atoms_error':  n_atoms_error,
        'eval_cutoff':    eval_cutoff,
    }
```

#### 键长违规率

```python
def bond_length_violation_rate(pred_cart, pred_types, bond_constraints):
    """
    计算预测结构中违反物理键长约束的键的比例
    bond_constraints 来自 all_center_neighbors_summary.csv (B列=元素对, F列=min/max)
    """
    # 对所有原子对计算距离
    # 查约束表，检查是否在 [min, max] 范围内
    # 返回违规比例
```

#### 子群分析（复用 Exp1 框架）

```python
# 按以下维度分组，分别报告 mean RMSD 和 mean Type Accuracy：
# 1. eval_cutoff 分组：< 3.0 Å / 3.0-4.0 Å
# 2. n_true 分组：≤ 8（第一壳层）/ 9-14 / 15-20
# 3. 化合物类型（如果 data_inventory 有此信息）
```

#### 汇总指标报告格式

```
=== Test Set Metrics ===
N_samples: 1,627
RMSD (Å):            mean=X.XX,  median=X.XX,  std=X.XX
Type Accuracy:       mean=X.XX,  median=X.XX
Bond Violation Rate: X.XX%
N_atoms_error (MAE): X.XX

=== Subgroup: eval_cutoff < 3.0 Å (N=XXX) ===
RMSD: X.XX,  Type Acc: X.XX

=== Subgroup: eval_cutoff 3.0-4.0 Å (N=XXX) ===
RMSD: X.XX,  Type Acc: X.XX
```

---

## 输出文件清单

```
experiment2/step4/
├── step4_1_health_check.py
├── step4_2_train.py
├── step4_3_sample.py
├── step4_4_compute_metrics.py
├── checkpoints/
│   ├── epoch=XXX-val_loss=X.XXXX.ckpt   ← 最优 checkpoint ★
│   └── ...
├── predictions_val.pt   ★
├── predictions_test.pt  ★
└── metrics_report.txt   ★
```

---

## 注意事项

1. **Step4.1 健康检查结果汇报给 Main Agent 后才开始训练**，不要自行跳过

2. **训练日志保存**：每 epoch 的 train_loss 和 val_loss 都要记录，方便后续分析收敛曲线

3. **Exp1 的历史教训**：
   - val_loss 在 1.0 附近震荡 ≠ 没学到东西，可能是任务本身难度
   - 真正判断是否有效要看 Step4.4 的 RMSD：如果 RMSD < 1.5 Å 就说明模型在学习结构信息
   - RMSD ≈ 随机基线（约 3-4 Å，L=12 的均匀分布期望距离）→ 模型没有学到东西

4. **评估用匈牙利匹配**（scipy.optimize.linear_sum_assignment）不是 StructureMatcher，不需要周期性边界处理，直接用笛卡尔坐标距离

5. **eval_cutoff 来自 batch.eval_cutoff**，已在 Dataset 的 __getitem__ 中计算并存入，是每个样本在数据准备时确定的静态值

6. **holdout_ids.txt 中的样本此步骤禁止使用**，Step5 才用

---

## 完成后向 Main Agent 汇报

重点汇报：
- Step4.1 健康检查：pred_lengths 和 pred_frac 范围（等 Main Agent 确认再开训）
- 训练曲线：最终 val_loss，收敛 epoch 数，是否触发 early stop
- Step4.4 核心指标：test 集 mean RMSD，mean Type Accuracy，Bond Violation Rate
- 子群分析中哪个子群表现最好/最差
- 最优 checkpoint 路径
