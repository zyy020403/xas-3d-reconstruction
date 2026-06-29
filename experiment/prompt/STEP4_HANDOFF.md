# STEP4_HANDOFF.md
# Step 4 Agent 交接文档：评估指标计算与微调

> **任务编号**: Step 4（共 3 个子步骤：4.1 → 4.2 → 4.3）  
> **前置条件**: Step 1、2、3 均已完成  
> **核心原则**: 最大化复用 DiffCSP 原有评估脚本，只在必要处扩展  
> **最优 Checkpoint**: `experiment/step3/training_output/epoch=189-val_loss=0.9522.ckpt`  
> **输出目录**: `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step4\`  
> **完成标志**: val/test 集上的结构评估指标计算完毕，且完成一轮带 scheduler 的 fine-tune

---

## ⚡ 开始工作前——必须向用户索取的文件

```
请提供以下文件（只需阅读，不修改）：

1. scripts/compute_metrics.py    ← DiffCSP 原有指标计算脚本，直接复用
2. scripts/evaluate.py           ← DiffCSP 原有评估脚本
3. scripts/eval_utils.py         ← 评估工具函数
4. scripts/sample.py             ← 采样脚本（生成预测结构用）
```

**读完后，重点记录**：
- `compute_metrics.py` 计算哪些指标（match_rate、rmse、等）
- `sample.py` 的输入格式：它期望什么格式的 checkpoint 和 batch
- 这决定了 Step 4.1 的采样脚本需要做多少适配

---

## 背景：Step 4 的目标

Step 3 已验证训练可收敛（val_loss=0.9522）。Step 4 做两件事：

1. **4.1 + 4.2**：用当前最优 checkpoint 在 val/test 集上做结构生成和评估，得到基准指标
2. **4.3**：加入 lr scheduler 从当前 checkpoint 继续 fine-tune，看是否能进一步提升

**不做的事**：不做大范围超参数搜索，不改网络结构，保留 holdout 集严格不碰。

---

## Step 4.1：生成预测结构（采样）

### 脚本名
```
experiment/step4/step4.1_sample_predictions.py
```

### 任务描述

使用训练好的模型，对 val 集和 test 集中的每个化合物进行结构预测（扩散采样），生成预测的晶体结构。

**核心策略**：读完 `scripts/sample.py` 后，尽量复用其采样逻辑，只做 DataModule 的替换（从原 `CrystDataModule` 换为 `XASDataModule`）和 checkpoint 加载适配。

**每个化合物生成 1 个预测结构**（节省时间，Step 5 盲测时再做 multi-sample）。

**输出格式**：
```python
# 对每个 mp_id，输出一个字典，保存为 predictions_val.pt / predictions_test.pt
{
    mp_id: {
        'pred_frac_coords': Tensor [N_atoms, 3],
        'pred_lengths':     Tensor [3],
        'pred_angles':      Tensor [3],
        'pred_atom_types':  Tensor [N_atoms],
        'gt_frac_coords':   Tensor [N_atoms, 3],  # ground truth
        'gt_lengths':       Tensor [3],
        'gt_angles':        Tensor [3],
        'gt_atom_types':    Tensor [N_atoms],
        'n_atoms':          int,
    }
}
```

保存至：
```
experiment/step4/predictions_val.pt
experiment/step4/predictions_test.pt
```

**注意**：采样时保持 `model.eval()` 和 `torch.no_grad()`，以及 `precision="bf16"` 与训练时一致。

---

## Step 4.2：计算评估指标

### 脚本名
```
experiment/step4/step4.2_compute_metrics.py
```

### 任务描述

读取 Step 4.1 的预测结果，计算以下指标。**优先复用 `scripts/compute_metrics.py` 和 `eval_utils.py` 中的函数**，不要重写结构匹配逻辑。

#### 必须计算的指标

**1. Match Rate（结构匹配率）**
- 使用 pymatgen 的 `StructureMatcher` 判断预测结构与 ground truth 是否匹配
- 阈值：`ltol=0.3`（晶格参数容差），`stol=0.5`（位置容差），`angle_tol=10`（角度容差）
- Match rate = 匹配成功数 / 总数

**2. RMSE（均方根误差）**
- 仅对匹配成功的结构计算
- 对齐原子后计算分数坐标的 RMSE

**3. 键长违规率（Bond Length Violation Rate）**
- 读取 `experiment/step1/bond_length_constraints.json`
- 对每个预测结构，检查所有原子对的键长是否在约束范围内
- 报告违规对数 / 总键对数

#### 按子群分析（非常重要）

除了总体指标，还需要按以下维度分别报告：

| 分析维度 | 说明 |
|----------|------|
| `is_ionic` 占比 | 纯 ionic 化合物 vs 混合/纯共价化合物的 match rate 分别是多少 |
| `n_sites` 分组 | 1 个位点 / 2-3 个位点 / 4+ 个位点各自的 match rate |
| `quality_tier` | A 级、B 级、C 级样本各自的 match rate（验证降权策略效果） |
| 元素种类 | 含 Fe 且仅含 Fe（无 ionic 元素）的 match rate，以及含混合元素的 match rate |

**输出格式**（保存至 `experiment/step4/metrics_report.txt`）：

```
=== Step 4 评估指标报告 ===

--- Val 集 ---
总样本数: XXX
Match Rate: XX.X%
Mean RMSE (matched only): X.XXXX Å
Bond Length Violation Rate: X.X%

按 is_ionic 分析:
  纯共价化合物 (is_ionic=False only): Match Rate = XX.X% (N=XXX)
  含 ionic 位点化合物: Match Rate = XX.X% (N=XXX)

按位点数分析:
  1 个位点: XX.X% (N=XXX)
  2-3 个位点: XX.X% (N=XXX)
  4+ 个位点: XX.X% (N=XXX)

按质量分级分析（样本质量分布）:
  全 A 级样本: Match Rate = XX.X% (N=XXX)
  含 B 级样本: Match Rate = XX.X% (N=XXX)
  含 C 级样本: Match Rate = XX.X% (N=XXX)

--- Test 集 ---
（同上格式）
```

---

## Step 4.3：Fine-tune（加入 lr scheduler）

### 脚本名
```
experiment/step4/step4.3_finetune.py
```

### 任务描述

从 Step 3 的最优 checkpoint 继续训练，加入学习率调度器，目标是突破 val_loss 平台期（当前 ~0.952）。

**基于 Step 3 的 `step3.3_train.py` 修改**，改动尽量小：

#### 改动 1：加入 CosineAnnealingLR

在 `CSPDiffusion` 的 `configure_optimizers` 方法中，读完原代码后，在原有 optimizer 基础上添加：

```python
# 在 configure_optimizers 返回前添加 scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,          # 100 个 epoch 完成一个余弦周期
    eta_min=1e-6        # 最小学习率
)
return {
    "optimizer": optimizer,
    "lr_scheduler": {
        "scheduler": scheduler,
        "monitor": "val_loss",
        "interval": "epoch",
        "frequency": 1,
    }
}
```

#### 改动 2：从 checkpoint 加载（resume）

```python
trainer.fit(
    model,
    datamodule=datamodule,
    ckpt_path=r"C:\...\experiment\step3\training_output\epoch=189-val_loss=0.9522.ckpt"
)
```

#### 改动 3：训练配置调整

```python
trainer = pl.Trainer(
    max_epochs=300,              # 从 epoch 189 继续到 300
    gradient_clip_val=1.0,
    precision="bf16",            # 开启 bf16（Step 3 已验证无 NaN）
    devices=1,
    accelerator='gpu',
    callbacks=[
        ModelCheckpoint(
            dirpath=r"...\experiment\step4\finetune_output",
            filename="epoch{epoch:03d}-val{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=30,          # 30 个 epoch 不改善则停止
            mode="min",
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ],
)
```

**注意**：`configure_optimizers` 是 `CSPDiffusion` 类的方法，修改它需要再次编辑 `diffcsp/pl_modules/diffusion.py`。修改前先**确认 Step 3 的备份 `diffusion_backup.py` 仍然存在**，并在修改前再做一次备份（`diffusion_backup_step3.py`）。

---

## 完成后提交的总结报告（额外内容）

```markdown
### 基准指标（最优 checkpoint epoch=189）
- Val Match Rate: XX.X%
- Val RMSE (matched): X.XXXX Å
- Val Bond Length Violation Rate: X.X%
- Test Match Rate: XX.X%
- Test RMSE (matched): X.XXXX Å

### 子群分析亮点
- （复制 metrics_report.txt 的内容）

### Fine-tune 结果
- Fine-tune 最优 val_loss: X.XXXX（epoch XXX）
- 相比 Step 3 基准改善: X.XXXX
- 是否触发 EarlyStopping: Y/N

### 需要 Main Agent 决策的问题
- （如果 ionic 样本 match rate 明显低于共价样本，说明 is_ionic embedding 的权重不够，可以考虑在 Step 5 前加大 ionic 样本的采样比例）
- （其他发现）
```

---

## 注意事项

1. **严禁使用 holdout_1000_ids.txt 中的任何 mp_id**，Step 4.1 采样只用 val 和 test 集
2. **`configure_optimizers` 修改前备份 `diffusion.py`**
3. **fine-tune 的 batch_size 保持与 Step 3 一致（=16）**，改 batch_size 会改变 loss scale，影响 scheduler 行为
4. 若 fine-tune 后 val_loss 反而上升（过拟合），立即停止，以 Step 3 的 checkpoint 为最终模型
5. `bond_length_constraints.json` 的键是 `"Fe-O"` 格式，计算违规时注意元素对的顺序需要做双向匹配（`"Fe-O"` 和 `"O-Fe"` 是同一对）
