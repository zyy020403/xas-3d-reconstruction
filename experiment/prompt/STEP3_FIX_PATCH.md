# STEP3_FIX_PATCH.md
# Step 3 训练修复指令（Patch）

> **性质**: 这不是新的 Step，是对 Step 3 训练配置的精确修复  
> **优先级**: 最高，修复完成后重新走 Step 4.1 → 4.2  
> **预计改动量**: 极小，只改训练脚本 + 修复一个 Dataset bug

---

## 诊断结论（来自 Step 4 Agent）

- **根本原因**: `lr=1e-4` 固定，无 scheduler。epoch 89 后 loss 完全震荡（0.95~1.05），从未收敛
- **坐标误差** 0.2548 ≈ 随机基线 0.25，模型未学到任何结构信息
- **次要 bug**: 数据加载中偶发两个样本的 XAS embedding 完全相同（cosine=1.0），需排查

---

## ⚡ 开始工作前——必须向用户索取的文件

```
请提供以下文件（你将直接修改它们）：

1. experiment/step3/step3.3_train.py         ← 主要修改对象
2. experiment/step3/xas_dataset.py           ← 修复 embedding 重复 bug
3. experiment/step3/xas_datamodule.py        ← 确认 batch 构建逻辑
4. diffcsp/pl_modules/diffusion.py           ← 确认 configure_optimizers 当前内容
```

---

## 修复 1：训练配置（step3.3_train.py）

**目标**：在 `step3.3_train.py` 中做以下 4 处改动，其余代码不动。

### 1a. Resume from checkpoint

找到 `trainer.fit(...)` 那一行，改为：

```python
# 修改前
trainer.fit(model, datamodule=datamodule)

# 修改后
RESUME_CKPT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step3\training_output\epoch=189-val_loss=0.9522.ckpt"
trainer.fit(model, datamodule=datamodule, ckpt_path=RESUME_CKPT)
```

### 1b. 初始学习率降为 1e-5

找到 optimizer 的学习率设置，改为：

```python
# 修改前（原始值，可能在 diffusion.py 或 yaml 中）
lr = 1e-4

# 修改后
lr = 1e-5
```

**注意**：`ckpt_path` resume 会恢复模型权重和 epoch 计数，但**不会**恢复 optimizer 状态里的 lr（PyTorch Lightning 的行为）。所以直接在代码/config 里把 lr 改为 1e-5 即可，resume 后的第一个 epoch 就会用新 lr。

### 1c. 加入 CosineAnnealingLR

在 `CSPDiffusion` 的 `configure_optimizers` 方法里加入 scheduler。找到该方法（在 `diffcsp/pl_modules/diffusion.py` 中），在返回 optimizer 之前加入：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,       # 100 个 epoch 完成一个余弦周期
    eta_min=1e-6     # 最小 lr
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

**修改前先备份**：
```python
import shutil
shutil.copy(r"...\diffcsp\pl_modules\diffusion.py",
            r"...\experiment\step3\diffusion_backup_before_fix.py")
```

### 1d. 加入 EarlyStopping 和调整 max_epochs

在 Trainer 的 callbacks 中加入 `EarlyStopping`，并把 `max_epochs` 改为足够大的值：

```python
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# max_epochs 改为 400（epoch 从 189 继续，给足空间）
trainer = pl.Trainer(
    max_epochs=400,
    gradient_clip_val=1.0,
    precision="bf16",
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
            patience=30,
            mode="min",
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ],
)
```

**注意输出目录改为 `experiment/step4/finetune_output/`**，不要覆盖 Step 3 的输出。

---

## 修复 2：embedding 重复 bug（xas_dataset.py）

Step 4 诊断发现部分样本对的 XAS embedding 完全相同（cosine=1.0），说明数据加载层面有 bug。

**排查目标**：找到 `XASCrystalDataset.__init__` 中预加载逻辑，检查是否存在以下情况：

```python
# 常见 bug：dict 浅拷贝导致多个 key 指向同一个 tensor 对象
for mp_id in mp_ids:
    self.data[mp_id] = item  # 若 item 是同一个对象的引用，就会出现 cosine=1.0
```

**修复方向**：
1. 在 `__getitem__` 末尾加一行断言，验证返回的 spectra tensor 地址与上一次不同
2. 若发现是浅拷贝问题，在存储时改为 `.clone()`：
   ```python
   self.data[mp_id]['spectra'] = spectra_tensor.clone()
   ```
3. 若是 `preprocess_chi` 返回的是同一个全零 tensor（多个无效谱都返回同一个对象）：
   ```python
   # preprocess_chi 中，确保每次返回新的 tensor
   return torch.zeros(1, self.k_grid_points, dtype=torch.float32)  # 每次 new
   # 不要用全局缓存的 zero_tensor
   ```

**验证方法**：修复后在脚本末尾加验证块：
```python
if __name__ == "__main__":
    # 加载前 20 个样本，检查谱数据唯一性
    from torch.nn.functional import cosine_similarity
    sample_specs = [dataset[i]['spectra'][0] for i in range(min(20, len(dataset)))]
    for i in range(len(sample_specs)):
        for j in range(i+1, len(sample_specs)):
            sim = cosine_similarity(sample_specs[i].flatten().unsqueeze(0),
                                    sample_specs[j].flatten().unsqueeze(0)).item()
            if sim > 0.999:
                print(f"WARNING: sample {i} and {j} cosine={sim:.4f} — 疑似重复！")
    print("唯一性检查完成")
```

---

## 修复完成后的验证步骤

1. 启动修复后的训练脚本，确认以下输出出现：
   ```
   Epoch 189, resuming...
   lr = 1e-5           ← 确认新 lr 生效
   val_loss = 0.XXXX   ← 应比 0.9522 更低，或至少不震荡
   ```

2. 跑 **10 个 epoch**，检查：
   - loss 是否继续单调下降（而不是震荡）
   - lr 是否在按 cosine 曲线下降（`LearningRateMonitor` 会打印）

3. 若 10 个 epoch 后 val_loss < 0.94（低于 Step 3 最优），说明修复有效，继续跑完

4. 若 val_loss 仍然震荡或上升，停止，把以下信息汇报给 Main Agent：
   - 前 10 个 epoch 的 train_loss 和 val_loss 序列
   - lr 在这 10 个 epoch 的实际值
   - GPU 利用率（排除计算瓶颈）

---

## 完成后提交的总结报告（额外内容）

```markdown
### 修复验证
- embedding 重复 bug 根因: （描述找到了什么）
- 修复后唯一性检查结果: PASS / FAIL
- 修复后前 10 epoch 的 val_loss 序列: [...]
- lr 实际值序列（确认 cosine 生效）: [...]

### 最终 fine-tune 结果
- 最优 val_loss: X.XXXX（epoch XXX）
- 相比 Step 3 基准（0.9522）改善: X.XXXX
- EarlyStopping 触发: Y/N（若触发，在第几个 epoch）
- 最优 checkpoint 路径: experiment/step4/finetune_output/...

### 给 Main Agent 的信号
- 可以重新进行 Step 4.1 → 4.2 评估: Y/N
```

---

## 注意事项

1. **`diffusion.py` 改完备份**，命名 `diffusion_backup_before_fix.py`，与已有的 `diffusion_backup.py`（Step 3 原始备份）区分开
2. **输出目录用 `experiment/step4/finetune_output/`**，不要写入 `experiment/step3/`
3. **不要改 batch_size、网络结构、latent_dim**，本次只动 lr 和 scheduler
4. **严禁加载 holdout_1000_ids.txt 中的任何 mp_id**
