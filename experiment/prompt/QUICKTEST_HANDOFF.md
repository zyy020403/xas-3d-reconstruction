# QUICKTEST_HANDOFF.md
# QuickTest Agent 交接文档：端到端快速验证

> **性质**: 这不是正式服的新 Step，是用极小数据集验证整条 pipeline 能跑通  
> **目标**: 100 个化合物，跑完训练 + 采样 + 指标计算，验证无崩溃、loss 能下降  
> **输出目录**: `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\`  
> **命名规则**: 所有脚本以 `step{X}.{Y}_qt_` 开头，对应正式服脚本编号

---

## QuickTest 的简化假设（与正式服的区别）

| 项目 | 正式服 | QuickTest |
|------|--------|-----------|
| 化合物数量 | ~9000 | **100 个** |
| 位点策略 | 所有位点（多位点聚合） | **只取每个化合物的第一个 Fe 位点** |
| 离子谱 | 包含（is_ionic=True） | **排除，只用 is_ionic=False** |
| k 权重 | k²χ(k)（加权到 2） | **k¹χ(k)（加权到 1）** |
| 多位点聚合器 | MultiSiteAggregator | **跳过，直接用单谱 embedding 作为 struct_emb** |
| 数据集划分 | train 80 / val 10 / test 10 | **train 70 / val 15 / test 15（约数）** |
| 保留集 | 1000 个严格隔离 | **不设保留集** |
| 训练 epoch | 200+ | **30 epoch，只看 loss 能否下降** |

---

## ⚡ 开始工作前——必须向用户索取的文件

你需要以下正式服脚本作为参考，**每写一个 qt 脚本前向用户索要对应的正式服脚本**：

```
第一批（写 step1.1_qt 前要）：
  experiment/step1/step1.1_scan_and_inventory.py

第二批（写 step1.4_qt 前要）：
  experiment/step1/step1.4_split_dataset.py

第三批（写 step2.1_qt 前要）：
  experiment/step2/step2_1_spectrum_encoder.py

第四批（写 step3 qt 前要）：
  experiment/step3/xas_dataset.py
  experiment/step3/xas_datamodule.py
  experiment/step3/step3.3_train.py
  diffcsp/pl_modules/diffusion.py
```

**不需要**重新索取 step1.2、step1.3（质量分级），QuickTest 直接用 `data_inventory.csv` 中已有的数据，不做额外筛查。

---

## 脚本清单与任务说明

### step1.1_qt_select_100.py
**对应正式服**: `step1.1_scan_and_inventory.py`  
**输入**: `experiment/step1/data_inventory.csv`（直接复用，不重新扫描）  
**任务**:
1. 读取已有的 `data_inventory.csv`
2. 过滤条件：`is_ionic=False` AND `files_complete=True` AND `element='Fe'`
3. 按 `mp_id` 分组，每个 mp_id 只保留 `site_id` 最小的那一行（即第一个 Fe 位点）
4. 从结果中随机抽取 **100 个 mp_id**（`random.seed(42)`）
5. 输出 `experiment/quicktest/qt_inventory.csv`，只含这 100 行

**输出字段**（只保留必要列）：
```
folder_name, mp_id, element, site_id, is_ionic, source_path, quality_tier
```

---

### step1.4_qt_split.py
**对应正式服**: `step1.4_split_dataset.py`  
**输入**: `experiment/quicktest/qt_inventory.csv`  
**任务**:
1. 从 100 个 mp_id 中按 70 / 15 / 15 划分（`random.seed(42)`）
2. 不做分层采样，直接随机划分
3. 输出三个文件：
   - `experiment/quicktest/qt_train_ids.txt`（约 70 个）
   - `experiment/quicktest/qt_val_ids.txt`（约 15 个）
   - `experiment/quicktest/qt_test_ids.txt`（约 15 个）

---

### step2.1_qt_encoder.py
**对应正式服**: `step2_1_spectrum_encoder.py`  
**输入**: 正式服的 `step2_1_spectrum_encoder.py`  
**任务**:  
复制正式服的 `SpectrumEncoder` 类，**只修改 `preprocess_chi` 函数中的 k 权重**：

```python
# 正式服（k²χ(k)）
k2chi = k_uniform ** 2 * chi_interp

# QuickTest 改为（k¹χ(k)）
k2chi = k_uniform ** 1 * chi_interp   # 只改这一行，变量名 k2chi 保持不变（避免牵连其他代码）
```

其余代码（归一化、边界情况处理、SpectrumEncoder 类结构）完全不动。

输出文件：`experiment/quicktest/step2.1_qt_encoder.py`

---

### step3_qt_dataset.py
**对应正式服**: `xas_dataset.py`  
**输入**: `experiment/quicktest/qt_inventory.csv`  
**任务**:  
基于正式服的 `XASCrystalDataset`，做以下简化：

1. **每个样本只有 1 个位点**（因为 qt_inventory 已经是每个 mp_id 一行），所以：
   - `n_sites` 永远为 1
   - `spectra` shape：`[1, 1, 512]`
   - `site_elements` shape：`[1]`，值永远是 Fe 的原子序数（26）
   - `is_ionic` shape：`[1]`，值永远是 0
   - `quality_weights` shape：`[1]`

2. **不需要 padding**：因为所有样本 n_sites=1，collate_fn 不需要 mask 逻辑，直接 stack 即可

3. 错误处理保持不变（POSCAR 读取失败则跳过）

输出文件：`experiment/quicktest/step3_qt_dataset.py`

---

### step3_qt_datamodule.py
**对应正式服**: `xas_datamodule.py`  
**任务**:  
基于正式服的 `XASDataModule`，改动：
- `setup` 中读取 `qt_train_ids.txt`、`qt_val_ids.txt`、`qt_test_ids.txt`
- 使用 `step3_qt_dataset.py` 中的 `QTCrystalDataset` 类（而不是正式服的 `XASCrystalDataset`）
- `batch_size=8`（数据只有 100 个，小 batch 就够）
- `num_workers=0`（Windows）

输出文件：`experiment/quicktest/step3_qt_datamodule.py`

---

### step3_qt_train.py
**对应正式服**: `step3.3_train.py` 和 `diffusion.py`  
**任务**:  
这是改动最关键的脚本。基于正式服训练脚本，做以下修改：

**A. 跳过 MultiSiteAggregator，直接用单谱 embedding 作为 struct_emb**

在 `diffusion.py` 的 `_encode_xas` 方法（或 `forward` 中的对应位置）中，正式服逻辑是：

```python
# 正式服
site_emb = self.spectrum_encoder(specs, elems, ionic)  # [n, 256]
# ... collate ...
struct_emb = self.site_aggregator(padded, mask)         # [B, 256]
```

QuickTest 改为：

```python
# QuickTest：n_sites 永远为 1，直接 squeeze 掉 site 维度
site_emb = self.spectrum_encoder(
    batch['spectra'][:, 0],          # [B, 1, 512]
    batch['site_elements'][:, 0],    # [B]
    batch['is_ionic'][:, 0]          # [B]
)                                    # → [B, 256]
struct_emb = site_emb                # 直接作为 struct_emb，不经过 aggregator
```

**⚠️ 实现方式**：不要直接修改 `diffusion.py`（避免污染正式服文件）。  
而是在 `step3_qt_train.py` 中，训练开始前用 monkey-patch 替换方法：

```python
import types

def _encode_xas_qt(self, batch):
    """QuickTest 版本：单 Fe 位点，跳过 aggregator"""
    struct_emb = self.spectrum_encoder(
        batch['spectra'][:, 0],
        batch['site_elements'][:, 0],
        batch['is_ionic'][:, 0],
    )
    return struct_emb   # [B, 256]，与正式服输出形状相同

# 在实例化 model 之后、trainer.fit 之前替换方法
model._encode_xas = types.MethodType(_encode_xas_qt, model)
```

这样 `diffusion.py` 完全不动，QuickTest 结束后直接还原。

**B. 其余训练配置**：

```python
trainer = pl.Trainer(
    max_epochs=30,               # 只跑 30 epoch
    gradient_clip_val=1.0,
    precision="bf16",
    devices=1,
    accelerator='gpu',
    callbacks=[
        ModelCheckpoint(
            dirpath=r"...\experiment\quicktest\qt_output",
            filename="epoch{epoch:03d}-val{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=2,
            mode="min",
        ),
    ],
)
# 不加 EarlyStopping（只跑 30 epoch，没必要）
# 不加 lr scheduler（只验证能否收敛，scheduler 留给正式服）
# lr = 1e-4（正常起点）
```

**C. sys.path 设置**（复用正式服中已确认可用的写法）：

```python
import sys, os
os.environ["PROJECT_ROOT"] = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main")
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest")
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step2")
# 导入 QuickTest 版本的 encoder（k¹ 权重）
from step2.1_qt_encoder import SpectrumEncoder, preprocess_chi
```

输出文件：`experiment/quicktest/step3_qt_train.py`

---

## 运行顺序

```
1. step1.1_qt_select_100.py      → qt_inventory.csv
2. step1.4_qt_split.py           → qt_train/val/test_ids.txt
3. step3_qt_train.py             → 训练（会自动调用 qt_dataset, qt_datamodule, qt_encoder）
```

step2.1_qt_encoder.py、step3_qt_dataset.py、step3_qt_datamodule.py 是被 step3_qt_train.py 导入的模块，不需要单独运行。

---

## 验收标准（你判断 quicktest 是否通过的依据）

| 检查项 | 通过条件 |
|--------|----------|
| 训练启动 | 无报错，第 1 个 epoch 完成 |
| Loss 趋势 | epoch 1→10 的 train_loss 单调下降 |
| 无数值异常 | 全程无 NaN / Inf |
| val_loss | 30 epoch 后 val_loss < epoch 1 的 val_loss（任何程度的下降即可） |
| 速度 | 30 epoch 总时间 < 30 分钟 |

**不需要**验证 match rate 或结构质量，quicktest 只验证 pipeline 能跑通、loss 能动。

---

## 完成后提交给 Main Agent 的报告

```markdown
### QuickTest 执行报告

### 执行状态: 通过 / 未通过

### Loss 曲线（前 30 epoch）
epoch 1:  train=X.XX, val=X.XX
epoch 5:  train=X.XX, val=X.XX
epoch 10: train=X.XX, val=X.XX
epoch 20: train=X.XX, val=X.XX
epoch 30: train=X.XX, val=X.XX

### 发现的 Bug（如有）
- （描述）

### 速度
- 每 epoch 耗时: 约 XX 秒
- 30 epoch 总时间: 约 XX 分钟

### 建议
- 若通过：可以进入正式服 Step 3 fix patch（lr 降到 1e-5 + CosineScheduler）
- 若未通过：（描述问题）
```
