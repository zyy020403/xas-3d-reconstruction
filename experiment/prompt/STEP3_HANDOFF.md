# STEP3_HANDOFF.md
# Step 3 Agent 交接文档：DiffCSP 条件化改造

> **任务编号**: Step 3（共 3 个子步骤：3.1 → 3.2 → 3.3）  
> **前置条件**: Step 1、Step 2 均已完成  
> **核心原则**: 最大化复用原始 DiffCSP 代码，只做最小必要修改  
> **输出目录**: `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step3\`  
> **完成标志**: 训练脚本可启动，loss 正常下降，无 NaN/Inf

---

## ⚡ 开始工作前——必须向用户索取的文件

**在写任何代码之前**，请向用户提供以下文件的完整内容：

```
请提供以下文件（核心改造目标，需要阅读全文）：

1. diffcsp/pl_modules/diffusion.py     ← 主要改造对象
2. diffcsp/pl_data/dataset.py          ← 理解数据格式，用于新建 Dataset
3. diffcsp/pl_data/datamodule.py       ← 理解 DataModule 结构，用于新建 DataModule
4. conf/model/diffusion.yaml           ← 需要添加 latent_dim: 256
5. diffcsp/run.py                      ← 理解训练入口，确认如何启动
```

**读完后，在开始写代码之前，先向用户输出一份"改动清单"**，格式如下：
```
=== 改动清单（请用户确认后再动手）===
文件1: diffcsp/pl_modules/diffusion.py
  - 第 XX 行：在 __init__ 中加入 SpectrumEncoder 和 MultiSiteAggregator 实例化
  - 第 XX 行：在 forward 中加入 struct_emb 计算 + cat 到 time_emb
  - 第 XX 行：在 sample 中同步上述修改
文件2: conf/model/diffusion.yaml
  - 第 XX 行：新增 latent_dim: 256
新建文件: experiment/step3/xas_dataset.py
新建文件: experiment/step3/xas_datamodule.py
新建文件: experiment/step3/step3.3_train.py
```
**等用户确认改动清单正确后，再开始写代码。**

---

## 背景：改造思路

Step 2 报告已确认 DiffCSP 的条件注入机制：
```python
# 原始 DiffCSP（伪代码）
t_emb = time_embedding(t)                          # [batch*N_atoms, time_dim]
# CSPNet 用 t_emb 作为条件

# 改造后
struct_emb = aggregator(encoder(spectra))          # [batch, latent_dim=256]
struct_emb_expanded = struct_emb.repeat_interleave(n_atoms, dim=0)  # [batch*N_atoms, 256]
t = torch.cat([struct_emb_expanded, t_emb], dim=-1)  # [batch*N_atoms, 256+time_dim]
# 其余不变
```

**关键约束**：`cspnet.py` 完全不动，只改 `diffusion.py` 里 time embedding 的拼接方式，以及 `diffusion.yaml` 里加一个维度参数。

---

## Step 3.1：新建 XASCrystalDataset 和 XASDataModule

### 脚本名
```
experiment/step3/xas_dataset.py
experiment/step3/xas_datamodule.py
```

### 设计原则
- **不修改** `diffcsp/pl_data/dataset.py` 和 `datamodule.py`
- 新建的类**继承或平行于**原有类，返回的 `batch` 字典在原有字段基础上新增谱相关字段
- 原有 DiffCSP 的所有 crystal 数据加载逻辑（pymatgen 解析 POSCAR、lattice/frac_coords/atom_types 提取）**直接复用**，不要重写

### XASCrystalDataset

```python
class XASCrystalDataset(Dataset):
    """
    读取 data_inventory.csv，按 mp_id 组织样本。
    每个样本 = 一个化合物（一个 mp_id）的所有位点。
    
    __getitem__ 返回字典，包含：
    
    # === 原有 DiffCSP 格式的晶体结构字段 ===
    'frac_coords':    torch.Tensor [N_atoms, 3]   分数坐标
    'atom_types':     torch.Tensor [N_atoms]       原子序数
    'lengths':        torch.Tensor [3]             晶格参数 a, b, c（Å）
    'angles':         torch.Tensor [3]             晶格角 α, β, γ（度）
    'num_atoms':      int                           原子数
    
    # === 新增谱相关字段 ===
    'spectra':        torch.Tensor [n_sites, 1, 512]  各位点归一化 k²χ(k)
    'site_elements':  torch.Tensor [n_sites]           各位点元素原子序数
    'is_ionic':       torch.Tensor [n_sites]           各位点 is_ionic 标记（0/1）
    'quality_weights':torch.Tensor [n_sites]           各位点质量权重（A=1.0,B=0.5,C=0.1,unknown=0.3）
    'n_sites':        int                              位点数量
    ```
    """
```

**POSCAR 解析方法（直接复用 DiffCSP 的方式）**：
- 读完 `dataset.py` 后，找到它解析晶体结构的代码段（通常用 `pymatgen.core.Structure.from_file()`）
- 直接复制那部分逻辑，或 import 它的工具函数，不要重写

**谱加载**：
```python
# 对每个位点调用 Step 2 的预处理函数
from experiment.step2.step2_1_spectrum_encoder import preprocess_chi
spec_tensor = preprocess_chi(chi_path)  # [1, 512]
```

**错误处理**：
- 若某位点的 chi.dat 读取失败（返回全零） → `quality_weights` 对应位置设为 0.0（训练时不贡献 loss）
- 若某 mp_id 的 POSCAR 读取失败 → 跳过该 mp_id（在初始化时就过滤掉），不在 `__getitem__` 中崩溃

**初始化逻辑**：
```python
def __init__(self, mp_ids: List[int], inventory_df: pd.DataFrame, ...):
    # 1. 按 mp_id 分组，建立 mp_id → 位点文件夹列表 的映射
    # 2. 预验证：尝试读取每个 mp_id 的 POSCAR，过滤掉读取失败的
    # 3. 不在 __init__ 中加载任何谱数据（懒加载，在 __getitem__ 中按需读取）
    # 4. 建立 self.valid_mp_ids 列表，__len__ 返回 len(self.valid_mp_ids)
```

### collate_fn（处理变长 n_sites）

```python
def xas_collate_fn(batch: List[dict]) -> dict:
    """
    将 XASCrystalDataset 的一个 batch（list of dict）整理为 padded batch。
    
    晶体结构字段：按 DiffCSP 原有方式处理（参考原 dataset.py 的 collate_fn，直接复用）
    谱相关字段：
        spectra → pad 到 [batch, n_sites_max, 1, 512]
        site_elements, is_ionic, quality_weights → pad 到 [batch, n_sites_max]
        spectra_mask → [batch, n_sites_max]，True=padding 位置
    """
```

**重要**：读完原有 `dataset.py` 和 `datamodule.py` 后，找到它们的 `collate_fn`，直接复用晶体结构部分的处理逻辑，只新增谱字段的 padding 处理。

### XASDataModule

```python
class XASDataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        # 读取 experiment/step1/ 的4个 id 文件
        # 实例化 XASCrystalDataset(train_ids, ...) 等
        # 注意：holdout_1000_ids.txt 中的 mp_id 严禁传入任何 Dataset

    def train_dataloader(self):
        return DataLoader(..., collate_fn=xas_collate_fn, 
                         num_workers=4, pin_memory=True)
    # val_dataloader, test_dataloader 同理
```

---

## Step 3.2：修改 diffusion.py（最小改动）

### 要改动的文件
```
diffcsp/pl_modules/diffusion.py  （直接修改原文件，改完后保留备份至 experiment/step3/diffusion_backup.py）
```

**在修改前，先把原文件完整备份**：
```python
# 脚本开头执行（只需一次）
import shutil
shutil.copy(r"C:\...\DiffCSP-main\diffcsp\pl_modules\diffusion.py",
            r"C:\...\DiffCSP-main\experiment\step3\diffusion_backup.py")
```

### 具体改动（精确到位置，读完文件后填写行号）

#### 改动 1：`__init__` 中新增模块实例化

在 `CSPDiffusion.__init__` 的末尾（或合适位置）新增：

```python
# === XAS conditioning modules (新增) ===
import sys
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step2")
from step2_1_spectrum_encoder import SpectrumEncoder, preprocess_chi
from step2_2_multisite_aggregator import MultiSiteAggregator, collate_multisite_batch

self.spectrum_encoder = SpectrumEncoder(d_site=256)
self.site_aggregator = MultiSiteAggregator(d_site=256, d_struct=hparams.latent_dim)
# === 新增结束 ===
```

#### 改动 2：`forward` 中计算 struct_emb 并拼接

读完 `forward` 方法后，找到计算 `time_emb`（或 `t_emb`）的那一行，在其**后面**插入：

```python
# === XAS conditioning (新增) ===
# batch 中新增字段: spectra [B, n_sites, 1, 512], site_elements [B, n_sites],
#                  is_ionic [B, n_sites], spectra_mask [B, n_sites], quality_weights [B, n_sites]
B = batch['num_atoms'].shape[0]
site_embs_list = []
for b in range(B):
    n = batch['n_sites'][b]
    specs = batch['spectra'][b, :n]           # [n, 1, 512]
    elems = batch['site_elements'][b, :n]     # [n]
    ionic = batch['is_ionic'][b, :n]          # [n]
    site_emb = self.spectrum_encoder(specs, elems, ionic)  # [n, 256]
    site_embs_list.append(site_emb)

padded, mask, qw = collate_multisite_batch(site_embs_list,
                                           quality_weights_list=[
                                               batch['quality_weights'][b, :batch['n_sites'][b]]
                                               for b in range(B)])
struct_emb = self.site_aggregator(padded, mask)  # [B, latent_dim]

# 按原子数广播：每个原子都接收对应化合物的 struct_emb
struct_emb_expanded = struct_emb.repeat_interleave(
    batch['num_atoms'], dim=0)                # [sum(N_atoms), latent_dim]

# 拼接到 time_emb（找到原代码中 t_emb 或 time_emb 的变量名，替换下面的 time_emb）
time_emb = torch.cat([struct_emb_expanded, time_emb], dim=-1)
# === 新增结束 ===
```

⚠️ **注意**：上述代码中 `time_emb` 变量名需根据你读到的 `diffusion.py` 实际代码替换为正确的变量名。

#### 改动 3：`sample` 方法同步修改

`sample` 方法中同样有生成 time embedding 的逻辑，用相同方式将 struct_emb 拼入。  
`sample` 方法的输入需要额外接收 `spectra_batch` 参数（包含谱数据）。  
具体参数名读完代码后确定。

#### 改动 4：conf/model/diffusion.yaml

在文件合适位置新增一行：
```yaml
latent_dim: 256
```

---

## Step 3.3：训练启动脚本

### 脚本名
```
experiment/step3/step3.3_train.py
```

### 任务描述

这个脚本的目标是**复用 DiffCSP 原有的训练逻辑**，只替换 DataModule 和 Model（加入谱条件化后的版本）。

**读完 `run.py` 后，按以下思路改造**：

```python
# run.py 原本大概是这样（伪代码，以实际为准）：
model = CSPDiffusion(cfg)
datamodule = CrystDataModule(cfg)
trainer = pl.Trainer(...)
trainer.fit(model, datamodule)

# 改造后：
model = CSPDiffusion(cfg)            # 模型已被修改，无需额外操作
datamodule = XASDataModule(cfg)      # 替换为新的 DataModule
trainer = pl.Trainer(...)            # 完全复用，不改
trainer.fit(model, datamodule)       # 完全复用，不改
```

**训练配置（写入 experiment/step3/train_config.yaml）**：

```yaml
# 继承原有 diffusion.yaml，只覆盖必要参数
defaults:
  - /model: diffusion          # 使用改造后的 diffusion.yaml

# 训练参数（针对 A4000 16GB）
train:
  batch_size: 16               # 保守起点，可根据显存情况调整
  num_workers: 4
  max_epochs: 200
  learning_rate: 1e-4

# 输出路径
hydra:
  run:
    dir: experiment/step3/training_output
```

**GPU/显存保护措施**（在训练脚本中必须加入）：
```python
# 梯度裁剪，防止谱编码器早期训练不稳定
trainer = pl.Trainer(
    gradient_clip_val=1.0,          # 必须
    precision=16,                   # 混合精度，节省显存
    devices=1,                      # 单 GPU
    accelerator='gpu',
    ...
)
```

**Loss 权重（site quality weight 的使用）**：

在 `CSPDiffusion` 的 loss 计算部分，读完代码后找到 loss 计算行，在计算之前加入：

```python
# quality_weight 对每个化合物是标量（各位点权重的加权平均）
# 用于对整个样本的 loss 进行缩放
sample_quality_weight = (qw * (~mask).float()).sum(dim=1) / (~mask).float().sum(dim=1)
# 广播到原子维度
sample_qw_expanded = sample_quality_weight.repeat_interleave(batch['num_atoms'], dim=0)
loss = (loss_per_atom * sample_qw_expanded).mean()  # 替换原来的 loss.mean()
```

---

## 完成后提交的总结报告（额外内容）

```markdown
### DiffCSP 改动摘要
- diffusion.py 实际修改行数: XX 行
- 改动 1（__init__）位置: 第 XX 行
- 改动 2（forward）位置: 第 XX 行
- 改动 3（sample）位置: 第 XX 行
- diffusion.yaml 改动: 第 XX 行新增 latent_dim

### 训练启动验证
- 能否成功启动（Y/N）:
- 前 10 个 step 的 loss 值: （复制粘贴日志）
- 显存占用（MB）: 
- 是否出现 NaN/Inf: 

### 发现的问题
- （如有）
```

---

## 注意事项

1. **先备份 diffusion.py**，改之前一定要做，出问题可以还原
2. **改动清单先给用户确认**，再动手写代码，避免改错位置
3. **`sys.path.insert` 的路径是绝对路径**，用 Windows 的 `r"..."` 字符串，注意反斜杠
4. **混合精度（precision=16）可能与 float32 的谱 tensor 产生冲突**，若出现 loss NaN，先关掉 `precision=16` 调试，确认无 NaN 后再打开
5. **严禁加载 holdout_1000_ids.txt 中的任何 mp_id 的谱或 POSCAR**，XASDataModule 的 setup 中必须有过滤逻辑
6. **Step 3 不做超参数搜索**，只验证训练可以正常启动且 loss 单调下降，超参数调优放到 Step 4
