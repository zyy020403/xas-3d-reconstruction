# STEP2_HANDOFF.md
# Step 2 Agent 交接文档：谱编码模块

> **任务编号**: Step 2（共 3 个子步骤：2.1 → 2.2 → 2.3）  
> **前置条件**: 读完所有 SHARED_00 ~ SHARED_04，并确认 Step 1 已完成  
> **输入依赖**: `experiment/step1/data_inventory.csv`  
> **输出目录**: `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step2\`  
> **完成标志**: 两个模块类（SpectrumEncoder、MultiSiteAggregator）可独立实例化并通过 forward pass 测试

---

## ⚡ 开始工作前——必须向用户索取的文件

**在写任何代码之前**，请向用户索取以下文件，原因是你需要了解 DiffCSP 的条件化接口尺寸，确保本步骤输出维度与之匹配：

```
请提供以下文件内容（只需阅读，不修改）：

1. diffcsp/pl_modules/cspnet.py     ← 核心网络，了解它接受哪些条件输入，conditioning dim 是多少
2. diffcsp/pl_modules/diffusion.py  ← 了解 condition 如何进入扩散过程（是 concat、cross-attention 还是 AdaLN）
```

**读完后重点记录**：
- CSPNet 中 condition embedding 的输入维度（变量名可能是 `hidden_dim`、`cond_dim`、`time_emb_dim` 等）
- condition 进入网络的方式（加法、拼接、attention 等）
- 这个维度决定了本步骤最终输出的 `d_struct` 维度，两者必须一致

---

## 背景：本步骤在整体 pipeline 中的位置

```
[chi.dat × N sites]
       ↓ Step 2.1 (预处理 + 单谱编码)
[site_embedding_1, ..., site_embedding_N]  每个维度 d_site
       ↓ Step 2.2 (多位点聚合)
[structure_embedding]  维度 d_struct
       ↓ Step 3 (注入 DiffCSP 扩散过程)
[预测晶体结构]
```

---

## Step 2.1：谱预处理 + 单谱编码器

### 脚本名
```
step2.1_spectrum_encoder.py
```

### 任务描述

本脚本定义并测试 `SpectrumEncoder` 类，同时包含 chi.dat 预处理函数。

#### A. chi.dat 预处理（作为函数，不是类）

```python
def preprocess_chi(chi_path: str, k_grid_points: int = 512) -> torch.Tensor:
    """
    读取 chi.dat，返回归一化后的 k²χ(k) 信号，shape = [1, k_grid_points]
    
    处理步骤：
    1. 读取文件，跳过 '#' 开头的注释行
    2. 解析第一列（k 值，Å⁻¹）和第二列（χ(k)）
    3. 截取 k ∈ [0, 20] 范围内的数据点（k < 0 的部分丢弃，是仪器 artifact）
    4. 在均匀 k 网格（np.linspace(0, 20, k_grid_points)）上线性插值
    5. 计算 k²χ(k)（用插值后的 k 值平方乘以 χ(k)）
    6. 归一化：除以 max(abs(k²χ(k)))，若全为零则返回零向量
    7. 转为 torch.Tensor，shape = [1, k_grid_points]，dtype=torch.float32
    """
```

**边界情况处理**（必须处理，否则脚本会在实际数据上崩溃）：
- 文件可能只有一列（chi 值，无 k 列）→ 若只有一列，生成均匀 k 网格后直接用该列作为 χ(k)
- 数据点数不足 50 → 记录警告，返回全零向量（标记为无效，不崩溃）
- 文件为空或全为注释行 → 返回全零向量
- 插值时若 k 范围不覆盖 [0, 20]，用边界值外推（`fill_value='extrapolate'` 或直接填 0）

#### B. SpectrumEncoder 类

```python
class SpectrumEncoder(nn.Module):
    """
    将单条 k²χ(k) 信号 + 元素类型 → 局部结构 embedding
    
    输入:
        spectrum: [batch, 1, k_grid_points]  归一化的 k²χ(k)
        atomic_number: [batch]               int，元素原子序数（1-94）
        is_ionic: [batch]                    bool/int，0 或 1
    
    输出:
        site_embedding: [batch, d_site]
    """
```

**网络结构（1D CNN，保持简单）**：

```
spectrum [batch, 1, 512]
    → Conv1d(1, 32, kernel=7, padding=3) + ReLU
    → Conv1d(32, 64, kernel=5, padding=2) + ReLU + MaxPool1d(2)  → [batch, 64, 256]
    → Conv1d(64, 128, kernel=5, padding=2) + ReLU + MaxPool1d(2) → [batch, 128, 128]
    → Conv1d(128, 256, kernel=3, padding=1) + ReLU + MaxPool1d(2) → [batch, 256, 64]
    → AdaptiveAvgPool1d(1) → [batch, 256, 1]
    → Flatten → [batch, 256]
    = cnn_feat

element_embedding_table(atomic_number)  # nn.Embedding(95, d_elem=64)
    = elem_emb  [batch, 64]

ionic_embedding_table(is_ionic)  # nn.Embedding(2, d_ionic=16)
    = ionic_emb  [batch, 16]

concat([cnn_feat, elem_emb, ionic_emb])  → [batch, 256+64+16=336]
    → Linear(336, d_site) + ReLU
    → Linear(d_site, d_site)
    = site_embedding  [batch, d_site]
```

**超参数**：
- `d_site = 256`（默认，可通过构造函数参数调整）
- `k_grid_points = 512`

**重要**：`d_site` 需要在读完 cspnet.py 后确认是否需要调整，使得 `d_struct`（Step 2.2 输出）能匹配 DiffCSP conditioning dim。

#### C. 快速验证（在脚本末尾 `if __name__ == "__main__":` 块中）

```python
# 用真实数据测试一条谱
import random, os, pandas as pd
inventory = pd.read_csv(r"C:\...\experiment\step1\data_inventory.csv")
sample = inventory[inventory['files_complete']==True].iloc[0]
chi_path = os.path.join(sample['source_path'], 'chi.dat')
spec = preprocess_chi(chi_path)
print(f"spectrum shape: {spec.shape}")  # 应为 [1, 512]

encoder = SpectrumEncoder(d_site=256)
out = encoder(spec.unsqueeze(0), torch.tensor([26]), torch.tensor([0]))
print(f"site embedding shape: {out.shape}")  # 应为 [1, 256]
```

### 验收标准
- `preprocess_chi` 对正常 chi.dat 输出 `[1, 512]` tensor，值域 `[-1, 1]`
- `SpectrumEncoder.forward` 在 batch_size=4 时输出 `[4, d_site]`，无报错
- 对全零输入（无效谱）不崩溃，输出有限值

---

## Step 2.2：多位点排列不变聚合器

### 脚本名
```
step2.2_multisite_aggregator.py
```

### 任务描述

本脚本定义 `MultiSiteAggregator` 类，将 N 个（可变数量）site embedding 聚合为一个 structure embedding。

#### MultiSiteAggregator 类

```python
class MultiSiteAggregator(nn.Module):
    """
    排列不变的多位点聚合器（Attention Pooling）
    
    输入:
        site_embeddings: [batch, N, d_site]   N 个位点的 embedding
        padding_mask: [batch, N]              True = 该位置是 padding，需屏蔽
    
    输出:
        structure_embedding: [batch, d_struct]
    """
```

**网络结构（Attention Pooling，保持简单且排列不变）**：

```
site_embeddings: [batch, N, d_site]

# 1. 自注意力（捕捉位点间的相互关系）
LayerNorm(d_site) → MultiheadAttention(d_site, num_heads=4, batch_first=True)
    使用 padding_mask 屏蔽无效位点
    残差连接
→ [batch, N, d_site]

# 2. Attention Pooling（排列不变聚合）
attention_scores = Linear(d_site, 1) → [batch, N, 1]
    对 padding 位置填充 -inf，再 softmax → attention_weights [batch, N, 1]
weighted_sum = (attention_weights × site_embeddings).sum(dim=1)
→ [batch, d_site]

# 3. 投影到 d_struct
Linear(d_site, d_struct) + ReLU
Linear(d_struct, d_struct)
→ [batch, d_struct]
```

**超参数**：
- `d_struct = 256`（默认，读完 cspnet.py 后调整为匹配 DiffCSP conditioning dim）

**⚠️ 排列不变性验证**（在测试中必须验证）：
```python
# 打乱 site 顺序后，输出必须相同（误差 < 1e-5）
emb1 = aggregator(sites, mask)
perm = torch.randperm(N)
emb2 = aggregator(sites[:, perm, :], mask[:, perm])
assert torch.allclose(emb1, emb2, atol=1e-5), "不满足排列不变性！"
```

#### 数据整理辅助函数

```python
def collate_multisite_batch(site_embedding_list: list) -> tuple:
    """
    将不同化合物（不同 N）的 site embedding 列表 padding 为统一 batch
    
    输入: list of tensors，每个 shape = [n_i, d_site]，n_i 各不同
    输出: 
        padded: [batch, N_max, d_site]  padding 值为 0
        mask: [batch, N_max]            True = padding 位置
    """
```

### 验收标准
- `MultiSiteAggregator.forward` 对 N=3 和 N=7 均能正确输出 `[batch, d_struct]`
- 排列不变性验证通过（assert 不报错）
- `collate_multisite_batch` 能正确处理 N=[1, 3, 5, 2] 的混合 batch

---

## Step 2.3：端到端前向传播测试

### 脚本名
```
step2.3_e2e_forward_test.py
```

### 任务描述

使用真实数据（从 data_inventory.csv 中取 4 个不同 mp_id，各有不同数量的位点），跑通完整的 Encoder → Aggregator 前向传播，验证数据流没有问题。

#### 测试流程

```python
# 1. 从清单中取 4 个 mp_id，各有不同位点数量（如 1个位点、2个、3个、5个）
# 2. 读取每个 mp_id 的所有 chi.dat，用 preprocess_chi 处理
# 3. 用 SpectrumEncoder 分别编码每个位点的谱 → site_embeddings
# 4. 用 collate_multisite_batch 打包成 batch
# 5. 用 MultiSiteAggregator 聚合 → structure_embeddings
# 6. 打印输出形状和数值范围
```

#### 输出要求

```
=== Step 2 端到端测试 ===
mp_id=XXX: 3 sites → site_embeddings shape: [3, 256]
mp_id=XXX: 1 site  → site_embeddings shape: [1, 256]
mp_id=XXX: 5 sites → site_embeddings shape: [5, 256]
mp_id=XXX: 2 sites → site_embeddings shape: [2, 256]

Batch collated: padded shape [4, 5, 256], mask shape [4, 5]
Structure embeddings: [4, 256]
Value range: min=X.XX, max=X.XX, has_nan=False, has_inf=False
排列不变性验证: PASSED
=== 测试通过 ===
```

将以上输出内容保存至 `experiment/step2/e2e_test_log.txt`

### 模块文件存放位置

```
experiment/step2/
├── step2.1_spectrum_encoder.py      ← SpectrumEncoder 类 + preprocess_chi 函数
├── step2.2_multisite_aggregator.py  ← MultiSiteAggregator 类 + collate 函数
├── step2.3_e2e_forward_test.py      ← 端到端测试脚本
└── e2e_test_log.txt                 ← 测试输出日志（Step 2.3 生成）
```

---

## 完成后提交的总结报告（额外内容）

除标准格式外，额外需要包含：

```markdown
### DiffCSP 接口信息（读完 cspnet.py 后填写）
- CSPNet conditioning dim: （从代码中读到的数值）
- condition 进入方式: （concat / 加法 / cross-attention / AdaLN）
- 因此本步骤设定 d_struct = （数值）

### 模块尺寸汇总
- SpectrumEncoder 参数量: XX万
- MultiSiteAggregator 参数量: XX万
- 合计: XX万

### 测试结果
- e2e_test_log.txt 的内容（复制粘贴）
```

---

## 注意事项

1. **不训练，只定义结构**：Step 2 全部是模块定义 + 前向传播测试，没有任何训练循环
2. **d_struct 必须与 cspnet.py 中的 conditioning dim 一致**，读完 cspnet.py 再决定具体数值
3. **保持 quality_tier 权重字段**：`collate_multisite_batch` 输出中可以额外包含一个 `quality_weights` tensor（A=1.0，B=0.5，C=0.1），供 Step 3 训练时使用
4. **不要读取 holdout_1000_ids.txt 中任何 mp_id 的谱文件**
5. 两个模块文件（2.1 和 2.2）将在 Step 3 中被直接 import，确保它们是干净的模块，没有副作用的顶层执行代码（测试代码放在 `if __name__ == "__main__":` 块中）
