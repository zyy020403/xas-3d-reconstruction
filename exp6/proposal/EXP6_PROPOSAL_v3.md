# Experiment 6 Proposal: DETR-style Set Prediction
# XAS → 局部原子结构预测(替换扩散框架)

> **状态**: DRAFT v3 (incorporating MA5 review round 2 feedback + resolution of mod 4 internal inconsistency)
> **日期**: 2026-04-29
> **v3 变更**: 4 处修改 — §7.1 五指标公式锁定、§10.1 holdout 阈值收紧至严格 beat Exp4、§6.1 DETR 超参行号 trace、§4.1(c) vocab 索引明确分离 (center vs neighbor)
> **v3 中的 MA5 mod 4 resolution**: MA5 同时说"output 用 88-class no_object=88"(暗示 output=center vocab)和"neighbor vocab 必须独立"(暗示 output=neighbor vocab,可能 ≠ 88)。v3 resolution: 两 vocab 严格分离,output 维度 = `N_NEIGHBOR_TYPES + 1`,no_object 索引 = `N_NEIGHBOR_TYPES`,**所有 §7.1 / §附录 B.5 公式中字面量 88 改为符号常量 N_NEIGHBOR_TYPES**。SA1 smoke test 阶段确定实际值并 assert 一致性。请 MA5 round 3 review 时优先确认此 resolution
> **v2 变更**: 6 处修改 — Set-Level baseline 处理、collapse 命名分离、center embedding 决策、lambda_pos 量级 caveat、持平区间收紧、附录 B 公式锁定
> **定位**: Exp4/5 扩散框架的并行替代方案,在 Exp5 跑出结果前先开训以避免空等
> **设计原则**: **工作量最小化** — 优先复用 Exp4 已验证组件,新代码尽量从 facebookresearch/detr 直接拷贝

---

## 0. 一句话目标

把 Exp4 的"扩散 decoder"换成"DETR-style Transformer encoder-decoder + 集合预测",其余(Dataset、SpectrumEncoder 前端、评估指标、数据划分)全部沿用 Exp4。

---

## 1. 为什么转向 DETR

### 1.1 Exp4 errata 暴露的扩散框架根本问题

`_density_loss` 是把 x0_hat 朝原点压的全局 L2,在 Fe-only(Exp2)合理,在 88 元素(Exp4)变成塌缩剂。但**问题不止在这一行 loss**——扩散框架本身依赖辅助 loss 给硬约束,而辅助 loss 跨任务复用就是这种隐式偏置坑的源头。

### 1.2 DETR 范式的对应优势

| Exp2-5 痛点 | DETR 范式如何解决 |
|---|---|
| `_density_loss` 塌缩剂(errata 2) | **没有**辅助物理约束 loss,直接监督学习 |
| Position-by-position TypeAcc 是虚假指标(Exp3 教训) | **天生**用 Hungarian 匹配,匹配后做监督,与"位置顺序"无关 |
| 多张谱聚合需要专门设计(Exp5 MV-attention) | **天生**用 Transformer self-attention,N 张谱就是 N 个 token |
| 扩散约束太软,RMSD 被评估机制顶住(errata 2) | **直接**回归 3D 坐标,投影 ablation 不再有意义(没东西可塌) |
| 88 元素让距离先验稀释(errata 2) | **不依赖**距离先验,每个元素自己学自己的 query |

### 1.3 这不是替代 Exp5,而是并行

- **Exp5 验证的是**: encoder 端从多张谱抽取信息能否变好
- **Exp6 验证的是**: decoder 端从扩散换成 set prediction 能否变好
- 两者**正交**。Exp5 跑通后(若有效),其多视角聚合思路可直接合并进 Exp6 的 transformer encoder——成本极低

---

## 2. 推荐的 GitHub 起点

### 2.1 主仓库

**`facebookresearch/detr`** — https://github.com/facebookresearch/detr
- 15.1k stars,Apache 2.0 license
- **作者明确说"50行 PyTorch 代码就能实现简化版"**(Standalone Colab Notebook)
- 整个仓库 99.8% Python,无需编译扩展
- 已 archive(2024-03-12),代码不再更新——对我们正好,API 稳定

### 2.2 推荐先做的两件事(0.5 天)

1. **跑一遍 Standalone Colab Notebook**
   链接: https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
   不看就开始改代码会踩很多坑。这个 notebook 50 行实现了 DETR 推理,2 小时内能完全看懂。

2. **clone 仓库到本地阅读关键 4 个文件**
   ```
   git clone https://github.com/facebookresearch/detr.git
   cd detr
   ```
   只需要看:
   - `models/transformer.py` (Transformer encoder-decoder,~300 行)
   - `models/matcher.py` (HungarianMatcher,~80 行)
   - `models/detr.py` 的 `SetCriterion` 类(set loss,~150 行)
   - `models/position_encoding.py` (position embedding,~80 行)

   其他文件(`backbone.py`, `engine.py`, `main.py`, COCO datasets/...)对我们没用。

### 2.3 拷贝清单

| DETR 文件 | 我们的位置 | 改动量 |
|---|---|---|
| `models/transformer.py` | `experiment6/shared/transformer.py` | **几乎零改动** — 直接复制 |
| `models/matcher.py` | `experiment6/shared/matcher.py` | 改 cost 计算: 把 GIoU/L1 box cost 换成 L2 position cost(min-image) |
| `models/detr.py` 的 `SetCriterion` | `experiment6/shared/criterion.py` | 改 loss 计算: 去掉 `loss_boxes` 的 GIoU,留 L2 position;`loss_labels` 的 CE 不变 |
| `models/position_encoding.py` | 不需要 | DETR 原版是 2D image 用,我们 token 序列短,用 nn.Embedding 学习式 position embedding 更简单 |

**实际新写代码量**: matcher 改 ~30 行,criterion 改 ~50 行,加起来 < 100 行。

---

## 3. Exp4 复用清单(关键)

### 3.1 直接复用,零改动

| Exp4 资产 | 复用理由 |
|---|---|
| `xas_local_dataset_v2.py` (Dataset) | L=6,min-image 折叠,过滤逻辑全部正确 |
| `holdout_*.txt` / train/val/test split files | 数据划分严格保持,确保和 Exp4 数字可比 |
| feff_features 标准化(z-score with train stats) | 与 Exp4 训练集统计量一致 |
| xmu XANES 窗口截取(150点)| 不变 |
| chi1 EXAFS 处理(200点) | 不变 |
| Set-Level TypeAcc 评估脚本 | Exp3 末期建立,Exp4 用过,直接搬 |
| Multiset F1 评估脚本 | 同上 |
| Hungarian RMSD 评估脚本(min-image) | 同上 |
| `pred_in_cutoff` / `true_in_cutoff` 评估脚本 | 同上 |

### 3.2 部分复用(只搬 SpectrumEncoder 的 CNN 前端)

```python
# Exp4 的 SpectrumEncoder 三路结构(xmu CNN + chi1 CNN + feff_feats MLP)
# 在 Exp6 里依然作为"单条谱 → token"的编码器使用,但不再 fuse 成 256d
# 而是输出 (B, 1, d_model=256) 作为 transformer encoder 的输入 token

class SpectrumTokenizer(nn.Module):  # 改自 Exp4 SpectrumEncoder
    def __init__(self, d_model=256):
        super().__init__()
        self.xmu_encoder = ...      # 完全沿用 Exp4
        self.chi_encoder = ...      # 完全沿用 Exp4
        self.feat_encoder = ...     # 完全沿用 Exp4
        self.fusion = nn.Linear(448, d_model)  # 不变,输出 256d

    def forward(self, xmu, chi, feff):
        # 输出 shape: (B, d_model)
        # 上游会 unsqueeze 成 (B, 1, d_model) 当 token 用
        ...
```

**改动**: 把 Exp4 SpectrumEncoder 最后一层 nn.Linear(latent_dim, latent_dim) 去掉,直接输出 256d 当 token。其余 100% 复用。

**关于 Exp5 v1 SA1 的 center_element_Z embedding (95→16d, output_dim=272) — 明确决策: 不 carry over**。

Exp6 中心元素信息通过 **separate learnable token**(作为 Transformer encoder input 的额外 token,与谱 token 并列)或 query embedding bias 注入,不依赖 SpectrumEncoder 内部 cat。这与 Exp5 v2 的多视角 attention 在 encoder 端的 center 处理是不同设计,Exp6 故意走另一条路——**正交假设: 同一信息不同注入路径,允许独立验证哪种更有效**。

具体实现细节由 Exp6-SA1 在 detr_xas.py 中决定,但必须在 SA1 handoff 中显式定义,**不允许悄悄复用 Exp5 v1 代码**。

### 3.3 完全废弃(不用搬)

- ❌ `diffusion_w_type_xas.py` — 整个扩散 decoder 不要
- ❌ `_density_loss` — errata 2 已确认是塌缩剂,不要
- ❌ Tweedie 公式相关代码 — 不要
- ❌ DDIM/DDPM 采样代码 — 不要
- ❌ TypeClassifier head — Exp3 双重证伪 + 自然分类是高斯分布的根本问题,不要

---

## 4. Exp6 整体架构

```
                        ┌──────────────────────────────────────┐
                        │   Spectrum 1   Spectrum 2  ...  N    │  ← 输入: N 张谱(N=1 时退化为 Exp4)
                        │      │            │           │      │
                        │      ▼            ▼           ▼      │
                        │  Tokenizer    Tokenizer  Tokenizer   │  ← 复用 Exp4 SpectrumEncoder
                        │      │            │           │      │
                        │      ▼            ▼           ▼      │
                        │  token_1       token_2      token_N  │  ← 每张谱 → 256d token
                        │      └──────┬─────┴───────────┘      │
                        │             ▼                         │
                        │  ┌──────────────────────────────┐    │
                        │  │  Transformer Encoder         │    │  ← DETR 直接拷贝
                        │  │  (self-attention over tokens)│    │
                        │  └──────────┬───────────────────┘    │
                        │             │ memory                  │
                        │             ▼                         │
                        │  ┌──────────────────────────────┐    │
                        │  │  Transformer Decoder         │    │  ← DETR 直接拷贝
                        │  │  (cross-attention with       │    │
                        │  │   20 learned object queries) │    │
                        │  └──────────┬───────────────────┘    │
                        │             │                         │
                        │     ┌───────┴───────┐                │
                        │     ▼               ▼                │
                        │  ┌──────┐       ┌──────┐            │
                        │  │ MLP  │       │ MLP  │            │  ← 新写,各 ~3 层
                        │  │ type │       │ pos  │            │
                        │  └──┬───┘       └──┬───┘            │
                        │     ▼               ▼                │
                        │ (B, 20, K+1)   (B, 20, 3)           │  ← K = N_NEIGHBOR_TYPES, +1 for no_object
                        │   logits        frac coords          │
                        └──────────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────┐
                        │  Hungarian Matching (min-image L2)   │  ← matcher.py 改 cost
                        │  Loss = λ_cls * CE + λ_pos * L2     │  ← criterion.py 改
                        └──────────────────────────────────────┘
```

### 4.1 关键设计决策

**(a) Phase 1 先做单谱(N=1),Phase 2 再扩展多谱**
- Phase 1 直接对标 Exp4 单谱基线,任何提升都干净归因到 transformer decoder
- Phase 2 等 Exp5 多谱结果出来再决定要不要做(若 Exp5 多谱无显著增益,Phase 2 也没必要做)

**(b) 20 个 object queries**
- 与 Exp4 的"20 个邻居"直接对齐
- DETR 原版用 100,我们 20 足够覆盖第一/第二配位壳

**(c) `N_NEIGHBOR_TYPES + 1` 类(neighbor element vocab + 1 "no_object")**
- DETR 经典设计:不是每个 query 都要对应真实原子,允许 query 输出"空"
- 解决了 Exp4 的"硬性 20 邻居"问题——若实际只有 17 个原子在 cutoff 内,3 个 query 输出 "no_object" 即可

**vocab 索引来源(SA1 必须显式实现,与 Exp4/5 严格区分)**:

Exp6 用 **TWO 独立 vocab**(MA5 review round 2 mod 4 resolution):

| Vocab | 用途 | size | 索引规则 |
|---|---|---|---|
| `center_vocab` | SpectrumEncoder 之外的中心元素 token / query bias 注入 | `N_CENTER_TYPES` (≈88) | sorted unique Z from `train_samples_v2.csv` `center_element` 列,Z → 0..N-1 dense 映射 |
| `neighbor_vocab` | 模型输出 type prediction 的类别空间 | `N_NEIGHBOR_TYPES` (TBD by SA1, ≥ N_CENTER_TYPES) | sorted unique Z from train spectra 实际邻居 atom_types,Z → 0..N-1 dense 映射 |

**关键约束**:
1. **不复用 Exp5 v1 SA1 的 95-class sparse vocab** (Z ∈ [2,94] 直接索引,含跳号空 slot) — Exp6 用 dense vocab
2. **output dimension = `N_NEIGHBOR_TYPES + 1`**,no_object 索引 = `N_NEIGHBOR_TYPES`
3. **所有 §7.1 / §附录 B.5 公式中的 no_object 索引必须用符号常量 `N_NEIGHBOR_TYPES`,不允许硬编码 88**
4. SA1 smoke test 阶段必须 assert `N_NEIGHBOR_TYPES ≥ N_CENTER_TYPES`(邻居元素至少包含所有中心元素)

**SA1 build phase 实现指南**:

```python
# SA1 build phase, 一次性
import pandas as pd
import json
from pymatgen.core import Element

# Step 1: build center_vocab (~88 类)
df = pd.read_csv('<EXP4_DATA_ROOT>/train_samples_v2.csv')
# 注意 Exp4 train_samples_v2 schema: mp_id, center_element, sample_name, 
#                                    site_equivalence_tag (没有 Z 列)
# 用 pymatgen 把 center_element 转 Z:
center_unique_Z = sorted(set(Element(e).Z for e in df['center_element'].unique()))
N_CENTER_TYPES = len(center_unique_Z)
# 历史预期 ~88,但 SA1 必须从 data 实算,不能硬编码:
print(f"N_CENTER_TYPES = {N_CENTER_TYPES}")
center_Z_to_idx = {z: i for i, z in enumerate(center_unique_Z)}

# Step 2: build neighbor_vocab (TBD,可能 > 88 因为 H 等中心未出现的元素可能作邻居)
# 具体实现取决于 Exp4 dataset 的邻居 atom_types 存储方式:
#   - 若 sample 级别有 neighbor_types 列: 直接 union 所有 sample 的 neighbor types
#   - 若需从 spectrum file 解析: SA1 写一次性扫描脚本,缓存结果
neighbor_unique_Z = sorted(<all unique Z appearing as neighbor in train set>)
N_NEIGHBOR_TYPES = len(neighbor_unique_Z)
print(f"N_NEIGHBOR_TYPES = {N_NEIGHBOR_TYPES}")  # 期望 ≥ N_CENTER_TYPES
assert N_NEIGHBOR_TYPES >= N_CENTER_TYPES, \
    f"邻居元素集应至少包含中心元素集: {N_NEIGHBOR_TYPES} vs {N_CENTER_TYPES}"
neighbor_Z_to_idx = {z: i for i, z in enumerate(neighbor_unique_Z)}

# Step 3: 保存
with open('experiment6/shared/exp6_element_vocab.json', 'w') as f:
    json.dump({
        'center': {
            'N_TYPES': N_CENTER_TYPES,
            'Z_to_idx': center_Z_to_idx,
            'idx_to_Z': {v: k for k, v in center_Z_to_idx.items()},
        },
        'neighbor': {
            'N_TYPES': N_NEIGHBOR_TYPES,
            'Z_to_idx': neighbor_Z_to_idx,
            'idx_to_Z': {v: k for k, v in neighbor_Z_to_idx.items()},
            'no_object_idx': N_NEIGHBOR_TYPES,  # 即 dense vocab 之外多 1 位
        }
    }, f, indent=2)
```

**模型输出维度**: `pred_logits.shape == (B, 20, N_NEIGHBOR_TYPES + 1)`,而非硬编码 89。

**OOV 邻居处理**: 若 val/test/holdout 出现 train 未见的邻居元素 Z,SA1 必须在 dataset 加载时检测并 raise(不允许 silent OOV → 错误索引)。具体策略 SA1 决定后写入 final report。

**(d) 坐标表示: 分数坐标 [-0.5, 0.5],L=6**
- 与 Exp4 完全一致,确保数字可比
- MLP pos head 末层 tanh × 0.5 把输出限制在 [-0.5, 0.5] 内(温和约束,无塌缩剂)

**(e) Hungarian cost 用 min-image L2**
- 周期性盒子,跨边界配对必须用 min-image,否则距离虚高
- DETR 原版 box cost 是 L1 + GIoU,我们去掉 GIoU(没有 box 概念),L1 换 L2

---

## 5. Loss 设计

```python
# 对每个 batch,得到 transformer 输出 (B, 20, 89_logits) + (B, 20, 3_pos)
# ground truth: (B, n_atoms, 1_type) + (B, n_atoms, 3_pos),n_atoms 可变

# Step 1: Hungarian matching (per sample)
cost_matrix[i, j] = lambda_cls * (-prob_pred_i[gt_type_j])  \
                  + lambda_pos * min_image_l2(pred_pos_i, gt_pos_j)
matching = hungarian(cost_matrix)

# Step 2: 计算 loss
# 已匹配的 query: 监督 type CE + position L2
# 未匹配的 query: 监督 type 为 "no object" (索引 NO_OBJECT_IDX = N_NEIGHBOR_TYPES),无 position loss

loss_cls = CE(pred_logits, target_classes_with_no_object)
loss_pos = L2(pred_pos[matched], gt_pos[matched])  # 只在匹配的对上算

total_loss = lambda_cls * loss_cls + lambda_pos * loss_pos
```

**初始超参**:
- `lambda_cls = 1.0`
- `lambda_pos = 5.0`(DETR 风格起点,见下方 caveat)
- `no_object_weight = 0.1`(DETR 原版 0.1,降低空类权重,与 Exp4 元素分布偏斜匹配)

**⚠️ lambda_pos 量级 caveat**: DETR 原版 5.0 是配 box L1 (4 维差绝对值和)。Exp6 用 position L2 (3 维差平方和开根),**量级不等价**。直接照搬可能让 pos loss 过强或过弱。

Phase 2 (训练 sanity) 必须**先观察 cls_loss / pos_loss 实际比值**,若超 10× 失衡即重调,目标比值 1×-3× 之间。预算 1 个 hyperparam tuning 来回(<1 天 wall time,只跑 5 epoch sanity)。

**没有 `_density_loss`,没有任何辅助物理约束** — 这是 Exp6 的核心 thesis。

---

## 6. 训练配置

```yaml
硬件: 2× RTX 4090 24GB(用 DDP)
batch_size: 64 per GPU(共 128) — 4090 显存充裕,扩散模型时代的 32 太保守
optimizer: AdamW
  lr_transformer: 1e-4
  lr_tokenizer: 1e-5  # SpectrumEncoder 前端学习率小一档,避免抖动
weight_decay: 1e-4
gradient_clip: 0.1   # DETR 推荐
max_epochs: 300      # DETR 标准 schedule,我们任务比 COCO 简单,可能更早收敛
lr_scheduler: StepLR(step_size=200, gamma=0.1)  # DETR 风格
early_stop_patience: 30  # 监控 val Set-Level TypeAcc + Hungarian RMSD
mixed_precision: bf16
```

**开训前强制检查**:
1. 5 样本 forward(): `pred_logits.shape == (5, 20, N_NEIGHBOR_TYPES + 1)`, `pred_pos.shape == (5, 20, 3)` ✓
2. pred_pos 无 NaN,数值范围合理([-1, 1] 内即可,tanh 已约束) ✓
3. Hungarian matcher 在 5 样本上跑通,matching 输出合理(20 query 中 ~17 与真实原子配对,~3 输出 no_object) ✓
4. 第 1 个 batch loss 在 [10, 100] 范围(过低或 NaN 都说明初始化有问题)✓

### 6.1 DETR 超参出处表(SA1 改超参前必查)

注: 仓库已 archive (2024-03-12),行号永久稳定。但 SA1 clone 后必须 verify 一次,在 SA1 handoff 中显式记录 verified status。

| 超参 | Exp6 v3 设值 | DETR 仓库出处 | 链接 |
|---|---|---|---|
| `lambda_cls` (`set_cost_class`) | 1.0 | `main.py` L48 | https://github.com/facebookresearch/detr/blob/main/main.py#L48 |
| `lambda_pos` (~`set_cost_bbox`) | 5.0 起步,见 §5 caveat | `main.py` L48 (DETR 是 box L1) | 同上 |
| `no_object_weight` (`eos_coef`) | 0.1 | `models/detr.py` L77 (`empty_weight[-1] = self.eos_coef`) | https://github.com/facebookresearch/detr/blob/main/models/detr.py#L77 |
| `gradient_clip` (`clip_max_norm`) | 0.1 | `main.py` L88 | https://github.com/facebookresearch/detr/blob/main/main.py#L88 |
| `max_epochs` (`epochs`) | 300 | `main.py` L51 | https://github.com/facebookresearch/detr/blob/main/main.py#L51 |
| `lr_drop` (StepLR step) | 200 | `main.py` L52 | 同上 |
| `weight_decay` | 1e-4 | `main.py` L42 | 同上 |
| `lr_transformer` | 1e-4 | `main.py` L41 | 同上 |
| `lr_tokenizer` (映射 DETR `lr_backbone`) | 1e-5 | `main.py` L40 | 同上 |
| `optimizer` | AdamW | `main.py` ~L150 | 同上 |

**Exp6 显式偏离 DETR 默认的项目(必须记录原因)**:

| 项目 | DETR 原版 | Exp6 v3 | 偏离原因 |
|---|---|---|---|
| `set_cost_giou` | 2.0 | **删除** | 我们不预测 box,只预测 position,无 IoU 概念 |
| `loss_bbox` (L1) | 5.0 | **替换为 L2** | 量级未必相同,见 §5 caveat |
| `mixed_precision` | fp32 | **bf16** | 4090 + bf16 加速 ~1.5×,任务规模 < COCO bf16 数值稳定无忧 |
| `early_stop` | 无 (跑满 300/500) | **patience=30** | 任务比 COCO 简单更早收敛,300 epoch 上限 + patience 30 兜底 |
| `auxiliary_loss` (6 层 decoder 各算 loss) | 启用 | **启用,沿用** | DETR 原版加速收敛设计,无理由偏离 |
| `position_encoding` | sin (2D image) | **learned (1D token)** | 我们 token 序列短(单谱时只有 1 个 token + center token),sin 没意义 |

---

## 7. 评估配置

**评估指标(全部沿用 Exp4)**:
- 主指标 1: **Hungarian RMSD (min-image)** — 与 Exp4 一致,目标 ≤ Exp4 holdout 1.4866 Å
- 主指标 2: **Set-Level TypeAcc** — Exp3 教训,绝不用 position-by-position
- 辅助 1: **Multiset F1** — 元素分布层面,不依赖位置对齐
- 辅助 2: **pred_in_cutoff / true_in_cutoff** — Exp2 起监控
- 辅助 3: **近配对 TypeAcc**(配对距离 < 0.5Å 才算)— Exp3 末期建立

**新增 Exp6 专属诊断指标**:
- `no_object_ratio`: 每个 query 输出 no_object 的比例。预期 ~3/20,若飙升至 15/20 说明模型学崩了(全部预测 no_object 是低 cls loss 但无意义的解)
- `query_diversity`: 20 个 query 输出位置的方差。若 < 0.01 说明所有 query 输出位置雷同(`query_pile-up`,DETR 训练初期常见,通常 30 epoch 后散开 — 严格遵循 §11.0 命名分离,**不用 collapse**)

### 7.1 五个继承指标公式锁定(SA1 不许改)

防止 Exp5 v1 时代"Set-Level / Multiset 没给公式被新 MA 自由发挥推导"的教训重演。

**公共前置(单样本)**:
- `pred_logits: (20, N_NEIGHBOR_TYPES + 1)` tensor
- `pred_pos: (20, 3)` tensor in frac coord [-0.5, 0.5]
- `pred_types_argmax: (20,)` = `pred_logits.argmax(-1)`
- 真实邻居数 n ≤ 20,`gt_types: (n,)`, `gt_pos: (n, 3)`
- `L = 6.0` (Å), `lengths = torch.tensor([6.0, 6.0, 6.0])`
- `NO_OBJECT_IDX = N_NEIGHBOR_TYPES`(注: v3 用符号常量,SA1 在 detr_xas.py 顶部 `from .vocab import N_NEIGHBOR_TYPES, NO_OBJECT_IDX`)

**公共工具函数**(matcher.py 已实现,eval 直接复用,不重复造):

```python
def min_image_l2(pred, gt, lengths):
    """min-image L2 distance, frac coord 输入, Å 输出"""
    diff = pred[:, None] - gt[None, :]      # (20, n, 3) frac
    diff = diff - torch.round(diff)         # min-image fold
    cart = diff * lengths                   # (20, n, 3) Å
    return torch.norm(cart, dim=-1)         # (20, n) Å
```

#### 指标 1: Hungarian RMSD (min-image)

```python
from scipy.optimize import linear_sum_assignment

def hungarian_rmsd(pred_pos, pred_types_argmax, gt_pos, gt_types, lengths):
    """
    Returns: rmsd (scalar Å), matched_pairs (list of (pred_idx, gt_idx))
    """
    # 只配对非 no_object 的 pred query
    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)   # (20,)
    if valid_pred_mask.sum() == 0:
        return float('inf'), []
    valid_pred_pos = pred_pos[valid_pred_mask]               # (k, 3)

    cost = min_image_l2(valid_pred_pos, gt_pos, lengths)     # (k, n) Å
    row, col = linear_sum_assignment(cost.cpu().numpy())

    matched_dists = cost[row, col]                           # (min(k,n),)
    rmsd = torch.sqrt((matched_dists ** 2).mean()).item()
    return rmsd, list(zip(row.tolist(), col.tolist()))
```
- Dataset-level: per-sample rmsd 然后平均
- 注意: pred 端先 filter no_object,gt 端不动

#### 指标 2: Set-Level TypeAcc (Exp3 教训核心)

```python
from collections import Counter

def set_level_type_acc(pred_types_argmax, gt_types):
    """
    Per-sample 多重集交集大小 / max(|pred|, |gt|)
    与坐标完全解耦,只看元素分布是否重合
    """
    valid_pred = pred_types_argmax[pred_types_argmax != NO_OBJECT_IDX].tolist()
    gt_list = gt_types.tolist()

    pred_counter = Counter(valid_pred)
    gt_counter = Counter(gt_list)

    intersection = sum((pred_counter & gt_counter).values())
    denominator = max(len(valid_pred), len(gt_list))
    if denominator == 0:
        return 0.0
    return intersection / denominator
```
- Dataset-level: per-sample 平均

#### 指标 3: Multiset F1 (Exp3 §1.1 警觉的"全猜 majority class"诊断)

```python
def multiset_f1_macro(all_pred_types_list, all_gt_types_list, n_elements):
    """
    Dataset-level macro-F1 across element classes.
    每个元素类 c 独立算 precision/recall/F1,然后 macro 平均。

    与 Set-Level 区别:
    - Set-Level 是 per-sample 平均(局部诊断)
    - Multiset F1 是 dataset-level macro across class(系统性偏差诊断)
    若 model 全猜 majority class O,Set-Level 看不出,Multiset macro F1
    会因 minority 元素 F1 拉低而显著降。

    Args:
        all_pred_types_list / all_gt_types_list: list of per-sample tensor,
            all_pred 已 filter 掉 no_object
        n_elements: N_NEIGHBOR_TYPES (不含 no_object)
    """
    f1_per_class = []
    for c in range(n_elements):
        tp = fp = fn = 0
        for pred, gt in zip(all_pred_types_list, all_gt_types_list):
            pc = Counter(pred.tolist())
            gc = Counter(gt.tolist())
            tp += min(pc.get(c, 0), gc.get(c, 0))
            fp += max(pc.get(c, 0) - gc.get(c, 0), 0)
            fn += max(gc.get(c, 0) - pc.get(c, 0), 0)
        if tp + fp == 0 or tp + fn == 0:
            continue                 # 类 c 在 dataset 不出现,跳过
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * p * r / (p + r))

    if len(f1_per_class) == 0:
        return 0.0
    return sum(f1_per_class) / len(f1_per_class)
```
- Dataset-level scalar(不是 per-sample 平均)
- macro 不是 micro,确保 minority 元素权重等同 majority

#### 指标 4: pred_in_cutoff / true_in_cutoff

```python
def in_cutoff_counts(pred_pos, pred_types_argmax, gt_pos, eval_cutoff, lengths):
    """
    eval_cutoff: per-sample 标量 (Å), 来自 dataset 的 eval_cutoff 字段
    Returns: (n_pred_in, n_true_in)
    """
    # pred 端: 笛卡尔距离原点 ≤ eval_cutoff 的非 no_object query 数
    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)
    valid_pred_pos = pred_pos[valid_pred_mask]
    pred_cart_dist = torch.norm(valid_pred_pos * lengths, dim=-1)
    n_pred_in = (pred_cart_dist <= eval_cutoff).sum().item()

    # true 端
    gt_cart_dist = torch.norm(gt_pos * lengths, dim=-1)
    n_true_in = (gt_cart_dist <= eval_cutoff).sum().item()

    return n_pred_in, n_true_in
```
- Dataset-level: 各 per-sample 平均

#### 指标 5: 近配对 TypeAcc (配对距离 < 0.5 Å)

```python
def close_pair_type_acc(pred_pos, pred_types_argmax, gt_pos, gt_types,
                        lengths, distance_threshold=0.5):
    """
    Hungarian 匹配后,只统计配对 cartesian 距离 < threshold (Å) 的对的 type
    命中率。这是 Exp3 末期建立的"可信配对"指标。
    """
    rmsd, matched = hungarian_rmsd(pred_pos, pred_types_argmax, gt_pos,
                                    gt_types, lengths)
    if len(matched) == 0:
        return 0.0

    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)
    valid_pred_pos = pred_pos[valid_pred_mask]
    valid_pred_types = pred_types_argmax[valid_pred_mask]

    cost = min_image_l2(valid_pred_pos, gt_pos, lengths)

    correct = total = 0
    for pred_i, gt_j in matched:
        if cost[pred_i, gt_j] < distance_threshold:
            total += 1
            if valid_pred_types[pred_i].item() == gt_types[gt_j].item():
                correct += 1

    if total == 0:
        return 0.0  # 没有 < 0.5 Å 的可信配对
    return correct / total
```
- Dataset-level: per-sample 平均
- 注意: 这指标 sensitive,RMSD ~1.5 Å 时可能很多样本 total=0(报告时同时报"有效样本数")

---

## 8. 文件结构

```
experiment6/
├── shared/
│   ├── xas_local_dataset_v2.py       # ← 直接从 Exp4 拷贝,零改动
│   ├── exp6_element_vocab.json       # ← SA1 build phase 生成,见 §4.1(c)
│   ├── spectrum_tokenizer.py          # ← 改自 Exp4 SpectrumEncoder,~10 行改动
│   ├── transformer.py                 # ← 直接从 DETR 拷贝
│   ├── matcher.py                     # ← 改自 DETR matcher.py,~30 行
│   ├── criterion.py                   # ← 改自 DETR SetCriterion,~50 行
│   ├── detr_xas.py                    # ← 新写,主模型类,~200 行 (含 N_NEIGHBOR_TYPES / NO_OBJECT_IDX 常量声明)
│   └── eval_metrics.py                # ← §7.1 五公式实现 + Exp4 评估 glue,SA1 直接复制锁定公式
├── step1/
│   ├── step1.0_build_vocab.py         # ← SA1 一次性,从 train_samples_v2.csv 建 center+neighbor vocab
│   ├── step1.1_recompute_exp4_setlevel.py  # ← SA1 一次性,从 Exp4 predictions_val.pt 重算 Set-Level baseline (~30 min,见 §10.1)
│   └── step1.2_smoke_test.py          # ← 5 样本 forward + matcher + loss 跑通
├── step2/
│   └── step2.1_train.py               # ← 训练主脚本,~200 行(DDP, AMP, ckpt, logging,§7.1 + 附录B.5 全指标)
├── step3/
│   └── step3.1_eval.py                # ← val/test 评估,完全调用 eval_metrics.py 锁定公式
├── step4/
│   └── step4.1_holdout.py             # ← Holdout 检验,与 Exp4 holdout 数字直接对比
└── EXP6_PROPOSAL_v3.md                # ← 本文档
```

**新写代码总量估算**: ~580 行(detr_xas.py 200 + train 200 + matcher/criterion 改 80 + smoke + eval glue 100)
**拷贝代码总量估算**: ~700 行(transformer 300 + Exp4 dataset/eval 复用 ~400)

对比 Exp4 的工程量(几千行扩散代码),**Exp6 工作量减少 60%+**。

---

## 9. 时间表

| Phase | 内容 | 时间 | 输出 |
|---|---|---|---|
| Phase 0 | 跑通 DETR Colab + 阅读 4 个核心文件 | 0.5 天 | 个人理解笔记 |
| Phase 1 | 拷贝 + 改造 4 个 DETR 文件,实现 detr_xas.py | 2 天 | smoke_test 通过 |
| Phase 2 | 训练脚本,跑 5 epoch sanity check | 1 天 | val loss 稳定下降,no_object_ratio 正常 |
| Phase 3 | 完整训练(双 4090 估计 1.5 天/300 epoch) | 1.5-2 天 | val Hungarian RMSD ≤ 1.5 Å 即视为达标基线 |
| Phase 4 | val/test/holdout 评估 + 与 Exp4 全面对比 | 1 天 | EXP6_FINAL_REPORT_v1.md |
| **总计** | | **~6 天** | |

(对比 Exp4/5 各自的 2-3 周工程,这是数量级的减少)

---

## 10. 验收标准

### 10.1 通过(进入 Exp7)

**v3 阈值采用"修法 A 严格 beat",MA5 round 2 mod 2 推荐**。理由: 验收阈值不一致是潜在隐患——v2 原版 val < 1.45(严格优于 Exp4)但 holdout < 1.50(允许退步 0.014 Å)是矛盾闸门。0.014 Å 与单 ckpt 训练随机性同量级,留这个 buffer 会让"通过 vs 部分成功"判决变模糊。§10.2 部分成功区间(val 1.45-1.49 / holdout 1.50-1.55)已经为"差不多但没严格优于"的情况留好台阶,所以 §10.1 应当严格 beat。

- val Hungarian RMSD **< 1.45 Å**(优于 Exp4 holdout 1.4866 Å)
- holdout Hungarian RMSD **< 1.4866 Å** (严格 beat Exp4 holdout)
- **Set-Level TypeAcc 报数即可,不预设阈值** — Exp4 没有 Set-Level baseline 可比(Exp4 的 0.197 是 position-by-position 虚假指标,见 ERRATA_2 §2)。Exp6 SA1 在 smoke test 阶段用 Exp4 best-ckpt-366 跑一次 Exp4 Set-Level baseline(预计 ~30 min,基于 Exp4 `predictions_val.pt` 直接重算,**无需重新 sample**),记录该数后回填 §10.1 阈值
- no_object_ratio 收敛在 [1/20, 6/20] 之间(说明 query 学会了"哪些位置该输出空")

### 10.2 部分成功(可继续优化但不替代 Exp4)

- val RMSD 1.45 - 1.49 Å / holdout RMSD 1.50 - 1.55 Å — 与 Exp4 视为持平,需进一步分析为何 transformer 没体现优势,但仍可作 Exp7 起点 ckpt
- 此时建议继续 Exp7 的 cost_density=0 投影 ablation,在扩散框架下重测真实上限

### 10.3 失败(放弃 transformer 方向)

- val RMSD ≥ 1.6 Å — 说明 transformer 在该任务上不如扩散,需重新设计或回归扩散框架

---

## 11. 风险与回退方案

### 11.0 术语命名约定 ⚠️

**Exp6 整个 proposal、SA handoff、训练日志、final report 严格遵守以下命名分离**:

| 术语 | 含义 | 性质 |
|---|---|---|
| `query_degeneracy` / `query_pile-up` | DETR 已知早期现象: 20 个 query 输出位置雷同 | **良性**,通常 30 epoch 后散开 |
| `pred_collapse` | Exp4/5 的 hard sample 上预测原子塌缩到中心 | **失败模式**,见 ERRATA_2 §1.3 塌缩根因 |

本术语区分继承自 EXP4_FINAL_REPORT_ERRATA_2.md §1.3 塌缩根因诊断,**Exp6 训练日志和 final report 必须严格遵守**——两个现象意义相反,混用会导致诊断错误。

### 11.1 已知风险

**风险 1: DETR 训练初期不稳定(query_pile-up)**(著名问题)
- 表现: 前 50 epoch loss 几乎不降,20 个 query 全部输出相近位置(query_pile-up / query_degeneracy)
- 应对: 这是 DETR 已知问题,DETR 论文也描述过。DETR 原版 500 epoch 正是因此。我们用 300 epoch 起步,若 50 epoch 还没起色,加 lr warmup(前 10 epoch 从 1e-6 线性升到 1e-4)
- 回退: 若 100 epoch 仍处于 query_pile-up 状态(`query_diversity` < 0.01),改用 Conditional DETR(收敛快 10×,见 dimiz51/DETR-Factory-PyTorch)
- **注意**: 这里讨论的是 DETR query 端的良性早期现象,**不是 Exp4/5 errata 中的 pred_collapse**(Exp6 由于没有 `_density_loss`,理论上不会出现 pred_collapse,若出现则是新问题需独立诊断)

**风险 2: cls loss 主导(N_NEIGHBOR_TYPES + 1 类),position loss 学不到**
- 表现: cls loss 收敛但 RMSD 不降
- 应对: 增大 lambda_pos 至 10 或 15,或对 cls 用 focal loss(DETR 后续工作 Deformable DETR 已用)

**风险 3: Multi-spectrum 接入(Phase 2)时 token 序列变长导致显存超**
- 应对: 不会发生。即使 N=10 谱,序列长度 = 10,远小于 DETR 原版处理的 800×1333 image 的 token 数

### 11.2 回退方案

如果 Exp6 完全失败(主验收 #10.3),按 Exp4 errata 2 的方向树回到扩散框架:
- 优先做 cost_density=0 + 投影 ablation 实验(原 Exp7 计划提前)
- 这套实验本来就要做,Exp6 失败只是顺序换一下

---

## 12. 与 Exp5 的接口

**Exp5 出结果后,根据其结果决定 Exp6 Phase 2 的开展方式**:

| Exp5 结果 | Exp6 Phase 2 行动 |
|---|---|
| MV-attention 显著提升(RMSD < 1.40) | Phase 2 优先做多谱 transformer encoder,大概率叠加收益 |
| MV-attention 持平(1.40 - 1.50) | Phase 2 也做多谱,但作为对比基线,不是主线 |
| MV-attention 退步(> 1.50) | **Phase 2 跳过** — 单谱 transformer 已是 Exp6 主结论 |

---

## 附录 A: DETR 关键概念速查(供 main agent 参考)

**Object query**: 一组可学习的 d_model 维向量,数量 = 预期最大目标数(我们 20)。每个 query 通过 cross-attention 从 encoder memory 中"查询"自己负责的目标,最终通过 MLP head 输出该目标的类型和位置。**类比**: 每个 query 是一个"侦察员",cross-attention 是它"扫描全图"找自己的目标。

**Hungarian matching**: 解二分图最大权匹配。我们 20 个 prediction vs n 个 ground truth,构造 20×n cost 矩阵(常用 padding 到 20×20,padding 用 no_object cost),scipy.optimize.linear_sum_assignment 一行解决。**关键**: 训练时每个 step 都要 match 一次,**这一步 detach,不传梯度**(因为 matching 本身是离散的)。

**No-object class**: 索引 = `NO_OBJECT_IDX` = `N_NEIGHBOR_TYPES`(即 dense neighbor vocab 之外多 1 位,见 §4.1(c) v3 vocab resolution)。若一个 query 没匹配到任何 ground truth,它的目标类别就是 no_object,cls loss 用 weight 0.1 降低权重,position 不算 loss。

**Auxiliary decoding loss**: DETR 默认在每一层 decoder 输出都算一次 loss(共 6 次),加速收敛。我们沿用,代码 DETR 已实现,无需新写。

---

## 附录 B: 给 main agent 的具体指令

执行顺序:

1. **不要**碰 Exp4/5 的代码,Exp6 全部新建在 `experiment6/` 目录下
2. **第一件事**:用户 clone facebookresearch/detr 仓库到 `experiment6/_detr_reference/`,你阅读 4 个核心文件后再开始实现,**禁止**凭记忆写 transformer 架构
3. **Phase 1 第一个产出**:`step1.1_smoke_test.py`,用 5 个样本跑通 forward + matcher + loss 反向。在这通过之前**禁止**写训练脚本
4. **训练前必须打印**:模型参数总量(预计 < 50M,DETR 原版含 ResNet 是 41M,我们没 ResNet 应该更小),所有模块的 grad_required 状态
5. **Logging 必须包含**:除 train/val loss 外,还要每 epoch 打印 `no_object_ratio` 和 `query_diversity`——这两个是 DETR 训练健康度的关键指标。**公式锁定如下,SA1 不许改**(防止 Exp5 v1 的"Set-Level/Multiset 没给公式被自由发挥"教训重演):

   **`no_object_ratio` (per epoch, dataset-level avg)**:
   ```python
   # NO_OBJECT_IDX = N_NEIGHBOR_TYPES, 见 §4.1(c) v3 resolution
   argmax = pred_logits.argmax(dim=-1)        # (B, 20)
   is_no_object = (argmax == NO_OBJECT_IDX)
   per_sample_ratio = is_no_object.float().mean(dim=-1)  # (B,)
   epoch_metric = per_sample_ratio.mean()     # scalar
   ```
   - **Healthy 区间**: [1/20, 6/20] (即 0.05 - 0.30)
   - **Failure**: > 0.5 = 模型学崩了(全部预测空,低 cls loss 的退化解)
   - **Failure**: < 0.05 = 模型不会输出空,可能强行 padding 真实原子

   **`query_diversity` (per epoch, dataset-level avg)**:
   ```python
   per_sample_std = pred_pos.std(dim=1).mean(dim=-1)  # (B,)
   # std over 20 query positions per axis, then mean over xyz
   epoch_metric = per_sample_std.mean()       # scalar
   ```
   - **Healthy 区间**: [0.05, 0.5] (frac coord 单位,等价 0.3 - 3.0 Å @ L=6)
   - **Failure**: < 0.01 = `query_pile-up`(DETR 早期常见,30 epoch 后应散开 — 严格遵循 §11.0 命名分离,**不用 collapse**)
   - **Failure**: > 0.8 = query 散布到 box 边界,异常
6. **禁止**:任何形式的辅助物理 loss(包括 density、shell、distance prior),Exp6 的 thesis 就是"不靠这些约束 transformer 也能学到结构"
7. **禁止**:重新引入 TypeClassifier head(Exp3 双重证伪 + Exp5 三重证伪 + 自然分类高斯分布的根本问题)
8. **必须**:Holdout 数字直接与 Exp4 对比,不能改 holdout 划分

---

*Main Agent 6 撰写,2026-04-29*
*v1 (initial) → v2 (MA5 round 1, 6 mods) → v3 (MA5 round 2, 4 mods + mod 4 internal inconsistency resolution)*
*基于: EXP4_FINAL_REPORT_ERRATA_2.md(扩散框架根因诊断 + collapse 命名约定)+ EXPERIMENT2_FINAL_REPORT.md(评估指标体系)+ Exp3 总结(.detach 教训和虚假指标教训)+ Exp5 v1 总结(公式自由发挥教训)+ facebookresearch/detr archive 2024-03-12(模型基础)*
