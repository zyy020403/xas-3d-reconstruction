# Experiment 6 Proposal: DETR-style Set Prediction
# XAS → 局部原子结构预测(替换扩散框架)

> **状态**: DRAFT v7 (incorporating MA5 EXPERIMENT5_FINAL_REPORT_v2 lessons learned — thesis 修订 + Exp5' 三件套物理 loss 移植 + 用户物理 sanity 必经)
> **日期**: 2026-05-01
> **v7 重大变更**: 1 处主线变更 + 4 处级联调整
> - **变更 (loss-side, thesis 修订)**: 吸收 MA5 EXPERIMENT5_FINAL_REPORT_v2 的 §6.2 / §6.6 教训 — Exp5 v2 95% 物理违反率证明"训练目标没要求的事,模型不会自己学"。Exp6 v6 之前坚持的"thesis: 不靠 attractive prior 也能学到结构"已被 MA5 用硬数据证伪。**v7 直接把 Exp5' 的三件套物理 loss 移植进 Exp6**: `cost_pairwise_min` (λ=1.0) + `cost_shell_dist` (λ=0.5) + `cost_shell_count` (λ=0.2),配合 v4 已有的 repulsion hinge 形成完整物理监督
> - thesis 重写: 从"不靠 attractive prior"调整为"**不靠 implicit attractive prior(`_density_loss`-like 全局 L2 朝原点压,无方向 grounding),允许 explicit data-driven physical loss(基于 GT shell_boundaries.pkl + min_d 的硬性物理事实,有 ground truth grounding)**"。两者本质区别: implicit prior 是作者直觉,explicit data-driven loss 是数据本身约束
> - 级联调整: §5 加三件套 loss 公式 + §5.1 thesis 重写 + §5.2 lambda 量级表;§附录B 第 6 条 / 第 11 条修订;§10 验收加"用户物理 sanity 必经"前置;§8 文件结构 step1.3 baseline_cps 强制运行 Exp4 best ckpt 物理对照 (从 v6 的"可选"升级为"必跑");§11 加风险 8 (三 loss 项干扰)
> - **Exp6 定位调整**: 从"thesis 验证 + 架构对照"降级为"架构对照"(transformer vs diffusion),thesis 验证已被 Exp5 v2 的负结果完成,Exp6 不再走同一条坑
> **v6 变更**: shell 划分根因修正 — 用 GT shell boundaries 替换全局固定 shell 范围
> **v5 变更**: MIN_PDIST data-driven calibration
> **v4 变更**: 加 pairwise repulsion hinge loss + Composite Physical Score (CPS) 主验收
> **v3 变更**: 4 处 — §7.1 五指标公式锁定、§10.1 holdout 阈值收紧至严格 beat Exp4、§6.1 DETR 超参行号 trace、§4.1(c) vocab 索引明确分离 (center vs neighbor)
> **v2 变更**: 6 处 — Set-Level baseline 处理、collapse 命名分离、center embedding 决策、lambda_pos 量级 caveat、持平区间收紧、附录 B 公式锁定
> **定位**: Exp4/5 扩散框架的并行替代方案。**v7 起 Exp6 不再做 thesis 验证(已被 Exp5 v2 完成),专注 transformer vs diffusion 架构对照**
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
| **`shell_boundaries.pkl`** (v6 关键) | Exp4 step 2.5 产出,128,382 sample 的 sample-specific shell 划分 (gap-based,threshold p10=0.1563 Å)。v6 CPS 评估直接 load 这份,作为 ground truth shell 归属 — 不再用 v5 的全局固定 SHELL1_RANGE/SHELL2_RANGE。文件 schema 见 §7.2.0 |
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
# 对每个 batch,得到 transformer 输出 (B, 20, K+1_logits) + (B, 20, 3_pos)
# K = N_NEIGHBOR_TYPES, K+1 包含 no_object 类
# ground truth: (B, n_atoms, 1_type) + (B, n_atoms, 3_pos),n_atoms 可变
# v7 新增: 从 shell_boundaries.pkl 加载 per-sample shell_starts/ends/n_atoms (前 2 个 shell)

# Step 1: Hungarian matching (per sample) - 不变
cost_matrix[i, j] = lambda_cls * (-prob_pred_i[gt_type_j])  \
                  + lambda_pos * min_image_l2(pred_pos_i, gt_pos_j)
matching = hungarian(cost_matrix)

# Step 2: 标准 DETR 监督 - 不变
loss_cls = CE(pred_logits, target_classes_with_no_object)
loss_pos = L2(pred_pos[matched], gt_pos[matched])

# Step 3 (v4): pairwise repulsion hinge — 强制原子间最小距离
loss_repulsion = compute_repulsion_hinge(pred_pos, pred_types_argmax, lengths,
                                         min_pdist=MIN_PDIST)

# Step 4 (v7 新增): pairwise minimum distance HARD penalty (Exp5' 三件套之 1)
# 与 repulsion hinge 不同: hinge 是 "max(0, d-r)^2" 软约束 (违反一点点惩罚一点点)
# pairwise_min 是 "1.0 if any pair < threshold else 0" 硬指示器,通过 BCE 监督
loss_pairwise_min = compute_pairwise_min_penalty(pred_pos, pred_types_argmax,
                                                  lengths, min_pdist=MIN_PDIST)

# Step 5 (v7 新增): shell distance loss (Exp5' 三件套之 2)
# 让 model 学到 "原子应该按 GT 真实壳层距离分布"
loss_shell_dist = compute_shell_distance_loss(pred_pos, pred_types_argmax,
                                               gt_shell_starts, gt_shell_ends,
                                               gt_shell_n_atoms, lengths)

# Step 6 (v7 新增): shell count loss (Exp5' 三件套之 3)
# 让 model 学到 "每壳层应该有多少原子"
loss_shell_count = compute_shell_count_loss(pred_pos, pred_types_argmax,
                                             gt_shell_starts, gt_shell_ends,
                                             gt_shell_n_atoms, lengths)

# v7 total loss
total_loss = (lambda_cls * loss_cls
              + lambda_pos * loss_pos
              + lambda_rep * loss_repulsion
              + lambda_pmin * loss_pairwise_min       # v7
              + lambda_sdist * loss_shell_dist        # v7
              + lambda_scount * loss_shell_count)     # v7
```

### 5.1 Thesis 重写(v7) — implicit attractive prior 禁,explicit data-driven physical loss 允许

#### 5.1.1 v6 thesis 的失败 — MA5 EXPERIMENT5_FINAL_REPORT_v2 §6.2 / §6.6 教训

**v6 之前的 thesis**: "transformer 不靠 attractive 物理 prior 也能学到结构"。这个 thesis 把所有 attractive prior 一刀切归为禁忌,基于 errata 2 的 `_density_loss` 教训外推。

**MA5 EXP5_v2 final report 用 95% 物理违反率证明此 thesis 错误**:
- Exp5 v2 沿用了类似 thesis(没加任何物理 loss,只靠 RMSD 监督 + 软 density)
- 结果: gate_pass_rate 5-11%,Shell-1 distance score = 0.0000(即使 gate-pass 子集),composite mean 0.005-0.011
- MA5 §6.6 lesson: "训练目标没要求的事,模型不会自己学" — 这是 ML 第一原理
- MA5 §6.2 lesson: "Step 2.5 ground truth 应进训练,不只评估"
- Exp5' 因此被设计为 from-scratch 重训 + 三件套物理 loss

**Exp6 v6 还坚持 v6 thesis = 主动重蹈 Exp5 v2 覆辙 = 浪费 GPU**。v7 修订。

#### 5.1.2 v7 thesis: 区分 implicit prior 和 explicit data-driven loss

**新二分法**(v7 取代 v4 的 attractive vs repulsive 二分):

| 类型 | 例子 | 是否允许 | 判断标准 |
|---|---|---|---|
| **Implicit attractive prior**(作者直觉性 prior) | `_density_loss` 把 x0_hat 朝原点压;距离先验"Fe-O 应该 2.0 Å"硬编码;假设原子分布对称的 attractive loss | **禁** | loss 项里有作者主观选择的目标值/分布,跨数据集复用就坑 |
| **Explicit data-driven physical loss**(GT-grounded loss) | shell_dist loss(loss target 来自 GT shell_boundaries.pkl);shell_count loss(同);pairwise_min loss(target 来自 train RDF 的 MIN_PDIST 校准) | **允许且必须** | loss 项的"target 值"完全来自数据集本身 (GT 或 RDF 校准),不含作者直觉 |
| **Repulsive hinge constraint**(物理底线,保留 v4 用法) | repulsion_hinge: pred-pred 对距离 < MIN_PDIST 罚 | **允许**(v4 已加) | 无方向偏置,只在违反时激活 |

类比:
- implicit prior 像告诉学生"答案应该是 5"(作者预设)
- explicit data-driven loss 像给学生看"这是这道题型的标准答案" (题型本身的事实)
- repulsive hinge 像告诉学生"答案不能是负数"(普适物理底线)

**v7 决定加入三件套 loss 的依据**:
- shell_dist / shell_count 的 target 直接来自 `shell_boundaries.pkl` (Exp4 step 2.5 算的 sample-specific GT)
- pairwise_min 的 target 来自 SA1 step1.0 RDF 校准 (train pair distance 1st percentile)
- 这三项都是 explicit + data-driven,不含作者直觉,与 `_density_loss` 完全不同
- MA5 EXP5_v2 §6.2 已用 95% 违反率证明: 不加这些 loss,模型不会自己学

#### 5.1.3 v7 vs MA5 Exp5' 的差别

| 维度 | Exp5' | Exp6 v7 | 备注 |
|---|---|---|---|
| 模型架构 | DiffCSP 扩散 (Exp5 v2 沿用) | DETR Transformer | **v7 唯一 thesis 项** |
| 三件套物理 loss | ✅ 三项 | ✅ 三项 (移植自 Exp5') | 完全相同设计 |
| repulsion hinge | 可能没显式有 | ✅ 有 (v4 已加) | v7 比 Exp5' 多一项 hinge,作冗余安全 |
| shell_boundaries.pkl 用途 | 进训练 loss + 评估 | 进训练 loss + 评估 | 完全一致 |
| Best ckpt selection | val_loss 复合 (α/β/γ) | CPS-based | v7 主指标 CPS 已经 covers 物理性 |

**v7 唯一与 Exp5' 真正不同的实验变量是模型架构**。这正是 Exp6 应该贡献的: transformer vs diffusion 在同等物理监督下谁好。Exp6 verdict 后才能说"transformer 范式对此任务更/不如扩散范式"。

#### 5.1.4 v6 二分法保留作历史档案

v4/v5/v6 用的"attractive vs repulsive"二分法在 v7 不再是主要框架,但仍正确(repulsive hinge 仍是允许 loss 的一种 — 见上表)。v7 的新二分法是一个更精细的划分:把 attractive 进一步细分为"implicit (禁)"和"explicit data-driven (允许)"。

**v6 公式锁定 (compute_repulsion_hinge) 完全保留,不动**。

### 5.2 三件套物理 loss 公式锁定 (v7,SA1 不许改)

公式设计严格沿用 MA5 Exp5' 提案,唯一改动是张量 shape 适配 DETR (B, 20, 3) 而非扩散 (n_atoms_total, 3)。

#### 5.2.1 Pairwise minimum distance penalty

```python
def compute_pairwise_min_penalty(pred_pos, pred_types_argmax, lengths,
                                  min_pdist=MIN_PDIST):
    """
    硬指示器版本: 任意 pred-pred 对 < min_pdist 即 sample-level 触发 1,否则 0
    与 compute_repulsion_hinge 区别: hinge 是连续软约束,
    pairwise_min 是离散硬指示 (per-sample 0/1),通过对 batch 求 mean 得到 batch-level 违反率
    
    设计理由: hinge 训练后期梯度衰减到 0 后无法再 push,pairwise_min 用 BCE-like 形式持续监督
    
    Returns: scalar tensor in [0, 1] (batch-level violation rate)
    """
    B = pred_pos.shape[0]
    violations = torch.zeros(B, device=pred_pos.device)
    
    for b in range(B):
        valid_mask = (pred_types_argmax[b] != NO_OBJECT_IDX)
        if valid_mask.sum() < 2:
            continue
        valid_pos = pred_pos[b][valid_mask]
        
        diff = valid_pos[:, None] - valid_pos[None, :]
        diff = diff - torch.round(diff)
        cart = diff * lengths
        pdist = torch.norm(cart, dim=-1)
        
        eye = torch.eye(valid_pos.shape[0], device=pdist.device, dtype=torch.bool)
        min_d = pdist[~eye].min()
        
        # 软 sigmoid 指示: min_d < min_pdist 时接近 1, min_d > min_pdist 时接近 0
        # 比硬 (min_d < min_pdist).float() 多了梯度
        violations[b] = torch.sigmoid(10.0 * (min_pdist - min_d))
    
    return violations.mean()
```

**关键设计**:
1. **soft sigmoid** 而非硬指示器: 保留梯度
2. **温度 10.0**: 让 sigmoid 在 min_pdist 附近陡峭 (transition zone ~0.1 Å),避免 "差一点也罚很多" 的过强惩罚
3. **与 repulsion_hinge 的关系**: 两者**互补**, hinge 在违反程度大时主导 (||max(0, r-d)||^2 跟违反程度成正比), pairwise_min 在 violation 接近 0 时仍提供持续梯度 (sigmoid 在边界附近梯度最大)

#### 5.2.2 Shell distance loss

```python
def compute_shell_distance_loss(pred_pos, pred_types_argmax,
                                 gt_shell_starts, gt_shell_ends, gt_shell_n_atoms,
                                 lengths, n_shells=2):
    """
    让 model 学到"原子应该按 GT shell 真实距离分布"
    
    Per-sample 算法:
    1. 把 valid pred 按距 sort,取最近 sum(GT shell_n_atoms[:n_shells]) 个原子作为 "pred 前几壳"
    2. 这些原子的距离与 GT 期望距离 (shell_starts/ends 中点) 算 L2
    3. 距离 sort 后做 monotonic alignment,避免 Hungarian 二次开销
    
    Returns: scalar tensor
    """
    B = pred_pos.shape[0]
    losses = []
    
    for b in range(B):
        valid_mask = (pred_types_argmax[b] != NO_OBJECT_IDX)
        if valid_mask.sum() == 0:
            continue
        valid_pos = pred_pos[b][valid_mask]
        
        # pred 距 sorted ascending
        pred_dist = (valid_pos * lengths).norm(dim=-1)
        pred_dist_sorted, _ = pred_dist.sort()
        
        # GT 期望距离: 每壳层中点重复 n_atoms 次
        gt_targets = []
        for s in range(min(n_shells, len(gt_shell_starts[b]))):
            shell_mid = (gt_shell_starts[b][s] + gt_shell_ends[b][s]) / 2.0
            n_in_shell = gt_shell_n_atoms[b][s]
            gt_targets.extend([shell_mid] * n_in_shell)
        
        if len(gt_targets) == 0:
            continue
        gt_targets = torch.tensor(gt_targets, device=pred_pos.device)
        
        # 取前 min(len(gt_targets), len(pred_dist_sorted)) 个对齐
        n_compare = min(len(gt_targets), len(pred_dist_sorted))
        pred_aligned = pred_dist_sorted[:n_compare]
        gt_aligned = gt_targets[:n_compare]
        
        sample_loss = ((pred_aligned - gt_aligned) ** 2).mean()
        losses.append(sample_loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    return torch.stack(losses).mean()
```

**关键设计**:
1. **sort-based monotonic alignment**: 不用 Hungarian (O(k^2) 而非 O(k^3)),足够准确因为距离按物理本来就近似 sorted
2. **shell 中点作 target**: 不用 shell_starts 也不用 shell_ends,避免边界 bias
3. **最多评 2 个壳**: 与 §7.2 CPS 一致,远壳的 loss 噪声大不评

#### 5.2.3 Shell count loss

```python
def compute_shell_count_loss(pred_pos, pred_types_argmax,
                              gt_shell_starts, gt_shell_ends, gt_shell_n_atoms,
                              lengths, n_shells=2, tol_band=TOL_SHELL_BAND):
    """
    让 model 学到"每壳层应该有多少原子"
    用 GT shell 边界 (扩张 tol_band) 数 pred 落入数,与 GT 数对比
    
    使用 differentiable count via sigmoid: 
        count(d in [a, b]) ≈ sum_i sigmoid(τ(d_i - a)) - sigmoid(τ(d_i - b))
    
    Returns: scalar tensor
    """
    B = pred_pos.shape[0]
    losses = []
    tau = 20.0  # 温度,让 sigmoid 在边界陡峭 (transition ~0.05 Å)
    
    for b in range(B):
        valid_mask = (pred_types_argmax[b] != NO_OBJECT_IDX)
        if valid_mask.sum() == 0:
            continue
        valid_pos = pred_pos[b][valid_mask]
        pred_dist = (valid_pos * lengths).norm(dim=-1)
        
        for s in range(min(n_shells, len(gt_shell_starts[b]))):
            lo = gt_shell_starts[b][s] - tol_band
            hi = gt_shell_ends[b][s] + tol_band
            n_gt = float(gt_shell_n_atoms[b][s])
            
            # differentiable count
            n_pred_soft = (torch.sigmoid(tau * (pred_dist - lo)) -
                          torch.sigmoid(tau * (pred_dist - hi))).sum()
            
            # L1 loss on count difference
            shell_loss = torch.abs(n_pred_soft - n_gt)
            losses.append(shell_loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    return torch.stack(losses).mean()
```

**关键设计**:
1. **differentiable count via sigmoid**: 硬 count(`(dist >= lo) & (dist < hi)`) 不可微,sigmoid 近似让梯度通过
2. **L1 而非 L2**: count 差是离散值,L1 对 ±1 的小差不过度惩罚
3. **共享 tol_band (TOL_SHELL_BAND = 0.1 Å)**: 与 §7.2 CPS 评估一致,训练 / 评估目标对齐

### 5.3 初始超参 (v7 — 6 项 lambda)

| Lambda | v7 设值 | 来源 / 校准方式 |
|---|---|---|
| `lambda_cls` | 1.0 | DETR `set_cost_class` |
| `lambda_pos` | 5.0 | DETR `set_cost_bbox`,L1→L2 量级 caveat 见下 |
| `lambda_rep` | 1.0 | v4 设定 |
| `lambda_pmin` | 1.0 | v7 起步,与 `lambda_rep` 互补 |
| `lambda_sdist` | 0.5 | Exp5' 提案设定,v7 沿用 |
| `lambda_scount` | 0.2 | Exp5' 提案设定,v7 沿用 |
| `no_object_weight` | 0.1 | DETR 原版 0.1 |

**⚠️ lambda_pos 量级 caveat**: DETR 原版 5.0 是配 box L1。Exp6 用 position L2,**量级不等价**。Phase 2 sanity 必须先观察 cls_loss / pos_loss 实际比值,失衡即重调,目标比值 1×-3× 之间。

**⚠️ 6 项 lambda 总比值 caveat (v7)**: 训练 sanity 阶段 (5 epoch) 必须 logging 6 项 loss 的实际数值,目标后期总比值约:

```
loss_cls : loss_pos : loss_rep : loss_pmin : loss_sdist : loss_scount
   1.0   :   1.0   :   0.1    :   0.05    :   0.5      :   0.3
```

(loss_rep + loss_pmin 双重监督最小距离,合计 ~0.15 是健康的;shell loss 主导次于位置回归 loss 是设计意图)

如果训练初期 (epoch 5) 三件套 loss 中任一项 dominant (> 0.5 × loss_cls),触发 hyperparam tuning re-run。预算 1-2 个 tuning 来回 (~1 天 wall time)。

**⚠️ Exp5' lambda schedule 经验** (MA5 final report §4.1 提到): Exp5' 设计了 `cost_pairwise_min` λ=1.0 起步,但 SA-METRICS-V3 dry-run 显示初期 violation 大,可能需要 schedule (e.g. 前 50 epoch λ=2.0,后期降回 1.0)。**v7 SA1 同样需要观察,如果 epoch 5 的 pairwise violation rate > 50%,提前考虑 schedule**。

### 5.4 v7 thesis 简短总结 (供 SA / agent / 后续 ExpN 引用)

**Exp6 v7 的核心 hypothesis**:

1. ✅ 接受: 物理监督 loss 必须显式加入,基于 GT shell_boundaries.pkl + RDF 校准的 explicit data-driven loss 不破坏好实验设计
2. ✅ 接受: implicit attractive prior (作者直觉的 distance/density target) 仍然禁止
3. **要验证**: 在同等物理监督下 (三件套 loss + repulsion hinge),DETR-style transformer 是否优于 DiffCSP 扩散 (Exp5')

**Exp6 verdict 类型**:
- 如果 Exp6 CPS > Exp5' CPS → transformer 范式胜 (Exp7+ 主线)
- 如果 Exp6 CPS ≈ Exp5' CPS → 物理 loss 是主导贡献,架构 secondary,Exp6 仍有价值作"另一种实现的对照"
- 如果 Exp6 CPS < Exp5' CPS → diffusion 范式胜,Exp7 不再做 transformer

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

### 7.2 Composite Physical Score (CPS) — v4 主验收指标 + v6 shell 划分根因修正

**用户观察 round 1**(2026-04-29): "用 Hungarian 一一对应然后看元素准确率,太严格了"——near-pair TypeAcc (§7.1 指标 5) 在 RMSD ~1.5 Å 时大量样本 total=0,无判别力。CPS 用物理意义的 shell-level 评估,容错合理,作为 v4 主验收。

**用户观察 round 3**(2026-04-30): "Exp4/5 模型根本没按真实分布划分壳层"。深入分析 `shell_boundaries.pkl` 后确认 — Exp4 step 2.5 算出的 sample-specific shell 划分(gap-based,128,382 sample 全做了)**只用在 Dataset 过滤和评估端,从未进 loss**(errata 2 §1 已部分指出)。Exp4/5 模型本质上从未学过壳层概念,只学了"原子应该在原点附近"(`_density_loss`)。

**v6 解决方案**:

不在 loss 里直接监督壳层(违反 thesis,会变 attractive prior)。改为**在评估端用 ground truth shell 边界**给预测原子分壳层,让 shell-matching 成为 CPS 主导项。这样 best ckpt selection 自然朝 shell 结构靠拢,而 loss 仍保持 v4 的 cls + pos + repulsion 三项 thesis-clean 设计。

**v5 vs v6 关键差别(不要混淆)**:

| 维度 | v5 设计 | v6 设计 | 改动理由 |
|---|---|---|---|
| Shell 边界来源 | 全局固定 SHELL1_RANGE=(0.5, 2.8), SHELL2_RANGE=(2.8, 4.5) | per-sample 从 `shell_boundaries.pkl` 读 GT `shell_starts`/`shell_ends` | Exp4 88 元素中心,真实 shell 高度 sample-specific |
| Shell 数量 | 强制 2 个壳 (1st + 2nd) | per-sample variable,但 CPS 只评估 GT 中的"前 2 个 shell" (即 GT shell index 0 + 1) | 真实样本 shell 数 1-17 不等,前 2 壳是物理最重要的部分 |
| eval_cutoff | 不使用 | sample-specific eval_cutoff (Exp4 step 2.5 已为每 sample 算好) | 第 20 邻居所在壳的外缘,自然边界 |
| Pred 原子分壳 | pred dist ∈ [0.5, 2.8) → shell 1 | pred dist ∈ [GT shell_starts[0], GT shell_ends[0]] → 算入 shell 1, etc. | 与 GT 的物理壳层对齐 |

#### 7.2.0 shell_boundaries.pkl schema (SA1 必读)

数据来自 Exp4 step 2.5,L=128,382 个样本,每个 sample 是一个 dict:

```python
{
    'threshold': 0.1563,           # gap_threshold (Å), p10 of train gap distribution
    'distances': np.float32 (n,),  # n 个邻居距离 (笛卡尔 Å),已 sorted ascending
    'species_Z': np.int8 (n,),     # 各邻居的 Z (与 distances 同序)
    'shell_starts': np.float32 (S,),  # 每个 shell 的最小距离 (S 个 shell)
    'shell_ends': np.float32 (S,),    # 每个 shell 的最大距离
    'shell_n_atoms': np.int32 (S,),   # 每个 shell 的原子数
    'shell_of_atom': np.int32 (n,),   # 每个原子的 shell index (0..S-1)
    'eval_cutoff': float,             # 包含第 20 邻居的那个 shell 的外缘 (Å)
    'n_center_sites': int,            # 多位点信息,evaluation 不直接用
}
```

**实际样本举例(用户上传数据真实样本)**:

```
mp-556058__mp-556058-EXAFS-O-K (O 中心,氧化物):
  shell_starts: [2.04, 2.30, 2.48, 2.98, 3.51, 4.62, 6.44]
  shell_ends:   [2.08, 2.31, 2.64, 3.03, 4.26, 6.15, 9.99]
  shell 0 (1st): [2.04, 2.08] — 2 atoms (氧化物中 O-Cation 短键)
  shell 1 (2nd): [2.30, 2.31] — 2 atoms (氧化物中 O-Cation 中键)
  ↑ v5 全局 SHELL1_RANGE=(0.5, 2.8) 会把 shell 0 + shell 1 + shell 2 部分合并,完全错!

mp-6026__mp-6026-EXAFS-C-K (C 中心,碳化物):
  shell_starts: [1.29, 2.90, 3.69, 4.13, 4.68, 6.57]
  shell 0: [1.29, 1.29] — 3 atoms (C-X 短键)
  ↑ v5 SHELL1_RANGE 起点 0.5 包含 1.29 还行,但 v5 终点 2.8 切掉 [1.29, 2.79] 的 shell 0 之后却把 shell 1 起点 2.90 误当 shell 1 了 (实际 GT shell 1 是 2.90-3.45,但 v5 把 [2.8, 4.5) 全划成 shell 2)

mp-1043472__mp-1043472-EXAFS-Mo-K (Mo 中心):
  shell 0: [1.79, 1.79] — 1 atom
  shell 1: [1.96, 2.02] — 5 atoms
  ↑ shell 0 与 shell 1 紧挨着 (gap 仅 0.17 Å,接近 gap_threshold),v5 全局边界根本切不出来
```

#### 7.2.1 设计原则 (v6)

- **Hard gate**: 物理可行性 (Physical Validity, PV)。pred-pred 任意对距离 < MIN_PDIST → CPS = 0(MIN_PDIST 由 SA1 step1.0 RDF calibration 后写入 `min_pdist_calibration.json`,见 §7.2.5 任务 1)
- **6 加权子项**: 按用户给定"重要程度"严格降序权重
- **Shell 划分(v6 关键)**: per-sample 从 `shell_boundaries.pkl` 加载 GT `shell_starts[0:2]` 和 `shell_ends[0:2]` 作为 **1st shell** 和 **2nd shell** 的边界。**只评估前 2 个壳层**,后续壳层 (shell index ≥ 2) 不计入 CPS
- **Pred 原子分壳**: 对每个 valid (非 no_object) pred 原子,算笛卡尔距离 d:
  - 若 `GT_shell_starts[0] - tol_shell_band ≤ d ≤ GT_shell_ends[0] + tol_shell_band` → 算入 1st shell
  - 若 `GT_shell_starts[1] - tol_shell_band ≤ d ≤ GT_shell_ends[1] + tol_shell_band` → 算入 2nd shell
  - 否则 → 不算入任何壳 (但仍参与 PV 计算)
  - `tol_shell_band = 0.1 Å` (壳层边界扩张容错,避免边界硬切)
- **Edge case**: 若 sample GT 没有 2nd shell (shell 数 = 1),T2/D2/C2 三项跳过(weight 重新归一化到 [C1, D1, T1] 上)。SA1 必须在 step1.0a integrity check 报告这种 sample 的占比
- **CNO 合并**: C(Z=6), N(Z=7), O(Z=8) 在 type score 中视为同一虚拟类(用户实测谱图几乎不可区分)
- **per-sample CPS ∈ [0, 1]**, dataset-level CPS = per-sample 平均

#### 7.2.2 权重表(SA1 不许改)

| 项 | 含义 | 容错 | weight | 排序理由 |
|---|---|---|---|---|
| **PV** | Physical Validity (hard gate) | 任意 pred-pred 对距离 < MIN_PDIST → CPS=0 | (gate, 不参与加权) | feff 不能算 → 一票否决 |
| C1 | 1st shell 配位数 | ±1.5 个 | **0.25** | 用户排序 #1, 配位环境核心 |
| D1 | 1st shell mean 键长(到中心)| ±0.2 Å | **0.20** | 用户排序 #2, 键长几何核心 |
| T1 | 1st shell 元素种类 (CNO 合并) | (multiset overlap) | **0.17** | 用户排序 #3 |
| C2 | 2nd shell 配位数 | ±3 个 | **0.15** | 用户排序 #4 |
| D2 | 2nd shell mean 键长(到中心) | ±0.2 Å | **0.13** | 用户排序 #5 |
| T2 | 2nd shell 元素种类 (CNO 合并) | (multiset overlap) | **0.10** | 用户排序 #6 |
| **Sum** | | | **1.00** | |

注: 严格降序 (0.25 > 0.20 > 0.17 > 0.15 > 0.13 > 0.10),与用户排序完全对齐。

#### 7.2.3 公式锁定(SA1 不许改)

```python
import torch
from collections import Counter
import json
import pickle

# 模块级常量(SA1 在 detr_xas.py / eval_metrics.py 顶部声明)

# v6 关键: shell 边界不再全局固定,而是 per-sample 从 shell_boundaries.pkl 加载
# load 一次,SA1 启动时:
#   with open('experiment6/shared/shell_boundaries.pkl', 'rb') as f:
#       SHELL_BOUNDARIES = pickle.load(f)  # dict: sample_name -> shell info dict
SHELL_BOUNDARIES = None     # SA1 必须显式 load

TOL_SHELL_BAND = 0.1        # 壳层边界扩张容错 (Å), v6 新增,避免边界硬切

# v5 calibration: MIN_PDIST 从 SA1 step1.0 输出文件加载,proposal 阶段不硬定
# 加载方式: 
#   with open('experiment6/shared/min_pdist_calibration.json') as f:
#       MIN_PDIST = json.load(f)['min_pdist']
# 若 calibration 文件不存在 (smoke test 早期),fallback 到 1.5 Å placeholder 并 warning
MIN_PDIST = None            # SA1 必须显式 load,不允许直接硬编码

TOL_C1 = 1.5                # 1st shell 配位数容错
TOL_C2 = 3.0                # 2nd shell 配位数容错
TOL_D = 0.2                 # 距离容错 (两壳共用)
COUNT_DECAY = 5.0           # 配位数 score 超容错后线性衰减到 0 的范围
DIST_DECAY = 1.0            # 距离 score 超容错后衰减到 0 的范围
WEIGHTS = {                 # v4 锁定权重
    'C1': 0.25, 'D1': 0.20, 'T1': 0.17,
    'C2': 0.15, 'D2': 0.13, 'T2': 0.10,
}
CNO_VIRTUAL_CLASS = -1      # C/N/O 合并后的虚拟 idx


def physical_validity(pred_pos_valid, lengths, min_pdist=MIN_PDIST):
    """Tier 0 hard gate. Returns: bool"""
    if len(pred_pos_valid) < 2:
        return True
    diff = pred_pos_valid[:, None] - pred_pos_valid[None, :]
    diff = diff - torch.round(diff)
    cart = diff * lengths
    pdist = torch.norm(cart, dim=-1)
    eye = torch.eye(len(pred_pos_valid), dtype=torch.bool, device=pdist.device)
    return pdist[~eye].min().item() >= min_pdist


def split_shells_from_gt(pos, types, lengths, gt_shell_starts, gt_shell_ends,
                          tol=TOL_SHELL_BAND):
    """
    v6 核心: 用 GT sample-specific shell 边界给 pred 原子分壳。
    
    Args:
        pos: (k, 3) frac coord pred valid positions
        types: (k,) pred types argmax
        lengths: (3,)
        gt_shell_starts: np.float32 (S,), GT shell starts in Å (笛卡尔)
        gt_shell_ends: np.float32 (S,), GT shell ends in Å
        tol: shell 边界扩张容错 (Å)
    
    Returns:
        (s1_pos, s1_types), (s2_pos, s2_types):
            shell 1 (GT shell index 0) 和 shell 2 (GT shell index 1) 的 pred 原子
        n_shells_in_gt: int, GT 中的 shell 总数 (用于 edge case 处理)
    """
    n_shells_in_gt = len(gt_shell_starts)
    if n_shells_in_gt == 0:
        # GT 完全没有 shell — 不应出现 (Exp4 step 2.5 至少给出 1 个 shell);raise
        raise ValueError(f"GT shell count is 0, unexpected. Check shell_boundaries.pkl integrity")
    
    dist = (pos * lengths).norm(dim=-1)  # (k,) cart Å
    
    # Shell 1 (GT shell 0)
    s1_lo = gt_shell_starts[0] - tol
    s1_hi = gt_shell_ends[0] + tol
    s1_mask = (dist >= s1_lo) & (dist <= s1_hi)
    
    # Shell 2 (GT shell 1) — 若 GT 只有 1 个 shell,返回空
    if n_shells_in_gt >= 2:
        s2_lo = gt_shell_starts[1] - tol
        s2_hi = gt_shell_ends[1] + tol
        s2_mask = (dist >= s2_lo) & (dist <= s2_hi)
    else:
        s2_mask = torch.zeros_like(s1_mask)
    
    return ((pos[s1_mask], types[s1_mask]),
            (pos[s2_mask], types[s2_mask]),
            n_shells_in_gt)


def split_shells_from_gt_for_truth(gt_distances, gt_species_Z, gt_shell_of_atom,
                                     n_shells_in_gt):
    """
    GT 端从 shell_of_atom 直接分壳,无需用 distance 二次计算 — 用 Exp4 已算好的归属。
    
    Args:
        gt_distances: np.float32 (n,) GT 邻居距离
        gt_species_Z: np.int8 (n,) GT 邻居 Z (后续 SA1 转 neighbor_vocab idx)
        gt_shell_of_atom: np.int32 (n,) GT 每原子的 shell index
    
    Returns: ((s1_dist, s1_Z), (s2_dist, s2_Z))
    """
    s1_mask = (gt_shell_of_atom == 0)
    s2_mask = (gt_shell_of_atom == 1) if n_shells_in_gt >= 2 else \
              torch.zeros_like(torch.from_numpy(gt_shell_of_atom), dtype=torch.bool).numpy()
    return ((gt_distances[s1_mask], gt_species_Z[s1_mask]),
            (gt_distances[s2_mask], gt_species_Z[s2_mask]))


def count_score(n_pred, n_gt, tolerance):
    diff = abs(n_pred - n_gt)
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (diff - tolerance) / COUNT_DECAY)


def distance_score_from_dists(pred_dists, gt_dists, tolerance=TOL_D):
    """
    v6: 距离 score 直接用 1D distance arrays 算,不再用 frac pos+lengths
    (因为 pred 端用 split_shells_from_gt 出来的是 pos,但 gt 端是 distance,统一为 distance 形式)
    
    Args:
        pred_dists: np.array (k_pred,) cart Å
        gt_dists: np.array (k_gt,) cart Å
    """
    n_pred, n_gt = len(pred_dists), len(gt_dists)
    if n_pred == 0 and n_gt == 0:
        return 1.0   # both empty, perfect
    if n_pred == 0 or n_gt == 0:
        return 0.0   # one empty, can't compare → 0
    pred_mean = float(pred_dists.mean()) if hasattr(pred_dists, 'mean') else \
                sum(pred_dists)/len(pred_dists)
    gt_mean = float(gt_dists.mean()) if hasattr(gt_dists, 'mean') else \
              sum(gt_dists)/len(gt_dists)
    err = abs(pred_mean - gt_mean)
    if err <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (err - tolerance) / DIST_DECAY)


def merge_cno(types_array, idx_to_Z_or_passthrough):
    """
    C(6), N(7), O(8) → CNO_VIRTUAL_CLASS (-1)
    
    Args:
        types_array: 可以是 pred neighbor_vocab idx (需 idx_to_Z map 转 Z), 
                     或 GT species_Z (idx_to_Z_or_passthrough='passthrough')
    """
    if len(types_array) == 0:
        return types_array
    if idx_to_Z_or_passthrough == 'passthrough':
        Z_list = [int(z) for z in types_array]
    else:
        Z_list = [idx_to_Z_or_passthrough[int(t)] for t in types_array]
    out = []
    for z in Z_list:
        if z in (6, 7, 8):
            out.append(CNO_VIRTUAL_CLASS)
        else:
            out.append(z)
    return out


def type_score_cno_merged(pred_types, gt_types_Z, neighbor_idx_to_Z):
    """
    Args:
        pred_types: pred neighbor_vocab idx (k_pred,)
        gt_types_Z: GT species_Z (k_gt,)
    """
    n_pred, n_gt = len(pred_types), len(gt_types_Z)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    pred_m = merge_cno(pred_types.tolist() if hasattr(pred_types, 'tolist') else pred_types,
                       neighbor_idx_to_Z)
    gt_m = merge_cno(gt_types_Z.tolist() if hasattr(gt_types_Z, 'tolist') else gt_types_Z,
                     'passthrough')
    pc = Counter(pred_m)
    gc = Counter(gt_m)
    intersection = sum((pc & gc).values())
    denom = max(len(pred_m), len(gt_m))
    return intersection / denom


def composite_physical_score(pred_pos, pred_types_argmax, sample_name, lengths,
                              neighbor_idx_to_Z):
    """
    Per-sample CPS (v6 — uses GT shell boundaries from shell_boundaries.pkl).
    
    Args:
        pred_pos: (20, 3) frac
        pred_types_argmax: (20,) neighbor_vocab idx (含 NO_OBJECT_IDX 占位)
        sample_name: str, key into SHELL_BOUNDARIES
        lengths: (3,)
        neighbor_idx_to_Z: dict
    
    Returns: (score: float, breakdown: dict)
    """
    # Load GT shell info
    gt_info = SHELL_BOUNDARIES[sample_name]
    gt_distances = gt_info['distances']
    gt_species_Z = gt_info['species_Z']
    gt_shell_of_atom = gt_info['shell_of_atom']
    gt_shell_starts = gt_info['shell_starts']
    gt_shell_ends = gt_info['shell_ends']
    n_shells_in_gt = len(gt_shell_starts)
    
    # Filter pred no_object
    valid_mask = (pred_types_argmax != NO_OBJECT_IDX)
    pred_pos_v = pred_pos[valid_mask]
    pred_types_v = pred_types_argmax[valid_mask]
    
    # Tier 0: Physical Validity hard gate
    pv = physical_validity(pred_pos_v, lengths)
    if not pv:
        return 0.0, {'PV': False, 'reason': 'pairwise_distance_violation',
                     'C1': 0, 'D1': 0, 'T1': 0, 'C2': 0, 'D2': 0, 'T2': 0,
                     'n_shells_in_gt': n_shells_in_gt}
    
    # Tier 1: shell split — pred 端用 GT 边界,GT 端用 GT shell_of_atom
    (ps1_pos, ps1_t), (ps2_pos, ps2_t), _ = split_shells_from_gt(
        pred_pos_v, pred_types_v, lengths, gt_shell_starts, gt_shell_ends)
    (gs1_d, gs1_Z), (gs2_d, gs2_Z) = split_shells_from_gt_for_truth(
        gt_distances, gt_species_Z, gt_shell_of_atom, n_shells_in_gt)
    
    # 转 pred pos → cart distance
    ps1_dists = (ps1_pos * lengths).norm(dim=-1).cpu().numpy() if len(ps1_pos) else \
                __import__('numpy').array([])
    ps2_dists = (ps2_pos * lengths).norm(dim=-1).cpu().numpy() if len(ps2_pos) else \
                __import__('numpy').array([])
    
    # Tier 2: subscores
    c1 = count_score(len(ps1_pos), len(gs1_d), TOL_C1)
    d1 = distance_score_from_dists(ps1_dists, gs1_d)
    t1 = type_score_cno_merged(ps1_t, gs1_Z, neighbor_idx_to_Z)
    
    if n_shells_in_gt >= 2:
        c2 = count_score(len(ps2_pos), len(gs2_d), TOL_C2)
        d2 = distance_score_from_dists(ps2_dists, gs2_d)
        t2 = type_score_cno_merged(ps2_t, gs2_Z, neighbor_idx_to_Z)
        # Standard 6-term weighted composite
        composite = (WEIGHTS['C1']*c1 + WEIGHTS['D1']*d1 + WEIGHTS['T1']*t1 +
                     WEIGHTS['C2']*c2 + WEIGHTS['D2']*d2 + WEIGHTS['T2']*t2)
    else:
        # Edge case: GT 只有 1 shell — 重新归一化 weight 到 [C1, D1, T1] 上
        c2 = d2 = t2 = None  # 不计入
        w_sum = WEIGHTS['C1'] + WEIGHTS['D1'] + WEIGHTS['T1']
        composite = (WEIGHTS['C1']*c1 + WEIGHTS['D1']*d1 + WEIGHTS['T1']*t1) / w_sum
    
    return composite, {
        'PV': True, 'C1': c1, 'D1': d1, 'T1': t1,
        'C2': c2, 'D2': d2, 'T2': t2, 'total': composite,
        'n_shells_in_gt': n_shells_in_gt,
        'pred_n_in_shell1': len(ps1_pos),
        'pred_n_in_shell2': len(ps2_pos),
        'gt_n_in_shell1': len(gs1_d),
        'gt_n_in_shell2': len(gs2_d) if n_shells_in_gt >= 2 else 0,
    }
```

**v6 关键设计实现细节**:

1. **`SHELL_BOUNDARIES = None`**: SA1 在 train/eval 启动时一次性 load `experiment6/shared/shell_boundaries.pkl` (Exp4 step 2.5 已经产出)
2. **per-sample 调用接口变更**: v5 是 `composite_physical_score(pred_pos, pred_types, gt_pos, gt_types, lengths, ...)`, v6 是 `composite_physical_score(pred_pos, pred_types, sample_name, lengths, ...)` — **gt_pos 和 gt_types 不再传入**,因为 v6 完全用 `shell_boundaries.pkl` 里的 GT(已包含 distances + species_Z + shell_of_atom)
3. **SA1 train loop 必须把 sample_name 传入 evaluate 函数**: 这要求 dataloader 输出 sample_name(Exp4 dataset 应该已支持,SA1 verify)
4. **Edge case GT 只有 1 shell**: 1 shell 的 sample 在 Exp4 数据中不罕见(看 mp-6026 C 中心样本就是 6 个 shell,但有些样本可能只有 1 个),v6 weight 归一化处理
5. **`shell_boundaries.pkl` integrity check 在 SA1 step1.0a 必做**: 见 §7.2.5 task 0

#### 7.2.4 Logging 要求(SA1 必须实现)

每 epoch validation 必须 logging 的 CPS-related 项:

| Metric | 公式 | 用途 |
|---|---|---|
| `val_cps_mean` | per-sample CPS 平均 | **主验收指标** |
| `val_pv_pass_rate` | % of samples with PV=True | 物理可行性,期望训练后期 > 95% |
| `val_C1_mean` | 仅对 PV=True 样本平均 | 子项诊断 (排除 hard-gate 0 干扰) |
| `val_D1_mean` | 同上 | |
| `val_T1_mean` | 同上 | |
| `val_C2_mean`, `val_D2_mean`, `val_T2_mean` | 同上;若 GT 仅 1 shell 跳过该 sample | |
| `val_cps_breakdown_hist` | 6 个子项各自的分布 (10-bin histogram) | 诊断哪些子项拖后腿 |
| **`val_pred_in_shell1_mean`** (v6 新增) | per-sample `pred_n_in_shell1` 平均 | 诊断 model 是否学到 1st shell 几何 — 训练后期应接近 GT shell 1 平均原子数 |
| **`val_pred_in_shell2_mean`** (v6 新增) | per-sample `pred_n_in_shell2` 平均 | 同上 2nd shell |
| **`val_pred_outside_shells_ratio`** (v6 新增) | per-sample `(20 - n_no_object - pred_in_s1 - pred_in_s2) / (20 - n_no_object)` 平均 | 关键诊断: 落在 GT 前两壳之外的 valid pred 比例。训练后期应 < 0.3,> 0.5 表示 model 仍把原子撒在远处 |
| **`val_n_shells_in_gt_dist`** (v6 新增, 一次性) | val set 中 GT shell 数的 histogram | SA1 sanity report,确认 1-shell sample 占比是否在合理范围 |

#### 7.2.5 SA1 sanity 阶段必做的三项基线计算 (v6 +1 task)

**任务 0: shell_boundaries.pkl integrity check**(v6 新增,先做,作为 §7.2 CPS 前置)

`step1.0a_shell_integrity_check.py` 的职责:

```python
# 输入: experiment6/shared/shell_boundaries.pkl (从 Exp4 step 2.5 直接复制过来)
# 输出: experiment6/shared/shell_integrity_report.json
#       + 失败时 raise,SA1 必须把样本踢出训练 (与 Exp4 v2 dataset 对齐)

import pickle
import numpy as np
import json
import pandas as pd

with open('experiment6/shared/shell_boundaries.pkl', 'rb') as f:
    SB = pickle.load(f)

# Step 1: 与 Exp4 dataset 的 sample list 对齐
inv = pd.read_csv('<EXP4_DATA_ROOT>/data_inventory_v2.csv')
exp4_sample_names = set(inv['sample_name'].tolist())
sb_keys = set(SB.keys())

missing_in_sb = exp4_sample_names - sb_keys
extra_in_sb = sb_keys - exp4_sample_names
print(f"Exp4 dataset samples not in shell_boundaries.pkl: {len(missing_in_sb)}")
print(f"shell_boundaries samples not in Exp4 dataset: {len(extra_in_sb)}")

# 重要: Exp4 dataset 中所有 sample 都必须在 shell_boundaries 里
if len(missing_in_sb) > 0:
    print(f"WARNING: {len(missing_in_sb)} samples missing from shell_boundaries.pkl")
    print(f"Exp6 dataloader 必须 skip 这些样本")

# Step 2: 统计 GT shell 数分布
n_shells_dist = []
n_atoms_in_shell1 = []
n_atoms_in_shell2 = []
shell1_widths = []
shell2_widths = []
samples_with_only_1_shell = 0

for name, info in SB.items():
    if name not in exp4_sample_names:
        continue
    n_shells = len(info['shell_starts'])
    n_shells_dist.append(n_shells)
    if n_shells >= 1:
        n_atoms_in_shell1.append(info['shell_n_atoms'][0])
        shell1_widths.append(info['shell_ends'][0] - info['shell_starts'][0])
    if n_shells >= 2:
        n_atoms_in_shell2.append(info['shell_n_atoms'][1])
        shell2_widths.append(info['shell_ends'][1] - info['shell_starts'][1])
    else:
        samples_with_only_1_shell += 1

# Step 3: 关键 sanity assertions
n_total = len(n_shells_dist)
frac_1_shell = samples_with_only_1_shell / n_total
print(f"Total samples: {n_total}")
print(f"Mean n_shells: {np.mean(n_shells_dist):.2f}")
print(f"Samples with only 1 shell: {samples_with_only_1_shell} ({frac_1_shell*100:.1f}%)")
print(f"Mean atoms in shell 1: {np.mean(n_atoms_in_shell1):.2f}")
print(f"Mean shell 1 width: {np.mean(shell1_widths):.3f} Å")

# Sanity assertions (SA1 必须通过)
assert frac_1_shell < 0.10, \
    f"超过 10% 的样本 GT 仅 1 shell ({frac_1_shell*100:.1f}%),CPS 评估有问题"
assert np.mean(n_atoms_in_shell1) >= 2.0, \
    f"shell 1 平均原子数过低 ({np.mean(n_atoms_in_shell1):.2f}),数据有问题"
assert np.mean(shell1_widths) <= 0.5, \
    f"shell 1 平均宽度过大 ({np.mean(shell1_widths):.3f} Å),gap_threshold 太松"

# 写报告
result = {
    'n_total_samples_in_exp4': n_total,
    'samples_missing_from_shell_boundaries': len(missing_in_sb),
    'samples_extra_in_shell_boundaries': len(extra_in_sb),
    'n_shells_distribution_summary': {
        'mean': float(np.mean(n_shells_dist)),
        'median': float(np.median(n_shells_dist)),
        'p10': float(np.percentile(n_shells_dist, 10)),
        'p90': float(np.percentile(n_shells_dist, 90)),
        'min': int(np.min(n_shells_dist)),
        'max': int(np.max(n_shells_dist)),
    },
    'samples_with_only_1_shell': samples_with_only_1_shell,
    'frac_only_1_shell': float(frac_1_shell),
    'shell1_n_atoms_mean': float(np.mean(n_atoms_in_shell1)),
    'shell2_n_atoms_mean': float(np.mean(n_atoms_in_shell2)),
    'shell1_width_mean_A': float(np.mean(shell1_widths)),
    'shell2_width_mean_A': float(np.mean(shell2_widths)),
}

with open('experiment6/shared/shell_integrity_report.json', 'w') as f:
    json.dump(result, f, indent=2)
```

**SA1 必须额外产出 shell n_atoms 直方图** (`shell_n_atoms_hist.png`),覆盖范围 0-20 bin 1,人工 review 确认 shell 1 的中位数在合理化学范围内(典型氧化物 4-6 配位)。

**任务 1: MIN_PDIST RDF calibration**(v5 新增,基于 v6 的 task 0 完成后做)

`step1.0_rdf_analysis.py` 的职责:

```python
# 输入: train set (xas_local_dataset_v2.py)
# 输出: experiment6/shared/min_pdist_calibration.json

import torch
import json
import numpy as np
from collections import defaultdict

def collect_pair_distances(train_loader, lengths, n_samples=2000):
    """
    收集 train set 的所有 atom-atom pair distance (min-image, cartesian Å)。
    n_samples = 2000 已经足够稳定统计 (~ 200k pairs)。
    """
    all_pdist = []
    for i, batch in enumerate(train_loader):
        if i >= n_samples:
            break
        gt_pos = batch['frac_coords']      # (n, 3) per sample
        gt_types = batch['atom_types']     # (n,) — for 后续 element-stratified
        if len(gt_pos) < 2:
            continue
        diff = gt_pos[:, None] - gt_pos[None, :]
        diff = diff - torch.round(diff)
        cart = diff * lengths
        pdist = torch.norm(cart, dim=-1)
        eye = torch.eye(len(gt_pos), dtype=torch.bool)
        all_pdist.append(pdist[~eye].flatten().cpu().numpy())
    return np.concatenate(all_pdist)


def calibrate_min_pdist(all_pdist, percentile=1.0, safety_margin=0.1):
    """
    取 train pair distance 的 1st percentile,减 safety margin (0.1 Å)。
    1st percentile 而非 absolute min,避免 outlier(数据噪声、数值精度伪假对)误伤。
    """
    p1 = np.percentile(all_pdist, percentile)
    abs_min = all_pdist.min()
    proposed_min_pdist = max(0.7, p1 - safety_margin)  # 0.7 Å 是 absolute 下界 (氢键最短)
    
    return {
        'min_pdist': float(proposed_min_pdist),
        'calibration_method': f'{percentile}th_percentile_minus_{safety_margin}A',
        'train_pair_dist_p1': float(p1),
        'train_pair_dist_p5': float(np.percentile(all_pdist, 5)),
        'train_pair_dist_min': float(abs_min),
        'train_pair_dist_median': float(np.median(all_pdist)),
        'n_pairs_sampled': int(len(all_pdist)),
        'n_pairs_below_proposed_min_pdist': int((all_pdist < proposed_min_pdist).sum()),
        'fraction_violating_proposed': float((all_pdist < proposed_min_pdist).mean()),
    }


# 必须 sanity check
assert result['fraction_violating_proposed'] < 0.005, \
    f"超过 0.5% 的 train pair 在 proposed MIN_PDIST 之下,proposed 阈值过高: {result}"
assert 0.7 <= result['min_pdist'] <= 1.8, \
    f"MIN_PDIST 跑出合理范围 [0.7, 1.8] Å,SA1 必须人工 review RDF 直方图: {result}"

with open('experiment6/shared/min_pdist_calibration.json', 'w') as f:
    json.dump(result, f, indent=2)
```

**SA1 必须额外产出 RDF 直方图** (`min_pdist_rdf_hist.png`),覆盖范围 [0, 4 Å] bin 0.05 Å,人工 review 一眼能看出第一峰位置(典型化学键 ~1.0-2.5 Å)。如果直方图显示 < 1.5 Å 区域有显著质量(e.g. > 1% 的 pair),proposed_min_pdist 自动降到 1st percentile - margin,**这正是数据驱动校准的目的**。

**预期 calibration 结果**(SA1 跑完后回填,proposal 阶段仅给 placeholder):
- 若 train 数据是典型氧化物/氟化物 → MIN_PDIST 落在 1.4-1.6 Å,与用户初始直觉吻合
- 若 train 数据含氢化物/有机配体 → MIN_PDIST 可能 0.8-1.0 Å (O-H ~0.96, C-H ~1.09)
- 若 SA1 跑出 MIN_PDIST < 0.7 Å → 数据有问题(原子重合),触发 dataset 检查

calibration 完成后 MIN_PDIST 写入 `min_pdist_calibration.json` 并 **冻结**;后续所有 SA 不允许修改此值,除非重新跑 step1.0。

---

**任务 2: CPS baselines**(原 §7.2.5 内容)

阈值 §10 由 SA1 实测后回填,不在 proposal 阶段硬定。SA1 必须计算:

1. **Random baseline CPS**: 用 `xas_local_dataset_v2.py` val set,每个 sample 把 pred_pos 随机撒在 [-0.5, 0.5]、pred_types 随机 sample from neighbor vocab 分布,跑 1000 sample 平均 CPS。预期 ~0.05-0.15
2. **Exp4 best ckpt CPS**: 用 Exp4 `predictions_val.pt` 的 (pos, types),套 Exp6 composite score 公式跑一次。预期 ~0.30-0.50(Exp4 RMSD ~1.49 + 元素 acc 0.20 经 CNO 合并后会跳到 ~0.50,但 PV 可能拖累——这正是用户观察的物理不合理样本)
3. **(可选) Exp5 best ckpt CPS**: 若 Exp5 完成,同样跑一次

三个数字写入 `experiment6/step1/baseline_cps.json`,后续 §10 验收阈值基于这三个数 + 0.10 absolute margin 定。

**三个任务的依赖关系**: 任务 0 (shell integrity) → 任务 1 (MIN_PDIST RDF) → 任务 2 (CPS baselines)。任务 0 必须最先,因为 CPS 公式整个依赖 shell_boundaries.pkl;任务 1 在任务 2 之前完成,因为 CPS 公式中 PV gate 需要 MIN_PDIST。

---

## 8. 文件结构

```
experiment6/
├── shared/
│   ├── xas_local_dataset_v2.py       # ← 直接从 Exp4 拷贝,零改动
│   ├── shell_boundaries.pkl          # ← v6 关键,直接从 Exp4 step 2.5 输出复制 (387 MB,但只 load 一次)
│   ├── shell_integrity_report.json   # ← v6 新增,SA1 step1.0a 输出
│   ├── shell_n_atoms_hist.png        # ← v6 新增,SA1 step1.0a 输出的人工 review 图
│   ├── exp6_element_vocab.json       # ← SA1 build phase 生成,见 §4.1(c)
│   ├── min_pdist_calibration.json    # ← v5 新增,SA1 step1.0 RDF 分析后写入,后续冻结
│   ├── min_pdist_rdf_hist.png         # ← v5 新增,SA1 step1.0 输出的 RDF 直方图,人工 review 用
│   ├── spectrum_tokenizer.py          # ← 改自 Exp4 SpectrumEncoder,~10 行改动
│   ├── transformer.py                 # ← 直接从 DETR 拷贝
│   ├── matcher.py                     # ← 改自 DETR matcher.py,~30 行
│   ├── criterion.py                   # ← 改自 DETR SetCriterion,~50 行 + v4 加 compute_repulsion_hinge ~30 行 + **v7 加三件套 loss (compute_pairwise_min_penalty + compute_shell_distance_loss + compute_shell_count_loss) ~150 行**
│   ├── detr_xas.py                    # ← 新写,主模型类,~200 行 (含 N_NEIGHBOR_TYPES / NO_OBJECT_IDX 常量声明)
│   ├── eval_metrics.py                # ← §7.1 五公式实现 + Exp4 评估 glue,SA1 直接复制锁定公式
│   └── composite_score.py             # ← v4 新增,§7.2 CPS 全套公式 + baseline 计算辅助函数,~250 行 (v6: shell-aware,从 shell_boundaries.pkl 加载 GT 边界)
├── step1/
│   ├── step1.0a_shell_integrity_check.py   # ← v6 新增,扫 shell_boundaries.pkl → shell_integrity_report.json (~10 min)
│   ├── step1.0_rdf_analysis.py        # ← v5 新增,扫 train pair RDF → min_pdist_calibration.json (~30 min)
│   ├── step1.1_build_vocab.py         # ← SA1 一次性,从 train_samples_v2.csv 建 center+neighbor vocab
│   ├── step1.2_recompute_exp4_setlevel.py  # ← SA1 一次性,从 Exp4 predictions_val.pt 重算 Set-Level baseline (~30 min,见 §10.1)
│   ├── step1.3_baseline_cps.py        # ← v4 新增,**v7 必跑** (升级自 v6 "可选"): 跑 random + Exp4 (must) + Exp5' (若已完成) 的 CPS baseline,产出 baseline_cps.json — 直接套 v6 shell-aware CPS 公式,Exp4 ckpt 用 predictions_val.pt 不重新 sample
│   └── step1.4_smoke_test.py          # ← 5 样本 forward + matcher + loss (含 v4 repulsion + **v7 三件套 6 项 loss**) 跑通 + CPS 计算
├── step2/
│   └── step2.1_train.py               # ← 训练主脚本,~250 行(DDP, AMP, ckpt, logging,§7.1 + §7.2 + 附录B.5 全指标 + v6 shell logging)
├── step3/
│   └── step3.1_eval.py                # ← val/test 评估,完全调用 eval_metrics.py + composite_score.py 锁定公式
├── step4/
│   └── step4.1_holdout.py             # ← Holdout 检验,与 Exp4 holdout CPS 直接对比
└── EXP6_PROPOSAL_v7.md                # ← 本文档
```

**新写代码总量估算**: ~750 行(detr_xas.py 200 + train 250 + matcher/criterion 改 80 + repulsion 30 + composite_score 150 + smoke + baseline 100 + eval glue 100 — 比 v3 多 ~170 行,因为加了 composite score 模块)
**拷贝代码总量估算**: ~700 行(transformer 300 + Exp4 dataset/eval 复用 ~400)

对比 Exp4 的工程量(几千行扩散代码),**Exp6 工作量减少 50%+**。

---

## 9. 时间表

| Phase | 内容 | 时间 | 输出 |
|---|---|---|---|
| Phase 0 | 跑通 DETR Colab + 阅读 4 个核心文件 + 阅读 Exp5' 三件套实现参考 | 0.5 天 | 个人理解笔记 |
| Phase 1 | 拷贝 + 改造 4 个 DETR 文件,实现 detr_xas.py + composite_score.py + shell integrity check + RDF analysis + **三件套物理 loss 实现** (v7) | 4 天 | smoke_test 通过,shell_integrity_report.json + baseline_cps.json (含 Exp4 ckpt 物理对照,v7 必跑) + min_pdist_calibration.json 完成 |
| Phase 2 | 训练脚本,跑 5 epoch sanity check + **6 项 lambda tuning** (v7,从 4 项增至 6 项 caveat 见 §5.3) | 2 天 | val loss 稳定下降,所有 v6/v7 logging 指标正常,**6 个 loss 项比例合理** |
| Phase 3 | 完整训练(双 4090 估计 1.5 天/300 epoch) | 1.5-2 天 | val CPS 显著优于 max(Exp4_CPS, Exp5'_CPS),RMSD < 1.5 Å,PV pass rate > 95%,pred_outside_shells_ratio < 0.3 |
| Phase 4 | val/test/holdout 评估 + 与 Exp4 + Exp5' 三方对比 + **用户物理 sanity 必经** (v7) | 1.5 天 | EXP6_FINAL_REPORT_v1.md (含用户物理 sanity 通过 attestation) |
| **总计** | | **~9.5 天** (v3 ~6 → v4 +1 lambda tuning → v5 +0.5 RDF → v6 +0.5 shell integrity → v7 +1.5 三件套实现 + 6 lambda tuning + 用户 sanity 必经) | |

(对比 Exp4/5 各自的 2-3 周工程,这是数量级的减少)

---

## 10. 验收标准

**v4 重大变更**: 主验收指标从 Hungarian RMSD 改为 **Composite Physical Score (CPS, §7.2)**。

**v7 重大变更 1 (verdict 框架)**: 验收对照对象从 "vs Exp4" 升级为 "vs Exp4 + vs Exp5'"。Exp5' 是与 Exp6 v7 同期的 diffusion 范式 + 三件套物理 loss 实验,真正的对照应该是 "Exp6 v7 (transformer + 三件套) vs Exp5' (diffusion + 三件套)"。Exp4 仅作为 "无三件套" 的历史 baseline。

**v7 重大变更 2 (用户物理 sanity 必经)**: MA5 EXP5_v2 final report §6.7 教训 — "用户对实验数据的物理直觉远比 metric 算法可靠"。Exp5 v2 训完后所有自动指标看起来 OK,直到用户 2026-05-01 物理统计才发现 95% min_d 违反。Exp6 v7 必须把 "用户独立物理 sanity 统计" 加为 verdict 必经一步,**不允许纯靠自动 CPS declare success**。

**阈值采用 baseline-relative 框架**:proposal 阶段不硬定阈值,SA1 阶段按 §7.2.5 三任务跑完后回填。**v7 起 SA1 step1.3_baseline_cps 必须强制运行 Exp4 best ckpt 物理对照**(v6 是"可选",v7 升级为"必跑"):
- Exp4 ckpt 路径: `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` (依据 MA5 final report §3.3)
- 直接用现有 Exp4 `predictions_val.pt` 套 Exp6 v6 shell-aware CPS 公式,不需要重新 sample
- 估计耗时 ~30 min,不影响时间表

| 区间 | 定义 |
|---|---|
| 通过(§10.1) | val CPS ≥ max(Exp4_CPS, Exp5'_CPS) + 0.05 absolute,且 holdout CPS ≥ max(Exp4_CPS, Exp5'_CPS) absolute |
| 部分成功(§10.2) | val CPS ≥ Exp5'_CPS 但 < Exp5'_CPS + 0.05 |
| 失败(§10.3) | val CPS < Exp5'_CPS,即 Exp6 transformer 范式不如 Exp5' diffusion 范式 |

### 10.1 通过(进入 Exp7)

**自动指标 (必须全过)**:
- **val CPS ≥ max(Exp4_CPS, Exp5'_CPS) + 0.05**(absolute,SA1 baseline 跑完后填具体数;若 Exp5' 训练未完,先用 Exp4_CPS,等 Exp5' 完成后回填)
- **holdout CPS ≥ max(Exp4_CPS, Exp5'_CPS)**(防过拟合 buffer)
- **val PV pass rate ≥ 95%**(物理可行性,任何不满足这条 CPS 都被 hard gate 卡死,所以这是先决条件)
- **val Hungarian RMSD < 1.4866 Å**(副指标,严格 beat Exp4 holdout 1.4866 Å,继承 v3 标准)
- **Set-Level TypeAcc 报数即可,不预设阈值** — Exp4 没有 Set-Level baseline 可比(Exp4 的 0.197 是 position-by-position 虚假指标,见 ERRATA_2 §2)。SA1 在 smoke test 阶段从 Exp4 `predictions_val.pt` 直接重算 baseline,记录该数后回填
- no_object_ratio 收敛在 [1/20, 6/20] 之间
- `pairwise_violation_rate` 训练后期 < 5%
- `val_pred_outside_shells_ratio` < 0.3 (v6 新增)
- **val_shell_count_loss / val_shell_dist_loss 训练后期 < 训练初期的 30%** (v7 新增,确认三件套 loss 真起作用)

**用户物理 sanity 必经 (v7 新增,自动指标过不能跳过)**:
- SA-final-eval 阶段必须把 Exp6 best ckpt sample 输出 (predictions_val + holdout) 交给用户跑 **independent 物理统计**,内容至少包括:
  1. 全 sample min pred-pred distance histogram (整个 val + holdout 各 1 张图)
  2. shell-1 / shell-2 配位数实际分布 vs GT 分布对比 (各 1 张图)
  3. shell-1 / shell-2 距离实际分布 vs GT 分布对比 (各 1 张图)
  4. 随机抽 5 个 best CPS sample + 5 个 worst CPS sample 的 3D 可视化
- **用户独立判定 "物理上看着合理" 后,Exp6 verdict 才能 declare 通过**;若用户判定 "看着像 Exp5 v2 那种"(即使 CPS 自动指标过),verdict 降为 §10.2 部分成功并触发额外诊断
- 这一步是流程闸门,不是数字闸门,设计目的是防止"自动指标完备性盲区"再次发生

### 10.2 部分成功(可继续优化但不替代 Exp4 + Exp5')

- val CPS ∈ [Exp5'_CPS, Exp5'_CPS + 0.05) — Exp6 与 Exp5' 接近但优势不显著
- 或: val CPS ≥ Exp5'_CPS + 0.05 但 PV pass rate ∈ [80%, 95%] — CPS 高但物理可行性不够稳
- 或: 用户物理 sanity 不通过(无论自动指标如何)
- 此时建议继续 Exp7 的 cost_density=0 投影 ablation 等扩散侧实验,与 Exp6 横向对比

### 10.3 失败(放弃 transformer 方向)

- val CPS < Exp5'_CPS — transformer 范式在该任务上不如 diffusion 范式 + 三件套
- 或: val PV pass rate < 80% — 即便加了三件套物理 loss,transformer 仍无法保证物理可行性,说明 transformer 架构的归纳偏置与该任务不匹配
- 或: val Hungarian RMSD ≥ 1.6 Å (传统 RMSD 维度的失败 line,与 v3 一致)
- 失败的话,Exp7 不再做 transformer,转向 Exp5'' (Exp5' 沿用扩散架构 + ablation 三件套各项贡献) 或 Exp7 + cost_density=0 投影 ablation

---

## 11. 风险与回退方案

### 11.0 术语命名约定 ⚠️

**Exp6 整个 proposal、SA handoff、训练日志、final report 严格遵守以下命名分离**:

| 术语 | 含义 | 性质 |
|---|---|---|
| `query_degeneracy` / `query_pile-up` | DETR 已知早期现象: 20 个 query 输出位置雷同 | **良性**,通常 30 epoch 后散开 |
| `pred_collapse` | Exp4/5 的 hard sample 上预测原子塌缩到中心(`_density_loss` attractive prior 副作用) | **失败模式**,见 ERRATA_2 §1.3 塌缩根因 |
| `repulsion_degradation` (v4 新增) | Exp6 lambda_rep 过大导致 model 把 valid query 都输出 no_object 来 trivially 满足约束 | **失败模式**,见 §11.1 风险 4 |

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

**风险 4 (v4 新增): Repulsion hinge loss 退化为塌缩剂**
- 表现: lambda_rep 过大 → model 把所有 valid query 输出 no_object 来 trivially 满足约束(因为 no_object 不参与 repulsion 计算)→ no_object_ratio 飙升 > 0.5
- 根因: hinge loss 的优化最便宜路径是"消灭 valid query"而不是"把 query 挪开"
- 监控: SA1 必须 logging `pairwise_violation_rate` + `no_object_ratio`,二者同时升高 = 退化信号
- 应对: lambda_rep 下调到 0.5 或 0.3;或对 no_object 这个类的预测增加 penalty(让 trivial solution 不便宜)
- 与 errata 2 区别: 这是 v4 新引入的 loss 项可能的失败模式,**不是 errata 2 的 attractive prior 退化**(那个是 pred_collapse 到中心,v4 这个是 pred 全部 → no_object)。建议命名 `repulsion_degradation` 与 `pred_collapse` 区分

**风险 5 (v4 新增 + v5 修订 + v6 重新表述): CPS hard gate (PV) 过严导致训练后期 CPS 频繁 = 0**
- 表现: val PV pass rate 长期 < 50%,即便 RMSD 不错,CPS 也卡在 ~0.1-0.2
- 根因: 训练初期 query_pile-up 时几乎所有 sample PV=False,后期若 lambda_rep 太弱 PV 也救不回来
- 监控: `val_pv_pass_rate` 应在 epoch 50 后稳定上升至 > 80%
- 应对: 优先调 lambda_rep ↑(0.5 → 1.0 → 2.0);次选调整 TOL_SHELL_BAND (默认 0.1 Å,可微调到 0.15 Å,但不允许超过 0.2 Å,否则 shell 之间会重叠)
- **v5 修订**: MIN_PDIST 由 SA1 step1.0 RDF calibration 后写入 `min_pdist_calibration.json` 并冻结,**SA1 calibration 完成后所有 SA 不许再改**
- **v6 修订**: 不再有"调 SHELL1_RANGE 边界"这个 lever,因为 v6 用 GT shell 边界

**风险 6 (v6 新增 + v7 修订): pred 原子大量落在 GT shell 1/2 之外 → CPS 子项 c1/c2 永远低**
- 表现: `val_pred_outside_shells_ratio` 长期 > 0.5,即便 PV 通过,CPS 也低
- 根因: model 学到了"原子在 box 内"但没学到"原子按物理壳层分布"
- 监控: `val_pred_outside_shells_ratio` 应在 epoch 100 后稳定下降至 < 0.3
- **v7 修订应对**:
  1. **首选 (v7)**: 增大 `lambda_sdist` (0.5 → 1.0 → 2.0) — 三件套已直接监督 shell 距离,加大权重应直接生效
  2. 增大 `lambda_scount` (0.2 → 0.5) — shell count loss 加大也能间接拉 pred 进 shell
  3. 增大 lambda_pos (5.0 → 8.0): 让 model 更精准回归到 GT 配对原子位置
  4. 训练时间延长(300 → 500 epoch): DETR + 多 loss 收敛慢
- **v6 旧应对** "Exp7 加 shell-aware 监督" 在 v7 被取消 — v7 已加,见 §5.2.2 / §5.2.3
- **不接受的应对**: 改 TOL_SHELL_BAND 到 > 0.3 Å 让 pred 凑数 (这是评估端造假)

**风险 7 (v6 新增): GT 仅 1 shell 的 sample 占比超预期 → 影响 CPS dataset-level mean**
- 表现: SA1 step1.0a 报告 `frac_only_1_shell > 10%` 触发 sanity check 失败
- 根因: 数据本身,或 Exp4 step 2.5 gap_threshold 选得不合适
- 应对: SA1 raise → 用户决策(可选: 接受并把 sanity 阈值放宽到 15%,或重跑 Exp4 step 2.5 用更小 gap_threshold)
- 不修 Exp6 proposal,这是数据问题

**风险 8 (v7 新增): 6 项 loss 项相互干扰,总优化失衡**
- 表现 (按严重度):
  1. 一项 loss dominant (> 0.5 × loss_cls 长期),其他项学不到 — Exp5 v2 时代的 `_density_loss` 类型问题
  2. shell_dist 和 shell_count 同向但梯度互相抵消 — 例如 model 把原子放在 shell 1 边缘,shell_dist 罚 (距 shell 中点远) + shell_count 暗罚 (边缘原子被 sigmoid 部分计数) → 优化方向不清
  3. pairwise_min 和 repulsion_hinge 重复监督 → 相当于 lambda_rep 加倍,可能过强
- 监控: 每 epoch logging 6 项 loss 实际数值 + 比值,目标比值见 §5.3
- 应对 (按顺序):
  1. **首选**: SA1 在 5 epoch sanity 后,根据实际比值调整 lambda 权重。以 loss_cls 为单位 1.0 基准,目标比值 cls:pos:rep:pmin:sdist:scount = 1.0 : 1.0 : 0.1 : 0.05 : 0.5 : 0.3
  2. 若 shell_dist 和 shell_count 干扰严重 → 暂时降低 lambda_scount 至 0.1 (shell_count 是 secondary monitor,shell_dist 才是主要)
  3. 若 pairwise_min 和 repulsion_hinge 重复 → 取消 repulsion_hinge (lambda_rep = 0),保留 pairwise_min — pmin 的 sigmoid 形式比 hinge 的 max(0,..) 形式更适配训练后期
- **必须 logging** (新增 §附录B.5 logging 项): `loss_breakdown_per_epoch` 必报 6 项 + 总和,SA1 train.py 实现

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
3. **Phase 1 第一个产出**: 按 §8 顺序 — `step1.0a_shell_integrity_check.py` (v6 新增,shell_boundaries.pkl 完整性) → `step1.0_rdf_analysis.py` (v5 新增 MIN_PDIST 校准) → `step1.1_build_vocab.py` → `step1.2_recompute_exp4_setlevel.py` → `step1.3_baseline_cps.py` → `step1.4_smoke_test.py`。**最终 `step1.4_smoke_test.py` 用 5 个样本跑通 forward + matcher + loss(含 repulsion)反向 + per-sample CPS 计算(用 shell_boundaries.pkl 的 GT 边界)**。在这通过之前**禁止**写训练脚本
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
6. **禁止 implicit attractive prior,允许 explicit data-driven physical loss + repulsive hinge**(v7 重大修订,弃用 v4-v6 的 attractive vs repulsive 二分法): 严格按 §5.1.2 v7 三类划分执行。
   - **禁止**: implicit attractive prior — loss 项的 target 值/分布是作者直觉硬编码 (e.g. `_density_loss` 朝原点压;距离 prior "Fe-O 应该 2.0 Å" 硬编码;假设原子分布对称的 attractive loss)
   - **允许且已加入** (v4): repulsive hinge loss (`compute_repulsion_hinge`,无方向偏置)
   - **允许且已加入** (v7 新): explicit data-driven physical loss — `compute_pairwise_min_penalty` (target = MIN_PDIST,来自 SA1 RDF 校准) + `compute_shell_distance_loss` (target = GT shell 中点) + `compute_shell_count_loss` (target = GT shell n_atoms)。这三项都从 `shell_boundaries.pkl` 或 RDF 校准取 target,不含作者直觉
   - **判断标准 (v7)**: loss 项的 target 值是否完全来自数据 (GT 或校准) → 允许;target 值有任何作者主观选择 → 禁
   - **v7 thesis 修订基础**: MA5 EXP5_v2 final report §6.2 / §6.6 / §6.7 用 95% 物理违反率证明"训练目标没要求的事,模型不会自己学"。坚持 v6 thesis 等于主动重蹈覆辙。Exp6 v7 不再做 thesis-clean 验证,转为架构对照 (transformer vs diffusion 同等物理监督下谁优)
7. **禁止**:重新引入 TypeClassifier head(Exp3 双重证伪 + Exp5 三重证伪 + 自然分类高斯分布的根本问题)
8. **必须**:Holdout 数字直接与 Exp4 对比,不能改 holdout 划分
9. **必须**(v5 新增): SA1 step1.0 完成后 MIN_PDIST 冻结。后续任何 SA(包括训练 agent)看到 PV pass rate 低不允许调 MIN_PDIST,只能调 lambda_rep / lambda_pmin。这避免"训练中途调阈值掩盖问题"反 pattern
10. **必须**(v6 新增): CPS 评估必须用 `shell_boundaries.pkl` 的 GT shell 边界,**不允许**回退到全局固定 SHELL1_RANGE/SHELL2_RANGE(v4/v5 的设计已被弃用)
11. **禁止**(v6 → v7 修订): ~~在 loss 端任何形式的 shell-aware 监督信号~~ — **v7 完全推翻** v6 这条。MA5 EXP5_v2 final report §6.2 已证明"shell ground truth 必须进训练 loss"。v7 已加 `compute_shell_distance_loss` 和 `compute_shell_count_loss` 两项 shell-aware 监督,所有遵循 §5.1.2 explicit data-driven 标准
12. **必须**(v7 新增): SA-final-eval 阶段必须把 Exp6 best ckpt sample 输出 (val + holdout predictions) 交给用户跑 independent 物理 sanity 统计 (min_d histogram + shell count/distance 分布对比 + 5 best/worst 3D 可视化),用户独立判定后才能 declare verdict。详见 §10.1 末尾用户物理 sanity 必经清单。**自动指标完备性盲区** (Exp5 v2 95% 违反却 metric 看着 OK) 必须由用户独立物理 sanity 兜底,流程闸门不是数字闸门
13. **禁止**(v7 新增,MA5 EXP5_v2 §6.3 lesson): R_max / shell 边界**禁止任何 fallback** (per-sample lookup 必须 hit,miss 时 raise 不是退回全局值)。SA1 dataloader 阶段 assert 所有 train/val/test/holdout sample 都在 `shell_boundaries.pkl` 里;若有 missing 必须 skip (与 Exp4 v2 dataset 一致),不允许"miss 就用全局默认值"
14. **禁止**(v7 新增,MA5 EXP5_v2 §5.1 工程债 lesson): 不沿用 Exp5 v2 `step5_2_compute_metrics.py` 的投影 ablation 代码 (R_max=5.5 fallback bug 锚点)。Exp6 v7 主指标用 `step1.3_baseline_cps.py` + `composite_score.py` (shell-aware),不依赖 step5_2 的旧实现
15. **必须**(v7 新增,MA5 EXP5_v2 §3.3 强建议): SA1 step1.3_baseline_cps **必须** 跑 Exp4 best ckpt 物理对照 (从 v6 的"可选"升级为"必跑")。Exp4 ckpt: `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt`;直接用 Exp4 `predictions_val.pt` 套 v6 shell-aware CPS 公式,~30 min。**目的**: 给 Exp6 v7 verdict 一个 v6 之前缺失的 "Exp4 物理 baseline" 锚点

---

*Main Agent 6 撰写,2026-04-29 → v6 update 2026-04-30 → v7 update 2026-05-01*
*v1 (initial) → v2 (MA5 round 1, 6 mods) → v3 (MA5 round 2, 4 mods + mod 4 internal inconsistency resolution) → v4 (用户 round 1, 物理约束 + composite shell-based eval) → v5 (用户 round 2, MIN_PDIST data-driven calibration) → v6 (用户 round 3, GT shell boundaries 替换全局固定 shell 范围) → v7 (吸收 MA5 EXPERIMENT5_FINAL_REPORT_v2 lessons learned, thesis 重写 + Exp5' 三件套物理 loss 移植 + 用户物理 sanity 必经)*
*基于: EXP4_FINAL_REPORT_ERRATA_2.md(扩散框架根因诊断 + collapse 命名约定 + attractive vs repulsive 区分基础 + 跨实验复用隐式偏置教训)+ EXPERIMENT2_FINAL_REPORT.md(评估指标体系)+ Exp3 总结(.detach 教训和虚假指标教训)+ Exp5 v1 总结(公式自由发挥教训)+ **EXPERIMENT5_FINAL_REPORT_v2.md (MA5 撰写 2026-05-01,8 条 Lessons Learned + 95% 物理违反率硬证据 → v7 thesis 重写依据)** + Exp4 step 2.5 shell_boundaries.pkl(128,382 sample 的 sample-specific shell 划分)+ 用户 2026-04-29 round 1 物理可行性观察 + 用户 2026-04-29 round 2 数据驱动校准要求 + 用户 2026-04-30 round 3 shell 划分根因发现 + 用户 2026-05-01 round 4 选项 B "MA5 教训全改" 决议 + facebookresearch/detr archive 2024-03-12(模型基础)*
