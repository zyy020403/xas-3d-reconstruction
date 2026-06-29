# Experiment 6 Proposal: DETR-style Set Prediction
# XAS → 局部原子结构预测(替换扩散框架)

> **状态**: DRAFT v1
> **日期**: 2026-04-29
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
                        │ (B, 20, 89)    (B, 20, 3)           │  ← 89 = 88 元素 + 1 "no object"
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

**(c) 89 类(88 元素 + 1 "no object")**
- DETR 经典设计:不是每个 query 都要对应真实原子,允许 query 输出"空"
- 解决了 Exp4 的"硬性 20 邻居"问题——若实际只有 17 个原子在 cutoff 内,3 个 query 输出 "no object" 即可

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
# 未匹配的 query: 监督 type 为 "no object" (88 这一类),无 position loss

loss_cls = CE(pred_logits, target_classes_with_no_object)
loss_pos = L2(pred_pos[matched], gt_pos[matched])  # 只在匹配的对上算

total_loss = lambda_cls * loss_cls + lambda_pos * loss_pos
```

**初始超参**:
- `lambda_cls = 1.0`
- `lambda_pos = 5.0`(DETR 原版 box L1 是 5.0,我们 position L2 也用 5.0 起步)
- `no_object_weight = 0.1`(DETR 原版 0.1,降低空类权重,与 Exp4 元素分布偏斜匹配)

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
1. 5 样本 forward(): pred_logits shape == (5, 20, 89), pred_pos shape == (5, 20, 3) ✓
2. pred_pos 无 NaN,数值范围合理([-1, 1] 内即可,tanh 已约束) ✓
3. Hungarian matcher 在 5 样本上跑通,matching 输出合理(20 query 中 ~17 与真实原子配对,~3 输出 no_object) ✓
4. 第 1 个 batch loss 在 [10, 100] 范围(过低或 NaN 都说明初始化有问题)✓

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
- `query_diversity`: 20 个 query 输出位置的方差。若 < 0.01 说明所有 query collapse 到同一处(DETR 训练初期常见,通常 30 epoch 后散开)

---

## 8. 文件结构

```
experiment6/
├── shared/
│   ├── xas_local_dataset_v2.py       # ← 直接从 Exp4 拷贝,零改动
│   ├── spectrum_tokenizer.py          # ← 改自 Exp4 SpectrumEncoder,~10 行改动
│   ├── transformer.py                 # ← 直接从 DETR 拷贝
│   ├── matcher.py                     # ← 改自 DETR matcher.py,~30 行
│   ├── criterion.py                   # ← 改自 DETR SetCriterion,~50 行
│   ├── detr_xas.py                    # ← 新写,主模型类,~200 行
│   └── eval_metrics.py                # ← 直接从 Exp4 拷贝(Set-Level/Multiset/Hungarian)
├── step1/
│   └── step1.1_smoke_test.py          # ← 5 样本 forward + matcher + loss 跑通
├── step2/
│   └── step2.1_train.py               # ← 训练主脚本,~200 行(DDP, AMP, ckpt, logging)
├── step3/
│   └── step3.1_eval.py                # ← val/test 评估,直接复用 Exp4 框架
├── step4/
│   └── step4.1_holdout.py             # ← Holdout 检验,与 Exp4 holdout 数字直接对比
└── EXP6_PROPOSAL_v1.md                # ← 本文档
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

- val Hungarian RMSD **< 1.45 Å**(优于 Exp4 holdout 1.4866 Å)
- holdout Hungarian RMSD **< 1.50 Å**(防过拟合)
- Set-Level TypeAcc **≥ 0.20**(与 Exp4 0.197 持平或更好,因为 Exp4 是用虚假 position-by-position 算的,Exp6 用真实 Set-Level 应至少不差)
- no_object_ratio 收敛在 [1/20, 6/20] 之间(说明 query 学会了"哪些位置该输出空")

### 10.2 部分成功(可继续优化但不替代 Exp4)

- val RMSD 在 1.45 - 1.55 Å — 与 Exp4 持平,需进一步分析为何 transformer 没体现优势
- 此时建议继续 Exp7 的 cost_density=0 投影 ablation,在扩散框架下重测真实上限

### 10.3 失败(放弃 transformer 方向)

- val RMSD ≥ 1.6 Å — 说明 transformer 在该任务上不如扩散,需重新设计或回归扩散框架

---

## 11. 风险与回退方案

### 11.1 已知风险

**风险 1: DETR 训练初期不稳定**(著名问题)
- 表现: 前 50 epoch loss 几乎不降,query 全部 collapse 到同一位置
- 应对: 这是 DETR 已知问题,DETR 论文也描述过。DETR 原版 500 epoch 正是因此。我们用 300 epoch 起步,若 50 epoch 还没起色,加 lr warmup(前 10 epoch 从 1e-6 线性升到 1e-4)
- 回退: 若 100 epoch 仍 collapse,改用 Conditional DETR(收敛快 10×,见 dimiz51/DETR-Factory-PyTorch)

**风险 2: 88 类 cls loss 主导,position loss 学不到**
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

**No-object class**: 第 89 类(类别索引 88,如果元素是 0-87)。若一个 query 没匹配到任何 ground truth,它的目标类别就是 no_object,cls loss 用 weight 0.1 降低权重,position 不算 loss。

**Auxiliary decoding loss**: DETR 默认在每一层 decoder 输出都算一次 loss(共 6 次),加速收敛。我们沿用,代码 DETR 已实现,无需新写。

---

## 附录 B: 给 main agent 的具体指令

执行顺序:

1. **不要**碰 Exp4/5 的代码,Exp6 全部新建在 `experiment6/` 目录下
2. **第一件事**:用户 clone facebookresearch/detr 仓库到 `experiment6/_detr_reference/`,你阅读 4 个核心文件后再开始实现,**禁止**凭记忆写 transformer 架构
3. **Phase 1 第一个产出**:`step1.1_smoke_test.py`,用 5 个样本跑通 forward + matcher + loss 反向。在这通过之前**禁止**写训练脚本
4. **训练前必须打印**:模型参数总量(预计 < 50M,DETR 原版含 ResNet 是 41M,我们没 ResNet 应该更小),所有模块的 grad_required 状态
5. **Logging 必须包含**:除 train/val loss 外,还要每 epoch 打印 no_object_ratio 和 query_diversity——这两个是 DETR 训练健康度的关键指标
6. **禁止**:任何形式的辅助物理 loss(包括 density、shell、distance prior),Exp6 的 thesis 就是"不靠这些约束 transformer 也能学到结构"
7. **禁止**:重新引入 TypeClassifier head(Exp3 双重证伪 + Exp5 三重证伪 + 自然分类高斯分布的根本问题)
8. **必须**:Holdout 数字直接与 Exp4 对比,不能改 holdout 划分

---

*Main Agent 6 撰写,2026-04-29*
*基于: EXP4_FINAL_REPORT_ERRATA_2.md(扩散框架根因诊断)+ EXPERIMENT2_FINAL_REPORT.md(评估指标体系)+ Exp3 总结(.detach 教训和虚假指标教训)+ facebookresearch/detr(模型基础)*
