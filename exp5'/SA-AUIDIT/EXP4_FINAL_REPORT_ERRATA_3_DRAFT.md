# EXP4_FINAL_REPORT_ERRATA_3_DRAFT.md
# Exp4/5 系列勘误 #3 — L=6 虚拟晶格 fold artifact + pairwise loss 设计缺陷

> **撰写者**: SA-EXP5'-STEP1-AUDIT
> **日期**: 2026-05-02
> **触发**: SA-EXP5'-STEP1 在 §1.5 前主动自查 dataset 输出物理性,发现 fold artifact,
>          SA-EXP5'-STEP1-AUDIT 完成 A1-A3 硬证据核实后出具本 errata 草稿
> **状态**: DRAFT,待 Exp5'-MA review + 决议 §5(路径 A 具体 L 值)
> **不重写原文档**,独立 errata 存档
> **影响范围**:
>   - EXPERIMENT4_FINAL_REPORT.md §7.2 / §10
>   - EXP4_FINAL_REPORT_ERRATA_2.md §1.4(RMSD 1.49 归因)
>   - EXP5_PRIME_PROPOSAL.md §2.1(`_pairwise_min_distance_penalty` 设计前提)

---

## §1 SA-EXP5'-STEP1 自查发现(物理 sanity SOP 来源)

SA-EXP5'-STEP1 在完成 §1.4(model + yaml 改动)后、§1.5 之前,对 dataset 输出的
ground truth `frac_coords` 做了物理 sanity 检查,发现:

- **1% 样本两两 frac 距离 < 0.01**(对应 cart < 0.06 Å,原子近乎重合)
- **94% 样本存在至少一对 frac 两两距离 < 0.25**(对应 cart < 1.5 Å)
- 100 样本诊断:64% 是 fold artifact,36% 是 fold 前后均重叠的真 overlap

SA1 自查流程和发现被记录在 `EXP5_PRIME_STEP1_HANDBACK_PARTIAL.md`。
本 errata 对 64% fold artifact 做精确根因分析,对 36% 真 overlap 做来源鉴定。

**这条 SOP 写进 ExpN 不变量**:任何 dataset `__getitem__` 改动后,输出的
ground truth coordinates 必须在原始物理量纲(cartesian Å)下验证两两距离分布,
不允许仅在 frac 空间验证。

---

## §2 fold artifact 精确根因

### 2.1 根因代码

`xas_local_dataset_v2.py`(Exp4/5/5' 共用,三代继承),`__getitem__` 末段:

```python
relative_cart = coords_top - center_cart[None, :]  # (20, 3) 真实 cartesian 位移 Å
frac = relative_cart / L_VIRTUAL                    # / 6.0 → frac ∈ (-∞, +∞)
frac = frac - np.round(frac)                        # ← min-image wrap → [-0.5, 0.5]
```

### 2.2 几何机制

`frac - np.round(frac)` 对虚拟晶格执行周期性折叠:超过 `L_VIRTUAL/2 = 3 Å` 的原子被
映射到 box 的"另一侧"(符号翻转,绝对值从 >0.5 变为 <0.5)。

**典型触发案例**:两个真实邻居分别在中心两侧,距离各自 > L/2:

```
邻居 A: relative_cart = [+3.2, 0, 0] Å
  frac = +0.533  →  fold → -0.467   (apparent cart: -2.80 Å)

邻居 B: relative_cart = [-3.2, 0, 0] Å
  frac = -0.533  →  fold → +0.467   (apparent cart: +2.80 Å)

真实 A-B 距离 = 6.4 Å
frac 差 = 0.467 - (-0.467) = 0.934
min-image of 0.934 → |0.934 - 1| = 0.066
pairwise cart (as seen by loss) = 0.066 × 6 = 0.40 Å  ← 严重虚假违反
```

**触发条件**:邻居对满足"两个原子分别在中心两侧且各自距中心 > L/2 = 3 Å"。
等价于:真实两两 cartesian 距离 > L = 6 Å,且中心在两者连线上(近似)。
CUTOFF_R = 10 Å 的搜索半径导致大量邻居对满足此条件。

### 2.3 对三件套 loss 的影响

`_pairwise_min_distance_penalty` 对 pred_frac_coords 做 min-image pairwise 计算,
threshold = 1.5 Å。训练初期 pred 坐标来自 Tweedie x0_hat,继承 true frac_coords
的 fold 分布,因此同样包含大量虚假近距离对。**loss 会惩罚这些虚假违反**,试图
把本来就是真实结构的预测推离(错误方向的梯度)。

64% fold artifact 意味着每个 batch 中 ~64% 的"违反"都是虚假的。训练信号严重污染。

---

## §3 shell_boundaries.pkl 干净性硬证(A2 audit 结论)

**结论**: shell_boundaries.pkl 的 `distances` 字段是 pymatgen `get_neighbors(r=10)`
返回的 cartesian Å 距离,在构建时**未经过 frac fold**。

硬证:
- fold_suspect 样本 `mp-555067__mp-555067-EXAFS-C-K` (碳化合物):
  - `distances[0:2] = [1.334, 1.444] Å` → C-C 化学键(石墨烯 ~1.42 Å,苯环 ~1.39 Å),物理合理
- 全局统计(前 2000 样本):
  - `fraction < 1.5 Å: 0.0006`(0.06%,全部为轻元素真实短键)
  - `fraction < 0.5 Å: 0.0000`

**推论**: `shell_boundaries.pkl` 的 `true_shell1_d_mean` 等 5 字段是真实物理距离,
inject 进训练是正确的。A2 CLEAN。

---

## §4 36% 真 overlap 根因(A3 audit 结论)

36% "fold 前后均重叠"的样本中,真实两两 cartesian 距离在 fold 前就已 < 1.5 Å,
来源是**轻元素真实化学键**。

典型来源:
- C-C 键 ~1.34-1.54 Å(碳化合物、有机金属)
- B-N 键 ~1.45 Å
- C-O 键 ~1.23-1.43 Å
- N-H 键 ~1.01 Å(但 H 通常不在 FEFF 邻居列表)

这不是 pymatgen 行为 bug,也不是 dataset 后处理引入,是物理现实。

**对 1.5 Å 阈值设计的影响**: 1.5 Å 对过渡金属(Fe-O ~1.95 Å,Ni-O ~2.0 Å)是
合理的最小距离。但对 88 元素数据集中的轻元素而言,1.5 Å 过于保守,会把合法的
C-C 键误判为违反。这是 proposal §2.1 设计时的遗漏。

---

## §5 errata 2 §1.4 重归因

### 5.1 原归因(errata 2)

> RMSD 1.49 不是物理顶住,是三层评估保护机制顶住(L=6 box + min-image + Hungarian)。

### 5.2 修正后归因

RMSD 1.49 由**三层叠加**决定:

**层 1(errata 2 已知)**: 评估保护机制(Hungarian min-image RMSD)提供地板。

**层 2(errata 3 新增)**: Fold-distorted training target 造成的**表示上限**。
外壳层邻居(真实距离 3-10 Å)经 fold 后,在 frac 空间 apparent cart 距离最大为
L/2 = 3 Å(因为 fold 后 |frac| ≤ 0.5)。模型学习的是这个 fold 后的分布,因此
**预测坐标的"径向"范围被压缩在 [-L/2, L/2] = [-3, 3] Å**,RMSD 上限 ≈ L/2 × √(1/3)
≈ 1.73 Å(均匀分布期望)。实际 RMSD 1.49 在此理论上限以内,合理。

**层 3(errata 2 _density_loss 主犯)**: `_density_loss` 进一步把预测坐标推向原点,
使 RMSD 低于 fold 几何上限。

**三层共同决定 RMSD 卡在 ~1.49 Å**。errata 2 的结论不变,但加入层 2 使归因更完整。

---

## §6 影响传播链

**正向传播(从源到下游)**:

```
Exp2 (Fe-only, L=6)
  → Fe-O ~2 Å < L/2=3 Å → fold 几乎不触发 → 训练数据干净 → 结果合理 ✅

Exp4 (88 元素, L=6, CUTOFF_R=10 Å 沿用)
  → 大量邻居 > 3 Å → fold artifact 进入 training ground truth
  → 模型学习 fold-distorted 分布
  → RMSD 1.49 部分由 fold 表示上限决定(层 2)
  → `_pairwise_min_distance_penalty` 未引入,问题未被发现 ⚠️

Exp5 v2 (沿用 Exp4 dataset)
  → 同 Exp4,且 min_d gate pass 5-11% 被归因为"训练无物理约束"
  → 真正根因是 fold artifact 使 ground truth 本身已有 94% 两两 < 1.5 Å 对 ❌

Exp5' (三件套 loss 基于 fold-distorted frac_coords)
  → `_pairwise_min_distance_penalty` 惩罚 64% 虚假违反
  → loss 信号严重污染,训练方向错误 ❌
```

**说明 Exp2 为何合理**: `L=6, Fe-O ~2 Å`,最近邻在 frac ~0.33,fold 无效。
88 元素扩展后 CUTOFF_R=10 保持不变,但邻居距离跨度 0-10 Å,fold 大量触发。
**这是从 Exp2 设计扩展到 88 元素时的遗漏**,不是 bug,是设计边界未更新。

---

## §7 ExpN 不变量级 Lessons(本 errata 贡献)

### 7.1 dataset ground truth 的物理验证必须在 cartesian Å 下做

任何 dataset 改动(或初次设计)后:

```python
# MUST-DO: cartesian sanity check BEFORE使用 frac_coords 进训练
frac = dataset[i]['frac_coords']           # (N, 3)
cart = frac * L                            # (N, 3) Å
d_pairs = pairwise_distances(cart)          # (N, N)
assert d_pairs.min() >= MIN_BOND_LENGTH, f"overlap found: {d_pairs.min():.3f} Å"
```

其中 `MIN_BOND_LENGTH` 应为 dataset 中心元素集合对应的物理最小值(0.7 Å 保守下限)。
**不允许仅在 frac 空间验证两两距离**。

### 7.2 fold-based 坐标系与 pairwise 距离 loss 的兼容性必须显式验证

`frac - round(frac)` 周期性折叠在 diffusion 训练中是正确操作,但 pairwise min distance
loss 需要在**折叠后的坐标系**中正确处理周期边界。当 L 较小(如 L=6)而搜索半径较大
(如 10 Å)时,fold artifact 会产生虚假近距离对。在启用 pairwise loss 前,必须
验证 ground truth 中该 loss 的"真实违反率"是否与物理预期一致。

### 7.3 L 值选取的物理约束

L_VIRTUAL 必须满足 `L/2 ≥ 最大期望邻居距离`,即:
- 若 CUTOFF_R = 10 Å,则 L ≥ 20 Å(避免任何邻居被 fold)
- 若 L 仍保持 6 Å(设计约束),则有效训练范围仅为 [0, 3] Å 的邻居,
  **必须明确文档化**,且不能使用以"全部 20 邻居正确空间位置"为前提的 loss 函数

---

## §8 路径 A 待决项(MA 决议)

以下决策 **SA-EXP5'-STEP1-AUDIT 不自行决定**,提交 Exp5'-MA:

### 8.1 L 值如何改

| 选项 | 描述 | 成本 | 风险 |
|---|---|---|---|
| A. L=6 → L=12 | 扩大 box,fold artifact 减少但不消失(CUTOFF_R=10 < L/2=6 ✓) | dataset cache 重建(大量 POSCAR 重新处理) | 训练分布改变,所有 Exp4/5 baseline 不可比 |
| B. L=6 → L=20 | 完全消除 fold artifact(CUTOFF_R=10 < L/2=10 ✓) | 同 A,更大 | 同 A |
| C. 保留 L=6,修 loss | `_pairwise_min_distance_penalty` 加 fold-aware 修正:检测 fold artifact 对并跳过 | 仅改 loss 函数 | 修正逻辑复杂,需要验证 |
| D. 保留 L=6,降 CUTOFF_R | CUTOFF_R 10→3 Å,仅取 3 Å 内邻居(N_NEIGHBORS 可能不足 20) | dataset 重建 | 丢失外壳层信息 |

**Exp5'-MA 决策需考虑**:4 个锁定 md5 代码不能动(但 dataset 重建不会改 md5 锁定文件
本身,只改 cache pkl——若 MA 选 A/B/D,需确认 cache 重建不违反红线)。

### 8.2 `_pairwise_min_distance_penalty` 的阈值是否需要元素分类

轻元素(C、B、N、O)的最小合理键长为 ~1.2 Å,而非 1.5 Å。
若 MA 选择 C(保留 L=6,修 loss),阈值也需要 element-aware 设计。

---

## §9 与 errata 1、errata 2 的关系

| | 内容 | 状态 |
|---|---|---|
| errata 1 | Phase 6.5 状态修正(SA4-续 2 报 5/5 PASS 未独立验证) | 与本 errata 无交叉 |
| errata 2 | `_density_loss` 塌缩根因 + Exp3 真实历史 + 方向 menu 调整 | **§1.4 归因被本 errata §5 扩充** |
| **errata 3(本文)** | L=6 fold artifact + pairwise loss 前提缺陷 + RMSD 1.49 完整归因 | DRAFT,待 MA 审核 |

三份 errata **并列存档,不合并**。errata 2 的结论不被推翻,§5.2 的层 2 是新增。

---

*SA-EXP5'-STEP1-AUDIT 撰写,2026-05-02*
*基于 A1(grep 捕获所有 L=6/fold 操作)+ A2(shell_boundaries.pkl cartesian Å 验证)*
*+ A3(dataset fold 几何机制推导)三项 audit 证据。*
*§8 路径 A 决策全部留给 Exp5'-MA,本 SA 不决议。*
