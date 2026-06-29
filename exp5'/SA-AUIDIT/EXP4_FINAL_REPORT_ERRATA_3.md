# EXP4_FINAL_REPORT_ERRATA_3.md
# Exp4/5 系列勘误 #3 — L=6 虚拟晶格 fold artifact + pairwise loss 设计缺陷

> **撰写者**: SA-EXP5'-STEP1-AUDIT
> **日期**: 2026-05-02
> **MA final 化**: Exp5'-MA 同日审核,三处修改后定 final
> **触发**: SA-EXP5'-STEP1 在 §1.5 前主动自查 dataset 输出物理性,发现 fold artifact,
>          SA-EXP5'-STEP1-AUDIT 完成 A1-A3 硬证据核实后出具本 errata
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
不允许仅在 frac 空间验证。MIN_BOND_LENGTH 推荐 **0.7 Å**(H-H 物理下限 ~0.74 Å),
不是 1.5 Å。1.5 Å 是 Exp5' pairwise loss 的阈值,不是 dataset sanity 阈值,两件事不混。

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
- fold_suspect 样本 `mp-555067__mp-555067-EXAFS-C-K`(碳化合物):
  - `distances[0:2] = [1.334, 1.444] Å` → C-C 化学键(石墨烯 ~1.42 Å),物理合理
- 全局统计(前 2000 样本):
  - `fraction < 1.5 Å: 0.0006`(0.06%,全部为轻元素真实短键)
  - `fraction < 0.5 Å: 0.0000`

**推论**: `shell_boundaries.pkl` 的 `true_shell1_d_mean` 等 5 字段是真实物理距离,
inject 进训练是正确的。A2 CLEAN,不需要重建 pkl。

---

## §4 36% 真 overlap 根因(A3 audit 结论)

36% "fold 前后均重叠"的样本中,真实两两 cartesian 距离在 fold 前就已 < 1.5 Å,
来源是**轻元素真实化学键**:

- C-C 键 ~1.34-1.54 Å
- B-N 键 ~1.45 Å
- C-O 键 ~1.23-1.43 Å

这不是 pymatgen 行为 bug,也不是 dataset 后处理引入,是物理现实。

**对 1.5 Å 阈值的影响**:Exp5'-MA 决议保留全局 1.5 Å 阈值,不做 element-aware(见 §8)。
36% 真短键样本在新 L=20 dataset 下 ground truth 两两 cart 也 < 1.5 Å,
coord_loss 会把预测拉回真值,loss 平衡可接受。

---

## §5 errata 2 §1.4 重归因

### 5.1 原归因(errata 2)

> RMSD 1.49 不是物理顶住,是三层评估保护机制顶住(L=6 box + min-image + Hungarian)。

### 5.2 修正后归因(三层叠加)

**层 1(errata 2 已知)**: 评估保护机制(Hungarian min-image RMSD)提供地板。

**层 2(errata 3 新增)**: Fold-distorted training target 造成的**表示上限**。
外壳层邻居(真实距离 3-10 Å)经 fold 后,在 frac 空间的绝对值被压缩到 ≤ 0.5,
对应 apparent cart ≤ L/2 = 3 Å。模型学习的是这个 fold 后的分布,因此
**预测坐标的径向范围被限制在 ≤ L/2 = 3 Å 的硬几何上限**。
实测 RMSD 1.49 < 3 Å,进一步说明层 3 的 `_density_loss` 把它进一步压低。

**层 3(errata 2 _density_loss 主犯)**: `_density_loss` cost=0.2 继续把预测坐标推向原点,
使 RMSD 低于 fold 几何上限。

**三层共同决定 RMSD 卡在 ~1.49 Å**。errata 2 的结论不被推翻,层 2 是新增补充。

---

## §6 影响传播链

```
Exp2 (Fe-only, L=6)
  → Fe-O ~2 Å < L/2=3 Å → fold 几乎不触发 → 训练数据干净 → 结果合理 ✅

Exp4 (88 元素, L=6, CUTOFF_R=10 Å 沿用)
  → 大量邻居 > 3 Å → fold artifact 进入 training ground truth
  → 模型学习 fold-distorted 分布
  → RMSD 1.49 部分由 fold 表示上限(层 2)决定
  → `_pairwise_min_distance_penalty` 未引入,问题未被发现 ⚠️

Exp5 v2 (沿用 Exp4 dataset)
  → 同 Exp4,且 min_d gate pass 5-11% 被归因为"训练无物理约束"
  → 真正根因是 fold artifact 使 ground truth 本身已有 94% 两两 < 1.5 Å 对 ❌

Exp5' 原设计 (三件套 loss 基于 fold-distorted frac_coords)
  → `_pairwise_min_distance_penalty` 惩罚 64% 虚假违反
  → loss 信号严重污染,训练方向错误 ❌
  → STEP1-FIX 路径 B(L=6→20)修复
```

**关键发现**: Exp2 Fe-only 时 L=6 合理,是因为 Fe-O ~2 Å < L/2。88 元素扩展时
CUTOFF_R=10 沿用但 L 未调整,设计边界未更新。这不是 bug,是扩展时的遗漏。

---

## §7 ExpN 不变量级 Lessons

### 7.1 dataset ground truth 的物理验证必须在 cartesian Å 下做

任何 dataset 改动(或初次设计)后:

```python
# MUST-DO: cartesian sanity check BEFORE 用 frac_coords 进训练
frac = dataset[i]['frac_coords']           # (N, 3)
cart = frac * L                            # (N, 3) Å
d_pairs = pairwise_distances(cart)          # (N, N) Å
assert d_pairs[d_pairs > 0].min() >= 0.7, \
    f"unphysical overlap: {d_pairs[d_pairs>0].min():.3f} Å"
# MIN_BOND_LENGTH = 0.7 Å (H-H 物理下限 ~0.74 Å)
# 不是 1.5 Å — 1.5 是 Exp5' pairwise loss 阈值,不是 sanity 阈值
```

**不允许仅在 frac 空间验证两两距离**。

### 7.2 pairwise 距离 loss 启用前必须验证 ground truth 违反率

在启用 `_pairwise_min_distance_penalty` 类 loss 前,先在 ground truth 上统计
"有多少对违反阈值"。若违反率 >> 物理预期(如 >1%),则 loss 设计前提不成立,
需先修 dataset 或修阈值。

### 7.3 L_VIRTUAL 与 CUTOFF_R 的物理约束

L_VIRTUAL 必须满足:

```
L_VIRTUAL / 2 ≥ CUTOFF_R
```

否则 min-image wrap 会产生 fold artifact。L=6, CUTOFF_R=10 → 违反此约束。
L=20, CUTOFF_R=10 → 满足(L/2 = 10 ≥ CUTOFF_R = 10,边界等号成立,
若需严格无 fold 可取 L=21 留 0.5 Å 余量)。

---

## §8 路径 A 决议

**Exp5'-MA 2026-05-02 决议:选 B(L_VIRTUAL = 6 → 20)+ 保留全局 1.5 Å 阈值**。

排除 A(L=12):修一半,L/2=6 = CUTOFF_R,数值边界仍有折叠风险,既然 rebuild 一次直接修干净。

排除 C(保留 L=6 修 loss):只修 pairwise loss 一处局部症状,fold-distorted ground truth
仍喂进 coord_loss / type_loss / density_loss / shell_dist_loss 全链路,根因不动。

排除 D(降 CUTOFF_R):砍掉 shell-2 信号(~3-5 Å),`_shell_distance_loss` shell-2 项废,
违背 Exp5' 三件套设计意图。

选 B(L=20):完全消除 fold artifact,三件套 loss 物理意义直接复原,shell_boundaries.pkl 不动。
代价:dataset cache 重建 + Exp4/5/5' baseline 不可比(已知,接受)。

§8.2 element-aware 阈值:不做。36% 真短键样本 coord_loss 会拉到真值,loss 平衡可接受;
element-aware 增加工程复杂度,留 Exp5'' ablation。

---

## §9 STEP1-FIX 待改项(提前标注,供 STEP1-FIX launch note 参考)

基于 A1 grep 结果,需要更新 L_VIRTUAL 的文件:

| 文件 | 位置 | 改动 |
|---|---|---|
| `xas_local_dataset_v2.py` | L69 `L_VIRTUAL = 6.0` | → 20.0;加 §7.1 cartesian sanity check |
| `xas_local_datamodule_v2.py` | L56 `L_VIRTUAL = 6.0` | → 20.0 |
| `diffusion_w_type_xas.py` | L99 `L_VIRTUAL = 6.0` | → 20.0(三件套 loss 内用此常量) |
| `diffusion_xas.yaml` | 无 L_VIRTUAL hardcode | 不改 |
| `step5_2_compute_metrics.py` | `L=6.0` default args | → 20.0(评估代码跟随) |
| `step5_3_smoke_test.py` | `L = 6.0` | → 20.0 |
| `step6_visualize_v2.py` | `L = 6.0` | → 20.0 |
| `pick_samples_for_feff.py` | `L = 6.0` | → 20.0 |

**STEP1-FIX 的 dataset 改动解锁 4 个 md5 锁定**:AUDIT 任务完成后,
SA-EXP5'-STEP1-FIX 在修改前 cp `.bak_pre_step1_fix` 锚点,改后报告新 md5。

dataset cache 重建完成后,必须重跑 §1 cartesian sanity check(阈值 0.7 Å)确认
fold artifact 消除。

---

## §10 与 errata 1、errata 2 的关系

| | 内容 | 状态 |
|---|---|---|
| errata 1 | Phase 6.5 状态修正(SA4-续 2 报 5/5 PASS 未独立验证) | 与本 errata 无交叉 |
| errata 2 | `_density_loss` 塌缩根因 + Exp3 真实历史 + 方向 menu 调整 | §1.4 归因被本 errata §5 扩充,结论不推翻 |
| **errata 3(本文)** | L=6 fold artifact + pairwise loss 前提缺陷 + RMSD 1.49 完整归因 + STEP1-FIX 路径 B 决议 | **FINAL** |

三份 errata 并列存档,不合并。errata 2 §1.4 RMSD 归因以本文 §5.2 为准(三层版本)。

---

*SA-EXP5'-STEP1-AUDIT 撰写,2026-05-02*
*Exp5'-MA 审核,同日定 final(三处修改:§5.2 层 2 删不严谨数值;§7.1 MIN_BOND_LENGTH=0.7 Å;§8 append MA 决议)*
*A1-A3 硬证据:grep 全捕获 + shell_boundaries.pkl cartesian Å 验证 + fold 几何机制推导*
