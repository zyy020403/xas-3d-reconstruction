# EXP5_PRIME_FINAL_REPORT_ERRATA_5.md
# Exp5' 系列勘误 #5 — Shell loss gap-based 设计缺陷(鸡蛋问题)+ Exp5'' 方向决议

> **撰写者**: Exp5'-MA(基于 SA-EXP5'-STEP3-SAMPLE hand-back v1+v2 + Exp5'-MA 全程 paper trail)
> **日期**: 2026-05-09
> **触发**: STEP3 sample + step5_3 复合分输出后 verdict 显示 shell-1 distance score 0.035 / shell-1 elem score 0.007 / pred_shell1_mean 6.3 Å vs true 2.3 Å,SA §10 根因初判 → Exp5'-MA 综合全 STEP1-3 paper trail 落 final 根因
> **本文档定位**: 继承 errata 1/2/3/4 格式,独立存档,与 STEP3 hand-back 配套
> **影响范围**:
>   - EXP5_PRIME_PROPOSAL.md §2.2 (`_shell_distance_loss`) + §2.3 (`_shell_count_loss`) 设计前提
>   - EXP4_FINAL_REPORT.md §10 方向 menu(errata 2 §3.2 已重排,本 errata 再补 Exp5'' 方向)
>   - Exp5'' proposal(本 errata §6 给方向决议,proposal 由 Exp5'-MA 后续写)

---

## §1 STEP3 sample verdict 摘要(本 errata 起点)

| 指标 | 实测 (val) | 阈值 | verdict |
|---|---|---|---|
| **gate_pass_rate** | 64.0% | ≥ 80% GREEN / ≥ 60% AMBER | AMBER 边缘 |
| **composite (step5_3 7 项)** | 0.0801 | ≥ 0.40 GREEN | RED ❌ |
| **shell-1 distance score** | 0.0346 | ≥ 0.50 GREEN | RED ❌ |
| **shell-1 elem score** | 0.0071 | — | 极低 |
| **collapse rate** | 0.00% | ≤ 1% | GREEN ✅ |
| **pred shell-1 mean radial dist** | 6.32 Å | true 2.27 Å | 严重偏外 |

**对比 Exp5 v2**:gate 5-11% → 64%(6-13× 改进),composite 0.005-0.011 → 0.08(10-16× 改进),shell-1 dist score 0.0000 → 0.035(从零到有但仍极低)。

**Exp5' 部分胜利**:
- ✅ Fold artifact 修复(errata 3 L=6→20)+ pairwise_min loss 生效:gate 改进 6-13×
- ❌ Shell loss(`_shell_distance_loss` + `_shell_count_loss`)实际未生效:shell-1 距离误差 4 Å,模型不知道 shell-1 该在 2-3 Å

本 errata 定位 shell loss 未生效根因。

---

## §2 错误声明: Shell loss 设计前提的鸡蛋问题

### 2.1 proposal §2.2 + §2.3 设计原意

**`_shell_distance_loss`** 设计逻辑(proposal §2.2):

```python
def _shell_distance_loss(pred_frac_coords, num_atoms,
                         true_shell1_d_mean, true_shell2_d_mean, has_shell2,
                         L=L_VIRTUAL, threshold_gap=0.1563):
    for sample i:
        coords_i = pred_frac_coords[i] * L          # cart Å
        radial = coords_i.norm(dim=1)                # center-to-atom dist
        sorted_d = radial.sort()
        gaps = sorted_d[1:] - sorted_d[:-1]
        boundaries = (gaps > threshold_gap).nonzero()
        if len(boundaries) >= 1:
            shell1_end = boundaries[0] + 1
            pred_s1_d_mean = sorted_d[:shell1_end].mean()
            loss += (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2
```

**核心假设**:pred_frac_coords 已具有 "壳层结构"(壳内紧密 + 壳间稀疏),gap 算法能切出 shell-1 / shell-2 → loss 衡量预测 shell 半径与真值差距 → 梯度回传修正 shell 位置。

### 2.2 实际机制(STEP1-FIX-C smoke + STEP2 训练曲线 + STEP3 sample verdict 三方证据)

模型从 random init 开始,**初始 pred_frac_coords 没有壳层结构** — 原子在 box 内随机散布,radial 分布近似单峰连续,**找不到 gap > 0.1563 Å 的边界**:

| 时刻 | shell loss 状态 |
|---|---|
| Epoch 0(random init)| sorted_d 几乎全连续,gaps 全 < 0.1563 → boundaries 空 → n_active = 0 → loss_shell_dist 用 `total_loss / max(n_active, 1)` 回退到 0 |
| Epoch 1-50(STEP2 早期)| 偶尔 sample 有 boundary → n_active 极少(~1-3 / batch=64)→ loss_shell_dist 数值 finite 但**梯度信号被 mean-over-batch 稀释为接近零** |
| Epoch 50+(STEP2 中后期)| pred 开始有粗略 shell 形态(因为 pairwise_min 把原子推开)→ gap 偶尔切出 boundary,但 boundary 位置随机(不一定在物理 shell 边界处)→ loss 数值计算正确,但**对应的真值 true_shell1_d_mean 是基于 pkl 真实物理 shell-1 的,与随机切出的 pred shell-1 是不同概念** → loss 仍降但不指向真物理修正 |

**鸡蛋问题精确定义**:loss_shell_dist 需要"pred 已有清晰壳层结构"才能产生有效梯度;而"pred 有清晰壳层结构"需要 loss_shell_dist 提供梯度引导。两者互为条件,**没有外部信号打破循环**。

**实测证据**(SA §10 + Exp5'-MA 训练曲线复盘):

1. STEP1-FIX-C smoke epoch 0-1: shell_count_loss 16~189(数值正常),但 SA 当时未 dump n_active 值,无法验证
2. STEP2 epoch 0: shell_count_loss=399(主导 total 95%)— 数值大但梯度未必有效
3. STEP2-CONTINUE 全程 shell_count_loss 在 ~ 400 平台,**与训练初期 epoch 0 相同量级**,165 epoch 续训未产生显著改进
4. STEP3 verdict: pred_shell1_mean 6.32 Å vs true 2.27 Å — 模型完全没收敛到真实 shell-1 位置
5. shell-1 elem score 0.0071 — 模型**连"shell-1 该是哪种元素"都没学**,因为 shell-1 都没找对位置

### 2.3 反观 `_pairwise_min_distance_penalty` 为什么生效

```python
def _pairwise_min_distance_penalty(pred_frac_coords, num_atoms, L, threshold):
    for sample:
        d = pairwise distances
        violation = ReLU(threshold - d)
        loss += (violation ** 2).mean()
```

**不依赖 pred 已有任何结构**。从 random init 起,只要任意两原子 cart 距离 < 1.5 Å 就有 violation > 0 → 梯度回传推开原子。**自启动**,不需要"鸡或蛋"。

这就是 STEP3 verdict 中 gate_pass_rate 64%(pairwise_min 大幅生效)+ shell-1 dist score 0.035(shell loss 几乎没生效)的根因差异。

### 2.4 为什么 errata 2/3/4 都没发现这个问题

| errata | 发现时机 | 为什么没看到 |
|---|---|---|
| errata 2(`_density_loss` 塌缩)| Exp4 final report 反思期 | shell loss 是 Exp5' 引入的,Exp4 没这套 |
| errata 3(L=6 fold artifact)| STEP1 SA 自查 dataset 物理性 | 关注的是 dataset 输出层,不是 loss 梯度有效性 |
| errata 4(ckpt selection bug)| STEP2 hand-back 后验尸 | shell_count_loss=399 我标 watch-only 但没追"为什么不降",当时假设是 expected behavior |

**整个 Exp5'-MA 的 watch-only 列表里 4 次提到 shell_count_loss 数量级异常,但都没追到鸡蛋问题根因**:
- proposal §2.5 警告 "epoch 0-10 ill-defined"
- launch note §11 #1 watch-only
- STEP1-FIX-C C5 SA 报 16~189
- STEP2 epoch 0 我标 95% dominate watch-only

**这是 Exp5'-MA 监督失职**,但根因不是检查不勤,是 **watch-only 这个机制对"loss 数值看起来正常但梯度无效"型 bug 不灵敏** — 6 active loss 全部 finite + 数值在合理范围 → 看起来一切正常。

**Lesson(进 ExpN 不变量)**:任何"依赖 pred 已有结构"的 loss(gap-based / cluster-based / topology-based),设计文档必须显式标注"鸡蛋启动条件",并在 dry-run 阶段 dump n_active 值,验证非零比例 ≥ 阈值(如 ≥ 50%)才能进训练。

---

## §3 errata 4 §6 决议修订(由 STEP2-CONTINUE 生效)

### 3.1 errata 4 §6 原决议

> Exp5'-MA 2026-05-03 决议:STEP3 用 `last.ckpt`(epoch 154,composite 0.576)进 sample。

### 3.2 修订后决议

> Exp5'-MA 2026-05-09 决议(基于 STEP2-CONTINUE 续训结果):**STEP3 用 `composite_epoch169_score0.5881.ckpt`**(STEP2-CONTINUE 续训 BEST,composite 0.5881,md5 `127afa44a850d8f7e4fcdae17e2761a1`)。

**修订理由**:
- STEP2-CONTINUE 续训 165 epoch,composite 从 STEP2 末 0.576 推到 epoch 169 BEST 0.5881(+0.012 真改进)
- STEP2-CONTINUE 用 `strict=False` 修复了 errata 4 §2 的 callback bug,本次 ckpt selection 可信
- last.ckpt(epoch 319 末尾)composite=0.573 显示末尾过训练,不如 epoch 169 BEST

### 3.3 本修订对 final report v3 的影响

final report v3 verdict 表 ckpt 来源必须标注 `composite_epoch169_score0.5881.ckpt`,**不是** errata 4 §6 旧决议的 last.ckpt。

---

## §4 与 errata 1/2/3/4 的关系

| errata | 内容 | 与本 errata 关系 |
|---|---|---|
| 1 | Phase 6.5 状态修正 | 无交叉 |
| 2 | `_density_loss` 塌缩 + Exp3 历史 + 方向 menu | §3.2 方向 4 "anti-collapse loss → Exp6 候选" 现升级为 Exp5'' 主线(详 §6) |
| 3 | L=6 fold + L=20 决议 | §5.2 RMSD 三层归因 + §7.3 ExpN 不变量(L_VIRTUAL ≥ 2 × CUTOFF_R)沿用 |
| 4 | STEP2 ckpt bug + verdict 双指标 SOP | §6 决议被本 errata §3 修订,§5.3 双指标 SOP 沿用 |
| **5(本文)** | Shell loss 鸡蛋问题 + Exp5'' 方向 | **FINAL** |

五份 errata 并列存档,不合并。errata 5 是 Exp5'' proposal 的 prerequisite。

---

## §5 ExpN 不变量级 Lessons(本 errata 贡献)

### 5.1 依赖 pred 结构的 loss 必须 dry-run 验证 n_active

任何 loss 函数:
- gap-based(本 errata 揭示)
- cluster-based(future-proof)
- topology-based(future-proof)
- structure-aware regularization(future-proof)

**必须**在 dry-run 阶段 dump 关键内部统计量:

```python
# 例:_shell_distance_loss
n_active_per_batch = ...  # 每 batch 中 boundaries ≥ 1 的 sample 数
n_active_ratio = n_active_per_batch / batch_size

# 进训练前必须验证:
assert n_active_ratio.mean() >= 0.5, \
    f"loss inactive on {(1-n_active_ratio.mean())*100:.1f}% samples — 鸡蛋启动问题"
```

**ExpN+ launch note 强制要求**:任何新 loss 实施时,SA dry-run hand-back 必报 n_active 比例。

### 5.2 训练时 composite_ckpt_score ≠ 评估时 step5_3 composite

errata 4 §5.3 已写"双 verdict 指标并列报告",本 errata 进一步精化:**这两个公式不是同一个东西**。

- 训练时 `val_composite_ckpt_score`:LightningModule 公式,3 项加权(loss + gate + pairwise),用于 ckpt selection
- 评估时 step5_3 `composite`:7 项加权(gate + 4 项 shell + type + RMSD),用于 verdict

**ExpN+ 强制**:final report verdict 必须**全部用** step5_3 复合分,**不允许**用训练时 ckpt selection 公式作为 verdict 数字。final report 表头必须明示"composite (step5_3 7 项)"以避免混淆。

### 5.3 watch-only 机制对"数值正常但梯度无效"型 bug 不灵敏

errata 5 §2.4 揭示:Exp5'-MA 4 次 watch-only 标记 shell_count_loss 数量级异常,都没追到鸡蛋问题根因。**根本是因为 watch-only 检查的是"loss 数值",不检查"loss 是否产生有效梯度"**。

**ExpN+ 改进**:watch-only 项必须明确"check 什么、什么是 fail"。对 loss 函数:
- 数值 finite → ✅(已检)
- 数值在合理范围 → ✅(已检)
- **梯度有效 → 必须显式检查**(本 errata 加)
  - dry-run dump n_active(如 §5.1)
  - 训练曲线对比该 loss 与"应受其影响的 metric"是否同时改进(如 shell_count_loss 应与 shell-1 distance score 同步降)

### 5.4 阶段性失败值得记录,不掩盖

Exp5' final report v3 不应隐藏 shell loss 失效。errata 5 是 Exp5'' proposal 的 prerequisite,Exp5'' 设计将基于本 errata 的根因诊断。**记录失败本身是研究价值**,publishable 比"假装 GREEN" 更诚实。

---

## §6 Exp5'' 方向决议

**Exp5'-MA 2026-05-09 决议**:Exp5'' 主线方向是"shell loss 重设计 + 不动其他"。

### 6.1 不动的内容(Exp5' 已验证有效)

- `_pairwise_min_distance_penalty`(λ=1.0)— gate 64% 硬证,生效
- L_VIRTUAL = 20(errata 3 修复)
- cost_density = 0.2(errata 2 揭示是塌缩剂但 Exp5' 沿用 OK)
- batch=64 / num_workers=16 / PreCollatedDataset(STEP2-CONTINUE 已验证可用)
- 三件套 cost 框架(只改 shell 部分公式,cost 保持 0.5 / 0.2)
- shell_boundaries.pkl 干净 ground truth(errata 3 §3 验证)

### 6.2 候选方向(Exp5'' proposal 二选一,等 Exp5' final report v3 完成后再选)

#### 候选 A:Distance-supervised shell loss(SA §10 候选 1)

```python
def _shell_distance_loss_v2(pred_frac_coords, num_atoms,
                             true_shell1_d_mean, has_shell2, true_shell2_d_mean,
                             L=L_VIRTUAL):
    """
    不依赖 gap 切壳,直接用真值 shell-1 / shell-2 半径作 attractor。
    每个 sample:取所有原子 radial,用 KNN 或固定切片找 shell-1 候选,
    把 shell-1 候选半径均值拉向 true_shell1_d_mean。
    """
    for sample i:
        coords_i = pred_frac_coords[i] * L
        radial = coords_i.norm(dim=1).sort()  # ascending
        # shell-1 候选:取 true_shell1_n 个最近原子
        n_s1 = int(true_shell1_n[i])
        pred_s1_d_mean = radial[:n_s1].mean()
        loss += (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2
        if has_shell2[i]:
            n_s2 = int(true_shell2_n[i])
            pred_s2_d_mean = radial[n_s1:n_s1+n_s2].mean()
            loss += (pred_s2_d_mean - true_shell2_d_mean[i]) ** 2
```

**优点**:
- 不需要"鸡蛋"启动条件 — 即使 random init 也能用 KNN 切片
- 直接 supervise 真物理量,不需要中间 gap 边界

**缺点**:
- 用了 ground truth 的 `true_shell1_n / true_shell2_n` 作为切片大小 — 训练时是 cheating?proposal §2.2 注释里讨论过"truth uses all neighbors,pred uses N=20 truncated",这里更近一步用 true count。是否合规?**Exp5'' proposal 必须讨论**

#### 候选 B:Distance-aware origin-attractor(errata 2 §3.2 方向 4)

把 `_density_loss` 改为 shell-target attractor(errata 2 §3.2 候选):

```python
def _density_loss_v2(input_frac_coords, pred_x, sigmas, sigmas_norm,
                     true_shell1_d_mean, has_shell2, true_shell2_d_mean):
    """
    Tweedie x0_hat → 不再 attract 原点,而 attract shell-1 / shell-2 半径。
    每个原子 i,根据其最近邻属性(shell-1 or shell-2 by distance proximity)
    pull 到对应 shell 半径。
    """
    x0_hat = ...
    radial_hat = (x0_hat * L).norm(dim=-1)
    # 软分配每原子到 shell-1 / shell-2(距哪个均值近)
    d1 = (radial_hat - true_shell1_d_mean).abs()
    d2 = (radial_hat - true_shell2_d_mean).abs() if has_shell2 else inf
    target = where(d1 < d2, true_shell1_d_mean, true_shell2_d_mean)
    return ((radial_hat - target) ** 2).mean()
```

**优点**:
- 复用 `_density_loss` 已有梯度路径(Tweedie x0_hat),工程改动小
- attractor 方向是真物理 shell 半径,不是 origin
- 保留 `_shell_distance_loss` 作辅助监督(已生效部分不动)

**缺点**:
- 软分配 shell-1/shell-2 可能不稳(初期距离都接近时分配震荡)
- 与现有 `_density_loss` 的 cost 关系需重新平衡

### 6.3 决议路径

**Exp5'-MA 暂不在本 errata 选 A 或 B**。理由:
- 候选 A 依赖 ground truth shell count,合规性需要 Exp5'' proposal 阶段讨论
- 候选 B 与 `_density_loss` 已有耦合,需要 Exp5'-MA 重审两者协同
- final report v3 完成后,基于 Exp5' 全数据 + 师兄 / 用户 / Exp5'-MA 三方决议 A/B/C(C 可能是 A+B 混合)

**STEP4-FINAL-REPORT 期间**,final report v3 §X 章节"未来工作"会列 A 和 B 作为 Exp5'' 候选,具体选哪个由 Exp5'' proposal 阶段决议。

### 6.4 不开 Exp5'' 直到 final report v3 完成

**Exp5'-MA 2026-05-09 决议**:Exp5'' proposal 必须等 Exp5' final report v3 完成后再写。理由:
- final report v3 是 Exp5'' proposal 的 baseline,baseline 数据不固化,proposal 比较没意义
- final report v3 写作过程会暴露 Exp5'-MA 之前没察觉的 paper trail 细节,可能影响 Exp5'' 方向
- 用户师兄可能在 review final report v3 时给新见解

---

## §7 给 STEP4-FINAL-REPORT 和 Exp5'' 阶段的提醒

如未来 Exp5'' SA 引用 STEP3 sample 输出 + Exp5' 训练曲线,**必须同时引用本 errata § 2 鸡蛋问题根因**。任何"Exp5' shell loss 工程实现 bug"的简化叙述都是 errata 5 之前的旧判断,需修正为"loss 函数设计前提失败,鸡蛋启动条件未满足"。

final report v3 §X "Limitations & Lessons" 章节必须明确写:
- shell loss 公式数学正确,代码实现无 bug
- 失败根因是设计前提(假设 pred 有壳层结构才能 gap 切壳)与训练实际(random init 没有结构)矛盾
- 这是 ExpN+ 不变量级 lesson(§5.1),不限于 Exp 系列

---

## §8 关于 Exp5'-MA 监督责任的自我评估

errata 5 §2.4 已客观记录 4 次 watch-only 标记但未追根因。**Exp5'-MA 不掩饰**:

1. proposal §2.5 警告 "epoch 0-10 ill-defined",未要求 dry-run dump n_active
2. STEP1 launch note §11 watch-only,未机制化"loss 梯度有效性 check"
3. STEP1-FIX-C C5 SA 报 16~189 → Exp5'-MA 接受为正常
4. STEP2 epoch 0 shell_count=399 → Exp5'-MA 标 watch-only 不强制 ping

**修正措施**:errata 5 §5.1 / §5.3 lesson 已落,ExpN+ launch note 必须强制 n_active dry-run check + 梯度有效性 watch-only 升级为 active monitor。

**这次 30+ h GPU 投入没有完全浪费**:
- pairwise_min loss 生效是真贡献
- shell loss 鸡蛋问题根因诊断本身是 publishable lesson
- 5 份 errata 完整 paper trail 在工业 / 学术界都是高质量记录

---

*Exp5'-MA 撰写,2026-05-09*
*基于 SA-EXP5'-STEP3-SAMPLE hand-back v1+v2 + 全 STEP1-3 paper trail + 4 份前置 errata*
*errata 5 是 Exp5'' proposal 的 prerequisite,Exp5'' 设计将基于本 errata §6 候选 A/B 之间二选一*
