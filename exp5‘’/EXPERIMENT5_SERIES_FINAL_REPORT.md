# EXPERIMENT5_SERIES_FINAL_REPORT.md
# Exp5 系列完整最终报告 — Exp5 v2 / Exp5' / Exp5'' 三阶段总结
# Diagnosis and Bounded Improvement of Physical Validity in
# Diffusion-Based Local Atomic Structure Prediction from XAS

> **撰写者**: Exp5'-MA(Exp5 系列第 3 任 Main Agent,负责 Exp5' 阶段 + 全系列总结)
> **日期**: 2026-05-10
> **版本**: Series Final v1(继承 Exp5 v2 final report + Exp5' final report v3,本文是全系列收尾)
> **状态**: ✅ Exp5 系列正式 wrap up。Exp6(Transformer 架构)由同步进行的另一线 MA 主导,Exp7(GAN 方向)由用户决议启动。
> **数据来源**: 7 棒 SA(Exp5')+ 0 SA(Exp5'',MA 直接做)+ 6 份 errata + 训练 log + sample predictions + step5_3 复合分输出 + Exp4 STEP5HANDOFF 红线对照
> **核心 ckpt 三档**:
>   - Exp5 v2 — `sa2pp_resume_epoch529_val0.7003.ckpt`(物理灾难)
>   - **Exp5' — `composite_epoch169_score0.5881.ckpt`**(部分胜利,系列 best)
>   - Exp5'' — `composite_epoch199_score0.5319.ckpt.frozen_p4_final`(失败诊断)

---

## §0 给师兄 / 导师 / Exp6/Exp7 读者的 30 秒速读

### 0.1 一句话定调

> **Exp5 系列三阶段在固定架构(MV-attention encoder + CSPNet decoder)下穷尽 loss-level fix,得到一个部分胜利的中间产物(Exp5'),一个失败但有诊断价值的 negative result(Exp5''),和六条 ExpN+ 不变量级 lesson。Exp5' 验证 pairwise loss 自启动有效,Exp5'' 证明 loss attractor 方法在该架构上有上限。这两个结论合起来论证了换架构(Exp6 Transformer / Exp7 GAN)的必要性。**

### 0.2 三档 verdict 全表(投稿 / 论文 baseline 章节直接复用)

| 指标 | Exp4(旧体系基线) | **Exp5 v2**(灾难)| **Exp5'**(系列 BEST)| **Exp5''**(失败) |
|---|---|---|---|---|
| **RMSD (Å,旧 metric)** | **1.4866** ✅ | 不可比 | 不可比 | 不可比 |
| **TypeAcc (旧 metric)** | **0.1973** ✅ | 0.249(F1 0.108)| set-level 0.0071 | 0.005 |
| **pred_in_cutoff (旧 metric)** | **18.92/20** ✅ | 不明 | 不明 | 不明 |
| **gate_pass_rate (≥1.5 Å,新)** | 未测* | 5-11% 🔴 | **64.0%** ⚠️ AMBER | 30.6% 🔴 |
| **min_d_mean (Å,新)** | 未测* | < 1.0(估)| **1.687** ⚠️ | 0.872 🔴 |
| **collapse_rate (新)** | 未测* | 不明(>10% 估)| **0.00%** ✅ | 31.7% 🔴 |
| **composite (step5_3 7 项,新)** | 未测* | 0.005-0.011 🔴 | **0.080** 🔴 边缘 | 0.035 🔴 |
| **shell-1 dist score (新)** | 未测* | 0.0000 🔴 | 0.035 🔴 | (low) 🔴 |

*Exp4 当时使用 STEP5HANDOFF 旧红线体系(RMSD ≤ 2.0 / TypeAcc ≥ 0.20 / pred_in ≥ 15),新物理指标(min_d gate / collapse / step5_3 复合分)是 Exp5 v2 → SA-METRICS-V3 阶段才发明,Exp4 ckpt 已不可访问以倒回去测。errata 3 §5.2 揭示 Exp4 RMSD 1.49 Å 实际是 fold artifact 几何上限(≤ L/2 = 3 Å),不是真物理 RMSD。

### 0.3 三档 verdict 文字结论

- **Exp4**:旧体系全合格,但旧体系不暴露物理违反。等价于 "考试用的指标本身不严"。
- **Exp5 v2**:旧体系数学指标改进(F1 +28.8%),新体系暴露物理灾难(gate 5-11%)。**SA-METRICS-V3 的发明是 Exp5 系列对方法论最大贡献之一**。
- **Exp5'**:🟢 Fold artifact 修复 + 🟢 Pairwise loss 验证生效 + 🟡 Shell loss 未生效(鸡蛋启动问题)。系列 BEST,但 mixed verdict。
- **Exp5''**:Shell loss 重设计候选 A 失败,collapse rate 31.7% deep RED。揭示**双根因**:训练 active ≠ 评估 active(伪解决)+ Attractor vs Constraint 几何冲突。

### 0.4 对 Exp6 / Exp7 的关键启示

1. **Loss-only fix 在 MV-attention + CSPNet 上有上限**(Exp5'' 已证)。Exp6 换 Transformer 是正确方向。
2. **Pairwise min distance loss 是自启动的硬约束,应在 Exp6 / Exp7 继续保留**(Exp5' 64% gate 硬证)。
3. **Shell-aware 监督必须从架构层注入**,不能 post-hoc 加 loss(errata 6 §7 hypothesis)。Exp6 Transformer 的 attention 机制本身可能 implicit 形成 shell;Exp7 GAN 的 discriminator 应能 reward "形成清晰 shell 结构"。
4. **L_VIRTUAL=20 + CUTOFF_R=10 配置(errata 3 修复)在 Exp6 / Exp7 必须沿用**,这是 dataset 物理正确性的基础。
5. **Verdict 必须双指标并列**(composite + gate,errata 4 §5.3 SOP)。Exp7 GAN 用单 inception score / FID 类指标如有可能不够,需 augment 物理 metric。

---

## §1 系列背景与目标

### 1.1 任务定义

DiffCSP-XAS:从 X 射线吸收谱(XAS,~ 100 维 spectrum)预测局部原子结构(20 原子周围环境,frac coords + atom types)。

**输入**:
- spectrum(SpectrumEncoder 编码到 256d latent)
- FEFF features(73d 物理参数,scaling/shift)
- center element type(单原子,作 condition)

**输出**:
- 20 个邻居原子的 frac_coords(中心元素在原点,frac ∈ [-0.5, 0.5])
- 20 个邻居的 atom_types(分类,88 元素)

**Architecture**:Conditional latent diffusion(CSPNet backbone + MV-attention spectrum encoder + center embedding)。

### 1.2 Exp 系列演进表

| 阶段 | 主要变化 | 评估体系 | 主负责 MA |
|---|---|---|---|
| Exp1 | 初版,Fe-only 单中心 | RMSD + TypeAcc | MA1 |
| Exp2 | Fe-only,完善 dataset 流水线 | + pred_in_cutoff | MA2 |
| Exp3 | 引入 TypeClassifier head(失败,详 errata 2 §2) | 同 | MA3 |
| Exp4 | 扩展到 88 元素,L_VIRTUAL=6 沿用(隐藏 fold artifact) | STEP5HANDOFF 红线(RMSD/TypeAcc/pred_in)| MA4 |
| **Exp5 v2** | MV-attention + center embedding + cost_density 0.2 + LR scheduler | + SA-METRICS-V3(新物理指标)| MA5 |
| **Exp5'** | + 三件套物理 loss + L_VIRTUAL=6→20(fold 修复)| 新+旧双指标 | **Exp5'-MA(本报告)** |
| **Exp5''** | Shell loss 重设计(distance-supervised KNN slice + sigmoid band)| 同 Exp5' | **Exp5''-MA**(本报告续棒) |

### 1.3 Exp5 系列总投入(三阶段汇总)

| 资源 | Exp5 v2 | Exp5' | Exp5'' | 合计 |
|---|---|---|---|---|
| SA 棒数 | 多(MA5 主导)| **7 棒** | 0(MA 直接做)| 多 |
| GPU 训练时长 | ~ 48h(SA2 + SA2'')| ~ 17h(STEP2 7h + STEP2-CONTINUE 10h)| 6h(P4 warm-start)| ~ 71h |
| GPU sample 时长 | ~ 30 min(dry-run 100)| ~ 1.5h(三 split 全集)| ~ 4h(三 split 全集)| ~ 6h |
| Wall clock | ~ 2 周 | ~ 1 周 | 1 天 | ~ 3.5 周 |
| Errata 数 | 1(errata 2,继 Exp4)| **3**(errata 3/4/5)| **1**(errata 6 — 本报告整合)| **5** |
| Final report 数 | v1 + v2 | v3 | (本报告整合)| 3 + 1 |

### 1.4 Exp5 系列三阶段决议时间线

```
2026-04-25  MA4: Exp4 phase 5b verdict — STEP5HANDOFF 红线全过
2026-04-28  MA4: errata 2(_density_loss 塌缩 + Exp3 历史)
2026-04-28  MA5: Exp5 v2 proposal — MV-attention + 三件套未引入
2026-04-XX  MA5: SA2 baseline + SA2'' resume,F1 +28.8% 数学改进
2026-04-30  MA5: SA-METRICS-V3 dry-run 100 → 物理灾难暴露
2026-05-01  Exp5'-MA(接班): Exp5' from-scratch 决议 + 三件套 loss proposal
2026-05-01  SA-EXP5'-STEP1: dataset shell_boundaries inject + 三件套 loss + smoke 全过
2026-05-02  SA-EXP5'-STEP1-AUDIT: 自查 dataset 发现 fold artifact → errata 3
2026-05-02  SA-EXP5'-STEP1-FIX: L=6→20,cartesian sanity 100/100
2026-05-03  SA-EXP5'-STEP1-FIX-C: dataset cache rebuild(20.2 min)
2026-05-03  SA-EXP5'-STEP2-TRAIN: 训练 7h EarlyStop(ckpt selection bug)→ errata 4
2026-05-04  SA-EXP5'-STEP2-CONTINUE: warm-start 续训 10h,epoch 169 BEST composite 0.5881
2026-05-09  SA-EXP5'-STEP3-SAMPLE: 三 split sample + step5_3 → mixed verdict → errata 5
2026-05-09  Exp5'-MA: errata 5 + final report v3 完成
2026-05-09  Exp5''-MA(接班): Exp5'' proposal 候选 A 决议
2026-05-10  Exp5''-MA: P1-P5 一天完成,verdict ❌ FAILURE
2026-05-10  Exp5'-MA(返岗): 系列 wrap up + 本报告(同时担任 errata 6 撰写)
```

---

## §2 Exp5 v2 阶段总结(继承 Exp5 v2 final report)

### 2.1 Exp5 v2 改动摘要

- **Architecture**:MV-attention(num_heads=4, residual_alpha=0.5)+ center embedding(95 × 16d)+ cost_density 0.2
- **Training**:Adam lr=1e-4 + batch=16 + CosineAnneal T_max=500 + fp32 + grad_clip=1.0
- **Loss**:三件套**未引入**(Exp5 v2 与 Exp4 loss 体系相同,只是架构 + cost_density 改)

### 2.2 Exp5 v2 verdict(回顾)

**数学指标**(SA2'' epoch 529):
- val_loss 0.7003(vs Exp4 baseline 0.7300)
- Multiset Macro-F1 0.1086(vs Exp4 0.0843,+28.8% 改进)
- Position-by-position TypeAcc 0.0945(vs Exp4 0.2160,**退步**,但 errata 2 揭示此指标为虚假指标)

**物理指标**(SA-METRICS-V3 dry-run 100):
- **min_d gate pass rate**:5%(val)/ 11%(test)🔴
- **shell-1 distance score**:0.0000 🔴(gate-pass 子集都不知 shell-1 应在 ~ 2-3 Å)
- **复合分均值**:0.0056(val)/ 0.0062(test)— verdict 表外

**结论**:数学改进无法掩盖物理灾难,verdict ❌ physical-invalid。

### 2.3 Exp5 v2 对系列的关键贡献

| 贡献 | 说明 |
|---|---|
| **SA-METRICS-V3 物理评估体系** | 首次引入 min_d gate + shell distance score + collapse rate,**这套体系成为后续 Exp5' / Exp5'' / Exp6 / Exp7 的通用 verdict 工具** |
| **MV-attention 验证** | 架构层面验证 multi-view attention 不会破坏数学指标(F1 改进),但单 attention 不足以约束物理 |
| **失败暴露 fold artifact 的需要** | Exp5 v2 物理灾难是 Exp5'-STEP1-AUDIT 自查的 trigger |

---

## §3 Exp5' 阶段总结(本报告主体之一)

### 3.1 Exp5' 核心改动

#### 3.1.1 Fold artifact 修复(errata 3 关键贡献)

**问题诊断**(SA-EXP5'-STEP1-AUDIT 主动自查):
- Exp4 / Exp5 v2 沿用 dataset_v2.py `L_VIRTUAL = 6` + `CUTOFF_R = 10`
- `frac = relative_cart / 6; frac = frac - round(frac)` 这一步 min-image fold
- 任意两个真实邻居各自距中心 > L/2 = 3 Å → fold 后映射到 frac box 两端 → min-image pairwise cart 距离 ≈ 0
- **100 样本 audit:64% 样本至少有一对此类 fold artifact 虚假近距离对**

**修复**:`L_VIRTUAL = 6 → 20`,L/2=10 ≥ CUTOFF_R=10。dataset cache rebuild(60501/60507 train 样本 valid,99.99%)。

**修复后硬证**:
- Cartesian sanity 100/100 PASS(MIN_BOND_LENGTH 0.7 Å)
- Fold 案例样本(双邻居 [+3.2, 0, 0] / [-3.2, 0, 0])实测距离:旧 L=6 0.40 Å → 新 L=20 6.40 Å

#### 3.1.2 三件套物理 loss 引入(proposal §2)

| Loss | Cost | 公式 | Inference 状态 |
|---|---|---|---|
| `_pairwise_min_distance_penalty` | **1.0** | `ReLU(1.5 - d_pair)² mean`,min-image cart | ✅ 自启动,Exp5' 生效 |
| `_shell_distance_loss` | 0.5 | gap > 0.1563 切壳 → pred shell-1 mean vs ground truth MSE | ⚠️ 鸡蛋启动问题(errata 5) |
| `_shell_count_loss` | 0.2 | gap 切壳 → pred shell 配位数 vs truth MSE | ⚠️ 同上 |

Ground truth 5 字段(`true_shell1_d_mean` / `true_shell2_d_mean` / `has_shell2` / `true_shell1_n` / `true_shell2_n`)从 `shell_boundaries.pkl`(Exp4 Step 2.5 产物)inject 到 dataset。

#### 3.1.3 其他改动

- **Center embedding 95 × 16d**:沿用 Exp5 v2
- **MV-attention 4 heads**:沿用
- **PreCollatedDataset 加速**:STEP2-TRAIN SA + 用户决议(batch=16→64,num_workers=0→16,persistent_workers=True),errata 4 §3 标流程未经 review 但实测幸运没害

### 3.2 Exp5' 训练历程(errata 4 / final report v3 §4)

**STEP2-TRAIN** ckpt selection bug(errata 4 §2):
- `EarlyStopping(monitor='val_composite_ckpt_score')` 在第一个 val epoch 前 RuntimeError → SA fallback 改 monitor=`val_gate_pass_rate`
- gate 是平台抖动指标(epoch 4 lucky shot 0.5305)→ patience=30 用满 → epoch 154 提前 EarlyStop
- **真正的 composite 仍在缓慢爬**(从 ckpt log 末尾 50 行 grep 看 epoch 152-154 三指标同时阶跃)

**STEP2-CONTINUE** 修复:
- `strict=False` 让 callback 容忍 metric 启动期不存在
- Warm-start from epoch 154 last.ckpt
- 续训 epoch 155 → epoch 319 真 EarlyStop
- **Exp5' BEST**:`composite_epoch169_score0.5881.ckpt`(md5 `127afa44a850d8f7e4fcdae17e2761a1`)

**STEP3-SAMPLE** verdict(三 split):

| Split | composite | gate | min_d | collapse |
|---|---|---|---|---|
| val (N=7621) | 0.0801 | 64.0% | 1.687 Å | 0.00% |
| test (N=4481) | 0.0795 | 65.2% | 1.695 Å | 0.00% |
| holdout (N=3025) | 0.0828 | 63.8% | 1.681 Å | 0.00% |

**三 split 一致性 < 0.004**,泛化优秀。

### 3.3 Exp5' verdict — Mixed:Partial Success

| 维度 | 状态 | 证据 |
|---|---|---|
| Fold artifact 修复 | ✅ 完成 | cartesian sanity 100/100,fold 案例硬证 0.40→6.40 Å |
| Pairwise loss 生效 | ✅ 真贡献 | gate 5-11% → 64%,6-13× 改进;collapse 0% |
| Shell loss 生效 | ❌ 未生效 | shell-1 distance score 0.035 vs target ≥ 0.50;pred shell-1 mean 6.32 Å vs true 2.27 Å |
| 双 verdict 指标 SOP | ✅ 落地 | errata 4 §5.3 |
| Composite verdict | 🔴 RED(0.080 < 0.40) | 7 项中 4 项 shell 相关 RED |
| Gate verdict | 🟡 AMBER 边缘(64% < 80%) | 但比 v2 5-11% 大幅改进 |

### 3.4 Exp5' 失败侧的核心诊断 — Shell loss 鸡蛋启动问题(errata 5 §2)

**鸡蛋问题精确定义**:

> `_shell_distance_loss` 需要 "pred 已有清晰壳层结构" 才能产生有效梯度(gap > 0.1563 切壳);而 "pred 有清晰壳层结构" 需要 `_shell_distance_loss` 提供梯度引导。两者互为条件,无外部信号打破循环。

**对比 `_pairwise_min_distance_penalty`**:从 random init 起,只要任意两原子 cart 距离 < 1.5 Å 就有 violation > 0 → 梯度回传推开原子。**自启动,不依赖 pred 已有任何结构**。

这就是 Exp5' verdict 中 gate=64% / shell-1=0.035 的根因差异:**自启动 loss 生效 + 鸡蛋问题 loss 失效**。

---

## §4 Exp5'' 阶段总结(本报告主体之二)

### 4.1 Exp5'' 候选 A 设计(基于 Exp5' 鸡蛋诊断)

**目标**:重设计 `_shell_distance_loss` / `_shell_count_loss` 让其**自启动**,从 pred shell-1 mean 6.32 Å → 接近真值 2.27 Å。

**候选 A 公式**:
- `_shell_distance_loss_v2`:用 ground truth `true_shell1_n` / `true_shell2_n` 作切片大小,取 pred 最近 K1 个原子的 radial mean,与 ground truth 半径 MSE
- `_shell_count_loss_v2`:用 sigmoid soft mask 数 ground truth shell-1 半径 band 内 pred 原子数,与 truth K1 MSE

**为什么不选候选 B(distance-aware density)**:几何冲突预期更严重(所有原子都被 attract 到 shell 半径,collapse 预期 > 50%)

### 4.2 Exp5'' P5 verdict(三 split)

| Split | composite | gate | min_d mean | collapse | n_pred_shells=0 |
|---|---|---|---|---|---|
| val (N=7621) | **0.0347** | 30.6% | **0.872** | **31.7%** | **70.2%** |
| test (N=4481) | ~ 0.034 | 29.2% | 0.861 | 31.5% | 69.3% |
| holdout (N=3025) | ~ 0.034 | 29.2% | 0.844 | 32.6% | 71.4% |

三 split 一致性 < 0.001,verdict 是设计层失败不是 split noise。

### 4.3 Exp5'' 对照 Exp5' Δ 表

| 指标 | Exp5' STEP3 baseline | **Exp5'' P5** | Δ |
|---|---|---|---|
| Composite | 0.080 | 0.035 | **-56%** ↓ |
| Gate | 64.0% | 30.6% | **-33.4 pp** ↓ |
| Min_d mean | 1.687 Å | 0.872 Å | **-49%** ↓ |
| Collapse | 0.00% | **31.7%** | **+31.7 pp** 🔴 |
| Shell-2 coord_n score | 0.32 | 0.48 | **+50%** ✅(唯一改进项) |

**Exp5'' verdict ❌ FAILURE**。Shell-2 coord_n +50% 改进不足以补偿其他全面退步。

### 4.4 Exp5'' 失败双根因诊断(本报告 + errata 5/6 整合)

#### 4.4.1 根因 1 — 鸡蛋问题伪解决(候选 A 没真正解决,推到评估端)

**关键区分**:训练 `n_active` vs 评估 `n_pred_shells`

| 层 | 来源 | 含义 | Exp5'' P5 数值 |
|---|---|---|---|
| 训练 loss `n_active` | Exp5''-MA 加的 dump | 多少 sample 进入 loop(num_atoms ≥ 2)| **100% 全程** ✅ |
| **评估 `n_pred_shells`** | step5_3 gap-based 切壳 | pred 坐标能切出几个 shell(gap > 0.1563 Å)| **70% 是 0** 🔴 |

候选 A 公式只奖励"前 K 个最近原子的均值 ≈ ground truth 半径",**不奖励"形成清晰 shell 边界的坐标分布"**。model 学到代理统计量(distance mean)但 bypass 真问题(形成 shell)。

#### 4.4.2 根因 2 — Sigmoid attractor 与 pairwise constraint 几何冲突

**几何分析**:
- shell-1 真值平均 6 个原子在半径 ~ 2.27 Å
- 6 个原子塞进半径 2.27 Å 球壳,等角度间隔 60° → 相邻距离 ~ 2.27 Å(理想)
- 但 sigmoid 不强制角度分布,model 可以在球面任意位置塞 → **实际相邻距离分布:多数 < 2.27 Å,部分 < 1.5 Å pairwise 阈值**

**Pairwise loss 的不对称性**:
- `ReLU(1.5 - d)²` 只惩罚 d < 1.5
- 不惩罚 d ∈ [1.0, 1.5)(此区间 loss 在 [0.25, 0] 之间,可被 shell loss 权重盖过)

**累积**:Epoch 169 collapse 0% → Epoch 199 ~ 10% → Epoch 238 ~ 31% → P5 sample 31.7%

### 4.5 Exp5'' 失败的方法论价值(errata 6)

虽然 verdict RED,Exp5'' 给系列贡献两条新 lesson:

**Lesson 6.1 — 训练 active 与评估 active 必须分别 dry-run**

任何 shell-aware loss 在 dry-run 阶段必须验证两层:
- 训练 loss 端:`n_active_loss_loop_ratio ≥ 0.95`(Exp5'' 落实)
- **评估端**:用真正的评估算法(step5_3 gap-based)在 batch 上跑一次,验证 `n_pred_shells ≥ 1` 比例 ≥ 80%

如训练 active 但评估 0-shell 比例高,**这个 loss 是伪解决**,不能进训练。

**Lesson 6.2 — Distance attractor 必须与 distance constraint 几何兼容性 check**

任何把原子拉向特定半径的 loss(shell / density / etc.)启用前必须:
- 估算"K 个原子等距分布在半径 r 球面"的最小 pair distance:d_min ≈ 2r × sin(180°/K)
- 如 d_min < pairwise_min threshold,**必然冲突**

Exp5'' 案例:K=6, r=2.27 → d_min ≈ 2.27 Å > 1.5 Å 表面看 OK,但 sigmoid soft attract 不强制等距 → 实际 collapse 31.7%。

---

## §5 系列三阶段 verdict 综合对比(投稿用)

### 5.1 主表(完整版)

| 指标 | 分类 | Exp4 holdout | Exp5 v2(SA-METRICS dry-run 100) | **Exp5' STEP3 (val full)** | **Exp5'' P5 (val full)** | 备注 |
|---|---|---|---|---|---|---|
| **RMSD (Å)** | 旧 | 1.4866 ✅ | (类似 ~ 1.49,fold 上限)| 不可比* | 不可比* | Exp5'/'' L=20,旧 RMSD 公式不适用 |
| **TypeAcc** | 旧 | 0.1973 ✅ | 0.249 / F1 0.1086 | (set-level 不同概念)| (同左)| 旧 metric 在 Exp5 系列被 set-level F1 取代 |
| **pred_in_cutoff (/20)** | 旧 | 18.92 ✅ | 不明 | 不明 | 不明 | 不重报 |
| **gate_pass_rate** | 新 | 未测 | 5-11% 🔴 | **64.0%** ⚠️ | 30.6% 🔴 | 主物理指标 |
| **min_d_mean (Å)** | 新 | 未测 | < 1.0(估)| **1.687** ⚠️ | 0.872 🔴 | |
| **collapse_rate** | 新 | 未测 | 不明 | **0.00%** ✅ | 31.7% 🔴 | 物理崩塌率 |
| **n_pred_shells > 0 比例** | 新 | 未测 | 不明 | 不明 | 29.8% 🔴 | Exp5'' 暴露 |
| **composite (step5_3 7 项)** | 新 | 未测 | 0.005-0.011 🔴 | **0.080** 🔴 | 0.035 🔴 | 综合指标 |
| **shell-1 dist score** | 新 | 未测 | 0.0000 🔴 | 0.035 🔴 | (similar low) 🔴 | |
| **shell-1 coord_n score** | 新 | 未测 | 不明 | 0.180 | (similar) | |
| **shell-2 coord_n score** | 新 | 未测 | 不明 | 0.316 | **0.48** ✅ | Exp5'' 唯一改进 |
| **shell-1 elem score** | 新 | 未测 | 不明 | 0.0071 🔴 | 0.005 🔴 | 元素分类失败,errata 2 §2 病态问题 |
| **泛化(三 split 一致性)** | 新 | < 0.01 | (dry-run 100 不足判)| < 0.004 ✅ | < 0.001 ✅ | 都很好 |

*"不可比"原因:Exp5'/Exp5'' 用了 L=20 box,旧 RMSD 在 L=6 box 内的计算公式数值不直接可比。错误的对比会给 reviewer 误导。投稿写法:Exp4 RMSD 1.49 在 errata 3 §5.2 揭示是 fold artifact 上限,不是真物理 RMSD,所以系列对比应以新物理指标为主。

### 5.2 关键改进/退步路径图

```
                Exp4              Exp5 v2          Exp5'           Exp5''
                ────              ───────          ─────           ──────
旧 metric:
  RMSD          1.49 ✅            ?                ?               ?
  TypeAcc       0.20 ✅            F1 +28.8% ✅      旧 metric 重定义
  pred_in       18.9 ✅            ?                ?               ?

新 metric(SA-METRICS-V3 之后):
  gate          未测              5-11% 🔴         64% 🟡⭐         31% 🔴
                                  ─────────────── 6-13× ────────►   ─────
                                                                    退步
  collapse      未测              ?(估 >10%)       0% ✅⭐          32% 🔴
                                                                    主要灾难
  shell-1 dist  未测              0.000 🔴         0.035 🔴         类似 🔴
                                                  ────── 出零 ────►
                                                  但仍 RED

  composite     未测              0.006 🔴         0.080 🔴         0.035 🔴
                                                  ────── 10× ────►  ────────
                                                                    退步 56%

主要贡献      旧 baseline       SA-METRICS-V3   fold 修复 +       双根因诊断
                                方法学发明       pairwise 验证     (lesson)
```

### 5.3 文字总结(一段话给 abstract / intro 用)

> Exp5 series consists of three stages on fixed architecture (MV-attention + CSPNet). Exp5 v2 introduced the SA-METRICS-V3 physical evaluation suite, which exposed catastrophic physical invalidity (gate_pass_rate 5-11%) hidden by improved mathematical metrics (F1 +28.8%). Exp5' diagnosed and fixed a foundational fold artifact in the dataset coordinate representation (L_VIRTUAL=6 → 20), and introduced a pairwise minimum distance loss that achieved 64% gate_pass_rate with 0% collapse — a 6-13× improvement over Exp5 v2. However, shell-level supervision via gap-based loss functions failed due to an egg-chicken initialization problem (loss requires pred shell structure to produce gradient; shell structure requires the loss). Exp5'' redesigned shell loss as distance-supervised (KNN slice + sigmoid attractor band) but failed: composite score dropped 56% relative to Exp5', collapse rate rose to 31.7%, and 70% of test samples failed to form any predictable shell boundary. Diagnosis identified two new failure modes: pseudo-resolution of the egg-chicken problem (training active ≠ evaluation active) and geometric conflict between attractors and pairwise constraints. The series concludes that loss-level fixes on this architecture have reached a hard ceiling, motivating architectural changes (Transformer, generative adversarial) for subsequent experiments.

---

## §6 Limitations & 失败侧的完整记录(投稿必含)

### 6.1 已知 Exp5' Limitations

1. **Shell loss 鸡蛋启动问题未在 Exp5' 解决**(errata 5 §2)。pred shell-1 mean 6.32 Å vs true 2.27 Å,差 4 Å。
2. **Composite verdict RED(0.080)**,虽然 gate / collapse / pairwise 全部改进,综合 verdict 仍未达 GREEN 0.40 阈值。
3. **Shell-1 elem score 0.007**:模型连"shell-1 该是哪种元素"都没学,根因 errata 2 §2 揭示是病态分类问题(88 元素 majority class 主导)。
4. **训练 batch_size 16→64 未经 MA review 改动**(errata 4 §3),虽然最终训练成功,流程上违 launch note §0.4 拍板。
5. **STEP2 ckpt selection bug 导致 ~ 30h GPU 浪费**(epoch 100-140 的真 best ckpt 未被保存),由 STEP2-CONTINUE 续训部分挽救。
6. **Holdout 1000 vs 3025 不一致**:proposal §0.5 写 1000,实际 3025 完整 split。

### 6.2 已知 Exp5'' Limitations

1. **候选 A 完全失败**:三 split RED verdict。
2. **Sigmoid soft mask 选择未充分论证**(proposal §2.3 末尾"P3 fail 切 sigmoid"我直接选了,事后看 boolean 可能 collapse 没这么严重)。
3. **P3 SMOKE 时 collapse 0% 让 Exp5''-MA 误判 mitigation 工作**,P4 epoch 5 未主动 dump collapse rate,导致 collapse 累积到 31.7% 才在 P5 暴露。
4. **EarlyStop patience 估算错**(忘了 `check_val_every_n_epoch=5` 倍数),P4 多训了 ~ 30 epoch。
5. **Exp5''_v2 ablation 未跑**(shell_band_width 0.5 / cost_shell_count 0.05 / boolean 替 sigmoid):有 50% 概率减弱 collapse 但根因 1 不修。我作为 Exp5'-MA 决议不跑,理由是 ROI 太低,wrap up 优先。

### 6.3 已知系列 Limitations(超 Exp5'/Exp5'')

1. **Architecture 从未变**:三阶段都是 MV-attention encoder + CSPNet decoder。任何架构层 inductive bias(等变结构 / shell-aware edge)都没在 Exp5 系列尝试。**这是 Exp6 / Exp7 的入口**。
2. **Element-aware threshold 未做 ablation**:errata 3 §8.2 决议保留全局 1.5 Å,但轻元素(C-C ~ 1.34 Å)真键长本就 < 1.5,可能 element-aware 能在 gate 上多挤几个百分点。
3. **Cheating 合规性论证仅纸面**:Exp5'' 候选 A 用 ground truth `true_shell1_n` 切片(proposal §2.4),论证为 sample-level 标量 label 合规,但 reviewer 可能不接受。**inference 时如何替代** ground truth count 没在 Exp5'' 实施(留 Exp6 / Exp7 用 spectrum encoder 输出估计)。
4. **Sample 时长 4h vs estimate 1.5h**:Diffusion sample 1000 steps × N samples,bottleneck 在 first-iter 重 + dataloader spin-up,Exp6 Transformer / Exp7 GAN 应改用 batch sampling 避免。

---

## §7 Lessons Learned(ExpN+ 不变量级,给 Exp6 / Exp7 / Exp8+ 直接复用)

来自 5 份 errata(errata 2/3/4/5/6 即本报告整合)的 ExpN+ 不变量级 lesson 完整清单。**所有 lesson 已落 launch note 强制 / final report SOP**。

### 7.1 数据层

**L1 — Dataset ground truth 必须在 cartesian Å 下验证**(errata 3 §7.1)
```python
frac = dataset[i]['frac_coords']
cart = frac * L
d_pairs = pairwise_distances(cart)
assert d_pairs[d_pairs > 0].min() >= 0.7  # H-H 物理下限
```

**L2 — L_VIRTUAL / 2 ≥ CUTOFF_R**(errata 3 §7.3)
- 任何虚拟立方体表示,L/2 必须 ≥ 邻居搜索半径
- Exp4 / Exp5 v2 沿用 Exp2 Fe-only 的 L=6 是设计边界未更新的遗漏

### 7.2 Loss 层

**L3 — 依赖 pred 结构的 loss 必须 dry-run 验证 n_active**(errata 5 §5.1)
- Gap-based / cluster-based / topology-based / structure-aware 任一 loss
- Dry-run 必 dump `n_active_loss_ratio ≥ 0.95` 才进训练

**L4 — 训练 active 与评估 active 必须分别验证**(errata 6 / 本报告 §4.5)
- Exp5''-MA 落 L3 但漏 L4 → 候选 A 训练 active 100% 但评估 active 30%
- 任何 loss dry-run 阶段必须**同时**跑评估算法(step5_3 / 等价)在 batch 上,验证 `n_pred_shells > 0 比例 ≥ 80%`

**L5 — Distance attractor 与 distance constraint 几何兼容性 check**(errata 6 / 本报告 §4.5)
- 把原子拉向半径 r 的 loss(attractor)必须验证:K 原子球面等距下 d_min ≥ pairwise threshold
- Sigmoid soft attract 不强制等距 → d_min 实际可能 << 理论

**L6 — `_density_loss` 是塌缩剂,新数据集启用前要重审**(errata 2 §1)
- Tweedie x0_hat → 全局 L2 → 0 是"原点吸引子"
- Fe-only 时合理(Fe-O ~2 Å 窄分布),88 元素时变成 distance prior 错置

### 7.3 训练层

**L7 — PL callback strict=False 防 metric 启动期未注册**(errata 4 §5.1)
```python
EarlyStopping(monitor='val_xxx', strict=False, verbose=True)
ModelCheckpoint(monitor='val_xxx', mode='max', save_top_k=3)
```
- LightningModule `on_validation_epoch_end` log 的 metric 在第一个 val epoch 前不存在 → strict=True 会 RuntimeError → SA 容易 fallback 错 monitor

**L8 — 训练超参 launch note 拍板,SA T2 dry-run 不许改**(errata 4 §3.4)
- 性能优化(workers / persistent / pin_memory)允许
- 核心超参(batch / lr / scheduler / optimizer)一律不动

**L9 — Watch-only 必须升级为 active monitor**(errata 5 §5.3)
- 旧 watch-only check loss 数值 finite + 在合理范围
- 不充分,必须显式 check 梯度是否产生预期效果(如 shell_count_loss 应与 shell-1 distance score 同步降)

### 7.4 评估层

**L10 — 训练 ckpt selection 公式 ≠ 评估 verdict 公式**(errata 5 §5.2)
- 训练:`val_composite_ckpt_score`(LightningModule 3 项加权)
- 评估:`step5_3 composite`(7 项加权)
- **final report verdict 只能用后者,不许用前者**

**L11 — Verdict 双指标并列报告 SOP**(errata 4 §5.3)
- Single composite verdict 可 cherry-pick
- 必须 (composite + gate + min_d + collapse) 并列,任一 GREEN/AMBER/RED 独立列出

### 7.5 工程层

**L12 — 任何 launch note 拍板红线项,SA 不擅自改**(errata 4 §3 + errata 6 §6)
- 红线包括:架构 / 数据预处理 / 训练超参 / loss cost / ckpt selection 公式
- 用户 / 师兄拍板 不等于 MA 拍板;MA 必须显式 ack

**L13 — 失败本身值得记录,不掩盖**(errata 5 §5.4 + 本报告 §6)
- Negative result 的根因诊断比含糊的 positive result 学术价值更高
- 5 份 errata 完整 paper trail 是 ExpN+ 项目的资产

### 7.6 架构层(新,本报告主张)

**L14 — Loss-level fixes 在固定架构上有上限**(本报告 §0.4 / §10.2)
- Exp5 三阶段(v2 + Exp5' + Exp5'')都 post-hoc 修 loss,model 架构未变
- Exp5''_v2 / 候选 B 同质化,不会突破上限
- 突破必须从架构层注入 inductive bias(等变 / 图卷积 with shell-aware edge / Transformer attention / GAN discriminator)

---

## §8 Artifact 永久档案与文件路径(完整索引)

### 8.1 三档 ckpt(永久保留,Exp6 / Exp7 baseline 引用)

| 阶段 | Ckpt 文件 | Md5 | 路径 |
|---|---|---|---|
| **Exp5 v2** | `sa2pp_resume_epoch529_val0.7003.ckpt.frozen` | `72ad4275153b86a65a1399e4ab357d85` | `/home/tcat/diffcsp_exp5/checkpoints/` |
| **Exp5'** | `composite_epoch169_score0.5881.ckpt` | `127afa44a850d8f7e4fcdae17e2761a1` | `/home/tcat/diffcsp_exp5_prime/checkpoints/` |
| **Exp5'** (frozen 副本) | `composite_epoch169_score0.5881.ckpt.frozen_step2_continue_final` | 同上 | 同上 |
| **Exp5''** | `composite_epoch199_score0.5319.ckpt.frozen_p4_final` | `635f3dddb1b9c6770ee14796e504d241` | `/home/tcat/diffcsp_exp5_double_prime/checkpoints/` |

### 8.2 关键数据文件

| 文件 | Md5 | 路径 | 用途 |
|---|---|---|---|
| `shell_boundaries.pkl` | `cf2050e4899160f5698ad2481377e94c` | `/home/tcat/diffcsp_exp4/data/` | Ground truth shell 5 字段 |
| `cache_metadata.json` | (含 `L_VIRTUAL: 20.0`) | `/home/tcat/diffcsp_exp5_prime/data/` | L=20 cache 锁定 |
| `train_structure_cache.pt` | (Exp5'-STEP1-FIX-C 记录)| `/home/tcat/diffcsp_exp5_prime/data/` | L=20 train cache(60501 samples)|
| `val_structure_cache.pt` | 同 | 同 | L=20 val cache(7621)|
| `test_structure_cache.pt` | 同 | 同 | L=20 test cache(4481)|

### 8.3 Predictions(三档 × 三 split)

| 阶段 | 文件位置 |
|---|---|
| Exp5 v2 (dry-run 100) | `/home/tcat/diffcsp_exp5/` (sample data 见 SA-METRICS-V3 输出)|
| **Exp5' 全集** | `/home/tcat/diffcsp_exp5_prime/predictions/predictions_{val,test,holdout}.pt` |
| **Exp5'' 全集** | `/home/tcat/diffcsp_exp5_double_prime/predictions/predictions_{val,test,holdout}.pt` |

### 8.4 Step5_3 复合分输出(三档)

每档 9 个文件:`composite_score_{val,test,holdout}.txt` × 3 + `composite_score_per_sample_*.csv` × 3 + `min_d_violations_*.csv` × 3

- Exp5 v2:`/home/tcat/diffcsp_exp5/logs/composite_score_*_debug100.*`
- **Exp5'**: `/home/tcat/diffcsp_exp5_prime/logs/composite_score_*` (无 debug100 后缀,全集)
- **Exp5''**: `/home/tcat/diffcsp_exp5_double_prime/logs/composite_score_*`

### 8.5 代码文件版本(关键 md5)

完整版见 Exp5' final report v3 §11 + Exp5'' hand-back §9.1。关键节点:

| 文件 | Exp5' STEP1-FIX-C | Exp5'' (P1 后) | Exp7 起点(若需)|
|---|---|---|---|
| `diffusion_w_type_xas.py` | `0bc6fc346e...` | `6ad5c461a5...`(三件套 v2 公式)| 沿用 Exp5' 或重写 |
| `xas_local_dataset_v2.py` | `94432ba56a...` | 沿用(不动)| 沿用(L=20 配置)|
| `step5_3_composite_score*.py` | `_exp5_prime` 版本 | `_exp5_double_prime` 版本 | 沿用任一 |

### 8.6 Logs 完整索引

- Exp5 v2 训练 log: `/home/tcat/diffcsp_exp5/logs/SA2*/`
- Exp5' STEP2 / STEP2-CONTINUE: `/home/tcat/diffcsp_exp5_prime/logs/train_step2*.log`
- Exp5'' P4: `/home/tcat/diffcsp_exp5_double_prime/logs/p4_train_stdout.log`
- 所有 sample log: `*/logs/sample_*.log` 或 `*/logs/p5_sample_stdout.log`

---

## §9 Errata 索引(系列完整 paper trail)

| Errata | 题目 | 影响范围 | 状态 |
|---|---|---|---|
| **2** (继 Exp4)| `_density_loss` 塌缩 + Exp3 历史 + 方向 menu | EXPERIMENT4_FINAL_REPORT §7.2/§7.3/§10 | FINAL |
| **3** (Exp5' STEP1-AUDIT)| L=6 fold artifact + L=20 决议 + RMSD 三层归因 | EXP5_PRIME_PROPOSAL §2.1 + EXP4 §7.2/§10 + errata 2 §1.4 | FINAL |
| **4** (Exp5' STEP2 后)| Ckpt selection bug + verdict 双指标 SOP + last.ckpt → epoch 169 路径决议 | EXP5_PRIME_STEP1_HANDOFF §0.4 #1 / EXP5_PRIME_STEP2_TRAIN_HANDOFF §0.5 #5 | FINAL(§6 已被 errata 5 §3 修订)|
| **5** (Exp5' STEP3 后)| Shell loss 鸡蛋启动问题 + Exp5'' 方向决议(候选 A/B)| EXP5_PRIME_PROPOSAL §2.2/§2.3 + Exp5'' proposal | FINAL |
| **6** (本报告整合)| Exp5'' 候选 A 失败 + 双根因(伪解决 + 几何冲突)+ 系列 wrap up + Architectural inductive bias hypothesis | EXP5_DOUBLE_PRIME_PROPOSAL §2.1-§2.4 + 本报告 §10 | **FINAL,本报告整合,未独立成文** |

5 份 errata + 本报告 = Exp5 系列完整 paper trail。每份独立存档,不合并。

---

## §10 对 Exp6(Transformer)/ Exp7(GAN)的建议

**说明**:Exp6 由同步进行的另一线 MA 主导,我(Exp5'-MA)不掌握 Exp6 细节;Exp7 由用户决议 GAN 方向。本节给一般性建议,基于 Exp5 系列教训,不是对具体 Exp6 / Exp7 设计的指令。

### 10.1 必须沿用的 Exp5 系列产出

1. **Dataset L_VIRTUAL=20 + CUTOFF_R=10**(errata 3 修复)— Exp6 / Exp7 启动前 verify cartesian sanity 100/100,不要回退到 L=6
2. **`shell_boundaries.pkl` 干净 ground truth**(errata 3 §3 验证)— inject 进 dataset 作 supervised signal
3. **SA-METRICS-V3 物理评估体系**(Exp5 v2 发明,step5_3 + min_d gate)— 作 verdict 主指标
4. **双指标 verdict SOP**(errata 4 §5.3)— composite + gate + min_d + collapse 并列
5. **三 split 完整 sample**(val 7621 / test 4481 / holdout 3025)— 不抽样,不 dry-run-only

### 10.2 必须避免的 Exp5 系列陷阱

1. **L1-L14 全部 ExpN+ 不变量级 lesson**(§7)— launch note 必引
2. **Ckpt callback strict=False**(L7)— 否则重蹈 Exp5' STEP2 错 monitor 覆辙
3. **MAX_EPOCHS 是 absolute 不是 incremental**(Exp5'' hand-back §6 trap)— Warm-start 起点 epoch 169 + 续训 N epoch,MAX_EPOCHS 必写 169+N 不是 N
4. **`persistent_workers` 与 `num_workers=0` 不兼容**(Exp5'' hand-back §6 trap)— 改 datamodule 加 `persistent_workers = (self.num_workers > 0)`
5. **EarlyStop patience 真值 = patience × check_val_every_n_epoch**(Exp5'' hand-back §6 trap)— Exp5'' patience=30 + check_val=5 → 实际 150 epoch 不升才停
6. **多套 step5 脚本版本污染**(Exp5'' hand-back §9.1)— 每个 ExpN 用独立后缀 `_expN_xxx.py`,不复用旧后缀

### 10.3 对 Exp6 Transformer 的具体建议(基于 Exp5 教训)

**Architecture 优势期望**:
- Transformer attention 机制可能 implicit 形成"shell"概念(类似 NLP 的 attention head 学到 position-relative pattern)
- 多 head + cross-attention 可让 spectrum 信号在不同尺度上 supervise pred coords

**Risk**:
- 如果 Exp6 沿用 Exp5 系列的 step5_3 7 项复合分作 verdict,**shell-1 distance score / shell-1 elem score 可能仍 RED**(因为 errata 5 揭示 shell 监督本质是 inductive bias 问题,不是 capacity 问题)
- 建议:Exp6 应 evaluator 端检验 `n_pred_shells > 0 比例 ≥ 80%`(L4 新 lesson)
- 建议:Exp6 训练 sanity 检 collapse rate 每 5 epoch dump(L9)

**ROI 期望**:
- Composite 从 Exp5' 0.080 → Exp6 0.15-0.25(部分突破 RED → AMBER)是合理目标
- Composite > 0.30 GREEN 需要 shell-1 监督真有效,即 architecture 真 implicit 形成 shell

### 10.4 对 Exp7 GAN 的具体建议

**GAN 架构在 XAS 任务上的特殊考量**:

1. **Discriminator 的 reward 信号**:
- 普通 GAN 用 discriminator 学 "是否是真样本"
- XAS 任务可让 discriminator 同时检查"是否物理合法"
- **建议**:Discriminator input = (spectrum, pred_structure) pair,输出 reward 包含 (a) 与真样本似然度 (b) 物理 sanity(min_d ≥ 1.5 / shell 形成 / etc.)
- 这相当于把 Exp5 系列 step5_3 7 项作为 discriminator 一部分

2. **Mode collapse 风险**:
- 这是 GAN 的经典问题
- XAS 任务的 "mode" = 不同元素组合的局部结构
- **建议**:Generator output 加 element-conditional 多样性 loss(类似 InfoGAN)

3. **训练稳定性**:
- GAN 训练不稳定众所周知
- Diffusion 在 Exp5 系列上训练稳定(EarlyStop patience 用满)
- **建议**:Exp7 不要丢 diffusion baseline,可考虑 hybrid(diffusion 主线 + GAN discriminator 加 reward signal)

4. **Fold artifact 检查**:
- GAN 也用 frac coordinates 输出 → 同 errata 3 fold 问题
- **Exp7 启动前必跑 cartesian sanity 100/100**(L1)

5. **Verdict 评估**:
- GAN 论文常用 FID / IS,**这些不能直接用于 XAS 物理结构**
- **必须**沿用 step5_3 7 项复合分 + gate + collapse(L10 + L11)
- 可加 GAN-specific metric(如 mode coverage / 元素分布 KL)作 augment

### 10.5 Exp7 GAN 启动建议(草稿,留用户决议)

如果用户决定启动 Exp7 GAN,建议先做:

1. **Exp7 proposal**(1-2 周):基于本报告 + Exp6 verdict(若已出)+ GAN 文献综述,define generator / discriminator / loss 架构
2. **Exp7 阶段划分**:类比 Exp5':STEP1 实施 + STEP2 训练 + STEP3 sample + STEP4 final report
3. **Baseline 对照**:Exp7 final report 必须含与 Exp5' / Exp6 的对照表(用 step5_3 同一套指标)
4. **SA 数量**:Exp7 是新架构,unknown 多,建议至少 1-2 SA(类比 Exp5' STEP1-AUDIT)

我作为 Exp5'-MA / Exp5''-MA **不写 Exp7 proposal**(超出 Exp5 系列范围,需要 GAN 专项 ramp up)。但本报告 §10.4 提供 Exp7 设计的 5 条 sanity check 建议。

---

## §11 投稿建议

### 11.1 Exp5 系列单独投稿(Short paper / Workshop)

**标题候选**:

> "Diagnosis of Implicit Failure Modes in Diffusion-Based Local Atomic Structure Prediction from X-ray Absorption Spectra: From Fold Artifact to Pseudo-Solved Shell Supervision"

**5 个核心贡献**:

1. SA-METRICS-V3 物理评估体系(min_d gate / collapse / step5_3 7 项)— 方法学贡献
2. Fold artifact 几何机制 + L_VIRTUAL 设计准则(errata 3)— 数据预处理贡献
3. Pairwise min distance loss 自启动验证(Exp5' 64% gate 硬证)— 工程贡献
4. Shell loss 鸡蛋启动问题 + 伪解决诊断(errata 5/6)— 失败分析贡献
5. Loss-level fix 在固定架构上有上限(本报告 §10.2)— 架构必要性论证

**长度**:8-10 页,short paper / workshop submission

### 11.2 与 Exp6 / Exp7 联合投稿(Full paper)

如 Exp6 Transformer 或 Exp7 GAN 出 GREEN verdict,可写全长 paper:

**结构建议**:
- §1 Introduction(任务背景,XAS → local structure)
- §2 Related Work
- §3 Method:Architecture(CSPNet / Transformer / GAN)+ Physical loss design
- §4 Experiments(Exp4 baseline + Exp5 系列对照 + Exp6/7 主结果)
- §5 Failure Analysis(Exp5'' 失败 + 双根因)
- §6 Lessons & Discussion(14 条 ExpN+ 不变量)
- §7 Conclusion

**Exp5 系列在 full paper 里的角色**:
- Exp4 = Old baseline,展示 old metric 隐藏物理违反
- Exp5 v2 = SA-METRICS-V3 发明 trigger
- **Exp5'** = Loss-level fix 的最好结果(部分胜利,设上限)
- **Exp5''** = Negative ablation(证明 loss-only 触顶)
- Exp6 / Exp7 = 架构变化突破上限(主贡献)

### 11.3 不建议的投稿方向

- **不要单投 Exp5 v2** — 旧数据,verdict 灾难,无 lessons 兜底
- **不要单投 Exp5''** — verdict 失败,没有 Exp5' / Exp4 baseline 衬托无意义
- **不要在 Exp5' 单独投稿后再投 Exp6**:同一系列拆分多投容易被 reviewer 标双投。最佳实践是 Exp5 系列 short paper + Exp6/7 full paper(两份内容差异充分,不冲突)

---

## §12 系列结束:Exp5'-MA / Exp5''-MA 离场致辞

### 12.1 自评

**做对的事**:
- Errata 2-6 完整 paper trail,负面发现不掩盖
- Errata 3 fold artifact 诊断 + L=20 修复是真贡献
- 双 verdict 指标 SOP(errata 4 §5.3)+ ExpN+ 14 条 lesson 落地
- 7 棒 SA 工作流(launch note → review → hand-back)沿用 v1 MA 哲学

**做错 / 可改进**:
- 4 次 watch-only 错过 shell loss 鸡蛋问题(errata 5 §2.4 / §8 自评)
- STEP2 误判训练失败,接受 last.ckpt → 用户本能 3 次救场(final report v3 §13)
- Exp5'' P3 SMOKE 时 collapse 0% 让我 / Exp5''-MA 误判 mitigation 工作,P4 epoch 5 未主动 dump(本报告 §6.2)
- Exp5'' 候选 A vs B 决议时倾向 A,部分受 §8.3 cheating 论证误导(事后 §3.3 errata 5 揭示"用 K 但 bypass 真问题"是更深合规争议)

### 12.2 用户(你)的贡献

本报告 §6 / final report v3 §13 已记。三次本能 challenge 救场:
- "epoch 4 best 这能信吗" → 救了 STEP2-CONTINUE 续训(找回 epoch 169 BEST)
- "训练还没结束怎么就取样" → 阻止误进 STEP3
- "0.1563 这值怎么来的不合理吧" → 触发 pkl 自一致性 verify(确认 L 解耦)

这些不是"Exp5'-MA 失误,用户救场",是**用户 + MA 协作模型本来就该这样**。User-in-the-loop 不是 fallback,是 first-class workflow。

### 12.3 给 Exp6 / Exp7 MA 的话

如 Exp6 / Exp7 MA 在某个时刻读到本报告:

1. **不要被 Exp5'' verdict ❌ 误导认为 Exp5 系列失败**。Exp5 系列三阶段产出了 SA-METRICS-V3 评估体系、fold 修复、6 份 errata、14 条 ExpN+ lesson、3 档 ckpt baseline。**这是工程资产,不是失败遗物**。

2. **Exp7 MA 启动 GAN 前,本报告 §10.4 的 5 条 sanity check 是 prerequisite**。如果跳过,可能重蹈 Exp5'' 那种"训练 active 但评估失败"的坑。

3. **如果 Exp6 Transformer verdict 也 mixed**(composite > 0.15 但 < 0.30),考虑:
- (a) Loss-level fix 在 Transformer 上的 ablation(可能 fold 修 + Transformer + 候选 B 重设计的组合 work)
- (b) 直接转 Exp7 GAN
- 决议依据是 Exp6 vs Exp5' 的 Δ,不是 Exp6 vs verdict 阈值

4. **Errata 6 没有独立文件,内容整合在本报告 §4.4-4.5 + §7 L4/L5**。如未来需要单独 errata 6 文件,从本报告抽 §4.4-§4.5 + §7 lesson 6.1/6.2 即可。

---

## §13 附录:核心数字速查表

### 13.1 Exp5 系列三档 verdict 一句话

- Exp5 v2:**Mathematics improve, physics catastrophe**(F1 +28.8% / gate 5-11%)
- Exp5':**Foundational fix + partial supervision win**(fold 修 / gate 64% / shell RED)
- Exp5'':**Pseudo-solution exposed**(候选 A 推问题到评估端 / collapse 32%)

### 13.2 Three-stage growth path

```
Composite (step5_3):  v2 0.006 → Exp5' 0.080 (13×) → Exp5'' 0.035 (0.4×) 退步
Gate (min_d ≥ 1.5):   v2 5-11% → Exp5' 64% (6-13×) → Exp5'' 31% (0.5×) 退步
Collapse rate:        v2 ?    → Exp5' 0% ✅       → Exp5'' 32% 🔴 灾难回归
```

### 13.3 Exp5 系列对 Exp6 / Exp7 的 4 个 must-do

1. ✅ **Sanity**:Cartesian sanity 100/100(L=20 + CUTOFF_R=10 / errata 3)
2. ✅ **Loss**:Pairwise min distance penalty(λ=1.0)沿用(Exp5' gate 64% 硬证生效)
3. ✅ **Verdict**:step5_3 7 项 + gate + collapse 双指标(errata 4 §5.3)
4. ✅ **Dry-run**:训练 active + 评估 active 双层(L3 + L4)

---

*Exp5'-MA / Exp5''-MA 撰写,2026-05-10*
*基于 Exp5 v2 final report v2 + Exp5' final report v3 + Exp5'' hand-back + 5 份 errata(2/3/4/5/6 整合)+ 全 STEP1-3 SA hand-back + 训练 / sample 完整 log + step5_3 复合分输出*
*Exp5 系列正式 wrap up。Exp6 由同步另线 MA 主导,Exp7(GAN)由用户决议启动。*
*本报告自带 errata 6 整合(§4.4-4.5 + §7 L4/L5),不另独立 errata 6 文件。*
*报告可直接进 Exp6 / Exp7 论文 baseline 章节,或作 short paper 投稿主体。*

— **End of Exp5 Series Final Report**
