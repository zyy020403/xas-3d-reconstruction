# EXPERIMENT5_FINAL_REPORT_v2.md
# Exp5 v2 — FINAL REPORT (v2.0, frozen with SA-METRICS-V3 verdict)

> **撰写者**: MA5(Exp5 v2 Main Agent,移交 Exp5'-MA 前 final 版本)
> **日期**: 2026-05-01
> **取代**: EXPERIMENT5_FINAL_REPORT_v1.md(归档为 _DEPRECATED)
> **状态**: Exp5 v2 阶段任务完成,verdict ❌ physical-invalid,Exp5' from-scratch 启动
> **本文档定位**: Exp5 v2 最终历史档案,所有数据 frozen。Exp5' 完成后另出 final report v3
> **配套**: EXP5_PRIME_PROPOSAL.md / EXP5_FILE_GUIDE_v2.md / EXP5_PRIME_MA_HANDOFF.md

---

## §0 Final Verdict

### 0.1 Exp5 v2 verdict

❌ **PHYSICAL-INVALID**

| 维度 | Exp4 baseline | Exp5 v2 | 评估 |
|---|---|---|---|
| **物理 gate pass rate (min_d ≥ 1.5 Å)** | 未测 | **5-11%** | ❌ 灾难性失败 |
| **Shell-1 distance score** | 未测 | **0.0000(即使 gate-pass 子集)** | ❌ 模型不知 shell 结构 |
| **Composite score 均值** | 未测 | **0.005-0.011** | ❌ proposal §B.5 verdict 表外 |
| Multiset Macro-F1(数学) | 0.0843 | **0.1086(+28.8%)** | ⚠️ 数学评分有 noise 之上的 signal,但物理无效 |
| RMSD(Hungarian min-image)| 1.4849 | 1.4954 | ≈ 持平 |
| Set-Level TypeAcc(数学)| 0.3309 | 0.3408 | ≈ 持平 |
| Collapse Ratio(std-based)| 0.0% | 0.013% | ≈ 持平 |
| Projection Δ RMSD(R_max=5.5 Å fallback)| 未测 | 0.0000 | ⚠️ R_max=5.5 Å fallback bug,实际无诊断价值 |

### 0.2 Verdict 原因分解

**Exp5 v2 训练有效但不充分**:
- ✅ 架构改造(MV-attention)+ center embedding + cost_density 减弱组合,数学指标 +28.8% 真信号
- ✅ SA2 → SA2'' 续训 LR warm restart 验证有效(val_loss 再降 0.88%)
- ❌ **训练目标完全没约束原子两两距离**,模型学了"原子在 box 内"但没学"原子互相不重合"
- ❌ Exp4 Step 2.5 已有的 shell_boundaries.pkl ground truth **从未真正进训练 loss**(当时只用于评估)
- ❌ SA1' 设计的 4 个 v2 metric 全部不检测原子两两距离(评估盲区)
- ❌ MA5 review SA1' 设计时未质疑指标 menu 完备性

**简言之**: Exp5 v2 在 ML 第一原理上的 root cause 是 "训练目标没要求的事,模型不会自己学"。物理 loss 项 + Step 2.5 ground truth 必须真正进训练 → Exp5'。

### 0.3 不是"作废",是"warm-start 起点"?(用户 2026-05-01 拍板)

**用户决策**: from-scratch 重训,**不 warm-start**。理由:
- SA2 epoch 529 ckpt 已"学坏"(95% 输出重合),warm-start 要先"忘"再"学"
- from-scratch + 物理 loss 从 epoch 0 加,设计更干净
- 工程上 ~32-40h 训练 vs ~14-19h 续训,~ 20h 差,但去掉"学坏"代价值得

**结论**: Exp5 v2 ckpt 不是 Exp5' warm-start 起点,**只是历史档案 + verdict 锚点**。

---

## §1 Exp5 v2 接力链时间轴

| 日期 | Agent | 关键事件 |
|---|---|---|
| 2026-04-28 | v1 SA1+SA2 | 复刻 Exp3 head 设计,SA2 训到 epoch 36 collapse 后 kill |
| 2026-04-28 | v1 MA self-audit | MA 工作哲学 5 条产生 |
| 2026-04-28 | MA5 + 用户 | v2 proposal 撰写,锁定 MV-attention + cost_density 0.2 |
| 2026-04-28 | SA1' | 架构 surgery,forward_test 5/5 PASS + 1 SKIPPED,smoke PASS |
| 2026-04-28 → 04-29 | SA2' | 28h from-scratch 训练,best epoch 484 val_loss 0.7065 |
| 2026-04-29 → 04-30 | SA3' | 3.5h sample val+test,Multiset F1 0.1086(val)/ 0.1096(test)|
| 2026-04-30 | SA2'' (ssh-only) | 11h 续训,best epoch 529 val_loss 0.7003,early stop @ epoch 679 |
| **2026-05-01 早** | **用户** | **物理统计发现 min_d < 1.5 Å 大量违反** |
| 2026-05-01 中 | MA5 | EXP5_PROPOSAL_v2_AMENDED + final report v1 + file guide + MA2 handoff |
| 2026-05-01 中 | Exp5 MA2 + SA-METRICS-V3 | 写 step5_3_composite_score.py + dry-run 100 样本 ×2 split 验证灾难 |
| **2026-05-01 晚** | **用户 + MA5** | **决议 from-scratch 重训 → Exp5' 启动** |

---

## §2 Exp5 v2 完整产出(永久档案)

### 2.1 服务器代码改动 surface(Exp4 → Exp5 v2)

| 文件 | 状态 | 关键改动 |
|---|---|---|
| step2/spectrum_encoder.py | ✅ active(Exp5 v2)| MV-attention(num_heads=4 + residual_alpha=0.5 + post-LN);chi/feff 末端升 256d;output_dim=272;center_emb cat at end |
| step3/diffusion_w_type_xas.py | ✅ active | 撤 v1 head + 实例化 + 3-mode loss;保留 center_emb + Patch 1 `.to(c0.dtype)`;4-arg encoder 调用 |
| step3/xas_local_dataset_v2.py | ✅ active | v1 SA1 加 center_element_Z 字段保留 |
| step3/xas_local_datamodule_v2.py | ✅ active | v1 SA1 加 LongTensor collate;v1→v2 命名 .train_ds(SA2' α' patch 修过 train.py)|
| step3/conf_xas/model/diffusion_xas.yaml | ✅ active | 删 head 6 字段 + 加 mv_attention.num_heads + cost_density 0.5→0.2 |
| step3/forward_test.py | ✅ active | Phase 6.6 测 MV-attention(view-order invariance 7.45e-9);Phase 6.5 SKIPPED-by-design |
| step4/step4_1_smoke_test.py | ✅ active | v1 4-mode → v2 1-mode |
| step4/step4_2_train.py | ✅ active | SA1' fork + SA2' α' patch + MA5 SA2'' resume hardcode + MAX_EPOCHS line 83 |
| step5/step5_1_sample.py | ✅ active(SA3' fork from Exp4)| 11 项 v2 surgery,硬阻断 holdout |
| step5/step5_2_compute_metrics.py | ✅ active(SA1' + SA3' patch)| 4 个 v2 metric 函数;⚠️ projection R_max fallback bug |
| step5/step5_3_composite_score.py | ✅ active(SA-METRICS-V3 写)| 7 项复合评分 + min_d 1.5 Å gate + per-sample shell_boundaries.pkl |

所有改动 surface 都有 .bak* 锚点保留(详见 EXP5_FILE_GUIDE_v2.md)。

### 2.2 服务器 ckpt(永久保留)

| ckpt | 大小 | epoch | val_loss | 状态 | 用途 |
|---|---|---|---|---|---|
| `epoch=529-val_loss=0.7003.ckpt` | 44 MB | 529 | 0.7003 | ✅ active | Exp5 v2 best,可作 Exp5'' baseline |
| `last.ckpt` | 44 MB | 679 | n/a | ✅ active | SA2'' 训练自然终点 |
| `sa2_baseline_epoch484_val0.7065.ckpt.frozen` | 44 MB | 484 | 0.7065 | 🔒 frozen | SA2' best 永久 safety net |
| `sa2pp_resume_epoch529_val0.7003.ckpt.frozen` | 44 MB | 529 | 0.7003 | 🔒 frozen | SA2'' best 永久 safety net |

**Exp5 v2 ckpt 不删,作历史档案**。Exp5' 训完独立 ckpt 落 `/home/tcat/diffcsp_exp5_prime/checkpoints/`。

### 2.3 服务器 predictions / metrics(永久保留)

| 文件 | 大小 | 用途 |
|---|---|---|
| `step5/predictions_v2_val.pt` | 9.8 MB | SA3' sample 输出(SA2 baseline epoch 484)|
| `step5/predictions_v2_test.pt` | 5.8 MB | 同 test |
| `logs/v2_val_metrics.txt` / `_test.txt` | — | SA1' 4 个 metric 主报告 |
| `logs/v2_val_per_sample.csv` / `_test.csv` | — | 7621 / 4481 行 |
| `logs/v2_projection_ablation_val.log` / `_test.log` | — | ⚠️ R_max=5.5 fallback,实际 0 诊断价值 |
| `logs/composite_score_val_debug100.txt` / `_test.txt` | — | SA-METRICS-V3 dry-run 主报告(灾难锚点)|
| `logs/composite_score_per_sample_{val,test}_debug100.csv` | — | 100 + 100 = 200 样本完整记录 |
| `logs/min_d_violations_{val,test}_debug100.csv` | — | ⭐ Exp5' λ schedule 设计依据 |

### 2.4 v2 vs Exp4 完整对照表

| 指标 | Exp4 val | Exp4 test | Exp5 v2 val | Exp5 v2 test | Δ val |
|---|---|---|---|---|---|
| RMSD (Å)(Hungarian min-image)| 1.4849 | 1.4852 | 1.4954 | 1.4928 | -0.7‰ 微退 |
| pred_in_cutoff (/20)| 18.93 | 18.93 | 18.92 | 18.94 | 持平 |
| Set-Level TypeAcc | 0.3309 | 0.3330 | 0.3408 | 0.3397 | +3.0% |
| Multiset Macro-F1 | 0.0843 | 0.0846 | **0.1086** | **0.1096** | **+28.8%** |
| Position-by-position TypeAcc[VIRTUAL] | 0.1877 | 0.1877 | 0.1979 | 0.1969 | +5.4% (虚假) |
| Collapse Ratio (std-based) | 0.0% | 0.0% | 0.013% | 0.000% | ≈ 持平 |
| **min_d gate pass rate** | **未测** | **未测** | **5%(N=100)** | **11%(N=100)** | **未知** |
| **Composite total mean** | **未测** | **未测** | **0.0056** | **0.0062** | **未知** |
| **Shell-1 distance score (gate-pass subset)** | **未测** | **未测** | **0.0000** | **0.0000** | **未知** |

---

## §3 Exp5 v2 vs Exp4 — 用 evidence-based 框架重新审视

### 3.1 数学评估指标:Exp5 v2 真有改进

**+28.8% Multiset F1** 是统计显著真信号(val/test 一致 +28.8%/+29.6%,not noise)。架构改造(MV-attention)+ center embedding + cost_density 减弱**作为整体集合**有效。

**但**: v2 没做 architecture ablation,**MV-attention vs center embedding vs density 减弱各自贡献分离不出来**。我作为 MA5 倾向猜测:MV-attention 占 30-50%,center_emb 占 30-50%,density 减弱 + 续训占 10-20%。**纯属猜测**。

### 3.2 物理评估指标:Exp5 v2 vs Exp4 不能直接比较

Exp4 时代根本没测 min_d gate / shell distance score。**强烈怀疑 Exp4 也是 80-95% min_d 违反**(同样的 `_density_loss` 把原子推向中心,RMSD 1.49 在 box 半对角 5.2 Å 框架下推算两两距离分布的概率论)。

**没有 Exp4 的物理对照数据**,**说"Exp5 v2 物理上比 Exp4 好/坏"是 unfounded**。

### 3.3 Exp5'-MA 决议:Exp5' 训完后是否补跑 Exp4 物理对照?

**强烈建议**: SA-EXP5'-sample 阶段顺便用 Exp4 best ckpt(`/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt`)重 sample → step5_3 全量,产出 Exp4 物理 baseline。这样 Exp5' final report v3 才有真正的 Exp4 vs Exp5 v2 vs Exp5' 三方物理对照。

**额外成本**: ~3.5h sample + ~10 min metrics。**强 ROI**(否则 Exp5' verdict 只能 vs Exp5 v2,vs Exp4 永远是空白)。

### 3.4 单视角 vs 多视角:用户问的核心问题

**用户原问**: "如果完全没提升那 MV-attention 就没必要"

**MA5 回答**:

1. **Exp5 v2 +28.8% Multiset F1 是真改进,不是 noise** — 这点确认
2. **但 MV-attention 是否是改进的主要功劳,Exp5 v2 没做 ablation 无法证实** — 不能确认
3. **Exp5'(本提案)沿用 MV-attention,因为**:
   - 物理 loss 加进去是正交关注点,不应同时拆架构
   - warm-start 不可行(用户拍板 from-scratch),但保 architecture consistency 让 v2 vs v5' 对照"只比物理 loss"
4. **如 Exp5' 复合分 ≥ 0.40 + gate pass ≥ 80% 成功** → 之后开 Exp5'' 做 ablation:**Exp5'' = Exp4 architecture + center_emb + cost_density 0.2 + 物理 loss 三件套(去 MV-attention)**,~30h from-scratch,产出 architecture 贡献 isolation
5. **如 Exp5' 失败** → 不做 Exp5'' ablation(没意义,先解决 architecture 之外的问题),直接 Exp6

**Exp5'-MA 在 Exp5' verdict 后决定 Exp5'' / Exp6,本 final report 不预设**。

---

## §4 数据处理沿用清单(Exp5' 关键参考)⭐

用户 2026-05-01 提的核心问题: "exp5'数据处理的部分可能不能沿用之前的,要大改"。

MA5 详细分类:

### 4.1 完全不能沿用 — Exp5' 必须重新做

| 项 | Exp4 / Exp5 v2 现状 | Exp5' 必须 |
|---|---|---|
| **min pairwise distance 1.5 Å 训练约束** | 完全无 | **新加 loss 项 `cost_pairwise_min`(λ=1.0 起步)** |
| **Shell-1 / Shell-2 真实距离训练约束** | 完全无 | **新加 loss 项 `cost_shell_dist`(λ=0.5 起步)** |
| **Shell-1 / Shell-2 配位数训练约束** | 完全无 | **新加 loss 项 `cost_shell_count`(λ=0.2 起步)** |
| **Shell ground truth inject 进 dataset** | 完全无(shell_boundaries.pkl 仅 evaluate 用)| **新加 dataset/datamodule inject 5 字段** |
| **Best ckpt selection criterion** | val_loss 单一指标 | **复合 α·val_loss + β·gate_pass_rate + γ·composite_total**(α/β/γ = 0.2/0.5/0.3)|

### 4.2 半沿用 — Exp4 / Exp5 v2 已有,但 Exp5' 用法不同

| 项 | Exp4 / Exp5 v2 用法 | Exp5' 用法 |
|---|---|---|
| **`shell_boundaries.pkl`(387 MB)** | 仅 evaluate 用(step5_2 stratified RMSD)| **进训练 loss(§4.1)** |
| **`step5_3_composite_score.py`(SA-METRICS-V3 产出)** | dry-run 100 样本验证灾难 | **Exp5' 训完直接重跑全量,主指标** |
| **MV-attention 架构** | Exp5 v2 引入 + 28.8% Multiset F1 | **Exp5' 沿用,不 ablation**(避免引入太多变量)|
| **Center embedding(95×16d)** | Exp5 v2 引入 | **Exp5' 沿用** |
| **cost_density: 0.2** | Exp5 v2 减弱(0.5 → 0.2)| **Exp5' 沿用** |
| **训练超参(LR/batch/epochs/optimizer)** | Exp5 v2 SA1' 设定 | **Exp5' 沿用,但 best ckpt criterion 不同(§4.1)** |

### 4.3 完全沿用 — Exp5' 不动

| 项 | 来源 |
|---|---|
| L=6 / [-0.5, 0.5] / min-image 折叠 | Exp2 step4d |
| split 60507/7624/4481/3025 | Exp4 |
| N_NEIGHBORS=20 | Exp4 |
| 邻居搜索半径 10 Å | Exp2 |
| FEFF feature 维度 74 | Exp4 |
| holdout 永久封存 | Exp4 → Exp5 全程 |
| incompat_pool.csv | Exp4 |
| 7 守卫包(scikit-learn 1.7.2 / numpy 2.2.6 / scipy 1.15.3 / pymatgen 2025.10.7 / torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / torch-scatter 2.1.2+pt24cu124)| Exp5 全程 |
| Phase 6.5 SKIPPED-by-design fp32 hardcoded | v1 SA1 OUTPUT §5.7 |
| Step 2.5 gap=0.1563 Å threshold(壳层切分阈值)| Exp4 MA2 拍板 |
| step5_1_sample.py 11 项 v2 surgery | SA3' |
| Smoke test / forward_test pattern | SA1' |

### 4.4 不沿用,**但作历史档案保留**

| 项 | 状态 |
|---|---|
| Exp5 v2 ckpt(epoch 529 + last + 2 frozen)| 永久保留 `/home/tcat/diffcsp_exp5/checkpoints/`,不删 |
| Exp5 v2 predictions_v2_*.pt | 永久保留作历史 baseline |
| Exp5 v2 metrics report(SA1' 4 metric)| 永久保留作 v2 数学评分档案 |
| step5_2_compute_metrics.py | 留作 v2 历史,Exp5' 主指标用 step5_3 |
| step5_2 投影 ablation R_max=5.5 fallback bug | **不修**(锚点 bug,作 lessons learned 案例)|

### 4.5 用户提的"Exp4 数据处理可能也有问题" 评估

用户原话: "exp4 的数据处理可能也有问题,没有达到我要的第一壳层第二壳层根据真实数据统计的划分加约束,也没有 1.5 的约束就训练了"

**MA5 分析**:
- ✅ Step 2.5 gap 算法 + p10=0.1563 Å threshold 是数据驱动的,**Exp4 这部分做对了**(MA2 拍板,产出 shell_boundaries.pkl)
- ❌ Exp4 把 shell_boundaries.pkl 当 evaluate-only 用,**没有进训练 loss** — 这是真问题,Exp5' 修复
- ❌ Exp4 / Exp5 v2 都没 1.5 Å pairwise 约束 — 这也是真问题,Exp5' 修复

**结论**: Exp4 data processing 算法本身正确,但**应用方式有缺**(评估而非训练)。Exp5' 把 shell_boundaries.pkl 真正进训练 loss 是正确补丁,**不需要重新跑 Step 2.5**。

---

## §5 已知 bug / 工程债务(留给 Exp5'-MA 知情)

### 5.1 SA1' 投影 ablation R_max=5.5 Å fallback bug
- 位置: `step5_2_compute_metrics.py::compute_projection_ablation_rmsd`
- 行为: 该 fallback 到全局 5.5 Å 而不是 per-sample 真实 R_max
- 后果: SA3' 跑出 Δ=0 / 0 atoms_projected,被解读为"v2 真物理改进",实际只证明 pred 在 box 内
- **MA5 决议: 不修**(留作 lessons learned 锚点;Exp5' 主指标用 step5_3,不依赖 step5_2)

### 5.2 MAX_EPOCHS 在 train.py 不在 yaml
- 位置: `step4_2_train.py` line 83 `MAX_EPOCHS = 500`
- 决策: SA1' 设计(epoch 数是训练 orchestration 不是模型 hyper)
- 后果: 改 max_epochs 改 train.py,LR scheduler T_max=MAX_EPOCHS 自动跟随
- **Exp5'-MA: 沿用此设计,fork train.py 改 MAX_EPOCHS line 83**

### 5.3 LR scheduler T_max 行为
- SA2'' 续训发现: 改 MAX_EPOCHS 500→700 让 epoch 484 处 LR 从 1.25e-6 跳到 22.5e-6(22.5×)
- **Exp5'-MA: from-scratch 训练此问题不存在**,但若 Exp5' 也续训需注意

### 5.4 PL ModelCheckpoint save_top_k=1 删旧 best
- 行为: PL 训练新 best 出现时删旧 active best
- **Exp5'-MA: Exp5' 启动前不需要 cp(新建 /home/tcat/diffcsp_exp5_prime/checkpoints/ 空目录)**

### 5.5 PL Callback `on_validation_epoch_end` 与 `current_epoch == 200` 不触发
- SA2 时代发现 milestone Callback bind val 周期 + check_val_every_n_epoch=5 → epoch 200 不是 val 周期
- **Exp5'-MA: 写 epoch milestone Callback 时用 `on_train_epoch_end` 钩子或一次性 latch**

### 5.6 v1→v2 datamodule API 命名 contract
- v1 SA1 改 `.train_dataset → .train_ds`
- SA2' 已踩(line 219 α' patch)
- **Exp5'-MA: 任何 fork train.py 后用 dm.train_ds(不是 .train_dataset)**

---

## §6 Lessons Learned(写进 ExpN 不变量级)

### 6.1 数学完备 ≠ 物理完备

Set-Level / Multiset Macro-F1 / RMSD / Hungarian / Collapse Ratio 是数学完备的解耦指标,**但物理上不完备**。任何 diffusion 生成原子坐标的 Exp,评估必须包含:
- min pairwise distance ≥ 1.5 Å gate(原子不重合)
- shell-1 distance score(第一配位壳层结构)
- shell-1 配位数 score(配位数化学常识)

写进 EXP4_FINAL_REPORT_ERRATA_2.md §4 Lessons Learned 第 5 条(Exp5'-MA 写 Exp5' final report 时补 push)。

### 6.2 Step 2.5 ground truth 应进训练,不只评估

Exp4 时代花了 7 个 phase 算 shell_boundaries.pkl(387 MB),但只用于评估时 stratified RMSD。**正确用法是 inject 进训练 loss**。Exp5' 是这个 lesson 的 first implementation。

### 6.3 R_max / shell 边界禁用 fallback

SA1' 的 5.5 Å fallback bug 隐藏到 SA3' 投影 ablation Δ=0 才暴露,直到用户 2026-05-01 物理统计才追到根因。任何 ExpN 评估涉及 R_max / shell 边界的,**必须 per-sample lookup**,fallback 是工程上的懒,会隐藏评估盲区。

### 6.4 MA review SA 设计时主动质疑指标完备性

MA5 review SA1' 4 个 metric 函数时只 verify 算法实现是否正确,没问"这 4 项加 Exp4 已有的 RMSD/TypeAcc 共 6 项,够不够覆盖所有 collapse 模式"。**Exp5'-MA / 后续 ExpN MA: review SA 设计时必问指标 menu 完备性**。

### 6.5 step6 picker subset 不是 verdict

SA-METRICS-V3 §2.2 提到的 step6 picker "1% 选样率"等价于"在 99% 失败里捞 1% 假装成功"。**Exp5' / 后续 ExpN final report 必须报告全 7621/4481 样本的 gate pass 率,不允许 picker subset 当 verdict**。

### 6.6 训练目标没要求的事,模型不会自己学

Exp5 v2 的 95% 物理违反不是"bug",是"设计遗漏的逻辑必然" — 模型严格按 loss function 优化,loss 没要求的物理性质模型不会自己学。**ExpN 设计 loss 时,不变量级别先列"模型应该学到什么物理性质",再 invert 到 loss 项**。

### 6.7 用户的物理统计是 ground truth(算法 / metric 都不是)

2026-05-01 用户物理统计发现 95% 违反 → SA-METRICS-V3 dry-run 100 样本确认 → 一致。**用户对实验数据的物理直觉远比 metric 算法可靠**。**ExpN 流程应包含: sample 完成后让用户跑物理 sanity 统计,作 final verdict 必经一步**。

### 6.8 LR warm restart 对续训有真实价值(但 from-scratch 仍然合适)

SA2'' 续训 22.5× LR 跳跃让 val_loss 再降 0.88%,模型确实跳出局部最小。但**Exp5' 用户拍板 from-scratch**,因为 ckpt 已"学坏"95% 重合,warm restart 要先"忘"再"学"代价高。LR warm restart 是续训技术,不是续训必然选择。

---

## §7 Exp5' 启动 4 件套(MA5 移交)

| 文件 | 行数 | 作用 |
|---|---|---|
| **EXP5_PRIME_PROPOSAL.md** | ~ 600 | Exp5' 详细 proposal,三件套 loss + 工作目录决策 + verdict 阈值 |
| **EXPERIMENT5_FINAL_REPORT_v2.md**(本文件)| ~ 500 | Exp5 v2 final 历史档案 + Lessons Learned + 数据处理沿用清单 |
| **EXP5_FILE_GUIDE_v2.md** | ~ 400 | 服务器 / 本地 / ckpt / log 完整索引 + verify 块 |
| **EXP5_PRIME_MA_HANDOFF.md** | ~ 350 | Exp5'-MA 一文上手 + 立即任务清单 |

四份文件落 `/mnt/user-data/outputs/`,用户上传给 Exp5'-MA。

---

## §8 final 移交宣告

**MA5 上下文接近 70% 闸门,主动 transition 到 Exp5'-MA**。

**Exp5 v2 状态**: ❌ Physical-invalid,但训练有效产出永久保留作 baseline + 历史档案 + lessons learned 来源。

**Exp5' 决策已锁定**:
- from-scratch 重训(用户拍板)
- 三件套物理 loss(pairwise + shell_dist + shell_count)
- 沿用 MV-attention + center_emb + cost_density 0.2(架构 v2 整体)
- 工作目录新建 `/home/tcat/diffcsp_exp5_prime/`

Exp5'-MA 接手见 EXP5_PRIME_MA_HANDOFF.md。

---

*MA5 撰写,2026-05-01,移交 Exp5'-MA 前最后一份 final report。
基于 SA1'/SA2'/SA3'/SA2''/SA-METRICS-V3 全程产出 + 用户 2026-05-01 物理统计发现 + from-scratch 重训决策。*
