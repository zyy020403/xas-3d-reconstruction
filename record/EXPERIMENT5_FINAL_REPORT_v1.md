# EXPERIMENT5_FINAL_REPORT_v1.md
# Experiment 5 v2 — Comprehensive Report (MA5 时代,移交 Exp5 MA2 前)

> **撰写者**: MA5(Exp5 v2 Main Agent)
> **日期**: 2026-05-01
> **状态**: SA2'' 续训完成 / SA-METRICS-V3 待启动 / Exp5' 待启动
> **本文档定位**: 留给 Exp5 MA2(Exp5 v2 extension Main Agent)的全 Exp5 v2 历程记录,
>   包括 v1 失败 → v2 启动 → SA1' 架构 → SA2' 训练 → SA3' 评估 → SA2'' 续训 → 物理约束发现
> **前置阅读**: EXP4_FINAL_REPORT.md / EXP4_FINAL_REPORT_ERRATA_2.md / EXP5_PROPOSAL_v2_AMENDED.md
> **配套文档**: EXP5_FILE_GUIDE_FINAL.md(脚本/数据/ckpt 索引)/ EXP5_MA2_HANDOFF.md(Exp5 MA2 启动包)

---

## §0 执行摘要(Exp5 MA2 一屏掌握)

### 0.1 Exp5 v2 做了什么

接续 Exp4(holdout RMSD=1.4866 / Multiset F1 = 0.0846),Exp5 v2 在 v1 失败被 kill 后由 MA5 重启,
设计两主线:
1. SpectrumEncoder fusion 块从 cat→MLP 改为 **MV-attention(MultiheadAttention with learnable query)**
2. `_density_loss` 权重 **0.5 → 0.2**(EXP4_ERRATA_2 §1 揭示 Exp4 collapse 主犯,减弱不删除)

不做的:不加 TypeClassifier head(Exp3+v1 双重证伪)/ 不 fine-tune(decoder 第一层 shape mismatch)/
不 multi-sample averaging(独立任务以后单起)。

### 0.2 三个时间节点 + 关键数字

| 阶段 | 状态 | 关键指标 |
|---|---|---|
| SA1' 架构 surgery + smoke + forward_test | ✅ 5/5 PASS + 1 SKIPPED | param 3,650,388 / view-order invariance 7.45e-9 |
| SA2' from-scratch 训练(28h) | ✅ best epoch 484 | val_loss 0.7065 / val_type_loss 0.00593 / val_coord 0.684 |
| SA3' sample val+test + metrics + 投影 ablation | ✅ 完成 | Multiset F1 0.1086(val)/ 0.1096(test)/ Δ Projection=0.0000 / Collapse 0.013% |
| SA2'' 续训(MAX_EPOCHS 500→700,从 best 续 + LR cosine 跳 22.5×) | ✅ best epoch 529 + early stop @679 | val_loss 0.7003(改进 0.88%) |
| **SA-METRICS-V3 复合评分(Exp5 MA2 启动)** | ⏳ 待启动 | 输出 SA2 baseline 7 项分 + min_d 违反率 |
| **Exp5' 物理约束加 pairwise 1.5 Å penalty 重训(Exp5 MA2 后续)** | ⏳ 待启动 | 输出 final 物理有效 ckpt |

### 0.3 verdict 历史轨迹

| 时点 | verdict | 信号 |
|---|---|---|
| SA3' 完成时 | ⚠️ AMBER + 🟢 geometry strong | Multiset F1 0.1086 vs Exp4 0.0843(+28.8%);Projection Δ=0;Collapse 0.013% |
| SA2'' 续训完成时 | ⏳ pending re-sample | val_loss 0.7003(再降 0.88%);LR warm restart 验证 |
| **2026-05-01 用户物理统计** | ❌ **physical-invalid** | 大量样本 min pairwise distance < 1.5 Å,FEFF 算不动 |

**verdict 修订**: 几何 / type 信号 vs 物理有效性是两个独立通道。
v2 在前两个通道部分胜利(geometry GREEN, type AMBER),
**第三个通道(物理两两距离)失败,需要 Exp5' 加 pairwise penalty 修复**。

---

## §1 v1 失败前史(继承自原 EXP5_PROPOSAL_v2)

Exp5 v1(独立 MA + SA1+SA2)被用户 kill,根因:
1. SA1+SA2 复刻 Exp3 已经砸过盘的 TypeClassifier head 设计
2. SA2 训练到 epoch 36 出现 head collapse 后 kill
3. v1 MA self-audit 暴露:用户意向被前 MA 的"推荐"覆盖 → MA 工作哲学 5 条产生(EXP5_PROPOSAL_v2.md 头部)

v1 产出已清理或归档,本 Exp5 v2 全程**不依赖 v1 任何东西**。

---

## §2 SA1' 架构 surgery(2026-04-28)

**输入**: v1 SA1 已改 7 文件 + bak_exp4 锚点。**输出**: v2 重命名继承策略(同名文件不带 _exp5 后缀)。

### 2.1 改动 surface

| 文件 | 改动 |
|---|---|
| `step2/spectrum_encoder.py` | 撤 v1 head → 加 MV-attention fusion(num_heads=4, residual_alpha=0.5 固定 + post-residual LN);chi/feff 末端升至 256d |
| `step3/diffusion_w_type_xas.py` | 撤 TypeClassifierHead 类 + 实例化 + 3-mode loss + head_predict_types;保留 center embedding + Patch 1(`F.one_hot(...).to(c0.dtype)`)+ 4-arg encoder 调用 |
| `step3/conf_xas/model/diffusion_xas.yaml` | 删 head 字段 6 个 + 加 mv_attention.num_heads/residual_alpha + cost_density 0.5→0.2 |
| `step3/forward_test.py` | 改 Phase 6.6 测 MV-attention(组件存在 / shape (4,272) / view-order invariance / cost_density yaml 加载) |
| `step4/step4_1_smoke_test.py` | v1 4-mode → v2 1-mode |
| `step3/xas_local_dataset_v2.py` | v1 已加 center_element_Z 字段,SA1' 完整保留 |
| `step3/xas_local_datamodule_v2.py` | v1 已加 LongTensor collate,SA1' 完整保留 |

### 2.2 PASS gates

- forward_test: 5/5 PASS + 1 SKIPPED-by-design(Phase 6.5 GPU bf16,3 处 hardcoded fp32 永久跳过)
- view-order invariance: max diff 7.45e-9(gate <1e-4,严苛 4 个数量级)
- param count: 3,650,388(v1 4,511,204,Δ -860K = head 删除量精确匹配)
- smoke test: 2 epoch × 10 batch PASS

### 2.3 Exp4 baseline 重算(SA1' dry-run)

v1 SA1 metrics 脚本无 Exp4 baseline 数,SA1' 在写 4 个 v2 函数后 dry-run 生成对照锚点:

| 指标 | Exp4 val | Exp4 test |
|---|---|---|
| RMSD | 1.4849 | 1.4852 |
| pred_in_cutoff | 18.93/20 | 18.93/20 |
| Set-Level TypeAcc | 0.3309 | 0.3330 |
| Multiset Macro-F1 | 0.0843 | 0.0846 |
| Collapse Ratio | 0.0% | 0.0% |
| Position-by-position TypeAcc | 0.1877 | 0.1877 |

⚠️ **SA1' 隐藏 bug**: 投影 ablation `compute_projection_ablation_rmsd` 函数对 `shell_boundaries.pkl`
fallback 到 R_max=5.5 Å,**未实现 per-sample lookup**。SA3' 跑出 Δ=0 / 0 原子需投影时未追究,
直到 2026-05-01 用户物理统计才暴露这是评估盲区(详 §6.4)。

### 2.4 carry-over to SA2'

`EXP5_STEP1_PRIME_OUTPUT.md` §8 完整 carry-over,关键是 PYTHONPATH 优先级写法 +
Phase 6.5 SKIPPED rationale + 服务器磁盘清理建议。

---

## §3 SA2' from-scratch 训练(2026-04-28 → 2026-04-29,28h)

### 3.1 关键事件

#### Pre-flight stage — line 219 AttributeError + α' patch
- 首次 launch 在 datamodule.setup() 后 line 219 crash:
  `AttributeError: 'XasLocalDataModuleV2' object has no attribute 'train_dataset'`
- Root cause: SA1' fork Exp4 step4c_2_train.py 模板,v1→v2 datamodule API 命名不一致
  (v1 `.train_dataset` → v2 `.train_ds`)
- MA5 决议 (γ): 授权 SA2' apply,scope 严格限 line 219-220 两行 rename
- Patch 应用: `step4_2_train.py` line 219-220 的 `.train_dataset/.val_dataset` → `.train_ds/.val_ds`
- 备份保留: `step4_2_train.py.bak_pre_alpha`
- 影响: ~1h(诊断+patch+verify+relaunch),不阻塞 32h 训练

#### Training stage — 0 异常
- 28h 训练,无 NaN / Inf / OOM / 卡住事件
- nohup 在 ssh 断开后正常存活
- val_type_loss epoch 5 → 484: 0.036 → 0.00593(~6× 改进)
- val_coord_loss epoch 0 → 484: 1.325 → 0.684(稳定下降)
- val_density_loss 全程 ~ 0.085 plateau(× cost_density=0.2 = 0.017,贡献小)

### 3.2 Best ckpt 信息

```
路径:        /home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt
size:        43,959,458 B (44 MB)
epoch:       484 (max_epochs=500,自然完结,non-early-stop)
val_loss:    0.7065
val_coord:   0.684
val_type:    0.00593
val_density: 0.0854
val_lattice: 1.360 (× cost_lattice=0)
param count: 3,650,388 (matches SA1' expected)
PL version:  2.5.5
global_step: 1,833,785
```

### 3.3 Wall time

- 起: 2026-04-28 17:39+12:00
- 终: 2026-04-29 21:21+12:00
- 总: ~28h(LAUNCH_NOTE 估 32h,实际省 12%)
- 平均 epoch: ~3.4 min/epoch

### 3.4 Epoch 200 milestone marker — cosmetic miss
- LAUNCH_NOTE §5 设计的 milestone Callback 在 `on_validation_epoch_end` + `current_epoch == 200` 触发
- 但 SA1' 设的 `check_val_every_n_epoch = 5`,epoch 200 不是 val 周期 → marker 未触发
- 实质 milestone review 通过 MA5 周期 ssh check 替代完成
- Lessons learned 给 Exp5 MA2: 用 `on_train_epoch_end` 钩子或 `current_epoch >= 200 and not self._fired` 一次性 latch

---

## §4 SA3' sample + metrics(2026-04-29 → 2026-04-30,~3.5h)

### 4.1 step5_1_sample.py fork(SA1' 没写)

SA3' 从 `/home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py`(305 行)fork 到
`/home/tcat/diffcsp_exp5/code/step5/step5_1_sample.py`(341 行)。

11 项 v2 surgery(C1-C11):
- C1: ROOT 改 diffcsp_exp5
- C2: CKPT_PATH 改 epoch=484-val_loss=0.7065.ckpt
- C3: 输出 predictions_v2_{val,test}.pt(避免与 SA0 multisample 撞名)
- C4: cost_density==0.2 断言(防误用 Exp4 ckpt)
- C5: holdout gate 硬阻断(扩展 error msg 防未来 SA bypass)
- C6: --debug-n-batches N flag(dry-run)
- C7: --debug-no-save flag(dry-run)
- C8: 删 Exp4 phase 5b 注释残留
- C9: spectrum_encoder 4-arg 内化(model.sample 一层封装,sample 脚本不改)
- C10: dead import 清理(xas_collate_fn_v2 / XasLocalDatasetV2 / DataLoader)
- C11: dm.setup("test") fail-fast(不掩盖错误)

### 4.2 step5_2_compute_metrics.py 改造

SA3' 在 SA1' 619 行基础上 +11 行:
- argparse 加 `--debug-n-samples N`(dry-run)
- compute_metrics() line 410 后加 slice block(getattr fail-safe)
- Scope 严格限,**不动 4 个 v2 算法函数**(LAUNCH_NOTE §1.3 红线)

### 4.3 Sample 历程

| split | n_nominal | n_eff | silent_drop | wall | ms/sample |
|---|---|---|---|---|---|
| val  | 7,624 | 7,621 | 3 (0.039%) | 133.5 min | 1051 |
| test | 4,481 | 4,481 | 0 | 76.0 min | 1018 |
| 总 | 12,105 | 12,102 | 3 | 209.5 min (3h29m) | ~1034 |

LAUNCH_NOTE 估 ~8h,实际 3.5h(4090 比预期快)。

### 4.4 v2 主指标(双 split + Exp4 baseline 对照)

| 指标 | v2 val | v2 test | Exp4 val | Exp4 test | v2 vs Exp4 | 档位 |
|---|---|---|---|---|---|---|
| **Multiset Macro-F1** | **0.1086** | **0.1096** | 0.0843 | 0.0846 | **+28.8% / +29.6%** | ⚠️ AMBER |
| Set-Level TypeAcc | 0.3408 | 0.3397 | 0.3309 | 0.3330 | +3.0% / +2.0% | 边缘改进 |
| Collapse Ratio | 0.013% | 0.000% | 0.0% | 0.0% | 实质 0 | 🟢 GREEN |
| RMSD | 1.4954 | 1.4928 | 1.4849 | 1.4852 | +0.7‰ / +0.5‰ | 🟢 持平 |
| pred_in_cutoff | 18.92/20 | 18.94/20 | 18.93/20 | 18.93/20 | 持平 | 🟢 |
| Position-by-position TypeAcc | 0.1979 | 0.1969 | 0.1877 | 0.1877 | [VIRTUAL] | n/a |

**Generalization**: val ↔ test Multiset F1 Δ=0.001,完美 generalize。

### 4.5 投影 Ablation

| split | n | R_max | RMSD before | RMSD after | Δ | atoms_projected_avg |
|---|---|---|---|---|---|---|
| val  | 7621 | 5.50 Å | 1.4954 | 1.4954 | **0.0000** | **0.00** |
| test | 4481 | 5.50 Å | 1.4928 | 1.4928 | **0.0000** | **0.00** |

⚠️ **R_max 是 fallback 5.5 Å,不是 per-sample 真实边界**。SA1' 实现 bug,SA3' 未追究。
当时解读为 "v2 真物理改进,不是 metric artifact"。**后续 §6.4 暴露这是 evaluation blind spot**。

### 4.6 Per-class Multiset detail(val,top by support)

- Z=8 O: F1=0.7972, support_true=42040, support_pred=47259, ratio 1.124 (中度 over-predict)
- Z=3 Li: F1=0.3123, ratio 1.424 (强 over-predict)
- Z=34 Se: F1=0.1809, ratio 0.322 (强 under-predict)
- Z=26 Fe: F1=0.0844 (test), ratio 0.209 (强 under-predict)

诊断: 模型学了 element prior(全局元素分布),没学好 element-context(per-sample 关联)。

### 4.7 SA3' verdict 自评

⚠️ AMBER(主信号 0.1086 落 [0.10, 0.20])+ 🟢 geometry strong。
推荐 MA5: 续训 200 epoch 试跨 0.20 GREEN(当时认为 LR 已到 eta_min=1e-6,续训涨幅有限但 ROI OK)。

---

## §5 SA2'' 续训(2026-04-30 ssh-only,无新 SA 上下文,~11h)

### 5.1 用户决策 + 工作流

用户选 (B) ssh 自己跑(避免开新 sub-agent 的 round-trip overhead)。MA5 给 paste-ready 命令包。

### 5.2 关键设计决策

#### LR scheduler T_max 行为问题

`CosineAnnealingLR(T_max=MAX_EPOCHS)` 是 step4_2_train.py line 183 配置。
改 MAX_EPOCHS 500→700 后,scheduler 会按新 T_max 重新规划:

```
T_max=500: LR at epoch 484 = 1.25e-06  (1.25× eta_min,SA2' 训练结束)
T_max=700: LR at epoch 484 = 2.25e-05  (22.5× eta_min,续训起点)
```

**LR 跳跃 22.5×** 是关键 trade-off:
- ✅ 给优化器新动能,跳出 SA2' 局部最小值(LR warm restart 机制)
- ✅ 2.25e-5 仍在合理 fine-tune 区间(< 5e-5)
- ⚠️ 头几 epoch val_loss 大概率先涨后跌

**用户拍板方案 (1)**: GO 续训接受 22.5× LR 跳跃。

#### last.ckpt vs best.ckpt 选择

**关键发现**(SA2 OUTPUT §3 写错): SA2' 自然完结到 epoch 499,但 best 在 epoch 484。
md5 不同,**last.ckpt ≠ best.ckpt**。

MA5 决: 从 **best (epoch 484) ckpt** 续训(不是 last),保持 verdict 一致性。
`step4_2_train.py` line 224-225 的 `last_ckpt = ...` 改为硬编码 best 路径。

#### Best ckpt 防覆盖

PL ModelCheckpoint save_top_k=1 会删旧 best,SA2'' 续训出新 best 时 SA2' best 会被 PL 删除。
MA5 决: 续训前 `cp epoch=484-val_loss=0.7065.ckpt sa2_baseline_epoch484_val0.7065.ckpt.frozen`,
`.frozen` 后缀防 PL pattern 匹配,SA2' best 永久保留作 safety net。

### 5.3 续训成果

| 指标 | SA2'(484 epoch) | SA2''(529 epoch) | Δ |
|---|---|---|---|
| best val_loss | 0.7065 | **0.7003** | **-0.88%** |
| best epoch | 484 | 529 | +45 epoch 找到新 best |
| 续训 wall | — | epoch 484 → 679 ≈ 195 epoch | ~11h |
| early stop | No | Yes(@ epoch 679,best+150 epoch 没改进)| ✓ |

**LR warm restart 验证为正确选择**:模型确实跳出 SA2' 局部最小,落到 epoch 529 更深最小值。

### 5.4 ckpt 状态(2026-04-30 末)

```
/home/tcat/diffcsp_exp5/checkpoints/
  epoch=529-val_loss=0.7003.ckpt           ← SA2'' best (current)
  last.ckpt                                ← SA2'' epoch 679 (训练结束)
  sa2_baseline_epoch484_val0.7065.ckpt.frozen  ← SA2' best (frozen safety net)
  sa2pp_resume_epoch529_val0.7003.ckpt.frozen  ← SA2'' best (frozen,2026-05-01 加)
```

### 5.5 SA2'' OUTPUT.md(本应有,实际未单写)

由于 SA2'' 是 ssh-only 执行,没有 sub-agent context,MA5 直接整合到本 final report §5。

---

## §6 物理约束发现(2026-05-01,用户)

### 6.1 现象

用户在 SA2'' 续训完成后,统计 `predictions_v2_*.pt` 的预测原子结构,发现:
**大量样本中预测原子两两距离 < 1.5 Å,等于物理上原子重合,无法跑 FEFF 计算。**

这与 SA3' 投影 ablation 报告 "Δ=0 / 0 原子需投影"严重矛盾 —— 用户的物理统计是 ground truth,
SA3' 报告是评估盲区误导。

### 6.2 评估盲区根因

v2 设计的 6 项 metric **全部未检测原子两两距离**:

- **RMSD**: per-atom 距 ground truth(Hungarian 一对一,不看 pred 内部)
- **pred_in_cutoff**: 每 pred 距原点(不看两两)
- **Set-Level / Multiset**: 元素 multiset(完全 ignore 几何)
- **Collapse Ratio**: pred std vs true std。如果 20 原子全挤在 1 Å 球内 std≈0.5 Å,
  与 true std 1.5 Å 比 0.5/1.5=0.33 > threshold 0.5 不触发 → 漏检
- **Projection Ablation**: 每 pred 距原点(同 pred_in_cutoff)+ R_max fallback bug

### 6.3 SA3' 投影 ablation 数据的真实含义

Δ=0 / 0 atoms_projected 真实意义:
**所有 pred 原子在距原点 5.5 Å 球内**(物理 R_max 应该是 per-sample 真实边界,
但 SA1' fallback 5.5 Å 实际是 L=6 box 的 √3/2 ≈ 5.196 Å 上限,任何 pred 在 box 内都算"球内")。

**不是**"pred 物理合理":
- 没看两两距离
- R_max=5.5 Å 是 box 几何上限,不是化学物理上限
- SA1' 实现 bug 把这个本应 per-sample 的指标降级为 trivial gate

### 6.4 责任划分

- SA1' 写投影 ablation 时,`R_max` 的 per-sample lookup 写成 fallback 常数 5.5 Å
- SA3' 跑 ablation 时报告 Δ=0 + 0 atoms,未追究"为什么 R_max 是 5.5 Å 不是 per-sample"
- MA5 review SA3' OUTPUT §3.1 时已看到"shell_boundaries.pkl 是 per-sample dict 但 SA3' 用默认 5.5 Å fallback",**没意识到这是评估盲区**,以为只是"非 critical 实现细节"
- MA5 review SA1' 4 个 metric 算法时,只 verify 算法实现是否正确,**没问指标 menu 完备性**

### 6.5 修复方向(2 个并行)

#### 修复 A — SA-METRICS-V3 评估改造(Exp5 MA2 立刻启动)
- 加 `min pairwise distance` 检测 + 1.5 Å gate
- 实现 7 项复合评分(详 EXP5_PROPOSAL_v2_AMENDED §B)
- 修 shell_boundaries.pkl 的 per-sample lookup(取代 5.5 Å fallback)
- 用 SA3' 已有 predictions_v2_*.pt(SA2 baseline)算复合分,出 Exp5 v2 真实物理 verdict
- 输出 min_d 违反样本清单,作为 Exp5' 设 lambda 的 ground truth
- **不重 sample,~ 2-3h 工作量**

#### 修复 B — Exp5' 物理约束加 pairwise penalty 重训(Exp5 MA2 后续)
- 在 `diffusion_w_type_xas.py` forward 内加 `_pairwise_min_distance_penalty` loss
- yaml 加 `cost_pairwise_min: λ`(λ 待 SA-METRICS-V3 数据出来后调)
- 从 SA2'' epoch 529 ckpt warm-start(架构兼容)
- max_epochs 200 续训(epoch 529 → 729)
- 估时 ~10-15h 训练 + ~3.5h sample + 0.5h metrics + 0.5h 报告 ≈ 14-19h 总

---

## §7 当前服务器/本地状态(2026-05-01)

### 7.1 服务器(scsmlnprd02.its.auckland.ac.nz)

```
/home/tcat/diffcsp_exp5/
├── code/
│   ├── step2/spectrum_encoder.py            (127 行,SA1' MV-attention 版,有 .bak_exp4)
│   ├── step3/
│   │   ├── diffusion_w_type_xas.py          (415 行,SA1' 撤 head 版,有 .bak_exp4)
│   │   ├── xas_local_dataset_v2.py          (374 行,v1 SA1 加 center_Z)
│   │   ├── xas_local_datamodule_v2.py       (257 行,v1 SA1 加 LongTensor collate)
│   │   ├── conf_xas/model/diffusion_xas.yaml (含 cost_density=0.2 + mv_attention.num_heads=4)
│   │   └── forward_test.py                  (546 行,SA1' Phase 6.6 测 MV-attention)
│   ├── step4/
│   │   ├── step4_1_smoke_test.py            (193 行,SA1' 1-mode 改写)
│   │   └── step4_2_train.py                 (300 行,SA1' + SA2' α' patch + MA5 SA2'' resume hardcode)
│   └── step5/
│       ├── step5_1_sample.py                (341 行,SA3' fork from Exp4)
│       └── step5_2_compute_metrics.py       (630 行,SA1' + SA3' --debug-n-samples;有 .bak_pre_sa3)
├── checkpoints/
│   ├── epoch=529-val_loss=0.7003.ckpt      (44 MB,SA2'' best,active for sample)
│   ├── last.ckpt                            (44 MB,SA2'' epoch 679 训练结束)
│   ├── sa2_baseline_epoch484_val0.7065.ckpt.frozen  (44 MB,SA2' best 永久保留)
│   └── sa2pp_resume_epoch529_val0.7003.ckpt.frozen  (44 MB,SA2'' best 永久保留)
├── data/  (软链接到 Exp4 data)
├── logs/  (训练 / sample / metrics / pre-flight 全部 log)
└── EXP5_SA2_PRIME_OUTPUT.md, EXP5_SA3_PRIME_OUTPUT.md, EXP5_STEP1_PRIME_OUTPUT.md

/home/tcat/diffcsp_exp4/data/
├── shell_boundaries.pkl  (387 MB,md5 cf2050e4899160f5698ad2481377e94c) ⭐ Exp5'/Exp5 MA2 关键
├── ...其他 Exp4 data 文件
```

### 7.2 SA3' 已有 predictions(Exp5 MA2 SA-METRICS-V3 直接用,不重 sample)

```
/home/tcat/diffcsp_exp5/code/step5/predictions_v2_val.pt    9.8 MB (7621 samples)
/home/tcat/diffcsp_exp5/code/step5/predictions_v2_test.pt   5.8 MB (4481 samples)
```

⚠️ 这是 **SA2 epoch 484 baseline** 的 sample 输出,**不是** SA2'' epoch 529 的。
SA2'' 续训后未重 sample(MA5 决策"先评估改造再决定是否 sample SA2''")。

### 7.3 本地(用户 Windows 机)

主要文档(Markdown):
- 各 launch note / OUTPUT / handoff
- 本 report(EXPERIMENT5_FINAL_REPORT_v1.md)
- 配套 file guide / MA2 handoff(后续生成)

无关键代码或数据本地化(全在服务器)。

---

## §8 给 Exp5 MA2 的 carry-over

### 8.1 Exp5 MA2 立刻要做的 3 件事

1. **读 EXP5_FILE_GUIDE_FINAL.md** 把脚本/数据/ckpt 索引内化
2. **启动 SA-METRICS-V3**(评估改造,~ 2-3h):
   - 加 7 项复合评分(详 EXP5_PROPOSAL_v2_AMENDED §B.2)
   - min_d 1.5 Å gate
   - 修 shell_boundaries.pkl per-sample lookup(SA1' fallback bug)
   - 用 SA3' predictions_v2_*.pt 算 SA2 baseline 7 项分
   - 输出 min_d 违反样本清单(Exp5' lambda 调度依据)
3. **启动 Exp5'**(物理约束重训,~ 14-19h):
   - 等 SA-METRICS-V3 数据出来后定 lambda
   - 写 SA-EXP5'-TRAIN handoff,从 epoch 529 ckpt warm-start
   - 评估用 SA-METRICS-V3 的复合分体系

### 8.2 决策树

```
SA-METRICS-V3 完成,SA2 baseline 数据出齐:

├─ min_d 通过率 > 90% + 复合分 > 0.50
│   → v2 真正成功,Exp5' 不需要,直接 SA4' figure + Exp5 final v2
│
├─ min_d 通过率 60-90% + 复合分 > 0.50
│   → v2 数学上 OK 物理上有 warn,Exp5' 加轻 lambda(0.1) 试修
│
├─ min_d 通过率 < 60% + 复合分 0.30-0.50
│   → 重物理违反 + 中等数学,Exp5' 加重 lambda(0.5-1.0) + 严监控
│
└─ min_d < 60% + 复合分 < 0.30
    → 双 fail,可能 MV-attention 这条路不通
    → 不开 Exp5',直接 Exp6 转向(distance-aware loss / CFG / hierarchical type)
```

### 8.3 MA 工作哲学(继承自 v1 self-audit)

1. 用户意向 = default,不让任何"前 MA 推荐"override
2. 任何技术判断先 conversation_search + 列证据,不直接套结论
3. 写完 SA handoff 必先给用户 review,不直接发
4. SA 中期报告交回必先识别"是否在 proposal 锁定方向上",不在则停
5. 70% 上下文闸门是硬线
6. **MA review SA 设计时,要主动质疑指标 menu 的完备性**(MA5 临走加,§6.4 教训)

### 8.4 红线(全程不动)

- ❌ 不动 holdout(Exp4 final 后从未解禁)
- ❌ 不动 incompat_pool.csv
- ❌ 不动 7 守卫包(scikit-learn 1.7.2 / numpy 2.2.6 / scipy 1.15.3 / pymatgen 2025.10.7 /
   torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / torch-scatter 2.1.2+pt24cu124)
- ❌ 不动 .frozen ckpt(2 个 SA2/SA2'' 永久保留,作 safety net)
- ❌ 不修 Phase 6.5 hardcoded fp32(永久 SKIPPED-by-design)
- ❌ 不动 cspnet.py 等 Exp4 backbone

### 8.5 已知 bug / 工程债务

1. **SA1' 投影 ablation R_max fallback** — SA-METRICS-V3 修
2. **MAX_EPOCHS in code not yaml** — SA1' 决策(line 83 写常量,不 yaml 字段),Exp5' 重训改 MAX_EPOCHS 时记得改 train.py 而不是 yaml
3. **ModelCheckpoint save_top_k=1 删旧 best** — Exp5' 训练前再 cp 一份 sa2'' ckpt 到 .frozen
4. **PL Callback milestone marker bind 到 val 周期** — Exp5 MA2 写新 callback 时改 `on_train_epoch_end`
5. **predictions_v2_*.pt 是 SA2 baseline 不是 SA2''** — Exp5 MA2 决定是否需要重 sample SA2''(看 SA-METRICS-V3 verdict)

---

## §9 接力链时间轴

| 日期 | Agent | 关键事件 |
|---|---|---|
| 2026-04-28 早 | v1 SA1 | 工作目录建立 + 7 文件改 + bak_exp4 锚点 |
| 2026-04-28 中 | v1 SA2 | 训练 36 epoch head collapse → kill |
| 2026-04-28 晚 | v1 MA self-audit | 5 条工作哲学产生 |
| 2026-04-28 | MA5 | v2 proposal 撰写 + 接管 |
| 2026-04-28 | SA1' | MV-attention surgery + forward_test 5/5 + smoke PASS |
| 2026-04-28 → 04-29 | SA2' | 28h from-scratch 训练 + line 219 α' patch + best epoch 484 val 0.7065 |
| 2026-04-29 → 04-30 | SA3' | 3.5h sample val+test + metrics + 投影 ablation + verdict AMBER |
| 2026-04-30 | SA2'' (ssh-only) | 11h 续训 from best → epoch 529 val 0.7003 + early stop @ 679 |
| 2026-05-01 | 用户 | 物理统计发现 min_d < 1.5 Å 大量违反 |
| 2026-05-01 | MA5 | 评估盲区 root cause + EXP5_PROPOSAL_v2_AMENDED + 本 final report + EXP5_FILE_GUIDE_FINAL + EXP5_MA2_HANDOFF |
| **2026-05-01+** | **Exp5 MA2 接手** | SA-METRICS-V3 + 决策 Exp5' / SA4' / Exp6 |

---

## §10 给 Exp5 MA2 / 未来 ExpN 的核心 Lessons

1. **数学完备 ≠ 物理完备** — Set-Level / Multiset Macro-F1 是数学解耦指标,
   不能替代物理 motivated 评分(配位数 / 距离 / 元素 / pairwise distance)。

2. **Min pairwise distance 是 ExpN 不变量** — 任何 diffusion 生成原子坐标的 Exp,
   评估必须包含原子两两距离检测。1.5 Å 是物理硬下限。

3. **R_max / shell 边界禁用 fallback** — Exp4 已有 per-sample ground truth,任何 ExpN 评估
   "懒一懒用 5.5 Å fallback"会隐藏评估盲区,直到下游(用户物理统计 / FEFF 跑不动)才暴露。

4. **MA review SA 设计时主动问"指标 menu 够不够"** — 我 review SA1' 4 个新函数
   只 verify 算法是否正确,没问"覆盖所有 collapse 模式吗"。Exp5 MA2+ 应主动补这一问。

5. **last.ckpt ≠ best.ckpt** — PL 训练自然完结到 max_epochs 时,last 是终点(可能已过 best),
   不是 best 的副本。续训前必 md5 verify。

6. **LR warm restart 对续训有真实价值** — 22.5× LR 跳跃让 SA2'' 跳出 SA2' 局部最小,
   找到更深最小值。Exp5' 续训也应有 LR warm restart 机制。

7. **小补丁也要 MA5 ack + diff** — line 219 α' patch 是 2 字符 rename,但 MA5 ack 流程
   保证了 scope 不扩大,SA2' 不会顺便清 logger config 等无关债务。

---

## §11 final 状态宣告

| 模块 | 状态 |
|---|---|
| Exp5 v2 架构(MV-attention + cost_density 0.2)| ✅ 落地,best ckpt epoch 529 val 0.7003 |
| Exp5 v2 评估(Multiset F1)| ⚠️ AMBER 0.1086 / 0.1096 |
| **Exp5 v2 评估(7 项复合分)** | ⏳ **Exp5 MA2 SA-METRICS-V3 启动** |
| Exp5'(物理约束 1.5 Å pairwise penalty)| ⏳ Exp5 MA2 决策启动时机 |
| Exp5 v2 收尾(SA4' figure / final report v2)| ⏳ 待 Exp5 MA2 / SA-METRICS-V3 后续决策 |

**MA5 移交。Exp5 MA2 接手见 EXP5_MA2_HANDOFF.md + EXP5_FILE_GUIDE_FINAL.md。**

---

*MA5 撰写,2026-05-01,基于 SA1'/SA2'/SA3'/SA2'' 全程产出 + 用户 2026-05-01 物理约束发现。*
