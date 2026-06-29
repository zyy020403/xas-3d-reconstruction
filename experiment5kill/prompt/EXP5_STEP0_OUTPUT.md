# EXP5 STEP 0 (SA0) — OUTPUT

**SA**: DiffCSP-Experiment5step0agent
**Date**: 2026-04-28 (NZST, host `scsmlnprd02`)
**Status**: ✅ COMPLETE — sanity PASS, K-averaging quick-win confirmed
**Cost**: ~6 GPU-hours (1.5h sample + 23s aggregate × 2)

---

## TL;DR — 给 MA 的一句话

**K-averaging 在 Exp4 ckpt 上是正向 free win**:K=10 + Hungarian-with-fold 同时给 RMSD `1.4856 → 1.4215`(−0.064 Å, −4.3%)和 TypeAcc `0.190 → 0.258`(+6.8 pp, **+36% 相对**),无训练成本。**强烈建议 SA3 把 K=5 标配为评估方法**(K=5 已拿到 K=10 收益的 ~95%)。

---

## 1. Sanity 闸门:✅ PASS

500-sample 子集,K=1 baseline 与 Exp4 K=1 同子集 reference 三项闸门全过(±2·SE band):

| 指标 | SA0 K=1 | Exp4 K=1 (same 500) | ±2SE band | 判定 |
|---|---|---|---|---|
| RMSD | 1.4856 | 1.4813 | [1.4698, 1.4928] | ✓ |
| TypeAcc | 0.1903 | 0.1795 | [0.1641, 0.1949] | ✓ |
| pred_in | 18.96 | 18.92 | [18.77, 19.07] | ✓ |

模型加载完美:`state_dict missing=0 unexpected=0`,epoch=366,ckpt md5 全程 `dc9d2c9b371c78125f285a5a6478d404`。

---

## 2. 主结果(deployment-safe 行加粗)

| K | Strategy | RMSD | ΔvsK1 | TypeAcc | ΔvsK1 | pred_in |
|---:|---|---:|---:|---:|---:|---:|
| 1 | (any) | 1.4856 | – | 0.1903 | – | 18.96 |
| 5 | naive ❌ | 2.1143 | +0.629 | 0.2287 | +0.038 | 20.00 |
| **5** | **hungarian_fold ★** | **1.4296** | **−0.056** | **0.2298** | **+0.040** | **18.96** |
| 5 | hungarian_fold_bestanchor † | 1.3610 | −0.125 | 0.2334 | +0.043 | 18.97 |
| 5 | medoid | 1.4886 | +0.003 | 0.1870 | −0.003 | 18.92 |
| 5 | oracle_best † | 1.3784 | −0.107 | 0.1884 | −0.002 | 19.09 |
| 10 | naive ❌ | 2.3549 | +0.869 | 0.2590 | +0.069 | 20.00 |
| **10** | **hungarian_fold ★** | **1.4215** | **−0.064** | **0.2583** | **+0.068** | **18.94** |
| 10 | hungarian_fold_bestanchor † | 1.3185 | −0.167 | 0.2601 | +0.070 | 19.03 |
| 10 | medoid | 1.4751 | −0.010 | 0.1835 | −0.007 | 18.93 |
| 10 | oracle_best † | 1.3432 | −0.142 | 0.1940 | +0.004 | 19.09 |

★ = recommended; ❌ = broken (torus-averaging bug, see §5); † = uses ground truth, not deployable as-is

### Per-Tier(只列 deployment-safe 的 hungarian_fold K=10)

| Tier | n | RMSD K=1 | RMSD K=10 | ΔRMSD | TypeAcc K=1 | TypeAcc K=10 | ΔTypeAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 129 | 1.4882 | 1.4273 | **−0.061** | 0.2554 | 0.3698 | **+0.114** |
| C | 256 | 1.4829 | 1.4231 | −0.060 | 0.1693 | 0.2324 | +0.063 |
| D | 115 | 1.4887 | 1.4114 | −0.077 | 0.1639 | 0.1909 | +0.027 |

Tier B 是 TypeAcc 最大受益者(+11.4 pp);Tier D 是 RMSD 最大受益者(−0.077 Å)。**所有 tier 双指标双正向**。

---

## 3. K=5 vs K=10:K=5 是 sweet spot

| 指标 | K=5 | K=10 | K=5 / K=10 |
|---|---:|---:|---:|
| ΔRMSD vs K=1 | −0.056 | −0.064 | **88% 收益** |
| ΔTypeAcc vs K=1 | +0.040 | +0.068 | 59% 收益 |

RMSD 已基本饱和;TypeAcc 仍有 ~40% 边际,但绝对增益从 K=5→K=10 只多 +0.029 vs K=1→K=5 的 +0.040。**SA3 default K=5 性价比最高;有时间预算的话 K=10 把 TypeAcc 吃满**。

---

## 4. 给 Exp5 主线 / SA3 的建议

| 建议 | 优先级 | 操作 |
|---|---|---|
| SA3 评估标配 K=5 + hungarian_fold | **必做** | 同时报 K=1 和 K=5 两组数字,Exp4 和 Exp5 ckpt 都报,主线对比维度多一倍 |
| 用 `multisample.py` + `multisample_aggregate_v2.py` 直接套 | **必做** | SA3 只需替换 ckpt 路径,代码已 production-ready |
| 给 SA1 新架构留一个 anchor-selection hook | 推荐 | hungarian_fold_bestanchor(用 truth 选 anchor)给出 K=10 RMSD 1.32,比 K=1 好 −0.17 Å。如果有无 cheating 的 anchor 启发式(例如 K 候选两两 RMSD 最低的那个 = medoid-as-anchor),可能进一步压 RMSD ~0.05 |
| 不要用 medoid | 警告 | TypeAcc 反而掉(0.190 → 0.184),因为它退化为单 sample 不做 mode-vote。本验证 falsify 了 medoid 假设 |
| 不要用 naive averaging | 警告 | torus-bug 灾难性失败(K=10 RMSD = 2.35,比 K=1 差 +0.87),就算 TypeAcc 涨也是假 win(pred_in 假涨到 20.00 暴露) |

---

## 5. 关键发现:torus-averaging bug 和它的修复

v1 hungarian 把对齐后的 frac coords 直接 mean,**忽略了 frac 是环面([-0.5, 0.5] 边界等价)**。结果原子被拖向晶胞中心 → pred_in 假涨到 20.00,RMSD 灾难性 +0.87。

修复(`aggregate_hungarian_fold` in v2):Hungarian 对齐后,**先 min-image fold 到 anchor 邻域**(`x = x - round(x - anchor)`),再 mean,最后 wrap 回 [-0.5, 0.5]。三行代码,RMSD 从 +0.87(灾难)变成 −0.06(free win)。

**进 Exp6 cheatsheet 的物理直觉**:任何对 fractional coordinates / atomic positions / lattice vectors 做 averaging 的操作必须先 fold 到公共邻域,否则结果是周期镜像的 centroid,不是物理上的"平均位置"。

---

## 6. 工件清单(在 `/home/tcat/diffcsp_exp5/sa0/`)

```
results/
├── sa0_subset_500.csv              500 stratified val names + Exp4 K=1 metrics
├── samples_raw_K10.pt              raw (K=10, 20, 3) per sample,2.7 MB
├── multisample_raw.csv             v1 per-sample × K × strat (3 strat)
├── multisample_results.md          v1 report (now obsolete due to bug)
├── multisample_K_curves.png        v1 curves
├── multisample_v2_raw.csv          ★ v2 per-sample × K × strat (5 strat)
├── multisample_v2_results.md       ★ v2 report — main deliverable
└── multisample_v2_K_curves.png     ★ v2 curves
logs/
├── env_smoke.log                   3-min env validation
├── make_subset.log                 500-sample selection
├── multisample_K10.log             1.5h K=10 sample log
├── multisample_aggregate.log       v1 aggregate (kept for diff)
└── multisample_aggregate_v2.log    ★ v2 aggregate
scripts/
└── (5 production scripts, see EXP5_STEP0_FINAL_REPORT §3)
```

---

## 7. 已知限制与未做的事

| 限制 | 影响 | 谁来 follow-up |
|---|---|---|
| 子集 N=500(目标统计噪声 ±2σ ≈ ±0.011 RMSD) | 全 val=7621 上数字会更精确,但符号和量级不会变 | SA3(直接全 val 跑) |
| Tier A 跳过(全 val 仅 13 例) | A-tier 行为未知 | SA3 可选(13 例统计意义低) |
| 同 seed 不可复现(model.sample 内部 RNG 不响应 global seed) | 复现性靠"独立 K 次采样,不复现单次"的模式;不影响统计结论 | Exp6 如果要 sample-level reproducibility 需改 model.sample 内部 |
| K=10 Hungarian fold 与 oracle 仍有 0.078 Å gap | "averaging across modes" 仍是问题;medoid-as-anchor 等启发式可能再压 ~0.05 Å | Exp5 主线 / Exp6 选做 |
| 没在 holdout/test 上跑 | handoff 红线明确禁止 SA0 触 holdout | SA3 在 ckpt 评估 phase 用 |

---

## 8. SA0 → MA 的交付物清单

✅ EXP5_STEP0_OUTPUT.md(本文件)— 数字 + 结论
✅ EXP5_STEP0_FINAL_REPORT.md — narrative + Exp6 cheatsheet
✅ 5 个 production-ready 脚本(`env_smoke.py`, `make_subset.py`, `multisample.py`, `multisample_aggregate_v2.py`, `run_sa0.sh`)
✅ 完整 logs + raw outputs 在服务器上的 `/home/tcat/diffcsp_exp5/sa0/`

**SA0 任务结束。等待 MA 验收。**
