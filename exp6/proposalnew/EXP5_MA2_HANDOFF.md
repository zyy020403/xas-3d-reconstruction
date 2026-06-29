# EXP5_MA2_HANDOFF.md
# Exp5 v2 Exp5 MA2 接班 Handoff(MA5 → Exp5 MA2)

> **From**: MA5(Exp5 v2 Main Agent,即将上下文 70% 闸门)
> **To**: Exp5 MA2(下一棒 Main Agent,接 Exp5 v2 收尾 + Exp5' 启动)
> **Date**: 2026-05-01
> **Status**: SA2'' 续训完成,用户物理统计发现评估盲区,SA-METRICS-V3 + Exp5' 待启动
> **本文档定位**: Exp5 MA2 一文上手。读完本文 + 4 份 critical 文件,立即可写第一棒 SA handoff

---

## §0 一屏掌握

### 0.1 你是谁,做什么

你是 Exp5 v2 Exp5 MA2(MA5 的接班),接 MA5 上下文闸门后的工作。Exp5 v2 已完成训练 + 续训
(SA2 → SA3 → SA2''),最 active best ckpt 是 epoch=529 val 0.7003。但用户在 2026-05-01
统计预测结构发现**严重物理违反**(min pairwise distance < 1.5 Å,大量样本物理无效)。

你的任务:
1. **SA-METRICS-V3** 评估改造(~ 2-3h):加复合评分 + min_d gate + 修 shell_boundaries.pkl bug
2. **基于 SA-METRICS-V3 数据决定 Exp5' 启动**(物理约束 pairwise penalty 重训,~14-19h)
3. **Exp5 v2 收尾**(SA4' figure / final report v2)

**不需要重启 Exp5 编号** — Exp5' 是 v2 的 extension,共享 codebase / data / ckpt。

### 0.2 必读 4 份(按顺序)

1. **EXP5_PROPOSAL_v2_AMENDED.md** — 修订后的 v2 proposal(加 §A 物理发现 + §B 7 项复合评分 + §C Exp5'),你所有工作的锚点
2. **EXPERIMENT5_FINAL_REPORT_v1.md** — MA5 时代的 Exp5 全程记录,§6/§8/§10 必看
3. **EXP5_FILE_GUIDE_FINAL.md** — 服务器 / 本地 / ckpt / log 完整索引,§9 verify 块照着跑一遍
4. **EXP4_FINAL_REPORT_ERRATA_2.md** — `_density_loss` 塌缩根因 + Exp3 真实历史(MA5 的根基,你也要内化)

**不读**: 各 sub-agent OUTPUT 全文(摘要在 final report §1-§5)/ EXP5_PROPOSAL_v2 原版(被 AMENDED 取代)。

### 0.3 启动后第一条回复请按以下格式

```
我已读完 4 份必读文档。复述 Exp5 v2 当前状态:
[列 5 条:架构/best ckpt/已知 bug/物理违反发现/SA3 carry-over]

我注意到 3 个最容易出错的点:
[列 3 条,例如: SA1' 投影 ablation R_max fallback bug / MAX_EPOCHS in code not yaml /
 PL ModelCheckpoint save_top_k=1 删旧 best 风险]

我下一步:
[列 SA-METRICS-V3 launch note 撰写计划 + 第 1 件让用户做的 verify 命令]
```

---

## §1 当前 Exp5 状态(给 Exp5 MA2 速查表)

### 1.1 已完成

| 阶段 | 产出 |
|---|---|
| SA1' 架构 surgery | MV-attention + cost_density 0.2 + center_emb 落地 |
| SA2' 28h training | best epoch 484 val 0.7065 |
| SA3' 3.5h sample + metrics | val Multiset F1 0.1086 / test 0.1096(vs Exp4 +28.8%/+29.6%)|
| SA2'' 11h 续训(ssh-only) | best epoch 529 val 0.7003(改进 0.88%)+ early stop @ 679 |

### 1.2 待启动

| 阶段 | 估时 | 触发条件 |
|---|---|---|
| **SA-METRICS-V3 评估改造** | 2-3h | **Exp5 MA2 立刻** |
| Exp5' 物理约束重训 | 14-19h | SA-METRICS-V3 数据出来后决定 |
| SA4' figure 6 张 + Exp6 决议 | 3-4h | 看 Exp5' verdict 决定 |
| Exp5 final report v2 | 2-3h | 全部完成后 |

### 1.3 服务器 active 资产

```
/home/tcat/diffcsp_exp5/checkpoints/
├── epoch=529-val_loss=0.7003.ckpt          ← active best (SA2'' 续训产物)
├── last.ckpt                                ← SA2'' 训练自然结束 (epoch 679)
├── sa2_baseline_epoch484_val0.7065.ckpt.frozen  ← 永久 SA2' best
└── sa2pp_resume_epoch529_val0.7003.ckpt.frozen  ← 永久 SA2'' best

/home/tcat/diffcsp_exp5/code/step5/
├── predictions_v2_val.pt   ← SA3' sample (from SA2 epoch 484, not 529)
└── predictions_v2_test.pt  ← 同上

/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl  ← Exp4 ground truth (387 MB, md5 cf2050e4...)
```

⚠️ predictions_v2_*.pt 是**SA2 baseline**(epoch 484)的输出,**不是 SA2''**。SA-METRICS-V3 用这个算
出的复合分是 SA2 时期的物理指标。如要算 SA2'' 复合分,**Exp5 MA2 决定是否值得多花 3.5h 重 sample**。

---

## §2 第一棒 — SA-METRICS-V3 任务规格(Exp5 MA2 写 launch note 用)

### 2.1 任务范围(2-3h 工程)

新写 `/home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py`,实现:
1. 7 项复合评分(详 EXP5_PROPOSAL_v2_AMENDED §B.2)
2. min pairwise distance ≥ 1.5 Å gate(违反 → 总分 0)
3. shell_boundaries.pkl per-sample 正确读取(取代 SA1' 5.5 Å fallback)
4. 用 SA3' 已有 predictions_v2_*.pt 算 SA2 baseline 7 项分
5. 输出 min_d 违反样本清单(Exp5' lambda 调度依据)

**不动**: step5_2_compute_metrics.py 的 4 个 v2 算法函数(保留作历史对照)。
**不重 sample**: 用 SA3' 已有 predictions(节省 3.5h)。

### 2.2 7 项评分定义(SA-METRICS-V3 实施)

```
Gate (前置物理硬下限):
  G. min pairwise distance ≥ 1.5 Å
     pred 20 原子两两 cartesian 距离最小值 ≥ 1.5 Å
     违反 → 该样本总分 = 0,其余 6 项不算
     dataset-level: gate 通过率(% of samples 通过)

6 项加权评分(总和 1.0,gate 通过才计算):
  1. 第一壳层配位原子数  权重 0.20  容错 ±1.5 个
  2. 第一壳层距离        权重 0.20  容错 ±0.2 Å
  3. 第一壳层元素种类    权重 0.20  CNO 等价(C/N/O 视为同类)
  4. 第二壳层配位原子数  权重 0.10  容错 ±3 个
  5. 第二壳层距离        权重 0.10  容错 ±0.2 Å
  6. 第二壳层元素种类    权重 0.10  CNO 等价

评分函数:
  - 配位数:1 if |Δn|≤tolerance else max(0, 1 - (|Δn|-tolerance)/3.0)
  - 距离:  1 if |Δd|≤0.2 else max(0, 1 - (|Δd|-0.2)/0.5)
  - 元素:  multiset 交集 / 总数(C/N/O 替换为合并 token "CNO")
```

### 2.3 shell_boundaries.pkl 读取(详 §3)

per-sample lookup by `sample_name`:
```python
import pickle
with open('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl', 'rb') as f:
    sb = pickle.load(f)  # dict[sample_name] → {threshold, distances, species_Z, shell_starts, ...}

# 对每个 prediction 样本:
# 1. 用 sample_name(predictions_v2_*.pt 的 sample_name 字段)查 sb
# 2. shell_1 边界 = (distances[shell_starts[0]], distances[shell_ends[0]-1])  # 注意 starts/ends 含义
# 3. shell_2 边界 = (distances[shell_starts[1]], distances[shell_ends[1]-1]) if len(shell_starts) >= 2 else None
# 4. true 数据:同上的 distances/species_Z
# 5. pred 数据:从 predictions[i] 算 cartesian 距离 → 用同一 threshold(p10=0.1563 Å)分 shell
```

**关键不确定**: SA-METRICS-V3 上线第 1 件事必须先 load shell_boundaries.pkl,print sample 字段,
**贴给 Exp5 MA2 verify schema 与本 handoff 描述一致**,再写实施细节。

### 2.4 输出文件

```
/home/tcat/diffcsp_exp5/logs/composite_score_val.txt
/home/tcat/diffcsp_exp5/logs/composite_score_test.txt
/home/tcat/diffcsp_exp5/logs/composite_score_per_sample_val.csv
/home/tcat/diffcsp_exp5/logs/composite_score_per_sample_test.csv
/home/tcat/diffcsp_exp5/logs/min_d_violations_val.csv     ← ⭐ Exp5' 调 lambda 用
/home/tcat/diffcsp_exp5/logs/min_d_violations_test.csv
```

`composite_score_<split>.txt` 格式:
```
=== EXP5 V2 SA-METRICS-V3 COMPOSITE SCORE - <split> ===
Total samples:           7621
min_d gate pass:         X / 7621 (XX.X%)
min_d gate fail:         X / 7621 (XX.X%)

--- Composite score (gate-pass samples only) ---
Total weighted mean:     0.XXX
  shell-1 coord_n F1:    0.XXX  (weight 0.20)
  shell-1 distance F1:   0.XXX  (weight 0.20)
  shell-1 elem (CNO eq): 0.XXX  (weight 0.20)
  shell-2 coord_n F1:    0.XXX  (weight 0.10)
  shell-2 distance F1:   0.XXX  (weight 0.10)
  shell-2 elem (CNO eq): 0.XXX  (weight 0.10)

--- Distribution detail ---
min_d distribution:      mean=X.XX, median=X.XX, p10=X.XX, p1=X.XX (Å)
samples with min_d <1.0: XX
samples with min_d <0.5: XX
samples with min_d <0.1: XX  (essentially overlap)
```

### 2.5 SA-METRICS-V3 红线(Exp5 MA2 写 handoff 时强调)

- ❌ 不重 sample(用 SA3' 已有 predictions)
- ❌ 不动 step5_2 的 4 个 v2 算法函数(写新文件 step5_3)
- ❌ 不动 ckpt
- ❌ 不动 holdout
- ❌ 不动 yaml / 训练代码
- ❌ 任何不确定的事 → 写脚本让用户跑 confirm,不靠记忆(用户原话)

---

## §3 关键已知 bug + 工程债务(Exp5 MA2 接手前必看)

### 3.1 SA1' 投影 ablation R_max fallback bug

`step5_2_compute_metrics.py` 内 `compute_projection_ablation_rmsd` 函数:
- **应做**: 从 shell_boundaries.pkl per-sample lookup R_max(每样本不同)
- **实做**: SA1' fallback 到 R_max=5.5 Å 全局值(SA3' OUTPUT §3.1 已暴露)

**后果**: SA3' 投影 ablation 报告 Δ=0 / 0 atoms_projected,被解读为 "v2 真物理改进",
实际只证明 pred 在 box 内(box 半对角线 √3×3=5.196 Å,5.5 Å 涵盖整个 box),什么都没说。

**Exp5 MA2 任务**: SA-METRICS-V3 写 step5_3 时正确读 shell_boundaries.pkl,**不动** step5_2 的旧函数
(留作历史对照,后续 v3+ 可能要 deprecate)。

### 3.2 MAX_EPOCHS in code not yaml

`step4_2_train.py` line 83 `MAX_EPOCHS = 500`(SA2'' 改成 700)是 Python 常量,
**不在 yaml 字段里**。理由:SA1' 决策"epoch 数是训练 orchestration 不是模型 hyper-parameter"。

**Exp5 MA2 注意**: Exp5' 续训时改 max_epochs 改 train.py line 83,不要去 yaml 找。
同时 line 183 `CosineAnnealingLR(T_max=MAX_EPOCHS)` 也会自动跟随,**LR schedule 会按新 T_max 重新规划**(详 §3.3)。

### 3.3 LR scheduler T_max 行为(SA2'' 续训关键发现)

`CosineAnnealingLR` 会根据当前 epoch / T_max 比例算 LR,改 T_max 会改整条曲线:

| T_max | epoch 484 处 LR | 比例 |
|---|---|---|
| 500(SA2 训练时)| 1.25e-6 | 1.25× eta_min |
| 700(SA2'' 续训时)| 2.25e-5 | **22.5× eta_min** |

**Exp5' 续训设计需考虑**: 从 epoch 529 + 加 pairwise penalty 续训:
- 选项 a: max_epochs=729(只 +200 epoch),epoch 529 处 LR ≈ ???(需计算)
- 选项 b: 重启 LR scheduler from fresh cosine
- 选项 c: 用 ConstantLR freeze(SA2'' 时排除过,但 Exp5' 不一样)

Exp5 MA2 决定时跑 dry-run 算公式(类似 SA2'' 的做法):
```python
import math
lr_init, eta_min = 1e-4, 1e-6
for T_max in [729, 800]:  # 候选
    for epoch in [529, 600, 700, 729]:
        t = min(epoch, T_max)
        lr = eta_min + 0.5 * (lr_init - eta_min) * (1 + math.cos(math.pi * t / T_max))
        print(f'T_max={T_max} epoch={epoch}: lr={lr:.4e}')
```

### 3.4 PL ModelCheckpoint save_top_k=1 删旧 best

PL 训练新 best 出现时,删旧 active best(保留 .ckpt 内容到新文件名)。
**Exp5' 训练前**: 先 cp 当前 active(epoch 529)到 .frozen 防覆盖。

### 3.5 PL Callback `on_validation_epoch_end` 与 `current_epoch == 200` 不触发

SA2 时代 milestone Callback bind 在 val 周期上,但 `check_val_every_n_epoch=5` 让 epoch 200 不是 val epoch。
**Exp5 MA2 写 Exp5' callback 时**: 用 `on_train_epoch_end` 钩子或 `current_epoch >= 200 and not self._fired` 一次性 latch。

### 3.6 v1 SA1 改的 datamodule 命名 v1→v2 contract

v1 SA1 把 datamodule 内部从 `.train_dataset` 改成 `.train_ds`(SA1' 实施时延续):
- `xas_local_datamodule_v2.py` 公开属性 `.train_ds / .val_ds / .test_ds`
- 任何 fork Exp4 模板的脚本都可能误用 `.train_dataset` → AttributeError
- SA2' 已踩过(line 219 α' patch)

**Exp5 MA2 注意**: SA-METRICS-V3 不直接用 datamodule(读 predictions_v2_*.pt),无 risk。
但 Exp5' 任何新代码动 datamodule 时,grep `.train_dataset|.val_dataset|.test_dataset` 防误用。

---

## §4 红线汇总(Exp5 MA2 全程不动)

### 4.1 数据红线
- ❌ 不动 holdout(`/home/tcat/diffcsp_exp4/data/holdout_samples_v2.csv`)
- ❌ 不动 incompat_pool.csv
- ❌ 不动 shell_boundaries.pkl(只读)
- ❌ 不动 .frozen ckpt(永久 safety net)

### 4.2 代码红线
- ❌ 不升级 7 守卫包
- ❌ 不动 cspnet.py(Phase 6.5 site 3,Exp4 backbone)
- ❌ 不修 Phase 6.5 hardcoded fp32(永久 SKIPPED-by-design)
- ❌ 不动 step5_2 的 4 个 v2 算法函数(SA-METRICS-V3 写新文件 step5_3)
- ❌ 不修 SA1' 的 5.5 Å fallback(留作历史 bug 锚点,SA-METRICS-V3 在新脚本里做对就行)

### 4.3 流程红线
- ❌ Exp5 MA2 写完 SA handoff 必先 review 给用户,不直接发
- ❌ Sub-agent 中期报告不在 proposal 锁定方向上必停
- ❌ Exp5 MA2 上下文 70% 闸门是硬线,接近时主动 transition 到 MA7
- ❌ 任何技术判断先 conversation_search + 列证据,不直接套结论
- ❌ 任何不确定的事写脚本让用户跑 confirm

---

## §5 Exp5 MA2 启动前用户应跑的 verify 块

转给 Exp5 MA2 时,**用户先跑 EXP5_FILE_GUIDE_FINAL.md §9 完整 verify 块**,贴回输出给 Exp5 MA2。

Exp5 MA2 看 verify 输出后第 1 件事:
1. 复述本 handoff §0.3 格式回复
2. 写 SA-METRICS-V3 launch note 草稿(参考 EXP5_SA3_PRIME_LAUNCH_NOTE 格式,~200 行)
3. 给用户 review 后再发

---

## §6 给 Exp5 MA2 的 7 条 lessons learned(MA5 临走)

1. **数学完备 ≠ 物理完备** — Multiset Macro-F1 是数学指标,不是物理指标。Exp5 MA2 设计任何评估应主动问"这个 metric 真的反映物理吗?"

2. **Min pairwise distance 是 ExpN 不变量** — 任何 diffusion 生成原子坐标的 Exp,评估必须包含原子两两距离检测,1.5 Å 物理硬下限。

3. **R_max / shell 边界禁用 fallback** — Exp4 已有 per-sample ground truth(shell_boundaries.pkl),
   任何 ExpN 评估"懒一懒用 5.5 Å fallback"会隐藏盲区,直到下游(用户物理统计)才暴露。

4. **MA review SA 设计时主动质疑指标 menu 的完备性** — MA5 review SA1' 4 个新函数时
   只 verify 算法是否正确,没问"覆盖所有 collapse 模式吗"。Exp5 MA2+ 应主动补这一问。

5. **last.ckpt ≠ best.ckpt** — PL 训练自然完结到 max_epochs 时,last 是终点(可能已过 best)。
   续训前必 md5 verify。

6. **LR warm restart 对续训有真实价值** — 22.5× LR 跳跃让 SA2'' 跳出 SA2' 局部最小,
   找到更深最小值。Exp5' 续训也应有 LR warm restart 机制。

7. **小补丁也要 MA ack + diff** — line 219 α' patch 是 2 字符 rename,但 MA5 ack 流程
   保证了 scope 不扩大。Exp5 MA2 接手任何"小改动" — 全程 ack + diff,不放过。

---

## §7 final 移交宣告

- **MA5 上下文**: 接近 70% 闸门,主动 transition
- **当前 Exp5 v2 verdict**: ⚠️ AMBER + 物理无效(min_d 违反),需 SA-METRICS-V3 量化 + Exp5' 修复
- **Exp5 MA2 立刻动作**: SA-METRICS-V3 launch note 草稿 → 用户 review → SA-METRICS-V3 启动
- **预计 Exp5 MA2 一棒到 Exp5 v2 final report v2**: SA-METRICS-V3(2-3h)+ Exp5' 重训(14-19h)+ SA4' figure(3-4h)+ final report(2-3h)≈ 21-29h 总,跨多个对话窗口

MA5 的所有 deliverable 落 `/mnt/user-data/outputs/`(已 present 给用户):
- EXP5_PROPOSAL_v2_AMENDED.md
- EXPERIMENT5_FINAL_REPORT_v1.md
- EXP5_FILE_GUIDE_FINAL.md
- EXP5_MA2_HANDOFF.md(本文件)

用户上传给 Exp5 MA2: 上面 4 份 + EXP4_FINAL_REPORT_ERRATA_2.md(从 Exp4 资产拿)。

---

*MA5 撰写,2026-05-01,移交 Exp5 MA2 前最后一份 deliverable。基于 SA1'/SA2'/SA3'/SA2'' 全程产出 +
用户 2026-05-01 物理统计发现 + Exp4 Step 2.5 ground truth(shell_boundaries.pkl,md5 cf2050e4...)。*
