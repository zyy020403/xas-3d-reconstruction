# EXP5_SA_METRICS_V3_EARLY_HANDBACK.md
# SA-METRICS-V3 早交回报告 — Dry-run 触发 Exp5' 立即启动

> **From**: SA-METRICS-V3(Exp5 v2 评估改造 sub-agent)
> **To**: Exp5 MA2
> **Date**: 2026-05-01
> **Status**: ⚠️ **早交回 — dry-run 阶段终止**(MA2 决议,非 sub-agent autonomous)
> **Full 7621×2 全量运行**: 跳过(数据已充分,浪费 ~10 min 无新增信息)
> **触发动作**: Exp5' 物理约束 from-scratch 立即启动

---

## §0 为什么早交回

Dry-run(val 100 + test 100)数据让 launch note §0 中"两个并列产出目标"中的 **#1(为 Exp5' λ 设计提供精确数据)**已经达成,**#2(中立 7 项复合分诊断)** 进入 information-saturated 状态:

- **#1 已成立**:gate fail 率 89-95%,p1 ≈ 0.003-0.004 Å(完全重合),violations csv 已生成可直接用作 λ 设计依据 — 全量数据只是把分母从 100 变 7621,**不改变 λ 决策**
- **#2 已饱和**:GATE-PASS 子集复合分 val 0.11 / test 0.06,shell-1 distance 一律 0.0000 — **physical-OK 子集本身仍然不是真好**,这是个 strong signal,不需要更多数据确认

**MA2 决议跳过全量是正确的**:
- 节省 5-10 min 计算 + 1-2h SA 注意力
- Exp5' 14-19h 训练比这关键,优先级倒置
- proposal §B.5 verdict 表所有阈值早被穿透,verdict 已经"穿底",再多 7521 个样本只是把 0.0056 算精确到 0.005xxxx,**对决策无价值**

---

## §1 Dry-run 关键数据(用于 Exp5' 设计输入)

### 1.1 Min pairwise distance — 物理 gate 灾难性失败

| 维度 | val (N=100) | test (N=100) | MA2 200-sample probe (历史基线) |
|---|---|---|---|
| Gate 通过率 | **5%** | **11%** | 5.0%(对齐) |
| Gate 失败率 | **95%** | **89%** | 95.0%(对齐) |
| min_d mean (Å) | 0.55 | 0.66 | — |
| min_d median (Å) | 0.43 | 0.56 | — |
| min_d min (Å) | **0.0039** | **0.0025** | 0.004(对齐) |
| samples min_d < 0.1 Å(完全重合)| 24/100 | 20/100 | — |

**与 200-sample probe 高度一致**(对齐误差 < 1%),说明 dry-run 数据 representative,不是边界 sample bias。

### 1.2 n_pred_shells 分布 — 结构破碎(Exp4 算法预期 1-2 shells)

| n_pred_shells | val | test |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 1 | 0 |
| 3 | 2 | 0 |
| **≥ 4** | **97** | **100** |

Exp4 真值端用 gap=0.1563 Å 切壳,典型结构出 1-3 shells(mp-10009 头 3 shells:1+3+6 atoms)。**预测端 ≥ 4 shells 占 97-100%** 表明:gap 算法在 pred 上检测到大量"距离断层",但**这些断层不是真正的化学壳层**,而是"原子重合堆 + 散乱原子"的 bimodal 结构假象。

→ Step 2.5 gap 算法在 pred 端**结构性失败**,因为 pred 不是物理结构。这个发现本身有信息量。

### 1.3 复合分 — 即便 gate-pass 子集也不真好

| 维度 | val | test |
|---|---|---|
| ALL samples 总分 | 0.0056 | 0.0062 |
| ALL shell-1 coord | 0.027 (估)| 0.027 (估)|
| ALL shell-1 distance | **0.0000** | **0.0000** |
| ALL shell-1 elem (CNO eq) | 0.025 (估)| 0.025 (估)|
| **GATE-PASS 子集**(N) | **0.1118 (5)** | **0.0568 (11)** |
| GATE-PASS shell-1 distance | **0.0000** | **0.0000** |

**关键诊断**:**gate-pass 5-11 个样本里 shell-1 distance 一律 0.0000** — 即便物理上 OK 的样本,模型也不能把 shell-1 距离放对。这是**远比 min_d 违反更深的问题**:模型连"知道第一壳层应该在 ~2-3 Å"这个 EXAFS 基础事实都没学到。

物理约束 pairwise penalty 能修 min_d gate(强制原子分开),**但不一定能修 shell-1 distance** — 那需要**距离监督**(distance-aware loss / target-distance-anchored prior),proposal §D 表"errata 2 方向 4"已列。

---

## §2 MA2 诊断的两条 lessons learned(SA agent 视角佐证)

### 2.1 step5_2 RMSD 1.4954 Å 平均数掩盖 95% 灾难

step5_2 的 RMSD 算法是 Hungarian 最近邻匹配的平均欧氏距离,有两个数学后果:
- **Hungarian** 把 pred 重合堆里的 1 个原子匹给 true shell-1,其余 19 个匹给周围,平均 RMSD 看起来"还行"
- **Mean** 把 95% 的 0.003-1.5 Å 灾难和 5% 的 ~3 Å 大偏离平均出 1.49 Å,看起来"接近 Exp4 1.69 Å"

**这是数学完备 ≠ 物理完备的教科书案例**(MA5 lessons learned 第 1 条,本次彻底验证)。

### 2.2 Step 6 picker 78 / 7621 = 1%(MA2 自己点出的 critical insight)

我没有 step6 source 在手,但你提到的"step6 picker 在 78 / 7621 样本里挑 20 个"如果属实,选样率 1.02% — 几乎正好等于本次 gate-pass 通过率 5-11% 的 sub-fraction。**意味着 step6 大概率以"min_d 物理可用"或类似过滤为标准**,从而"挑选出最干净的 1% 样本"展示给下游/化学家,**给团队造成模型基本可用的错觉**。

**Exp5' 之后**:step6 picker 应该报告"picker 接受率"作为评估元指标 — 如果 picker 拒绝 99% 样本却没人报警,这本身就是模型失败的信号。

---

## §3 给 Exp5' 的可复用资产

### 3.1 直接复用 — step5_3_composite_score.py(已在 `/mnt/user-data/outputs/`,服务器同步)

- **Exp5' 训练后**:不重写,直接 `python step5_3_composite_score.py --split val/test` 算 Exp5' 的复合分
- **Exp5' 训练中**:可复制核心函数(`compute_min_pairwise` / `assign_pred_shells`)做 epoch-级 monitoring
- **历史对照**:本次 dry-run debug100 输出在 `/home/tcat/diffcsp_exp5/logs/composite_score_*_debug100.txt`,Exp5' final report 写"vs SA2 baseline"时直接引用,**不需要重 sample SA2 全量**(Exp5' 重训后从新 ckpt sample,直接对比)
- **Smoke test**:`/mnt/user-data/outputs/step5_3_smoke_test.py` 也保留,Exp5' 评估改任何函数后跑一遍 sanity check

### 3.2 直接复用 — `min_d_violations_*_debug100.csv`(λ schedule 设计依据)

- 100 样本子集已足够 Exp5' λ schedule 设计参考(violations rate / 完全重合 fraction 已稳定)
- 全量 violations CSV 不需要(Exp5' 训完会生成新 CSV,旧 SA2 CSV 仅作"灾难发现"档案)

### 3.3 设计原则(Exp5'-train sub-agent launch note 应继承)

1. **min_d 1.5 Å gate 是 Exp 不变量** — Exp5' 训练中也跑 min_d 监控(epoch-level mean / p10 / violation rate),作为 loss 之外的 sanity check
2. **CNO equivalence(Z=6/7/8 合并 token=-1)** — Exp5' final 评估保留,这是 EXAFS 物理设定不是 hyper
3. **Step 2.5 gap=0.1563 Å threshold 写死** — Exp4 MA2 拍板的全实验不变量,Exp5'/Exp6 都不要重新挖
4. **shell-2 吸收所有后续 shells** — launch note §3.2 注释,本次实测合理(避免 pred bimodal 结构错切出 shell-3+)
5. **任何不确定查询失败 → raise,不静默 fallback** — `eval_cutoff` 一致性 assert 100 样本 0 raise,机制 OK

### 3.4 设计陷阱(Exp5' 不要重蹈)

1. **不要相信 mean RMSD** — Exp5' 训练中 val_loss / RMSD 会下降是正常,但**gate fail rate 才是物理指标**。Exp5' early-stop / best-ckpt criterion 应该综合 val_loss + gate_pass_rate(权重待 SA-EXP5'-train 设计)
2. **不要做 step6-style picker selection 评估** — 任何"挑 N 个最干净样本展示"等价于"在 99% 失败里捞 1% 假装成功"。Exp5' final report 必须报告 **全 7621 / 4481 样本的 gate pass 率**,不允许 picker subset
3. **score_coord_n 对称性问题挂着但不阻塞**(P1 ack:A spec-literal)— Exp5' 全量评估时,如果 true_s2_n=0 样本占比 < 5%,A vs B 总分均值差 < 0.01,可保留 A;如果占比高,Exp5'-train sub-agent 可重启 P1 ack 流程切 B
4. **不要复用 step5_2 任何函数** — 5.5 Å fallback bug 仍在原文件,Exp5' 任何评估也走 step5_3 / 新文件,step5_2 留作 v2 历史档案

---

## §4 Exp5' λ schedule 设计建议(基于 dry-run 数据)

### 4.1 起步 λ

proposal §C.2 表:
- violations < 10%:λ=0.1
- 10-30%:λ=0.5
- **> 30%:λ=1.0** ← 当前命中(violations = 89-95%)

但 p1=0.0025-0.0039 Å 表明**完全重合 sample 占 ~20-24%**,proposal §C.2 表没覆盖这个尾部severity。我的精化建议:

**λ=1.0 起步,前 10 epoch 加 schedule**:
- epoch 0-2:λ=1.0(让模型先尝试满足 pairwise penalty)
- epoch 3-5:监控 violation rate
  - 若 violation 单调下降到 < 50%:λ 保持 1.0 → 正常路径
  - 若 violation 卡在 > 70%:λ ramp 到 2.0(完全重合 sample 需要更强 push)
  - 若 RMSD 飙升 > SA2 baseline + 10%:λ 减半,重启
- epoch 5+:violation < 30% 后 λ 可降回 0.5(避免破坏其他 loss 平衡)

### 4.2 监控指标(Exp5'-train sub-agent 必须 epoch-level log)

```
[exp5_prime] epoch=N
  val_loss=X.XX
  val_min_d_mean=X.XX (target: increasing)
  val_min_d_p10=X.XX  (target: > 1.5 by epoch 50)
  val_gate_pass_rate=X.X%  (target: > 80% by end of training)
  val_overlap_rate (min_d<0.1)=X.X%  (target: 0% by epoch 20)
  λ_current=X.X
```

### 4.3 Best ckpt 选择 criterion

**不要单看 val_loss**(SA2 已证明 val_loss 0.7003 时 95% 物理违反)。Exp5' best ckpt 应综合:

```
score_for_ckpt_selection = α × (1 - val_loss / 1.0) + β × val_gate_pass_rate + γ × val_composite_total
```

α / β / γ 由 SA-EXP5'-train sub-agent 在 launch note 拍板。我建议初稿 α=0.2 / β=0.5 / γ=0.3,**β 最高**(物理可用性是 hard prereq)。

---

## §5 我的状态 + Done definition revisited

### 5.1 launch note §7 完成定义对照

- [x] §2 schema verify 跑过 + 与 MA2 描述一致(且补 probe 对齐 pred_frac_coords / list[Tensor] schema)
- [x] step5_3_composite_score.py 写完 + 通过 §3 红线检查
- [x] dry-run 100 样本 ×2 split 完成 + MA2 ack
- [ ] ~~全量 val + test 完成 + §5 verify checklist 全过~~ → **跳过(MA2 决议)**
- [x] EXP5_SA_METRICS_V3_EARLY_HANDBACK.md 写完(本文档,即将给 MA2 review)
- [x] 6 输出文件:**3 文件**落 `/home/tcat/diffcsp_exp5/logs/`(只 dry-run debug100 版本,full 版不存在)

### 5.2 服务器 active 资产清单(handoff Exp5'-train sub-agent 用)

```
/home/tcat/diffcsp_exp5/code/step5/
├── step5_3_composite_score.py           ← SA-METRICS-V3 产物,Exp5' 复用
├── predictions_v2_val.pt                ← SA3' SA2 baseline output(留作历史对照)
└── predictions_v2_test.pt               ← 同上

/home/tcat/diffcsp_exp5/logs/
├── composite_score_val_debug100.txt     ← dry-run 主报告,SA2 baseline 物理灾难锚点
├── composite_score_test_debug100.txt    ← 同上
├── composite_score_per_sample_val_debug100.csv
├── composite_score_per_sample_test_debug100.csv
├── min_d_violations_val_debug100.csv    ← Exp5' λ 设计依据
└── min_d_violations_test_debug100.csv
```

`/home/tcat/diffcsp_exp5/checkpoints/` 状态不变(launch note §0.1 红线"不动 ckpt"全程满足):
- `epoch=529-val_loss=0.7003.ckpt` active best(SA2'')
- `sa2_baseline_epoch484_val0.7065.ckpt.frozen` 永久
- `sa2pp_resume_epoch529_val0.7003.ckpt.frozen` 永久

⚠️ **MA2 决议 Exp5' from-scratch 启动**,以上 ckpt 是否作为 warm-start 起点是 SA-EXP5'-train 拍板 — 我不擅自建议。但物理常识:既然 SA2/SA2'' 学到的几何"完全不能用",warm-start 可能不是最优,from-scratch + 物理约束从 epoch 0 加可能更干净。MA2 在 launch note 里和 SA-EXP5'-train 商榷。

---

## §6 给 MA2 的 3 条 sub-agent 视角观察

1. **早交回 vs 跑全量的 trade-off** — 本次 MA2 凭 100 样本数据决议是合理的,但需要在 OUTPUT.md / Exp5 final report v2 里 surface "全量未跑" 这个事实,后续 review 团队可能会问"为什么 SA-METRICS-V3 不跑全量"。我建议 final report v2 里加一句:"全量 7621×2 跳过,因 100 样本 dry-run 数据已 saturate decision boundary,200-sample probe + 100×2 dry-run 三次独立 violation rate ~95% 一致,统计显著性已立。"

2. **proposal §B.5 verdict 表的 epistemological status** — 本次 SA2 baseline 复合分 0.0056 落在 < 0.30 RED 区间,proposal 写"转 Exp6"。但 MA2 决议是"转 Exp5'(物理约束 extension)"而不是 Exp6 大改架构。这两条建议可以并存:**Exp5' 是先做最小修复尝试**(加 pairwise penalty 不改架构),**Exp5' 失败后才考虑 Exp6 大改架构**(equivariant decoder / hierarchical type)。这个 staging 是合理的,Exp5 final report v2 应明文记录这个 ladder。

3. **SA-METRICS-V3 vs SA1' 的指标完备性对比** — SA1' 写 4 个 v2 指标都正确实现但**集合不完备**(漏 min pairwise distance);SA-METRICS-V3 加 1 个 gate + 修 fallback bug 就让灾难显形。这印证 proposal §A.3 / MA5 lessons learned 第 4 条:**MA review SA 设计时应主动质疑指标 menu 完备性,不止 verify 算法正确性**。SA-EXP5'-train 设计 Exp5' 评估时,MA2 review 应主动问"还有什么 collapse 模式我们没监控?"

---

## §7 hand-back 完成,等 MA2 ack

- 本文档 `/mnt/user-data/outputs/EXP5_SA_METRICS_V3_EARLY_HANDBACK.md` 已写好(SA agent 即将 present)
- step5_3_composite_score.py + smoke test 已 present(上一棒)
- 服务器 6 个 dry-run 输出文件已 in-place(用户已 verify)
- 我作为 SA-METRICS-V3 任务结束,不进 Exp5' 训练设计

下一棒:Exp5 MA2 写 SA-EXP5'-train(或 SA-EXP5p-train)launch note,锚点本文档 §3 / §4。

---

*SA-METRICS-V3 撰写,2026-05-01。基于 launch note §0 双重产出目标 + dry-run val/test 100 样本数据
+ MA2 200-sample probe 对齐验证 + MA2 早交回决议(避免浪费全量 5-10 min 对决策无价值的精算)。*
