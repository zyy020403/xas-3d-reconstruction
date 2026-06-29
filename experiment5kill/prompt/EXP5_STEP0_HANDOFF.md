# EXP5_STEP0_HANDOFF.md
# Exp5 Step 0 Sub-Agent 交接文档:Multi-Sample Averaging Quick Win

> **撰写者**: Exp5 Main Agent
> **日期**: 2026-04-28
> **接收人**: Exp5-SA0(quick win Sub-Agent)
> **与 SA1 的关系**: 完全独立,并行启动,零依赖。SA0 只读 Exp4 资源(ckpt + sample 脚本 + val 数据),不碰 Exp5 任何代码

---

## 0. 你的使命(一句话)

用 **Exp4 best ckpt + 现有 sample 脚本**,跑 K-sample TTA(test-time augmentation)看能不能 0 训练成本把 RMSD / TypeAcc 提一档。**不训练。不读 holdout。不动 Exp4 任何文件。** 半天到 1 天交付。

---

## 1. 必读背景

1. **EXPERIMENT4_FINAL_REPORT.md §10 方向 5** — 这是你做的事情的源头(MA5 "Multi-Sample Test-Time Averaging",标 ⭐,排名 #5)
2. **EXPERIMENT4_FINAL_REPORT.md §5.1** — 你的 K=1 baseline 数字(val RMSD=1.4849, TypeAcc=0.1877, pred_in=18.93/20)。SA0 跑 K=1 必须复现到 3 位小数,否则环境 / 代码 / 数据有 drift
3. **EXPERIMENT4_FINAL_REPORT.md §9.2** — 文件读取代码片段(predictions_*.pt schema 在这里)
4. **EXP4_STEP5AGENT 的 step5_1_sample.py**(`/home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py`) — 这是你的 sampling 入口,改造它跑 K 次

---

## 2. 关键设计抉择(SA0 必须读完再开始)

### 2.1 SA0 要回答的核心问题

| 问题 | 期望答案 |
|------|---------|
| K-averaging 真有提升吗? | 量化提升幅度,K=5/10 vs K=1,看 ΔRMSD / ΔTypeAcc |
| 哪种聚合策略对? | naive slot-wise 平均 vs Hungarian-aligned 平均,哪个 work |
| 跨 Tier 是否一致? | hard sample(O2 collapse 的)是不是受益更多?或反之 |
| K=10 vs K=5 边际收益如何? | 决定 Exp5 后续是否标配 K-averaging |

### 2.2 关键 ambiguity:K 个 sample 之间的 slot 顺序

DiffCSP 反扩散从随机噪声出发,**K 个独立 sample 之间 slot 顺序未知是否稳定**:

- **假设 A(slot 稳定)**:slot k 在 K 个 sample 里都对应"距中心第 k 近"那个真实原子。naive slot-wise 平均直接 work。
- **假设 B(slot permutation-symmetric)**:slot k 在 sample 1 和 sample 2 里可能对应不同真实原子。naive 平均会把两个不同原子的位置糊在一起,**反而变差**,必须先两两 Hungarian 对齐。

**Exp2/Exp4 没人验证过 A/B 哪个对**。SA0 必须**两种聚合策略都实现并对比**,数据说话。

### 2.3 两种聚合策略

#### 策略 X(naive,假设 A 成立时正确)

```python
# K samples per evaluation case, each (20, 3) coord + (20,) type
all_coords = np.stack([sample_k_coords for k in range(K)])  # (K, 20, 3)
all_types  = np.stack([sample_k_types for k in range(K)])    # (K, 20)
agg_coords = all_coords.mean(axis=0)                         # (20, 3)
agg_types  = scipy.stats.mode(all_types, axis=0).mode        # (20,) majority vote per slot
# 然后用 (agg_coords, agg_types) 与 true 做 Hungarian 评估(同 Exp4 标准)
```

#### 策略 Y(Hungarian-aligned,假设 B 成立时正确)

```python
# 选 sample 0 当 anchor
anchor_coords, anchor_types = samples[0]
aligned_coords = [anchor_coords]
aligned_types  = [anchor_types]
for k in range(1, K):
    # 把 sample k 通过 Hungarian 对齐到 anchor
    cost = pairwise_distance(samples[k].coords, anchor_coords)  # min-image
    row, col = linear_sum_assignment(cost)
    # 让 sample k 的 row[i] slot 对齐到 anchor 的 col[i] slot
    reordered_coords = np.zeros_like(anchor_coords)
    reordered_types  = np.zeros_like(anchor_types)
    reordered_coords[col] = samples[k].coords[row]
    reordered_types[col]  = samples[k].types[row]
    aligned_coords.append(reordered_coords)
    aligned_types.append(reordered_types)
agg_coords = np.stack(aligned_coords).mean(axis=0)   # (20, 3)
agg_types  = scipy.stats.mode(np.stack(aligned_types), axis=0).mode
# 然后同样与 true 做 Hungarian 评估
```

**SA0 必须两种都跑**,在 results 表里报。

### 2.4 K 的扫描范围

- **必跑 K ∈ {1, 5, 10}**
- **可选 K = 20**(如果 K=10 vs K=5 还有显著提升,跑 K=20 看是否平台;如果 K=10 已平台,不必)

K=1 必须**与 Exp4 final report §5.1 的 val 数字到 3 位小数一致**(RMSD=1.4849, TypeAcc=0.1877)。这是环境和代码没 drift 的证据。

---

## 3. 数据子集(降低 wall time)

跑全 val(7,621 sample)× K=10 大约 30+ 小时,**不可接受**。子采样:

- **随机抽 500 个 val sample,seed=0**,**按 eval_cutoff Tier 分层**(防止某 Tier 抽空)
- 大致比例(参考 Exp4 §5.2 val Tier 分布):Tier B ~25%、Tier C ~50%、Tier D ~22%、Tier A 跳过(全 val 才 13 个,统计噪声大)
- 每 Tier 抽样比例与 val 真实比例匹配

预算 wall time(单 RTX 4090):
- Exp4 Step 5 报告 9h / ~12000 sample = ~2.7 s/sample(含 I/O)
- 500 sample × K=1: ~25 min
- 500 sample × K=5: ~2.1 h
- 500 sample × K=10: ~4.2 h
- 总计 ~6.5 h(K=1+5+10),加 K=20 多 ~8 h

如果 K=10 vs K=5 显示明显边际收益,K=20 值得;否则 K=20 可跳。

---

## 4. 实施步骤

### 4.1 准备

```bash
# 工作目录(独立于 SA1 的 /home/tcat/diffcsp_exp5/code/)
mkdir -p /home/tcat/diffcsp_exp5/sa0/{scripts,results,logs}
cd /home/tcat/diffcsp_exp5/sa0/scripts

# 复制 Exp4 sample 脚本作改造起点
cp /home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py multisample.py
cp /home/tcat/diffcsp_exp4/code/step5/step5_2_compute_metrics.py multisample_metrics.py
```

### 4.2 改造 multisample.py 的关键点

1. **加载 Exp4 ckpt**:`/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt`(沿用 Exp4 现有 model 实例化路径,**model 用 Exp4 的,不要用 Exp5 SA1 还在改的代码**)
2. **子采样 500 sample**:用 pandas + seed=0 + Tier 分层,产出固定的 sample_name 列表(存 `results/sa0_subset_500.csv` 作 audit anchor)
3. **K-loop**:对每个 sample,跑 K 次 `model.sample(...)`,保存每次的 (frac_coords, atom_types)
4. **聚合**:实现 §2.3 两种策略
5. **评估**:用 Exp4 现成的 Hungarian 函数(从 step5_2_compute_metrics.py 抽)算 RMSD / TypeAcc / pred_in_cutoff

### 4.3 输出

| 路径 | 内容 |
|------|------|
| `results/sa0_subset_500.csv` | 抽样的 500 sample_name + tier(audit anchor) |
| `results/multisample_raw.csv` | per-sample-per-K-per-strategy 原始指标(N=500 × K × 2 行) |
| `results/multisample_results.md` | summary 表 + 结论(详见 §5) |
| `results/multisample_K_curves.png` | K vs RMSD / TypeAcc 曲线(2 panel,naive 和 Hungarian 各一条) |
| `logs/multisample.log` | 跑完的完整 log(包括 wall time、K=1 与 Exp4 的一致性核对) |

---

## 5. multisample_results.md 模板(SA0 必产)

```markdown
# Exp5 SA0: Multi-Sample Averaging Quick Win Results

## Subset
- 500 val samples, seed=0, stratified by Tier
- Distribution: Tier A=0, B=125, C=250, D=125 (or 实际值)

## Sanity check: K=1 vs Exp4 final report §5.1
| Metric | SA0 K=1 (subset 500) | Exp4 full val (7621) | Δ | Pass? |
|---|---|---|---|---|
| RMSD | 1.4XXX | 1.4849 | <0.02 | ✓/✗ |
| TypeAcc | 0.1XXX | 0.1877 | <0.01 | ✓/✗ |
| pred_in_cutoff | 18.9X | 18.93 | <0.05 | ✓/✗ |

(K=1 必须与 Exp4 数字大致一致到子集采样误差范围。如果偏差 > Tier 加权采样标准误,环境有问题,SA0 在这里停下来上交 Main Agent。)

## Main results table
| K | Strategy | RMSD | ΔRMSD vs K=1 | TypeAcc | ΔTypeAcc vs K=1 | pred_in_cutoff |
|---|----------|------|--------------|---------|-----------------|-----------|
| 1 | (n/a) | x | 0 | y | 0 | z |
| 5 | naive | | | | | |
| 5 | hungarian | | | | | |
| 10 | naive | | | | | |
| 10 | hungarian | | | | | |
| 20 | naive | (optional) | | | | |
| 20 | hungarian | (optional) | | | | |

## Per-Tier breakdown(可选,but useful)
[同样的表,按 Tier B/C/D 分别报,看 hard sample 是否受益更多]

## Conclusion
- 哪种策略 win:naive 还是 Hungarian-aligned?
- K=5 vs K=10 边际收益:还有用还是平台?
- ROI 评估:K=5 wall time 多 5×,RMSD 提多少 / TypeAcc 提多少?对 Exp5 后续是否值得标配?

## Open questions for Main Agent
[任何 SA0 不确定的事]
```

---

## 6. 验收闸门

**5/5 必须通过才能交棒**:

1. ✅ K=1 复现 Exp4 §5.1 数字到 3 位小数(子采样误差范围内)
2. ✅ K=5 和 K=10 都跑完了 naive + Hungarian 两个策略
3. ✅ multisample_results.md 包含 §5 模板的所有表格(K=20 可选)
4. ✅ multisample_K_curves.png 渲染成功,曲线趋势清晰
5. ✅ 写完 EXP5_STEP0_OUTPUT.md 上交 Main Agent,内容包含:
   - 实测各 K + 策略的指标
   - 你对 "naive vs Hungarian 哪个 work" 的明确判断
   - 你对 "K-averaging 是否应该作为 Exp5 后续标配" 的明确建议
   - 任何 OPEN QUESTION

---

## 7. 红线(绝对不能动)

- `holdout_samples_v2.csv` / `spectra_holdout.pkl`:**全程不读**
- `incompat_pool.csv`:**封存**
- `/home/tcat/diffcsp_exp4/`:**read-only**(包括 ckpt、code、data),不写不删
- `/home/tcat/diffcsp_exp5/code/`:**SA1 的领地,SA0 不写不读**(SA0 只用 `/home/tcat/diffcsp_exp5/sa0/`)
- 不训练,不改 model 权重,**只 inference**
- 网络环境守卫包不升级(详见 Exp4 final report §9.3)

---

## 8. 时间预算

预估半天写代码 + 1 天跑 + 半天分析 = 1-2 天 wall time。其中纯 GPU 占用 ~6-8 小时。

如果 K=1 sanity check 就过不了(数字与 Exp4 偏差太大),**立即停下来上交 Main Agent**——这是基础环境问题,不是 SA0 自己能 debug 的。

---

## 9. 与 SA1 的关系

- **SA1**(架构改造):写 Exp5 baseline_v2 新代码,完全独立路径
- **SA0**(本任务):用 Exp4 现成 ckpt 跑 TTA 实验,完全独立路径
- 两者**零文件冲突**(SA1 写 `/home/tcat/diffcsp_exp5/code/`,SA0 写 `/home/tcat/diffcsp_exp5/sa0/`)
- 两者**零 GPU 冲突**(都是单 GPU 任务,服务器只一张卡的话需排队;但 SA0 ~6h GPU,SA1 主要是 CPU+小 forward test,基本不冲突)
- 一个 GPU 的话:**SA0 跑 GPU,SA1 写代码** —— 几乎完美 dovetail

如果服务器允许两张卡同时跑,完全并行;如果只一张卡,SA0 跑 K=5 时 SA1 正好可以做代码 review 和 forward_test 准备。**Main Agent 协调,SA0 不必管这件事**。

---

## 10. SA0 的小 deliverable 也是 Exp5 的大 input

SA0 的结论决定 Exp5 后续两件事:

1. **Phase A 的 SA3(评估)是否在 sample 时跑 K-averaging?** 如果 SA0 证明 K=5 hungarian 提升 RMSD 0.05+ TypeAcc 0.03+,SA3 就标配 K=5。如果 SA0 证明 naive 也行,更省事。
2. **Exp4 vs Exp5 baseline_v2 的对比是否 K-aligned?** 为了 fair compare,Exp4 数字也要用同一个 K。如果 SA0 的 K=5 提升明显,Exp4 数字要重报一组 K=5 的 baseline,Exp5 baseline_v2 也用 K=5。

所以 SA0 不只是 "free win",**它也是 Exp5 评估方法论的 calibration step**。重要性不低。

---

*Exp5 Main Agent 撰写,2026-04-28。SA0 接收后请在 24h 内回 ack 并报当日 K=1 sanity check 结果(优先于 K=5/10)。*
