# STEP 2.5 PHASE B + PHASE C 合并报告

**报告对象**：DiffCSP-Exp4-Main-Agent 2
**Sub-Agent**：DiffCSP-Exp4-Step2.5-SubAgent
**阶段**：Phase B (阈值应用) + Phase C (多位点诊断)
**状态**：两项全部完成。**请 MA 决策多位点策略**。
**总 wall-clock**：12.8 s (Phase B 5.7 s + Phase C 7.1 s)

---

## Part 1: Phase B — 壳层边界产出

### 1.1 执行情况

| 指标 | 值 |
|---|---|
| Threshold | **0.1563 Å** (MA-selected, p10) |
| Built | 128,382 / 128,382 |
| Skipped | 0 |
| pickle size | **369.5 MB** |
| 不变量检查 (100 随机样本) | ✓ 全部通过 |
| Wall-clock | 5.7 s |

**交付物**（在 `experiment4/step2_5/`）：
- `shell_boundaries.pkl` — 核心产物，Step 3/5 直接 load
- `shell_stats_by_split.csv` — 4 行
- `shell_stats_by_element.csv` — 88 行
- `step2_5b_summary.txt`

### 1.2 关键统计（By split）

| split | n | mean_eval_cutoff | median | p5 | **p95** | mean_shell1_n | mean_shell1_outer | mean_shell2_outer | mean_n_shells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 102,660 | 4.71 | 4.40 | 3.60 | **6.74** | 4.58 | 2.24 | 3.08 | 10.25 |
| val | 12,912 | 4.72 | 4.41 | 3.60 | 6.80 | 4.54 | 2.24 | 3.08 | 10.28 |
| test | 7,696 | 4.71 | 4.40 | 3.60 | 6.67 | 4.68 | 2.26 | 3.11 | 10.30 |
| holdout | 5,114 | 4.75 | 4.39 | 3.60 | **9.39** | 4.54 | 2.22 | 3.07 | 10.04 |

**观察**：4 个 split 的 mean/median 几乎一模一样（4.7 / 4.4 Å），证明 mp-id-level split 没有产生分布偏移 ✓。shell1 mean 原子数 4.54-4.68，shell1 外缘 2.22-2.26 Å——物理合理。

### 1.3 ⚠ 观察：5-10% 样本 eval_cutoff 落到 r_cutoff = 10 Å 极限

**现象**：holdout split p95=9.39 Å，且 `by-element` top-10 里 **O / Li / P / F** 四个元素的 p95 全部 ≈ 9.99 Å（贴近 r_cutoff=10 Å）。

**物理解释**：这些样本在 4-10 Å 区间所有相邻距离间隙都 < 0.1563 Å，切不出"含第 20 邻居的壳层"的清晰外缘。fallback 到 `shell_ends[-1]`，把整个 [4, 10] 区域当一个大壳层。

**sanity check 的第一个样本就是这种情况**：
- shell_n_atoms = [5, 1, 2, 1, **212**]（最后一壳 212 原子）
- shell_ends = [1.77, 2.05, 3.23, 3.66, **9.98**] Å
- eval_cutoff = 9.98 Å（全 10 Å 域）

**对评估的影响**：对这 5-10% 样本，"分层 RMSD" 的第 2/3 壳与"尾部"区分会弱化，因为整个外层是一个大壳。**不是错误，但 Step 5 的分层评估报告里 MA 可能想单独标记这类样本**（例如按 eval_cutoff 分位数 tag "outer-diffuse" 子集）。

**完全遵守 handoff §5.7 规格**（fallback = `shell_ends[-1]`），无偏离。

### 1.4 By-element top-10 观察（物理合理性）

| element | n | shell1_n | shell1_outer | 解读 |
|---|---:|---:|---:|---|
| O | 22,441 | 2.01 | 1.78 Å | O 作为吸收原子的 shell 1 是 ~2 个配阳离子，典型 O-M 距离 1.8 Å ✓ |
| Li | 7,147 | 4.43 | 2.16 Å | Li 通常 4-6 配位 ✓ |
| Fe | 3,481 | 5.39 | 2.15 Å | Fe-O 八面体部分配位，~5-6 ✓ |
| Mn | 3,313 | 5.20 | 2.17 Å | 同上 ✓ |
| Cu | 2,518 | 4.40 | 2.21 Å | Cu 常见 4-5 配位（Jahn-Teller） ✓ |

所有元素的 shell 1 配位数与外缘距离都符合化学预期。**Phase B 产出质量达标**。

---

## Part 2: Phase C — 多位点诊断

### 2.1 执行情况

- 采样：5 per bucket × {2, 4, 8, 16} = 20 train 样本，`random_state=42`
- 中位处理时间 < 0.1 s/样本（首样本冷启动 5.6 s）
- Wall-clock 7.1 s
- 错误：0

### 2.2 两套判等指标给出不同结论

| 判等口径 | 等价比例 | MA 阈值 70% 对照 | 建议方案 |
|---|---:|---|---|
| Signature 严格相等 (含 `round(d, 0.01)` 离散化) | **40.0%** | < 70% | Option B 变体 |
| MAE ≤ 0.05 Å | 65.0% | < 70% | Option B 变体 |
| **MAE ≤ 0.10 Å** | **70.0%** | = 70% | **骑墙** |
| MAE ≤ 0.20 Å | 75.0% | > 70% | Option A |

**MAE 分布**（排除 INCOMPAT）：median = **0.0036 Å**，p90 = 0.0852 Å，max = 0.1863 Å。等价位点的实际距离差**极小**。

### 2.3 分桶细看（关键）

按 `n_center_sites` 分档看，性质完全不一样：

| 桶 | signature 等价 | MAE = 0 | MAE 最大值 | INCOMPAT 数 |
|---|---:|---:|---:|---:|
| n=2 | **5/5 (100%)** | 5/5 | 0.000 | 0 |
| n=4 | 3/5 (60%) | 3/5 | 0.105 | 0 |
| n=8 | 0/5 (0%) | 0/5 | 0.040* | **1** |
| n=16 | 0/5 (0%) | 0/5 | 0.186** | **3** |

\* 排除 INCOMPAT 样本后的最大 MAE
\** 同上

**分档观察**：
- **n=2 桶**：全部严格等价。2 位点 → 几乎必然是单一 Wyckoff 的对称等价对。
- **n=4 桶**：多数等价，少数有 MAE ~0.1 Å 的"近等价"。
- **n=8, n=16 桶**：signature 无一严格相等，但多数 MAE 很小（0.02-0.04 Å）——签名判严了。
- **INCOMPAT 占整体 20% (4/20)**：shell1 原子数本身不等，是真正不等价。

### 2.4 INCOMPAT 样本详情 — 这 20% 是"硬不等价"

| 样本 | center | shell1_sizes | 解读 |
|---|---|---|---|
| mp-20846 Ca | Ca | `[13,13,13,13, 2,2,2,2]` | 明显两种 Wyckoff：4 个位点有 13 邻居，4 个有 2 邻居 |
| mp-560146 Y | Y | `[7,6,7,7,6,7,7,7,7,7,6,7,7,7,6,7]` | 两类位点（shell1 size 7 vs 6）混合 |
| mp-559833 Na | Na | `[6,5,6,6,5,5,5,5,6,6,5,5,6,6,5,6]` | 两类位点（6 vs 5）混合 |
| mp-775145 O | O | `[3,4,4,4,4,3,3,3,4,2,4,4,4,3,2,3]` | **12 种不同 signature**，多种 Wyckoff |

这 4 个样本的位点实际属于不同的配位环境，**"选第一位点"对它们系统性偏差最大**。

### 2.5 MP EXAFS 协议（site-specific vs site-averaged）

**未查到**：Sub-Agent 无 mp-api 凭证，无法程序化访问 Materials Project metadata。

Summary 中留下三条人工查询指路：
1. MP docs: https://docs.materialsproject.org/ (搜 'XAS' or 'EXAFS')
2. 具体样本 XAS tab: https://next-gen.materialsproject.org/materials/mp-18658
3. 源文献: Mathew et al., Scientific Data 5, 180151 (2018)

**这条信息对 MA 决策至关重要**，建议 MA 查明后再下 Step 3 的多位点决策。

---

## Part 3: 综合诊断 + 策略选项

### 3.1 Phase C 结果对 MA 原决策矩阵的冲击

MA 原决策规则：
> ≥70% shell1 完全一致 → Option A（选第一位点）
> <70% → Option B 变体（位点随机采样）

实际数据**在分界点上**：按严格 signature 是 40%，按 MAE<0.1 Å 是 70%——规则本身对容差敏感。

### 3.2 四种可能的 Step 3 策略（列出来让 MA 拣选）

**选项 I：纯 Option B 变体（所有多位点样本都做位点采样）**
- 实现：Dataset.__getitem__ 里，对 `n_center_sites ≥ 2` 样本，每次随机选一个等价位点重新计算邻居 + shell boundaries
- 代价：中等（每 step 要重做 `get_neighbors`，但可以 cache 每样本所有位点的预计算结果）
- 覆盖：90% 样本
- 风险：如果 MP 是 site-specific 谱而某些位点其实是被 MP 特定选中的，随机采样会把本来对的位点打成"平均"

**选项 II：分层 Option B（按等价度 flag 选择）**
- 实现：对全 128K 样本预跑 Phase C 式诊断（~30 min on 8 workers）→ 产出 `site_equivalence_tag ∈ {equal, near_equal, distinct}`
  - `equal` (MAE<0.01): 走 Option A
  - `near_equal` (0.01 ≤ MAE < 0.1): 走 Option A + 质量标记
  - `distinct` (MAE ≥ 0.1 或 INCOMPAT): 走 Option B 变体
- 代价：中等，一次性多 30 min + Dataset 逻辑更复杂
- 覆盖：精细分层
- 价值：只对真正有问题的 10-20% 样本做采样，其余 80-90% 保持确定性

**选项 III：扩大 Phase C 样本量后再决策**
- 实现：把 Phase C 采样从 20 扩到 200（每 bucket 50）或 500（每 bucket 125）
- 代价：~1-2 min
- 价值：现在 20 样本的 40%/70%/75% 统计误差较大。扩到 200 样本能把决策置信度提升一个量级

**选项 IV：先查清 MP 协议再决策**
- MA 人工查 MP docs / 论文，确认 site-specific 还是 site-averaged，然后：
  - site-averaged → 几乎不受位点选择影响，选项 A 即可
  - site-specific → 必须选项 II 或更严格的位点匹配策略

### 3.3 Sub-Agent 的观察式建议（不替 MA 决策）

**我若被迫选一个**，倾向**选项 III + IV 并行**：
- 先花 2 min 跑扩大版 Phase C（200 样本）获取更稳定的等价度分布
- 同时 MA 手查 MP 协议
- 两条信息到齐后，MA 在 Option A / 分层 Option B / 纯 Option B 中做真正有依据的决策

**理由**：
1. 20 样本的统计误差在 70% 附近高达 ±10%，决策置信度不够
2. Phase C 已经在 7 秒跑完 20 样本，扩到 200 就是 70 秒，成本可忽略
3. 在没有 MP 协议信息的情况下，任何选择都可能系统性错误（比如 MP 是 site-averaged 但我们加了位点采样，反而把本来对齐的数据打散）

### 3.4 Phase B 与多位点策略无依赖

**重要**：`shell_boundaries.pkl`（Phase B 产物）**已经产出并固化**。它使用第一位点的邻居（与 Phase A neighbor_distances 完全一致），符合 handoff §5.2 的 LOCKED 策略。**Step 3 Dataset 的多位点策略决策不需要重跑 Phase B**——只影响 Dataset 里"拿到一个 sample_name 后怎么找 shell"的那一步，下游逻辑可以在 Step 3 里灵活实现。

---

## Part 4: 请 MA 决策

**决策点 1：Step 3 多位点策略**
- [a] 选项 I — 纯 Option B 变体
- [b] 选项 II — 分层 Option B（需要跑全量诊断 30 min）
- [c] 选项 III — 扩大 Phase C 样本至 200 再决策（Sub-Agent 推荐先走这条）
- [d] 选项 IV — MA 手查 MP 协议后决策
- [e] 选项 III + IV 并行（Sub-Agent 倾向）
- [f] 其他

**决策点 2（可选）：Phase B 的外层弥散样本标记**
对 5-10% eval_cutoff ≈ 10 Å 的样本，是否在 Step 5 评估时单独 tag "outer-diffuse"？
- [a] 是 — 在 Step 3/5 加一列 `eval_cutoff_fallback: bool`
- [b] 否 — 视同普通样本
- [c] 先跑实验看结果再决定

---

## Part 5: 交付物清单（Phase B + C 已全部产出）

Phase B：
- `shell_boundaries.pkl` (369.5 MB) — 128,382 条 shell 结构记录
- `shell_stats_by_split.csv` (4 行)
- `shell_stats_by_element.csv` (88 行)
- `step2_5b_summary.txt`
- `step2_5b_apply.log`

Phase C：
- `step2_5c_multisite_diagnostic.csv` (20 行详细)
- `step2_5c_summary.txt`（含 Q1/Q2 答案 + MP 查询指南）
- `step2_5c_diagnose.log`

Phase A（未动）：
- `step2_5_neighbor_distances.pkl` (196.1 MB)
- `step2_5_gap_histogram.png`
- `step2_5_gap_stats.csv` / `step2_5_gap_stats_by_element.csv`
- `step2_5_candidate_thresholds.csv`
- `step2_5a_summary.txt`

**Step 2.5 的核心交付物（Step 3 依赖）**：`shell_boundaries.pkl` ✓ 已就绪。

---

**Sub-Agent 待命。等 MA 决策：**
1. 是否立即进入 Step 3 交接文档（若 MA 选定多位点策略）
2. 还是先执行一条额外验证（扩大 Phase C 样本 / MP 协议手查 / 其他）

**Phase B 产物已 final**，Step 3 可随时开始使用 `shell_boundaries.pkl`。
