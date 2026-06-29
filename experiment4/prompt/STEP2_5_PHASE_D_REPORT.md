# STEP 2.5 PHASE D v2 完成报告

**报告对象**：DiffCSP-Exp4-Main-Agent 2
**Sub-Agent**：DiffCSP-Exp4-Step2.5-SubAgent
**阶段**：Phase D v2 — 全数据集多位点等价度打 tag
**状态**：**完成。请 MA 决策 Step 3 多位点策略**。
**Wall-clock**：628.7 s（compute 598.9 s ≈ 10 min）

---

## 1. 执行情况

| 指标 | 值 |
|---|---|
| 算法 | Pool(8) + group-by-mp_id + pure-numpy 周期镜像 brute-force（绕开坏掉的 pymatgen Cython） |
| Sanity check（启动时） | **5/5 max diff = 0.00000 Å** vs Phase A 的 saved distances ✓ |
| 处理样本数 | 128,382 / 128,382 |
| primitive_error | 0 |
| neighbor_error | 0 |
| phase_a_mismatch | 0 |
| Compute wall-clock | 598.9 s（≈ 10 min，与预估 1-3 min 偏差较大但仍可接受） |

**交付物**（在 `experiment4/step2_5/`）：
- `site_equivalence_tag.csv`（128,382 行：sample_name, mp_id, center_element, split, n_center_sites, tag, max_shell1_MAE, n_unique_shell1_multisets）
- `step2_5d_summary.txt`
- `step2_5d_tag.log`

---

## 2. ⭐ HEADLINE：Tag 分布与原估计严重偏离

| Tag | 计数 | 占比 | Phase C 20-样本估计 | 偏差 |
|---|---:|---:|---|---|
| single_site | 13,018 | **10.14%** | ~10% | ✓ |
| equivalent | 53,877 | **41.97%** | 40-60% | ✓ |
| near_equivalent | 9,741 | **7.59%** | 20-30% | **低** |
| **incompat** | **51,746** | **40.31%** | 10-20% | **⚠⚠ 超出 2 倍** |

**结论**：实际 incompat 比例（40.31%）是我之前 §3.1 给 MA 估计的 10-20% 的 2 倍以上。Phase C 的 20 样本太小不足以代表全集，N=128K 才是真相。

**这条数据决定了 Step 3 多位点策略的难度**：原以为只需要处理 ~10-20% 边缘样本，现在是处理 ~40% 主流情况。

---

## 3. Tag × split 跨 split 分布（关键诊断：是否有 split leakage）

| split | single_site | equivalent | near_equivalent | incompat |
|---|---:|---:|---:|---:|
| train | 10.13% | 42.00% | 7.58% | **40.29%** |
| val | 10.39% | 42.00% | 7.40% | **40.21%** |
| test | 10.43% | 41.16% | 7.45% | **40.96%** |
| holdout | 9.33% | 42.37% | 8.35% | **39.95%** |

**4 个 split 的 incompat 比例几乎完全一致**（39.95% - 40.96%，标准差 < 0.5%）。

✓ **确认 mp_id-level split 无 site_equivalence 偏移** — Step 1 的 split 决策依然干净，不需要重新 split。
✓ Step 5 的 test/holdout 评估时不会因为 incompat 分布异常而偏差。

---

## 4. Incompat 元素集中度（Step 3 策略关键）

| element | n_total | n_incompat | incompat_pct |
|---|---:|---:|---:|
| **O** | 22,441 | **16,806** | **74.89%** |
| **F** | 3,762 | 2,413 | **64.14%** |
| Li | 7,147 | 3,401 | 47.59% |
| Cl | 1,818 | 860 | 47.30% |
| Se | 1,589 | 719 | 45.25% |
| S | 3,258 | 1,437 | 44.11% |
| Na | 2,367 | 1,007 | 42.54% |
| Fe | 3,481 | 1,410 | 40.51% |
| Mn | 3,313 | 1,341 | 40.48% |
| K | 1,985 | 796 | 40.10% |
| V | 2,913 | 1,065 | 36.56% |
| P | 5,495 | 1,960 | 35.67% |
| B | 2,097 | 745 | 35.53% |
| N | 2,196 | 715 | 32.56% |
| Ni | 2,478 | 768 | 30.99% |

**关键观察**：

1. **O 主导（incompat_pct = 74.89%）**：22,441 个 O 样本里有 16,806 个 incompat。光 O 一个元素就占 incompat 总量 32.5%（16,806 / 51,746）。物理上合理 — 氧化物里 O 经常处在多个不等价 Wyckoff 位置（terminal O / bridging O / 不同配位环境），site-averaging 是 site-specific 谱的真实加权平均。

2. **F、Cl、Se、S、Li、Na、K 等也高**（30-65%）：离子型晶格里这些原子也常常多 Wyckoff。

3. **过渡金属相对低**（Fe/Mn/Ni 30-40%，Cu/Co 应该更低）：TM 通常每相 1 个 Wyckoff 位置。

**对 Step 3 的影响**：如果选 Option A（维持 first-site），74.89% 的 O 谱将被打上"不准确" label。考虑到 O 是数据集里数量最多的中心元素（22,441 占 17.5%），这是大问题。

---

## 5. n_center_sites 与 incompat 的关联（incompat 分布桶）

| n_sites bucket | incompat 数 | 占 incompat 总量% |
|---|---:|---:|
| 2 | 1,076 | 2.08% |
| 3 | 2,029 | 3.92% |
| 4 | 4,890 | **9.45%** |
| 6 | 4,745 | 9.17% |
| 8 | 7,177 | **13.87%** |
| 12 | 5,101 | 9.86% |
| 16 | 4,320 | 8.35% |
| 24 | 2,966 | 5.73% |
| 其他 | ~10K | ~37% |

**incompat 在中等位点数（4-16）最集中**。n_sites=8 桶贡献了 incompat 总量的 13.87%。这与 Phase C 观察一致：n_sites 越大 → 越可能存在多 Wyckoff 位置组合。

但 **n_sites=2 也有 1,076 个 incompat（2 个位点都不等价）** — 这种是"明明只有 2 个位点还各不相同"的小晶格高对称性破缺情形。

---

## 6. Sub-statistic: near_equivalent 内部的隐藏不一致

按 MA 字面规则，"near_equivalent" 只要求 *count* 相等 + MAE < 0.1，**不要求 multiset 相等**。意思是"同样数量的原子，但元素配比可能不同也能算 near_equivalent"。

实际触发情况：

> **999 out of 9,741 near_equivalent (10.26%) 有 multiset mismatch**
> （即 shell1 size 一致，但元素组合不同，例如某位点 shell1 是 [O,O,O,F]，另一位点是 [O,O,F,F]）

**MA 决策点**：这 999 个样本物理上不是"小数值漂移"，而是化学环境不同——理论上应该归 incompat 而非 near_equivalent。

**给 MA 的两个选项**：
- **a**. 把这 999 个重分类到 incompat → incompat 总数变 52,745 (40.79%)，near_equivalent 变 8,742 (6.81%)
- **b**. 维持 MA 原字面规则，999 留在 near_equivalent

我倾向 **a**（重分类），因为 multiset 不等意味着真实化学环境不同，Step 3 处理时应该和其他 incompat 一视同仁。但这是 MA 字面规则的修订，需要 MA 明示。

---

## 7. Step 3 多位点策略 — 现在数据清楚了

之前我列了 4 个选项，现在用 40% incompat 重新评估：

### Option A. 全数据维持 first-site
**代价**：40% 数据 label-spectrum 系统性偏差（不是噪声而是系统偏差）。O 元素 75% 受影响。
**优点**：实现零成本，Dataset 不需改动。
**风险**：模型学到的是"first-site 局部环境 → 谱"，而真实关系是"site-averaged 局部环境 → 谱"。在 holdout 上的泛化能力存疑，特别是 O-K edge。

### Option B. 分层策略（Sub-Agent 推荐）
**实现**：Step 3 Dataset 的 `__getitem__` 按 site_equivalence_tag 分支：
- `single_site` (10.14%)：first-site = only-site，确定性 ✓
- `equivalent` (41.97%)：所有位点完全等价，first-site = 任意位点，确定性 ✓
- `near_equivalent` (7.59% 或 6.81%)：first-site，小数值漂移作训练噪声忍受
- `incompat` (40.31% 或 40.79%)：每次 `__getitem__` **随机从 n_center_sites 个等价类位点选一个**，重新算 shell boundaries 当 label

**代价**：
- 实现：Step 3 Dataset 类需要 site_equivalence_tag 字段 + 对 incompat 走随机分支
- 训练时间：incompat 样本 `__getitem__` 要做 `find_neighbors_brute` + 切 shell（非 vectorized 的话每次 ~5-10 ms）
- 缓存优化：可以预算所有位点的 shell boundaries 存盘，`__getitem__` 只是 dict lookup。需要 ~3 GB（128K × 平均 8 sites × 每 site 几十字节）

**优点**：
- 物理上对齐——多 epoch 平均下，模型隐式学到"对所有 Wyckoff 位置加权平均"的映射，呼应 site-averaged 谱
- single_site / equivalent 占 52% 数据完全确定性，训练稳定
- 只对 ~40% incompat 引入随机性，但这随机性物理上是正确的

### Option C. 显式平均 label
**实现**：预计算所有 Wyckoff 位点的 shell boundaries，按 multiplicity 加权平均出"平均 shell"。
**代价**：复杂——平均 shell starts/ends 没有清晰定义（怎么平均一个长度变化的数组？）。
**结论**：实现难度过高，不推荐。

### Option D. 直接丢弃 incompat
**代价**：损失 40.31% 数据，从 128K 降到 76K。
**优点**：剩余样本严格对齐。
**结论**：损失太多 + Exp4 全元素覆盖目标受损（O 数据砍 75%），不推荐。

---

## 8. Sub-Agent 推荐

**Option B 分层策略**，配合 §6 的 a 决策（999 multiset-mismatch near_equivalent 重分类到 incompat）。

理由：
1. 物理上正确：site-averaged 谱 ↔ 随机位点采样 = 训练时隐式平均
2. 大头数据（52%）保持确定性，训练稳定
3. 只对真正有问题的 ~40% 样本引入随机性
4. 实现可行：Dataset 加分支即可，预算 shell boundaries 缓存可以加速

**实现细节备忘（供 Step 3 用）**：
- 预计算文件 `shell_boundaries_all_sites.pkl`：每个 multi-site sample 存所有 n_center_sites 个位点各自的 shell boundaries（不是只存第一个）
- Dataset `__getitem__` 对 incompat 样本 `np.random.choice(n_center_sites)` 选位点，从字典取对应 shell
- Train mode 用随机；val/test/holdout mode 用确定性 first-site（评估稳定性）

这条预计算大约会增加 `shell_boundaries.pkl` 体积 5-8×（从 369 MB 涨到 ~2 GB），但 IO 一次永久受用。

---

## 9. 请 MA 决策

**决策点 1**：`near_equivalent` 内部 999 个 multiset-mismatch 是否重分类到 incompat？
- [a] 是，重分类（Sub-Agent 推荐）
- [b] 否，维持原字面规则
- [c] 其他

**决策点 2**：Step 3 多位点策略
- [a] Option A — 维持 first-site，40% 数据带系统偏差
- [b] **Option B — 分层策略（Sub-Agent 推荐）**
- [c] Option C — 显式平均 label
- [d] Option D — 丢弃 incompat
- [e] 其他

**决策点 3**：是否预计算 `shell_boundaries_all_sites.pkl`？
- [a] 是 — Sub-Agent 立即在 Step 2.5 阶段产出（Phase E 类似 Phase D 的另一次扫描，~10 min）
- [b] 否 — Step 3 的 Dataset 在 `__getitem__` 即时算（每个 step 多 5-10 ms）
- [c] 其他

---

## 10. 给 Sub-Agent 的下一步动作（取决于 MA 决策）

- 若 MA 选 Option B + 决策 3=a → Sub-Agent 立刻写 `step2_5e_compute_all_site_shells.py`，Step 2.5 还需 ~10-15 min 收尾
- 若 MA 选 Option B + 决策 3=b → Sub-Agent 直接进入 **Step 3 交接文档** 撰写，Dataset 实现里包含 brute-force 邻居函数
- 若 MA 选 Option A → Sub-Agent 直接进入 Step 3 交接文档（最简单路径）
- 若 MA 选 Option D → Sub-Agent 加一个数据过滤脚本砍掉 incompat，再写 Step 3
- 若 MA 选 Option C → Sub-Agent 需要先和 MA 讨论"平均 shell"的精确定义

---

**Sub-Agent 报告结束。等 MA 三个决策点的回复后即刻执行下一步。**
