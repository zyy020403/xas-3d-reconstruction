# EXP4_PROGRESS_LOG.md
# Step 1 / Step 2 / Step 2.5 完整工作记录

> **撰写者**：Main Agent 2
> **日期**：2026-04-25
> **目的**：让 Main Agent 3 完整了解前情，避免重复决策或踩相同的坑

---

## Step 0：信息收集（Main Agent 1 完成）

**Action**：Main Agent 1 与用户讨论确认了 Exp4 完整方案，包括：数据集范围、文件格式、缺失值处理、异常剔除、特征标准化、不可变量。所有 Step 0 决策见 `EXP4_PROPOSAL_v2.md`。

**Result**：Main Agent 2 接管时，所有 Step 0 决策已锁定，但**部分预估值在执行后被实际数据修正**（详见各 Step）。

---

## Step 1：数据清洗与切分（Main Agent 2 + Sub-Agent，已完成）

### Action 1.1：写 Step 1 Sub-Agent 交接文档

- 路径：`STEP1_SUBAGENT_HANDOFF.md`
- 关键决策：
  - 文件名解析：`mp-{id}__mp-{id}-EXAFS-{Element}-K`
  - 三方 JOIN key = 完整 sample_name 字符串
  - IQR × 50 异常按**中心元素分组**判（不是全局 IQR）
  - 切分：mp_id 级 stratified split，比例 80/10/6/4
  - primary_element 定义：mp_id 所含元素中**全局 count 最小的非稀有元素**（不是最常见）
  - 含稀有元素的 mp_id 整 mp_id 进 train

### Action 1.2：用户运行 Step 1，反馈 3 条 deviation

| # | Deviation | Sub-Agent 处理 |
|---|-----------|--------------|
| 1 | holdout = 5,114 vs 区间 [3000, 5000] | 假警报：5,114/128,382 = 3.98%，区间是按 126K 估算紧了 |
| 2 | H_element = 479 vs [1300, 1800] | 计数归属问题：raw H 总数 2,209，其余 1,730 在前序过滤器（chi_invalid/missing_poscar/iqr_outlier）已剔 |
| 3 | iqr_outlier = 2,156 vs [3000, 10000] | 数据干净，pipeline 正确（按元素分组 IQR 比全局严）|

### Action 1.3：Main Agent 2 裁决

- 全部裁决 **A（接受，不重跑）**，理由：
  - p10 IQR 阈值锁定原则
  - grouped IQR 比全局更合理（不同元素 E0 跨度极大）
  - 数据真实清洁度 1.68% 是合理值

### Action 1.4：用户跑 raw H 核实命令

```python
inv = pd.read_pickle("step1_1_raw_inventory.pkl")
print((inv["center_element"]=="H").sum())  # 输出：2209
```

确认 H 总数 2,209，与 Sub-Agent 计数归属解释吻合。

### Action 1.5：missing_poscar_list.csv 失效（重要技术债）

Sub-Agent 发现：missing_poscar_list.csv 里 8,003 条中 7,751 条磁盘上实际存在 POSCAR。这个列表是上游流水线的旧日志，已失效。

**修正语义**：从"剔 mp_id ∈ missing_poscar_list" 改为"剔 not os.path.isfile(...)"。最终 missing_poscar = 790（精准命中预估 [600, 1000]）。

### Result 1：Step 1 产出

```
Total samples (raw):    133,718
parse_fail:                   0
chi_invalid:              1,911
missing_poscar:             790
H_element:                  479（计数归属，实际 H 全部 2,209 已排除）
poscar_invalid:               0
xmu_invalid:                  0
iqr_outlier:              2,156
TOTAL DROPPED:            5,336
KEPT:                   128,382  (96%)

Splits (mp_id-level):
  train:    102,660 samples / 33,147 mp_ids  (79.96%)
  val:       12,912 samples /  4,142 mp_ids  (10.06%)
  test:       7,696 samples /  2,485 mp_ids  ( 5.99%)
  holdout:    5,114 samples /  1,657 mp_ids  ( 3.98%)

Pairwise mp_id intersection: 0 / 0 / 0 / 0 / 0 / 0 ✓
Rare elements (Ar, He, Kr, Ne, 14 samples) all in train ✓
Center elements covered: 88

feff dimension: 74 (73 + has_pre_edge) ✓
has_pre_edge=1: 95,827 / 128,382 = 74.64%
RobustScaler fit on train (102,660 × 74) ✓
max|scaler.center_ - train_median| = 0 (perfect) ✓

Wall-clock: 53 min (6 scripts serial)
```

**关键产物**：data_inventory.csv (56.8 MB), feff_features_imputed.pkl (40.3 MB), feff_feature_scaler.pkl, train/val/test/holdout_ids.txt + train/val/test/holdout_samples.csv

---

## Step 2：谱预处理（Main Agent 2 + Sub-Agent，已完成）

### Action 2.1：写 Step 2 Sub-Agent 交接文档

- 路径：`STEP2_SUBAGENT_HANDOFF.md`
- 关键决策：
  - xmu 截 [E0-50, E0+150]，插值到 150 点
  - chi 用 chi1 列（不是 chi 或 chi2），截 k ∈ [0,12]，插值到 200 点
  - 保持原始物理尺度，不做 normalization（Step 3 encoder 第一层会处理）
  - 用 `np.interp` 线性插值 + 常数外推
  - 4 split 独立产 pkl（holdout 物理隔离）
  - 必做 5 元素视觉 QC（O/Fe/Cu/La/U）

### Action 2.2：用户运行 Step 2

**执行总览**：
- Wall-clock 6.3 min（比预估 25-40 min 快 4-6×，SSD + itertuples 红利）
- 处理样本数 128,382 / 128,382 ✓
- 失败 0
- 跨 split sample_name 冲突 0 ✓

**Per-split 大小**：

| split | N | pkl size |
|-------|---|----------|
| train | 102,660 | 148.4 MB |
| val | 12,912 | 18.7 MB |
| test | 7,696 | 11.1 MB |
| holdout | 5,114 | 7.4 MB |

**统计 sanity**：4 split 的 mean/std 相对差 ≤ 3.7%（远低于 20% 阈值），切分高度均衡。

**外推计数**：
- xmu_right_extrap: 0（FEFF 输出全部覆盖 E0+150） ✓
- chi_right_extrap: 0（FEFF 输出全部覆盖 k=12） ✓
- xmu_left_extrap: 115,670 (90.1%)（FEFF 输出从 E0 附近开始，[E0-50, E0] 是常数平台）

### Action 2.3：90% xmu_left_extrap 决策

Main Agent 2 决策：选 (a) 保持现状。理由：
- Exp2 用的是 [E0-50, E0+150]，Exp4 改窗口破坏可比性
- Conv1D 自动学到"前段低信息"
- 不是 critical issue，留待 Exp5 优化

### Action 2.4：用户视觉 QC 通过

`step2_qc_samples.png` 五个样本（O/Fe/Cu/La/U）红点（resampled）紧贴蓝线（native），E0 全部在该元素 K-edge 能量附近 <1% 偏差。

### Result 2：Step 2 产出

```
spectra_train.pkl     148.4 MB  (102,660 samples)
spectra_val.pkl        18.7 MB  ( 12,912 samples)
spectra_test.pkl       11.1 MB  (  7,696 samples)
spectra_holdout.pkl     7.4 MB  (  5,114 samples)
TOTAL                 185.6 MB

每个 pkl schema:
  sample_names: list[str]
  xmu:          (N, 150) float32
  chi1:         (N, 200) float32
  name_to_idx:  dict[str, int]
  E0:           (N,) float32
  meta:         {xmu_window_eV: [-50,150], chi_k_range: [0,12], chi_column: "chi1", ...}
```

---

## Step 2.5：物理约束 + 多位点处理（Main Agent 1 原 proposal 没有此步！）

> **重要**：Step 2.5 不在 Main Agent 1 原 proposal 中。是用户在 Step 2 之后提出"应该基于数据统计而不是经验值定壳层"，Main Agent 2 据此设计的新步骤。最终发现 40% 样本是多 Wyckoff 不等价中心位点的 incompat 样本，导致整个 Exp4 数据集需要剔除重整。

### 用户的原始动机（直接引用）

> "我认为不能只是根据经验值来告诉模型第一壳层的范围和第二壳层的范围等等，我们应该先根据我们的数据集做统计，知道真正合理的范围"

### Action 2.5.1：Main Agent 2 与用户讨论壳层算法

讨论结果（用户全部接受）：
- **细节 1**：算法改成"沿排序后距离序列，相邻间隙 > gap_threshold 处切分"（不是固定 +0.3）
- **细节 2**：gap_threshold 从训练集间隙分布定（不写死 0.3）
- **决策 2 选项 B**：壳层约束**只用在评估端**（Step 5 分层 RMSD + Step 3 Dataset 的 eval_cutoff），**不改训练 loss、不改采样**

### Action 2.5.2：写 Step 2.5 Sub-Agent 交接文档

- 路径：`STEP2_5_SUBAGENT_HANDOFF.md`
- 两阶段设计：Phase A（贵，邻居计算）+ Phase B（便宜，应用阈值）

### Action 2.5.3：Phase A 执行

```
处理样本数:    128,382 / 128,382 ✓
失败:               0
Wall-clock:    2.4 min (Pool(8) + group-by-mp_id)
```

**关键发现 ⚠**：n_center_sites >= 2 占比 = **89.86%**（预期触发阈值 20% 的 4.5 倍）！多位点是数据集主流。

**间隙分布**：高度右偏单峰（晶体对称等价原子贡献大量 gap=0），无 valley。

**候选阈值**：
| 候选 | 值 (Å) | shell1_n | shell1_outer | merged% | iso_single% |
|---|---|---|---|---|---|
| p10 | 0.1563 | 4.58 | 2.24 | 0.62% | 11.19% |
| p15 | 0.1047 | 4.07 | 2.21 | 0.05% | 15.63% |
| p20 | 0.0762 | 3.74 | 2.19 | 0.00% | 20.34% |
| empirical 0.3 | 0.30 | 9.85 | 2.41 | 6.97% | 6.78% |

### Action 2.5.4：Main Agent 2 决策阈值

选 **p10 = 0.1563 Å**：
- 唯一 shell1 各项物理指标全部在合理区间的候选
- merged% = 0.62%（次低）
- empirical 0.30 在 7% 样本上把 shell 1/2 合并掉，否决

### Action 2.5.5：Phase B 应用阈值

```
Threshold:     0.1563 Å (MA decision)
Built:         128,382 / 128,382 ✓
pickle size:   369.5 MB
Wall-clock:    5.7 s
```

**Phase B 副作用**：5-10% 样本 eval_cutoff fallback 到 ~10 Å（外层弥散，主要 O/Li/P/F 元素）。Main Agent 2 决策：Step 5 加 `eval_cutoff_fallback: bool` 标记。

### Action 2.5.6：Phase C 多位点采样诊断（20 样本）

| 判等口径 | 等价比例 |
|---|---|
| signature 严格相等 | 40% |
| MAE ≤ 0.05 Å | 65% |
| MAE ≤ 0.10 Å | 70% |（骑墙）
| MAE ≤ 0.20 Å | 75% |

**Sub-Agent 建议扩大到 200 样本再决策**。

### Action 2.5.7：Main Agent 2 拒绝扩 Phase C，改为查 MP 协议

理由：扩到 200 样本不会改变结构性结论。**关键问题是 MP 谱是 site-specific 还是 site-averaged**，这决定 INCOMPAT 样本的处理策略。

Main Agent 2 web_search 查到：
- MP wiki 原文："The computed absorption coefficient for an element in the given structure is set to the absorption coefficient averaged over all the sites in the structure with that element"
- Mathew 2018 Scientific Data 原文："This will facilitate comparison with experimental spectra, where the averaging over each element is unavoidable"

**结论**：MP EXAFS 数据是 **site-averaged over symmetrically unique sites**。

### Action 2.5.8：用户师兄确认

用户向师兄确认：是的，每个 (mp_id, element) 谱是该元素所有不等价 Wyckoff 位点的物理加权平均。

**Main Agent 2 必须坦白**：之前 4 个回合在错的方向上推进——基于"选第一位点"做了 Phase A/B/C/D 的工作。问题根源不是位点选哪个，而是**数据集本身的标签和谱在 40% 样本上没有 1:1 对应关系**。

### Action 2.5.9：Phase D 全量 site_equivalence_tag

第一次跑（Phase D v1）失败：pymatgen 2024.8.9 + numpy 1.26.4 在 Windows 的 `find_points_in_spheres` Cython 函数 buffer dtype mismatch（int64_t vs long），100% multi-site 调用失败。

第二次跑（Phase D v2）：用纯 numpy brute-force 邻居函数绕过 pymatgen Cython。Sanity check 5/5 max diff = 0.00000 Å。

**Phase D v2 结果（128,382 样本全跑完）**：

| Tag | 计数 | 占比 |
|---|---:|---:|
| single_site | 13,018 | 10.14% |
| equivalent | 53,877 | 41.97% |
| near_equivalent | 9,741 | 7.59% |
| **incompat** | **51,746** | **40.31%** |

**incompat 元素集中度**：
- O: 22,441 → 16,806 incompat (74.89%)
- F: 3,762 → 2,413 incompat (64.14%)
- Li: 7,147 → 3,401 incompat (47.59%)

**Tag × split 一致性**：4 个 split 的 incompat 比例 39.95%-40.96%（标准差 < 0.5%）。Step 1 的 mp_id-level split 干净，无 site_equivalence 偏移。

**near_equivalent 内部 999 个 multiset-mismatch**：原子数同但元素配比不同，本质上是化学环境不同，应归 incompat。Sub-Agent 建议重分类。

### Action 2.5.10：Main Agent 2 决策（多位点策略）

discuss 4 个选项后用户问"选项 234 都是什么我不懂"。Main Agent 2 用具体例子（Ca₂FeO₄ 的 4 个 O 位点）逐个解释：
- 选项 1（拒掉，选第一位点）：错位
- 选项 2（删 incompat）：丢 40% 数据但干净
- 选项 3（多目标 loss 平均）：DiffCSP 数学要改，物理可疑
- 选项 4（随机采样隐式平均）：在位点差异大时是噪声不是平均

用户问："128K 样本 → 76K 是谱，那对于 mpid 来说，减少了吗"

Main Agent 2 解释：mp_id 损失是次级问题，需 Sub-Agent 跑 Phase F 诊断。

### Action 2.5.11：Phase F 剔除诊断

```
保留样本:           75,637 / 128,382 = 58.92%（与 MA 预估完美吻合）
保留 mp_ids:         35,445 / 41,431 = 85.55%（损失 14.45%）
完全消失 mp_ids:     5,986
unique 元素覆盖:     88 / 88（零损失）
```

**Split 比例自动保持**（无需重做 split）：

| split | orig pct | keep pct | target |
|---|---|---|---|
| train | 79.96% | 80.00% | 80% |
| val | 10.06% | 10.08% | 10% |
| test | 5.99% | 5.92% | 6% |
| holdout | 3.98% | 4.00% | 4% |

**每 split 元素覆盖**：train=88, val/test/holdout=84（与原数据一致）。

**11 元素 < 200 样本**：7 个原本就少（Ne/Ar/He/Kr/Xe/Ac/Pm），4 个过滤后才掉到 200 以下（Pa/Np/Tc/Pu）。Step 5 报告时 caveat 这些元素。

### Action 2.5.12：Main Agent 2 最终决策 → Option D

选项 D：剔除 incompat。理由：
- 75K 对深度学习够（CIFAR-10 量级）
- 88 元素全保留 + 主流元素都 ≥ 1.2K
- 实现简单 = 更少 bug 面
- 评估干净 = Step 5 归因清晰
- 可逆性好（D 跑通后可加回 incompat 走 B；反过来难）

### Action 2.5.13：Phase G 产 v2 数据集

```
data_inventory_v2.csv     75,637 行 × 15 列（v1 14 列 + site_equivalence_tag）
train_samples_v2.csv      60,507 rows
val_samples_v2.csv         7,624 rows
test_samples_v2.csv        4,481 rows
holdout_samples_v2.csv     3,025 rows
incompat_pool.csv         52,745 rows (sealed)

assertions all PASS:
  ✓ len(data_inventory_v2) == 75637
  ✓ sum(4 v2 samples) == 75637
  ✓ len(incompat_pool) == 52745
  ✓ unique mp_ids in v2 == 35445
  ✓ 88 elements present
  ✓ 4 split mp_id zero intersection
```

### Result 2.5：Exp4 Final Dataset Name Card

```
═══════════════════════════════════════════════════════════════
                  Exp4 Final Dataset (Option D)
═══════════════════════════════════════════════════════════════
Total samples:      75,637
Total mp_ids:       35,445
Element coverage:   88

Splits:
  train      60,507  (80.00%)   28,297 mp_ids
  val         7,624  (10.08%)    3,580 mp_ids
  test        4,481  ( 5.92%)    2,139 mp_ids
  holdout     3,025  ( 4.00%)    1,429 mp_ids

Tag composition (kept):
  single_site         13,018  (17.21%)
  equivalent          53,877  (71.23%)
  near_equivalent      8,742  (11.56%)

Sealed (incompat_pool.csv):
  52,745 samples (40.31% of original 128,382)
  Reserved for Exp5 site-averaging strategy

vs Exp2:
  Exp2: 11,636 samples, Fe-oxide focus, 1 element
  Exp4: 75,637 samples, 35,445 mp_ids, 88 elements
        → 6.5× more samples, 88× more elements
═══════════════════════════════════════════════════════════════
```

**Total Step 2.5 wall-clock**：~25 min compute + 7 phases of decision iteration

---

## 已遗留给 Step 3+ 的事项

### 1. 服务器准备
- 用户已上传 POSCAR 到 `/home/tcat/mp-9_POSCAR`（路径需要 mv 到 `/home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/`）
- 其他 ~700 MB 数据待 Step 3 启动时 scp 上传（清单见 EXP4_FILE_INVENTORY.md）
- conda env `jhub_env` 包版本未确认（Step 3 第一步要 `pip freeze` 看）

### 2. Pymatgen Linux 兼容性
Phase D v1 在 Windows 失败，brute-force fallback 在 Phase D v2 用上了。Linux 服务器是否有相同问题**未知**，Step 3 Sub-Agent 必须做 sanity check（5 个 multi-site 样本 vs Phase A 的 distances 对比）。

### 3. Step 3 关键改动
- xas_local_dataset.py 重写（中心原子改 dynamic、文件读取改 csv-based、incompat 已剔除无需多位点分支）
- spectrum_encoder.py feff Linear 73 → 74（一行）
- diffusion_w_type_xas.py 路径常量更新

### 4. Step 5 必含的 caveat
final report 必须引述："本工作 Exp4 训练了 75,637 样本（剔除 52,745 'incompat' 样本，详见 Step 2.5 Phase D 报告）。incompat 样本结构上含多个不等价 Wyckoff 中心位点，与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"

---

## 关键工作教训（给 Main Agent 3 参考）

1. **物理理解 > 数据统计**：Step 2.5 折腾 7 个 phase 才发现是 site-averaged vs site-specific 的根本性不对齐。如果 Main Agent 1 在 Step 0 就和用户师兄确认 MP EXAFS 协议，能省 4 个回合。
2. **承认错误比硬推更省时间**：Main Agent 2 在 Phase D 后明确承认"我之前理解错了"，给用户 4 个选项让她拍板。这反而比"用 Option B 的精确版分层处理"省时间。
3. **数据驱动 > 经验值**：用户提议"基于数据定壳层阈值而不是用 0.3 经验值"是关键洞察。最终 p10 = 0.1563 Å，比 0.3 严格得多但物理更对。
4. **Sub-Agent 推荐 ≠ Main Agent 决策**：Sub-Agent 倾向 Option B（数据全），Main Agent 综合考虑后选 Option D（数据干净）。Sub-Agent 视角偏工程实现，Main Agent 视角偏整体科学价值。
5. **Pymatgen Cython bug 是真实风险**：不是绕过去就完事，要在文档里明确告诉 Step 3+ 这个隐患。

---

*Main Agent 2 撰写，2026-04-25*
