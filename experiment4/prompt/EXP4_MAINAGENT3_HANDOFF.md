# EXP4_MAINAGENT3_HANDOFF.md
# DiffCSP-Experiment4 Main Agent 3 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 2
> **接收者**：DiffCSP-Exp4-Main-Agent 3
> **日期**：2026-04-25
> **背景**：Main Agent 2 的对话上下文窗口耗尽，由 Main Agent 3 接管。

---

## 1. 你是谁，你要做什么

你是 Experiment 4 的 **Main Agent 3**。前任（Main Agent 2）已经完成了 **Step 1 / Step 2 / Step 2.5** 三个阶段，所有产出文件都在用户本地 Windows 机器的 `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\` 下。

**你的工作从 Step 3 Sub-Agent 交接文档开始**：

1. 写 **Step 3 Sub-Agent 交接文档**（服务器端，最大改动）—— Dataset + SpectrumEncoder + 前向测试
2. 等用户跑完 Step 3 汇报，确认前向 loss 正常后，写 **Step 4 交接文档**（训练）
3. 等用户跑完 Step 4 汇报，确认 val_loss 收敛后，写 **Step 5 交接文档**（评估 + holdout）
4. 用户跑的中间结果你都要核对，发现问题及时修正

**你不写代码**，代码由各 Sub-Agent 实现。**Step 0 / Step 1 / Step 2 / Step 2.5 的所有决策都已锁死**，不要重新和用户讨论（下面有完整决策清单）。

---

## 2. 必读文档（按顺序）

我打包了 4 份文档给你（用户会一起发给你）：

| 文档 | 必读？ | 作用 |
|------|-------|------|
| **本文档（EXP4_MAINAGENT3_HANDOFF.md）** | ✅ 必读 | 你的工作全景图 |
| **EXP4_PROPOSAL_v2.md** | ✅ 必读 | 更新后的 Exp4 方案，覆盖原 EXP4_MAINAGENT_HANDOFF.md |
| **EXP4_PROGRESS_LOG.md** | ✅ 必读 | Step 1/2/2.5 完整 action+result 记录 |
| **EXP4_FILE_INVENTORY.md** | ✅ 必读 | 所有产出文件位置和 schema |

历史归档（用户**可选**发给你做 deep context，但不是必读）：
- STEP1_SUBAGENT_HANDOFF.md + STEP1_COMPLETION_REPORT.md
- STEP2_SUBAGENT_HANDOFF.md + Step 2 完成报告
- STEP2_5_SUBAGENT_HANDOFF.md + Phase A/BC/D/F/Final 完成报告
- 共享文档 SHARED_00_v2.md / SHARED_01_DATA_MANIFEST.md（Exp2 的，仅参考设计理念）
- EXPERIMENT2_FINAL_REPORT.md（Exp2 历史结果）

---

## 3. 用户的执行环境

- **本地**（Windows）：Step 1、Step 2、Step 2.5 已全部完成
- **服务器**（Step 3 / 4 / 5 在这里跑）：
  - Host: `scsmlnprd02.its.auckland.ac.nz`，SSH 用户 `tcat`
  - OS: Ubuntu 22.04.4 LTS
  - Python 环境: `jhub_env`（登录时默认激活）
  - GPU: 2 × RTX 4090（24 GB each），CUDA 12.2
  - **存储应急方案**：根盘 99% 满只剩 30 GB；管理员说要等到 Exp5 才有独立盘位
    - 持久数据和代码放 `/home/tcat/diffcsp_exp4/`
    - 训练启动前 `cp -r .../data /tmp/diffcsp_cache/`（`/tmp` 是 tmpfs，256 GB RAM）
    - checkpoints 只留 best + last，其余立即清理
  - 相对 Exp2（RTX A4000 16GB），训练时间应显著缩短

**用户已经完成的事**：POSCAR 目录已 scp 上传到 `/home/tcat/mp-9_POSCAR`（注意路径，不在 `diffcsp_exp4/data/`）。其他文件用户**还没上传**，等 Step 2.5 完成后一起传，避免反复覆盖。

---

## 4. 关键决策速记（🔒 LOCKED，绝对不要重新讨论）

### 4.1 数据集（Step 0 锁定，Step 2.5 修正）

- **数据集**：MP 全元素 EXAFS，~132K 原始样本
- **每个 (mp_id, 中心元素) 是一个独立样本**
- **🆕 Step 2.5 修正**：剔除 incompat 样本（多 Wyckoff 不等价中心位点）后，**最终训练数据集 = 75,637 样本 / 35,445 mp_ids / 88 元素**
- **52,745 incompat 样本封存到 `incompat_pool.csv`**，Exp4 全程不动，留给 Exp5

### 4.2 切分（Step 1 完成，Step 2.5 自动保持比例）

- mp_id 级 stratified split，比例 80/10/6/4
- v2 切分（剔除 incompat 后）：
  - train: 60,507 samples / 28,297 mp_ids
  - val: 7,624 samples / 3,580 mp_ids
  - test: 4,481 samples / 2,139 mp_ids
  - holdout: 3,025 samples / 1,429 mp_ids
- 4 个 split mp_id 零交集（已 assert）
- **Step 3 训练用 `*_samples_v2.csv` 文件**，**不要**用 v1 的 `*_ids.txt`

### 4.3 文件格式（已确认）

- **chi.csv**：表头 `k,chi,chi1,chi2`，逗号分隔，401 行
  - **模型输入用 `chi1` 列**（k¹χ(k) 加权），**不要**用 `chi` 或 `chi2`
  - 已在 Step 2 处理为 (200,) np.float32 张量，k ∈ [0, 12] Å⁻¹
- **xmu.csv**：表头 `x,y`，401 行
  - 已在 Step 2 处理为 (150,) np.float32 张量，窗口 [E0-50, E0+150] eV
  - E0 从 feff_features 的 E0 列读
- **POSCAR**：标准 VASP，`Structure.from_file()` 直接读
  - **保留 `get_primitive_standard_structure(symprec=0.1)`** 防御
- **feff_features**：73 原始 + 1 has_pre_edge = **74 维**
  - Step 1 已 RobustScaler fit on train，存 `feff_feature_scaler.pkl`

### 4.4 不可变量（继承 Exp2，绝对不能改）

- L = 6 Å（虚拟晶格边长）
- 坐标系 [-0.5, 0.5]，`frac -= np.round(frac)` min-image 折叠
- forward() 无 `% 1.`
- N_NEIGHBORS = 20
- batch_size = 16，lr = 1e-4，bf16，num_workers = 0（Windows 限制；服务器先保持一致，必要时再调）
- 三路 SpectrumEncoder（xmu 150 + chi1 200 + feff 74 → latent 256）
- DiffCSP 扩散框架，cost_lattice = 0
- **不加 TypeClassifier**（Exp3 已证伪）

### 4.5 物理约束（Step 2.5 新增）

- **gap_threshold = 0.1563 Å**（基于训练集 train-only 的相邻距离间隙 p10）
- **算法**：相邻距离间隙 > threshold 处切分壳层，整组紧密原子算一个壳层
- **eval_cutoff** = "包含第 20 个邻居的那个完整壳层的外缘"（替代 Exp2 的 `min(d20, 4.0)`）
- **5-10% 样本** eval_cutoff fallback 到 ~10 Å（外层弥散），Step 5 评估时加 `eval_cutoff_fallback: bool` 标记
- **shell_boundaries.pkl** 已产出（369 MB），覆盖全 128K 样本（包括 incompat），Step 3/5 用 `data_inventory_v2.csv` 过滤后访问

### 4.6 多位点策略（Step 2.5 最终决定 → Option D）

- 经过 Phase D 全量诊断：incompat = 40.31%（O 高达 75%）
- 经过 Phase F 剔除诊断：剔除后 88 元素全保留 + split 比例自动保持
- **MA2 决策**：选 Option D，**剔除 incompat 样本，保留的 75,637 样本均直接用第一位点**
- Step 3 Dataset **不需要任何多位点分支逻辑**（已被 Option D 简化）
- 物理依据：MP EXAFS 是 site-averaged over symmetrically unique sites（已查证，Mathew 2018 + MP wiki）

### 4.7 异常剔除（Step 1 完成）

- H 中心元素（479 个走到 H_element 过滤器，但 raw 总数 2,209；其余 ~1,730 在 chi_invalid/missing_poscar/iqr_outlier 阶段被剔）
- IQR × 50 极端异常（按中心元素分组 grouped IQR）：2,156 个
- chi_invalid 1,911 + missing_poscar 790 + parse_fail 0 + xmu_invalid 0 + poscar_invalid 0
- **总剔除 5,336**（5,336 = 128,382 - 122,046 ... 等等，实际数字：raw 133,718 → final 128,382 = 剔除 5,336）

### 4.8 缺失值处理（Step 1 完成）

- 强度/面积/比值类列填 0
- 能量/位置类列按中心元素分组取中位数
- 新增 `has_pre_edge ∈ {0, 1}` 标志位（标记原始 pre_peak_I 是否为 NaN）
- 最终 feff 维度 73 + 1 = 74

### 4.9 Pymatgen Cython bug（重要技术债）

Phase D v1 在 Windows 上发现 pymatgen 2024.8.9 + numpy 1.26.4 的 `find_points_in_spheres` 有 buffer dtype 不匹配 bug（期望 int64_t 拿到 long），**100% multi-site 调用失败**。Phase D v2 用纯 numpy brute-force 邻居函数绕过。

**对 Step 3 的影响**：服务器（Linux）上 pymatgen 大概率没问题，但 Sub-Agent 必须做一次 sanity check（5 个 multi-site 样本 vs Phase A 的 distances 对比）。如果 Linux 也坏，Step 2.5 的 brute-force 函数已存在 `step2_5d_full_multisite_tag_v2.py` 中，可作为 fallback 抽出来用。

---

## 5. 你的工作流程

### 5.1 你的第一条回复（建议格式）

```
我已阅读完 Main Agent 2 交接的所有文档（EXP4_MAINAGENT3_HANDOFF + 
EXP4_PROPOSAL_v2 + EXP4_PROGRESS_LOG + EXP4_FILE_INVENTORY）。

[简要复述：Step 1/2/2.5 的核心结论 + 当前数据集状态 + Step 3 待处理事项]

在写 Step 3 交接文档之前，我需要向你确认几件事：

1. SSH 配置：你已有 ssh key 还是密码登录？scp 脚本要不要交互密码？
2. 服务器 conda env `jhub_env` 里 pymatgen / torch / numpy 当前版本？
   （建议跑 `conda activate jhub_env && pip freeze > /tmp/jhub_env_packages.txt` 
   并把这个文件发给我，确认兼容性）
3. Exp2 代码仓库当前在你本地哪里？需要我让 Step 3 Sub-Agent 上传哪些 Exp2 文件？
4. 服务器上 /home/tcat/ 还剩多少空间？是否要先清理 /home/tcat/mp-9_POSCAR
   （这个路径看起来是测试性的，应该 scp 到 /home/tcat/diffcsp_exp4/data/）

确认后我开始写 Step 3 Sub-Agent 交接文档。
```

### 5.2 Step 3 交接文档应该包含的章节

Step 3 是整条 pipeline 风险最高的一步。我建议的章节结构：

1. **Step 3 总览** —— 这一步的目标 + 不做什么
2. **服务器准备** —— SSH、conda env 检查、目录创建、数据上传清单
3. **Exp2 代码仓库审计** —— 哪些文件可以原样用、哪些要改、改在哪行
4. **Dataset 改造（最大改动）**：
   - 文件读取：从 .dat 改成 .csv（chi 用 chi1 列，xmu 用 y 列）—— 但 Step 2 已经预处理成 pkl，Step 3 直接 load `spectra_*.pkl`
   - 中心原子定位：从硬编码 Fe 改成读 `data_inventory_v2.center_element`
   - 主索引文件改成 `data_inventory_v2.csv` + `*_samples_v2.csv`
   - 多位点：直接用第一位点（incompat 已被 Option D 剔除，剩下样本第一位点 = site-averaged）
   - 邻居计算：服务器先尝试 pymatgen，失败 fallback brute-force（utility module）
   - feff RobustScaler transform 在 Dataset `__getitem__` 里做（因为 Step 1 只 fit 没 transform）
   - shell_boundaries.pkl 也在 Dataset 里 load，每个 sample 取它的 shell 信息（Step 4 训练用不到，Step 5 用）
5. **SpectrumEncoder 改动** —— feff 分支 `nn.Linear(73, ...)` → `nn.Linear(74, ...)`，**仅此一行**
6. **diffusion_w_type_xas.py** —— 路径常量更新，cost_lattice=0 保持，框架本身不变
7. **前向测试协议** —— 服务器上跑 batch_size=4 的小批量，手算预期 loss 范围、检查 sample 维度对齐
8. **训练前 checklist** —— 最后一道关
9. **Step 3 Sub-Agent 完成后的汇报模板**

### 5.3 推进节奏（建议）

| Step | 用户在哪里跑 | 你的任务 |
|------|------------|---------|
| Step 3 | 服务器 | 写 Sub-Agent 交接文档 → 用户跑 → 你审 forward loss → OK 就进 Step 4 |
| Step 4 | 服务器 | 写训练交接 → 用户跑 → 你审 val_loss 曲线 → 收敛了进 Step 5 |
| Step 5 | 服务器 | 写评估交接 → 用户跑 → 你审最终 RMSD/Type Acc/pred_in_cutoff |

每一步用户汇报后**都要检查后再推进**，不跳步。

---

## 6. 你不要做的事（继承自 Main Agent 1/2 工作原则）

- ❌ 不写代码，只出交接文档
- ❌ 不重新讨论 Step 0 / Step 1 / Step 2 / Step 2.5 的任何决策（已锁死）
- ❌ 不要让 Sub-Agent 触碰 incompat_pool.csv 里的样本（Exp4 全程封存）
- ❌ 不要让 Sub-Agent 用 v1 的 `*_ids.txt`（用 v2 的 `*_samples_v2.csv`）
- ❌ 不要在训练前接触 holdout（Step 4 训练只用 train + val）
- ❌ 不要让 Sub-Agent 改虚拟晶格 L=6、坐标系 [-0.5, 0.5]、N_NEIGHBORS=20 等不可变量
- ❌ 不要让 Sub-Agent 加 TypeClassifier（Exp3 已证伪）

---

## 7. Exp4 的预期指标（Step 5 评估时核对）

| 指标 | Exp2 Holdout | Exp4 目标 |
|------|-------------|-----------|
| RMSD | 1.47 Å | ≤ 1.8 Å |
| Type Accuracy | 0.241 | 0.15-0.25（类别 ~20 → 88，绝对值降是预期） |
| pred_in_cutoff | 17.5/20 | ≥ 15/20 |

**Exp4 主要目标**：验证 Exp2 架构在全元素数据集上是否仍能成立，为 Exp5（多视角 attention 聚合）打 baseline。**不是刷 RMSD**。

Step 5 final report 必须引述："本工作 Exp4 训练了 75,637 样本（剔除 52,745 'incompat' 样本，详见 Step 2.5 Phase D 报告）。incompat 样本结构上含多个不等价 Wyckoff 中心位点，与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"

---

## 8. Sub-Agent 交接文档的格式要求（继承）

每个 Sub-Agent 交接文档必须包含：

1. **背景**：这一步要做什么、与 Exp2 的区别
2. **需要用户提供的文件清单**：精确到文件名和路径
3. **改动内容**：精确描述每个文件改哪里、改成什么（不写代码，写意图）
4. **服务器命令**：所有 `cd / conda activate / python` 完整命令
5. **验证方法**：改完后如何确认改对了
6. **输出文件清单**：改好的文件存哪里
7. **汇报模板**：完成后按这个格式汇报

---

## 9. 关键提醒

### 9.1 服务器存储紧张

`/home/tcat/` 只剩 30 GB，要严格管理：

- 持久数据：`/home/tcat/diffcsp_exp4/data/`（v2 训练数据 + spectra + shell_boundaries 共 ~700 MB）
- 代码：`/home/tcat/diffcsp_exp4/code/`（Exp2 仓库 fork + Exp4 改动）
- 训练前 cache：`cp -r .../data /tmp/diffcsp_cache/`（tmpfs，256 GB RAM）
- checkpoints：只留 best + last，其余立即清理

### 9.2 Pymatgen 版本兼容性

服务器上 jhub_env 的 pymatgen 版本可能和 Step 2.5 的本地版本不同。**Step 3 Sub-Agent 第一件事**应该是 sanity check：
```python
from pymatgen.core import Structure
s = Structure.from_file("/home/tcat/.../mp-12345_POSCAR")
neighbors = s.get_neighbors(s[0], r=10.0)
print(len(neighbors), neighbors[0].nn_distance)  # 看是否报错
```
如果报错 → 用 brute-force fallback（从 step2_5d_full_multisite_tag_v2.py 抽函数）

### 9.3 数据 key 对齐 sanity check

`spectra_*.pkl`、`feff_features_imputed.pkl`、`shell_boundaries.pkl`、`data_inventory_v2.csv` 都用 `sample_name` 作 key。**Step 3 Sub-Agent 写 Dataset 后第一个 forward pass 之前必须验证**：随机取 100 个 v2 sample，确认 4 个数据源都能 lookup 到。

### 9.4 v1 vs v2 不要混用

| v1（保留作历史归档） | v2（Step 3+ 必用） |
|-------------------|------------------|
| `data_inventory.csv` 128,382 行 | `data_inventory_v2.csv` 75,637 行 |
| `train_ids.txt` mp_id 级 | `train_samples_v2.csv` sample 级 |
| `train_samples.csv` sample 级（已含 incompat） | `train_samples_v2.csv` sample 级（已剔 incompat） |

Step 3 Dataset 入口用 v2，**不要**用 v1。

### 9.5 incompat_pool.csv 的处理

52,745 个 incompat 样本封存。Exp4 全程：
- Step 3 Dataset 不 load
- Step 4 训练不用
- Step 5 评估不报告
- 但 Step 5 final report 要引述（参见 §7）

这些样本留给 Exp5。Sub-Agent 会试图"为完整性也跑一下"，**你要明确禁止**。

---

## 10. 给你的最后一条提醒

**Step 2.5 比 Main Agent 1 原 proposal 多花了 3-4 个回合**（A→B→C→D v1 失败→D v2 修复→F 诊断→G 剔除）。这不是浪费——是因为发现了关键的物理问题（site-averaged 谱 vs site-specific 标签不对齐）。**最终方案 Option D 是基于这个发现的正确响应**。

如果你接下来在 Step 3-5 中又发现"原假设错了"的情况，**不要硬推**。停下来和用户讨论，参考 Main Agent 2 在 Phase D 之后的处理方式：

1. 先承认错误（"我之前理解错了 X"）
2. 解释新理解（"实际情况是 Y"）
3. 给出几个选项（A/B/C/D），列各自代价
4. 不替用户做决定，让用户拍板
5. 如果用户拍板了某个选项但你后来发现还是有问题，**再次停下来**，不要为了"推进"而继续

用户对你信任，你要回报这份信任的方式是**诚实**，不是"流畅"。

---

*Main Agent 2 撰写，2026-04-25，最后一次发言*
