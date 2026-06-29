# STEP 2.5 完整收工报告

**报告对象**：DiffCSP-Exp4-Main-Agent 2
**Sub-Agent**：DiffCSP-Exp4-Step2.5-SubAgent
**状态**：**Step 2.5 全部交付物 final，6/6 验收 assertions 通过**
**总耗时**：~25 min compute + 多轮 MA 决策迭代

---

## 1. 执行总览

Step 2.5 设计目标是产出"按 gap 切壳层 + 给样本打 site_equivalence_tag"的两件套。最终走完了 **A → B → C → D(v1 失败) → D(v2 修复) → F(诊断) → G(剔除产 v2 数据集)** 七个 phase，全部成功。

| Phase | 任务 | 耗时 | 关键产物 | 状态 |
|---|---|---:|---|---|
| A | 全 128K 样本邻居距离（r=10 Å） | 2.4 min | `step2_5_neighbor_distances.pkl` (196 MB) | ✓ |
| B | 应用 gap 阈值 0.1563 Å 切壳层 | 5.7 s | `shell_boundaries.pkl` (369 MB) | ✓ |
| C | 20 样本多位点等价度抽样诊断 | 7 s | `step2_5c_multisite_diagnostic.csv` | ✓ |
| **MA 决策** | 查 MP 协议确认 site-averaged | — | 锁定 Phase D 范围 | ✓ |
| D v1 | 全量 128K site_equivalence tagging | 2 min | **失败**：pymatgen Cython buffer 全部炸 | ✗ |
| D v2 | 用纯 numpy brute-force 重写 | 10 min | `site_equivalence_tag.csv` | ✓ |
| F | 诊断 Option D 剔除影响 | 5 s | `step2_5f_filter_diagnosis.txt` | ✓ |
| **MA 决策** | 选 Option D（剔除 incompat） | — | 锁定 v2 数据集口径 | ✓ |
| G | 应用过滤产 Exp4 v2 数据集 | 30 s | 6 个 v2 文件 + name card | ✓ |

---

## 2. 关键决策路径（4 个 MA 拍板时刻）

### 决策 1：Threshold 选择 → **p10 = 0.1563 Å**
- Phase A 跑出全局间隙分布是高度右偏单峰（valley 不存在）
- 5 个候选模拟显示 p10 是唯一一个 shell-1 各项物理指标全部在合理区间的选择（mean shell1 atoms = 4.58, shell1 outer = 2.24 Å, merged% = 0.62%）
- empirical 0.30 Å 在 7% 样本上把 shell 1/2 合并掉了，否决

### 决策 2：MP 协议确认 → **site-averaged spectra**
- Phase C 发现 89.86% 样本是 multi-site，触发 ⚠ FLAG
- MA 查 MP wiki + Mathew 2018 论文确认：MP EXAFS 是按对称唯一位点做加权平均的谱
- 这条事实锁定了"必须给所有 128K 样本打 site_equivalence_tag"的工作必要性

### 决策 3：999 个 multiset-mismatch near_equivalent → **重分类到 incompat**
- Phase D v2 发现：MA 字面规则下，"near_equivalent" 不要求 multiset 相等，导致 999 个样本"原子数同但元素配比不同"也归 near_eq
- 这 999 个物理上是化学环境不同，应该 incompat
- MA 同意重分类

### 决策 4：多位点策略 → **Option D（剔除 incompat）**
- Phase D v2 揭示 incompat 实际是 40.31%（不是预估的 10-20%）
- Phase F 诊断证明剔除后：88 元素全保留 + split 比例自动保持 + mp_id 损失仅 14.45%
- MA 综合判断：实现简单 + 评估干净 + 75K 数据量足够 + 可逆性好 → 选 D
- **51,746 incompat + 999 multiset-mismatch = 52,745 样本封存到 `incompat_pool.csv`**，留给可能的 Exp5 site-averaging 实验

---

## 3. 关键技术事件 — Phase D v1 failure 与 brute-force 修复

### 故障现象
Phase D v1 用 `multiprocessing.Pool` + `pymatgen.Structure.get_neighbors` 在 r=2-3 Å 调用，**100% multi-site 样本（115,364 个）全部走 `neighbor_error` 失败路径**。

### 根因（用诊断脚本暴露）
```
ValueError: Buffer dtype mismatch, expected 'const int64_t' but got 'long'
File "src\pymatgen\optimization\neighbors.pyx", line 48,
  in pymatgen.optimization.neighbors.find_points_in_spheres
```

pymatgen 2024.8.9 + numpy 1.26.4 在 Windows 环境下，Cython 底层 `find_points_in_spheres` 期望 `int64_t` buffer，但 numpy 默认整数 dtype 在 Windows 上是 `long`(32-bit)，buffer 尺寸不匹配。**且与 r 值无关 — Phase A 用 r=10 也会爆**。Phase A 之前能跑，说明环境在 Phase A → Phase D 之间发生了某种变更（可能 pip 装了什么副作用安装、或 Windows update）。

### 修复（Phase D v2）
完全绕开 pymatgen Cython，用纯 numpy 实现周期镜像 brute-force neighbor finder：
- 用 lattice 体积/面积算每方向需要的镜像层数（perpendicular distance 法）
- meshgrid 构造所有镜像 offset 矩阵
- 一次 vectorized 距离计算 + mask 过滤

**Sanity check**：5 个样本对照 Phase A 已存盘的 distances，**5/5 max diff = 0.00000 Å**（cast 到 float32 后完全相等），证实 brute-force 与 pymatgen 数学上等价。

### 性能
- Pool(8) + 41K 个 mp_id groups
- 全 128K 样本 + 多位点 ~920K queries 共 10 min 跑完
- 比预估慢 5×（IPC + 调度 overhead），但绝对时间可接受

### Step 3 必须知道的事
**brute-force neighbor finder 必须随 Step 3 代码一起部署到服务器**（Linux），否则一旦本地环境再坏一次（或服务器 pymatgen 版本不一致），整个 Dataset 就跑不起来。建议把 `find_neighbors_brute()` 函数封装成 utility module（如 `exp4_utils/neighbors.py`），Step 3 Dataset 直接 import。本地我已经验证算法在玩具 NaCl 上正确（8 Cl@3.464 Å + 6 Na@4.0 Å = 14 邻居）。

---

## 4. Exp4 数据集最终 Name Card

```
═══════════════════════════════════════════════════════════════
                  Exp4 数据集（Option D 剔除版本）
═══════════════════════════════════════════════════════════════
总样本数：           75,637
总 mp_ids：          35,445
元素覆盖：           88 种

切分：
  train      60,507  (80.00%)   28,297 mp_ids
  val         7,624  (10.08%)    3,580 mp_ids
  test        4,481  ( 5.92%)    2,139 mp_ids
  holdout     3,025  ( 4.00%)    1,429 mp_ids

Tag 组成（kept 样本）：
  single_site         13,018  (17.21%)  原胞唯一中心位点
  equivalent          53,877  (71.23%)  多位点全对称等价
  near_equivalent      8,742  (11.56%)  小数值漂移（< 0.1 Å MAE）

封存（incompat_pool.csv）：
  52,745 样本不参与 Exp4 训练
  Step 5 final report 引述："已知盲区 = 52,745 samples in incompat_pool"

vs Exp2：
  Exp2: ~11,636 samples (Fe-oxide focus, single element)
  Exp4: 75,637 samples (35,445 mp_ids, 88 elements)
        → 6.5× 更多样本, 88× 更广元素覆盖
═══════════════════════════════════════════════════════════════
```

---

## 5. 完整交付物清单

### 5.1 Step 3 主索引（必读）

| 文件 | 大小 | 用途 |
|---|---|---|
| `data_inventory_v2.csv` | **33.51 MB** | **Step 3 主索引**，75,637 行 × 15 列（v1 14 列 + `site_equivalence_tag`）|
| `train_samples_v2.csv` | 3.30 MB | Step 3 Dataset (train mode) load |
| `val_samples_v2.csv` | 0.42 MB | Step 3 Dataset (val mode) |
| `test_samples_v2.csv` | 0.24 MB | Step 3 Dataset (test mode) |
| `holdout_samples_v2.csv` | 0.17 MB | Step 5 评估 |

样本 CSV 字段：`[mp_id, center_element, sample_name, site_equivalence_tag]`

v1 inventory 14 列 schema（v2 末尾追加 `site_equivalence_tag` = 15 列）：
```
[sample_name, mp_id, center_element,
 chi_path, xmu_path, poscar_path,
 prim_n_atoms, has_pre_edge,
 chi_valid, xmu_valid, poscar_valid, is_iqr_outlier,
 split, quality_tier]
```

### 5.2 Step 5 评估用

| 文件 | 大小 | 用途 |
|---|---|---|
| `shell_boundaries.pkl` | **369.5 MB** | Step 5 分层 RMSD 评估，每样本 9 字段 schema（threshold/distances/species_Z/shell_starts/shell_ends/shell_n_atoms/shell_of_atom/eval_cutoff/n_center_sites） |

`shell_boundaries.pkl` 含全 128K 样本（包括 incompat），Step 5 用 `data_inventory_v2.csv` 过滤后再使用。

### 5.3 文档归档

| 文件 | 大小 | 用途 |
|---|---|---|
| `site_equivalence_tag.csv` | 9.5 MB | 全 128K 样本含 tag 详情，可追溯 |
| `incompat_pool.csv` | 3.28 MB | 52,745 封存样本，Exp5 备用，Step 5 报告需引述 |
| `step2_5g_summary.txt` | 1 KB | Exp4 数据集 name card |
| `step2_5_neighbor_distances.pkl` | 196 MB | Phase A 中间产物（保留 for debug，可删） |

### 5.4 报告归档（推荐随代码一起 commit）

- `STEP2_5_PHASE_A_REPORT.md`
- `STEP2_5_PHASE_BC_REPORT.md`
- `STEP2_5_PHASE_D_REPORT.md`
- `STEP2_5_PHASE_F_REPORT.md`
- 本报告（`STEP2_5_FINAL_REPORT.md`）

---

## 6. 服务器上传清单（MA 参考）

scp 到 `/home/tcat/diffcsp_exp4/data/`：

```
Step 1（保留 v1 历史归档）:
  data_inventory.csv               (56.8 MB)
  feff_features_imputed.pkl        (40.3 MB)
  feff_feature_scaler.pkl          (1.6 KB)
  feff_feature_stats.csv           (6.8 KB)
  feff_feature_names.txt           (1.0 KB)
  {train,val,test,holdout}_ids.txt (~430 KB)

Step 2:
  spectra_{train,val,test,holdout}.pkl  (185.6 MB total)

Step 2.5（v2 真正训练数据）:
  data_inventory_v2.csv            (33.5 MB)  ★
  train_samples_v2.csv             (3.3 MB)   ★
  val_samples_v2.csv               (0.4 MB)   ★
  test_samples_v2.csv              (0.2 MB)   ★
  holdout_samples_v2.csv           (0.2 MB)   ★
  shell_boundaries.pkl             (369.5 MB) ★
  site_equivalence_tag.csv         (9.5 MB)
  incompat_pool.csv                (3.3 MB)
```

总 ~700 MB，scp 约 5-10 min。★ = Step 3 必读，其他归档/备查。

---

## 7. Step 3 入口指南（给 MA 写交接文档时参考）

### 7.1 Step 3 Sub-Agent 第一步该读什么

按顺序：
1. **本报告 §4 Name Card** — 知道数据集规模与组成
2. **`data_inventory_v2.csv` 第一行 header** — 知道字段
3. **Phase D v1 failure 故事（§3）** — 知道 pymatgen 在 Windows 不可用，Linux 服务器需要验证
4. **Exp2 现有 Dataset 类源码** — 找改造入口点

### 7.2 Step 3 Dataset 改造关键点

```python
class XasLocalDatasetV2(Dataset):
    def __init__(self, split, ...):
        # Load v2 sample list (already filtered, no incompat)
        self.samples = pd.read_csv(f"{DATA_DIR}/{split}_samples_v2.csv")
        # Load shell boundaries (full 128K, but we only access keys in samples)
        self.shells = pickle.load(open("shell_boundaries.pkl", "rb"))
        # Load FEFF features
        self.feff = pd.read_pickle("feff_features_imputed.pkl")
        self.scaler = joblib.load("feff_feature_scaler.pkl")
        # Load spectra
        self.spectra = pickle.load(open(f"spectra_{split}.pkl", "rb"))

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        sname = row["sample_name"]
        # No multi-site dispatching needed — incompat already filtered
        # Just use first-site shell boundaries
        shell = self.shells[sname]
        feff = self.scaler.transform(self.feff.loc[sname].values.reshape(1, -1))
        spectrum = self.spectra[sname]
        return {
            "spectrum": spectrum,
            "feff": feff,
            "shell_starts": shell["shell_starts"],
            "shell_ends": shell["shell_ends"],
            "eval_cutoff": shell["eval_cutoff"],
            ...
        }
```

**关键简化**：因为 Option D 已经过滤掉 incompat，**Dataset 不需要任何多位点分支逻辑**。所有保留样本的 first-site label 都 ≈ site-averaged label（在 0.1 Å MAE 容差内）。直接用 first-site 的 shell boundaries 即可。

### 7.3 Brute-force neighbor finder 部署

Step 3 在服务器（Linux）上**应该可以正常用 pymatgen.get_neighbors**，因为 Windows-specific 的 dtype 问题在 Linux 上不出现。但建议：
1. Step 3 服务器跑通后做一次 sanity check：用 5 个 multi-site 样本对比 pymatgen vs Phase A pkl 的 distances
2. 如果发现 server pymatgen 也坏，把 brute-force 函数从 `step2_5d_full_multisite_tag_v2.py` 抽出来作为 utility module fallback

### 7.4 Spectra pkl 的依赖（Step 2 产物，不在我职责范围）

Step 3 假设 `spectra_{split}.pkl` 已经按 v1 split 切好。**v2 数据集是 v1 的子集**，所以 v2 的所有 sample_names 都在 v1 的 spectra_{split}.pkl 里。Dataset 用 sample_name 作 key 查询即可，不需要重切 spectra。

如果 Step 3 sub-agent 发现 spectra pkl 缺某个 sample → 说明 Step 2 阶段哪里出了 inconsistency，应回头排查（不是 Step 2.5 的责任）。

---

## 8. 已知风险 / 需要 Step 3+ 注意

### 8.1 spectra-shell key 对齐
v2 inventory 用 `sample_name` 作 unique key。`shell_boundaries.pkl` 也用 `sample_name`。`spectra_{split}.pkl` 假设也是同 key（从 Step 2 命名推测）。Step 3 第一个 forward pass 之前**必须对随机 100 个 sample 验证三个 dict 都能 lookup 到**。

### 8.2 v1 ids 文件 vs v2 samples 文件
Step 1 产物 `train_ids.txt` 等是**mp_id 列表**（mp-id-level split），共 41K mp_ids。v2 的 `train_samples_v2.csv` 等是**sample 级**，共 60K samples（每 mp_id 多个 element 谱）。两套文件 purpose 不同：
- `*_ids.txt`：mp_id 切分文档（Step 5 论文表格用）
- `*_samples_v2.csv`：训练时 Dataset 实际加载的列表

Step 3 Dataset 用 v2，**不要混用 v1 ids**。

### 8.3 holdout 样本量小（3,025）
holdout 是最终评估用。3,025 样本对 88 元素分布而言，**某些稀有元素 holdout 样本可能 <5 个**。Step 5 报告分层 RMSD 时要 caveat 稀有元素的统计噪声。

### 8.4 Step 5 报告的"已知盲区"声明
Step 5 final report 必须引述：
> "本工作 Exp4 训练了 75,637 样本（剔除 52,745 'incompat' 样本，详见 Step 2.5 Phase D 报告）。incompat 样本结构上含多个不等价 Wyckoff 中心位点，与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"

---

## 9. Sub-Agent 状态

**Step 2.5 全部任务完成，待命中。**

下一步等待 MA 派发的 Step 3 子任务。可能的请求范围：
- 写 Dataset 类骨架代码
- 服务器上传后做对齐 sanity check
- 提供 brute-force utility module
- 前向测试协议
- 其他技术细节

不主动越权进入 Step 3 工程实现，等 MA 决定切分粒度。

---

**End of Step 2.5 Sub-Agent reports.**
