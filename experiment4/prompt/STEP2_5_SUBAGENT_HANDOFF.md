# STEP 2.5 SUB-AGENT 交接文档
# Experiment 4 — 数据驱动的壳层统计（Shell Boundary Stats）

> **发送对象**：DiffCSP-Exp4-Step2.5-SubAgent（新会话窗口）
> **撰写者**：DiffCSP-Exp4-Main-Agent 2
> **日期**：2026-04-23
> **执行环境**：本地 Windows（Python 3.9，无需 SSH）
> **前置依赖**：Step 1 PASS（`experiment4\step1\` 全部产出）、Step 2 PASS（`experiment4\step2\` 全部产出）

---

## 1. 为什么加这一步（读完再动手）

Exp2 的评估截断是硬编码的 **eval_cutoff = min(d20, 4.0 Å)**。这个 4 Å 在 Fe 氧化物里合理（Fe-O 键长 ~2 Å，第二壳层 ~3.5 Å），但 Exp4 跨 88 种元素（H 到 U），不同元素的壳层结构跨度巨大（U-O 键长 ~2.3 Å、Li-O ~2.1 Å、但重元素第二壳层可能到 4.5 Å 以上）。用单一经验值 4 Å 会对不同元素产生不一致的评估偏好。

**本步的任务是用训练集统计出"壳层边界的自然分隔点"**，把 eval_cutoff 从硬编码常量换成基于数据的物理量。

**算法（🔒 已和 Main Agent 对齐）**：
1. 把中心原子到所有邻居的距离排序：`d[0] < d[1] < d[2] < ...`
2. 相邻距离间隙 `d[i+1] - d[i]` 超过阈值 `gap_threshold` 时，在此处切分壳层
3. 一个壳层包含一组距离连续（间隙 < threshold）的原子
4. `gap_threshold` **不用经验值 0.3**，而是**从训练集的"相邻间隙分布"里选**，让数据自己定

**这一步的定位（🔒）**：
- **只用在评估端**（Step 5 的分层 RMSD 报告 + Step 3 Dataset 的 eval_cutoff 定义）
- **不改训练 loss**
- **不改扩散采样**
- Exp4 vs Exp2 的核心可比性保留

**你不做的事**：
- ❌ 不改训练 loss（保持 Exp2 架构完全不变）
- ❌ 不改采样/后处理
- ❌ 不动 Step 1/2 的任何产出
- ❌ 不涉及谱（xmu/chi/feff），本步纯粹基于 POSCAR 几何

---

## 2. 你的工作流程（两个阶段）

### 阶段 A：开销大的邻居距离计算（30-60 min）

对全部 128,382 样本：
- 加载 POSCAR → 转原胞 → 找中心原子 → 找所有邻居（截至 10 Å，含周期镜像）
- 输出排序后的距离数组 + 原子序数数组
- 基于 train 集聚合"相邻间隙分布" → 直方图 PNG + 候选阈值清单
- **停下来等 Main Agent 裁决 gap_threshold**

### 阶段 B：基于选定阈值生成壳层边界（<1 min）

- 运行一个轻量参数化脚本 `step2_5b_apply_threshold.py --threshold X.XX`
- 读阶段 A 的 `neighbor_distances.pkl`
- 每个样本按 threshold 切壳层
- 输出 `shell_boundaries.pkl`（Step 3/5 直接 load）

**你在阶段 A 完成后按 §8 模板汇报，用户把汇报 + 直方图发给 Main Agent，MA 拍板 threshold，你再跑阶段 B。**

---

## 3. 动手前：必须先做的三件事

### 事一：确认 Step 1/2 产出齐全可读

```python
import pandas as pd
inv = pd.read_csv(r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\data_inventory.csv")
assert inv.shape[0] == 128382
assert set(inv['split'].unique()) == {'train', 'val', 'test', 'holdout'}
assert 'poscar_path' in inv.columns
assert 'center_element' in inv.columns
```

### 事二：在 5 个样本上跑通单样本流水线

先挑 5 个中心元素差异大的样本（如 O / Fe / Cu / La / U），对每个样本：

```python
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

s_super = Structure.from_file(poscar_path)
prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()

# 找中心元素所有位点
center_sites = [i for i, site in enumerate(prim) if site.specie.symbol == center_element]
assert len(center_sites) >= 1, f"No {center_element} in primitive!"

# 用第一个位点算邻居
center_idx = center_sites[0]
neighbors = prim.get_neighbors(prim[center_idx], r=10.0)
# neighbors: list of pymatgen PeriodicNeighbor objects
distances = sorted([nbr.nn_distance for nbr in neighbors])
species   = [nbr.specie.symbol for nbr in neighbors]  # 排序前，和 distances 对齐
```

打印 5 个样本的前 25 个邻居距离。肉眼看：
- 距离应该从某个合理值（1.5-2.5 Å 量级）开始
- 应该能看到"密集 - 间隙 - 密集"的模式（这是壳层结构的视觉体现）
- 10 Å 内应有 20-100 个邻居

### 事三：估算单样本耗时 × 并行策略

对上面 5 个样本 `time` 一下总耗时，除以 5，估算单样本平均秒数。若：
- **< 0.2 s/sample**：预计 <7 min 单线程跑完，直接单线程
- **0.2 - 1.0 s/sample**：单线程 7-35 min，可接受，但推荐多进程
- **> 1.0 s/sample**：**必须**用多进程，否则 >35 min

**关于多进程**：Main Agent 1 说 `num_workers=0` 是针对 **PyTorch DataLoader**（Windows 多进程和 PyTorch 交互不稳定）。本步是纯 CPU pymatgen 计算，不涉及 PyTorch，用 `multiprocessing.Pool` 完全 OK。Windows 下用法：

```python
if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(processes=min(8, os.cpu_count() - 1)) as p:
        results = list(tqdm(p.imap(process_one_sample, sample_rows), total=len(sample_rows)))
```

推荐 `processes = min(8, cpu_count - 1)`，预留 1 核给 OS。如果用户机器 RAM 紧张，降到 4。

**primitive 缓存关键优化** 🔒：同一个 mp_id 的多个元素谱共享 POSCAR。**按 mp_id 缓存 primitive 结构**，41K 独立 mp_id × 2-3 个元素/id。缓存可以节省 65% 的 primitive 转换时间。具体方法见 §5.1。

---

## 4. 路径常量

```python
import os

EXP4_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR   = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR  = os.path.join(EXP4_ROOT, "step2_5")
os.makedirs(STEP25_DIR, exist_ok=True)

INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
```

建议 3 个脚本：
- `step2_5a_compute_neighbors.py` — 阶段 A 主脚本
- `step2_5a_plot_histogram.py` — 阶段 A 的直方图生成 + 候选阈值报告
- `step2_5b_apply_threshold.py` — 阶段 B（参数化，接受 `--threshold`）

---

## 5. 核心逻辑（🔒 LOCKED）

### 5.1 单样本邻居提取 🔒

```python
def process_one_sample(row):
    """row: data_inventory 的一行，含 sample_name, poscar_path, center_element"""
    
    # 1. 加载 + primitive（带缓存）
    prim = primitive_cache.get(row.mp_id)
    if prim is None:
        s_super = Structure.from_file(row.poscar_path)
        prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()
        primitive_cache[row.mp_id] = prim
    
    # 2. 找中心原子（选第一个匹配的位点）
    center_sites = [i for i, site in enumerate(prim)
                    if site.specie.symbol == row.center_element]
    
    if len(center_sites) == 0:
        return dict(sample_name=row.sample_name, status="no_center_atom",
                    distances=None, species=None, n_center_sites=0)
    
    center_idx = center_sites[0]
    n_center_sites = len(center_sites)  # 留档，后面统计多位点样本比例
    
    # 3. 找邻居（10 Å 内，含周期镜像）
    neighbors = prim.get_neighbors(prim[center_idx], r=10.0)
    
    if len(neighbors) == 0:
        return dict(sample_name=row.sample_name, status="no_neighbors",
                    distances=None, species=None, n_center_sites=n_center_sites)
    
    # 4. 按距离排序
    sorted_pairs = sorted([(nbr.nn_distance, nbr.specie.Z) for nbr in neighbors],
                          key=lambda x: x[0])
    distances = np.array([p[0] for p in sorted_pairs], dtype=np.float32)
    species_Z = np.array([p[1] for p in sorted_pairs], dtype=np.int8)
    
    return dict(sample_name=row.sample_name, status="ok",
                distances=distances, species_Z=species_Z,
                n_center_sites=n_center_sites)
```

**两条设计选择（🔒）**：
- **中心原子选第一个位点**：Exp4 FEFF 文件名不带 site 序号，无法精确还原 FEFF 实际用的位点。用"第一个"是稳定的确定性策略。多位点样本记录 `n_center_sites` 供后续分析。
- **距离 10 Å 截断**：足够覆盖前 5-6 个壳层，远超后续评估所需。
- **多进程内变量 `primitive_cache`**：每个 worker 进程自己的缓存，不跨进程共享（无 IPC 成本）。加速来自同一 mp_id 的多样本落到同一 worker 的概率（imap 默认 chunksize 保证一定程度聚集）。若想精准聚集，**按 mp_id 排序 sample_rows** 后再喂给 imap。

### 5.2 多位点样本的处理策略 🔒

**阶段 A 只用"第一个位点"**，但要**统计多位点分布**：

```python
n_center_sites_distribution = Counter([r['n_center_sites'] for r in results if r['status']=='ok'])
# 输出到 summary: "1 site: ?%, 2 sites: ?%, 3+: ?%"
```

若 `n_center_sites >= 2` 的样本比例 > 20%，**汇报里标注**——这提示"第一位点"可能对大量样本不是真正的 FEFF 计算位点，Step 3 的 Dataset 可能需要更精细策略。但**阶段 A 不解决这个问题**，只统计并上报。

### 5.3 间隙分布统计（train 集） 🔒

**只用 train 集（102,660 样本）算阈值**，不用 val/test/holdout（避免信息泄露）：

```python
all_gaps = []  # 全局收集
all_gaps_per_element = defaultdict(list)  # 分中心元素

for r in train_results:
    if r['status'] != 'ok' or len(r['distances']) < 2:
        continue
    d = r['distances']
    gaps = d[1:] - d[:-1]   # 相邻间隙
    all_gaps.extend(gaps.tolist())
    all_gaps_per_element[r['center_element']].extend(gaps.tolist())
```

**注意**：只统计**前 30 个邻居的间隙**（或距离 ≤ 6 Å 内的所有原子的间隙）。超过 6 Å 后，原子密度主导间隙分布，不是壳层切分所需的信号。

```python
# 修正：截断收集
gaps = d[1:] - d[:-1]
mask = d[:-1] <= 6.0   # 只收集"内层"的间隙
all_gaps.extend(gaps[mask].tolist())
```

### 5.4 候选阈值计算 🔒

输出 4 个候选：

1. **双峰谷值**（如果存在）：
   - 对 `all_gaps` 的直方图（`bins=np.arange(0, 1.0, 0.02)`）做 smoothing（简单移动平均窗口 3-5），找局部最小值
   - 若找到一个位于 [0.15, 0.5] 范围内的谷 → 记为 `threshold_valley`
   - 若没找到明显谷值 → 标记为 "not found"

2. **分位数候选**：
   - `threshold_p10 = np.percentile(all_gaps, 90)` —— 90% 的间隙都 ≤ 这个值 → 把最大 10% 的间隙切开
   - `threshold_p15 = np.percentile(all_gaps, 85)` —— 同理，切最大 15%
   - `threshold_p20 = np.percentile(all_gaps, 80)` —— 切最大 20%

   ⚠️ 注意方向：我们要的是"大于 threshold 视为壳层分界"，所以 threshold 应该是**高分位数**（上 10/15/20 分位点）。

3. **经验值参考**：`threshold_empirical = 0.3`（用户最初提议值，用作 sanity 对照）

以上 5 个候选值、各自的壳层切分效果（下一条）全部写进 `candidate_thresholds.txt`，让 Main Agent 选。

### 5.5 阈值效果模拟 🔒

对每个候选阈值，在 train 集上模拟壳层切分，报告：

- **平均壳层数**（前 6 Å 内）
- **第 1 壳层平均原子数**（应 ~4-8）
- **第 1 壳层平均外缘半径**（应 ~2-3 Å）
- **第 2 壳层平均外缘半径**（应 ~3.5-5 Å）
- **孤立单原子壳层占比**（壳层只含 1 个原子的比例——太高说明 threshold 太小，切得过碎）
- **合并异常壳层占比**（前 2 个壳层合并成 1 个的比例——太高说明 threshold 太大，切得不够）

输出表格如：

```
threshold   | mean_n_shells_in_6A | shell1_n_atoms | shell1_outer | shell2_outer | isolated_single_shell% | over_merged%
0.15        | ?                   | ?              | ?            | ?            | ?                      | ?
0.20        | ...                 | ...            | ...          | ...          | ...                    | ...
0.30        | ...                 | ...            | ...          | ...          | ...                    | ...
0.40        | ...                 | ...            | ...          | ...          | ...                    | ...
valley_X.XX | ...                 | ...            | ...          | ...          | ...                    | ...
```

### 5.6 直方图输出 🔒

**`step2_5_gap_histogram.png`**：
- 全局间隙分布直方图（bins=0.02 Å，x 范围 [0, 1.5]）
- 5 条垂直线标出 5 个候选阈值（不同颜色 + 图例）
- 下方 subplot：按中心元素 top-5 频次的元素分别画直方图（观察分布差异）
- 建议 dpi=120，不要太大，方便 Main Agent 看

### 5.7 阶段 B 脚本 `step2_5b_apply_threshold.py` 🔒

```
usage: python step2_5b_apply_threshold.py --threshold 0.25

读：
  - step2_5_neighbor_distances.pkl
  - data_inventory.csv (for sample_name iteration)

输出：
  - shell_boundaries.pkl: dict[sample_name] -> {
      "threshold":       float,
      "distances":       np.ndarray (N_neighbors,) float32, # 全部邻居距离，截至 10Å
      "species_Z":       np.ndarray (N_neighbors,) int8,
      "shell_starts":    np.ndarray (N_shells,) float32,    # 每个壳层起始距离
      "shell_ends":      np.ndarray (N_shells,) float32,    # 每个壳层结束距离
      "shell_n_atoms":   np.ndarray (N_shells,) int32,      # 每个壳层原子数
      "shell_of_atom":   np.ndarray (N_neighbors,) int32,   # 每个邻居属于第几壳层（0-indexed）
      "eval_cutoff":     float,                             # 建议：含 20 个最近邻的最小壳层外缘
      "n_center_sites":  int,                               # 原胞中该元素位点数
    }
  - shell_stats_by_split.csv: 各 split 的平均壳层特征
  - shell_stats_by_element.csv: 各中心元素的平均壳层特征
```

**`eval_cutoff` 计算规则（🔒）**：
```python
# 取前 20 个最近邻所在的最远壳层的外缘距离
if len(distances) >= 20:
    shell_of_20th = shell_of_atom[19]  # 第 20 个邻居（0-indexed: 19）在第几壳层
    eval_cutoff = shell_ends[shell_of_20th]
else:
    eval_cutoff = shell_ends[-1]  # 所有邻居不到 20 个，用最外壳层
```

这替代了 Exp2 的 `min(d20, 4.0)` —— 物理意义是"把第 20 个邻居所在的完整壳层全部纳入评估"，不会把一个壳层切一半。

---

## 6. 输出文件清单

阶段 A 完成后（放在 `STEP25_DIR`）：

| 文件 | 预估大小 | 内容 |
|------|---------|------|
| `step2_5_neighbor_distances.pkl` | ~80-150 MB | dict[sample_name] → {distances, species_Z, n_center_sites, status} |
| `step2_5_gap_histogram.png` | ~200 KB | 全局 + top-5 元素分直方图 |
| `step2_5_gap_stats.csv` | ~1 KB | 全局 gap 的 mean/median/std/分位数 |
| `step2_5_gap_stats_by_element.csv` | ~3 KB | 按中心元素的 gap 统计 |
| `step2_5_candidate_thresholds.csv` | ~1 KB | 5 个候选 + 效果模拟表（§5.5） |
| `step2_5a_summary.txt` | ~3 KB | 阶段 A 人类可读报告 |
| `step2_5_failures.csv` | <5 KB | status != "ok" 的样本清单（预期 0 或极少） |

阶段 B 完成后（threshold 选定后）：

| 文件 | 预估大小 | 内容 |
|------|---------|------|
| `shell_boundaries.pkl` | ~150-250 MB | §5.7 schema |
| `shell_stats_by_split.csv` | ~1 KB | 4 split 的壳层统计 |
| `shell_stats_by_element.csv` | ~3 KB | 按元素 |
| `step2_5b_summary.txt` | ~2 KB | 阶段 B 人类可读报告 |

---

## 7. 自查清单（阶段 A 汇报前必跑）

1. **成功处理样本数** = 128,382（任何 `status != 'ok'` 都汇报）
2. **每样本至少 5 个邻居**（否则壳层统计不够，汇报哪些样本）
3. **多位点分布**：`n_center_sites` 的 value_counts，特别报 `>=2` 的占比
4. **距离数组单调递增**（sanity）：抽 100 个样本 assert
5. **gap 分布非零**：`np.std(all_gaps) > 0`，避免所有 gap 全一样的退化
6. **直方图可视化 sanity**：肉眼看是否有双峰特征
7. **候选阈值合理性**：
   - `threshold_p10/p15/p20` 应在 [0.05, 0.60] 范围内
   - `threshold_valley`（若找到）应在 [0.15, 0.50] 范围内
   - 若某候选超出 [0.05, 1.0] **汇报异常**
8. **前 20 邻居平均距离** ≤ 5 Å（全局统计）—— 超出可能提示 10 Å 截断不够，但此值不会超
9. **Cache 命中率**：`cached_hits / total_calls` 应 ≈ `1 - 41431/128382 ≈ 68%` ± 5%

---

## 8. 阶段 A 汇报模板

```markdown
## Step 2.5 阶段 A 完成报告（邻居统计 + 候选阈值）

### 8.1 执行总览
- Wall-clock：? min
- 进程数：? (单线程 / multiprocessing Pool N)
- 处理样本数：? / 128,382（status="ok"）
- 失败样本数：?（status != "ok" 的拆解）
  - no_center_atom: ?
  - no_neighbors: ?
  - 其他: ?

### 8.2 多位点样本分布
| n_center_sites | 样本数 | 占比 |
|---|---|---|
| 1 | ? | ?% |
| 2 | ? | ?% |
| 3 | ? | ?% |
| 4+ | ? | ?% |

[如果 ≥2 占比 > 20%, 在此标红汇报]

### 8.3 邻居统计
- 全局平均：每个样本 ? 个邻居（10Å 内）
- 平均第 1 近邻距离：? Å
- 平均第 20 近邻距离：? Å

### 8.4 间隙分布
[粘贴 step2_5_gap_stats.csv 的全局行]
- 全局 gap mean: ?
- 全局 gap median: ?
- 全局 gap p85 / p90 / p95: ? / ? / ?

### 8.5 候选阈值表
| candidate | value | mean_n_shells_6A | shell1_n_atoms | shell1_outer | shell2_outer | isolated_single% | over_merged% |
|---|---|---|---|---|---|---|---|
| valley | ? | ? | ? | ? | ? | ? | ? |
| p10 (p90 cut) | ? | ? | ? | ? | ? | ? | ? |
| p15 (p85 cut) | ? | ? | ? | ? | ? | ? | ? |
| p20 (p80 cut) | ? | ? | ? | ? | ? | ? | ? |
| empirical 0.3 | 0.30 | ? | ? | ? | ? | ? | ? |

### 8.6 直方图视觉观察
[文字描述：是否有双峰？谷值位置？]
[附 step2_5_gap_histogram.png]

### 8.7 按元素分组 gap 分布
[表格：top-10 频次中心元素的 gap median 和 p90，观察是否元素间差异巨大]

### 8.8 Main Agent 决策请求

请从以下候选中选一个 threshold，我立即跑阶段 B：
- [a] valley = ?
- [b] p10 = ?
- [c] p15 = ?
- [d] p20 = ?
- [e] empirical = 0.30
- [f] 其他（指定具体值）

### 8.9 其他发现/异常
[若无写"无"]
```

---

## 9. 不要做的事

1. ❌ 不要用 val/test/holdout 的距离参与 gap 分布统计（信息泄露）
2. ❌ 不要先选 threshold 再动手 —— 阶段 A 产出候选，阶段 B 才实际切壳层
3. ❌ 不要删除 Step 1/2 任何产出
4. ❌ 不要写入 Step 1/2 的目录（`step2_5\` 是你专属）
5. ❌ 不要在阶段 A 里产 `shell_boundaries.pkl`（那是阶段 B 的事）
6. ❌ 不要把 `primitive_cache` 做成**跨进程共享**的全局变量（IPC 成本抵消收益）。每个 worker 自己的 local cache 就够。
7. ❌ 不要改 `symprec=0.1`（与 Exp2 一致，保持可比）
8. ❌ 不要对 POSCAR 做额外的 supercell 扩展（`get_neighbors(r=10.0)` 自己处理周期镜像）
9. ❌ 不要用 `CrystalNN` / `VoronoiNN` 等"智能"邻居识别器（它们对重元素或低对称结构不稳定），只用简单的距离截断
10. ❌ **不要在阶段 A 汇报前自作主张跑阶段 B** —— Main Agent 要看直方图才能定 threshold

---

## 10. 依赖

```
pandas
numpy
scipy          (仅用于 smoothing 直方图找谷值，可选)
pymatgen
matplotlib
tqdm
```

复用 Step 2 的环境即可。

---

## 11. 交付节奏

1. 跑完阶段 A → 按 §8 汇报 + 附直方图 PNG
2. 用户转发给 Main Agent
3. Main Agent 和用户讨论直方图，定 threshold
4. 你收到 threshold 后跑阶段 B（<1 min）
5. 阶段 B 简短汇报（产出文件列表 + 各 split 的壳层统计摘要）
6. Main Agent 审阶段 B → 进入 Step 3（服务器 Dataset + encoder 改造）

---

*DiffCSP-Exp4-Main-Agent 2 撰写，2026-04-23*
