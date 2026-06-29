# EXP4_STEP6_STEP6AGENT_HANDOFF.md
# DiffCSP-Experiment4 Step 6 可视化 Step6Agent 交接文档

> **撰写者**: DiffCSP-Exp4-Main-Agent 5
> **接收者**: Step 6 可视化 Sub-Agent(用户命名 = Step6Agent)
> **日期**: 2026-04-28
> **背景**: Step 5 完结,val/test/holdout 三 split 评估 PASS,所有 §6 红线全过。Step 6 启动闸门 CLEAR。
> **核心约束**: Step6Agent 只画图,**不写 final report**。Final report 由 MA5 在收到图后直接撰写。

---

## §0 你是谁,你的工作边界

你是 DiffCSP-Exp4 接力链最后一棒 **Step6Agent**(前棒 = Step5Agent)。**任务非常聚焦: 基于 Exp2 `step6_visualize.py` 模板改造,跑出 6 张 figure**。仅此而已。

**你做什么**:
1. Phase 6.0 hard check(env / 输入 CSV 文件齐全)
2. Phase 6.1 写 `step6_visualize.py`(基于 Exp2 模板,具体改动见 §3)
3. Phase 6.2 跑出 6 张 PNG(figures 1/2/2b/3/4/5)
4. Phase 6.3 自检 PNG 视觉合理性 + 写**简短**报告交回 MA5

**你不做什么**:
- ❌ 不写 final report(MA5 任务,你交完图就结束)
- ❌ 不动任何 dataset / model / training / evaluation 脚本
- ❌ 不重算 metrics(per_sample_metrics_*.csv 是你的唯一数据源,Step5Agent 已算好)
- ❌ 不跑 sample / 不加载 ckpt / 不接触 GPU
- ❌ 不深 debug 图渲染问题 ≤ 1 轮观察

**上下文闸门**: 70%(同前棒)。但本任务工作量小,预期 ~30% 完成。

---

## §1 必读文档清单

按读取顺序,**1-3 精读,4 速读**:

| # | 文档 | 必读? | 重点 |
|---|------|-------|------|
| 1 | **本文档** | ✅ 精 | 全文,尤其 §3 改动清单 + §4 红绿灯 |
| 2 | **Exp2 `step6_visualize.py`(用户提供)** | ✅ 精 | 你的代码模板,你看过即可 |
| 3 | **EXP4_STEP5AGENT_FINAL_REPORT.md** | ✅ 精 | 三 split 数据来源 + 数字 baseline + Tier 分层定义 |
| 4 | EXPERIMENT2_FINAL_REPORT.md | 速 | §1 指标定义 + §2.5 random baseline 推导 |

---

## §2 输入文件(Step5Agent 已交付)

**唯一数据源**:

| 文件 | 路径 | 行数 | schema |
|------|------|------|--------|
| `per_sample_metrics_val.csv` | `/home/tcat/diffcsp_exp4/code/step5/` | 7621 | sample_name, mp_id, rmsd, type_acc, n_pred_in, n_true_in, eval_cutoff |
| `per_sample_metrics_test.csv` | `/home/tcat/diffcsp_exp4/code/step5/` | 4481 | 同上 |
| `per_sample_metrics_holdout.csv` | `/home/tcat/diffcsp_exp4/code/step5/` | 3025 | 同上 |

**注意**: per_sample CSV 不含 `pred_frac_coords` / `pred_atom_types` / `true_frac_coords` / `true_atom_types`(这些是 array 字段,不能进 CSV)。**Figure 3(3D 结构对比)需要从 `predictions_*.pt` 读 array 字段**:

| 文件 | 路径 | 大小 | 用途 |
|------|------|------|------|
| `predictions_val.pt` | `/home/tcat/diffcsp_exp4/code/step5/` | 9.84 MB | Figure 3 唯一数据源 |
| (test 和 holdout .pt 也存在但 Figure 3 只用 val) | | | |

`predictions_*.pt` schema(由 Step5Agent 写,你 `torch.load` 后看 keys):
```python
preds = torch.load("predictions_val.pt", weights_only=False)
# 应有 keys: "mp_id", "pred_frac_coords", "pred_atom_types", 
#           "true_frac_coords", "true_atom_types"
# 每个 value 是 list(长度 = N_sample,每个元素是 numpy/tensor (20, 3) 或 (20,))
```

如果实际 schema 与你预期不符,**停下来报 MA5,不要猜**。

**辅助数据**(中心元素信息):

Exp4 是 88 元素中心,Figure 3 / Figure 6 需要知道每个 sample 的 `center_element`。从这里 join:
```python
import pandas as pd
inv = pd.read_csv("/home/tcat/diffcsp_exp4/data/data_inventory_v2.csv")
# 列: mp_id, center_element, sample_name, site_equivalence_tag, ...
```

---

## §3 6 张 Figure 详细清单(基于 Exp2 `step6_visualize.py` 改造)

### 主图 4 张(保留 Exp2 设计 + Exp4 改造)

#### Figure 1: RMSD 分布直方图(3 panel,val/test/holdout 各一)

**Exp2 原版**: 单 panel,只画 val。
**Exp4 改造**:
- **3 子图横排**(figsize ≈ (15, 5)),val / test / holdout 各一
- 每子图 hist + mean RMSD 虚线 + random baseline 虚线(2.32 Å)
- 每子图 title 含 `N` 和 `mean ± std`
- bins 仍 40,range (0, 4),与 Exp2 一致(便于 Exp2 vs Exp4 对照阅读)
- y 轴各子图独立(不强制同 scale,因为 N 不同)

**输出**: `fig1_rmsd_distribution.png`

#### Figure 2: TypeAcc 分布直方图(3 panel,val/test/holdout 各一)

**Exp2 原版**: 单 panel,21 bins(k/20,k=0..20)。
**Exp4 改造**:
- 3 子图横排同 Figure 1
- bin 设计不变(21 bins)
- 每子图 mean TypeAcc 虚线 + random baseline `1/88 ≈ 0.0114`(不是 Exp2 的 0.01)
- title 含 `N`、`mean ± std`

**输出**: `fig2_typeacc_distribution.png`

#### Figure 3: 3D 结构对比(沿用 Exp2 6 panel 设计,**只用 val**)

**Exp2 原版**: 6 panel(2 best / 2 mid / 2 worst),val 数据。
**Exp4 改造**:
- 6 panel 设计**保留**
- mid panel 选 RMSD 接近 1.485(Exp4 mean)的两个,不是 Exp2 的 1.47
- worst panel 仍 RMSD ≤ 3.5 上限筛选(避免极端 outlier 主导)
- **元素配色全改**: 不能用 Exp2 的 `tab10 + O 红 / Fe 橙` 硬编码。改用 **pymatgen Jmol 配色**:
  ```python
  from pymatgen.vis.structure_vtk import EL_COLORS
  jmol = EL_COLORS["Jmol"]   # dict: element_symbol -> (R, G, B) 0-255
  
  # element_symbol 从 Z 转
  from pymatgen.core import Element
  def z_to_symbol(z):
      return Element.from_Z(z).symbol
  
  def element_color(z):
      sym = z_to_symbol(int(z))
      rgb = jmol.get(sym, (128, 128, 128))  # fallback gray
      return tuple(c / 255.0 for c in rgb)
  ```
- **中心元素(原点红星)动态化**: 不再硬写 "Fe center",从 `data_inventory_v2.csv` join 该 sample 的 `center_element`,title 写成 `[Best #1] Fe center, RMSD=...` 或 `[Best #1] Cu center, RMSD=...`
- 全局图例右侧改为:
  - True atoms(大实心球 + 黑边)
  - Predicted atoms(小空心圆 + 同色边)
  - Center atom (origin)(红星 + 黑边)—— 不再写 "Fe center"
  - Matched pair(虚线)
  - **不再列具体元素例图**(O 红 / Fe 橙),因为 88 元素列不完;改为底部一行说明 "Atom colors follow Jmol convention"

**输出**: `fig3_structure_comparison.png`

#### Figure 4: RMSD vs TypeAcc 散点(Exp2 原版,改 3 split 颜色叠加)

**Exp2 原版**: 单色散点 + 单条线性回归线 + Pearson r。
**Exp4 改造**:
- val 蓝 / test 橙 / holdout 绿,三种颜色 alpha=0.3 叠加
- **3 条独立线性回归线**(每 split 一条),颜色与点对应
- annotation 框列 3 行 Pearson r:
  ```
  val:     r=X.XXX, p=X.XXe-XX
  test:    r=X.XXX, p=X.XXe-XX
  holdout: r=X.XXX, p=X.XXe-XX
  ```
- title `RMSD vs Type Accuracy (3-split overlay)`

**输出**: `fig4_rmsd_vs_typeacc.png`

### 补充图 2 张(Exp4 增量,与 Step 5 报告 Tier 分层呼应)

#### Figure 2b: TypeAcc 按 eval_cutoff Tier 分层(boxplot,3 split)

**这是 Exp4 vs Exp2 最核心的 differentiator**。Step5Agent §7.3 明确说"the single most informative figure"。

**设计**:
- x 轴: 4 tier(A: ≤3 / B: 3-4 / C: 4-5 / D: >5),tier 边界来自 Step5Agent §2.2 / §3.2
- y 轴: TypeAcc(0 - 1.0)
- 每 tier 处 3 个并排 box(val / test / holdout),宽度 0.25,中心错位 -0.25/0/+0.25
- 颜色: val 蓝 / test 橙 / holdout 绿(与 Figure 4 对齐)
- 横虚线 reference: `Exp2 Fe-only holdout TypeAcc = 0.241`(贴在 Tier B 区域作 baseline 比对)
- title `Type Accuracy by eval_cutoff Tier (val/test/holdout)`
- caption 子标题: `Tier B (3-4 Å, 1st/2nd shell) ≈ Exp2 parity. Monotone decrease reflects XANES near-shell information limit.`
- **Tier A 在 holdout 是 N=0**,该 tier 的 holdout box 留空(或写 "N/A in holdout"),val/test 正常画(N=13 / N=3 也照画但 box 会很窄)

**Tier 边界 + per-split N 的源数据**(给你 hardcode 用):

| Tier | val N | test N | holdout N |
|---|---|---|---|
| A: ≤3.0 Å | 13 | 3 | **0** |
| B: 3-4 Å | 1961 | 1164 | 797 |
| C: 4-5 Å | 3893 | 2302 | 1536 |
| D: >5.0 Å | 1754 | 1012 | 692 |

**注意**: 这些 N 是 Step 5 报告 §2.2 / §3.2 的快照,**Step6Agent 不要 hardcode 这些数,要从 CSV 实时算**。Tier 边界 hardcode 即可:
```python
def get_tier(eval_cutoff):
    if eval_cutoff <= 3.0: return "A"
    if eval_cutoff <= 4.0: return "B"
    if eval_cutoff <= 5.0: return "C"
    return "D"
```

**输出**: `fig2b_typeacc_by_tier.png`

#### Figure 5: TypeAcc 按"邻居距离 rank"分层(柱状,只用 val)

**目的**: 直接验证"XANES 是近邻探针"——第 k 近邻的预测准确率应单调下降。

**这张图需要从 `predictions_val.pt` 读 array 字段**(per_sample CSV 不够):
- 对每个 val sample,取 `true_frac_coords` 和 `pred_frac_coords`(都是 (20, 3))
- 真实邻居按距 Fe 原点距离 sort(笛卡尔距离 = `||true_frac * L||`),得到 rank 1-20
- 跑 Hungarian 配对(min-image,与 metrics 脚本一致),得到 pred → true 配对
- 对每个 true rank k(1-20),统计"配到的 pred 元素 = true 元素的样本数 / 总样本数",得到 20 个 TypeAcc 值
- 画 20 个柱(rank 1 → rank 20),y 轴 TypeAcc

**预期形态**: rank 1-2(最近邻)TypeAcc 0.4-0.6,单调下降到 rank 15-20 ≈ 0.05-0.10。
**如果不是这个形态**(比如平的 / U 形),报告里写明,这是值得注意的诊断信号。

**title**: `Type Accuracy by Neighbor Distance Rank (Val, N=7621)`
**caption**: `XANES near-shell sensitivity validated: rank-1 neighbors are predicted at >5x random baseline; rank-20 approaches information floor.`

**实现伪码**:
```python
import torch, numpy as np
from scipy.optimize import linear_sum_assignment

preds = torch.load("predictions_val.pt", weights_only=False)
N = len(preds["mp_id"])
L = 6.0
correct_at_rank = np.zeros(20)
total_at_rank = np.zeros(20)

for i in range(N):
    pf = np.asarray(preds["pred_frac_coords"][i])  # (20, 3)
    pt = np.asarray(preds["pred_atom_types"][i])   # (20,)
    tf = np.asarray(preds["true_frac_coords"][i])
    tt = np.asarray(preds["true_atom_types"][i])
    if pf.shape[0] != 20 or tf.shape[0] != 20:
        continue
    
    # rank true neighbors by distance to origin (Fe / center)
    true_dists = np.linalg.norm(tf * L, axis=1)
    rank_order = np.argsort(true_dists)  # rank 1 (closest) → rank 20
    
    # Hungarian match (same as Exp2)
    cost = np.zeros((20, 20))
    for ri in range(20):
        delta = pf[ri] - tf
        delta -= np.round(delta)
        cost[ri] = np.linalg.norm(delta * L, axis=1)
    row, col = linear_sum_assignment(cost)
    # row[k] is pred index, col[k] is true index, pairs (row[k], col[k])
    
    # for each true index, get its rank, then check pred type match
    rank_of_true = np.empty(20, dtype=int)
    for r, ti in enumerate(rank_order):
        rank_of_true[ti] = r  # 0-indexed rank
    
    for ri, ci in zip(row, col):
        rank = rank_of_true[ci]
        total_at_rank[rank] += 1
        if pt[ri] == tt[ci]:
            correct_at_rank[rank] += 1

acc_at_rank = correct_at_rank / np.maximum(total_at_rank, 1)
# bar plot acc_at_rank against ranks 1..20
```

**输出**: `fig5_typeacc_by_rank.png`

### Figure 6 我已 cut(原计划"按中心元素 top-20 RMSD boxplot")

理由: 工作量大但信息量重叠 Figure 2b 已经给出的 "Tier B/C/D" 解读。88 元素散布 boxplot 视觉拥挤,论文/报告里通常是 supplementary。**先不画,如果你或 MA5 看到 figure 1-5 后觉得需要,phase 6b 再补**。

如果你看到 figure 5 跑出来后觉得"信息密度还不够,需要 figure 6 元素分层"——**报告里写一句你的看法,不画**,留给 MA5 决议。

---

## §4 文件归属总表

### 4.1 你新建(放在 `/home/tcat/diffcsp_exp4/code/step6/`)

| 文件 | 阶段 | 备注 |
|---|---|---|
| `step6_visualize.py` | 6.1 | 基于 Exp2 模板改 |
| `figures/fig1_rmsd_distribution.png` | 6.2 | |
| `figures/fig2_typeacc_distribution.png` | 6.2 | |
| `figures/fig2b_typeacc_by_tier.png` | 6.2 | Exp4 新增 |
| `figures/fig3_structure_comparison.png` | 6.2 | |
| `figures/fig4_rmsd_vs_typeacc.png` | 6.2 | |
| `figures/fig5_typeacc_by_rank.png` | 6.2 | Exp4 新增 |

### 4.2 不动文件

- ❌ 任何 dataset / model / metrics / training 脚本(Step 1-5 已封存)
- ❌ Exp2 step6_visualize.py 原件(用户 Windows 端)—— 你不动它,只是参考逻辑
- ❌ predictions_*.pt 和 per_sample_metrics_*.csv —— read-only

---

## §5 Phase 子任务清单

### Phase 6.0:Hard check

```bash
# env(关键! Step5Agent §6.1 教训:确认 mlff env 而非 jhub_env)
which python   # 期望 /home/tcat/conda_envs/mlff/bin/python
python -c "import matplotlib, scipy, pandas, pymatgen; \
print(matplotlib.__version__, scipy.__version__, pandas.__version__, pymatgen.__version__)"

# 输入 CSV 齐
for f in val test holdout; do
  ls -la /home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_${f}.csv
done

# predictions_val.pt(figure 3 + 5 用)
ls -la /home/tcat/diffcsp_exp4/code/step5/predictions_val.pt

# data_inventory_v2.csv(figure 3 中心元素 join 用)
ls -la /home/tcat/diffcsp_exp4/data/data_inventory_v2.csv

# 输出目录
mkdir -p /home/tcat/diffcsp_exp4/code/step6/figures
```

任一 FAIL 立刻停。

### Phase 6.1:写 `step6_visualize.py`

按 §3 6 张图详细清单写。**主体逻辑直接 fork Exp2 step6_visualize.py**,改动点:
1. 路径常量改 Linux + step5/ + step6/
2. 数据源从 `predictions_val.pt` 改为 **`per_sample_metrics_*.csv` + `predictions_val.pt`**(figure 3 / 5 用 .pt,其他用 CSV)
3. element_color 函数全替换为 pymatgen Jmol
4. 6 个 figure 函数按 §3 详细规格逐个写
5. 加 argparse 让 main 可以单独跑某 figure(便于 debug)

**伪码骨架**:
```python
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["1","2","2b","3","4","5"])
    args = ap.parse_args()
    
    # load CSV inputs
    df_val  = pd.read_csv(".../per_sample_metrics_val.csv")
    df_test = pd.read_csv(".../per_sample_metrics_test.csv")
    df_hold = pd.read_csv(".../per_sample_metrics_holdout.csv")
    df_inv  = pd.read_csv(".../data_inventory_v2.csv")
    
    # only load .pt if needed (fig 3 or 5)
    preds_val = None
    if "3" in args.only or "5" in args.only:
        preds_val = torch.load(".../predictions_val.pt", weights_only=False)
    
    if "1" in args.only: plot_fig1(df_val, df_test, df_hold)
    if "2" in args.only: plot_fig2(df_val, df_test, df_hold)
    if "2b" in args.only: plot_fig2b(df_val, df_test, df_hold)
    if "3" in args.only: plot_fig3(df_val, preds_val, df_inv)
    if "4" in args.only: plot_fig4(df_val, df_test, df_hold)
    if "5" in args.only: plot_fig5(preds_val)

if __name__ == "__main__":
    main()
```

### Phase 6.2:跑出 PNG

```bash
cd /home/tcat/diffcsp_exp4/code/step6
/home/tcat/conda_envs/mlff/bin/python step6_visualize.py 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step6_render.log

ls -la figures/
# 期望 6 个 PNG 文件
```

预期 wall time: 1-3 分钟(纯 CPU 算 + 渲染,figure 5 的 Hungarian 7621 次会慢一些但仍 < 1 min)。

### Phase 6.3:自检 + 简短报告

每张图自检以下 sanity:

| Figure | 自检项 |
|---|---|
| 1 | 3 子图都有,mean RMSD ≈ 1.485,baseline 2.32 虚线在,bin 数对 |
| 2 | 3 子图都有,mean TypeAcc ≈ 0.19,21 bins,random baseline 0.0114 虚线在 |
| 2b | 4 个 tier × 3 个 box,Tier A holdout 空白,Tier B 三 split TypeAcc 都在 0.25-0.27 区间,Exp2 reference 虚线在 |
| 3 | 6 panel 都有,中心红星可见,True 实心 / Pred 空心区分清楚,title 含动态中心元素名 |
| 4 | 三色散点,3 条回归线,annotation 列三组 r 和 p |
| 5 | 20 个柱,从左(rank 1)到右(rank 20)单调下降,rank 1 ≈ 0.4-0.6,rank 20 ≈ 0.05-0.15 |

**报告模板**(给 MA5):

```markdown
# Step6Agent Final Report

## 6.0 Hard check: PASS/FAIL

## 6.1-6.2 6 张 figure 产出
- fig1: <wall time s, 文件大小, mean RMSD val/test/holdout 各值>
- fig2: <同>
- fig2b: <四 tier mean TypeAcc 各值,确认 Tier A holdout 是空白>
- fig3: <6 panel 选样描述: best/mid/worst RMSD 数值 + 中心元素>
- fig4: <三组 Pearson r 数值>
- fig5: <rank 1 / rank 5 / rank 10 / rank 20 的 TypeAcc 数值,确认形态单调下降>

## 6.3 视觉自检
[逐图 PASS/疑点 listed]

## 给 MA5 的开放点(可空)
- O1: <如 fig5 形态非单调下降的诊断观察>
- O2: <如 fig3 worst sample 显示明显物理异常>

## 上下文用量自估
- <%>

## 移交
- 6 PNG 路径已就绪 in /home/tcat/diffcsp_exp4/code/step6/figures/
- MA5 接管 final report 撰写
- Step6Agent 关窗口
```

---

## §6 红灯 / 绿灯

### 红灯(立刻停 + 报 MA5)

1. Phase 6.0 任一项 FAIL
2. CSV 行数与 Step5 报告不符(val 7621 / test 4481 / holdout 3025)
3. 元素配色实现 ImportError(pymatgen Jmol 路径)
4. fig5 的 Hungarian 结果与 Step5 metrics 报告 TypeAcc 不一致(说明算法实现有 bug)
   - 检查方法: fig5 的 weighted average TypeAcc 应 ≈ Step5 val aggregate 0.1877
5. 任一图 wall time > 5 分钟(可能死循环)

### 绿灯(可继续)

1. fig1 mean RMSD val/test/holdout 都在 1.48-1.49 区间
2. fig2 mean TypeAcc 都在 0.18-0.20 区间
3. fig2b Tier B 三 split 都在 0.25-0.27,Tier D 在 0.13-0.15
4. fig5 rank 1 > rank 10 > rank 20(形态单调)
5. fig3 6 panel 中心元素分布合理(应有多种元素,不是全 Fe——验证 88 元素中心多样性)

---

## §7 禁令

- ❌ 不写 final report(MA5 任务)
- ❌ 不动 Step 1-5 任何脚本
- ❌ 不重算 metrics(用 Step5 CSV 即可)
- ❌ 不深 debug 图渲染问题 ≤ 1 轮
- ❌ 不用 jhub_env(Step5Agent §6.1 教训,显式用 mlff env 绝对路径)
- ❌ 不画 figure 6(已 cut,留 phase 6b)
- ❌ 不用 emoji(可视化文档保持纯净)

---

## §8 第一条回复建议格式

```
我已读完 MA5 给我的 Step6Agent handoff + 必读文档清单 §1 的全部 4 份。

[简要复述: Step 5 完结,3 split 全 PASS;
我的工作 = Phase 6.0 hard check → 写 step6_visualize.py(基于 Exp2
模板,6 figure 改造)→ 跑出 6 PNG → 简短报告。final report 不归我]

我注意到三个关键约束:
1. 显式用 mlff env 绝对路径,避免 jhub_env 误用(Step5 教训)
2. 元素配色全改 pymatgen Jmol,不用 Exp2 的 tab10 硬编码
3. fig5 必须从 predictions_val.pt 读 array 字段(CSV 不够)

开始前需要确认 1 件事:
1. predictions_val.pt 实际 schema 我假设是 dict-of-lists 形式
   (preds["mp_id"] / preds["pred_frac_coords"] 等),与 Step5Agent 
   sample 脚本一致。请用户先 torch.load 一次确认 keys,我再写 fig 5。
```

---

## §9 最后提醒

**接力链工作哲学**(继承所有前棒):

1. **任务边界**: 你只画图。final report 是 MA5 任务,不要越线
2. **70% 闸门**: 远低于此(预期 ~30%),宽裕,但仍守
3. **不深 debug**: 任何 figure 跑不出来 1 轮观察后即停 + 报
4. **数字必校验**: figure 数字与 Step5 报告一致,差异 > 1% 即视为 bug

Step 6 是 Exp4 接力链最后一棒。**你的 figure 直接进 Exp4 final report**。

---

*MA5 撰写,2026-04-28,等用户 review 后转发到 Step6Agent 窗口启动*
