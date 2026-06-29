# EXP5_SA_METRICS_V3_LAUNCH_NOTE.md
# SA-METRICS-V3 Launch Note — 7 项复合评分 + min_d 1.5 Å gate(Exp5 v2 物理评估改造)

> **From**: Exp5 MA2(Exp5 v2 extension Main Agent,接 MA5)
> **To**: SA-METRICS-V3(评估改造 sub-agent,新棒)
> **Date**: 2026-05-01
> **Anchor**: EXP5_PROPOSAL_v2_AMENDED.md §B / EXP5_MA2_HANDOFF.md §2
> **估时**: 2-3h(设计+实现+dry-run+全量+hand-back)
> **本 launch note 是 SA agent 的唯一 ground truth**:与 proposal §B.2 / handoff §2.3 描述若有冲突,以本文件为准(MA2 已基于实际 schema verify 修正)

---

## §0 Framing — 这次 sub-agent 在做什么

**你不是在产出 verdict 决定要不要 Exp5'。Exp5' 几乎已确定启动**。MA2 已用 200-sample probe 跑出
SA2 baseline 的 min_d 违反率 = **95.0%**(190/200,p1=0.004 Å,完全重合),proposal §B.5 verdict 表
所有阈值都被穿透到表外,Exp5' 加 pairwise penalty 重训是定论。

**你的产出有两个目的**(并列,不分主次):

1. **为 Exp5' 的 `cost_pairwise_min` λ 设计提供精确数据** — min_d 分布 / 违反程度 / 完全重合占比,
   决定 λ 起步值与 schedule
2. **完整中立的 7 项复合分诊断** — gate-pass 子集质量是关键信号:若 5% gate-pass 子集复合分高,
   说明模型在物理 OK 子集上学到了真东西;若 gate-pass 子集仍然低,Exp5' 不光要修物理还要从根本改

两个目的不冲突 — 同一个 step5_3 脚本一次跑出两类输出。

### 0.1 红线(全程不动)

- ❌ 不重 sample(用 SA3' 已有 `predictions_v2_{val,test}.pt`,from SA2 epoch 484 baseline)
- ❌ 不动 `step5_2_compute_metrics.py`(及其 4 个 v2 算法函数,含 R_max 5.5 Å fallback bug — 留作历史对照)
- ❌ 不动 ckpt / 不动 holdout / 不动 yaml / 不动训练代码 / 不动 datamodule
- ❌ 不修 SA1' 5.5 Å fallback bug(留作历史锚点,新文件做对就行)
- ❌ 不 import `step5_2_compute_metrics.py` 任何 shell-related / R_max 相关函数(防 bug 传染)
- ❌ 任何不确定的事 → 写脚本让用户跑 confirm,不靠记忆

### 0.2 红线 — 不要扩 scope

本 sub-agent 只做评估改造,**不**做:
- Exp5' 训练相关任何工作(loss 加项 / yaml 改 / train.py 改 / warm-start 决策)
- SA4' figure
- step5_2 deprecation(可以的,但不在本棒任务里)

---

## §1 任务 scope

新写文件: `/home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py`(SA1' / SA3' 都没动过此文件名)。

实现:
1. shell_boundaries.pkl per-sample 正确读取(取代 SA1' 5.5 Å fallback;**真值端**直接 lookup,不重算)
2. **预测端壳层分配**:沿用 Exp4 Step 2.5 gap 算法,threshold=0.1563 Å 写死(MA2 拍板)
3. min pairwise distance ≥ 1.5 Å gate(violate → 总分 0,其余 6 项不算)
4. 6 项加权评分(权重 0.20×3 shell-1 + 0.10×3 shell-2,详 §3.3)
5. 输出 6 文件 + gate-pass 子集独立统计

**不动**: step5_2 任何东西(SA1' 4 个 v2 算法函数保留作历史对照)。

---

## §2 强制第 1 步 — Schema verify

**进 §3 之前必须先做这步,否则 sub-agent 立即 hand-back 让 MA2 review**。

MA2 已基于实际 schema verify 过 shell_boundaries.pkl 字段(handoff §2.3 描述与实际有出入 —
`shell_starts` / `shell_ends` 是 `float32` 径向距离 Å,不是 int index)。但 SA agent 要自己跑一遍,
确认数据没变 + 自己理解算法前提。

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

/home/tcat/conda_envs/mlff/bin/python <<'PY'
import pickle, torch
sb = pickle.load(open('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl', 'rb'))
p_val = torch.load('/home/tcat/diffcsp_exp5/code/step5/predictions_v2_val.pt',
                   map_location='cpu', weights_only=False)
sn = p_val['sample_name'][0]
print(f'predictions_v2_val.pt sample_name[0]: {sn!r}')
print(f'sb[sn] in dict: {sn in sb}')
sb_sn = sb[sn]
print(f'shell_boundaries fields: {list(sb_sn.keys())}')
for k in ['threshold', 'shell_starts', 'shell_ends', 'shell_n_atoms', 'eval_cutoff']:
    v = sb_sn[k]
    print(f'  {k}: type={type(v).__name__}, ' + 
          (f'shape={v.shape}, dtype={v.dtype}, head={v[:3] if hasattr(v, "size") and v.size > 0 else v}'
           if hasattr(v, 'shape') else f'value={v}'))
print(f"predictions L: {p_val['L']}, eval_cutoff[0]: {p_val['eval_cutoff'][0] if 'eval_cutoff' in p_val else 'NO'}")
print(f"sb[sn] eval_cutoff: {sb_sn['eval_cutoff']}")
PY
```

期望(MA2 已 verify):
- `shell_starts` shape=(5,) dtype=float32,值是径向距离 Å(NOT int index)
- `shell_ends` 同上
- `threshold = 0.1563`
- `shell_of_atom`:per-atom shell index(0-based)
- predictions L 是 scalar(6.0),eval_cutoff 是 per-sample list

若实际输出与 MA2 描述任何字段不一致,**stop 并 hand-back**。

---

## §3 step5_3_composite_score.py 实施细节

### 3.1 真值端 — shell_boundaries.pkl per-sample lookup

```python
sb_i = shell_boundaries[sample_name]  # 已切好

# 真值 shell-1
true_shell1_mask = (sb_i['shell_of_atom'] == 0)
true_shell1_distances = sb_i['distances'][true_shell1_mask]   # (n,) Å
true_shell1_species_Z = sb_i['species_Z'][true_shell1_mask]   # (n,)
true_shell1_n         = int(sb_i['shell_n_atoms'][0])         # scalar

# 真值 shell-2(可能不存在)
if len(sb_i['shell_n_atoms']) >= 2 and sb_i['shell_n_atoms'][1] > 0:
    true_shell2_mask = (sb_i['shell_of_atom'] == 1)
    true_shell2_distances = sb_i['distances'][true_shell2_mask]
    true_shell2_species_Z = sb_i['species_Z'][true_shell2_mask]
    true_shell2_n         = int(sb_i['shell_n_atoms'][1])
else:
    true_shell2_distances = None
    true_shell2_species_Z = None
    true_shell2_n         = 0
```

**不重算真值** — 直接用 Exp4 Step 2.5 已切好的产出。

### 3.2 预测端 — Step 2.5 gap 算法实时跑

**核心写死常量**(放文件顶部):

```python
SHELL_GAP_THRESHOLD = 0.1563  # Å, Exp4 MA2 拍板的 p10 全局阈值,跨实验不变
MIN_PAIRWISE_DIST   = 1.5     # Å, physical lower bound (FEFF/EXAFS 化学键最小), NOT a tunable hyper
```

```python
def assign_pred_shells(pred_frac_coords, pred_atom_types, L, eval_cutoff,
                       gap_threshold=SHELL_GAP_THRESHOLD):
    """Step 2.5 gap 算法用在 pred 上,镜像真值端切壳逻辑。"""
    # 1. frac → cart(absorber 在原点,frac coord 已减 0.5)
    pred_xyz = pred_frac_coords * L                          # (20, 3)
    radial   = pred_xyz.norm(dim=1)                          # (20,)

    # 2. eval_cutoff defensive 截(实际 0 裁,因 box 半对角线 5.196 < 任何 sample 的 eval_cutoff)
    in_mask = radial <= eval_cutoff
    radial_in = radial[in_mask]
    Z_in      = pred_atom_types[in_mask]

    # 3. 排序
    order     = radial_in.argsort()
    sorted_d  = radial_in[order]
    sorted_Z  = Z_in[order]

    # 4. adjacent gap > threshold 处切壳
    if len(sorted_d) < 2:
        # 极端情况:1 个或 0 个 in-cutoff atom
        return {
            'n_pred_shells': 1 if len(sorted_d) == 1 else 0,
            'shell1_distances': sorted_d, 'shell1_species_Z': sorted_Z,
            'shell1_n': len(sorted_d),
            'shell2_distances': None, 'shell2_species_Z': None, 'shell2_n': 0,
        }
    gaps = sorted_d[1:] - sorted_d[:-1]
    boundaries = (gaps > gap_threshold).nonzero(as_tuple=True)[0] + 1  # boundary indices

    # 5. 按 boundaries 切 shell-1 / shell-2
    n_pred_shells = int(boundaries.numel()) + 1
    if n_pred_shells == 1:
        shell1_end = len(sorted_d)
        shell2_start, shell2_end = None, None
    else:
        shell1_end = int(boundaries[0].item())
        shell2_start = shell1_end
        # NOTE: shell-2 includes any further shells (shell-3+ if pred has them)
        # see report header annotation in §3.4
        shell2_end = int(boundaries[1].item()) if n_pred_shells >= 3 else len(sorted_d)

    pred_shell1_distances = sorted_d[:shell1_end]
    pred_shell1_species_Z = sorted_Z[:shell1_end]
    pred_shell2_distances = sorted_d[shell2_start:shell2_end] if shell2_start is not None else None
    pred_shell2_species_Z = sorted_Z[shell2_start:shell2_end] if shell2_start is not None else None

    return {
        'n_pred_shells': n_pred_shells,
        'shell1_distances': pred_shell1_distances,
        'shell1_species_Z': pred_shell1_species_Z,
        'shell1_n': len(pred_shell1_distances),
        'shell2_distances': pred_shell2_distances,
        'shell2_species_Z': pred_shell2_species_Z,
        'shell2_n': len(pred_shell2_distances) if pred_shell2_distances is not None else 0,
    }
```

**关键性质**:`gap_threshold` 与真值端 `sb_i['threshold']` 同源(都是 0.1563),所以两边切法
同算法、同阈值 — apples-to-apples。

### 3.3 7 项评分(gate + 6 weighted scores)

**Gate**:
```python
def compute_min_pairwise(pred_xyz):
    """pred_xyz: (20, 3) cart Å"""
    pw = torch.cdist(pred_xyz, pred_xyz)
    pw.fill_diagonal_(float('inf'))
    return pw.min().item()

# Apply
min_d = compute_min_pairwise(pred_xyz)
gate_pass = (min_d >= MIN_PAIRWISE_DIST)

if not gate_pass:
    total_score = 0.0
    # Don't compute the 6 sub-scores; csv records min_d, n_pred_shells, gate_pass=0,
    # all 6 sub-scores = 0.0, total = 0.0
```

**6 项评分函数**(proposal §B.2 + 距离用 mean radial,MA2 拍板):

```python
def score_coord_n(pred_n, true_n, tol):
    delta = abs(pred_n - true_n)
    if delta <= tol:
        return 1.0
    return max(0.0, 1.0 - (delta - tol) / 3.0)

def score_distance(pred_d_mean, true_d_mean, tol=0.2):
    if pred_d_mean is None or true_d_mean is None:
        return 0.0  # shell-2 missing 视为分=0(公平惩罚,csv 单独记 n_pred_shells 让用户后期分析)
    delta = abs(pred_d_mean - true_d_mean)
    if delta <= tol:
        return 1.0
    return max(0.0, 1.0 - (delta - tol) / 0.5)

CNO_SET = {6, 7, 8}  # C, N, O

def cno_token(z):
    return -1 if int(z) in CNO_SET else int(z)  # -1 作为合并 token,避开真实 Z

def score_element(pred_Z, true_Z):
    if pred_Z is None or true_Z is None:
        return 0.0  # shell-2 missing
    pred_tokens = [cno_token(z) for z in pred_Z.tolist()]
    true_tokens = [cno_token(z) for z in true_Z.tolist()]
    # multiset 交集 / 总数
    from collections import Counter
    pred_c, true_c = Counter(pred_tokens), Counter(true_tokens)
    inter = sum((pred_c & true_c).values())
    total = max(sum(pred_c.values()), sum(true_c.values()))
    return inter / total if total > 0 else 0.0
```

**总分组装**:
```python
if gate_pass:
    s1_n  = score_coord_n(pred['shell1_n'], true_shell1_n, tol=1.5)
    s1_d  = score_distance(pred['shell1_distances'].mean().item() if pred['shell1_n']>0 else None,
                           true_shell1_distances.mean() if true_shell1_n>0 else None)
    s1_e  = score_element(pred['shell1_species_Z'], true_shell1_species_Z)
    s2_n  = score_coord_n(pred['shell2_n'], true_shell2_n, tol=3.0)
    s2_d  = score_distance(pred['shell2_distances'].mean().item() if pred['shell2_n']>0 else None,
                           true_shell2_distances.mean() if true_shell2_n>0 else None)
    s2_e  = score_element(pred['shell2_species_Z'], true_shell2_species_Z)
    total = 0.20*s1_n + 0.20*s1_d + 0.20*s1_e + 0.10*s2_n + 0.10*s2_d + 0.10*s2_e
else:
    s1_n = s1_d = s1_e = s2_n = s2_d = s2_e = 0.0
    total = 0.0
```

### 3.4 输出文件(6 个,落 `/home/tcat/diffcsp_exp5/logs/`)

```
composite_score_val.txt              ← 主报告 1
composite_score_test.txt             ← 主报告 2
composite_score_per_sample_val.csv   ← 每样本 12 字段
composite_score_per_sample_test.csv
min_d_violations_val.csv             ← gate fail 子集(Exp5' λ 设计依据)
min_d_violations_test.csv
```

**Per-sample csv schema**(12 字段):
```
sample_name, gate_pass, min_d, n_pred_shells,
score_shell1_coord, score_shell1_dist, score_shell1_elem,
score_shell2_coord, score_shell2_dist, score_shell2_elem,
total_score
```

**主报告 txt 格式**(注意 shell-2 注释 + gate-pass 子集独立统计):
```
=== EXP5 V2 SA-METRICS-V3 COMPOSITE SCORE - <split> ===
Source: predictions_v2_<split>.pt (from SA2 baseline, ckpt epoch=484-val_loss=0.7065)
Shell algorithm: Exp4 Step 2.5 gap-based, threshold=0.1563 Å (applied to both true and pred)
Note: shell-2 = first remaining gap-bounded group after shell-1; may include shell-3+ atoms.
Note: MIN_PAIRWISE_DIST = 1.5 Å is a physical lower bound, NOT a tunable hyper.

Total samples:           N
min_d gate pass:         X / N (XX.X%)
min_d gate fail:         X / N (XX.X%)

--- Composite score (ALL samples, gate-fail counts as 0) ---
Total weighted mean:     0.XXX
  shell-1 coord_n:       0.XXX  (weight 0.20)
  shell-1 distance:      0.XXX  (weight 0.20)
  shell-1 elem (CNO eq): 0.XXX  (weight 0.20)
  shell-2 coord_n:       0.XXX  (weight 0.10)
  shell-2 distance:      0.XXX  (weight 0.10)
  shell-2 elem (CNO eq): 0.XXX  (weight 0.10)

--- Composite score (GATE-PASS subset only, key diagnostic) ---
N_gate_pass:             X
Total weighted mean:     0.XXX
  (same 6 sub-scores)

--- min_d distribution ---
mean=X.XX, median=X.XX, p10=X.XX, p1=X.XX (Å), min=X.XX, max=X.XX
samples with min_d <1.5 (gate fail): X
samples with min_d <1.0:             X
samples with min_d <0.5:             X
samples with min_d <0.1 (essentially overlap): X

--- n_pred_shells distribution ---
0 shells: X
1 shells: X
2 shells: X
3 shells: X
>=4 shells: X
```

**`min_d_violations_<split>.csv`** schema:
```
sample_name, min_d, n_pred_shells
```
(只记 gate fail 样本,Exp5' λ schedule 直接用)

### 3.5 不要做的事

- 不要 import `step5_2_compute_metrics.py` 任何函数(防 bug 传染)
- 不要尝试"修复" SA1' 5.5 Å fallback 在 step5_2 里(留历史锚点)
- 不要碰 `predictions_v2_*.pt`(只读)
- 不要碰 `shell_boundaries.pkl`(只读)
- 不要碰 ckpt(本棒 0 个 ckpt 操作)
- 不要在 step5_3 里加任何"评估保护机制"或"fallback"(任何不确定的查询失败 → raise,不静默 fallback)

---

## §4 Dry-run gate(全量前必须先 dry-run)

加 argparse `--debug-n-samples N`,先跑两 split 各 100 样本:

```bash
cd /home/tcat/diffcsp_exp5/code/step5
export PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code

/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score.py \
    --split val --debug-n-samples 100
/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score.py \
    --split test --debug-n-samples 100
```

**Dry-run hand-back gate**:在贴回 dry-run stdout(主报告 txt 内容 + 任何 stderr)给 MA2 之前,
**不**跑全量。MA2 review:
- min_d 违反率 ~95%(应与 MA2 200-sample probe 一致)
- gate-pass 子集 N ≥ 5
- n_pred_shells 分布合理(probe 显示 ≥4 shells 占 ~98%)
- 0 stderr error / 0 missing sample_name

ack 后才跑全量(2 split full,~5-10 min)。

---

## §5 全量跑后 verify checklist

```
✅ 6 个输出文件全在(2 主报告 txt + 2 per-sample csv + 2 violations csv)
✅ val per-sample csv 行数 = 7621
✅ test per-sample csv 行数 = 4481
✅ violations csv 行数 ≈ 7621×0.95 / 4481×0.95(允许 ±5% 偏差)
✅ 所有 score_* 字段 ∈ [0, 1]
✅ total_score = 0.20×s1n + 0.20×s1d + 0.20×s1e + 0.10×s2n + 0.10×s2d + 0.10×s2e(抽 5 行 manual verify)
✅ gate_pass=0 的行 total_score=0(无例外)
✅ 主报告 txt 含 "GATE-PASS subset only" 独立统计 block
```

---

## §6 Hand-back 必交内容(SA → MA2)

写 `EXP5_SA_METRICS_V3_OUTPUT.md`,~150 行,目录:

1. Schema verify 输出(§2 跑出的实际)
2. Dry-run 输出 + MA2 ack 时间戳
3. 全量执行 wall time / cmd / 完成时间
4. 6 输出文件路径 + 大小 + md5
5. 主报告 txt 内容粘贴(val + test 各一份)
6. 关键数字摘录(min_d 违反率 / 复合分均值 / gate-pass 子集复合分 / n_pred_shells 分布)
7. **给 MA2 的 Exp5' λ 起步建议**(根据 min_d 违反率档位:proposal §C.2 表):
   - violations < 10%:λ=0.1
   - 10-30%:λ=0.5
   - > 30%:λ=1.0
   - 加你自己看到的数据后的精化建议(如 p1=0.004 Å 完全重合占比是否影响 λ schedule 设计)
8. 已知 issue / 边角情况(若 dry-run 或全量过程中遇到任何 raise / fallback / 异常 sample,详记)

---

## §7 完成定义(Done = 以下全 ✓)

- [x] §2 schema verify 跑过 + 与 MA2 描述一致
- [x] step5_3_composite_score.py 写完 + 通过 §3 红线检查
- [x] dry-run 100 样本 ×2 split 完成 + MA2 ack
- [x] 全量 val + test 完成 + §5 verify checklist 全过
- [x] EXP5_SA_METRICS_V3_OUTPUT.md 写完 + 给 MA2 review
- [x] 6 输出文件全部落 `/home/tcat/diffcsp_exp5/logs/`

---

## §8 估时拆分

| 阶段 | 估时 |
|---|---|
| §2 schema verify | 5 min |
| §3 step5_3 实现(~ 250 行)| 60 min |
| §4 dry-run + 等 MA2 ack | 15 min + ack 时间 |
| 全量两 split | 5-10 min |
| §5 verify checklist | 10 min |
| §6 OUTPUT.md 撰写 | 30 min |
| **总** | ~ 2-2.5h |

---

*Exp5 MA2 撰写,2026-05-01。基于 EXP5_PROPOSAL_v2_AMENDED §B.2 + MA2 schema verify(handoff §2.3 字段类型修正)
+ MA2 200-sample probe(min_d 违反率 95%,n_pred_shells ≥4 占 97.5%)+ 用户拍板(Step 2.5 gap 算法 / mean radial / 1.5 Å 写死 / 3 项 review 全采纳)。*
