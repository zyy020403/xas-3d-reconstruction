# EXP5_PRIME_STEP3_SAMPLE_HANDOFF.md
# SA-EXP5'-STEP3-SAMPLE 任务 launch note(Exp5'-MA → SA-EXP5'-STEP3-SAMPLE)

> **From**: Exp5'-MA(Exp5 系列第 3 任 Main Agent)
> **To**: SA-EXP5'-STEP3-SAMPLE(新一棒,起自干净窗口)
> **日期**: 2026-05-03
> **任务范围**: 用 STEP2 last.ckpt 跑 sample(val + test + holdout)→ 算 7 项复合分 + min_d gate → 物理 sanity → hand-back(~ 1-2 天)
> **预期 hand-back**: ckpt → predictions.pt → 7 项复合分 + min_d violations csv → Exp5'-MA review → 启动 SA-EXP5'-STEP4-FINAL-REPORT

---

## §0 一屏掌握

### 0.1 你是谁

**SA-EXP5'-STEP3-SAMPLE**,新一棒。前置:
- STEP2 训练完成(verdict mixed:composite 0.576 GREEN / gate 0.455 AMBER 边缘)
- ckpt selection bug(详 errata 4 §2)→ best ckpt 实际是 `last.ckpt`(epoch 154,全程 composite 平台峰值)
- 你用 `last.ckpt` 跑 sample,**不重训,不调参**

### 0.2 任务步骤

| 步 | 任务 | 工程量 |
|---|---|---|
| S1 | 服务器环境 verify(STEP2 ckpt + cache + 7 守卫包)| 10 分钟 |
| S2 | 写/复用 sample 脚本 跑 val + test + holdout 三 split | 1.5h(GPU sample 3 split)|
| S3 | 写 step5_3_composite_score.py(沿用 SA-METRICS-V3 实施)+ dry-run 100 sample | 0.5 天 |
| S4 | 完整跑 7 项复合分 val + test + holdout | 0.3h(CPU)|
| S5 | physical sanity 报告(min_d 分布、shell-1 距离、collapse 率)| 0.2 天 |
| S6 | 中期 hand-back 给 Exp5'-MA review | 0.2 天 |

### 0.3 必读 8 份(顺序)

1. **EXP5_PRIME_MA_HANDOFF.md** — Exp5' 接班背景
2. **EXP5_PRIME_PROPOSAL.md** §3.1 verdict 阈值表 + §3.3 复合分公式
3. **EXPERIMENT5_FINAL_REPORT_v2.md** §0 v2 verdict 与 Exp5' 对比基线
4. **EXP5_FILE_GUIDE_v2.md** §6 工作目录 + §8 PYTHONPATH
5. **EXP4_FINAL_REPORT_ERRATA_2.md** — `_density_loss` 旧归因
6. **EXP4_FINAL_REPORT_ERRATA_3.md** ⭐ — fold + L=20 决议
7. **EXP5_PRIME_FINAL_REPORT_ERRATA_4.md** ⭐ — **本 STEP 之前的 ckpt selection bug + STEP2 mixed verdict + last.ckpt 路径决议**
8. **EXP5_PRIME_STEP3_SAMPLE_HANDOFF.md** ⭐ — 本文件

### 0.4 启动后第一条回复格式

```
我已读完 8 份必读。复述任务 6-8 条(含 ckpt 选 last.ckpt 不选 epoch=4 的理由 / 7 项复合分构成 / verdict 双指标)。
最易踩坑 4 条。
计划: ssh 跑 §1 verify。
```

### 0.5 Exp5'-MA 已拍板的 7 条不再讨论

1. **ckpt 用 `last.ckpt`,不用 `epoch=004-gate=0.5305.ckpt`**(errata 4 §6 决议,你不重新评估)
2. **不重训,不调参**(errata 4 §6)
3. **Sample 三 split:val(7621)+ test(4481)+ holdout(1000)**,全集不抽样
4. **L_VIRTUAL = 20**(errata 3,sample 时 dataset / model 路径已是 exp5_prime 配置)
5. **复合分公式沿用 SA-METRICS-V3 设计**(7 项加权,proposal §B.2,详 §3)
6. **min_d gate 阈值 = 1.5 Å**(全局,不做 element-aware,errata 3 §8.2)
7. **shell_boundaries.pkl 不动,inject 进评估即可**(errata 3 §3 已确认 cart Å 干净)

---

## §1 Step S1 — 启动前 verify(10 分钟)

### 1.1 ssh 后跑

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# (A) STEP2 ckpt 完整
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/
# 期望: last.ckpt + last.ckpt.frozen_step2_final + epoch=004-gate=0.5305.ckpt + epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/last.ckpt
# 期望: 9cd39421187df8d02951b9389266de36 (STEP2 hand-back 记录)

# (B) cache 完整(L=20,STEP1-FIX-C 重建)
ls -la /home/tcat/diffcsp_exp5_prime/data/*.pt
cat /home/tcat/diffcsp_exp5_prime/data/cache_metadata.json  # L_VIRTUAL=20.0

# (C) shell_boundaries.pkl
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# 期望: cf2050e4899160f5698ad2481377e94c

# (D) PYTHONPATH 三段 import 全 exp5_prime
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code
/home/tcat/conda_envs/mlff/bin/python -c "
import xas_local_dataset_v2, xas_local_datamodule_v2, diffusion_w_type_xas
print(f'dataset: {xas_local_dataset_v2.__file__}')
print(f'  L_VIRTUAL={xas_local_dataset_v2.L_VIRTUAL}')
print(f'datamodule: {xas_local_datamodule_v2.__file__}')
print(f'model: {diffusion_w_type_xas.__file__}')
"

# (E) GPU 状态(sample 用 GPU 0,STEP2 已结束,GPU 应 idle)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# (F) 磁盘(STEP3 sample 输出 ~ 几 GB,7 项复合分 ~ 几 MB)
df -h /home/tcat
du -sh /home/tcat/diffcsp_exp5_prime/

# (G) holdout 1000 ID 列表
ls -la /home/tcat/diffcsp_exp4/data/holdout*.txt 2>/dev/null
ls -la /home/tcat/diffcsp_exp5_prime/data/holdout*.txt 2>/dev/null
find /home/tcat -name "holdout*1000*" 2>/dev/null
```

### 1.2 PASS gate S1

- ✅ last.ckpt md5 = `9cd39421187df8d02951b9389266de36`(STEP2 hand-back 一致)
- ✅ 3 个 cache .pt + cache_metadata.json L=20
- ✅ shell_boundaries.pkl md5 = `cf2050e4899160f5698ad2481377e94c`
- ✅ PYTHONPATH 全 exp5_prime,L_VIRTUAL=20
- ✅ GPU 0 idle
- ✅ 磁盘 ≥ 10G avail
- ✅ holdout_1000_ids.txt 找到(若不在 exp5_prime/data/,从 exp4 cp)

---

## §2 Step S2 — Sample 三 split(1.5h)

### 2.1 复用旧 sample 脚本

服务器上应有 Exp4/v2 时代的 `step5_1_sample.py`(file guide §2.5),改输入输出路径即可。**先 view 原脚本看接口**:

```bash
ls -la /home/tcat/diffcsp_exp5_prime/code/step5/step5_1_sample.py
cat /home/tcat/diffcsp_exp5_prime/code/step5/step5_1_sample.py | head -80
```

### 2.2 改动点(SA cp 一份 sample 脚本到 step5/sample_exp5_prime/)

```bash
mkdir -p /home/tcat/diffcsp_exp5_prime/code/step5/sample_exp5_prime
cp /home/tcat/diffcsp_exp5_prime/code/step5/step5_1_sample.py \
   /home/tcat/diffcsp_exp5_prime/code/step5/sample_exp5_prime/step5_1_sample_exp5_prime.py
```

改动 4 处:

```python
# 原(Exp4/v2 时代):
CKPT_PATH = "/home/tcat/diffcsp_exp5/checkpoints/sa2_baseline_epoch484_val0.7065.ckpt.frozen"
PRED_OUT_DIR = "/home/tcat/diffcsp_exp5/predictions/"
L_VIRTUAL = 6.0  # 或类似

# 改为(STEP3-SAMPLE):
CKPT_PATH = "/home/tcat/diffcsp_exp5_prime/checkpoints/last.ckpt"  # ⭐ errata 4 §6 决议
PRED_OUT_DIR = "/home/tcat/diffcsp_exp5_prime/predictions/"
L_VIRTUAL = 20.0  # ⭐ errata 3 §8 决议
```

**第 5 处可能需要改**(若 sample 脚本对 L 有内置假设):

```python
# 检查 sample 脚本里是否有 L=6 hardcode
grep -n "L\s*=\|L_VIRTUAL\|6\.0\|6\b" sample_exp5_prime/step5_1_sample_exp5_prime.py
# 如有,改成 20.0(同 errata 3 §9 surgery 流程)
```

### 2.3 跑 sample(三 split,GPU 0)

```bash
mkdir -p /home/tcat/diffcsp_exp5_prime/predictions
cd /home/tcat/diffcsp_exp5_prime/code/step5/sample_exp5_prime

export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# Sample val split(7621 samples,~ 25 分钟)
CUDA_VISIBLE_DEVICES=0 /home/tcat/conda_envs/mlff/bin/python step5_1_sample_exp5_prime.py \
    --split val \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/sample_val.log

# Sample test split(4481 samples,~ 15 分钟)
CUDA_VISIBLE_DEVICES=0 /home/tcat/conda_envs/mlff/bin/python step5_1_sample_exp5_prime.py \
    --split test \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/sample_test.log

# Sample holdout split(1000 samples,~ 4 分钟)
CUDA_VISIBLE_DEVICES=0 /home/tcat/conda_envs/mlff/bin/python step5_1_sample_exp5_prime.py \
    --split holdout \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/sample_holdout.log
```

### 2.4 verify 输出

```bash
ls -la /home/tcat/diffcsp_exp5_prime/predictions/
md5sum /home/tcat/diffcsp_exp5_prime/predictions/*.pt
# 期望 3 个 .pt 文件:predictions_val.pt + predictions_test.pt + predictions_holdout.pt

# 简单 sanity:每个 .pt 的样本数 + 字段
/home/tcat/conda_envs/mlff/bin/python -c "
import torch
for split in ['val', 'test', 'holdout']:
    p = torch.load(f'/home/tcat/diffcsp_exp5_prime/predictions/predictions_{split}.pt', map_location='cpu', weights_only=False)
    print(f'{split}: {len(p[\"sample_name\"])} samples, fields: {list(p.keys())}')
    print(f'  L = {p.get(\"L\", \"N/A\")}')
    print(f'  ckpt = {p.get(\"checkpoint\", \"N/A\")}')
"
```

### 2.5 PASS gate S2

- ✅ 3 个 predictions_{split}.pt 落盘
- ✅ 每个 .pt 含字段:sample_name / pred_frac_coords / pred_atom_types / true_frac_coords / true_atom_types / eval_cutoff / L=20.0 / checkpoint=last.ckpt
- ✅ 样本数:val=7621,test=4481,holdout=1000(可能差几个 silent_drop,< 0.1% 可接受)

---

## §3 Step S3 — 写 step5_3_composite_score.py(0.5 天)

### 3.1 原则:沿用 SA-METRICS-V3 设计,**from-scratch 实现**

旧 Exp5 v2 时代的 `step5/step5_3_composite_score.py` 在服务器上应已存在(SA-METRICS-V3 的产物)。**SA 不直接复用,先 view 旧版,以理解 7 项复合分 + min_d gate 实现细节,然后 fork 一份到 exp5_prime 路径**。

### 3.2 view 旧版 + 决议是否复用

```bash
ls -la /home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py 2>/dev/null
ls -la /home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score.py 2>/dev/null

# 如旧版存在,view 主要逻辑
head -100 /home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py
```

### 3.3 fork + 改动(若旧版存在)

```bash
cp /home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py \
   /home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score_exp5_prime.py

# 改动:
# - PRED_DIR = '/home/tcat/diffcsp_exp5_prime/predictions/'
# - SHELL_BOUND_PATH = '/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl' (不变)
# - LOG_DIR = '/home/tcat/diffcsp_exp5_prime/logs/'
# - L = 20.0 (cart 计算)
# - 新增 holdout split 支持(原脚本只有 val/test)
# - 输出文件名加 ckpt tag:composite_score_val_lastckpt.txt 等(errata 4 §5.4 命名规则)
```

### 3.4 7 项复合分定义(从 proposal §B.2 / SA-METRICS-V3 设计)

每个 sample 算 7 项 score(0-1,1 是完美),加权平均得 composite_score:

| # | 指标 | 权重 | 含义 |
|---|---|---|---|
| 1 | min_d_gate_pass | 0.20 | min pairwise d ≥ 1.5 Å,binary 0/1 |
| 2 | shell-1 distance score | 0.20 | exp(-(pred_s1_d_mean - true_s1_d_mean)² / σ²),σ=0.5 Å |
| 3 | shell-1 count score | 0.20 | exp(-(pred_s1_n - true_s1_n)² / σ²),σ=2 |
| 4 | shell-2 distance score | 0.10 | 同 #2,但 shell-2;若 has_shell2=False,跳过(权重重新分配)|
| 5 | shell-2 count score | 0.10 | 同 #3,但 shell-2 |
| 6 | type set-level acc | 0.10 | 1 - hamming_dist(pred_atom_set, true_atom_set) / N_atoms |
| 7 | overall RMSD score | 0.10 | exp(-RMSD/σ),σ=2.0 Å,Hungarian min-image RMSD |

**权重和 = 1.0**(检查项)。具体公式以旧脚本实现为准,SA 不擅自改公式,只搬迁路径。

### 3.5 输出文件(每 split 6 个)

```
composite_score_{split}_lastckpt.txt              # 主报告(均值 + 分位数 + verdict)
composite_score_per_sample_{split}_lastckpt.csv   # per-sample 7 项 + composite
min_d_violations_{split}_lastckpt.csv             # gate fail 的 sample(min_d < 1.5)
```

3 split × 3 文件 = 9 个输出。

### 3.6 dry-run gate(必跑)

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step5
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# 先跑 100 sample dry-run(每 split)
/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score_exp5_prime.py \
    --split val --debug-n-samples 100 \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/composite_dryrun_val.log

# 看输出文件
ls -la /home/tcat/diffcsp_exp5_prime/logs/composite_score*.txt
cat /home/tcat/diffcsp_exp5_prime/logs/composite_score_val_debug100_lastckpt.txt
```

### 3.7 PASS gate S3

- ✅ step5_3 fork + 4 处改动落盘
- ✅ dry-run 100 sample 跑通,无 NaN/Inf
- ✅ dry-run 输出文件 6 个齐(2 split × 3 / 或 1 split × 3 都可)
- ✅ composite 数值合理(应在 0.5-0.7 范围,与 STEP2 训练时报的 val_composite_ckpt_score=0.576 数量级接近)

---

## §4 Step S4 — 完整跑 7 项复合分(0.3h)

### 4.1 跑三 split 完整版

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step5
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# val 全集
/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score_exp5_prime.py --split val \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/composite_val_full.log

# test 全集
/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score_exp5_prime.py --split test \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/composite_test_full.log

# holdout 全集
/home/tcat/conda_envs/mlff/bin/python step5_3_composite_score_exp5_prime.py --split holdout \
    2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/composite_holdout_full.log
```

### 4.2 PASS gate S4

- ✅ 3 split × 3 文件 = 9 个输出落盘
- ✅ 每个 .txt 主报告 verdict 评级一致(预期 mixed:composite ≥ 0.40 GREEN / gate < 0.80 AMBER)
- ✅ val/test/holdout 三 split 数值差异 < 0.1(generalize 良好)

---

## §5 Step S5 — Physical sanity 报告(0.2 天)

### 5.1 写一个独立 sanity 脚本(不复用 step5_3,从 predictions.pt 直接算)

落盘 `/home/tcat/diffcsp_exp5_prime/code/step5/sample_exp5_prime/physical_sanity.py`,输出:

```
==== Physical Sanity Report ====
Split: val | test | holdout

1. min_d distribution:
   median: X.XX Å
   mean:   X.XX Å
   p10:    X.XX Å
   p90:    X.XX Å
   gate_pass_rate (≥1.5 Å): XX.X%

2. Sample-level shell-1 mean radial distance:
   pred_s1_d_mean: median X.XX, range [X.XX, X.XX]
   true_s1_d_mean: median X.XX, range [X.XX, X.XX]
   error (pred-true) RMSE: X.XX Å

3. Collapse rate (≥ 50% atoms within 0.5 Å):
   X.XX% of samples

4. Top-5 worst samples (by composite_score):
   sample_name  composite  min_d  s1_dist_err  ...
```

### 5.2 PASS gate S5

- ✅ 3 split sanity 报告落盘
- ✅ collapse rate ≤ 1%(STEP2 训练后 model 已学到不 collapse,此项验证)
- ✅ shell-1 distance error RMSE ≤ 1.0 Å(物理学到 shell 概念)

---

## §6 Step S6 — Hand-back

### 6.1 写 `EXP5_PRIME_STEP3_SAMPLE_HANDBACK.md` 落服务器根目录

```markdown
# EXP5_PRIME_STEP3_SAMPLE_HANDBACK.md
# SA-EXP5'-STEP3-SAMPLE hand-back

## §0 状态
- S1-S5 全完成
- ckpt: last.ckpt (epoch 154, composite=0.576)
- 三 split sample 完成 (val 7621 / test 4481 / holdout 1000)
- 7 项复合分 9 个输出文件齐

## §1-§5 各 PASS gate evidence
[逐 step 贴 log + 命令输出]

## §6 verdict 报告(双指标并列,errata 4 §5.3 SOP)
| split | composite | composite verdict | gate | gate verdict | min_d mean | min_d p10 |
|---|---|---|---|---|---|---|
| val   | ... | ... | ... | ... | ... | ... |
| test  | ... | ... | ... | ... | ... | ... |
| holdout | ... | ... | ... | ... | ... | ... |

## §7 7 项复合分明细(全 split)
[贴 9 个文件路径 + 关键数值]

## §8 Physical sanity 摘要
[min_d 分布 + collapse 率 + shell-1 dist RMSE]

## §9 与 Exp5 v2 对比表(关键证据)
| 指标 | Exp5 v2 (SA-METRICS-V3 dry-run 100) | Exp5' STEP3 (val 全集) |
|---|---|---|
| gate_pass_rate | 5-11% (灾难) | ~ 45-50% (大幅改进) |
| shell-1 dist score | 0.0000 | ... |
| composite | 0.0056-0.0062 | 0.576+ |
| collapse rate | ? | ... |

## §10 OPEN 问题
```

### 6.2 PASS gate S6

- ✅ hand-back 完整
- ✅ Exp5'-MA review → 启动 SA-EXP5'-STEP4-FINAL-REPORT
- ✅ 9 个 output 文件 + sanity 报告 + predictions.pt 永久档案

---

## §7 红线(SA-EXP5'-STEP3-SAMPLE 全程不动)

| 红线 | 说明 |
|---|---|
| ❌ 不重训,不调参,不 warm-start | errata 4 §6 |
| ❌ 不动 last.ckpt(只 read,不 write)| 永久档案 |
| ❌ 不动 STEP1-FIX-C / STEP2 训练代码 md5 | 11 文件全锁定 |
| ❌ 不动 cache .pt(L=20)| |
| ❌ 不动 shell_boundaries.pkl(errata 3 §3 干净)| |
| ❌ 不动 holdout 1000 ID 列表 | 永久封存 |
| ❌ 不擅自调 7 项复合分公式 / 权重 | proposal §B.2,Exp5'-MA 决议 |
| ❌ 不擅自调 min_d gate 阈值 1.5 Å | errata 3 §8.2 |
| ❌ 不擅自做 element-aware threshold | errata 3 §8.2 |
| ❌ 不擅自删 STEP2 ckpt(.frozen 永久档案)| |
| ❌ 不擅自启动 STEP4(Exp5'-MA 写)| |
| ❌ 不擅自做 Exp4 / Exp5 v2 cross-check sample(留 STEP4 引用)| |
| ❌ 任何不确定 → ping,不擅自 fix | MA 工作哲学 |

---

## §8 Watch-only 项(SA 报告 Exp5'-MA 决议)

1. **三 split 数值 generalize**:val ≈ test ≈ holdout(差异 < 0.1 composite),如某 split 显著低,SA 报告 Exp5'-MA 调查
2. **collapse rate**:STEP2 训练 val_min_d_mean=1.59,但 45% sample gate fail 说明仍有近距离对。具体 collapse rate(50%+ 原子重合)应 ≤ 1%
3. **shell-1 distance score**:proposal §3.1 verdict 阈值 ≥ 0.50。STEP2 训练曲线 val_shell_dist_loss 平台 ~ 3,对应 score 不一定到 0.5,SA 报告
4. **Exp5 v2 baseline cross-check**:旧的 SA-METRICS-V3 dry-run 100 在 v2 ckpt 上跑出 composite 0.0056-0.0062 / gate 5-11%,**SA 不需要重跑 v2 ckpt 做对比**,但要在 hand-back §9 引用旧数字作为基线

---

## §9 OPEN QUESTIONS(SA 不答,贴给 Exp5'-MA)

### Q1 — sample 脚本是否有 L=6 hardcode 残留

`step5_1_sample.py` 是 Exp4 时代写的(L=6),file guide §9 说改了一处但未必全。SA S2 改路径时 grep + view 全文,如发现 L=6 hardcode 报告 Exp5'-MA(可能要 STEP1-FIX-C 风格的小补丁)。

### Q2 — holdout split 在 step5_3 旧版可能不支持

旧 SA-METRICS-V3 写时只跑 val + test。SA S3 fork 时若发现旧版 hardcode `--split val|test`(没 holdout 选项),报告 Exp5'-MA 决议是否加,我倾向加(holdout 是真盲测,STEP4 final report 必引)。

### Q3 — predictions.pt 内的 L 字段

SA 在 sample 脚本里 SAVE L=20 进 .pt,**确认 SAVE 的 L 数值**。旧 v2 sample 脚本若 hardcode SAVE L=6,SA S2 verify 时必查。

---

## §10 你不做的事

- **STEP4 figure + final report v3**(Exp5'-MA 写)
- **修订 errata / proposal**(Exp5'-MA 工作)
- **Exp5 v2 ckpt 重跑 sample**(留作 forensic,STEP3 不需要)
- **元素-aware threshold ablation**(留 Exp5'')
- **resume from last.ckpt 续训**(errata 4 §6 排除)

---

## §11 工作哲学红线

1. 任何技术判断先列证据
2. 任何不确定 → 贴日志,不靠记忆
3. 小补丁也要贴 diff
4. 70% 上下文闸门是硬线,主动 hand-back
5. 不擅自调 7 项复合分公式 / min_d gate / 元素-aware
6. **跑 sample / 跑 metric 中途 ping Exp5'-MA 是好事**

---

*Exp5'-MA 撰写,2026-05-03,基于 STEP2 hand-back final + errata 4 §6 决议(用 last.ckpt)。SA-EXP5'-STEP3-SAMPLE 接此 launch note 启动 Exp5' sample + 7 项复合分一棒。*
