# EXP5_SA3_PRIME_LAUNCH_NOTE.md
# Exp5 v2 SA3' Launch — Sampling + Metrics + Verdict Sub-Agent

> **From**: MA5 (Exp5 v2 Main Agent)
> **To**: SA3' (Sampling & Metrics sub-agent prime)
> **Date**: 2026-04-29
> **Status**: SA2' training complete, best ckpt landed, MA5 decided (I) accept epoch 484 ckpt. SA3' GO.

---

## §0 你是谁,做什么,读什么

你是 Exp5 v2 SA3'。SA2' 已完成 28h 训练,best ckpt 落地 `/home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt`(val_type_loss=0.00593,~200× 优于 v1 SA1 baseline)。**你的任务是 sample val + test → 算 v2 metrics → 出 Multiset F1 verdict**(~9h sample + 0.5h metrics + 0.5h 报告)。

**必读 2 份**:
1. 本 launch note(任务规格 + verdict 框架 + 红线)
2. `EXP5_SA2_PRIME_OUTPUT.md`(SA2' hand-back,**特别 §6** carry-over)

**不读**: SA1' OUTPUT 全文 / proposal v2 / Exp4 final report / LAUNCH_NOTE(SA2 的)。如要看背景,SA2 OUTPUT §6 已抽提关键信息。

**你的 5 个动作**(对应本 note §2-§6):
1. Pre-flight check
2. Verify `step5_1_sample.py` 存在(若不存在 SA3' 写,见 §3)
3. Sample val + test(各 ~ 4-5h,**严禁 sample holdout**)
4. Compute v2 metrics(Set-Level / Multiset-F1 / Collapse / 投影 ablation)
5. 出 verdict 报告 → MA5 review → 三分支决策

**红线**: 不动 holdout / 不改训练代码 / 不动 yaml / 不删 ckpt / 不擅自决续训。

---

## §1 SA3' 不变量(MA5 拍板,SA3' 不动)

### 1.1 Verdict 框架(主信号 = Multiset Macro-F1)

| 状态 | val Multiset-F1 | 意义 | MA5 决策 |
|---|---|---|---|
| 🟢 GREEN | > 0.20(2.40× over Exp4 0.0843) | MV-attention + density 减弱组合成功 | 启动 SA4'(figure + Exp6 决议)|
| ⚠️ AMBER | 0.10 - 0.20 | 学到东西但不充分 | **续训** 100-200 epoch 触发条件,改 max_epochs=700 |
| ❌ RED | < 0.10 | MV-attention 这条路不通 | 转 Exp6 方向(distance-aware loss / CFG) |

**Set-Level TypeAcc 副阈值**: > 0.40(Exp4 0.331 的 1.21×)。

**Geometry 防退化阈值**: RMSD < 1.5 / pred_in_cutoff > 18 / Collapse Ratio < 5%。任一退化即使 F1 涨也算"不健康成功"需要 MA5 review。

### 1.2 投影 Ablation(SA3' 必做,不只是计算 baseline)

`step5_2_compute_metrics.py` 已有 `compute_projection_ablation_rmsd` 函数(SA1' 写的)。SA3' 必跑 val + test 两个 split:

**判读**(投影前后 Δ RMSD):
- Δ < 0.05: 输出本身合理,MV-attention + density 减弱组合有效(预测原子已经在 R_max shell 内)
- Δ 0.05 - 0.10: 部分有效,部分塌缩到原点
- Δ > 0.10: 仍塌缩,与 errata 2 §1 描述的 Exp4 同模式 → 即使 F1 在 amber/green 区,也提示"是评估保护机制顶住的,不是物理顶住的"

**R_max 取值**: 从 `/home/tcat/diffcsp_exp5/data/shell_boundaries.pkl`(或同名 Exp4 文件)读训练真实距离 99 percentile,大致 5.5 Å 左右。如文件不存在,SA3' 用默认 5.5 Å。

### 1.3 Set-Level / Multiset / Collapse 算法已锁定

SA1' 在 `step5_2_compute_metrics.py` 实现了 4 个新函数(SA2' OUTPUT §6.2 沿用):
- `compute_set_level_typeacc(p, t)` + `_dataset` wrapper
- `compute_multiset_f1_macro(all_p, all_t)` — macro avg over **classes-in-true** only
- `compute_collapse_ratio(all_p_frac, all_t_frac, threshold=0.5)`
- `compute_projection_ablation_rmsd(all_p_frac, all_t_frac, R_max)`

SA3' **不许改算法**。如发现实现有 bug ping MA5 不自己 patch。

---

## §2 红线(任一触发立即停 + 报 MA5)

| 红线 | 出处 |
|---|---|
| ❌ **不 sample holdout**(holdout 在 v2 还没到解禁条件) | SA2 OUTPUT §6.5 |
| ❌ 不改 metrics 算法实现 | 本 note §1.3 |
| ❌ 不动 best/last ckpt(SA3' 只读) | SA2 OUTPUT §6.5 |
| ❌ 不改训练 yaml / 不动训练代码 | 同上 |
| ❌ 不擅自启动续训(amber 区触发后由 MA5 决) | SA2 OUTPUT §6.4 |
| ❌ 不动 holdout 任何文件(`holdout_samples_v2.csv` / `spectra_holdout.pkl` / `predictions_holdout.pt`) | EXP4_FILE_GUIDE §7 |
| ❌ 不修 Phase 6.5 hardcoded fp32(永久 SKIPPED-by-design) | 全程沿用 |
| ❌ 不并发跑 sample val 和 sample test(GPU 资源 / 显存峰值未知,串行稳) | MA5 谨慎 |
| ❌ 不在没确认 metrics 函数能 import 时启动 sample(白等 ~ 5h) | 工程常识 |

---

## §3 Pre-flight Checklist

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# 1. Verify best ckpt 在
BEST_CKPT=/home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt
ls -la $BEST_CKPT
# 期望: 文件存在,~ 44 MB

# 2. Verify SA3' deliverables 起点
ls -la /home/tcat/diffcsp_exp5/code/step5/step5_2_compute_metrics.py
# 期望: SA1' 写的 619 行版本

ls -la /home/tcat/diffcsp_exp5/code/step5/step5_1_sample.py
# 关键: 这个文件可能不存在!SA1' OUTPUT 没说写过。
# 如不存在: SA3' fork Exp4 模板,详见 §4

# 3. Verify Exp4 baseline 文件还在
ls -la /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt \
       /home/tcat/diffcsp_exp5/logs/exp4_baseline_test_metrics.txt
# 期望: SA1' dry-run 产出的 2 文件还在(SA3' 直接 diff 用)

# 4. Verify shell_boundaries.pkl(投影 ablation R_max 来源)
ls -la /home/tcat/diffcsp_exp5/data/shell_boundaries.pkl 2>/dev/null || \
  ls -la /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl 2>/dev/null
# 如都不存在: 投影 ablation 用默认 R_max=5.5 Å

# 5. GPU + env
/home/tcat/conda_envs/mlff/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
# 期望: CUDA: True, devices: ≥ 1

# 6. 磁盘
df -h /
# 期望: < 95%(predictions_*.pt 估计每个 ~ 200-500 MB)

# 7. metrics 函数可 import(不算 baseline,只测 import 链)
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  /home/tcat/conda_envs/mlff/bin/python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step5')
from step5_2_compute_metrics import (
    compute_set_level_typeacc,
    compute_multiset_f1_macro,
    compute_collapse_ratio,
    compute_projection_ablation_rmsd,
)
print('PASS: 4 metrics 函数可 import')
"
```

任一 fail → kill 流程,贴 stderr,不自己 troubleshoot。

---

## §4 `step5_1_sample.py` 处理

SA1' 没说写过,SA2' 也没碰这个。SA3' 上线先 `ls`:

### 4.1 如文件**存在**

直接用,跳到 §5。

### 4.2 如文件**不存在**

SA3' fork Exp4 模板:

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
ls -la /home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py
# 如 Exp4 有,cat 出来给 SA3' 看
cat /home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py
# 如 Exp4 也没有,可能叫 step5_sample.py / sample.py / 类似名
ls /home/tcat/diffcsp_exp4/code/step5/
```

SA3' fork 后必做改动:
- 路径 hardcode 改 Exp5: ckpt 路径 / data 路径 / 输出路径
- PYTHONPATH self-check 头部加(carry-over from SA1' 写法)
- 输出 `/home/tcat/diffcsp_exp5/code/step5/predictions_<split>.pt`,与 SA1' metrics 脚本期望路径对齐
- **断言**: 加载 ckpt 后 `assert model.cost_density == 0.2` 防误用 Exp4 ckpt

如 Exp4 也无现成 sample 脚本,SA3' 依据 SA2' OUTPUT §6.2 模板自写(基础流程: load ckpt → DataLoader val/test → 反扩散 1000 步 sample → 收集 pred_atom_types + pred_frac_coords → torch.save)。这种情况下 SA3' 写完先 dry-run 4 个 batch 验证 shape,再启动全 sample。

**SA3' 写新文件需 MA5 ack**(类比 SA2' 改 train.py 流程)。如要写,先贴 diff/full source 给 MA5 review。

---

## §5 Sample 命令(串行,不并发)

```bash
PY=/home/tcat/conda_envs/mlff/bin/python
BEST_CKPT=/home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt
cd /home/tcat/diffcsp_exp5/code/step5

# Sample val 先(~ 4-5h)
nohup env PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  $PY step5_1_sample.py --split val --ckpt $BEST_CKPT \
  > /home/tcat/diffcsp_exp5/logs/step5_sample_val.log \
  2> /home/tcat/diffcsp_exp5/logs/step5_sample_val.err &

echo $! > /home/tcat/diffcsp_exp5/logs/step5_sample_val.pid

# 头 5 min 守屏:确认进入 sample loop,无 OOM,GPU util 起来
tail -f /home/tcat/diffcsp_exp5/logs/step5_sample_val.log

# val 完成后,sample test(~ 3h,test 4,481 vs val 7,621)
$PY step5_1_sample.py --split test --ckpt $BEST_CKPT \
  2>&1 | tee /home/tcat/diffcsp_exp5/logs/step5_sample_test.log

# DO NOT run holdout. 红线 §2.
```

**头 5 min 守屏**:
- ckpt 加载成功(`assert cost_density == 0.2` 过)
- 第一个 batch 反扩散 1000 步开始(stdout 应有 `[Step 1000/1000]` 或类似 progress)
- `nvidia-smi` 显存占用 ~ 2-3 GB(单样本 1000 步轨迹)
- 无 NaN / Inf

任一异常 kill,贴 stderr。

---

## §6 Compute v2 Metrics

```bash
PY=/home/tcat/conda_envs/mlff/bin/python
cd /home/tcat/diffcsp_exp5/code/step5

PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  $PY step5_2_compute_metrics.py --split val --predictions ./predictions_val.pt \
  > /home/tcat/diffcsp_exp5/logs/step5_metrics_val.log 2>&1

PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  $PY step5_2_compute_metrics.py --split test --predictions ./predictions_test.pt \
  > /home/tcat/diffcsp_exp5/logs/step5_metrics_test.log 2>&1

# 主报告路径(SA1' 模板规定):
ls -la /home/tcat/diffcsp_exp5/code/step5/metrics_report_val.txt
ls -la /home/tcat/diffcsp_exp5/code/step5/metrics_report_test.txt
```

**投影 ablation 单跑**(metrics 主流程未必默认调用,SA3' 显式触发):

```bash
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  $PY -c "
import sys, torch, pickle
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step5')
from step5_2_compute_metrics import compute_projection_ablation_rmsd

# 读 predictions
preds = torch.load('./predictions_val.pt')
all_pred_frac = preds['pred_frac_coords']  # list of (20, 3)
all_true_frac = preds['true_frac_coords']

# R_max 来源
try:
    with open('/home/tcat/diffcsp_exp5/data/shell_boundaries.pkl', 'rb') as f:
        sb = pickle.load(f)
    R_max = sb.get('shell_99percentile', 5.5)
except:
    R_max = 5.5

result = compute_projection_ablation_rmsd(all_pred_frac, all_true_frac, R_max_angstrom=R_max)
print(f'R_max = {R_max:.2f} Å')
print(f'RMSD before projection: {result[\"rmsd_before\"]:.4f}')
print(f'RMSD after  projection: {result[\"rmsd_after\"]:.4f}')
print(f'Δ RMSD                 : {result[\"rmsd_delta\"]:.4f}')
print(f'Avg atoms projected per sample: {result[\"n_atoms_projected_avg\"]:.2f}')
" > /home/tcat/diffcsp_exp5/logs/step5_projection_ablation_val.log

# test 同样跑一遍
```

---

## §7 SA3' 输出报告 deliverable

`/home/tcat/diffcsp_exp5/EXP5_SA3_PRIME_OUTPUT.md`,内容:

| 节 | 内容 |
|---|---|
| §1 Sample 历程 | val/test wall time、batch 数、GPU util 平均、OOM/异常事件 |
| §2 v2 主指标(val + test) | RMSD / pred_in_cutoff / Set-Level TypeAcc / **Multiset Macro-F1** / Collapse Ratio,每个数 + 与 Exp4 baseline 对照(直接 diff 数字) |
| §3 投影 Ablation(val + test) | R_max 取值 / Δ RMSD / 每样本平均投影原子数 / 解读(Δ < 0.05 / 0.05-0.10 / > 0.10 三档) |
| §4 历史对照(虚假指标) | Position-by-position TypeAcc(仅供 Exp4 对照,标 [VIRTUAL]) |
| §5 Multiset F1 per-class detail | Top-10 元素类的 F1 + support_true + support_pred,**特别看 Z=8 (O) 是否系统性 over-predict** |
| §6 Verdict 自评 | SA3' 给的 🟢/⚠️/❌ 自评 + 理由(MA5 ratify) |
| §7 OPEN QUESTIONS | 如有 |
| §8 给 SA4'(or 续训)的 carry-over | predictions_*.pt 路径 / metrics_report 路径 / 关键数字 / 续训需要的话 yaml diff 模板 |

---

## §8 SA3' 三分支决策树(SA3' 知情,MA5 决)

SA3' 出 verdict 自评后,**SA3' 不下决断,只列三选项 + 推荐**。MA5 ratify。

### 🟢 GREEN(val Multiset-F1 > 0.20)
- SA4' 启动:6 figure + collapse 比例统计 + Exp6 方向决议
- SA3' 不续训
- 主信号成功

### ⚠️ AMBER(val Multiset-F1 ∈ [0.10, 0.20])
- 续训机制启动:
  ```yaml
  # diffusion_xas.yaml diff
  - max_epochs: 500
  + max_epochs: 700
  ```
- 重 launch 用 `Trainer.fit(ckpt_path=last.ckpt)` 续训 200 epoch
- ~ 11h 续训(200 epoch × 3.4 min)
- 续训完后 SA3'' 重 sample + metrics
- 如续训后跨 0.20 → SA4'
- 如续训后仍 < 0.20 → 转 Exp6

### ❌ RED(val Multiset-F1 < 0.10)
- MV-attention 这条路不通
- 转 Exp6:errata 2 §3.2 列的方向 4(distance-aware loss)或方向 9(CFG)
- 写 Exp6 proposal,基于 v2 lessons learned

---

## §9 速查红线总结

| | |
|---|---|
| ❌ 不 sample holdout(任何形式都不行,包括"快速看一下") | |
| ❌ 不改 metrics 算法实现(发现 bug 也 ping MA5) | |
| ❌ 不并发 val + test sample | |
| ❌ 不动 ckpt | |
| ❌ 不动 yaml / 不动训练代码 | |
| ❌ 不擅自启动续训 | |
| ❌ 不在没 import test 时启动 4-5h sample | |
| ✅ 任何 unexpected ping MA5,不自己改方案 | |

---

## §10 SA3' 启动 checklist 你的第一条回复

转给 SA3' 时,SA3' 第一条回复应:

1. 复述 5 个动作(pre-flight / sample script verify / sample val+test / metrics / verdict 报告)
2. 复述本 note §2 红线(至少 5 条)
3. 复述 §1.1 verdict 框架(GREEN/AMBER/RED 三阈值)+ §1.2 投影 ablation 三档解读
4. 列 pre-flight 7 条命令的执行计划
5. 列 `step5_1_sample.py` 不存在时的 fork 计划(从 Exp4 复制 + MA5 ack diff)
6. 估时(应 0.5h pre-flight + ~5h sample val + ~3h sample test + 0.5h metrics + 0.5h 报告 = ~9.5h)

---

*MA5 撰写,2026-04-29。基于 SA2' OUTPUT §6 carry-over + MA5 决议 (I) + Q3 阈值 0.20。*
