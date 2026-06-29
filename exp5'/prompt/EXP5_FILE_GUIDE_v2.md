# EXP5_FILE_GUIDE_v2.md
# Exp5 v2 + Exp5' File Guide — 完整索引(MA5 移交版)

> **撰写者**: MA5(移交 Exp5'-MA 前)
> **日期**: 2026-05-01
> **格式**: 继承 EXP4_FILE_GUIDE.md
> **取代**: EXP5_FILE_GUIDE_FINAL.md(归档为 _DEPRECATED,不更新)
> **用途**: Exp5'-MA 启动 + Exp5' 训练 + 后续 ExpN 接力的文件清单速查
> **更新**: Exp5'-MA 接手后任何新文件落盘需 append 到本文件 §X 末尾,不删历史条目

---

## §1 服务器目录结构总览

```
scsmlnprd02.its.auckland.ac.nz, /home/tcat/

├── conda_envs/mlff/                    ← Python 环境(所有 ExpN 共用)
│   └── bin/python                      ← 7 守卫包锁定版
│
├── diffcsp_exp4/                       ← Exp4 历史档案,只读
│   ├── code/...                        ← Exp4 代码(Exp5 backbone PYTHONPATH 末尾)
│   ├── data/                           ← ⭐ 共享 data 中心
│   │   ├── shell_boundaries.pkl        ← 387 MB, md5 cf2050e4..., Exp5' inject 进训练
│   │   ├── data_inventory_v2.csv       ← 主索引 75637 样本
│   │   ├── holdout_samples_v2.csv      ← 永久封存
│   │   ├── incompat_pool.csv           ← 永久封存
│   │   ├── spectra_*.pkl               ← 谱图数据
│   │   └── feff_features_*.pkl         ← FEFF 73d 特征
│   └── checkpoints/best-epoch366-val0.7300.ckpt   ← Exp4 best,Exp5' 物理对照可用
│
├── diffcsp_exp5/                       ← Exp5 v2 历史档案(MA5 移交时锁定状态)
│   ├── code/...                        ← v2 代码(SA1' MV-attention + SA3' step5_3)
│   ├── checkpoints/                    ← 4 个 v2 ckpt(2 active + 2 frozen)
│   ├── data/                           ← symlink → exp4/data/
│   ├── logs/                           ← v2 全部 log + SA-METRICS-V3 dry-run 输出
│   ├── sa0/                            ← SA0 multisample(独立工具)
│   └── EXP5_*_OUTPUT.md / *_HANDOFF.md ← 各 sub-agent 报告
│
└── diffcsp_exp5_prime/                 ← ⭐ Exp5' 工作目录(Exp5'-MA 新建)
    ├── code/...                        ← cp from exp5/code + Exp5' 改动
    ├── checkpoints/                    ← 空,等 Exp5' 训练
    ├── data/                           ← symlink → exp4/data/
    └── logs/                           ← 空,等 Exp5' 训练
```

---

## §2 Exp5 v2 服务器代码文件(Exp5'-MA 起点 — fork from 这里)

### 2.1 step2/

| 路径 | 行数 | 状态 | Exp5' fork 改动 |
|---|---|---|---|
| `exp5/code/step2/spectrum_encoder.py` | 127 | ✅ active | **不改**,直接 cp 进 exp5_prime |
| `exp5/code/step2/spectrum_encoder.py.bak_exp4` | 95 | 🔒 frozen | 不动 |

### 2.2 step3/

| 路径 | 行数 | 状态 | Exp5' fork 改动 |
|---|---|---|---|
| `exp5/code/step3/diffusion_w_type_xas.py` | 415 | ✅ active | **改**: 加 3 个 loss 函数(pairwise_min / shell_dist / shell_count),forward 调用,total_loss + 5 个 output 字段 |
| `.bak_exp4` | 415 | 🔒 frozen | 不动 |
| `exp5/code/step3/xas_local_dataset_v2.py` | 374 | ✅ active | **改**: 加 shell_boundaries inject 进 Data |
| `.bak_exp4` | — | 🔒 frozen | 不动 |
| `exp5/code/step3/xas_local_datamodule_v2.py` | 257 | ✅ active | **改**: collate 加 5 字段 |
| `.bak_exp4` | — | 🔒 frozen | 不动 |
| `exp5/code/step3/conf_xas/model/diffusion_xas.yaml` | 79 | ✅ active | **改**: 加 cost_pairwise_min / cost_shell_dist / cost_shell_count 三字段 |
| `.bak_exp4` / `.bak_sa2` / `.bak_v1` | — | 🔒 frozen | 不动 |
| `exp5/code/step3/forward_test.py` | 546 | ✅ active | **改**: Phase 6.7 测三新 loss |
| `.bak_exp4` | 365 | 🔒 frozen | 不动 |

### 2.3 step4/

| 路径 | 行数 | 状态 | Exp5' fork 改动 |
|---|---|---|---|
| `exp5/code/step4/step4_1_smoke_test.py` | 193 | ✅ active | **改**: 加 6 active loss 字段验证(原 4 + 新 3 - lattice = 6) |
| `exp5/code/step4/step4_2_train.py` | 300 | ✅ active | **改**: 去 last_ckpt 硬编码(from-scratch);加 ckpt selection callback;MAX_EPOCHS 写死 500 |
| `.bak_pre_milestone` / `.bak_pre_alpha` / `.bak_pre_resume` | — | 🔒 frozen | 不动 |

### 2.4 step5/

| 路径 | 行数 | 状态 | Exp5' fork 改动 |
|---|---|---|---|
| `exp5/code/step5/step5_1_sample.py` | 341 | ✅ active(SA3' fork)| **不改**,直接 cp |
| `exp5/code/step5/step5_2_compute_metrics.py` | 630 | ✅ active(SA1'+SA3')| **不改**,留作历史档案 |
| `.bak_pre_sa3` | — | 🔒 frozen | 不动 |
| `exp5/code/step5/step5_3_composite_score.py` | (SA-METRICS-V3 写)| ✅ active | **不改**,直接 cp,Exp5' 主指标 |

---

## §3 Exp5 v2 服务器 ckpt 文件

| 路径 | 大小 | epoch | val_loss | 状态 | 用途 |
|---|---|---|---|---|---|
| `exp5/checkpoints/epoch=529-val_loss=0.7003.ckpt` | 44 MB | 529 | 0.7003 | ✅ active | Exp5 v2 best,Exp5'' baseline 候选 |
| `exp5/checkpoints/last.ckpt` | 44 MB | 679 | n/a | ✅ active | SA2'' 训练自然终点 |
| `exp5/checkpoints/sa2_baseline_epoch484_val0.7065.ckpt.frozen` | 44 MB | 484 | 0.7065 | 🔒 frozen | SA2' best 永久 safety net |
| `exp5/checkpoints/sa2pp_resume_epoch529_val0.7003.ckpt.frozen` | 44 MB | 529 | 0.7003 | 🔒 frozen | SA2'' best 永久 safety net |

**Exp5' 起点**: 不 warm-start。`exp5_prime/checkpoints/` 空目录起。

---

## §4 Exp4 关键 data ground truth(Exp5' 必读)

| 路径 | 大小 | md5 | 用途 |
|---|---|---|---|
| `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl` | 387 MB | `cf2050e4899160f5698ad2481377e94c` | ⭐ Exp4 Step 2.5 per-sample shell ground truth,Exp5' 训练 inject 进 batch |

**Schema**(per-sample dict[sample_name] → 9 字段):
```
{
  "threshold":      float (= 0.1563 全样本一致),
  "distances":      array of float (该样本所有邻居距中心距离, 升序),
  "species_Z":      array of int (对应原子 Z),
  "shell_starts":   array of int (各 shell 起始 index),
  "shell_ends":     array of int (各 shell 结束 index, exclusive),
  "shell_n_atoms":  array of int (各 shell 原子数),
  "shell_of_atom":  array of int (每原子的 shell index),
  "eval_cutoff":    float (评估半径),
  "n_center_sites": int
}
```

**Exp5' inject 用 5 字段**:
- `true_shell1_d_mean = mean(distances[shell_of_atom == 0])`
- `true_shell2_d_mean = mean(distances[shell_of_atom == 1]) if shell_n_atoms[1] > 0 else 0.0`
- `has_shell2 = (shell_n_atoms[1] > 0)`
- `true_shell1_n = shell_n_atoms[0]`
- `true_shell2_n = shell_n_atoms[1] if has_shell2 else 0`

---

## §5 Exp5 v2 sample / metrics 输出(Exp5' 历史对照档案)

### 5.1 SA3' sample 输出

| 路径 | 大小 | 内容 |
|---|---|---|
| `exp5/code/step5/predictions_v2_val.pt` | 9.8 MB | SA2 epoch 484 baseline,7621 samples |
| `exp5/code/step5/predictions_v2_test.pt` | 5.8 MB | 同 test, 4481 samples |

⚠️ Exp5'-MA: 这是 SA2 baseline,**不是** SA2'' epoch 529。如要全量 v2 物理对照,SA-EXP5'-sample 阶段重 sample epoch 529 + Exp4 best ckpt 各一次。

### 5.2 SA-METRICS-V3 dry-run 输出(灾难锚点)

| 路径 | 内容 |
|---|---|
| `exp5/logs/composite_score_val_debug100.txt` | 100 样本 dry-run 主报告 — gate pass 5%, 复合 0.0056 |
| `exp5/logs/composite_score_test_debug100.txt` | 同 test — gate pass 11%, 复合 0.0062 |
| `exp5/logs/composite_score_per_sample_val_debug100.csv` | 100 行 |
| `exp5/logs/composite_score_per_sample_test_debug100.csv` | 100 行 |
| `exp5/logs/min_d_violations_val_debug100.csv` | ⭐ Exp5' λ schedule 设计依据 |
| `exp5/logs/min_d_violations_test_debug100.csv` | 同 test |

### 5.3 v2 数学评分(Exp5' final report 引用)

| 路径 | 内容 |
|---|---|
| `exp5/logs/v2_val_metrics.txt` / `_test.txt` | RMSD / pred_in_cutoff / Set-Level / Multiset / Collapse / Pos-by-pos |
| `exp5/logs/v2_val_per_sample.csv` / `_test.csv` | 7621 / 4481 行 |
| `exp5/logs/v2_projection_ablation_val.log` / `_test.log` | ⚠️ R_max=5.5 fallback,无诊断价值 |

### 5.4 Exp4 baseline(SA1' dry-run 重算)

| 路径 | 内容 |
|---|---|
| `exp5/logs/exp4_baseline_val_metrics.txt` / `_test.txt` | Exp4 4 metric 重算 |
| `exp5/logs/exp4_baseline_val_per_sample.csv` / `_test.csv` | Exp4 历史对照 |

---

## §6 Exp5' 服务器目录(Exp5'-MA 启动时新建)

```
/home/tcat/diffcsp_exp5_prime/
├── code/
│   ├── step2/spectrum_encoder.py            (cp from exp5)
│   ├── step3/
│   │   ├── diffusion_w_type_xas.py          (cp + 加 3 loss 函数)
│   │   ├── xas_local_dataset_v2.py          (cp + shell_boundaries inject)
│   │   ├── xas_local_datamodule_v2.py       (cp + 5 字段 collate)
│   │   ├── conf_xas/model/diffusion_xas.yaml(cp + 加 3 cost 字段)
│   │   └── forward_test.py                  (cp + Phase 6.7)
│   ├── step4/
│   │   ├── step4_1_smoke_test.py            (cp + 6 loss 字段验证)
│   │   └── step4_2_train.py                 (cp + ckpt selection callback,from-scratch)
│   └── step5/
│       ├── step5_1_sample.py                (cp from exp5,不改)
│       ├── step5_2_compute_metrics.py       (cp,作 v2 历史对照)
│       └── step5_3_composite_score.py       (cp from exp5,不改)
├── checkpoints/                              (空,等 Exp5' 训练)
├── data/                                     (symlink → /home/tcat/diffcsp_exp4/data/)
└── logs/                                     (空)
```

**新建命令**(Exp5'-MA 启动第 1 步):

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# 1. 新建目录
mkdir -p /home/tcat/diffcsp_exp5_prime/{code,checkpoints,logs}

# 2. cp Exp5 v2 code 树作 starting point
cp -r /home/tcat/diffcsp_exp5/code /home/tcat/diffcsp_exp5_prime/

# 3. data symlink
ln -s /home/tcat/diffcsp_exp4/data /home/tcat/diffcsp_exp5_prime/data

# 4. 验证
ls -la /home/tcat/diffcsp_exp5_prime/
ls -la /home/tcat/diffcsp_exp5_prime/code/
ls -la /home/tcat/diffcsp_exp5_prime/data/  # 应显示 → 链接到 exp4/data/

# 5. 验证 shell_boundaries.pkl 可见
ls -la /home/tcat/diffcsp_exp5_prime/data/shell_boundaries.pkl
md5sum /home/tcat/diffcsp_exp5_prime/data/shell_boundaries.pkl
# 期望 md5 = cf2050e4899160f5698ad2481377e94c

# 6. ckpt 目录空(from-scratch)
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/  # 期望: 空
```

---

## §7 7 守卫包(Exp5 全程不升级,Exp5' 沿用)

```
scikit-learn  1.7.2
numpy         2.2.6
scipy         1.15.3
pymatgen      2025.10.7
torch         2.4.1+cu124
pytorch-lightning 2.5.5
torch-scatter 2.1.2+pt24cu124
```

**Verify 命令**:
```bash
/home/tcat/conda_envs/mlff/bin/python -c "
import sklearn, numpy, scipy, pymatgen, torch, pytorch_lightning, torch_scatter
print(f'sklearn  {sklearn.__version__}')
print(f'numpy    {numpy.__version__}')
print(f'scipy    {scipy.__version__}')
print(f'pymatgen {pymatgen.__version__}')
print(f'torch    {torch.__version__}')
print(f'PL       {pytorch_lightning.__version__}')
print(f'tscat    {torch_scatter.__version__}')
"
```

---

## §8 PYTHONPATH 优先级(Exp5' 沿用 v2 设计)

```bash
# Exp5' 任何脚本启动必带:
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code
```

**顺序**:
1. exp5_prime step3/step2 在前 → shadow Exp4/Exp5 同名
2. Exp4 code 末尾 → 找 backbone(`diffcsp.pl_modules.cspnet`)

⚠️ **不放 /home/tcat/diffcsp_exp5/** — Exp5 v2 是历史档案,Exp5' 不沿用 v2 的 import 路径(避免误用 v2 的 .py 而不是 exp5_prime 改后的)。

---

## §9 Exp5'-MA 启动 verify 清单

第 1 件让用户跑(Exp5'-MA 写 SA-EXP5'-train launch note 之前):

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

echo "=========================== Exp5'-MA startup verify ==========================="

echo "--- (1) Conda env + Python ---"
/home/tcat/conda_envs/mlff/bin/python --version
which python  # 不用,以下都 absolute path

echo ""
echo "--- (2) 7 守卫包 ---"
/home/tcat/conda_envs/mlff/bin/python -c "
import sklearn, numpy, scipy, pymatgen, torch, pytorch_lightning, torch_scatter
print(f'sklearn  {sklearn.__version__}')
print(f'numpy    {numpy.__version__}')
print(f'scipy    {scipy.__version__}')
print(f'pymatgen {pymatgen.__version__}')
print(f'torch    {torch.__version__}')
print(f'PL       {pytorch_lightning.__version__}')
print(f'tscat    {torch_scatter.__version__}')
"

echo ""
echo "--- (3) Exp4 关键 ground truth ---"
ls -la /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# 期望 md5 = cf2050e4899160f5698ad2481377e94c, size = 387 MB

echo ""
echo "--- (4) Exp5 v2 ckpt 永久档案完整 ---"
ls -la /home/tcat/diffcsp_exp5/checkpoints/
md5sum /home/tcat/diffcsp_exp5/checkpoints/*.frozen

echo ""
echo "--- (5) Exp5 v2 SA-METRICS-V3 dry-run 输出存在 ---"
ls -la /home/tcat/diffcsp_exp5/logs/composite_score_*_debug100.txt
ls -la /home/tcat/diffcsp_exp5/logs/min_d_violations_*_debug100.csv

echo ""
echo "--- (6) Exp5' 工作目录尚不存在(待 Exp5'-MA 新建) ---"
ls -la /home/tcat/diffcsp_exp5_prime/ 2>&1
# 期望: ls: cannot access ... No such file or directory
# 这是预期,Exp5'-MA 第 1 件事 mkdir

echo ""
echo "--- (7) 磁盘 + GPU ---"
df -h /
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "--- (8) shell_boundaries schema 确认 ---"
/home/tcat/conda_envs/mlff/bin/python -c "
import pickle
with open('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl', 'rb') as f:
    sb = pickle.load(f)
print(f'type: {type(sb)}, n_samples: {len(sb)}')
sn = list(sb.keys())[0]
print(f'first sample_name: {sn!r}')
print(f'fields: {list(sb[sn].keys())}')
for k, v in sb[sn].items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, (list, tuple)):
        print(f'  {k}: len={len(v)}, head={v[:5] if len(v) > 0 else \"empty\"}')
    else:
        print(f'  {k}: {v}')
"
```

---

## §10 本地(用户 Windows 机)关键文档

### 10.1 Exp5 v2 时代(已完结)

| 文件 | 状态 | 用途 |
|---|---|---|
| EXP5_PROPOSAL_v2.md | ⚠️ outdated | v2 原 proposal |
| EXP5_PROPOSAL_v2_AMENDED_DEPRECATED.md | 🗄️ archived | MA5 早期 amend(被 v2 final report 取代) |
| EXP5_STEP1_PRIME_HANDOFF.md | ✅ active | SA1' handoff |
| EXP5_STEP1_PRIME_OUTPUT.md | ✅ active | SA1' 中期报告 |
| EXP5_SA2_PRIME_LAUNCH_NOTE.md | ✅ active | SA2' launch |
| EXP5_SA2_PRIME_OUTPUT.md | ✅ active | SA2' hand-back |
| EXP5_SA3_PRIME_LAUNCH_NOTE.md | ✅ active | SA3' launch |
| EXP5_SA3_PRIME_OUTPUT.md | ✅ active | SA3' hand-back |
| EXP5_SA_METRICS_V3_EARLY_HANDBACK.md | ✅ active | SA-METRICS-V3 早交回 |
| EXPERIMENT5_FINAL_REPORT_v1_DEPRECATED.md | 🗄️ archived | 被 v2 取代 |
| EXP5_FILE_GUIDE_FINAL_DEPRECATED.md | 🗄️ archived | 被本文件 v2 取代 |
| EXP5_MA2_HANDOFF_DEPRECATED.md | 🗄️ archived | 被 PRIME_MA_HANDOFF 取代 |

### 10.2 Exp5' 启动 4 件套(MA5 移交)

| 文件 | 状态 | 用途 |
|---|---|---|
| ⭐ EXP5_PRIME_PROPOSAL.md | ✅ active | Exp5' proposal,三件套 loss |
| ⭐ EXPERIMENT5_FINAL_REPORT_v2.md | ✅ active | Exp5 v2 final 历史档案 |
| ⭐ EXP5_FILE_GUIDE_v2.md(本文件) | ✅ active | 完整索引 |
| ⭐ EXP5_PRIME_MA_HANDOFF.md | ✅ active | Exp5'-MA 一文上手 |

### 10.3 Exp4 / 历史依赖

| 文件 | 状态 | 用途 |
|---|---|---|
| EXPERIMENT4_FINAL_REPORT.md | ✅ 历史 | Exp4 final |
| EXP4_FINAL_REPORT_ERRATA_2.md | ✅ 强制阅读 | _density_loss 塌缩根因 + Exp3 真实历史 |
| EXP4_ERRATA_2026-04-28.md | ✅ 历史 | Phase 6.5 状态修正 |
| EXP4_FILE_GUIDE.md | ✅ 历史格式参考 | 本文件继承 |
| EXPERIMENT2_FINAL_REPORT.md | ✅ 基础 | L=6 + min-image 设计起源 |
| Exp3 总结报告 | ✅ 基础 | head 失败教训 |

### 10.4 SA-METRICS-V3 产出代码(本地副本)

| 文件 | 状态 | 备注 |
|---|---|---|
| step5_3_composite_score.py | ✅ 本地 + 服务器 | 7 项复合分,Exp5' 沿用 |
| step5_3_smoke_test.py | ✅ 本地 + 服务器 | sanity 测试,Exp5' 改函数后跑 |

---

## §11 风险点速查(Exp5' 启动前必看)

| 风险 | 位置 | 处理 |
|---|---|---|
| MAX_EPOCHS 在 train.py 不在 yaml | step4_2_train.py line 83 | Exp5' 沿用,改 max_epochs 改 train.py |
| `_pairwise_min_distance_penalty` 重合数据 numerical | 新加 loss 函数 | gradient_clip=1.0 + monitor 前 5 epoch |
| gap 算法在 pred 上初期 ill-defined | shell_dist_loss / shell_count_loss | epoch 0-10 数值大或 0 是预期,不慌张 |
| shell_boundaries.pkl inject 进 batch 工程 | dataset/datamodule 改 | smoke test + Phase 6.7 forward_test catch |
| Best ckpt α/β/γ 调参 | train.py callback | 起步 0.2/0.5/0.3,如某项 dominate 调 weight |
| from-scratch 训练时长 ~32-40h | 整体 | Exp5'-MA 写 SA handoff 估时给 SA-EXP5'-train |

---

## §12 数据处理沿用一句话总结

| Exp5' 必须重做 | Exp5' 半沿用 | Exp5' 完全沿用 |
|---|---|---|
| 物理 loss 三件套(pairwise + shell_dist + shell_count)| shell_boundaries.pkl(从 evaluate-only 升级到 inject 进训练)| L=6 / split / FEFF / 7 守卫包 |
| Best ckpt selection 复合 criterion | step5_3_composite_score.py(主指标用,代码不改) | holdout / incompat_pool 永久封存 |
| Dataset / Datamodule shell 字段 inject | MV-attention + center_emb + cost_density 0.2 沿用 | Phase 6.5 SKIPPED-by-design 永久 |

详见 EXPERIMENT5_FINAL_REPORT_v2.md §4。

---

*MA5 撰写,2026-05-01,继承 EXP4_FILE_GUIDE 格式 + Exp5 v2 全部产出 + SA-METRICS-V3 产出 + Exp5' 工作目录指引。Exp5'-MA 接手后维护本文件。*
