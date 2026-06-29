# EXP5_FILE_GUIDE_FINAL.md
# Exp5 v2 File Guide — 服务器 / 本地 / 脚本职责完整索引

> **撰写者**: MA5(移交 Exp5 MA2 前)
> **日期**: 2026-05-01
> **格式**: 继承 EXP4_FILE_GUIDE.md
> **用途**: Exp5 MA2 / SA-METRICS-V3 / Exp5' 启动时的文件清单速查
> **更新原则**: Exp5 MA2 接手后任何新文件落盘需 append 到本文件 §X 末尾,不删历史条目

---

## §1 服务器 (scsmlnprd02.its.auckland.ac.nz) 工作目录结构

```
/home/tcat/diffcsp_exp5/
├── code/
│   ├── step2/    ← SpectrumEncoder
│   ├── step3/    ← Diffusion model + Dataset/DataModule + yaml + forward_test
│   ├── step4/    ← Smoke test + train script
│   └── step5/    ← Sample + metrics(SA-METRICS-V3 写新文件落这)
├── checkpoints/  ← 4 个 ckpt(2 active + 2 frozen safety net)
├── data/         ← 软链接到 /home/tcat/diffcsp_exp4/data/
├── logs/         ← 全部 log(forward_test / smoke / train / sample / metrics)
├── sa0/          ← SA0 multisample(Exp5 范围外,不动)
└── EXP5_*_OUTPUT.md / EXP5_*_HANDOFF.md  (各 sub-agent 报告)

/home/tcat/diffcsp_exp4/data/
├── shell_boundaries.pkl  ← ⭐ Exp4 Step 2.5 ground truth (387 MB, md5 cf2050e4...)
├── ...其他 Exp4 data 文件(holdout / incompat_pool / 各 split CSV)
```

---

## §2 服务器代码文件清单(Exp5 v2 改动 / 新增,**不**含 Exp4 backbone)

### 2.1 step2/ (SpectrumEncoder)

| 路径 | 行数 | 状态 | 关键改动 |
|---|---|---|---|
| `step2/spectrum_encoder.py` | 127 | ✅ active | SA1' 改:删 cat→MLP fusion,加 MV-attention(MultiheadAttention num_heads=4 + learnable query + post-residual LN + center_emb cat at end);chi/feff 末端 Linear 升至 256d;output_dim=272 |
| `step2/spectrum_encoder.py.bak_exp4` | 95 | 🔒 frozen | Exp4 真版锚点(MA5 时代的 SpectrumEncoder,3-arg forward) |

### 2.2 step3/ (Model + Dataset + DataModule + yaml + forward_test)

| 路径 | 行数 | 状态 | 关键改动 |
|---|---|---|---|
| `step3/diffusion_w_type_xas.py` | 415 | ✅ active | SA1' 改:撤 TypeClassifierHead 类 + 实例化 + 3-mode loss + head_predict_types;保留 center embedding(SpectrumEncoder 4-arg)+ Patch 1 `F.one_hot(...).to(c0.dtype)` |
| `step3/diffusion_w_type_xas.py.bak_exp4` | 415 | 🔒 frozen | Exp4 真版锚点 |
| `step3/xas_local_dataset_v2.py` | 374 | ✅ active | v1 SA1 加 `_symbol_to_Z` lookup + `center_element_Z` 字段(Exp5 v2 完整保留) |
| `step3/xas_local_dataset_v2.py.bak_exp4` | — | 🔒 frozen | Exp4 真版 |
| `step3/xas_local_datamodule_v2.py` | 257 | ✅ active | v1 SA1 加 `center_element_Z` LongTensor (B,) collate;**v1→v2 命名 `.train_dataset` → `.train_ds`**(SA2' α' patch 修过 train.py) |
| `step3/xas_local_datamodule_v2.py.bak_exp4` | — | 🔒 frozen | Exp4 真版 |
| `step3/conf_xas/model/diffusion_xas.yaml` | 79 | ✅ active | SA1' 改:删 head 6 字段;加 mv_attention.num_heads=4 + residual_alpha=0.5;cost_density 0.5→0.2;latent_dim=272;decoder.latent_dim=528 |
| `step3/conf_xas/model/diffusion_xas.yaml.bak_exp4` | 50 | 🔒 frozen | Exp4 真版 |
| `step3/conf_xas/model/diffusion_xas.yaml.bak_sa2` | — | 🔒 frozen | v1 SA1 阶段 backup |
| `step3/conf_xas/model/diffusion_xas.yaml.bak_v1` | — | 🔒 frozen | v1 fallback 锚点 |
| `step3/forward_test.py` | 546 | ✅ active | SA1' 改 Phase 6.6 测 MV-attention(组件存在 / shape (4,272) / view-order invariance / cost_density yaml 加载);Phase 6.5 SKIPPED-by-design verbatim 保留 |
| `step3/forward_test.py.bak_exp4` | 365 | 🔒 frozen | Exp4 真版 |

### 2.3 step4/ (Smoke + Train)

| 路径 | 行数 | 状态 | 关键改动 |
|---|---|---|---|
| `step4/step4_1_smoke_test.py` | 193 | ✅ active | SA1' NEW(Exp4 没有);v1 4-mode → v2 1-mode 改写 |
| `step4/step4_2_train.py` | 300 | ✅ active | SA1' fork from Exp4 + SA2' α' patch(line 219-220 `.train_dataset` → `.train_ds`)+ MA5 SA2'' resume hardcode(line 224-225 last_ckpt 改硬编码 epoch=484 best path)+ MAX_EPOCHS line 83 = 700(SA2'' 续训改);MilestoneCallback class(epoch 200 marker,实际未触发,cosmetic) |
| `step4/step4_2_train.py.bak_pre_milestone` | — | 🔒 frozen | SA2' milestone-only baseline |
| `step4/step4_2_train.py.bak_pre_alpha` | — | 🔒 frozen | SA2' α' patch 前 baseline |
| `step4/step4_2_train.py.bak_pre_resume` | — | 🔒 frozen | SA2'' 续训 MAX_EPOCHS 改前 baseline(MA5 cp) |

### 2.4 step5/ (Sample + Metrics)

| 路径 | 行数 | 状态 | 关键改动 |
|---|---|---|---|
| `step5/step5_1_sample.py` | 341 | ✅ active | SA3' fork from Exp4(305 行);11 项 v2 surgery(C1-C11);硬阻断 holdout(扩展 error msg);--debug-n-batches + --debug-no-save flag;cost_density==0.2 断言 |
| `step5/step5_2_compute_metrics.py` | 630 | ✅ active | SA1' 写(619 行)+ SA3' 加 --debug-n-samples flag(+11 行);4 个 v2 算法函数(Set-Level / Multiset Macro-F1 / Collapse / Projection); ⚠️ Projection 函数 fallback R_max=5.5 Å bug(SA-METRICS-V3 必修) |
| `step5/step5_2_compute_metrics.py.bak_pre_sa3` | — | 🔒 frozen | SA1' 写完后 SA3' 改前 baseline |
| `step5/predictions_v2_val.pt` | 9.8 MB | ✅ active | SA3' sample 输出,7621 samples,from SA2 epoch 484 baseline |
| `step5/predictions_v2_test.pt` | 5.8 MB | ✅ active | SA3' sample 输出,4481 samples,from SA2 epoch 484 baseline |
| `step5/metrics_report_val.txt` | — | ✅ active | SA3' 算的 v2 主指标(Set-Level / Multiset / Collapse / RMSD / pred_in_cutoff)|
| `step5/metrics_report_test.txt` | — | ✅ active | 同上 |
| `step5/per_sample_metrics_val.csv` | 754 KB | ✅ active | 7621 行 |
| `step5/per_sample_metrics_test.csv` | 443 KB | ✅ active | 4481 行 |
| `step5/step5_3_composite_score.py` | TBD | ⏳ **SA-METRICS-V3 新写** | 7 项复合评分 + min_d 1.5 Å gate + per-sample shell_boundaries.pkl 读取 |

---

## §3 服务器 ckpt 文件清单

| 路径 | 大小 | 状态 | epoch | val_loss | 用途 |
|---|---|---|---|---|---|
| `checkpoints/epoch=529-val_loss=0.7003.ckpt` | 44 MB | ✅ **active best** | 529 | 0.7003 | SA-METRICS-V3 / Exp5' warm-start 起点 |
| `checkpoints/last.ckpt` | 44 MB | ✅ active | 679 | n/a | SA2'' 训练自然结束(early stop)|
| `checkpoints/sa2_baseline_epoch484_val0.7065.ckpt.frozen` | 44 MB | 🔒 **frozen** | 484 | 0.7065 | SA2' best 永久保留 safety net,SA3' predictions_v2_*.pt 来源 |
| `checkpoints/sa2pp_resume_epoch529_val0.7003.ckpt.frozen` | 44 MB | 🔒 **frozen** | 529 | 0.7003 | SA2'' best 永久保留 safety net |
| `best_checkpoint_path.txt` | — | ✅ active | n/a | n/a | 内容指向 epoch=529 ckpt |

⚠️ **PL ModelCheckpoint save_top_k=1 行为**: 训练新 best 出现时会删旧 active best。
**Exp5' 重训前**: Exp5 MA2 / SA-EXP5'-TRAIN 先 cp 一份当前 active best 到 .frozen,防 epoch 529 ckpt 被覆盖。

---

## §4 服务器数据文件清单

### 4.1 关键 Exp4 ground truth(Exp5 直接用)

| 路径 | 大小 | md5 | 用途 |
|---|---|---|---|
| `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl` | 387 MB | `cf2050e4899160f5698ad2481377e94c` | ⭐ **per-sample shell 边界 ground truth**,SA-METRICS-V3 必读取代 SA1' fallback |
| `/home/tcat/diffcsp_exp4/data/holdout_samples_v2.csv` | — | — | ❌ Exp5 v2 全程不动(holdout) |
| `/home/tcat/diffcsp_exp4/data/incompat_pool.csv` | — | — | ❌ 不动 |
| `/home/tcat/diffcsp_exp4/data/data_inventory_v2.csv` | 33.5 MB | — | 主索引 75637 样本 |

### 4.2 Exp5 软链接

```
/home/tcat/diffcsp_exp5/data/  ← symlink 全部到 /home/tcat/diffcsp_exp4/data/
```

不需要重新建立链接,**SA-METRICS-V3 读 `/home/tcat/diffcsp_exp5/data/shell_boundaries.pkl` 即等价**。

---

## §5 服务器 log 文件清单

### 5.1 SA1' 阶段(2026-04-28)
- `/home/tcat/diffcsp_exp5/logs/step1_forward_test_v2.log` — Phase 6.1-6.6 PASS
- `/home/tcat/diffcsp_exp5/logs/step1_smoke_v2.log` — 2 epoch × 10 batch SMOKE PASS
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt` — Exp4 baseline 重算,val
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_test_metrics.txt` — 同 test
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_val_per_sample.csv` — 7621 行
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_test_per_sample.csv` — 4481 行

### 5.2 SA2' 训练阶段(2026-04-28 → 04-29)
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2_stdout.log` — 28h 训练 stdout(epoch 5/8/394/484/500 关键行)
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2_stderr.log` — PL warning,无 traceback
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2.pid` — PID 记录

### 5.3 SA3' sample + metrics 阶段(2026-04-29 → 04-30)
- `/home/tcat/diffcsp_exp5/logs/step5_sample_val.log` / `.err` / `.pid`
- `/home/tcat/diffcsp_exp5/logs/step5_sample_test.log`
- `/home/tcat/diffcsp_exp5/logs/v2_val_metrics.txt` / `v2_val_per_sample.csv`
- `/home/tcat/diffcsp_exp5/logs/v2_test_metrics.txt` / `v2_test_per_sample.csv`
- `/home/tcat/diffcsp_exp5/logs/v2_projection_ablation_val.log` / `_test.log`
- `/home/tcat/diffcsp_exp5/logs/step5_dryrun.log` — dry-run 4 batch sample

### 5.4 SA2'' 续训阶段(2026-04-30)
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2_resume_stdout.log` — 11h 续训 stdout(epoch 484-679)
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2_resume_stderr.log` — best 后 150 epoch 没改进 + early stop signal
- `/home/tcat/diffcsp_exp5/logs/step4_train_v2_resume.pid`

### 5.5 SA-METRICS-V3 阶段(Exp5 MA2 即将)
- `/home/tcat/diffcsp_exp5/logs/composite_score_val.txt` — TBD
- `/home/tcat/diffcsp_exp5/logs/composite_score_test.txt` — TBD
- `/home/tcat/diffcsp_exp5/logs/min_d_violations_val.csv` — TBD ⭐ Exp5' 调 lambda 关键依据
- `/home/tcat/diffcsp_exp5/logs/min_d_violations_test.csv` — TBD

---

## §6 本地(用户 Windows 机)关键文档

### 6.1 Exp4 / 历史(用户机)

| 名称 | 路径 / 来源 |
|---|---|
| EXPERIMENT4_FINAL_REPORT.md | 用户机 + 服务器 `/home/tcat/diffcsp_exp4/...` |
| EXP4_FINAL_REPORT_ERRATA_2.md | ⭐ 强制阅读(`_density_loss` 塌缩根因 + Exp3 真实历史) |
| EXP4_ERRATA_2026-04-28.md | Phase 6.5 状态修正 |
| EXP4_FILE_GUIDE.md | Exp4 file guide(本文件继承格式) |
| EXPERIMENT2_FINAL_REPORT.md | L=6 + min-image 设计 |
| Exp3 总结报告 | head 失败教训 |

### 6.2 Exp5 v2 文档(本次产出)

| 名称 | 状态 |
|---|---|
| EXP5_PROPOSAL_v2.md | 原 proposal(MA5 早期写) |
| EXP5_PROPOSAL_v2_AMENDED.md | ⭐ MA5 临走 amend(加复合评分 + Exp5' 物理约束)|
| EXPERIMENT5_FINAL_REPORT_v1.md | ⭐ MA5 全 Exp5 时代 final report |
| EXP5_STEP1_PRIME_HANDOFF.md | SA1' handoff |
| EXP5_STEP1_PRIME_OUTPUT.md | SA1' 中期报告 |
| EXP5_SA2_PRIME_LAUNCH_NOTE.md | SA2' launch note |
| EXP5_SA2_PRIME_OUTPUT.md | SA2' hand-back |
| EXP5_SA3_PRIME_LAUNCH_NOTE.md | SA3' launch note |
| EXP5_SA3_PRIME_OUTPUT.md | SA3' hand-back |
| EXP5_FILE_GUIDE_FINAL.md | ⭐ 本文件 |
| EXP5_MA2_HANDOFF.md | ⭐ Exp5 MA2 启动包 |

### 6.3 不需要的文档(Exp5 MA2 不必读)

- v1 SA1+SA2 任何文档(已被用户清理或归档,Exp5 v2 不依赖)
- Exp3 中间 sub-agent handoff(只读总结报告)
- Exp4 中间 sub-agent handoff(只读 final report + 2 个 errata)

---

## §7 7 守卫包(Exp5 全程不升级)

```
scikit-learn  1.7.2
numpy         2.2.6
scipy         1.15.3
pymatgen      2025.10.7
torch         2.4.1+cu124
pytorch-lightning 2.5.5
torch-scatter 2.1.2+pt24cu124
```

**确认命令**(Exp5 MA2 任何疑问跑这):
```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
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

## §8 PYTHONPATH 优先级(任何脚本启动必带)

```bash
export PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code
```

**顺序解释**(继承自 v1 SA1 OUTPUT §5.6):
1. Exp5 step3/step2 在前 → shadow Exp4 同名文件(`spectrum_encoder.py` / `diffusion_w_type_xas.py` 等)
2. Exp4 code 末尾 → 找 backbone(`diffcsp.pl_modules.cspnet` 等 Exp5 不复制)

⚠️ **顺序错了的后果**: Python import 缓存拉到 Exp4 旧版,你以为在跑 v2 实际跑 Exp4 网络。
SA1' 设计 train.py 时加了 `assert "/diffcsp_exp5/" in module.__file__` 自检,
SA-METRICS-V3 / Exp5' 任何新脚本应继承同样自检模式。

---

## §9 Exp5 MA2 启动时的 verify 清单(防 v1→v2 transition 遗漏)

Exp5 MA2 接手后,**第 1 件事**让用户跑下面 verify 块,贴回输出:

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

echo "=========================== Exp5 MA2 startup verify ==========================="

echo "--- (1) Conda env ---"
which python
/home/tcat/conda_envs/mlff/bin/python --version

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
echo "--- (3) ckpt 状态 ---"
ls -la /home/tcat/diffcsp_exp5/checkpoints/

echo ""
echo "--- (4) shell_boundaries.pkl 完整性 ---"
ls -la /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# 期望 md5: cf2050e4899160f5698ad2481377e94c

echo ""
echo "--- (5) SA3' predictions 完整性 ---"
ls -la /home/tcat/diffcsp_exp5/code/step5/predictions_v2_*.pt
/home/tcat/conda_envs/mlff/bin/python -c "
import torch
for split in ['val', 'test']:
    p = torch.load(f'/home/tcat/diffcsp_exp5/code/step5/predictions_v2_{split}.pt',
                   map_location='cpu', weights_only=False)
    print(f'{split}: n_eff={p[\"n_effective\"]}, exp_version={p.get(\"exp_version\", \"missing\")}, '
          f'checkpoint={p[\"checkpoint\"].split(\"/\")[-1]}')
"

echo ""
echo "--- (6) 磁盘 ---"
df -h /

echo ""
echo "--- (7) GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

期望输出对照(从本 file guide):
- (1) `/home/tcat/conda_envs/mlff/bin/python` Python 3.11.x
- (2) 7 个版本严格匹配 §7 表
- (3) 4 个 ckpt 全在(2 active + 2 frozen)
- (4) md5 = cf2050e4899160f5698ad2481377e94c,大小 387 MB
- (5) val n_eff=7621, test n_eff=4481, exp_version='v2', checkpoint 含 'epoch=484' 或 '529'
- (6) 磁盘 < 95%(若 ≥ 95%,清 /tmp/diffcsp_cache/)
- (7) 至少 1 GPU 可用,memory.used < 1 GiB

---

## §10 Exp5 MA2 重要警告(从 SA-METRICS-V3 / Exp5' 之前必看)

1. **任何不确定的事 → 写脚本让用户跑 confirm,不靠记忆**(用户原话)。例如:
   - 文件存在/路径 → `ls -la /path/to/file`
   - schema 不确定 → load 后 print keys
   - md5 一致性 → `md5sum`
   - 函数签名 → `inspect.signature`

2. **Bak 文件全部 frozen,不删** — 历史可追溯。任何"看起来无用"的 .bak_* 文件删之前问用户。

3. **MAX_EPOCHS 在 train.py 不在 yaml** — line 83 写死(SA1' 决策)。Exp5' 改 max_epochs 时改 train.py。

4. **CosineAnnealingLR T_max=MAX_EPOCHS** — 改 max_epochs 会改 LR schedule。Exp5' 续训设计需考虑 LR warm restart 是否仍要(已有 epoch 529 vs Exp5' 新 max_epochs 729 的 cosine 曲线)。

5. **predictions_v2_*.pt 是 SA2 baseline(epoch 484),不是 SA2''(epoch 529)** — SA-METRICS-V3 算的是 SA2 baseline 的复合分。如要算 SA2'' 复合分,先重 sample(~3.5h)。

6. **shell_boundaries.pkl per-sample lookup** — sample_name 形如 `mp-555067__mp-...-EXAFS-As-K`,SA-METRICS-V3 实施时 load 后先 print `list(d.keys())[:3]` 和某 sample 的字段验证 schema。

7. **Holdout 全程不动** — 即使 SA-METRICS-V3 也只用 val + test。Holdout 在 Exp5' 完成 + Exp5 MA2 ratify 后才解禁。

---

## §11 Exp5 MA2 / SA-METRICS-V3 推荐 starting commands

```bash
# Verify(本 file guide §9)
ssh tcat@scsmlnprd02.its.auckland.ac.nz
# (paste §9 verify block)

# 看 SA1' 投影 ablation 的 R_max bug 真实代码
sed -n '290,340p' /home/tcat/diffcsp_exp5/code/step5/step5_2_compute_metrics.py
# 或 grep
grep -n "R_max\|shell_boundaries\|99percentile\|5\.5" /home/tcat/diffcsp_exp5/code/step5/step5_2_compute_metrics.py

# Load shell_boundaries.pkl 看 schema
/home/tcat/conda_envs/mlff/bin/python -c "
import pickle
with open('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl', 'rb') as f:
    sb = pickle.load(f)
print(f'type: {type(sb)}, n_samples: {len(sb)}')
sn = list(sb.keys())[0]
print(f'first sample_name: {sn!r}')
print(f'fields: {list(sb[sn].keys())}')
print(f'sample value:')
for k, v in sb[sn].items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, (list, tuple)):
        print(f'  {k}: len={len(v)}, head={v[:5] if len(v) > 0 else \"empty\"}')
    else:
        print(f'  {k}: {v}')
"

# Load predictions_v2_val.pt 看 schema
/home/tcat/conda_envs/mlff/bin/python -c "
import torch
p = torch.load('/home/tcat/diffcsp_exp5/code/step5/predictions_v2_val.pt',
               map_location='cpu', weights_only=False)
print('keys:', list(p.keys()))
print(f'n_effective: {p[\"n_effective\"]}')
print(f'first sample_name: {p[\"sample_name\"][0]!r}')
print(f'pred_frac_coords[0].shape: {p[\"pred_frac_coords\"][0].shape}')
print(f'pred_atom_types[0].shape: {p[\"pred_atom_types\"][0].shape}')
print(f'pred_atom_types[0][:5]: {p[\"pred_atom_types\"][0][:5]}')
"
```

跑完三块,Exp5 MA2 就有 SA-METRICS-V3 实施的全部上下文。

---

*MA5 撰写,2026-05-01。继承 EXP4_FILE_GUIDE 格式,加 Exp5 v2 全部产出 + ckpt + log。Exp5 MA2 接手后维护本文件。*
