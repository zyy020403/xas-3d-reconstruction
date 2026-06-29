# EXP4_FILE_GUIDE.md
# Exp4 项目文件指南 — 本地 / 服务器 / 取用方法

> **撰写者**: Main Agent 5(Exp4 完结后补写)
> **日期**: 2026-04-28
> **目的**: 任何后续 agent(Exp5 / Exp6 / 未来 Sub-Agent)启动后第一时间能定位每个文件,知道在哪取、怎么取。
> **本文档定位**: EXPERIMENT4_FINAL_REPORT.md §9 的展开,严格区分**本地 (Windows)** vs **服务器 (Linux scsmlnprd02)**。
> **使用场景**: Sub-Agent 在对话窗口里说"我需要 X 文件",用户对着本指南找路径 + 取用命令。

---

## §0 前提: 物理位置与权限

### 0.1 三个物理位置

| 位置 | 含义 | 谁能直接读? |
|------|------|---|
| **L (Local)** | 用户 Windows 工作站 `C:\Users\T-Cat\Desktop\DiffCSP-main\` | 用户 + 用户上传给 agent 之后的 agent |
| **S (Server)** | 远程 Linux `scsmlnprd02.its.auckland.ac.nz`,目录 `/home/tcat/diffcsp_exp4/` | 用户 ssh + 服务器内运行的脚本 |
| **C (Chat-uploaded)** | 用户上传到 agent 对话窗口的文件,在 agent 沙盒里 `/mnt/user-data/uploads/` | 当前对话窗口的 agent |

### 0.2 关键认知

**Sub-Agent 没有 ssh**。Sub-Agent 看不到 S 上的文件,除非:
- 用户主动 ssh 跑命令 → 把输出 / 文件内容贴到对话(L→C 或 S→C 路径)
- 用户 scp 服务器文件到 Windows → 上传到对话(S→L→C 路径)

**用户不需要知道 sub-agent 沙盒**。用户管 L 和 S,sub-agent 管 C 内可见的内容。

### 0.3 通用 ssh 取文件三招

后续每个 S 上的文件我都给具体取用命令,但通用三招记住:

```bash
# 招 1: 在服务器内 cat 后复制粘贴(适合 < 1000 行 文本文件)
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cat /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py
# 复制 stdout 到对话窗口

# 招 2: scp 下载到 Windows(适合大文件 / 二进制)
# Windows PowerShell:
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py C:\Users\T-Cat\Desktop\
# 然后从 Windows 上传到对话

# 招 3: 服务器内 cat + tee(查看 + 同时存本地 PowerShell 的临时文件)
ssh tcat@scsmlnprd02.its.auckland.ac.nz "cat /home/tcat/diffcsp_exp4/data/data_inventory_v2.csv | head -20"
# 适合"先 sanity 后决定要不要全文件"
```

---

## §1 文档类(L 本地全套 + C 部分上传)

这些是 markdown / txt 文档,主要在 **L 本地的 prompt 文件夹**(用户已经把它们一直在维护)。Exp5 启动时用户会上传部分到 C(对话沙盒)。

### 1.1 必传给 Exp5 Main Agent 的 5 份(从 L 上传到 C)

| 文档 | 本地路径 (L) | 大小 | 用途 |
|------|-------------|------|------|
| `EXPERIMENT4_FINAL_REPORT.md` | `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\prompt\` 或用户存档目录 | 46 KB | Exp5 主参考 |
| `EXPERIMENT2_FINAL_REPORT.md` | `C:\...\experiment2\prompt\EXPERIMENT2_FINAL_REPORT.md` | 14 KB | Exp2 baseline 对照 |
| `EXP4_PROPOSAL_v2.md` | `C:\...\experiment4\prompt\EXP4_PROPOSAL_v2.md` | 17 KB | 不变量继承 |
| `EXP4_FILE_INVENTORY.md` | `C:\...\experiment4\prompt\EXP4_FILE_INVENTORY.md` | 13 KB | MA2 写的旧版数据清单(本指南新版替代) |
| **`EXP4_FILE_GUIDE.md`(本文档)** | 用户从我这里下载到 `C:\...\experiment4\prompt\` 后再上传 | ~30 KB | **新增**,Exp5 SA 主参考 |

### 1.2 Exp5 选传(根据 SA 任务定)

| 文档 | 本地路径 (L) | 何时传 |
|------|-------------|---|
| `EXP4_STEP6_STEP6AGENT_FINAL_REPORT.md` | `C:\...\experiment4\prompt\` | Exp5 想看 6 张 figure 数字细节 |
| `EXP4_STEP5AGENT_FINAL_REPORT.md` | 同上 | Exp5 想 reproduce Step 5 评估流程 |
| `EXP4_STEP4_SUBAGENT5_INTERIM_REPORT.md` | 同上 | Exp5 想看 Step 4 训练曲线 / Phase 4.6 修复细节 |

### 1.3 Exp5 不要传

| 文档 | 为什么 |
|------|---|
| `EXP4_MAINAGENT1/2/3/4_HANDOFF.md` | MA5 final report 已浓缩 |
| `EXP4_PROGRESS_LOG.md` | Step 1/2/2.5 历史细节,Exp5 不直接用 |
| `EXP4_STEP*_HANDOFF.md` | handoff 是给 sub-agent 的,Main Agent 不需要 |
| `exp2tree.txt` | 信息已在 final report §9.1 |

---

## §2 代码类(S 服务器主体 + L 本地 Exp2 fork)

代码主体在服务器(被 SA 跑)。用户本地有 Exp2 时代的版本作 reference。**重大警告**: L 上 Exp2 fork 的代码 ≠ S 上 Exp4 改造后的代码。**绝不能用 Exp2 版替代 Exp4 版**(否则 ckpt 加载 shape mismatch / Phase 4.6 silent drop 行为消失等)。

### 2.1 Exp4 当前用代码(S 服务器,Sub-Agent 取用)

| 文件 | 服务器绝对路径 (S) | 当前版本说明 | 取用方法 |
|------|-------------------|------------|---------|
| **`xas_local_dataset_v2.py`** ⭐ | `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py` | Phase 4.6 silent-drop 版,Exp5 复用,SA1 加 `center_element_Z` 字段 | ssh cat 或 scp |
| **`xas_local_datamodule_v2.py`** ⭐ | `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py` | 含 `xas_collate_fn_v2`(None-filter),Exp5 复用 | ssh cat 或 scp |
| **`spectrum_encoder.py`** | `/home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py` | 5 处 73→74,Exp5 SA1 加 center embedding 在此 | ssh cat 或 scp |
| **`diffusion_w_type_xas.py`** ⭐ | `/home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py` | line 108 `feat_dim=74`,Exp5 SA1 加 TypeClassifierHead 在此 | ssh cat 或 scp |
| **`diffusion_xas.yaml`** ⭐ | `/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml` | line 18 `feat_dim=74`,**所有 hyperparam 来源,SA1 必须按此版改造**(否则 ckpt warm-start 报 shape) | ssh cat 或 scp |
| `forward_test.py` | `/home/tcat/diffcsp_exp4/code/step3/forward_test.py` | Phase 6.1-6.5 sanity test,SA1 改造架构后必须 5/5 PASS | ssh cat 或 scp |
| `step4_1_smoke_test.py` | `/home/tcat/diffcsp_exp4/code/step4/step4_1_smoke_test.py` | Step4Agent 写的 smoke 模板,Exp5 SA1 用作模板改 | ssh cat 或 scp |
| `step4_2_train.py` | `/home/tcat/diffcsp_exp4/code/step4/step4_2_train.py` | Exp5 SA2 训练参考 | SA2 用,SA1 不用 |
| `step5_1_sample.py` | `/home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py` | Phase 5b 直接构造 dataset 版,Exp5 SA3 评估参考 | SA3 用 |
| `step5_2_compute_metrics.py` | `/home/tcat/diffcsp_exp4/code/step5/step5_2_compute_metrics.py` | Hungarian metrics,Exp5 SA3 复用 | SA3 用 |
| `step6_visualize.py` | `/home/tcat/diffcsp_exp4/code/step6/step6_visualize.py` | 6 figure render,Exp5 SA4 fork 改 | SA4 用 |

#### Exp5 SA1 的 P0+P1 取文件命令一次发(用户复制粘贴跑)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp4/code

# 6 个 SA1 必拿文件,逐个 cat
echo "=== xas_local_dataset_v2.py ==="
cat step3/xas_local_dataset_v2.py
echo ""
echo "=== xas_local_datamodule_v2.py ==="
cat step3/xas_local_datamodule_v2.py
echo ""
echo "=== spectrum_encoder.py ==="
cat step2/spectrum_encoder.py
echo ""
echo "=== diffusion_w_type_xas.py ==="
cat step3/diffusion_w_type_xas.py
echo ""
echo "=== diffusion_xas.yaml ==="
cat step3/conf_xas/model/diffusion_xas.yaml
echo ""
echo "=== step4_1_smoke_test.py ==="
cat step4/step4_1_smoke_test.py
```

整段输出复制到对话给 SA1。如果输出过长(>200KB):

```bash
# 替代: scp 下载到 Windows 再上传
mkdir C:\Users\T-Cat\Desktop\exp4_fork
cd C:\Users\T-Cat\Desktop\exp4_fork
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py .
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py .
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py .
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py .
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml .
scp tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp4/code/step4/step4_1_smoke_test.py .
# 然后从 exp4_fork 文件夹拖文件到对话
```

### 2.2 Exp4 .bak 备份(S 服务器,只读历史)

**Sub-Agent 不主动用,只在回滚时用**。用户 ssh 检查或 scp 都行。

| 文件 | 服务器路径 | 何时用 |
|------|-----------|---|
| `xas_local_dataset_v2.py.bak_phase46` | `/home/tcat/diffcsp_exp4/code/step3/` | Phase 4.6 修复前的 raise 版,emergency 回滚 |
| `xas_local_datamodule_v2.py.bak_phase46` | 同上 | 无 None-filter 版 |
| `forward_test.py.bak3` | `/home/tcat/diffcsp_exp4/code/step3/` | fp32 改前的 4/5 PASS 版,emergency 回滚 |
| `step5_1_sample.py.bak_phase5` | `/home/tcat/diffcsp_exp4/code/step5/` | Phase 5a 含 holdout 闸门版 |
| `diffusion_w_type_xas.py.bak` | `/home/tcat/diffcsp_exp4/code/step3/` | Sub-Agent 4 早期备份 |
| `diffusion_xas.yaml.bak` | `/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/` | Sub-Agent 4 改 73→74 前的版本 |
| 其他 `.bak`/`.bak2` | 同 step3/ 目录散落 | 历史归档 |

### 2.3 Exp2 fork(L 本地,**仅作 reference,不复用**)

| 文件 | 本地路径 (L) | 与 Exp4 关系 |
|------|-------------|---|
| `xas_local_dataset_L6.py` | `C:\...\experiment2\step3\xas_local_dataset_L6.py` | Exp2 dataset,Exp4 v2 的源头,**架构相同**但缺 88 元素支持 / silent drop / `eval_cutoff` 字段 |
| `spectrum_encoder.py` | `C:\...\experiment2\step2\spectrum_encoder.py` | feat_dim=73,Exp4 改成 74 |
| `diffusion_w_type_xas.py` | `C:\...\experiment2\step3\diffusion_w_type_xas.py` | Exp2 Step4c 版,坐标系已 [-0.5, 0.5] |
| `diffusion_xas.yaml` | `C:\...\experiment2\step3\conf_xas\model\diffusion_xas.yaml` | feat_dim=73,**没 cost_density 字段**,与 Exp4 yaml 不同 |
| Step 1-6 全套脚本 | `C:\...\experiment2\step1\` 到 `step6\` | Exp4 fork 来源,Exp5 SA 参考但**不直接复用** |

**Exp5 SA1 红线**: 上面这些 L 上的 Exp2 文件**只能作 reference**(对照看历史结构),**绝不能上传给 SA1 当 Exp4 当前版用**。已有 sub-agent 因为这个犯过错。

---

## §3 数据类(S 服务器主体 + L 本地原始)

### 3.1 Exp4 训练 / 评估用数据(S 服务器)

| 文件 | 服务器路径 (S) | 大小 | 用途 |
|------|---------------|------|------|
| `data_inventory_v2.csv` | `/home/tcat/diffcsp_exp4/data/data_inventory_v2.csv` | 33.5 MB | 主索引,75637 行 × 15 列 |
| `train_samples_v2.csv` | `/home/tcat/diffcsp_exp4/data/train_samples_v2.csv` | 3.3 MB | 60507 train sample list |
| `val_samples_v2.csv` | 同目录 | 0.42 MB | 7624 val |
| `test_samples_v2.csv` | 同目录 | 0.24 MB | 4481 test |
| **`holdout_samples_v2.csv`** ⚠️ | 同目录 | 0.17 MB | 3025 holdout,**Exp5 训练期不可读** |
| `feff_features_imputed.pkl` | 同目录 | 40.3 MB | (128382, 74) float32 DataFrame |
| `feff_feature_scaler.pkl` | 同目录 | 1.6 KB | RobustScaler 对象 |
| `feff_feature_names.txt` | 同目录 | 1.0 KB | 74 个特征名 |
| `spectra_train.pkl` | 同目录 | 148.4 MB | xmu(150)+ chi1(200)预处理 |
| `spectra_val.pkl` | 同目录 | 18.7 MB | val |
| `spectra_test.pkl` | 同目录 | 11.1 MB | test |
| **`spectra_holdout.pkl`** ⚠️ | 同目录 | 7.4 MB | **Exp5 训练期不可读** |
| `shell_boundaries.pkl` | 同目录 | 369.5 MB | Step 5 Tier 评估必读 |
| **`incompat_pool.csv`** 🔒 | 同目录 | 3.3 MB | **全程封存,任何 agent 不读** |
| `site_equivalence_tag.csv` | 同目录 | 9.5 MB | 归档,默认不读 |
| `MP_all_POSCAR_flat/` | `/home/tcat/diffcsp_exp4/data/MP_all_POSCAR_flat/` | dir | POSCAR 文件目录 |

#### Sanity check 数据用 schema(SA 写代码前先 sample 看)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp4/data

# 看 inventory schema(用户跑,把 head 输出贴给 SA)
head -3 data_inventory_v2.csv
wc -l data_inventory_v2.csv  # 应是 75638(75637 + header)

# spectra_*.pkl 的 schema(用 python 看 keys)
python -c "
import pickle
with open('spectra_val.pkl', 'rb') as f:
    s = pickle.load(f)
print('keys:', list(s.keys()))
print('xmu shape:', s['xmu'].shape, s['xmu'].dtype)
print('chi1 shape:', s['chi1'].shape)
print('N samples:', len(s['sample_names']))
print('first 3 names:', s['sample_names'][:3])
"

# shell_boundaries 的单样本 schema
python -c "
import pickle
with open('shell_boundaries.pkl', 'rb') as f:
    sb = pickle.load(f)
k = list(sb.keys())[0]
print('one sample key:', k)
print('one sample fields:', list(sb[k].keys()))
print('eval_cutoff:', sb[k]['eval_cutoff'])
"
```

### 3.2 Exp4 ckpt(S 服务器)⭐⭐ Exp5 fine-tune 起点

| 文件 | 服务器路径 (S) | 大小 | 用途 |
|------|---------------|------|------|
| **`best-epoch366-val0.7300.ckpt`** ⭐⭐ | `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` | 40.2 MB | **Exp5 SA2 warm start 起点** |
| `last.ckpt` | `/home/tcat/diffcsp_exp4/checkpoints/last.ckpt` | 40.2 MB | epoch 395 末态,备用 |
| `_smoke/` | `/home/tcat/diffcsp_exp4/checkpoints/_smoke/` | dir | Phase 4.2 smoke test 残留,可清理 |
| `best_checkpoint_path.txt` | `/home/tcat/diffcsp_exp4/best_checkpoint_path.txt` | 单行 | 内容 = best ckpt 绝对路径 |

**Exp5 SA2 warm-start 命令模板**:
```python
import torch
ckpt = torch.load(
    "/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt",
    map_location="cpu", weights_only=False
)
# Exp5 加了 head + center_emb,strict=False 允许 missing keys
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("missing (新加的层):", missing)        # 期望: head.* / center_emb.* 等
print("unexpected (旧的不在新模型):", unexpected)  # 期望: 空,或只有 lattice_scaler
```

**红线**: SA2 必须**打印 missing 和 unexpected 给用户看**,如果 unexpected 非空(不是空 list 也不是 lattice_scaler),说明 hyperparam 不对齐,**立刻停**,这意味着 SA1 改架构时漂离了 Exp4 yaml 的 backbone 参数。

### 3.3 Exp4 评估产出(S 服务器,Step 6 / Exp5 ablation 用)

| 文件 | 服务器路径 (S) | 大小 | 用途 |
|------|---------------|------|------|
| `predictions_val.pt` | `/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt` | 9.84 MB | Step 6 fig3/fig5 数据,Exp5 ablation reference |
| `predictions_test.pt` | 同目录 | 5.79 MB | |
| `predictions_holdout.pt` | 同目录 | ~4 MB | |
| `per_sample_metrics_val.csv` | 同目录 | ~600 KB | 7621 行 × 7 列,Step 6 主输入 |
| `per_sample_metrics_test.csv` | 同目录 | ~350 KB | |
| `per_sample_metrics_holdout.csv` | 同目录 | ~240 KB | |
| `metrics_report_*.txt` | 同目录 | ~3 KB 各 | 人类可读汇总 |

#### predictions_*.pt schema(SA 拿前要确认)

```python
preds = torch.load("predictions_val.pt", weights_only=False)
# Step5Agent format,dict-of-lists:
# preds["mp_id"]:              list[str], len = N
# preds["sample_name"]:        list[str], len = N  (与 mp_id 对齐)
# preds["pred_frac_coords"]:   list of (20, 3) numpy/tensor
# preds["pred_atom_types"]:    list of (20,) int
# preds["true_frac_coords"]:   list of (20, 3)
# preds["true_atom_types"]:    list of (20,)
# preds["lengths"]:            list of (1, 3) = [[6,6,6]]
# preds["eval_cutoff"]:        list of float
```

### 3.4 Step 6 figure(S 服务器)

| 文件 | 服务器路径 (S) | 大小 | 用途 |
|------|---------------|------|------|
| `fig1_rmsd_distribution.png` | `/home/tcat/diffcsp_exp4/code/step6/figures/` | ~200 KB | val/test/holdout RMSD 直方图 |
| `fig2_typeacc_distribution.png` | 同目录 | ~200 KB | TypeAcc 直方图 |
| `fig2b_typeacc_by_tier.png` | 同目录 | ~200 KB | **Headline figure** |
| `fig3_structure_comparison.png` | 同目录 | ~500 KB | 6 panel 3D |
| `fig4_rmsd_vs_typeacc.png` | 同目录 | ~150 KB | scatter |
| `fig5_typeacc_by_rank.png` | 同目录 | ~150 KB | by rank,Exp5 重画对照 |

#### Exp5 何时需要这些 PNG

- 写 EXP5_PROPOSAL.md 时贴图引用 → 用户从 S scp 到 L,再上传到对话
- SA4 重画 fig5 时对照"Exp4 rank 1 = 0.243 vs Exp5 后是否爬升" → 看本指南 §3.3 数字直接对照,不必看图

### 3.5 Exp4 训练 log(S 服务器,通常不需 Sub-Agent 看)

| 文件 | 服务器路径 (S) | 用途 |
|------|---------------|------|
| `step4_train_v2_stdout.log` | `/home/tcat/diffcsp_exp4/logs/` | Phase 4.6 修复后的训练 log,含每 epoch val_loss |
| `step4_train_v2_stderr.log` | 同目录 | PL 输出 |
| `step4_red_light_2026-04-26_*.log` | 同目录 | 第一次启动红灯归档 |
| `step5_*.log`、`step6_*.log` | 同目录 | 评估 / 可视化 log |

### 3.6 L 本地原始数据(用户管,Exp5 不用)

Exp4 的 Step 1/2/2.5 输入数据在 L:
- `C:\...\DiffCSP-main\experiment4\data\MP_all_EXAFS_only_chi_csv\` chi.csv 源数据
- `C:\...\experiment4\data\MP_all_EXAFS_only_csv\` xmu.csv 源数据
- `C:\...\experiment4\data\POSCAR_zip\` POSCAR 源
- `C:\...\experiment4\step1\` Step 1 中间产出
- `C:\...\experiment4\step2\` Step 2 spectra pkl(已 scp 到 S)
- `C:\...\experiment4\step2_5\` Step 2.5 v2 split 产出(已 scp 到 S)

**Exp5 不需要回到 L 的原始数据**——所有 Step 1/2/2.5 产物已经在 S 上 ready,Exp5 直接复用 S 数据即可。

---

## §4 环境信息(S 服务器)

### 4.1 Conda env

```bash
which python          # /home/tcat/conda_envs/mlff/bin/python
python --version      # Python 3.10.x
```

**警告**: 服务器上还有 `(jhub_env)` 一个虚假 prompt,Step5Agent §6.1 教训。**任何 SA 跑命令必须用绝对路径** `/home/tcat/conda_envs/mlff/bin/python`,不要依赖 PATH。

### 4.2 守卫包(7 + 18 子依赖)

| 包 | 期望版本 | Exp5 不可升级 |
|---|---|---|
| scikit-learn | 1.7.2 | ✓ |
| numpy | 2.2.6 | ✓ |
| scipy | 1.15.3 | ✓ |
| pymatgen | 2025.10.7 | ✓ |
| torch | 2.4.1+cu124 | ✓ |
| pytorch-lightning | 2.5.5 | ✓ |
| torch-scatter | 2.1.2+pt24cu124 | ✓ |

Sub-Agent 4-续 装的 18 个 diffcsp 子依赖(einops 0.8.2 / p_tqdm 1.4.2 / smact 3.2.0 / matminer 0.9.3 / pyxtal 1.1.3 / torch_sparse 0.6.18+pt24cu124 / 等)都在 `/home/tcat/.local/lib/python3.10/site-packages/`,Exp5 自动可用。

### 4.3 GPU

```bash
nvidia-smi
# 期望: 2× NVIDIA GeForce RTX 4090,各 24 GB
# Exp4 用 GPU 0,GPU 1 全程闲
```

### 4.4 Disk

```bash
df -h ~
# Exp4 完结后 ~68 GB free
# Exp5 SA2 训练前应 ≥ 30 GB,清理目标: 旧 wandb / smoke ckpt
```

---

## §5 Exp5 工作目录建议(S 服务器,SA1 启动时 mkdir)

按 Exp4 同结构:

```
/home/tcat/diffcsp_exp5/
├── code/
│   ├── .env                          ← 复制 Exp4 .env,改 PROJECT_ROOT 等路径
│   ├── step3/                        ← Exp5 SA1 改架构产出
│   │   ├── xas_local_dataset_v2_exp5.py        ← 加 center_element_Z 字段
│   │   ├── xas_local_datamodule_v2.py          ← 复用 Exp4(若 batch 字段不变)
│   │   ├── spectrum_encoder_exp5.py            ← 加 center embedding
│   │   ├── diffusion_w_type_xas_exp5.py        ← 加 TypeClassifierHead
│   │   └── conf_xas/model/diffusion_xas_exp5.yaml  ← head + center_emb 字段
│   ├── step4/                        ← SA2 训练产出
│   ├── step5/                        ← SA3 评估产出
│   └── step6/                        ← SA4 可视化产出
├── data/                             ← 软链接到 Exp4 data,不复制
│   └── (ln -s /home/tcat/diffcsp_exp4/data/* .)
├── checkpoints/                      ← SA2 训练输出
└── logs/                             ← 全套 log
```

**软链接 data 而不是复制**: 节省 ~650 MB 磁盘 + 保证数据不会被误改。

```bash
mkdir -p /home/tcat/diffcsp_exp5/data
cd /home/tcat/diffcsp_exp5/data
for f in /home/tcat/diffcsp_exp4/data/*; do
    ln -s "$f" .
done
ls -la  # 确认全是软链接
```

---

## §6 给后续 Sub-Agent 的快速 ask 模板

Sub-Agent 启动后第一条消息建议用户附:

```
我是 Exp5-SAx (Step x agent)。我需要从服务器拿以下文件:

P0(阻塞,必须):
- /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py
- /home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml

P1(重要):
- /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py
- /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py

请用户:
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp4/code
cat step3/xas_local_dataset_v2.py
cat step3/conf_xas/model/diffusion_xas.yaml
cat step3/diffusion_w_type_xas.py
cat step2/spectrum_encoder.py

(把输出贴回对话)
```

用户照命令跑,把 4 段 cat 输出粘贴到对话——SA 立刻能动手。

---

## §7 红线汇总(任何 Sub-Agent 不能动)

| 红线 | 文件 | 位置 |
|------|------|---|
| 训练期不读 holdout | `holdout_samples_v2.csv`、`spectra_holdout.pkl` | S `/home/tcat/diffcsp_exp4/data/` |
| 全程封存 incompat | `incompat_pool.csv` | S 同上 |
| 不动 Exp4 任何 .bak* | 所有 `*.bak*` | S 各目录散落 |
| 不动 Exp4 best ckpt | `best-epoch366-val0.7300.ckpt` | S `/home/tcat/diffcsp_exp4/checkpoints/` |
| Exp5 不能用 Exp2 fork 替代 Exp4 当前版 | L 上 `experiment2\step3\` 等 | L 用户本地 |
| 不升级 7 守卫包 | env packages | S env |

---

## §8 文档版本与 reference

- 本指南覆盖 Exp4 完结时(2026-04-28)的文件状态
- 后续 Exp5 启动后,**Exp5 Main Agent 应在 Exp5 完结时类似补一份 EXP5_FILE_GUIDE.md**,记录 Exp5 新增文件位置
- 若服务器目录结构未来变动,本指南需更新(由 MA5 或后续 MA 维护)

---

*Main Agent 5 撰写补遗,2026-04-28,补 EXPERIMENT4_FINAL_REPORT.md §9 没说清的"本地 vs 服务器"取用方式。*
