# EXP5_STEP1_PRIME_HANDOFF.md
# Exp5 v2 SA1' Handoff — MV-Attention Encoder + Density-Loss De-anchoring

> **撰写者**: Exp5 v2 Main Agent(MA5 续作,Exp4 收尾后接 Exp5 v2)
> **日期**: 2026-04-28
> **接收人**: Exp5 v2 SA1'(架构改造 Sub-Agent)
> **本文档定位**: 给 SA1' 一窗一文,涵盖任务/约束/红线/代码骨架/PASS gate
> **配套必读**: EXP5_PROPOSAL_v2.md(主 proposal)、EXP5_STEP1_OUTPUT.md(v1 SA1 输出,**carry-over §5.6/§5.7 critical**)、EXP4_FILE_GUIDE.md(取文件命令)、EXP4_FINAL_REPORT_ERRATA_2.md(`_density_loss` 塌缩根因)
> **不读**: EXP4_PROPOSAL_v2.md / Exp4 中间 handoff / Exp5 v1 MA self-audit / EXPERIMENT4_FINAL_REPORT.md §10 方向 menu(已被 errata 2 推翻)

---

## §0 任务一屏掌握

### 0.1 你是谁、做什么

你是 Exp5 v2 的 SA1'(prime,区别于 v1 已被 kill 的 SA1)。Exp5 v1 在 v1-SA2 训到 epoch 36 出现 head collapse 后被用户 kill;v1-SA1 已经在服务器 `/home/tcat/diffcsp_exp5/` 改了 7 个文件并 PASS forward_test 5/5 + 1 skipped。**你的任务是在 v1 SA1 改后的代码基础上做 surgery**,撤销 head 部分 + 替换 fusion 块为 MV-attention + 改 cost_density,**不重头来**。

### 0.2 时间预算

2-3 天。**SA1' 禁止启动正式训练**。最后一步是中期报告交回 MA5,等 review 通过才开 SA2'。

### 0.3 v2 vs v1 SA1 改动总览(关键)

| v1 SA1 改的 | v2 处理 |
|---|---|
| `dataset_v2.py` 加 `_symbol_to_Z` + `center_element_Z` 字段 | **完全保留**,0 改动 |
| `datamodule_v2.py` 加 `center_element_Z` LongTensor collate | **完全保留**,0 改动 |
| `spectrum_encoder.py` 加 `nn.Embedding(95,16)` + 4-arg forward + `output_dim=272` + `cat([latent, center_emb])` 出口 | **保留 center 部分**;**替换 fusion 块**(`cat 448→Linear→SiLU→Linear` 这整段) 为 MV-attention;chi 分支末端 128→256、feff 末端 64→256 |
| `diffusion_w_type_xas.py` 加 `TypeClassifierHead` 类 + `self.type_head` + `type_loss_mode/diffusion_type_weight/head_type_weight` flags + head loss 计算 + `head_predict_types` 方法 | **撤销所有 head 痕迹**,还原 Exp4 总 loss = `cost_lattice*L + cost_coord*C + cost_type*T + cost_density*D`;**保留** SpectrumEncoder 实例化时 `n_center_elements/center_emb_dim` 参数 + Patch 1 (`F.one_hot(...).to(c0.dtype)`) + 两处 forward/sample 4-arg 调用 |
| `diffusion_xas.yaml` 加 8 个新字段 + latent_dim 256→272 + decoder.latent_dim 512→528 | **保留**: n_center_elements=95, center_emb_dim=16, latent_dim=272, decoder.latent_dim=528;**删除**: type_head_hidden_dim, n_atoms, n_elements, type_loss_mode, diffusion_type_weight, head_type_weight;**改值**: cost_density 0.5→0.2 ⭐;**新增**: mv_attention.num_heads=4 + mv_attention.residual_alpha=0.5 |
| `forward_test.py` Phase 6.6 测 head_logits/center_Z 强 conditioning/3 mode forward + Phase 6.5 SKIPPED + loss range [4,12] | **保留** Phase 6.5 SKIPPED 机制 + `_phase_65_legacy`(verbatim);**重写** Phase 6.6 测 MV-attention 性质(view 顺序 invariance / num_heads 维度 / cost_density yaml 加载);**改 range** [4,12]→[1.5,5.0] |
| `step4_1_smoke_test.py`(NEW from v1)跑 4 mode | **改写为 1 mode**(v2 没 mode flag) |
| ❌ `step4_2_train.py` v1 没写 | SA1' **新写**,fork Exp4 模板 |
| ❌ `step5_2_compute_metrics.py` v1 没碰 | SA1' **加 Set-Level TypeAcc + Multiset F1 函数**,dry-run Exp4 best ckpt 跑 baseline 数 |

### 0.4 SA1' 10 步任务清单(对应 proposal v2 §4.1)

| 步 | 任务 | 工程量 |
|---|---|---|
| 6.1 | 撤销 `diffusion_w_type_xas.py` 的 head 类 + 实例化 + 3-mode + forward 计算 + `head_predict_types`,还原 Exp4 形态(保留 center conditioning + Patch 1) | 0.5 天 |
| 6.2 | 撤销 yaml 的 head 字段(6 字段)+ 注释还原 | 10 分钟 |
| 6.3 | yaml `cost_density: 0.5 → 0.2`(一行) | 1 分钟 |
| 6.4 | 重写 `spectrum_encoder.py` 的 fusion 块为 MV-attention,chi/feff 末端升 256 | 1 天 |
| 6.5 | yaml 加 `mv_attention.num_heads=4` + `mv_attention.residual_alpha=0.5` | 5 分钟 |
| 6.6 | 新写 `step4_2_train.py`,fork Exp4 模板 + 去 warm-start + 加 PYTHONPATH self-check | 0.5 天 |
| 6.7 | `step5_2_compute_metrics.py` 加 Set-Level TypeAcc + Multiset F1 函数 + dry-run Exp4 baseline | 0.5 天 |
| 6.8 | `forward_test.py` 改写 Phase 6.6 + 调 6.4 range,跑通 5 PASS + 1 SKIPPED | 0.3 天 |
| 6.9 | `step4_1_smoke_test.py` 改写为 1 mode,2 epoch × 10 batch PASS | 0.3 天 |
| 6.10 | 中期报告交回 MA5(SA1' **不**启动正式训练) | 0.2 天 |

---

## §1 不变量(SA1' 不许动)

### 1.1 数据 / 训练参数(继承 proposal v2 §2)

| 项 | 值 | 来源 |
|---|---|---|
| 中心元素 | 88 元素(Z ∈ [2, 94] 实测) | Exp4 + v1 SA1 验证 |
| split | 60507/7624/4481/3025 | Exp4 |
| L | 6.0 Å | Exp2 step4d |
| coord 系 | [-0.5, 0.5] + min-image | Exp2 step4d |
| `cost_lattice` | 0.0 | MA4 决策 |
| `cost_coord` | 1.0 | Exp4 |
| `cost_type` | 1.0 | Exp4(v1 加的"被 diffusion_type_weight 覆盖"注释 SA1' **必须删除**) |
| **`cost_density`** | **0.2**(Exp4 是 0.5) | **proposal v2 §3.5 ⭐** |
| `n_center_elements` | 95 | v1 SA1 实测 max(Z)=94 |
| `center_emb_dim` | 16 | v1 SA1 |
| `latent_dim`(top-level yaml) | 272 | v1 SA1(= 256 + 16) |
| `decoder.latent_dim` | 528 | v1 SA1(= time_dim 256 + spectrum 272) |
| **`mv_attention.num_heads`** | **4**(每 head 64d) | **v2 新增** |
| **`mv_attention.residual_alpha`** | **0.5** 固定不可学 | **v2 新增** |
| N_NEIGHBORS | 20 | EXP4_PROPOSAL_v2 |
| 邻居搜索半径 | 10.0 Å | Exp2 |
| `<20 邻居 / frac 越界` | `return None` + collate filter | Exp4 Phase 4.6 |
| FEFF feature 维度 | 74 | Exp4 |
| 反扩散步数 | 1000 | Exp2 |
| precision | fp32 | MA4 D1 |
| batch_size | 16 | Exp4 |
| optimizer | Adam, lr=1e-4 | Exp4 |
| gradient_clip_val | 1.0 | Exp4 |
| max_epochs | 500 | Exp4 |
| early_stop patience | 30 | Exp4 |
| save_top_k | 1 | Exp4 |
| num_workers | 0(pymatgen SGA worker safety) | Exp4 |

### 1.2 PYTHONPATH 优先级(v1 SA1 OUTPUT §5.6 carry-over,必读)

服务器同时跑 Exp4(可能有遗留)、SA0、Exp5 SA2'/SA3,**全部用同一个 conda env、同一个 `diffcsp` package**。Exp4 和 Exp5 的 `diffusion_w_type_xas.py` / `spectrum_encoder.py` 是**重名不重内容的两套文件**。SA1' 写的 `step4_2_train.py` 训练入口必须**让 Exp5 的几个改动文件优先放到 PYTHONPATH 前面**,否则 Python import 缓存会拉到 Exp4 旧版,你以为在训 Exp5 v2 baseline,实际跑的是 Exp4 网络(没 center_emb、没 MV-attention)。

**v1 SA1 实测验证写法(forward_test v3 跑通的)**:

```bash
cd /home/tcat/diffcsp_exp5/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py
```

**末尾的 `/home/tcat/diffcsp_exp4/code` 不能省** —— Exp4 仓里 `diffcsp/` 子包(`diffcsp.pl_modules.cspnet` 等)在那。Exp5 没复制(也不该复制,SA1' 不动 Exp4 backbone)。三者顺序:Exp5 step3/step2 在前(shadow Exp4 同名)、Exp4 code 在末尾(找 backbone)。

### 1.3 Phase 6.5 SKIPPED-by-design(v1 SA1 OUTPUT §5.7 carry-over,verbatim 保留)

3 处 hardcoded fp32 site:
1. `diffusion_w_type_xas.py` Patch 1(已修,`F.one_hot(...).to(c0.dtype)`,fp32 下 bit-exact 等价,SA1' **必须保留**)
2. `SinusoidalTimeEmbeddings.forward` 内 `torch.arange` 默认 fp32 → time_emb 永远 fp32(未修)
3. `diffcsp/pl_modules/cspnet.py` L272-274 无 dtype-aware cast(Exp4 代码,out of scope)

Skip 决策: Exp4/Exp5 训练全程 fp32(MA4 D1),bf16 path 不在生产路径。`_phase_65_legacy()` 函数保留作未来 bf16 enabler 起点。SA1' **不修** site 2 / site 3,**不动** `_phase_65_legacy`。

---

## §2 红线(任一触发立即停 + 报 MA5)

| 红线 | 出处 |
|---|---|
| ❌ 不加任何 TypeClassifier head 或类似 head | proposal v2 §2 + Exp3 + v1 SA2 三重证伪 |
| ❌ 不 fine-tune from Exp4 ckpt(decoder 第一层 shape mismatch 528 vs 512,且 v2 是 from-scratch) | proposal v2 §2 |
| ❌ 不做 multi-sample test-time averaging | proposal v2 §2(独立任务,以后单起) |
| ❌ `cost_density` 不许调到 0(完全删可能 RMSD 退化到 2-3Å,Exp2 step4c 时代血书) | proposal v2 §2 + EXP4_FINAL_REPORT_ERRATA_2 §1.4 |
| ❌ 不动 holdout(`holdout_samples_v2.csv`、`spectra_holdout.pkl`) | EXP4_FILE_GUIDE §7 |
| ❌ 不动 incompat_pool.csv | 同上 |
| ❌ 不动 Exp4 任何 `.bak*` / Exp4 `forward_test.py.bak3` / Exp5 `.bak_exp4` 锚点 | 同上 |
| ❌ 不升级 7 守卫包(scikit-learn 1.7.2 / numpy 2.2.6 / scipy 1.15.3 / pymatgen 2025.10.7 / torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / torch-scatter 2.1.2+pt24cu124) | 同上 |
| ❌ Position-by-position TypeAcc 不许进 val 训练监控主面板(只作 metrics_report.txt 末尾历史对照栏) | Exp3 §1.1 + proposal v2 §3.4 |
| ❌ **SA1' 禁止启动正式训练**(只跑 forward_test + smoke test) | proposal v2 §6.4 |
| ❌ 不动 Exp4 `cspnet.py`(Phase 6.5 site 3) | v1 SA1 OUTPUT §5.7 |
| ❌ 不修 `_phase_65_legacy()` 函数 | 同上 |

---

## §3 服务器工作目录与文件取用

### 3.1 工作目录(已存在,v1 SA1 建过)

```
/home/tcat/diffcsp_exp5/
├── code/{step2,step3,step4}/...   ← v1 SA1 已改的 7 个文件 + 各 .bak_exp4 锚点
├── data/                          ← 软链接到 /home/tcat/diffcsp_exp4/data/(已建,99.99% 命中实测)
├── checkpoints/                   ← 空,等 SA2' 产出
├── logs/                          ← v1 forward_test / smoke 日志在此
└── sa0/                           ← SA0 multi-sample 独立(SA1' 不碰)
```

**SA1' 不需要 mkdir 任何新顶层目录**。新增文件就放进现有 step2/step3/step4/step5 子目录。

### 3.2 v1 SA1 已改的 7 个文件 + 6 个 bak 锚点(SA1' 起点)

```
/home/tcat/diffcsp_exp5/code/step2/spectrum_encoder.py            ← v1 改后(127 行)
/home/tcat/diffcsp_exp5/code/step2/spectrum_encoder.py.bak_exp4   ← Exp4 真版锚点(95 行)

/home/tcat/diffcsp_exp5/code/step3/xas_local_dataset_v2.py            ← v1 改后(374 行,SA1' **不碰**)
/home/tcat/diffcsp_exp5/code/step3/xas_local_dataset_v2.py.bak_exp4   ← Exp4 真版

/home/tcat/diffcsp_exp5/code/step3/xas_local_datamodule_v2.py            ← v1 改后(257 行,SA1' **不碰**)
/home/tcat/diffcsp_exp5/code/step3/xas_local_datamodule_v2.py.bak_exp4   ← Exp4 真版

/home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py            ← v1 改后(589 行,SA1' 大改还原)
/home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py.bak_exp4   ← Exp4 真版(415 行,撤销目标)

/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml            ← v1 改后(79 行,SA1' 大改)
/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml.bak_exp4   ← Exp4 真版(50 行)

/home/tcat/diffcsp_exp5/code/step3/forward_test.py            ← v1 改后(546 行,SA1' 改 Phase 6.6 + 6.4 range)
/home/tcat/diffcsp_exp5/code/step3/forward_test.py.bak_exp4   ← Exp4 真版(365 行)

/home/tcat/diffcsp_exp5/code/step4/step4_1_smoke_test.py   ← v1 NEW(193 行,无 bak;SA1' 改写 1 mode)
```

### 3.3 SA1' 需要新建的文件

```
/home/tcat/diffcsp_exp5/code/step4/step4_2_train.py            ← SA1' 新写(fork Exp4 模板)
/home/tcat/diffcsp_exp5/code/step5/step5_2_compute_metrics.py  ← SA1' 改造(fork Exp4 + 加 Set-Level/Multiset)
```

### 3.4 Exp4 模板文件(SA1' fork 起点,**只读不改**)

```
/home/tcat/diffcsp_exp4/code/step4/step4_2_train.py            ← Exp4 训练入口模板
/home/tcat/diffcsp_exp4/code/step5/step5_2_compute_metrics.py  ← Exp4 metrics 模板
/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt          ← SA1' dry-run baseline 输入
/home/tcat/diffcsp_exp4/code/step5/predictions_test.pt         ← 同
/home/tcat/diffcsp_exp4/code/step5/predictions_holdout.pt      ← 同(注意此文件 holdout 仅 SA3 期解禁,但 baseline 重算指标本身不动 holdout 真实 label,只读 prediction;若担心红线建议 SA1' 仅在 val/test 上 dry-run,holdout 留 SA3)
```

**SA1' 取上面 4 个 Exp4 文件**: 让用户跑

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp4/code

echo "=== step4/step4_2_train.py ==="
cat step4/step4_2_train.py
echo ""
echo "=== step5/step5_2_compute_metrics.py ==="
cat step5/step5_2_compute_metrics.py
```

predictions_*.pt 是二进制,SA1' 在服务器内 `torch.load` 读即可,不必传到对话。

### 3.5 v1 SA1 OUTPUT.md 必读章节(carry-over)

**SA1' 必读**(本地用户处下载或从对话历史拿):
- §3 关键实施决策(yaml 默认值表 + 4 条 deviation a/b/c/d)
- **§5.6 PYTHONPATH 优先级**(本 handoff §1.2 已 carry,但 OUTPUT 原文更详细)
- **§5.7 Phase 6.5 SKIPPED-by-design**(本 handoff §1.3 已 carry,OUTPUT 原文有 3 处 hardcoded fp32 site 完整描述)
- §6.1 forward_test v3 实测日志(看 v1 时各 phase 数字,SA1' 改 Phase 6.4 range / Phase 6.6 时参考)

**SA1' 可跳过**:
- §4 OPEN QUESTIONS(OQ-1/-2/-3 都涉及 head,v2 没 head,自动消解)
- §5.1 phased training(v2 from-scratch,不需要)
- §5.2 Exp4 ckpt warm-start(v2 不 warm-start)
- §5.3 三 mode flag(v2 没 mode)


---

## §4 SA1' 10 步任务详细规格

每步含: **输入文件 / 改动规格 / PASS gate / 红线**。建议按 6.1→6.10 顺序做,因为后步依赖前步产出。

> **命名约定**: 本节的 "6.1, 6.2, ..., 6.10" 是 **SA1' 工作步骤编号**(沿用 proposal v2 §4.1 的编号),与 `forward_test.py` 内的 "Phase 6.1, 6.2, ..., 6.6" 不同名空间。文中提及 "Phase 6.x" 时永远指 forward_test 的 phase,提及 "Step 6.x" / "6.x" 不带 "Phase" 前缀时指 SA1' 工作步骤。请勿混淆。

### 6.1 撤销 `diffusion_w_type_xas.py` 的 head 痕迹

**输入**:
- 起点: `/home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py`(v1 改后,589 行)
- 撤销目标参考: `/home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py.bak_exp4`(Exp4 真版,415 行)

**改动规格**(diff 思路: 在 v1 当前版基础上**逐项撤销**, **不**整文件回退到 bak):

(A) **删除整个 `TypeClassifierHead` 类**(v1 diff 第 47-91 行加的,~ 50 行)。整段连同前面 `# ── Exp5 SA1: TypeClassifierHead ────` 注释删干净。

(B) **`CSPDiffusion.__init__` 内**:
- ✅ **保留**: `self.spectrum_encoder = SpectrumEncoder(...)` 实例化 (含 `n_center_elements`、`center_emb_dim` 参数)
- ✅ **保留**: `self._spectrum_out_dim = self.spectrum_encoder.output_dim`(= 272)
- ✅ **保留**: `self.cost_density = float(self.hparams.get('cost_density', 0.5))`(默认值留 0.5,实际 yaml 给 0.2 覆盖)
- ❌ **删除**: `self.type_head = TypeClassifierHead(...)` 整段 + 注释
- ❌ **删除**: `self.type_loss_mode = ...` + `if self.type_loss_mode not in ...: raise ValueError(...)`
- ❌ **删除**: `self.diffusion_type_weight = ...` 和 `self.head_type_weight = ...` 两行 + 注释

(C) **`forward()` 内**:
- ✅ **保留**: SpectrumEncoder 4-arg 调用(`batch.center_element_Z` 第 4 参)
- ✅ **保留**: Patch 1 的 `F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).to(c0.dtype)`(**verbatim**,不许回 `.float()`)
- ❌ **删除**: `head_logits = self.type_head(spectrum_cond)` 行
- ❌ **删除**: `true_types_per_slot = batch.atom_types.view(batch_size, -1)` 行
- ❌ **删除**: `loss_type_ce_head = F.cross_entropy(...)` 整段
- ❌ **删除**: `if self.type_loss_mode == 'diffusion_only': ... elif ... else ...` 整 3-mode block
- ✅ **还原**: 总 loss 公式回到 Exp4 形态
  ```python
  loss = (self.hparams.cost_lattice * loss_lattice
          + self.hparams.cost_coord  * loss_coord
          + self.hparams.cost_type   * loss_type
          + self.cost_density        * loss_density)
  ```
- ✅ **还原**: output dict 回到 Exp4 5 字段(`loss / loss_lattice / loss_coord / loss_type / loss_density`),**删除** v1 加的 3 字段(`loss_diffusion_type / loss_type_ce_head / loss_type_total`)

(D) **`sample()` 内**:
- ✅ **保留**: SpectrumEncoder 4-arg 调用(`batch.center_element_Z` 第 4 参)
- 其他不变

(E) **`training_step` / `compute_stats` 内**:
- ❌ **删除** v1 加的 3 个 log 字段(`loss_diffusion_type / loss_type_ce_head / loss_type_total`)
- ✅ **还原**: log_dict 回到 Exp4 5 字段

(F) **删除 `head_predict_types` 方法整段**(v1 diff 最后 ~ 35 行)。

**PASS gate (本地静态检查)**:

```bash
cd /home/tcat/diffcsp_exp5/code/step3

# 1. import 不报错
/home/tcat/conda_envs/mlff/bin/python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step3')
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step2')
sys.path.append('/home/tcat/diffcsp_exp4/code')
import diffusion_w_type_xas as m
assert not hasattr(m, 'TypeClassifierHead'), 'head 类未删干净'
assert not hasattr(m.CSPDiffusion, 'head_predict_types'), 'head_predict_types 方法未删干净'
print('PASS: head 痕迹已撤干净')
"

# 2. grep 不剩 head 关键字
grep -nE 'TypeClassifierHead|type_head|head_predict|loss_type_ce_head|loss_type_total|loss_diffusion_type|type_loss_mode|diffusion_type_weight|head_type_weight' diffusion_w_type_xas.py
# 期望: 输出为空(不剩任何匹配)
```

**红线**:
- 不许回 `.float()` —— Patch 1 必须保留为 `.to(c0.dtype)`
- 不许动 `_density_loss` 函数本身(只改它在 yaml 的权重 0.5→0.2,不改实现)
- 不许动 `SinusoidalTimeEmbeddings`(Phase 6.5 site 2,out of scope)

---

### 6.2 撤销 yaml 的 head 字段

**输入**: `/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml`(v1 改后,79 行)

**改动规格**:

❌ **删除以下 6 字段**(v1 加的,~ 14 行含注释):
- `type_head_hidden_dim: 512`
- `n_atoms: 20`
- `n_elements: 100`
- `type_loss_mode: both`
- `diffusion_type_weight: 1.0`
- `head_type_weight: 0.5`

❌ **删除注释 block** "# ── Exp5 SA1: TypeClassifierHead ────" 和 "# ── Exp5 SA1: three-mode type-loss aggregation ────"

✅ **还原** `cost_type` 行(把 v1 加的 "⚠️ Exp5 SA1: 该字段被 diffusion_type_weight 覆盖" 注释删,回到 Exp4 形态):
```yaml
cost_type:    1.0
```

✅ **还原** yaml 头部 docstring(v1 加的 "Exp5 SA1 patch: + center embedding + TypeClassifierHead + three-mode type-loss flag" 改为 "Exp5 v2 patch: + center embedding + MV-attention encoder + cost_density 0.5→0.2"):

```yaml
# Exp5 v2 改动一览:
#   1. 继承 v1 SA1: center embedding (n_center_elements=95, center_emb_dim=16)
#      → SpectrumEncoder.output_dim = 256 + 16 = 272
#      → decoder.latent_dim = time_dim(256) + spectrum(272) = 528
#   2. 主线 1: SpectrumEncoder fusion 块替换为 MV-attention
#      (mv_attention.num_heads=4, residual_alpha=0.5 固定)
#   3. 主线 2: cost_density 0.5 → 0.2 (errata 2 §1 _density_loss 塌缩根因减弱)
#   4. 不加 TypeClassifierHead (Exp3 + Exp5 v1 双重证伪)
```

**保留**:
- `n_center_elements: 95` + `center_emb_dim: 16`
- `latent_dim: 272`
- `decoder.latent_dim: 528`(保留,因 spectrum 仍 272)
- `cost_lattice: 0.0` + 其修正注释(L=12→6)

**PASS gate**:

```bash
cd /home/tcat/diffcsp_exp5/code/step3

# yaml 不能再有 head 字段
grep -E 'type_head_hidden_dim|^n_atoms|^n_elements|type_loss_mode|diffusion_type_weight|head_type_weight' conf_xas/model/diffusion_xas.yaml
# 期望: 输出为空

# yaml 仍有保留字段
grep -E 'n_center_elements|center_emb_dim|latent_dim|cost_density|cost_type' conf_xas/model/diffusion_xas.yaml
# 期望: 6 行匹配
```

---

### 6.3 yaml `cost_density: 0.5 → 0.2`

**输入**: 同 6.2 的 yaml 文件

**改动**:

```yaml
# 原(v1 继承 Exp4):
cost_density: 0.5

# 改为(v2 主线 2):
cost_density: 0.2   # Exp5 v2: 0.5→0.2, 减弱 _density_loss 塌缩剂 (EXP4_FINAL_REPORT_ERRATA_2 §1.4)
```

**PASS gate**:

```bash
grep '^cost_density' /home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml
# 期望: cost_density: 0.2   # Exp5 v2: ...
```

**红线**:
- **不许调到 0**(proposal v2 §2 红线;Exp2 step4c 时代证明完全删 density 会 RMSD 退化到 4Å)
- **不许调到 0.1 / 0.05** —— 这些是 Exp6 候选,SA1' 时段锁 0.2

---

### 6.4 重写 `spectrum_encoder.py` 的 fusion 块为 MV-attention

**输入**: `/home/tcat/diffcsp_exp5/code/step2/spectrum_encoder.py`(v1 改后,127 行)

**改动规格**:

(A) **修改 chi 分支末端 Linear 输出 128 → 256**

定位 `self.chi_encoder = nn.Sequential(...)`,找其最后一个 `nn.Linear(...)`,把第二个参数从 128 改 256。

(B) **修改 feff 分支末端 Linear 输出 64 → 256**

定位 `self.feat_encoder = nn.Sequential(...)`,找其最后一个 `nn.Linear(...)`,把第二个参数从 64 改 256。

(C) **删除 `self.fusion = nn.Sequential(nn.Linear(448, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))` 整段**

(D) **新增 MV-attention 组件**(详见本 handoff §5 完整代码骨架)

(E) **修改 `forward()` 内 fusion 调用**

原 v1:
```python
fused = torch.cat([xmu_out, chi_out, feat_out], dim=-1)  # (B, 448)
latent = self.fusion(fused)                              # (B, 256)
center_e = self.center_emb(center_Z)                     # (B, 16)
return torch.cat([latent, center_e], dim=-1)             # (B, 272)
```

改为(v2):
```python
# 三 view 各 (B, 256)
views = torch.stack([xmu_out, chi_out, feat_out], dim=1)  # (B, 3, 256)

# Cross-attention with learnable query
B = views.shape[0]
q = self.mv_query.expand(B, -1, -1)                       # (B, 1, 256)
attn_out, _ = self.mv_attn(q, views, views, need_weights=False)  # (B, 1, 256)
attn_out = attn_out.squeeze(1)                            # (B, 256)

# Post-residual LayerNorm + projection
fused = attn_out + self.mv_residual_alpha * views.mean(dim=1)  # (B, 256)
fused = self.mv_layernorm(fused)
latent = self.mv_proj(fused)                              # (B, 256)

# 继承 v1: center embedding cat 在末尾
center_e = self.center_emb(center_Z)                      # (B, 16)
return torch.cat([latent, center_e], dim=-1)              # (B, 272)
```

(F) **修改 `__init__` 签名加 `mv_num_heads` / `mv_residual_alpha` 参数**(yaml 通过 hydra 传入,详见 §5)

**PASS gate**:

```bash
cd /home/tcat/diffcsp_exp5/code/step2

/home/tcat/conda_envs/mlff/bin/python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step2')
import torch
from spectrum_encoder import SpectrumEncoder

enc = SpectrumEncoder()  # 默认参数
B = 4

# 检查组件
assert hasattr(enc, 'mv_attn'), '缺 mv_attn'
assert hasattr(enc, 'mv_query'), '缺 mv_query'
assert hasattr(enc, 'mv_layernorm'), '缺 mv_layernorm'
assert hasattr(enc, 'mv_proj'), '缺 mv_proj'
assert hasattr(enc, 'center_emb'), '缺 center_emb'
assert not hasattr(enc, 'fusion'), '旧 fusion 块未删干净'

# 检查 forward shape
xmu = torch.randn(B, 150)
chi = torch.randn(B, 200)
feff = torch.randn(B, 74)
center_Z = torch.tensor([26, 8, 11, 47], dtype=torch.long)

out = enc(xmu, chi, feff, center_Z)
assert tuple(out.shape) == (B, 272), f'out shape {tuple(out.shape)} != (4, 272)'
assert not torch.isnan(out).any(), 'NaN in output'
assert enc.output_dim == 272, f'output_dim={enc.output_dim}'

# 参数量 sanity
n_params = sum(p.numel() for p in enc.parameters())
print(f'PASS: SpectrumEncoder n_params={n_params:,}, output_dim=272')
"
```

**红线**:
- 不许把 `mv_residual_alpha` 设成可学(`nn.Parameter`),必须 `float` 标量挂在 `self`
- 不许去掉 LayerNorm
- 不许把 num_heads 改成 1 / 8 / 其他值(锁 4)

---

### 6.5 yaml 加 mv_attention 字段

**输入**: 同 6.2/6.3 的 yaml

**改动**:

在 yaml 的 SpectrumEncoder 配置区(已有 `xmu_dim/chi_dim/feat_dim/n_center_elements/center_emb_dim/latent_dim`)末尾,加:

```yaml
# ── Exp5 v2: MV-attention fusion(主线 1) ─────────────────────────────────
# 三 view (xmu/chi/feff) 各 256d → MultiheadAttention(num_heads=4) → residual + LN → 256d
# residual_alpha 固定不可学(防 attention 早期 noise 拉飞 latent)
mv_attention:
  num_heads:      4    # 256 / 4 = 64 per head
  residual_alpha: 0.5  # 固定,不可学
```

**PASS gate**:

```bash
/home/tcat/conda_envs/mlff/bin/python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml')
assert cfg.mv_attention.num_heads == 4
assert cfg.mv_attention.residual_alpha == 0.5
assert cfg.cost_density == 0.2
assert cfg.n_center_elements == 95
assert cfg.center_emb_dim == 16
assert cfg.latent_dim == 272
assert cfg.decoder.latent_dim == 528
print('PASS: yaml fields all in place')
"
```

---

### 6.6 新写 `step4_2_train.py`

**输入**:
- 起点(fork): `/home/tcat/diffcsp_exp4/code/step4/step4_2_train.py`(SA1' 让用户 ssh cat 给你)
- 输出: `/home/tcat/diffcsp_exp5/code/step4/step4_2_train.py`(新建)

**改动规格**:

(A) **路径硬编码改 Exp5**:
- `DATA_DIR = "/home/tcat/diffcsp_exp5/data"` (软链到 Exp4 data)
- `CHECKPOINT_DIR = "/home/tcat/diffcsp_exp5/checkpoints"`
- `LOG_DIR = "/home/tcat/diffcsp_exp5/logs"`
- yaml: `/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml`

(B) **去除 Exp4 的 ckpt warm-start 逻辑**(如有 `model.load_state_dict(torch.load(...))` from Exp4 best ckpt 这类代码,全删 —— v2 是 from-scratch)

(C) **加 PYTHONPATH self-check**(脚本开头,在所有 `import diffusion_w_type_xas` / `import spectrum_encoder` 之前):

```python
import sys, os

# Exp5 v2: PYTHONPATH 优先级硬保证(carry-over from v1 SA1 OUTPUT §5.6)
EXP5_STEP3 = "/home/tcat/diffcsp_exp5/code/step3"
EXP5_STEP2 = "/home/tcat/diffcsp_exp5/code/step2"
EXP4_BACKBONE = "/home/tcat/diffcsp_exp4/code"
sys.path.insert(0, EXP5_STEP2)
sys.path.insert(0, EXP5_STEP3)  # 最高优先级
if EXP4_BACKBONE not in sys.path:
    sys.path.append(EXP4_BACKBONE)

# Self-check: 确认拉到的是 Exp5 而不是 Exp4 同名文件
import diffusion_w_type_xas
import spectrum_encoder
assert "/diffcsp_exp5/" in diffusion_w_type_xas.__file__, \
    f"WRONG diffusion_w_type_xas: {diffusion_w_type_xas.__file__} (期望 /diffcsp_exp5/...)"
assert "/diffcsp_exp5/" in spectrum_encoder.__file__, \
    f"WRONG spectrum_encoder: {spectrum_encoder.__file__} (期望 /diffcsp_exp5/...)"
print(f"[PYTHONPATH check] diffusion_w_type_xas: {diffusion_w_type_xas.__file__}")
print(f"[PYTHONPATH check] spectrum_encoder:     {spectrum_encoder.__file__}")
```

(D) **训练参数(从 yaml 读,但脚本里也写明 sanity)**:
- `max_epochs = 500`
- `early_stop patience = 30`
- `save_top_k = 1`
- `monitor = 'val_loss'`(主指标,与 Exp4 一致)
- `precision = 'fp32'`(MA4 D1)

(E) **新增 val 监控指标 callback**:

在 PyTorch Lightning trainer 的 callback 列表加一个 `EpochEndMetricsCallback`(SA1' 写),每 epoch end 时:
1. 在 val_dataloader 上跑一遍模型(已有 PL 内置 val_step 流程,不重复跑)
2. 调用 `step5_2_compute_metrics` 中的 `compute_set_level_typeacc` 和 `compute_multiset_f1` 函数
3. log 到 `val_set_level_typeacc` 和 `val_multiset_f1_macro`
4. **不 log** position-by-position TypeAcc 到 prog_bar(只保留在末尾 metrics_report.txt 历史对照栏)

注: 训练监控的 Set-Level / Multiset 计算可以是**简化版**(对 atom_types argmax 直接,不做 Hungarian)。完整版 metrics 在 SA3 阶段算。SA1' 决定简化版 vs 完整版,如选简化版需在 docstring 注明 "training monitor only, full metrics in step5_2".

**推荐**: 训练监控用简化版(直接 argmax + multiset 计算,~ 30 行 inline 在 `compute_stats` 内),SA3 阶段才用完整 Hungarian 版(在 step5_2_compute_metrics.py 中)。这样训练 epoch 不慢。

(F) **后台启动指令文档化**(写在 step4_2_train.py 头部 docstring 内,SA2' 要看):

```python
"""
Run from /home/tcat/diffcsp_exp5/code/step4/ with mlff env active:

    cd /home/tcat/diffcsp_exp5/code/step4
    PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
    nohup /home/tcat/conda_envs/mlff/bin/python step4_2_train.py \
        > /home/tcat/diffcsp_exp5/logs/step4_train_v2_stdout.log \
        2> /home/tcat/diffcsp_exp5/logs/step4_train_v2_stderr.log &

SA2' 启动后:
    1. 头 30 min 守屏看 val_loss 是否合理(初始应在 ~ 2-4)
    2. 关 ssh,等 ~ 32h 训练
    3. best ckpt 落到 /home/tcat/diffcsp_exp5/checkpoints/
"""
```

**PASS gate**:

```bash
# 1. import 不报错
cd /home/tcat/diffcsp_exp5/code/step4
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  /home/tcat/conda_envs/mlff/bin/python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step4')
# 仅 import 模块层,不跑 main()
import importlib.util
spec = importlib.util.spec_from_file_location('train', './step4_2_train.py')
# 这里只检查 file 能 parse
import ast
with open('step4_2_train.py') as f:
    ast.parse(f.read())
print('PASS: step4_2_train.py syntax OK + PYTHONPATH check 在脚本开头')
"

# 2. grep 确认无 warm-start 残留
grep -nE 'load_state_dict|best-epoch|val0\.7300|warm[-_]?start' /home/tcat/diffcsp_exp5/code/step4/step4_2_train.py
# 期望: 输出为空(或仅注释行说明 'no warm-start')
```

**红线**:
- **不许跑训练**(SA1' 只检查 import + ast parse,不 `python step4_2_train.py`)
- 不许加 warm-start 逻辑

---

### 6.7 `step5_2_compute_metrics.py` 加 Set-Level + Multiset F1 + dry-run baseline

**输入**:
- 起点(fork): `/home/tcat/diffcsp_exp4/code/step5/step5_2_compute_metrics.py`(SA1' 让用户 ssh cat 给你)
- 输出: `/home/tcat/diffcsp_exp5/code/step5/step5_2_compute_metrics.py`(新建/改造)

**改动规格**(详细算法定义见 §8):

(A) **保留 Exp4 已有的指标计算**: RMSD (Hungarian min-image)、position-by-position TypeAcc、pred_in_cutoff、true_in_cutoff。这些是历史对照,不删。

(B) **新增 4 个函数**(纯 pure function,无 PL 依赖):

```python
def compute_set_level_typeacc(pred_types: np.ndarray, true_types: np.ndarray) -> float:
    """
    Per-sample Set-Level TypeAcc = sum_c min(pred_count_c, true_count_c) / N

    Parameters
    ----------
    pred_types : (20,) int  predicted Z values
    true_types : (20,) int  ground-truth Z values

    Returns
    -------
    float in [0, 1]
    """
    from collections import Counter
    pred_cnt = Counter(pred_types.tolist())
    true_cnt = Counter(true_types.tolist())
    intersection = sum(min(pred_cnt[c], true_cnt[c]) for c in (pred_cnt.keys() | true_cnt.keys()))
    return intersection / max(len(true_types), 1)


def compute_multiset_f1_macro(all_pred_types: list[np.ndarray],
                               all_true_types: list[np.ndarray]) -> dict:
    """
    Dataset-level Multiset Macro-F1 across element classes.

    For each class c:
        TP_c = sum_samples min(pred_count_c, true_count_c)
        FP_c = sum_samples (pred_count_c - min(pred_count_c, true_count_c))
        FN_c = sum_samples (true_count_c - min(pred_count_c, true_count_c))
        precision_c = TP_c / (TP_c + FP_c)  if TP_c+FP_c > 0 else 0
        recall_c    = TP_c / (TP_c + FN_c)  if TP_c+FN_c > 0 else 0
        F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

    Macro-F1 = mean over all classes that appear in any true sample.

    Returns
    -------
    dict:
      'macro_f1':        float
      'per_class_f1':    dict[int, float]    Z → F1
      'per_class_support': dict[int, int]    Z → total true count
      'n_classes_evaluated': int
    """
    # 实现略 — 标准 multi-label macro-F1 思路
    ...


def compute_collapse_ratio(all_pred_frac: list[np.ndarray],
                            all_true_frac: list[np.ndarray],
                            L: float = 6.0) -> dict:
    """
    Per-sample collapse detection.

    For each sample:
        pred_xyz_std = np.std(pred_frac * L, axis=0).mean()  # avg per-axis std
        true_xyz_std = np.std(true_frac * L, axis=0).mean()
        is_collapsed = pred_xyz_std < 0.5 * true_xyz_std

    Returns dict:
      'collapse_ratio': float
      'n_collapsed':    int
      'n_total':        int
    """
    ...


def compute_projection_ablation_rmsd(all_pred_frac, all_true_frac,
                                      R_max: float, L: float = 6.0) -> dict:
    """
    SA3 投影 ablation 的 helper。SA1' 实现函数,SA3 调用。
    将笛卡尔距原点 > R_max 的预测原子投影回 R_max 球壳,重算 RMSD。
    """
    ...
```

(C) **修改 main() 流程**:

```python
# Exp4 main 已有: 算 RMSD/TypeAcc/pred_in_cutoff/true_in_cutoff 写 metrics_report_*.txt
# Exp5 v2 main 加: 算 Set-Level / Multiset F1 / Collapse ratio,写到同一 report

# metrics_report_val.txt 输出结构(替代 Exp4 版):
# ============ EXP5 V2 METRICS REPORT (val) ============
# RMSD:             1.4866 ± 0.xxx
# pred_in_cutoff:   18.92 / 20
# true_in_cutoff:   19.79 / 20
# 
# --- Type metrics (Exp5 v2 主面板) ---
# Set-Level TypeAcc (per-sample avg):  0.xxxx
# Multiset Macro-F1 (dataset-level):   0.xxxx
# Collapse Ratio:                      x.x%  (n_collapsed / n_total)
# 
# --- Type metrics (历史对照,Exp3 已证为虚假指标,仅供回溯) ---
# Position-by-position TypeAcc:       0.1973
```

(D) **dry-run Exp4 baseline**: SA1' 写完 metrics 后,在 val/test 上跑一遍 Exp4 best ckpt 的 predictions:

```bash
cd /home/tcat/diffcsp_exp5/code/step5
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
  /home/tcat/conda_envs/mlff/bin/python step5_2_compute_metrics.py \
    --predictions /home/tcat/diffcsp_exp4/code/step5/predictions_val.pt \
    --output /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt \
    --split val
# 同跑 test
```

baseline 数会在 SA1' 中期报告里写出,SA3 直接拿来作 Exp5 v2 vs Exp4 对照。

**PASS gate**:

```bash
# 1. 4 个新函数都能 import
/home/tcat/conda_envs/mlff/bin/python -c "
import sys
sys.path.insert(0, '/home/tcat/diffcsp_exp5/code/step5')
from step5_2_compute_metrics import (
    compute_set_level_typeacc,
    compute_multiset_f1_macro,
    compute_collapse_ratio,
    compute_projection_ablation_rmsd,
)
print('PASS: 4 函数可 import')
"

# 2. dry-run 产出 Exp4 baseline 文件
ls -la /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt
# 期望: 文件存在 + 含 Set-Level / Multiset / Collapse 三行
grep -E 'Set-Level|Multiset|Collapse' /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt
```

**红线**:
- 不读 `predictions_holdout.pt`(holdout 仅 SA3 期解禁)。SA1' dry-run 只用 val + test。
- position-by-position TypeAcc 不进 prog_bar / 主指标显示位

---

### 6.8 `forward_test.py` 改写 Phase 6.6 + 调 6.4 range

**输入**: `/home/tcat/diffcsp_exp5/code/step3/forward_test.py`(v1 改后,546 行)

**改动规格**:

(A) **保留**:
- Phase 6.1 / 6.2 / 6.3(13-field schema、(4, 272) 检查、center_Z sanity)— **不动**
- Phase 6.5 SKIPPED-by-design 完整机制 + `_phase_65_legacy()` 函数 — **verbatim 保留**
- Header docstring 中 Phase 6.5 的 3 处 hardcoded fp32 site 描述 — **verbatim 保留**(改 "Exp5 SA1" → "Exp5 v2 SA1' (carry-over from v1 SA1)" 即可)

(B) **改 Phase 6.4 loss range** [4, 12] → [1.5, 5.0]:

```python
# Exp5 v2: loss range adjusted from [4, 12] (v1 with head 'both') to [1.5, 5.0]
#   v1 'diffusion_only' 实测 ~ 2.42; v2 ≈ same minus 0.5*(0.5-0.2)*0.083 = 2.42 - 0.012
#   MV-attention random init 可能轻微偏移;[1.5, 5.0] 留余量
if not (1.5 <= loss_val <= 5.0):
    log(f"  WARN: loss={loss_val:.4f} outside Exp5 v2 expected [1.5, 5.0] (random-init can drift; not gating)")
```

(C) **改写 Phase 6.6**(整段重写,删 v1 的 6.6.a/b/c/d head 测试,改测 MV-attention):

```python
def phase_66(batch_cpu):
    """
    Phase 6.6 — Exp5 v2 architecture additions: MV-attention + cost_density 0.2

    (a) SpectrumEncoder 有 MV-attention 组件: mv_attn / mv_query / mv_layernorm / mv_proj
    (b) Encoder forward 在不同 num_heads 假设下产生 (B, 272) 输出无 NaN
    (c) View 顺序无关性: shuffle (xmu/chi/feff) 进 stack 顺序 → fused latent 输出 invariant
        (cross-attention with shared query 是 set-pooling, 不该依赖 view 顺序)
    (d) yaml cost_density=0.2 加载到模型 self.cost_density 字段
    """
    log("\n" + "=" * 72)
    log("Phase 6.6 — Exp5 v2 MV-attention + cost_density verification (CPU)")
    log("=" * 72)

    from spectrum_encoder import SpectrumEncoder

    # 6.6.a — MV-attention 组件存在
    log("--- 6.6.a SpectrumEncoder has MV-attention components ---")
    enc = SpectrumEncoder()
    for attr in ['mv_attn', 'mv_query', 'mv_layernorm', 'mv_proj', 'center_emb']:
        if not hasattr(enc, attr):
            fail("Phase 6.6.a", f"SpectrumEncoder missing attribute: {attr}")
        log(f"  ✓ has {attr}")
    if hasattr(enc, 'fusion'):
        fail("Phase 6.6.a", "Old fusion block not removed")
    log(f"  mv_attn.embed_dim={enc.mv_attn.embed_dim}, num_heads={enc.mv_attn.num_heads}")
    if enc.mv_attn.num_heads != 4:
        fail("Phase 6.6.a", f"num_heads={enc.mv_attn.num_heads}, expect 4")

    # 6.6.b — Forward 输出 (B, 272) 无 NaN
    log("--- 6.6.b Forward output (B, 272) no NaN ---")
    enc.eval()
    with torch.no_grad():
        z = enc(batch_cpu.xmu_xanes, batch_cpu.chi1, batch_cpu.feff_features,
                batch_cpu.center_element_Z)
    if tuple(z.shape) != (4, 272):
        fail("Phase 6.6.b", f"shape {tuple(z.shape)} != (4, 272)")
    if torch.isnan(z).any() or torch.isinf(z).any():
        fail("Phase 6.6.b", "NaN/Inf in encoder output")
    log(f"  z.shape={tuple(z.shape)}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")

    # 6.6.c — View 顺序无关性(set-equivariance check)
    # 直接 hack 进 forward 把三 view 顺序 shuffle, 应该不影响输出
    log("--- 6.6.c View order invariance (shuffled 3 views → same fused latent) ---")
    enc2 = SpectrumEncoder()
    enc2.load_state_dict(enc.state_dict())
    enc2.eval()
    # 注: 我们要测的是 fusion 阶段对 view 顺序无关(因 attention with shared query 是 set-pooler)
    # 但 SpectrumEncoder.forward 对外接 (xmu, chi, feff) 是固定语义的,不能直接 shuffle 输入
    # 改测: monkey-patch 内部 stack 顺序
    import torch.nn.functional as F_

    with torch.no_grad():
        # 标准顺序
        xmu_o  = enc.xmu_encoder(batch_cpu.xmu_xanes.unsqueeze(1))
        chi_o  = enc.chi_encoder(batch_cpu.chi1.unsqueeze(1))
        feat_o = enc.feat_encoder(batch_cpu.feff_features)

        views_normal = torch.stack([xmu_o, chi_o, feat_o], dim=1)
        views_shuffled = torch.stack([feat_o, xmu_o, chi_o], dim=1)  # 不同顺序

        B = views_normal.shape[0]
        q = enc.mv_query.expand(B, -1, -1)
        out_normal,   _ = enc.mv_attn(q, views_normal,   views_normal,   need_weights=False)
        out_shuffled, _ = enc.mv_attn(q, views_shuffled, views_shuffled, need_weights=False)

    diff = (out_normal - out_shuffled).abs().max().item()
    log(f"  max |out_normal - out_shuffled| = {diff:.6e} (expect < 1e-5)")
    if diff > 1e-4:
        fail("Phase 6.6.c", f"View order matters: max diff {diff:.6e} > 1e-4 — MV-attention is not set-pooler!")

    # 6.6.d — yaml cost_density=0.2 loaded into model
    log("--- 6.6.d yaml cost_density=0.2 loaded into CSPDiffusion.cost_density ---")
    model = _instantiate_model()
    if abs(model.cost_density - 0.2) > 1e-6:
        fail("Phase 6.6.d", f"model.cost_density={model.cost_density}, expect 0.2")
    log(f"  model.cost_density = {model.cost_density}")

    log("[Phase 6.6 PASS]")
```

(D) **main()** 末尾保持 v1 的 5/5 PASS + 1 SKIPPED 计数逻辑,不变。

**PASS gate**:

```bash
cd /home/tcat/diffcsp_exp5/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py 2>&1 | tee /home/tcat/diffcsp_exp5/logs/step1_forward_test_v2.log

# 期望末尾输出:
#   5/5 PHASES PASS  +  1 SKIPPED-BY-DESIGN (phase 6.5)
#   Phases run: 6.1 / 6.2 / 6.3 / 6.4 / 6.6   ALL PASS
#   Step 1 launch gate: CLEAR (Exp5 v2 architecture verified, fp32 production path)
```

**红线**:
- 不动 `_phase_65_legacy()`
- 不删 v1 已有的 13-field schema check / center_Z sanity / Phase 6.5 SKIP 机制

---

### 6.9 `step4_1_smoke_test.py` 改写为 1 mode

**输入**: `/home/tcat/diffcsp_exp5/code/step4/step4_1_smoke_test.py`(v1 NEW,193 行)

**改动规格**:

v1 跑 4 mode (`None / diffusion_only / head_only / both`),每 mode 验证 head_ce / diff_t / type_total 三 loss。v2 没 mode flag,简化为单路径:

(A) 删除 mode 循环 + mode 切换的所有代码

(B) 简化为单次 2-epoch × 10-batch 跑通验证:
- val_loss 在 2-epoch 后下降(简单合理性)
- 全 4 loss 合理(`loss / coord_loss / type_loss / density_loss`,无 `head_ce`)
- 无 NaN / Inf
- ckpt 落地 `/home/tcat/diffcsp_exp5/checkpoints/_smoke/`(后续可清)

(C) 加 PYTHONPATH self-check(同 6.6 D)

**PASS 标准**:
```
SMOKE PASS:
  loss(epoch 0):  ~ 2.4 ± 0.5  (random init)
  loss(epoch 2):  < loss(epoch 0)  (有下降)
  no NaN / Inf in any loss field
  ckpt 落地 /home/tcat/diffcsp_exp5/checkpoints/_smoke/
```

**PASS gate**:

```bash
cd /home/tcat/diffcsp_exp5/code/step4
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python step4_1_smoke_test.py 2>&1 | tee /home/tcat/diffcsp_exp5/logs/step1_smoke_v2.log

grep 'SMOKE PASS' /home/tcat/diffcsp_exp5/logs/step1_smoke_v2.log
# 期望: 输出 1 行 SMOKE PASS
```

**红线**:
- smoke test 跑完后 ckpt 立即清(`rm -rf /home/tcat/diffcsp_exp5/checkpoints/_smoke/`),不留垃圾占盘
- 不许跑超过 2 epoch × 10 batch(防滑入正式训练)

---

### 6.10 中期报告交回 MA5

**SA1' 中期报告内容**(写到 `/home/tcat/diffcsp_exp5/EXP5_STEP1_PRIME_OUTPUT.md`):

| 项 | 描述 |
|---|---|
| §0 完成度 | 10 步任务勾选表(对照本 handoff §0.4 表) |
| §1 改了什么 | 7 个文件 + 2 个新建文件(`step4_2_train.py`、`step5_2_compute_metrics.py` 改造)的 diff 摘要 |
| §2 PASS gate 实测证据 | 6.1-6.9 各 gate 的命令 + 输出截图 |
| §3 forward_test 完整 log | 5/5 PASS + 1 SKIPPED 末尾输出 verbatim |
| §4 smoke test 完整 log | epoch 0/1/2 loss 数 + SMOKE PASS 末尾 |
| §5 Exp4 baseline 重算结果 | val + test 两 split 的 Set-Level / Multiset Macro-F1 / Collapse ratio 数(SA3 拿来对照) |
| §6 OPEN QUESTIONS | SA1' 实施中遇到的疑问(若有),让 MA5 回 |
| §7 给 SA2' 的 carry-over | (a) PYTHONPATH 写法 verbatim;(b) 启动命令(nohup);(c) 监屏建议(头 30 min);(d) 磁盘清理建议 |

**SA1' 不**: 启动 SA2'。等 MA5 review 中期报告后再开。

---

## §5 MV-Attention 完整代码骨架(SA1' 可粘贴)

**位置**: 替换 `/home/tcat/diffcsp_exp5/code/step2/spectrum_encoder.py` 中的 `class SpectrumEncoder` 整段

> ⚠️ **重要警告**: 本骨架中 **xmu/chi/feff 三分支的 conv/pool/linear 内部结构是占位**(MA5 写本 handoff 时未读 `spectrum_encoder.py.bak_exp4` 的逐行内容)。SA1' **必须**先 `cat` v1 改后版 + bak_exp4 看真实 Exp4 分支编码器结构(kernel size、channel 数、pool size、几层 conv 等),**只修改三处**:
>
> 1. `self.chi_encoder` 末尾最后一个 `nn.Linear(..., 128)` → `nn.Linear(..., 256)`
> 2. `self.feat_encoder` 末尾最后一个 `nn.Linear(..., 64)` → `nn.Linear(..., 256)`
> 3. 删除整个 `self.fusion = nn.Sequential(nn.Linear(448, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))` 块
>
> conv 层 / pool / 激活函数 / 中间 hidden 数全部**保留 Exp4 原值**。新增的是 MV-attention 组件(`mv_attn / mv_query / mv_layernorm / mv_proj / mv_residual_alpha`),它们在 v1 的 `center_emb` 之前实例化。
>
> 下面骨架中的 conv 结构(`Conv1d(1, 32, kernel_size=5, ...)` 等)**仅作示例**,SA1' 实施时**以 .bak_exp4 实际结构为准**。

```python
import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    Exp5 v2 三路 XAS 谱编码器 + MV-attention fusion + center conditioning。

    分支结构(每分支输出 256d,平衡)
    --------------------------------
    XANES xmu (B, 150)  → Conv1d-Pool-Linear → (B, 256)   view_xmu
    EXAFS chi (B, 200)  → Conv1d-Pool-Linear → (B, 256)   view_chi  ← v2: 升 128→256
    FEFF feat (B,  74)  → MLP                → (B, 256)   view_feff ← v2: 升  64→256

    MV-attention fusion(v2 替换 v1 的 cat→MLP fusion)
    -------------------------------------------------
    views = stack([view_xmu, view_chi, view_feff], dim=1)   → (B, 3, 256)
    q     = mv_query.expand(B, -1, -1)                       → (B, 1, 256)
    attn  = MultiheadAttention(num_heads=4)(q, K=views, V=views) → (B, 1, 256)
    fused = attn.squeeze(1) + 0.5 * views.mean(dim=1)        → (B, 256) (post-residual)
    fused = LayerNorm(fused)
    latent = Linear(256, 256)(fused)                         → (B, 256)

    Center conditioning(继承 v1 SA1)
    -------------------------------
    center_Z (B,) → nn.Embedding(95, 16) → (B, 16)           center_emb
    output = cat([latent, center_emb], dim=-1)               → (B, 272)

    Parameters
    ----------
    xmu_dim           : int, 默认 150
    chi_dim           : int, 默认 200
    feat_dim          : int, 默认 74
    latent_dim        : int, 默认 256       — fusion 输出 = MV-attention proj 出口
    n_center_elements : int, 默认 95        — Embedding 表大小(实测 max(Z)=94 + slot 0 padding)
    center_emb_dim    : int, 默认 16
    mv_num_heads      : int, 默认 4         — MV-attention head 数(每 head 64d)
    mv_residual_alpha : float, 默认 0.5     — residual 系数,固定不可学
    """

    def __init__(self,
                 xmu_dim=150, chi_dim=200, feat_dim=74,
                 latent_dim=256,
                 n_center_elements=95, center_emb_dim=16,
                 mv_num_heads=4, mv_residual_alpha=0.5):
        super().__init__()

        # ── XANES 分支(v2: 末端输出 256,与 Exp4 同) ──
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256),  # 末端 → 256
        )

        # ── EXAFS chi 分支(v2: 末端 128 → 256 ★) ──
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256),  # ★ Exp5 v2: was 128, now 256
        )

        # ── FEFF MLP 分支(v2: 末端 64 → 256 ★) ──
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 256),  # ★ Exp5 v2: was 64, now 256
            nn.SiLU(),
        )

        # ── Exp5 v2 主线 1: MV-attention fusion ──
        # learnable query, small init (防 attention 早期偏向某一 view)
        self.mv_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # PyTorch 标准 MHA, batch_first=True 让 Q/K/V 形如 (B, seq, dim)
        self.mv_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=mv_num_heads,
            batch_first=True,
        )

        # post-residual LN(用户拍板顺序: attn → squeeze → + residual → LN → Linear)
        self.mv_layernorm = nn.LayerNorm(latent_dim)
        self.mv_proj = nn.Linear(latent_dim, latent_dim)

        # 固定标量,不可学
        self.mv_residual_alpha = float(mv_residual_alpha)

        # ── Exp5 SA1 继承: center embedding ──
        self.center_emb = nn.Embedding(n_center_elements, center_emb_dim)

        # output_dim 属性
        self._latent_out_dim = latent_dim + center_emb_dim   # 256 + 16 = 272

    @property
    def output_dim(self) -> int:
        """Final SpectrumEncoder output dim (latent + center_emb)."""
        return self._latent_out_dim

    def forward(self, xmu_xanes, chi1, feff_feats, center_Z):
        """
        Parameters
        ----------
        xmu_xanes  : Tensor (B, 150)
        chi1       : Tensor (B, 200)
        feff_feats : Tensor (B, 74)
        center_Z   : LongTensor (B,)

        Returns
        -------
        Tensor (B, 272) = MV-attention-fused latent (256) ⊕ center_emb (16)
        """
        # 三 view 各 (B, 256)
        view_xmu  = self.xmu_encoder(xmu_xanes.unsqueeze(1))   # (B, 256)
        view_chi  = self.chi_encoder(chi1.unsqueeze(1))         # (B, 256)
        view_feff = self.feat_encoder(feff_feats)               # (B, 256)

        # Stack to (B, 3, 256) for MHA
        views = torch.stack([view_xmu, view_chi, view_feff], dim=1)  # (B, 3, 256)

        # Cross-attention with learnable query
        B = views.shape[0]
        q = self.mv_query.expand(B, -1, -1)                    # (B, 1, 256)
        attn_out, _ = self.mv_attn(q, views, views, need_weights=False)  # (B, 1, 256)
        attn_out = attn_out.squeeze(1)                         # (B, 256)

        # Post-residual LayerNorm + projection
        fused = attn_out + self.mv_residual_alpha * views.mean(dim=1)  # (B, 256)
        fused = self.mv_layernorm(fused)
        latent = self.mv_proj(fused)                            # (B, 256)

        # 继承 v1: center embedding cat 末尾
        center_e = self.center_emb(center_Z)                    # (B, 16)
        return torch.cat([latent, center_e], dim=-1)            # (B, 272)
```

**SA1' 实施备注**:

1. `mv_query` 用 `torch.randn(...) * 0.02` 小初始化,防 attention 早期偏向某一 view
2. `nn.MultiheadAttention` 在 PT 2.4.1 中默认 `dropout=0.0` 是 OK 的,SA1' 不必显式设置
3. `batch_first=True` 让 Q/K/V 形如 `(B, seq, dim)`,与本骨架契合;**SA1' 不许改 `batch_first=False`**
4. 在 hydra 实例化时,如 yaml 用 `mv_attention.num_heads` 嵌套字段,SA1' 在 `diffusion_w_type_xas.py` 内 SpectrumEncoder 实例化处需要把 `cfg.mv_attention.num_heads` 平铺成 `mv_num_heads=cfg.mv_attention.num_heads` 传入 — 即:

```python
# 在 CSPDiffusion.__init__ 内的 SpectrumEncoder 实例化处:
mv_cfg = self.hparams.get('mv_attention', {})
self.spectrum_encoder = SpectrumEncoder(
    xmu_dim    = self.hparams.get('xmu_dim',   150),
    chi_dim    = self.hparams.get('chi_dim',   200),
    feat_dim   = self.hparams.get('feat_dim',  74),
    latent_dim = self.hparams.get('spectrum_latent_dim', 256),
    n_center_elements = self.hparams.get('n_center_elements', 95),
    center_emb_dim    = self.hparams.get('center_emb_dim', 16),
    mv_num_heads      = int(mv_cfg.get('num_heads', 4)),
    mv_residual_alpha = float(mv_cfg.get('residual_alpha', 0.5)),
)
```

---

## §6 Set-Level TypeAcc 与 Multiset Macro-F1 算法精确定义(用户拍板方案 B)

> **命名澄清**: 本节的 "6.1, 6.2, ..., 6.5" 是**指标算法定义编号**,与 §4 的 SA1' 工作步骤编号 "6.1-6.10" 不同名空间。本节 5 个小节都是 Step 6.7 的实施细节(那一步要 SA1' 实现这 4 个函数 + 1 个输出格式)。

### 6.1 Set-Level TypeAcc(per-sample,然后 dataset-level 平均)

**意图**: 衡量"每个样本里有多少元素能配上",与坐标完全解耦。

**算法**:

```python
def compute_set_level_typeacc(pred_types: np.ndarray, true_types: np.ndarray) -> float:
    """
    Per-sample Set-Level TypeAcc = sum_c min(pred_count_c, true_count_c) / N

    数学等价表述: 多重集交集大小 / N。
    与 Hungarian-on-type 在 |pred|=|true|=N 且 cost = 1{type_a != type_b} 时等价。

    Parameters
    ----------
    pred_types : (N,) int  predicted Z values, N=20 in this experiment
    true_types : (N,) int  ground-truth Z values

    Returns
    -------
    float in [0, 1], higher is better
    """
    from collections import Counter
    pred_cnt = Counter(pred_types.tolist())
    true_cnt = Counter(true_types.tolist())
    # Multiset intersection
    intersection = sum(
        min(pred_cnt[c], true_cnt[c])
        for c in (set(pred_cnt) | set(true_cnt))
    )
    N = len(true_types)
    return intersection / max(N, 1)


def compute_set_level_typeacc_dataset(all_pred_types: list, all_true_types: list) -> dict:
    """Dataset-level: per-sample Set-Level TypeAcc 的均值 + std"""
    vals = [
        compute_set_level_typeacc(p, t)
        for p, t in zip(all_pred_types, all_true_types)
    ]
    import numpy as np
    return {
        'set_level_typeacc_mean': float(np.mean(vals)),
        'set_level_typeacc_std':  float(np.std(vals)),
        'n_samples':              len(vals),
    }
```

**虚假指标对比**: 对 Fe-only Exp2 数据,position-by-position TypeAcc 可被"全猜 O"虚报到 ~ 0.60(Exp3 §1.1)。Set-Level TypeAcc 在同样情况下也会受 majority class 影响(若全猜 O,multiset 交集 = min(20, true_O_count) 也很大),**但** 它至少跟 prediction 实际多重集一致,不被坐标配对错误污染。**真正能曝光 majority bias 的是 §6.2 的 Multiset Macro-F1**。

### 6.2 Multiset Macro-F1(dataset-level,across element classes)

**意图**: 暴露系统性偏向多数类的塌缩(Exp3 §1.1 警觉的"全猜 O 虚报"问题)。

**算法**:

```python
def compute_multiset_f1_macro(all_pred_types: list, all_true_types: list,
                               eps: float = 1e-9) -> dict:
    """
    Dataset-level Multiset Macro-F1 across element classes.

    For each class c that appears in any true sample:
        TP_c = sum_samples min(pred_count_c, true_count_c)
        FP_c = sum_samples max(0, pred_count_c - true_count_c)   = sum_samples (pred_c - min)
        FN_c = sum_samples max(0, true_count_c - pred_count_c)   = sum_samples (true_c - min)
        precision_c = TP_c / (TP_c + FP_c + eps)
        recall_c    = TP_c / (TP_c + FN_c + eps)
        F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

    Macro-F1 = mean over all classes that appear in any true sample.

    Why this exposes majority bias:
      Suppose pred always = [O, O, ..., O] (20 O's). True has avg 12 O / 8 non-O.
      For class O:    TP=12 per sample → precision = 12/20 = 0.6, recall = 12/12 = 1.0, F1 = 0.75
      For class Fe:   TP=0 per sample  → precision = 0,           recall = 0,           F1 = 0
      For class Cu:   TP=0 per sample  → F1 = 0
      ...
      Macro-F1 over 30 element classes ≈ 0.75/30 = 0.025
      → 极低 macro-F1 暴露塌缩到 majority class

    Parameters
    ----------
    all_pred_types : list of (20,) int arrays
    all_true_types : list of (20,) int arrays

    Returns
    -------
    dict:
      'multiset_macro_f1':       float
      'per_class_f1':            dict[int, float]      Z → F1
      'per_class_precision':     dict[int, float]
      'per_class_recall':        dict[int, float]
      'per_class_support_true':  dict[int, int]        Z → total true count over dataset
      'per_class_support_pred':  dict[int, int]        Z → total pred count over dataset
      'n_classes_evaluated':     int                   (only classes appearing in true)
    """
    from collections import Counter, defaultdict
    import numpy as np

    # 累计每类的 TP / FP / FN 跨整个 dataset
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support_true = defaultdict(int)
    support_pred = defaultdict(int)
    classes_in_true = set()

    for p, t in zip(all_pred_types, all_true_types):
        p_cnt = Counter(p.tolist())
        t_cnt = Counter(t.tolist())
        all_classes = set(p_cnt) | set(t_cnt)
        for c in all_classes:
            pc = p_cnt[c]
            tc = t_cnt[c]
            tp[c] += min(pc, tc)
            fp[c] += max(0, pc - tc)
            fn[c] += max(0, tc - pc)
            support_true[c] += tc
            support_pred[c] += pc
            if tc > 0:
                classes_in_true.add(c)

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    for c in classes_in_true:  # 只对在 true 中出现过的类做 macro 平均
        prec = tp[c] / (tp[c] + fp[c] + eps)
        rec  = tp[c] / (tp[c] + fn[c] + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        per_class_precision[c] = prec
        per_class_recall[c] = rec
        per_class_f1[c] = f1

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

    return {
        'multiset_macro_f1':      macro_f1,
        'per_class_f1':           per_class_f1,
        'per_class_precision':    per_class_precision,
        'per_class_recall':       per_class_recall,
        'per_class_support_true': dict(support_true),
        'per_class_support_pred': dict(support_pred),
        'n_classes_evaluated':    len(classes_in_true),
    }
```

### 6.3 Collapse Ratio(per-sample)

```python
def compute_collapse_ratio(all_pred_frac: list, all_true_frac: list,
                            L: float = 6.0, threshold: float = 0.5) -> dict:
    """
    Per-sample collapse detection following proposal v2 §5.5.

    is_collapsed if pred std (averaged over 3 axes, in cartesian Å) is
    less than threshold * true std.

    Parameters
    ----------
    all_pred_frac : list of (20, 3) float arrays   (frac coords ∈ [-0.5, 0.5])
    all_true_frac : list of (20, 3) float arrays
    L          : box edge in Å (= 6.0)
    threshold  : default 0.5 (= "pred std < half of true std → collapsed")

    Returns
    -------
    dict:
      'collapse_ratio': float in [0, 1]
      'n_collapsed':    int
      'n_total':        int
      'pred_std_dist':  list of float — per-sample pred xyz std (Å), for histogram
      'true_std_dist':  list of float — per-sample true xyz std (Å), reference
    """
    import numpy as np
    n_total = len(all_pred_frac)
    n_collapsed = 0
    pred_std_dist = []
    true_std_dist = []

    for p, t in zip(all_pred_frac, all_true_frac):
        # frac → cartesian (Å) via L scaling
        pred_xyz_std = np.std(p * L, axis=0).mean()  # avg per-axis std
        true_xyz_std = np.std(t * L, axis=0).mean()
        pred_std_dist.append(float(pred_xyz_std))
        true_std_dist.append(float(true_xyz_std))
        if pred_xyz_std < threshold * true_xyz_std:
            n_collapsed += 1

    return {
        'collapse_ratio': n_collapsed / max(n_total, 1),
        'n_collapsed':    n_collapsed,
        'n_total':        n_total,
        'pred_std_dist':  pred_std_dist,
        'true_std_dist':  true_std_dist,
    }
```

### 6.4 投影 Ablation(SA3 用,SA1' 实现 helper 函数)

```python
def compute_projection_ablation_rmsd(all_pred_frac: list, all_true_frac: list,
                                      R_max_angstrom: float, L: float = 6.0) -> dict:
    """
    proposal v2 §5.4 的诊断 ablation。

    将每个样本预测原子中"笛卡尔距原点 > R_max"的原子,投影到 R_max shell 表面,
    保留方向。然后用 Hungarian min-image 重算 RMSD。

    Parameters
    ----------
    all_pred_frac : list of (20, 3) frac coords
    all_true_frac : list of (20, 3) frac coords
    R_max_angstrom : float — 投影外缘半径(Å);SA3 通常取训练真实距离 99 percentile
    L : float = 6.0

    Returns
    -------
    dict:
      'rmsd_before':     float   (Å,无投影,常规 Hungarian RMSD)
      'rmsd_after':      float   (Å,投影后,Hungarian RMSD)
      'rmsd_delta':      float   = rmsd_before - rmsd_after  (正值 = 投影改善)
      'n_atoms_projected_avg': float — 平均每样本被投影的原子数
    """
    import numpy as np
    # SA1' 实现细节: 对每个样本,
    #   1. p_xyz = p_frac * L  (cartesian)
    #   2. norms = np.linalg.norm(p_xyz, axis=-1)  (20,)
    #   3. mask = norms > R_max_angstrom
    #   4. p_xyz[mask] = p_xyz[mask] * (R_max_angstrom / norms[mask][:, None])  # 投影到球壳
    #   5. p_frac_projected = p_xyz / L
    #   6. 调用现有 Hungarian min-image RMSD 函数算 rmsd_after
    # rmsd_before: 直接调 Hungarian min-image RMSD
    # SA1' 把这写成 helper, SA3 在评估时调用
    ...
```

SA3 期决定 `R_max_angstrom`:从 `shell_boundaries.pkl` 读训练真实距离 99 percentile,大致 5.5 Å 左右(L=6 box,理论上限 √3 × 3 = 5.2 Å)。SA1' 不必跑此函数,只确保函数签名 + 实现正确,SA3 拿来调。

### 6.5 step5_2 monitor 输出格式

`metrics_report_<split>.txt` SA1' 改造后的输出模板:

```
=========================================================
EXP5 V2 METRICS REPORT — split: <val|test|holdout>
=========================================================
Total samples:        <N>
Effective samples:    <N_eff> (silent_drop count: <N - N_eff>)

--- Geometry (主面板,与 Exp4 对比) ---
RMSD (Å):             1.xxxx ± 0.xxxx
pred_in_cutoff (/20): xx.xx
true_in_cutoff (/20): xx.xx (reference)

--- Type metrics (Exp5 v2 主面板,真信号) ---
Set-Level TypeAcc:    0.xxxx ± 0.xxxx   (per-sample multiset intersection / 20)
Multiset Macro-F1:    0.xxxx            (dataset-level, across <K> element classes)
Collapse Ratio:       x.x%              (n_collapsed / n_total at threshold=0.5)

--- Type metrics (历史对照,Exp3 已证为虚假指标,仅供回溯) ---
Position-by-position TypeAcc: 0.xxxx    [VIRTUAL METRIC — DO NOT USE]

--- Top-10 element classes by support (Multiset F1 detail) ---
  Z=8  (O):    F1=0.xxx   support_true=xxxx   support_pred=xxxx
  Z=26 (Fe):   F1=0.xxx   support_true=xxxx   support_pred=xxxx
  ...
  Z=14 (Si):   F1=0.xxx   support_true=xxxx   support_pred=xxxx
=========================================================
```

---

## §7 SA1' 中期报告 deliverable 清单(展开 6.10)

SA1' 写到 `/home/tcat/diffcsp_exp5/EXP5_STEP1_PRIME_OUTPUT.md`,**MA5 review 后才开 SA2'**。

### 7.1 必备内容

| 节 | 内容 | 来源 |
|---|---|---|
| §0 完成度速览 | 10 步任务表(对照本 handoff §0.4) | SA1' 自填 |
| §1 改了什么 | 8 个文件 diff 摘要(7 旧 + 1 新):每文件改前→改后行数 + 关键改动概述 | git diff 风格 |
| §2 静态 PASS gate 实测 | 6.1-6.5 各 grep / python -c 命令 + 输出 | 实跑 |
| §3 forward_test 完整 log | `step1_forward_test_v2.log`,5/5 PASS + 1 SKIPPED 末尾 verbatim | log 文件路径 + 关键 phase 数 |
| §4 smoke test 完整 log | `step1_smoke_v2.log`,SMOKE PASS 末尾 + epoch 0/1/2 loss 数 | 同上 |
| §5 Exp4 baseline 重算结果 | val + test 两 split 的: Set-Level TypeAcc / Multiset Macro-F1 / Collapse Ratio / per_class_F1 top-10 | `exp4_baseline_<split>_metrics.txt` |
| §6 OPEN QUESTIONS | 实施中遇到的疑问(若有) | SA1' 视情况 |
| §7 给 SA2' 的 carry-over | (a) PYTHONPATH 完整命令(本 handoff §1.2);(b) nohup 启动命令(本 handoff §6.6 F);(c) 监屏建议:头 30 min 看 val_loss 不能从 ~ 2.4 起跳到 > 4 不回落;(d) 磁盘清理(`/tmp/diffcsp_cache/` + 旧 wandb + smoke ckpt) | SA1' verbatim 抄 |

### 7.2 必备 diff 抓取(SA1' 跑命令贴 log)

```bash
cd /home/tcat/diffcsp_exp5/code

# 6 文件 改后 vs bak 锚点
for f in step2/spectrum_encoder.py \
         step3/diffusion_w_type_xas.py \
         step3/conf_xas/model/diffusion_xas.yaml \
         step3/forward_test.py; do
  echo "============ diff: $f (v2 vs bak_exp4) ============"
  diff -u "$f.bak_exp4" "$f" | wc -l
  echo "---"
done

# 2 新建文件 行数
wc -l step4/step4_2_train.py step5/step5_2_compute_metrics.py 2>&1
```

### 7.3 SA1' 不做(给 SA2' 留)

- 不跑正式训练
- 不动 holdout
- 不做 Phase 6.5 修复(永久 SKIPPED-by-design)

---

## §8 关键 carry-over 与红线汇总(末尾速查)

| # | 项 | 出处 |
|---|---|---|
| 1 | PYTHONPATH 顺序: Exp5 step3/step2 在前,Exp4 code 末尾 | v1 SA1 OUTPUT §5.6 |
| 2 | Phase 6.5 SKIPPED-by-design,3 处 fp32 site 不修 | v1 SA1 OUTPUT §5.7 |
| 3 | Patch 1 `.to(c0.dtype)` verbatim 保留 | 同上 |
| 4 | `cost_density` 不许 0,锁 0.2 | proposal v2 §2 |
| 5 | 不加 head,position-by-position 不进主面板 | proposal v2 §2 + Exp3 §1.1 |
| 6 | 不 fine-tune from Exp4 ckpt,from-scratch | proposal v2 §2 |
| 7 | 不读 holdout(SA3 期才解禁) | EXP4_FILE_GUIDE §7 |
| 8 | SA1' **禁止启动正式训练**,只做静态/smoke 验证 | proposal v2 §6.4 |
| 9 | 7 守卫包不升级 | EXP4_FILE_GUIDE §7 |
| 10 | `_phase_65_legacy()` 函数不动 | v1 SA1 OUTPUT §5.7 |
| 11 | `_density_loss` 函数实现不动(只改 yaml 权重) | proposal v2 §3.5 |
| 12 | MV-attention `num_heads=4` / `residual_alpha=0.5` 锁死 | proposal v2 §3.1 |
| 13 | `mv_residual_alpha` 不许做成可学 | 同上 |
| 14 | `batch_first=True` 不许改 | 本 handoff §5 |

---

## §9 后续(SA2' / SA3 / SA4)预告

SA1' 不需关心,仅作上下文:

- **SA2'**: 后台 nohup ~ 32h from-scratch 训练。监主指标 `val_loss`、bonus 指标 `val_set_level_typeacc` / `val_multiset_f1_macro`。MA5 头 30 min 守屏 + SA2' 关 ssh 等。
- **SA3**: sample val/test (~ 9h) → 算 Set-Level / Multiset / Collapse / **投影 ablation**(本 handoff §6.4 函数)→ 红绿灯 → 解禁 holdout 重算 → 出 metrics_report_*.txt。
- **SA4**: 6 figure 重画(Exp5 vs Exp4 overlay)+ collapse 比例统计 + Exp6 决议建议。

---

## §10 SA1' 启动 checklist

转给 SA1' 时,SA1' 第一条回复应:

1. 复述本 handoff §0.4 的 10 步任务表
2. 复述 §2 红线(至少 5 条)
3. 复述 §1.2 PYTHONPATH 写法 + §1.3 Phase 6.5 状态
4. 列出**第一棒要从用户拿什么**:
   - `cat /home/tcat/diffcsp_exp4/code/step4/step4_2_train.py`(6.6 fork 模板)
   - `cat /home/tcat/diffcsp_exp4/code/step5/step5_2_compute_metrics.py`(6.7 fork 模板)
   - `ls -la /home/tcat/diffcsp_exp5/code/{step2,step3,step4}/` 确认 v1 改后文件 + bak_exp4 都在
5. 估时(应在 2-3 天)+ 第一日计划(优先级排序: 6.1 → 6.2 → 6.3 → 6.5 → 6.4 → 6.8 → ... 把 yaml/diffusion/spectrum_encoder 的 surgery 先打通 forward_test,再做 train/metrics)

---

*Exp5 v2 Main Agent (= MA5) 撰写,2026-04-28。基于 EXP5_PROPOSAL_v2 + EXP5_STEP1_OUTPUT (v1 SA1) + EXP4_FINAL_REPORT_ERRATA_2 + 用户拍板的 4 个决策点(Multiset F1 方案 B / post-residual LN / cost_type 注释还原 / smoke 改写 v1)。SA1' review 后 MA5 转发到新窗口。*
