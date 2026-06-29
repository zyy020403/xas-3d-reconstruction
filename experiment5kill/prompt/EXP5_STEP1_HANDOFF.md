# EXP5_STEP1_HANDOFF.md
# Exp5 Step 1 Sub-Agent 交接文档:架构改造(decoupled TypeClassifier head + center-element conditioning)

> **撰写者**: Exp5 Main Agent
> **日期**: 2026-04-28
> **接收人**: Exp5-SA1(架构改造 Sub-Agent)
> **交付方式**: 完成后产出 `EXP5_STEP1_OUTPUT.md` 上交 Main Agent,然后由 Main Agent 启动 Exp5-SA2(训练)

---

## 0. 你的使命(一句话)

把 Exp4 的 best ckpt(val_loss=0.7300)上"type prediction 与 diffusion decoder 耦合 + center 元素无显式注入"这两个被 Exp4 数据反证的设计,改造成 baseline_v2 架构,**通过 5/5 forward_test + smoke test 后交棒**。

不训练。**不读 holdout。** 不动 Exp4 的 `/home/tcat/diffcsp_exp4/` 任何文件——所有改动在新建的 `/home/tcat/diffcsp_exp5/` 下完成。

---

## 1. 必读背景(按这个顺序)

1. **EXPERIMENT4_FINAL_REPORT.md §7 三大发现 O1/O2/O3** — 这是你为什么要做这件事的依据
2. **EXPERIMENT4_FINAL_REPORT.md §10 方向 1 + 方向 2** — 这是你要实现的具体设计意图
3. **EXPERIMENT4_FINAL_REPORT.md §9.1 + §9.2** — 文件位置 + 读取方式参考代码
4. **EXPERIMENT2_FINAL_REPORT.md §3.2 direction A** — 早期对 decoupled head 的论述

不读 EXP4_PROPOSAL_v2.md(那是 Exp4 的)。**特别注意**:Exp4 PROPOSAL §1.3 写的"不加 TypeClassifier head"已被 Exp4 数据反证(详见 Exp4 final report §7.3 O3),Exp5 必须加。

---

## 2. 改动设计(意图 + 关键 shape,**不是抄的代码**)

### 2.1 架构概览(与 Exp4 对比)

```
                         Exp4(当前)                              Exp5 baseline_v2(你要改成的)
                         ─────────────                            ─────────────────────────────
   xmu (150)  ──┐                                                xmu (150)  ──┐
   chi1 (200) ──┼─ SpectrumEncoder ──── latent (256) ──┐         chi1 (200) ──┼─ SpectrumEncoder ──── latent (256) ──┐
   feff (74)  ──┘                                       │         feff (74)  ──┘                                       │
                                                        │                                                              │
   center_Z(没有显式注入,只在 feff 一维 one-hot)         │         center_Z ── nn.Embedding(89, 16) ──── center_emb (16)│
                                                        ↓                                                              ↓
                                            [latent → CSPDiffusion]                              [concat: latent ⊕ center_emb] (272)
                                                        │                                          │            │
                                                        ↓                                          ↓            ↓
                                          Output:(frac_coords, atom_types)              CSPDiffusion          TypeClassifierHead
                                            (二者由同一 decoder 共担)                          ↓                  ↓
                                                                                       frac_coords (B,20,3)   logits (B, 20, 89)
                                                                                       atom_types (B,20)        ↓
                                                                                       (按 dataset slot 顺序)   slot k ↔ dataset slot k
                                                                                       训练: slot-aligned CE(无 Hungarian)
                                                                                       Eval: 见 §2.3
```

### 2.2 改动一览表

| # | 文件 | 改动 | 关键 shape / 常量 |
|---|------|------|------|
| **1** | `xas_local_dataset_v2.py` | **基于 Exp4 当前用版本(`/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py`,不是 `.bak_phase46`)cp 后改**。Data 对象多吐一个字段 `center_element_Z` (int8 / int32),从 `data_inventory_v2.csv` 的 `center_element` 列经 `pymatgen.core.Element(symbol).Z` 转换得来。**Phase 4.6 的两处 `return None` 和 collate filter 一行不动**。 | `data.center_element_Z` shape: `(1,)` per sample,collate 后 `(B,)` |
| **2** | `spectrum_encoder.py` | (a) 新建 `nn.Embedding(89, 16)`(89 = 88 元素 + 0-padding,Z=1 到 Z=92 用 89 够吗?**注意:Z 最大值要确认,见 §3.1 sanity check**)<br>(b) `forward()` 接收 `center_Z`,lookup embedding,与 latent concat | center_emb_dim = 16(可调,但起步 16);output dim 从 256 → 272 |
| **3** | `diffusion_w_type_xas.py` | (a) 新建 `TypeClassifierHead` 类:`Linear(272, 512) → SiLU → Linear(512, 20*89) → reshape (B, 20, 89)`<br>(b) `forward()` 多 return `head_logits`<br>(c) 训练 loss 用 slot-aligned CE(详见 §2.3,**无 Hungarian**)<br>(d) loss 聚合按 `type_loss_mode` 三模式分流(详见 §2.4)<br>(e) training/validation_step 分别 log `loss_diffusion_coord`、`loss_diffusion_type`、`loss_type_ce_head`、`loss_total`<br>(f) 暴露 `head_predict_types(latent, center_emb) → (B, 20)` 整数 tensor 方法供 SA3 调用 | head 输出 (B, 20, 89);三个 type-loss-mode 都要支持 |
| **4** | `xas_local_dataset_v2.py`(再补) | true atom_types 必须按距离 sort 输出(rank 1 = 最近)。**这点 Exp4 应该已经是,但你必须 verify** | `data.frac_coords` 与 `data.atom_types` 必须按距离同序 |
| **5** | `conf_xas/model/diffusion_xas.yaml` | 新增字段:`center_emb_dim: 16`、`type_head: {hidden_dim: 512, n_elements: 89}`、`type_loss_mode: both`、`diffusion_type_weight: 1.0`、`head_type_weight: 0.5`(详见 §2.4) | YAML 整洁,与代码字段对齐 |
| **6** | `forward_test.py` | 加新 phase / 断言:(a) head_logits.shape == (B, 20, 89);(b) loss_type_ce_head 是 finite scalar;(c) center_Z=真值 vs center_Z=0 时 head_logits 不相同(证明 conditioning 真起作用);(d) **三种 `type_loss_mode` 各 forward 一次,total_loss 都 finite**(快速验证 mode switching 不 NaN) | 5/5 PASS 的 5 = 原 5 + 新断言合并;**至少 5/5,推荐扩到 8/8** |

### 2.3 训练 vs 评估的 atom-slot 顺序(关键,SA1 必须读完再写代码)

**训练时(slot-aligned,无 Hungarian)**:

dataset 输出的 20 个 atom 是按距中心距离 sort 后的固定顺序。Diffusion 反扩散在那 20 个 slot 上做 denoising,所以 pred 的 slot k ↔ dataset 的 slot k(同 ground truth)。Head 输出的 logits[:, k, :] 也对应 slot k。**所以训练时 head 的 type loss 直接按 atom-slot 算 CE,不需要 Hungarian 介入训练循环**(那会非常慢且无意义)。

```python
# 训练时(伪代码)
head_logits = type_head(latent_with_center)        # (B, 20, 89)
true_types  = data.atom_types                       # (B, 20),已按距离 sort
loss_type_ce_head = F.cross_entropy(
    head_logits.reshape(-1, 89),                    # (B*20, 89)
    true_types.reshape(-1),                          # (B*20,)
)
```

**评估时(Hungarian,只在 sample/eval 路径)**:

Sample 完成后,pred coords 可能 drift 到与 dataset slot 顺序完全不对应的位置(尤其 collapse 模式下)。所以 evaluation 必须用 Hungarian:

```python
# Sample 后(伪代码,SA3 真正实现)
pred_coords = diffusion.sample(...)                 # (20, 3)
head_argmax = head_logits.argmax(dim=-1)            # (20,) 整数 element Z
# Hungarian: 用坐标距离作 cost matrix
cost = pairwise_distance(pred_coords, true_coords)  # (20, 20),min-image
row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
# TypeAcc 用 Hungarian 配对结果(不是按 rank!)
type_acc = mean([head_argmax[row_ind[k]] == true_types[col_ind[k]] for k in range(20)])
rmsd = sqrt(mean([cost[row_ind[k], col_ind[k]]**2 for k in range(20)]))
```

**SA1 你的责任**:确保 head 在 forward 路径上的输入 / 输出 shape 正确,training_step 里 loss 用 slot-aligned CE。Eval 路径(Hungarian + head argmax 替换 diffusion type)由 SA3 写,**你不写 eval 逻辑,但要在 inference 路径里暴露一个 `head_predict_types(latent, center_emb) → (B, 20)` 整数 tensor 方法供 SA3 调用**。

### 2.4 Diffusion 内部 type prediction 与 head 的 3 模式 ablation flag(yaml 决定)

Exp4 的 `diffusion_w_type_xas.py` 已经有 type prediction 在 diffusion decoder 里。Exp5 加 head 之后,两条路径都用同一个 ground truth 算 type loss,**有潜在竞争问题**(diffusion 已经在学 type,head 又独立学,可能互相挤压)。

为了避免一旦 default 不行就要重训,**SA1 必须在 yaml 里加 3 个 mode flag,代码里支持三种 mode**:

```yaml
# conf_xas/model/diffusion_xas.yaml(新增段)
type_loss_mode: both       # 三选一: "diffusion_only" / "head_only" / "both"
diffusion_type_weight: 1.0  # diffusion 内部 type CE 的权重(原 Exp4 默认 1.0)
head_type_weight: 0.5       # head 的 type CE 权重(λ_type)
```

**代码逻辑**(在 diffusion_w_type_xas.py 的 loss 聚合处):
```python
if cfg.type_loss_mode == "diffusion_only":
    # 不实例化 head,或 head 实例化但 head_type_weight=0
    loss_type_total = cfg.diffusion_type_weight * loss_diffusion_type
elif cfg.type_loss_mode == "head_only":
    # diffusion 内部 type loss 不进 total(权重置 0,但模块仍然 forward 以保持 backbone 不变)
    loss_type_total = cfg.head_type_weight * loss_type_ce_head
elif cfg.type_loss_mode == "both":
    loss_type_total = (cfg.diffusion_type_weight * loss_diffusion_type
                       + cfg.head_type_weight * loss_type_ce_head)
total_loss = loss_diffusion_coord + loss_type_total
```

**SA2 训练时 default `both`**。如果训练完 TypeAcc 不达预期,SA2-续 试 `head_only`(只改 yaml,不改代码,不重训也能 quick check evaluate-only ablation)。

**SA1 的实现要点**:
- 三种 mode 都要测试 forward 路径不 NaN(forward_test 各跑一次)
- `head_only` 模式下 `loss_diffusion_type` 仍要计算(只是不进 total),否则 diffusion 内部的 type prediction 模块梯度断流可能出意外行为

---

## 3. Sanity check(写代码前先做)

### 3.1 88 元素 → Z 范围范围

```python
import pandas as pd
df = pd.read_csv("/home/tcat/diffcsp_exp4/data/data_inventory_v2.csv")
print(df["center_element"].nunique())  # 应是 88
# 把 88 元素的 Z 取出来
from pymatgen.core import Element
zs = sorted({Element(e).Z for e in df["center_element"].unique()})
print(zs[:5], zs[-5:], "max:", max(zs))
# 决定 nn.Embedding 的 num_embeddings:
#  如果 max(Z) == M,用 nn.Embedding(M+1, 16)(Z=0 留作 padding)
#  注意 88 元素中可能跳号(没有 H/He/惰性气体等),embedding table 会有"空位"——无所谓,反正不会用到
```

**关键**:embedding table 大小 = `max(Z_used) + 1`,**不是 89**(89 是临时占位)。一般用 89 或 93 即可,反正只 +20 个空位。但 yaml 里写实际值。

### 3.2 数据 key 对齐(继承自 Exp4 PROPOSAL_v2 §7.1,但用 Exp4 已部署的数据)

```python
# 取随机 100 个 v2 sample,确认 4 个数据源都能 lookup
import pandas as pd, pickle
samples = pd.read_csv("/home/tcat/diffcsp_exp4/data/train_samples_v2.csv").sample(100, random_state=0)
spectra = pickle.load(open("/home/tcat/diffcsp_exp4/data/spectra_train.pkl", "rb"))
feff = pd.read_pickle("/home/tcat/diffcsp_exp4/data/feff_features_imputed.pkl")
shells = pickle.load(open("/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl", "rb"))
for sname in samples.sample_name:
    assert sname in spectra["name_to_idx"]
    assert sname in feff.index
    assert sname in shells
print("✓ 4 sources aligned")
```

### 3.3 atom_types 是否按距离 sort 输出

```python
# 加载已有 dataset(Exp4 版),查一个 sample
import sys; sys.path.insert(0, "/home/tcat/diffcsp_exp4/code/step3"); sys.path.insert(0, "/home/tcat/diffcsp_exp4/code/step2")
from xas_local_dataset_v2 import XasLocalDatasetV2
ds = XasLocalDatasetV2(split="val", data_dir="/home/tcat/diffcsp_exp4/data")
d = ds[0]
import torch
# frac_coords (20, 3),atom_types (20,)
# 算每个原子到原点的距离(笛卡尔):dist = ||frac_coords * L|| with L=6
dists = (d.frac_coords * 6.0).norm(dim=-1)
print("distances rank-1 to rank-20:", dists.tolist())
# 期望:单调递增。如不单调,SA1 必须在 dataset 输出时显式 sort
```

如果不单调,你需要在 dataset 的 `__getitem__` 末尾加 sort:
```python
order = dists.argsort()
frac_coords = frac_coords[order]
atom_types = atom_types[order]
```

---

## 4. 实施步骤

### 4.1 准备

```bash
# 在服务器 scsmlnprd02 上
mkdir -p /home/tcat/diffcsp_exp5/{code,checkpoints,logs}
cd /home/tcat/diffcsp_exp5/code

# 复制 Exp4 已 working 的代码骨架
cp -r /home/tcat/diffcsp_exp4/code/step2 .
cp -r /home/tcat/diffcsp_exp4/code/step3 .
cp -r /home/tcat/diffcsp_exp4/code/step4 .  # 训练脚本骨架,SA2 用
cp /home/tcat/diffcsp_exp4/code/.env .

# diffcsp 包、conf 顶层不动,从 Exp4 那边的 sys.path 引用即可
# 如果 import 路径里有 hardcode 的 /diffcsp_exp4/,改成 /diffcsp_exp5/

# 备份 baseline 代码(改之前)
cd /home/tcat/diffcsp_exp5/code/step3
cp xas_local_dataset_v2.py xas_local_dataset_v2.py.bak_exp4
cp diffusion_w_type_xas.py diffusion_w_type_xas.py.bak_exp4
cp forward_test.py forward_test.py.bak_exp4
cp conf_xas/model/diffusion_xas.yaml conf_xas/model/diffusion_xas.yaml.bak_exp4

cd /home/tcat/diffcsp_exp5/code/step2
cp spectrum_encoder.py spectrum_encoder.py.bak_exp4
```

### 4.2 数据路径

**复用 Exp4 的数据,不复制** —— `/home/tcat/diffcsp_exp4/data/` 全部沿用。在你的 dataset 代码里直接 hardcode 这个路径,或者在 yaml 里加 `data_dir: /home/tcat/diffcsp_exp4/data`。

**例外**:训练时需要 cache 到 `/tmp/diffcsp_cache/` (tmpfs RAM)以 IO 加速。这一步由 SA2 在训练入口做,你 Step 1 不必管。

### 4.3 改代码顺序(我建议这个顺序,你可以调整)

1. **先 dataset**:`xas_local_dataset_v2.py` 多吐 `center_element_Z`,verify atom_types 按距离 sort
2. **再 encoder**:`spectrum_encoder.py` 加 embedding + concat
3. **再 diffusion**:`diffusion_w_type_xas.py` 加 `TypeClassifierHead`、改 `forward()`、改 loss、改 logging
4. **最后 yaml**:补字段
5. **再 forward_test**:加新断言
6. **跑 forward_test.py**,跑 `step4_1_smoke_test.py`,各 PASS

### 4.4 接力链工作哲学(从 Exp4 继承)

- **诚实 > 流畅**:不确定的代码改动,在 OUTPUT.md 里写"我做了 X,理由是 Y,担心 Z"。Main Agent 会判断
- **70% 上下文闸门**:context 用到 70% 时停下来交付现状,**不要硬撑**
- **不深 debug**:如果改完报错且 30 min 没头绪,在 OUTPUT.md 里详细贴 stacktrace + 你的诊断假设,**交回 Main Agent**
- **状态锚定**:每一步代码改动后,跑 `git status` 或人工检查 .bak 锚点,确保你知道哪些文件改了哪些没改

---

## 5. 验收闸门(SA1 完成的硬标准)

**5/5 必须通过才能交棒**:

1. ✅ **forward_test.py 通过新版 5/5(或 7/7)PASS**,日志干净无 NaN/Inf/RuntimeError
2. ✅ **`step4_1_smoke_test.py` 跑通 5-10 batches 无 crash**,可见四个 loss 各自有合理量级:
   - `loss_diffusion_coord` ~1
   - `loss_diffusion_type` ~4-5(接近 ln(89)=4.49)
   - `loss_type_ce_head` ~4-5(同上,刚初始化时)
   - `loss_total` ≈ coord + 1.0 × diff_type + 0.5 × head_type ≈ 7-8
3. ✅ **center_Z conditioning 真起作用**:在 forward_test 里加一个断言——同一条 spectra,把 center_Z 从真值改成 0,head_logits 的 argmax 应该不同(至少 5/20 个 rank 上不同)。证明 conditioning 路径可微、有效
4. ✅ **head_logits shape (B, 20, 89) 确认**,且 type prediction 与 diffusion 的 atom_types 输出**并存**(不是替换);三种 `type_loss_mode` 都能 forward 不 NaN
5. ✅ **写完 EXP5_STEP1_OUTPUT.md**,内容包含:
   - 改了哪些文件、每个文件改了什么(diff 摘要)
   - 5 个验收闸门各自的实测证据(forward_test 日志末尾、smoke test 四个 loss 数值、center_Z 改动 ablation 结果、三 mode forward 各自的 loss)
   - 你做出的所有 yaml 默认值 / 实现细节决定(center_emb_dim=16、head_hidden_dim=512、`type_loss_mode=both`、各权重值)
   - 任何你不确定 / 想让 Main Agent 拍板的事(用"OPEN QUESTION:"明确标记)
   - **§Notes_for_SA2**:必须包含以下三条信息(详见 §8 Notes for SA2):
     - ckpt warm-start 入口建议(strict=False 加载 + head 模块需独立初始化)
     - **phased training 警告**(§8 详述,这是关键 risk flag)
     - 三 mode flag 怎么切(SA2 default 用 `both`,如果 TypeAcc 不达预期 SA2-续 试 `head_only`)

---

## 6. 红线(绝对不能动)

- `holdout_samples_v2.csv` / `spectra_holdout.pkl`:**全程不读**
- `incompat_pool.csv`:**封存**
- L=6, min-image, [-0.5, 0.5] coord, cost_lattice=0, N_NEIGHBORS=20:**全部继承,不动**
- silent drop + collate filter (`xas_collate_fn_v2`):**继承,不改回 raise**
- precision=fp32(MA4 D1 决策):**继承,不试 bf16**
- `/home/tcat/diffcsp_exp4/`:**read-only**,不写不删
- 网络环境守卫包(7 个核心 + 18 个子依赖):**不升级**(详见 Exp4 final report §9.3)

---

## 7. 关键文件路径速查

### 7.1 输入(read-only)

| 类别 | 路径 | 用途 |
|------|------|------|
| Exp4 best ckpt | `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` | 不要在 SA1 加载,SA2 才用。但你需要在 OUTPUT.md 里建议 SA2 怎么 strict=False 加载 |
| 数据(全套) | `/home/tcat/diffcsp_exp4/data/` | dataset 直接 lookup,**不复制** |
| Exp4 代码骨架 | `/home/tcat/diffcsp_exp4/code/{step2,step3,step4}/` | cp 后改 |
| .env | `/home/tcat/diffcsp_exp4/code/.env` | cp,如果有 path hardcode 改 |

### 7.2 输出(写到 `/home/tcat/diffcsp_exp5/`)

| 路径 | 内容 |
|------|------|
| `code/step2/spectrum_encoder.py` | 改后版本(含 center embedding) |
| `code/step3/xas_local_dataset_v2.py` | 改后版本(吐 center_Z + verify sort) |
| `code/step3/diffusion_w_type_xas.py` | 改后版本(含 TypeClassifierHead + 三 loss) |
| `code/step3/conf_xas/model/diffusion_xas.yaml` | 改后版本(含新字段) |
| `code/step3/forward_test.py` | 改后版本(5-7 phases) |
| `code/step3/*.bak_exp4` | 备份(不动) |
| `code/EXP5_STEP1_OUTPUT.md` | **你的交棒文档** |
| `logs/step1_forward_test.log` | forward_test 输出 |
| `logs/step1_smoke.log` | smoke test 输出 |

---

## 8. Notes for SA2(SA1 必须 carry over 进 EXP5_STEP1_OUTPUT.md)

这一节的内容 SA1 不实现,但**必须原样写进 OUTPUT.md 的 §Notes_for_SA2 段落**,确保 SA2 看到。Main Agent 不会替你转达。

### 8.1 ⚠️ Fine-tune phased training 警告(关键 risk flag)

Exp5 baseline_v2 从 Exp4 best ckpt(`best-epoch366-val0.7300.ckpt`)warm-start。**新加的 `TypeClassifierHead` 和 `center embedding` 是随机初始化的**。如果 SA2 直接 unfreeze 全部 + 用 lr=1e-4 训练:

- head 的初始 gradient 会非常大(随机权重 + CE on 89 类初始 loss ~4.5)
- 这些大 gradient 会通过 backbone 传回去,**前 5-10 个 epoch 风险最高,可能扰乱已经收敛的 backbone weights**
- 表现会是:val_loss 在前几 epoch 反而比 0.7300 起点更差,然后慢慢恢复——但已经在好的 weights 上"绕了一圈",后面收敛点未必好于直接 fine-tune

**SA2 强烈建议采用 phased training**:

| Phase | epoch 范围 | 谁解冻 | lr |
|-------|-----------|--------|-----|
| Phase 1(head warmup) | 0 - 5 | 只 head + center_emb | head lr = 1e-3 或 1e-4 |
| Phase 2(joint fine-tune) | 6 - end | 全部 | head lr = 1e-4,backbone lr = 1e-5(differential) |

PL 实现可用 `freeze()` / `requires_grad_(False)` + 在 `on_train_epoch_start` 切换,或者两段式 `Trainer.fit()` 调用(更直观但要正确处理 ckpt 续接)。

如果 SA2 不愿做 phased,**至少要做 lr warmup**:前 1000-2000 steps lr 线性从 0 升到 1e-4,降低初始大 gradient 的破坏性。

### 8.2 ckpt warm-start 加载方式

```python
# SA2 训练入口建议:
ckpt = torch.load("/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt",
                  map_location="cpu", weights_only=False)
# Exp5 model 含新模块(head + center_emb),Exp4 ckpt 没有这些 keys
# 必须 strict=False
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
# 期望:missing 列表里有 type_head.* 和 center_emb.* 的 keys(因为 Exp4 ckpt 没有)
# 期望:unexpected 列表为空(Exp4 ckpt 里所有 key 都是 Exp5 backbone 应该有的)
# 如果 unexpected 非空 → SA1 改名了某个 backbone 模块,要回查
```

**SA2 必须打印 missing / unexpected 列表到训练日志开头**,作为加载正确性的存证。

### 8.3 三 mode flag 怎么用

- `type_loss_mode: both`(default):SA2 第一次训练用这个
- `type_loss_mode: head_only`:如果 baseline_v2 训完 TypeAcc 不达 0.30 目标,SA2-续 改这个 yaml + 重训(可能从 SA2 的 best 接着 fine-tune,而不是从 Exp4 ckpt)
- `type_loss_mode: diffusion_only`:理论上等价于 Exp4(只验证 SA1 没把 diffusion 弄坏),不期望训练用这个

### 8.4 验证 phased training 是否有效的诊断

SA2 训练时关注前 5 个 epoch 的 `val_loss_diffusion_coord`:如果它从 0.7300 起步快速上升(>0.85)且不回落,说明 backbone 被新 head 的 gradient 扰动了——是 phased training 没做好的信号。这种情况要回滚,改 phased。

---

## 9. 时间预算

预估 1-2 天 wall time。如果第二天结束还没 5/5 PASS,**不要硬撑**——按 §4.4 接力链哲学,在 OUTPUT.md 里写明现状交回 Main Agent,Main Agent 决定是分裂任务、降级方案,还是召唤 SA1-续。

---

## 10. 与 SA0(Multi-sample averaging quick win)的关系

SA0 完全独立,只用 Exp4 ckpt + Exp4 sample 脚本,把 K=1 → K∈{5,10,20} 改一下,跑 val 几百个样本看 RMSD/TypeAcc 提升幅度。**与 SA1 完全不冲突,可以并行启动**。

如果 Main Agent 决定先跑 SA0,你的 SA1 任务不变。

---

*Exp5 Main Agent 撰写,2026-04-28(v2,接受 4 个 flag 修订)。SA1 接收后请在 24h 内回 ack 并报当日 §3 sanity check 结果。*
