# Experiment 6 Phase 1 — SA1 Handoff
# DETR-style Set Prediction · 单谱 Phase 1 实现

> **From**: Exp6-MA1
> **To**: Exp6-SA1
> **日期**: 2026-04-29
> **状态**: 实施手册,SA1 收到后逐步执行
> **权威依据**: EXP6_PROPOSAL_v3.md(以下简称 *proposal*),其余文档为辅助

---

## 0. 怎么用这份文档

这份是你的执行手册。**读完整份再动手**——里面有 forbidden list 和 open questions,提前撞到不要自己改主意。

执行约定:
- 你在自己的 chat 里读这份 + EXP6_PROPOSAL_v3.md 是主要入口,其余文档按需 pull
- 凡是要在 S 服务器上跑的命令,**不是你自己跑**——你把命令贴给用户,让用户 ssh 执行,把 stdout 贴回你的 chat,你据此推进。**这与 MA1 的工作流一致**
- 每完成一步,**短报**给用户(一句话:"step X.Y 完成,产出 Z");MA1 在用户那边见到再决定下一步是否需要 review
- 不许自己脑补 schema、不许凭记忆写 transformer 架构、不许"为简化"省略 sanity check
- Phase 1 全程**禁止**写训练脚本(`step2.1_train.py` 是 SA2 的事)

---

## 1. SA1 Phase 1 Scope

### 1.1 你要交付的东西(Phase 1 完成的定义)

在 S 上 `/home/tcat/diffcsp_exp6/` 下产出以下文件,且 `step1.2_smoke_test.py` **5 样本通过 4 项硬检查**:

```
diffcsp_exp6/
├── _detr_reference/                         # 你 clone DETR 到这里(reference,不动)
├── shared/
│   ├── xas_local_dataset_v2.py              # 从 Exp4 直接 cp,零改动
│   ├── xas_local_datamodule_v2.py           # 从 Exp4 直接 cp,零改动
│   ├── exp6_element_vocab.json              # 你 build 出来的 vocab
│   ├── spectrum_tokenizer.py                # 改自 Exp4 spectrum_encoder.py(去末层 Linear)
│   ├── transformer.py                       # 从 DETR 直接 cp,零改动
│   ├── matcher.py                           # 改自 DETR matcher.py(~30 行改动)
│   ├── criterion.py                         # 改自 DETR detr.py SetCriterion(~50 行改动)
│   ├── detr_xas.py                          # 新写主模型,~200 行
│   └── eval_metrics.py                      # 新写,实现 proposal §7.1 五公式
├── step1/
│   ├── step1.0_build_vocab.py               # 一次性脚本,产 exp6_element_vocab.json
│   ├── step1.1_recompute_exp4_setlevel.py   # 一次性脚本,在 Exp4 predictions_val.pt 上算 Set-Level baseline
│   └── step1.2_smoke_test.py                # 5 样本 forward + matcher + loss + backward
└── EXP6_PHASE1_OUTPUT.md                    # 你的交付报告(见 §8 exit criteria)
```

### 1.2 你**不**做的事

- 训练脚本(SA2 territory)
- 完整 val/test 评估(SA2/SA3 territory,但 eval_metrics.py 要写好备用)
- holdout 检验(SA4 territory)
- 任何形式的"探索性"超参 sweep / loss design 改动
- 任何 Exp4/Exp5 服务器代码的修改(只允许 cp 出来再改自己的副本)

---

## 2. Phase 0:Setup(在你写第一行实现代码之前必须全部完成)

### 2.1 Step 0.1 — 验证 / 创建 Exp6 目录

让用户 ssh 跑:

```bash
ls -d /home/tcat/diffcsp_exp6 2>/dev/null || \
  mkdir -p /home/tcat/diffcsp_exp6/{shared,step1,step2,step3,step4}
ls /home/tcat/diffcsp_exp6
```

**期望输出**: 看到上面 4 个空子目录(`_detr_reference` 由 Step 0.2 创建)。

如果用户告诉你"已经 mkdir + clone 过了"(MA1 在你之前的对话里已经让用户开始过),不要重做,接着 Step 0.3 验证 clone。

### 2.2 Step 0.2 — Clone DETR

```bash
cd /home/tcat/diffcsp_exp6
[ -d _detr_reference ] || git clone https://github.com/facebookresearch/detr.git _detr_reference
ls _detr_reference/models/
```

**期望输出**: `backbone.py detr.py matcher.py position_encoding.py segmentation.py transformer.py __init__.py` 等。

> 注: DETR repo 已 archive(2024-03-12),但 default branch 可能仍是 `master` 或 `main`。`git clone` 默认拉 default branch,不必管。

### 2.3 Step 0.3 — Verify proposal §6.1 行号

proposal §6.1 把以下超参 trace 到了 DETR 仓库的具体行。**你必须 verify 一次,在 EXP6_PHASE1_OUTPUT.md 中标注 "verified" 或 "delta found"**(MA1 没做这步,你做)。

```bash
cd /home/tcat/diffcsp_exp6/_detr_reference
echo "=== main.py (proposal cited L40, L41, L42, L48, L51, L52, L88) ==="
sed -n '38,55p' main.py
echo "---"
sed -n '85,92p' main.py
echo ""
echo "=== detr.py L77 (empty_weight = self.eos_coef) ==="
sed -n '70,82p' models/detr.py
```

**SA1 在 OUTPUT 里要写**:每个 proposal §6.1 表行 → 实际命中的源码行号。一致就 ✓。如果 archive 后行号有偏移(几行内浮动很可能),记下偏移量,**不必修改 proposal**——只要源码语义正确即可。

### 2.4 Step 0.4 — Dump train_samples_v2.csv schema

让用户 ssh 跑:

```bash
cd /home/tcat/diffcsp_exp4/data
head -2 train_samples_v2.csv
echo "---"
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('train_samples_v2.csv')
print("shape:", df.shape)
print("dtypes:")
print(df.dtypes)
print("columns:", list(df.columns))
if 'center_element' in df.columns:
    print("center_element top 10:")
    print(df['center_element'].value_counts().head(10))
    print("n_unique:", df['center_element'].nunique())
EOF
```

**判定**: 看 dump 出来的列名,确认 vocab build 走哪条路:

| 你看到的 schema | step1.0 怎么写 |
|---|---|
| 有 `neighbor_types` 列(list 或 stringified list) | union 这一列即可建 neighbor_vocab |
| 只有 `center_element` + `mp_id` + `sample_name` 等 | neighbor_vocab 必须从 spectra/POSCAR 解析(见 §3.1) |

把 dump 结果记进 OUTPUT.md,选定的路径在 step1.0 实现里加注释引用 schema 行。

### 2.5 Step 0.5 — Dump predictions_val.pt schema

让用户 ssh 跑:

```bash
cd /home/tcat/diffcsp_exp4/code/step5
python3 << 'EOF'
import torch
d = torch.load('predictions_val.pt', weights_only=False)
print("type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys()))
    for k, v in list(d.items())[:5]:
        print(f"  {k}: type={type(v).__name__}", end="")
        if hasattr(v, 'shape'):
            print(f", shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f", len={len(v)}, first={v[0] if len(v) > 0 else 'empty'}")
        else:
            print(f", value={str(v)[:100]}")
elif isinstance(d, list):
    print("len:", len(d))
    print("d[0]:", d[0])
EOF
```

**判定**: 你需要确认 predictions_val.pt 里:
- 每个 sample 的 pred 类型存的是什么?**原子序数 Z** 还是 **dense 索引**?
- 类型张量的 shape 是 `(N_samples, 20)` 还是 `(20,)` per-sample?
- 是否包含 `pred_in_cutoff` 已计算值?(影响 step1.1 是否需要重算)

把 dump 结果记进 OUTPUT.md。step1.1 实现按实际 schema 写。

### 2.6 Step 0.6 — 阅读 DETR 4 个核心文件

不在 chat 里贴大段代码,你**自己 ssh `cat` 或 SCP 出来读**。下面是 MA1 已读过 transformer/matcher/detr 三份的关键 takeaway,你在 OUTPUT.md 中**报你自己读完后的认知**——若与 MA1 takeaway 有差,标出来。

| 文件 | 你必须确认的 |
|---|---|
| `models/transformer.py` | (1) `Transformer.__init__` 默认 d_model=512(我们改 256);(2) `forward(src, mask, query_embed, pos_embed)`,src 期望 `(N_seq, B, d_model)` seq-first;(3) `return_intermediate_dec=True` 时返回 `(num_layers, B, N_query, d_model)` 用于 aux loss |
| `models/matcher.py` | (1) `@torch.no_grad()` 装饰器**保留**;(2) cost = `cost_class * cost_class + cost_bbox * cost_bbox + cost_giou * cost_giou`(加权和);(3) `linear_sum_assignment` 在 `.cpu().numpy()` 上调用,per-batch |
| `models/detr.py` 的 `SetCriterion` | (1) `num_classes` 参数语义 = "**omit no_object**",no_object 落在 `num_classes` 索引位;(2) `empty_weight[-1] = eos_coef = 0.1`;(3) `loss_cardinality` 是 `@torch.no_grad()` 诊断,不传梯度——Exp6 的 `no_object_ratio` 与之等效但口径用 proposal §附录B.5 锁定的 |
| `models/position_encoding.py` | **只快速浏览**,不 cp 不用。Exp6 用 `nn.Embedding(num_tokens, d_model)` 替代——proposal §2.3 已决策 |

---

## 3. Phase 1:Implementation(顺序锁定,不许跳)

### 3.1 Step 1.0 — Build vocab

依据 proposal §4.1(c),实现 `step1/step1.0_build_vocab.py`,产出 `shared/exp6_element_vocab.json`。

**核心逻辑**:

1. 读 `train_samples_v2.csv`(路径见 EXP4_FILE_GUIDE.md §3.1)
2. `center_unique_Z = sorted(set(Element(e).Z for e in df['center_element'].unique()))`
3. `N_CENTER_TYPES = len(center_unique_Z)`,记录到 OUTPUT
4. **neighbor_vocab 怎么建**: 取决于 Step 0.4 dump 结果
   - 若 train CSV 有 neighbor_types 列 → union 直接拿
   - 若没有 → 走 fallback: 你写一次性脚本扫 `spectra_train.pkl` 或 POSCAR 文件夹,union 所有 `(sample, neighbor_atom_types)` pair。**仅扫 train,不扫 val/test/holdout**(避免泄漏)
5. assert `N_NEIGHBOR_TYPES >= N_CENTER_TYPES`,失败 raise
6. dump JSON 严格按 proposal §4.1(c) 给的 schema(含 `Z_to_idx`、`idx_to_Z`、`no_object_idx`)

**禁止**:
- 用 Exp5 v1 的 95-class sparse vocab(哪怕 S 上有 `/home/tcat/diffcsp_exp5/`,也不许 import / 参考 / cp)— proposal §3.2 明文
- 在 vocab 里塞 "all elements 1-94" 这种 "宽松起见多兜几个" 的写法,稀释损害 Exp4 数字可比性

**输出**: `shared/exp6_element_vocab.json` + OUTPUT.md 里报 `(N_CENTER_TYPES, N_NEIGHBOR_TYPES)` 实测值。

### 3.2 Step 1.1 — Recompute Exp4 Set-Level baseline

**为什么做这步**: proposal §10.1 明文 SA1 要在 smoke test 阶段重算 Exp4 Set-Level baseline,**否则 §10.1 主验收阈值无法落地**。预算 ~30 min(load + scalar 计算)。

**核心逻辑**:

1. `torch.load('/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt', weights_only=False)`
2. 按 Step 0.5 确认的 schema 取出 (pred_types, gt_types) per-sample
3. **关键**: Exp4 predictions 用 Z 直接索引(几乎肯定),你的 Set-Level 公式比较的是"两个 multiset 是否重合",vocab 不影响——直接拿 Z 跟 Z 比即可,**不需要先映射到 Exp6 的 dense vocab**
4. 应用 proposal §7.1 indicator 2(`set_level_type_acc`)**精确实现**(eval_metrics.py 里写的那一份;Step 1.0 之后写)
5. dataset-level 取 per-sample 平均 → scalar
6. **Exp4 prediction 的 valid_pred_mask**: Exp4 没有 no_object 概念,所有 20 个预测都"valid"。即 `valid_pred = pred_types_argmax`(无过滤)。把这一点在脚本里**注释明示**,SA2 review 时一眼看懂

**输出**: 一个 scalar `exp4_setlevel_typeacc_val` 报到 OUTPUT.md,作为 proposal §10.1 阈值 backfill 数字。

如果 schema 让 set_level 公式无法直接跑(例如 pred_types 是 logits 不是 argmax),先 argmax 再算,在 OUTPUT 中说明你做了一次 `argmax(-1)` 转换。

### 3.3 Step 1.2 — `eval_metrics.py`

**新写**,严格实现 proposal §7.1 的**全部 5 个指标 + 公共工具函数**。代码逐字符照 §7.1 实现,**不许重命名变量、不许"优化"**。

如果你看到 §7.1 的某个公式实现疑似有效率问题(比如 `min_image_l2` 用 `pred[:, None] - gt[None, :]` broadcast 占内存),**忍住**。proposal 是 contract,不是 suggestion。优化等 SA3 评估期再做。

**输出**: `shared/eval_metrics.py`,含:
- `min_image_l2(pred, gt, lengths)` 公共工具
- `hungarian_rmsd(...)` (indicator 1)
- `set_level_type_acc(...)` (indicator 2)
- `multiset_f1_macro(...)` (indicator 3)
- `in_cutoff_counts(...)` (indicator 4)
- `close_pair_type_acc(...)` (indicator 5)

每个函数加 docstring,引用 "proposal §7.1 indicator N"。

### 3.4 Step 1.3 — `spectrum_tokenizer.py`

1. ssh `cp /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py /home/tcat/diffcsp_exp6/shared/spectrum_tokenizer.py`
2. **唯一改动**: 去掉 forward 末尾的 `nn.Linear(latent_dim, latent_dim)`(如果存在),让最后输出就是 fusion 后的 256d
3. 保持 forward 签名 `forward(xmu, chi, feff)` → `(B, 256)`(上层 `unsqueeze(1)` 变 token)

**Exp4 文档与现实差异**(MA1 在交叉读 EXP4_FILE_GUIDE 时发现,SA1 cp Exp4 代码会撞到):
- proposal §3.1 写"feff 73 维",但 **Exp4 实际是 74 维**(FILE_GUIDE line 95: "spectrum_encoder.py 5 处 73→74";`feff_features_imputed.pkl` 是 (128382, 74))。看到 74 不要"修正回 73"
- proposal §3.1 写"z-score with train stats",但 Exp4 实际用 **RobustScaler**(`feff_feature_scaler.pkl` 1.6 KB)。功能等价,术语不同,**不要重新算 z-score**

在 `spectrum_tokenizer.py` 文件头加一段 docstring 说明:

```python
"""
Spectrum tokenizer for Exp6, derived from Exp4 spectrum_encoder.py.

Notes on Exp6 proposal vs Exp4 reality (do NOT "fix" these):
- feff dim is 74 (proposal §3.1 says 73, Exp4 reality is 74)
- feff scaling is RobustScaler (proposal says z-score, Exp4 reality is RobustScaler)
Both are inherited unchanged from Exp4. Source: EXP4_FILE_GUIDE.md §3.1.
"""
```

### 3.5 Step 1.4 — `matcher.py`

1. ssh `cp /home/tcat/diffcsp_exp6/_detr_reference/models/matcher.py /home/tcat/diffcsp_exp6/shared/matcher.py`
2. 改动列表:

| DETR 原版 | Exp6 替换 | 行数 |
|---|---|---|
| `from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou` | 删除 | 1 |
| `__init__(cost_class, cost_bbox, cost_giou)` | `__init__(cost_class, cost_pos)`,`cost_giou` 删 | 3 |
| `assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0` | `assert cost_class != 0 or cost_pos != 0` | 1 |
| `outputs["pred_boxes"]` | `outputs["pred_pos"]` | 2 |
| `cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)` | `cost_pos = min_image_l2(out_pos, tgt_pos, lengths)` (复用 eval_metrics.py 同名函数) | 1 |
| `cost_giou = -generalized_box_iou(...)` | 删除 | 2-3 |
| `C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou` | `C = self.cost_pos * cost_pos + self.cost_class * cost_class` | 1 |
| `build_matcher(args)` | 改为接受 `cost_class, cost_pos, lengths` 参数 | 5 |

**绝对保留**:
- `@torch.no_grad()` 装饰器
- `linear_sum_assignment` per-batch logic
- 返回格式 `[(torch.as_tensor(i), torch.as_tensor(j)), ...]`

**新参数**: `lengths` 是 `torch.tensor([6.0, 6.0, 6.0])`,L=6 box 边长,proposal §4.1(d)。从 detr_xas.py 传进来。

### 3.6 Step 1.5 — `criterion.py`

1. ssh `cp /home/tcat/diffcsp_exp6/_detr_reference/models/detr.py /home/tcat/diffcsp_exp6/shared/criterion.py`
2. **只保留** `SetCriterion` 类,其他全删(DETR class、PostProcess、MLP、build、PostProcess 等)。MLP 你单独在 detr_xas.py 里写自己的,不复用 DETR 的(因为 DETR MLP 是 num_layers 参数化,你做 type/pos head 也可以独立简单写)
3. 改动列表:

| DETR 原版 | Exp6 替换 |
|---|---|
| `from util.box_ops import box_ops` | 删除(no GIoU) |
| `from util.misc import ...` | 只保留 `is_dist_avail_and_initialized, get_world_size`,其余删 |
| `loss_boxes(self, outputs, targets, indices, num_boxes)` | 重写为 `loss_pos(...)`:`outputs['pred_pos']` 不是 `pred_boxes`;loss = `min_image_l2_squared(src_pos, tgt_pos, lengths).sum() / num_boxes`(L2 平方而非 L1);**删除 GIoU 计算和 'loss_giou' key** |
| `loss_masks(...)` | **整段删除**(no segmentation) |
| `self.losses = ['labels', 'boxes', 'cardinality', 'masks']` | `self.losses = ['labels', 'pos', 'cardinality']` |
| `loss_map = {'labels': ..., 'cardinality': ..., 'boxes': ..., 'masks': ...}` | `loss_map = {'labels': ..., 'cardinality': ..., 'pos': self.loss_pos}` |

**绝对保留**:
- `loss_labels` 完全不动(CE + empty_weight,empty_weight[-1]=eos_coef=0.1)
- `loss_cardinality` 不动(`@torch.no_grad()` diagnostic)
- `_get_src_permutation_idx`、`_get_tgt_permutation_idx`
- `forward(self, outputs, targets)` 主流程含 aux_outputs 循环

**关于 cardinality vs no_object_ratio**: 二者不冲突——`loss_cardinality` 是 DETR 自带 diagnostic(返回 `card_pred - target_count` 的 L1),`no_object_ratio` 是 proposal §附录B.5 锁定的 Exp6 自定义指标。两个**都保留**,各自打 log,不试图合并。

### 3.7 Step 1.6 — `detr_xas.py`(主模型,新写)

这是 SA1 主写工作量。proposal §4 给了架构图,§4.1 (a)-(e) 给了关键决策,proposal §3.2 给了 SpectrumTokenizer 接口,你要拼起来。

**类结构**:

```python
# Pseudocode reference, SA1 实现细节自决但接口签名锁定

class DETRXas(nn.Module):
    def __init__(self,
                 d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 num_queries=20,
                 n_neighbor_types: int,    # 从 vocab json 读
                 n_center_types: int,      # 从 vocab json 读
                 lengths=(6.0, 6.0, 6.0),
                 aux_loss=True):
        # 必含组件:
        # - self.tokenizer: SpectrumTokenizer (从 §3.4 import)
        # - self.center_token_embed: nn.Embedding(n_center_types, d_model)
        #     (proposal §3.2 "separate learnable token" 路径)
        # - self.token_pos_embed: nn.Embedding(2, d_model)
        #     (encoder 端 2 个 token: spectrum + center;Phase 1 单谱)
        # - self.transformer: from .transformer import Transformer
        #     (return_intermediate_dec=True)
        # - self.query_embed: nn.Embedding(num_queries, d_model)
        # - self.class_head: 3-layer MLP(d_model -> d_model -> d_model -> n_neighbor_types+1)
        #     (proposal §4 "MLP type 各 ~3 层" 锁定)
        # - self.pos_head: 3-layer MLP(d_model -> d_model -> d_model -> 3),末层 tanh*0.5
        #     (proposal §4.1(d) 锁定)
        # - self.aux_loss flag
    
    def forward(self, batch):
        # batch 来自 Exp4 datamodule 的 collate_fn,期望含:
        # - xmu: (B, 150)
        # - chi1: (B, 200)
        # - feff: (B, 74)            ← 注意 74 不是 73
        # - center_Z: (B,)            ← 中心元素原子序数 → 你 map 到 dense center_idx
        # 
        # 输出 dict (proposal §4 contract):
        # - pred_logits: (B, 20, n_neighbor_types + 1)
        # - pred_pos: (B, 20, 3)        ← in [-0.5, 0.5] frac
        # - aux_outputs: list of 5 dicts (decoder layers 0..4)
        ...
```

**forward 主流程**:

1. `spectrum_token = self.tokenizer(xmu, chi1, feff).unsqueeze(1)` → `(B, 1, 256)`
2. `center_token = self.center_token_embed(center_idx).unsqueeze(1)` → `(B, 1, 256)`
3. `src = torch.cat([spectrum_token, center_token], dim=1)` → `(B, 2, 256)`
4. **关键 permute**: DETR transformer 期望 seq-first `(N_seq, B, d_model)`。`src = src.permute(1, 0, 2)` → `(2, B, 256)`
5. `pos = self.token_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)` → `(2, B, 256)`
6. `query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)` → `(20, B, 256)`
7. `mask = None`(单谱,无 padding;Phase 2 多谱时再造 padding mask)
8. `hs = self.transformer(src, mask, query, pos)` → 注意 transformer.forward 的真实 signature(MA1 读出来 DETR 原版是 `forward(src, mask, query_embed, pos_embed)`,你照样调用)
9. transformer 在 image 路径会有 "flatten NxCxHxW to HWxNxC" 这一段,因为我们已经是 seq-first,**这段会出错**。**必须改 transformer.py forward**——见下方 §4 delta map 第 3 行
10. `hs[-1]` → `(B, 20, 256)`,过 class_head 得 logits,过 pos_head 得 pos
11. `aux_outputs = [{'pred_logits': self.class_head(hs[i]), 'pred_pos': torch.tanh(self.pos_head_inner(hs[i])) * 0.5} for i in range(num_layers - 1)]`
12. return dict

**注意**: DETR 原 transformer.py forward 第一行是 `bs, c, h, w = src.shape`,这是为 image 写的,你的 src 是 `(2, B, 256)` 不是 4D。**有两条路**:
- (A) 不改 transformer.py,而是在 detr_xas.py 里把 src 强行做成 4D `(B, 256, 1, 2)`,让 transformer 内部 flatten 后变 `(2, B, 256)` ——hacky 但 transformer.py 零改动
- (B) 改 transformer.py forward,第一行起前三行(flatten + permute)整段删掉,直接接受 seq-first input

**SA1 决策**: 走 (B)。理由: (A) 让 detr_xas.py 多一段歪门 reshape,可读性差,以后 Phase 2 多谱时也得维持 shape 兼容性。(B) 改 transformer.py 是 ~3 行删除 + forward signature 不变,Phase 2 也不用再改。**这条决策记进 OUTPUT.md "implementation choices" 一节**。

### 3.8 Step 1.7 — `step1.2_smoke_test.py`

**目标**: 5 样本跑通 forward + matcher + loss + backward,不依赖训练 loop。

骨架:

1. 加载 `xas_local_datamodule_v2`,build train DataLoader,取第一 batch(`bs=5`)
2. 实例化 `DETRXas`,加载 vocab 进 `n_neighbor_types`/`n_center_types`
3. `out = model(batch)` → 检查 shape
4. 构造 targets(从 batch 里取 gt_types + gt_pos + 转 dense vocab idx)
5. 构造 matcher 和 criterion,`loss_dict = criterion(out, targets)`,`total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)`
6. `total.backward()`
7. 打印参数总量、每个组件 grad_required、loss 各项数值

**4 项硬检查**(proposal §6 锁定):

```
[CHECK 1] pred_logits.shape == (5, 20, n_neighbor_types + 1):  PASS / FAIL
[CHECK 2] pred_pos in [-1, 1] (tanh*0.5 已约束),无 NaN:        PASS / FAIL
[CHECK 3] matcher 5 样本输出合理(20 query 中 ~17 配对,~3 no_object 在合理范围): PASS / FAIL
[CHECK 4] 第 1 个 batch total_loss 在 [10, 100]:                PASS / FAIL
```

任何一项 FAIL 都 Phase 1 不算完成。stdout 全 dump 到 OUTPUT.md。

**额外伸展检查**(不算硬要求,但若 PASS 写进 OUTPUT.md):
- 参数总量 < 50M(proposal §附录B.4 期望)
- 6 层 decoder aux_outputs 各自 loss 接近(数量级一致),不要某层 NaN

---

## 4. DETR → Exp6 Delta Map(快速参考)

| 改动文件 | DETR 原版位置 | Exp6 替换 | proposal 出处 |
|---|---|---|---|
| `transformer.py` | forward 第 1-4 行 `bs, c, h, w = src.shape; src.flatten(2).permute(2, 0, 1)` | 删除前 3 行,直接接受 seq-first src | 你 §3.7 决策 (B) |
| `transformer.py` | `pos_embed.flatten(2).permute(2, 0, 1)` | 删除,pos 已是 seq-first | 同 |
| `matcher.py` | `cost_giou` 整段 | 删除 | proposal §2.3,§4.1(e) |
| `matcher.py` | `cost_bbox = torch.cdist(p=1)` | `cost_pos = min_image_l2(...)` | proposal §4.1(e) |
| `matcher.py` | `outputs["pred_boxes"]` | `outputs["pred_pos"]` | proposal §4 |
| `criterion.py` | `loss_boxes` 含 GIoU | `loss_pos` 仅 L2 平方 / num_boxes | proposal §2.3 |
| `criterion.py` | `loss_masks` | **整段删除** | proposal §2.3(无 segmentation) |
| `criterion.py` | `losses = ['labels', 'boxes', 'cardinality', 'masks']` | `losses = ['labels', 'pos', 'cardinality']` | 同 |
| `detr_xas.py`(替 DETR.py) | `class_embed = nn.Linear(d_model, num_classes+1)` | 3-layer MLP | proposal §4 "各 ~3 层" |
| `detr_xas.py` | `bbox_embed = MLP(d_model, d_model, 4, 3); .sigmoid()` | 3-layer MLP `(d_model, d_model, 3, 3)`,末层 `tanh*0.5` | proposal §4.1(d) |
| `detr_xas.py` | `input_proj = nn.Conv2d(...)`(image backbone 出来过 1x1 conv) | 删除,SpectrumTokenizer 直接出 256d | proposal §3.2 |
| `detr_xas.py` | `backbone(samples)` → 解 NestedTensor | 删除,无 image,无 mask | proposal §4.1(a) |
| `position_encoding.py` | sin/cos 2D | **不 cp**,用 `nn.Embedding(num_tokens, d_model)` | proposal §2.3 |

---

## 5. Tensor Shape Contracts(SA1 调试时打印的对照表)

| 张量 | 期望 shape | 出处 | 备注 |
|---|---|---|---|
| `xmu` | `(B, 150)` | Exp4 dataset | xanes 窗口 |
| `chi1` | `(B, 200)` | Exp4 dataset | exafs |
| `feff` | `(B, 74)` | Exp4 dataset | **74 不是 73**,见 §3.4 |
| `center_Z` | `(B,)` | Exp4 dataset | atomic Z,你 map 到 `center_idx` `(B,)` 0..N_CENTER_TYPES-1 |
| `spectrum_token` | `(B, 1, 256)` | tokenizer 输出后 unsqueeze | |
| `center_token` | `(B, 1, 256)` | center_token_embed |
| `src`(permute 前) | `(B, 2, 256)` | cat |
| `src`(permute 后) | `(2, B, 256)` | seq-first 入 transformer |
| `query_embed.weight` | `(20, 256)` | nn.Embedding |
| `query`(repeat 后) | `(20, B, 256)` | unsqueeze(1).repeat(1, B, 1) |
| `pos`(token_pos_embed 后) | `(2, B, 256)` | 同 query 模式 |
| `hs`(transformer 输出) | `(num_layers=6, B, 20, 256)` | return_intermediate_dec=True |
| `pred_logits`(取 hs[-1]) | `(B, 20, n_neighbor_types+1)` | last decoder layer |
| `pred_pos`(取 hs[-1]) | `(B, 20, 3)` | tanh*0.5 后 in [-0.5, 0.5] |
| `aux_outputs` | `list[5]` of `{'pred_logits': ..., 'pred_pos': ...}` | layers 0..4 |
| matcher 输入 `outputs` | `dict` with `pred_logits` + `pred_pos` | criterion 内自动剥 aux_outputs |
| matcher 输入 `targets` | `list[B]` of `{'labels': (n,), 'pos': (n, 3)}` | 你在 datamodule collate_fn 里组装,或在 smoke_test 里手工组装 |
| matcher 输出 `indices` | `list[B]` of `(pred_idx, gt_idx)` 各为 LongTensor | per-batch |

---

## 6. Forbidden List(违反 = Phase 1 重做)

直接禁:

1. **任何辅助物理 loss**(density / shell / distance / center attraction)— proposal §附录B.6,ERRATA_2 §1 已确认 `_density_loss` 是塌缩剂。**Exp6 thesis 就是不靠这些**
2. **TypeClassifier head / 任何接 latent 的额外分类 head**— Exp3 双重证伪(`.detach()` 教训 + 虚假指标)+ Exp5 三重证伪。proposal §附录B.7
3. **修改 Exp4/Exp5 服务器代码**— 只 cp 出来再改自己的副本,绝不在 `/home/tcat/diffcsp_exp4/code/` 下做修改
4. **训练期读 holdout** — `holdout_samples_v2.csv`、`spectra_holdout.pkl`、`predictions_holdout.pt`(后者只在 step1.1 不算"训练期",但你也用不到 holdout pred,只用 val pred)
5. **`incompat_pool.csv`** — 永久封存,任何 SA 不读
6. **复用 Exp5 v1 SA1 的 95-class sparse vocab**— proposal §3.2 明文,Exp6 用自己的 dense neighbor_vocab
7. **自由发挥 §7.1 公式**— eval_metrics.py 严格逐字符复制 proposal §7.1。如果你认为某个公式有"明显 bug",**先 push 给 MA1**,不要自改

灰色地带需要 push back to MA1 不要自决:

8. proposal 与 Exp4 现实差异(feff 73 vs 74、z-score vs RobustScaler):本 handoff §3.4 已说明保留 Exp4 现实,**不是 SA1 自决**
9. 行号 verify 失败(proposal §6.1 引的 main.py L48 等行号在你 clone 的实际文件里偏移):记 OUTPUT.md,**不修 proposal**,跑通即可
10. predictions_val.pt schema 出乎预期(例如 pred 类型存的是 logits 而非 argmax):走 §3.2 Step 1.1 灰色处理,在 OUTPUT.md 里说明你做的 transform

---

## 7. Open Questions(开 Phase 1 之前 push 回 MA1)

如果碰到下面任何一条,**先停手,在你的 chat 里问用户(用户会转给 MA1)**,不要自己拍板:

| 触发 | 你怎么问 |
|---|---|
| Step 0.4 dump 出来 train_samples_v2.csv 没有 `center_element` 列,只有比如 `mp_id` + `composition` | "MA1: train CSV schema 是 X,vocab build 路径需要 reroute,提议走 Y(从 mp_id pull POSCAR 解析中心元素),要 ack 吗?" |
| Step 0.5 dump 出来 predictions_val.pt 是 list 不是 dict,或者 pred 类型用的是 dense 0..N idx 而非 Z | "MA1: predictions_val.pt schema 是 X,Set-Level baseline 计算 path 需要调整为 Y,要 ack 吗?" |
| Step 0.6 读 transformer.py 发现 d_model 默认不是 512 而是其他,或 forward signature 不是 `(src, mask, query_embed, pos_embed)` | "MA1: transformer.py 实际签名/默认值与你 takeaway 有差,见 X。这影响 detr_xas.py 拼装方式" |
| Step 1.6 写 detr_xas.py 时,SpectrumTokenizer 输出维度不是 256 而是 fusion 之后实际比如 448 / 272 | "MA1: tokenizer 末层去除后实际输出维度是 X,与 d_model=256 不匹配,需要加一个 projection nn.Linear(X, 256) 还是改 d_model=X?" |
| smoke_test loss > 100 或 < 10,或 NaN | "MA1: smoke loss 异常,数值是 X,完整 stdout 见附件,建议 debug 顺序是?" |
| 任何"我可以稍微优化一下"的诱惑 | 不要做。push: "MA1: 我观察到 X 处可优化为 Y,但 proposal 锁了 Z,Phase 1 是否走 Z 不动?" |

---

## 8. Phase 1 Exit Criteria(SA1 完成的硬性定义)

`EXP6_PHASE1_OUTPUT.md` 必须包含:

1. **文件清单**: `experiment6/` 下所有产出文件的相对路径 + 行数 + sha256
2. **DETR 行号 verification**: proposal §6.1 表 7 行 + detr.py L77 各自实际命中行号,一致 ✓ / 偏移记录
3. **vocab 实测值**: `(N_CENTER_TYPES, N_NEIGHBOR_TYPES)` + sample 的 Z→idx 映射前 5 条
4. **Exp4 Set-Level baseline 重算结果**: scalar 数字 `exp4_setlevel_typeacc_val = ?` 一句报告,proposal §10.1 backfill 用
5. **smoke_test 完整 stdout**: 4 项硬检查 PASS/FAIL + 参数总量 + 各 loss 数值
6. **implementation choices**: 你在 §3.7 (A vs B) 等灰色处选了哪条,理由一行
7. **schema dump 结果**: Step 0.4 + 0.5 dump 内容压缩贴出(关键列名 + dtypes)
8. **anything else worth flagging**: 比如发现 Exp4 dataset 某行为与 proposal 描述不符,记下供 MA1 review

OUTPUT.md 完成后,**push 给用户,用户转 MA1**。MA1 review 通过后,你 Phase 1 closed,SA2 接手 Phase 2 训练。

---

## 9. 时间估算

| 阶段 | 估时 | 备注 |
|---|---|---|
| Phase 0 setup(0.1-0.6) | 0.5 天 | 多数等用户 ssh 转发 |
| Step 1.0 + 1.1 vocab/baseline | 0.5 天 | 写 + 验证 |
| Step 1.2 eval_metrics.py | 0.5 天 | 5 公式逐字实现 + 单测 |
| Step 1.3 spectrum_tokenizer.py | 0.25 天 | cp + 改 1-2 行 |
| Step 1.4 + 1.5 matcher/criterion | 0.5 天 | DETR cp + 改造 |
| Step 1.6 detr_xas.py | 0.75 天 | 主要新写工作量 |
| Step 1.7 smoke_test + debug | 0.5 天 | 第一次跑可能撞 shape bug |
| OUTPUT.md 写 | 0.25 天 | |
| **合计** | **~3.75 天** | proposal §9 给 Phase 1 是 2 天,SA1 估实际更紧,可能要 3-4 天 |

如果第 5 天 smoke 还没过,**stop,push MA1**,不要自己 debug 超 1 天。

---

## 10. 从 MA1 来的话

读完这份手册的全部内容再开始动手。任何一条不清楚的地方,在动手前就问出来,不要"先做做看"。

proposal v3 比 v2 干净很多(§7.1 五公式已锁、§6.1 超参 trace 已引、§4.1(c) vocab 已严格分离),你在 v3 这个版本下应该不会再撞到"指标公式自由发挥"或"超参不知从哪拍的"这类前几代 SA1 撞过的坑。如果撞到了,几乎肯定意味着 v3 还有漏洞——**push 回来**,我打回 MA6 再迭代,不要自己补窟窿。

祝 Phase 1 顺利。

—— Exp6-MA1
