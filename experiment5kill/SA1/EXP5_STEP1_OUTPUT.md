# EXP5_STEP1_OUTPUT.md — Exp5 SA1 交棒文档(final-final)

> **撰写者**: Exp5-SA1 (架构改造 Sub-Agent)
> **日期**: 2026-04-28
> **接收人**: Exp5 Main Agent
> **状态**: ✅ **全部 acceptance gates 通过,实测证据完整**;SA2 可启动训练

---

## §0 完成度速览

| 闸门(handoff §5) | 状态 | 证据 |
|---|---|---|
| 1. forward_test.py 全 phase 通过,日志干净 | ✅ **5 PASS + 1 skipped-by-design** | `/home/tcat/diffcsp_exp5/logs/step1_forward_test.log`(v3) |
| 2. step4_1_smoke_test.py 5+ batch 无 crash,4 个 loss 量级合理 | ✅ **ALL SMOKE PASSES**(4 mode) | `/home/tcat/diffcsp_exp5/logs/step1_smoke.log`(v1) |
| 3. center_Z=true vs 0 时 head_logits ≥ 5/20 ranks 不同 | ✅ phase 6.6.c 自动断言通过 | log 内 phase 6.6.c |
| 4. head_logits.shape (B,20,100) + 三 mode 各自 forward 不 NaN | ✅ phase 6.6.a/d 自动断言通过 | log 内 phase 6.6 |
| 5. 写完 OUTPUT.md(本文件) | ✅ 本文件 final 版 | — |

**Phase 6.5 (GPU bf16) 状态**:**SKIPPED-by-design**,详见 §5.7。skip ≠ pass,显式区分。

---

## §1 改了哪些文件

7 个文件 + 6 个 `.bak_exp4` baseline 锚点。

| 文件 | 行数 (改前→改后) | 改动摘要 |
|---|---|---|
| `step2/spectrum_encoder.py` | 95 → 127 | 加 `nn.Embedding(95, 16)` + forward 接 `center_Z` + 内部 cat 输出 (B, 272) |
| `step3/xas_local_dataset_v2.py` | 357 → 374 | 加 `_symbol_to_Z` lookup dict + 两条 path 字典加 `center_element_Z` |
| `step3/xas_local_datamodule_v2.py` | 247 → 257 | `_dict_to_pyg_data` 加 `center_element_Z` (1,) long tensor |
| `step3/diffusion_w_type_xas.py` | 415 → 589 | `TypeClassifierHead` 类 + head 实例化 + 三 mode 字段 + forward 算 head CE + 三 mode loss 聚合 + log + `head_predict_types` 方法供 SA3。**Patch 1 (2026-04-28)**:onehot `.float()` → `.to(c0.dtype)` |
| `step3/conf_xas/model/diffusion_xas.yaml` | 50 → 79 | 加 8 个新字段;`latent_dim` 256→272;`decoder.latent_dim` 512→528 |
| `step3/forward_test.py` | 365 → 546 | 加 phase 6.6;**phase 6.5 改为 SKIPPED-by-design**(原代码保留为 `_phase_65_legacy`);phase 6.4 loss 闸门 [2,6]→[4,12] |
| `step4/step4_1_smoke_test.py` | NEW (193) | 新写;Exp4 服务器 step4_1_smoke_test.py 不存在 |

---

## §2 Sanity check 实测结果(从服务器日志)

### §2.1 max(Z) → embedding 表大小

```
[XasLocalDatasetV2] center_element_Z lookup built: 88 elements, Z ∈ [2, 94]   ← train split
[XasLocalDatasetV2] center_element_Z lookup built: 84 elements, Z ∈ [3, 94]   ← val split
```

**实测 max(Z) = 94**。`nn.Embedding(95, 16)` 足够。Handoff §3.1 的 89 占位被实测推翻。

### §2.2 数据 4-source 对齐

dataset_v2 init 阶段 first-5-samples defensive check 全部通过(无 RuntimeError raise)。Cache 实测覆盖率:
- train: **60,501 / 60,507 valid** (99.99%)
- val:   **7,621 / 7,624 valid** (99.96%)

### §2.3 atom_types 已按距离 sort

Exp4 真版 `__getitem__` 已显式 `np.argsort(dists)[:N_NEIGHBORS]`,SA1 不重复加排序。

---

## §3 关键实施决策(yaml 默认值)

| 字段 | 值 | 来源 |
|---|---|---|
| `n_center_elements` | **95** | §2.1 实测 max(Z)=94 |
| `center_emb_dim` | 16 | handoff §2.1 |
| `type_head_hidden_dim` | 512 | handoff §2.2 |
| `n_atoms` | 20 | `N_NEIGHBORS` 不变量 |
| `n_elements` | **100** | 跟 diffusion 内部 `MAX_ATOMIC_NUM=100` 对齐(MA5 也建议) |
| `type_loss_mode` | both | handoff §2.4 default |
| `diffusion_type_weight` | 1.0 | handoff §2.4 |
| `head_type_weight` | 0.5 | handoff §2.4 |
| `latent_dim` (top) | **272** | 256 + 16 |
| `decoder.latent_dim` | **528** | time_dim 256 + spectrum 272(Exp4: 512) |

### 偏离 handoff 的实施细节

**a. center embedding 嵌入 SpectrumEncoder 内部** —— 让 SpectrumEncoder 成为单一 `(xmu, chi1, feff, center_Z) → (B, 272)` 封装单元。fusion 层位置不变 → Exp4 ckpt 该层可完整加载。

**b. 不用 `_exp5` 后缀文件名** —— `diffusion_w_type_xas.py` L24 hardcode `from spectrum_encoder import SpectrumEncoder`,改名要改一堆 import,得不偿失。每个改动文件旁留 `*.bak_exp4` 锚点。

**c. `cost_type` 字段沦为 nominal** —— 新增 `diffusion_type_weight: 1.0` 接管。yaml 保留 `cost_type` 向后兼容,但**实际不进 total loss 计算**。SA2 改 `diffusion_type_weight` 才有效。

**d. `center_element_Z` 不进 cache** —— 通过 `self._symbol_to_Z` dict O(1) lookup 实现,Exp4 已生成的 cache `.pt` 文件 SA2 可继续用,**无需重建**(实测验证:cache 99.99% 命中)。

---

## §4 OPEN QUESTIONS(请 Exp5 MA / SA2 / SA3 决议)

### OQ-1: atom_types ∈ [1, 109] vs `MAX_ATOMIC_NUM=100`

`forward_test.py` L102 断言 `atom_types ≤ 109`,但 `diffusion_w_type_xas.py` L44 `MAX_ATOMIC_NUM=100`。Exp4 训练成功说明 max(Z) 实际 ≤ 100。SA1 `n_elements=100` 与此一致。

**SA2/SA3 决议建议**:在 train 数据上跑一次 `max(atom_types)` 实测确认。如果 ≥ 101,需要扩 `n_elements` 和 `MAX_ATOMIC_NUM`,并重训。

### OQ-2: handoff §5 item 2 对 `loss_diffusion_type` 量级的描述疑似笔误

handoff 写 "loss_diffusion_type ~4-5(接近 ln(89))"。但实测:

```
loss_diffusion_type:  1.3352   ← random init,正常 MSE 量级
loss_type_ce_head:    4.6028   ← random init,~ ln(100) = 4.605,正常 CE 量级
```

handoff 把 MSE 和 CE 的 init 量级混淆了。**实际 4-5 区间是 head CE 而非 diffusion type MSE**。OUTPUT 实测数据是基准。

### OQ-3: `cost_type` yaml 字段是否要彻底删

代码已经先读 `diffusion_type_weight`,`cost_type` 留着但不起作用。可以选 (a) yaml 删掉干净;(b) 保留 + 注释强调 nominal。SA1 选了 (b),Exp5 MA / SA2 觉得 (a) 更好就删一行,代码不用动。

---

## §5 Notes for SA2

### §5.1 ⚠️ Phased training(handoff §8.1 carry-over)

新加可学参数全部随机初始化:

```
spectrum_encoder.center_emb.weight              (95, 16)         ~1.5K params
type_head.fc.0.weight, fc.0.bias                (512, 272), (512)
type_head.fc.2.weight, fc.2.bias                (2000, 512), (2000)
decoder.<first Linear> (atom_latent_emb)        shape mismatch   见 §5.2
```

**phased training 强烈建议**:

| Phase | epoch 范围 | 解冻 | lr |
|---|---|---|---|
| 1(head warmup)| 0-5  | `type_head.*` + `spectrum_encoder.center_emb.*` | 1e-3 或 1e-4 |
| 2(joint)      | 6-end | 全部 | head 1e-4,backbone 1e-5(differential)|

诊断:前 5 epoch 关注 `val_coord_loss`,如从 0.7300 起步快速升到 >0.85 不回落 → phased training 没做好,回滚。

### §5.2 ⚠️ Exp4 ckpt warm-start: decoder 第一层 strict=False 跳过

`decoder.latent_dim` 从 512 → 528(因 spectrum 输出 256→272)。Exp4 ckpt 里 `decoder.atom_latent_emb.weight` shape (256, 512) 与 Exp5 (256, 528) **shape mismatch**。

```python
# SA2 训练入口推荐:
ckpt = torch.load("/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt",
                  map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)

# 期望 missing:
#   spectrum_encoder.center_emb.weight                                ← Exp5 新模块
#   type_head.fc.0.weight / .bias                                     ← Exp5 新模块
#   type_head.fc.2.weight / .bias                                     ← Exp5 新模块
#   decoder.atom_latent_emb.weight / .bias  (shape mismatch → SKIP)   ← Exp5 SA1 特有
```

**SA2 必须打印 missing / unexpected 列表到训练 log 开头**作存证。

### §5.3 三 mode flag 怎么用

- `type_loss_mode: both`(default):SA2 第一次训练用这个
- `type_loss_mode: head_only`:如果 baseline_v2 训完 TypeAcc < 0.30,SA2-续 改 yaml 这一行重训
- `type_loss_mode: diffusion_only`:理论上等价 Exp4 baseline,只用于回归测试

切换不需改代码,只改 yaml。

### §5.4 服务器磁盘 + swap 警告

```
/ 已用 92.0% (1.72TB 剩 ~140GB)
Swap 80% 占用
```

SA2 训练前**强烈建议清理**:旧 wandb runs / 旧 smoke ckpt / /tmp/diffcsp_cache/。预算:Exp5 全程预计 30-50 GB。

### §5.5 Exp5 工作目录(GUIDE §5)

```
/home/tcat/diffcsp_exp5/
├── code/{step2,step3,step4}/...    ← SA1 代码
├── data/                            ← 软链接到 Exp4 data
├── checkpoints/                     ← SA2 产出
├── logs/                            ← 全套 log
└── sa0/                             ← SA0 multi-sample(独立)
```

数据软链接已在服务器建好(实测 cache 99.99% 命中即证)。

### §5.6 ⚠️ PYTHONPATH 优先级(SA1 漏提的关键 carry-over)

服务器同时跑 Exp4(可能有遗留)、SA0、Exp5 SA2/SA3,**全部用同一个 conda env、同一个 `diffcsp` package**。Exp4 和 Exp5 的 `diffusion_w_type_xas.py` / `spectrum_encoder.py` 是**重名不重内容的两套文件**。SA2 训练入口必须**让 Exp5 的几个改动文件优先放到 PYTHONPATH 前面**,否则 Python import 缓存会拉到 Exp4 旧版,你会以为在训 Exp5 baseline_v2,实际跑的是 Exp4 网络(没 head、没 center_emb)。

**正确写法(任选其一)**:

a) shell 设置:

```bash
export PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:$PYTHONPATH
/home/tcat/conda_envs/mlff/bin/python train.py
```

b) 训练入口脚本开头硬插:

```python
import sys
sys.path.insert(0, "/home/tcat/diffcsp_exp5/code/step3")
sys.path.insert(0, "/home/tcat/diffcsp_exp5/code/step2")
# 然后再 import 任何 diffusion_w_type_xas / spectrum_encoder
```

**自检**:训练开始时打印 `import diffusion_w_type_xas; print(diffusion_w_type_xas.__file__)` 确认路径在 `/home/tcat/diffcsp_exp5/...` 而非 `/home/tcat/diffcsp_exp4/...`。如果走错,**整个训练都白费**。

`forward_test.py` 现在走 `cd step3/` + 同目录 import 所以没踩到。SA2 的 cwd 一变就会踩。

**SA1 实测验证写法**(2026-04-28 forward_test v3 跑通用的):

```bash
cd /home/tcat/diffcsp_exp5/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py
```

⚠️ **末尾的 `/home/tcat/diffcsp_exp4/code` 不能省** —— 这是 Exp4 仓里 `diffcsp/` 子包(decoder CSPNet、common utils 等)的位置,Exp5 没复制(也不该复制,SA1 不能动 Exp4 backbone)。**Exp5 自己的 step2/step3 必须在前面**(才能 shadow Exp4 同名文件),**Exp4 code 必须在末尾**(才能找 `diffcsp.pl_modules.cspnet` 等)。三者缺一会 fail。

### §5.7 phase 6.5 (GPU bf16) SKIPPED-by-design 决策记录

**背景**:Exp4 SA4-续 2 final report 声称 phase 6.5 PASS。Exp5 SA1 在 PT 2.4.1+cu124 复现时,patch 前 fail 在 `cspnet.py:268 self.node_embedding`;patch 1(`F.one_hot(...).to(c0.dtype)`)修了第一处后,fail 移到 `cspnet.py:274 self.atom_latent_emb` —— 第二处 dtype mismatch。

**3 处 hardcoded fp32 site**(从 stack trace 定位):

| # | 文件:行 | 内容 | 状态 |
|---|---|---|---|
| 1 | `diffusion_w_type_xas.py` ~L288 (Exp4) / L292 (Exp5 SA1 patched) | `F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()` | ✅ Patch 1 修了 (`.float()` → `.to(c0.dtype)`)。fp32 下 bit-exact 等价,零风险,为未来 bf16 留起点 |
| 2 | `diffusion_w_type_xas.py` ~L100 `SinusoidalTimeEmbeddings.forward` | `torch.arange(half_dim, device=device)` 默认 fp32 → emb 永远是 fp32 | ❌ 未修(skip 决策) |
| 3 | `diffcsp/pl_modules/cspnet.py` L272-274 | `t_per_atom = t.repeat_interleave(...)` + `node_features = self.atom_latent_emb(...)` 无 dtype-aware cast | ❌ 未修(Exp4 代码,**out of scope for SA1**) |

**Skip 决策依据**(用户拍板 B):

1. Exp4/Exp5 训练全程 fp32(MA4 决策 D1,用户 2026-04-28 确认)
2. phase 6.5 测试的 bf16 path **不在生产路径上**,fail 不影响 SA2 训练 / SA3 评估
3. 修第 3 点要动 Exp4 cspnet.py —— **out of scope for SA1**,且会引入第三方 backbone 改动风险
4. 强行修 1+2 也只是修一段 dead code,可能藏第四处 dtype bug

**Skip ≠ Pass**。forward_test 闸门 status:`5 PASS + 1 skipped-by-design`,显式区分。

**保留代码**:原 phase 6.5 代码以 `_phase_65_legacy()` 函数保留,**不被 main() 调用**。未来若启用 bf16/AMP 训练,enabler 可 diff 上述 3 处 site,从此函数开始恢复测试。

**给 Exp4 MA5 的反馈材料**(用户已收到,会更新 final report 和 GUIDE):

> Exp4 forward_test.py phase 6.5 (GPU bf16) 在 PT 2.4.1+cu124 下因 3 处 hardcoded fp32 site 与 model bf16 weights 产生 mat1/mat2 mismatch。Exp4 final report 声称 PASS 应是误记或当时跑了别的 PyTorch 版本。**只要训练用 fp32,这个 fail 不影响任何实际生产路径**,SA2/SA3 不需要回头修 Exp4。

---

## §6 实测日志摘要

### §6.1 forward_test.py 关键数据(v3 — patch 1 + phase 6.5 skip;实测 2026-04-28 10:56:38)

```
Phase 6.1 PASS — 60,507 train samples / 100 random samples / 0.1 ms each
                 frac range [-0.499898, 0.499898] (handoff §2.C in spec)
                 cache LOADED: 60,501/60,507 valid (99.99%)
                 center_element_Z lookup: 88 elements, Z ∈ [2, 94]

Phase 6.2 PASS — PyG Batch with 10 expected fields:
                 frac_coords (80,3), atom_types (80,), xmu_xanes (4,150),
                 chi1 (4,200), feff_features (4,74), lengths/angles (4,3),
                 eval_cutoff (4,), batch (80,), center_element_Z (4,)

Phase 6.3 PASS — SpectrumEncoder out (4, 272), mean=+0.0034, std=0.2336
                 center_Z=true vs zero: 4/4 samples changed ✓

Phase 6.4 PASS — CSPDiffusion 4,511,204 params
                 loss_total          : 4.8772
                 loss_coord          : 1.1976
                 loss_diffusion_type : 1.3368   (MSE, ~ Exp4 量级)
                 loss_type_ce_head   : 4.6031   (~ ln(100) = 4.605, 完美)
                 loss_type_total     : 3.6383   (= 1.0×1.3368 + 0.5×4.6031 ✓)
                 loss_density        : 0.0826
                 loss_lattice        :12.0533   (× cost_lattice=0)
                 grad_norm           : 9.6678

Phase 6.5 SKIPPED-BY-DESIGN — see §5.7

Phase 6.6 PASS — Exp5 SA1 architecture additions
   6.6.a head_logits.shape:        (4, 20, 100) ✓
   6.6.b loss_type_ce_head:        4.6031 (finite scalar) ✓
   6.6.c center_Z=true vs zero argmax differ:
         per sample = [19, 19, 16, 17] (out of 20); min = 16 (gate ≥ 5) ✓
         strong center conditioning observed
   6.6.d three type_loss_mode flags all forward without NaN:
         mode=diffusion_only:  total=2.4232  diff_type=1.3297  head_ce=4.6031  type_total=1.3297
         mode=head_only:       total=3.5406  diff_type=1.3549  head_ce=4.6031  type_total=2.3016
         mode=both:            total=4.7414  diff_type=1.3770  head_ce=4.6031  type_total=3.6786
         all three monotonically reasonable ✓

Total wall time: 21.3 s
Status: 5/5 PHASES PASS + 1 SKIPPED-BY-DESIGN
        Step 1 launch gate: CLEAR (Exp5 SA1 architecture verified, fp32 production path)
```

### §6.2 step4_1_smoke_test.py(已 PASS,无需重跑)

```
ALL SMOKE PASSES (mode=None / diffusion_only / head_only / both)
4 个 mode 损失值与预期吻合:
  head_ce avg ~ 4.60 (random init ~ ln(100))
  diff_t avg ~ 1.30 (random init MSE)
  type_total 按 mode 正确加权
```

(注:smoke 在 CPU fp32 跑,与 phase 6.5 patch 无关,v1 PASS 仍有效。)

---

## §7 Notes for Exp5 Main Agent

1. **`EXP4_FILE_GUIDE.md` 由 MA5 在 Exp4 完结后补写** —— 区分本地/服务器/沙盒三路文件取用,以及每个文件具体 ssh cat / scp 命令模板。**Exp5 后续任何 SAx 启动后第一时间应阅读这份指南 §6 ask 模板**,避免重复 SA1 卡文件这一晚。

2. **GUIDE §2.1 描述与现实有一处不符**:`/home/tcat/diffcsp_exp4/code/step4/step4_1_smoke_test.py` 在 GUIDE 列出但服务器实际没有这个文件。SA1 已补写,建议 ping MA5 在 GUIDE 里 fix。

3. **Exp4 phase 6.5 PASS 声称疑点已解** —— 用户 2026-04-28 确认 Exp4 训练全程 fp32 不用 bf16,phase 6.5 当时是否真 PASS 不追究。SA1 改成 SKIPPED-by-design,文档化 3 处 hardcoded fp32 site,合理结案。

4. **OQ-1 / OQ-2 / OQ-3 三个 OPEN QUESTION 见 §4** —— 都不阻塞 SA2 启动。

5. **§5.6 PYTHONPATH 警告必须 carry over 给 SA2** —— 这是 SA1 第一稿漏提、用户严肃批评后补的关键项。SA2 训练入口若不显式控制 PYTHONPATH 优先级,有静默 import 旧版 Exp4 代码风险,整训练白费。

---

*Exp5-SA1 撰写,2026-04-28(final 版)。Exp5 Step 1 launch gate CLEAR,SA2 可启动。*
