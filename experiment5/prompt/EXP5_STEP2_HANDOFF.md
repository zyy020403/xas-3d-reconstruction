# EXP5_STEP2_HANDOFF.md
# Exp5 Step 2 Sub-Agent 交接文档:Fine-tune training(decoupled head + center conditioning,from Exp4 ckpt)

> **撰写者**: Exp5 Main Agent
> **日期**: 2026-04-29
> **接收人**: Exp5-SA2(训练 Sub-Agent)
> **前置**: SA1 已完成(`EXP5_STEP1_OUTPUT.md` final 版),`/home/tcat/diffcsp_exp5/code/` 代码已落位且 forward_test 5 PASS + 1 skipped-by-design,smoke ALL PASS

---

## §0 使命(一句话)

把 SA1 落位的 Exp5 baseline_v2 网络(decoupled TypeClassifierHead + center-element conditioning + 三 mode flag)从 Exp4 best ckpt(`best-epoch366-val0.7300.ckpt`)warm-start 训到收敛,产出 `/home/tcat/diffcsp_exp5/checkpoints/best.ckpt`。

**不评估**(SA3 干)。**不读 holdout**。**不改 SA1 代码**(只读 + 加训练入口/recipe)。

---

## §1 必读背景(按顺序)

1. **EXP5_STEP1_OUTPUT.md §3 (yaml 默认值)、§5.1 (phased training)、§5.2 (ckpt warm-start)、§5.6 (PYTHONPATH)** — SA1 把所有训练前置都写明白了,**这四节是 SA2 的核心 input**
2. **EXP5_STEP1_OUTPUT.md §6 (实测日志摘要)** — 看一遍正常的 forward 各 loss 量级,训练时偏离这些量级要警觉
3. **EXP5_STEP0_OUTPUT.md §2 主结果** — 你训完的 ckpt 后续会被 SA3 用 K=5 hungarian_fold 评估,**SA2 不直接做这件事**,但要意识到 ckpt 的下游用途(影响 ckpt 保存策略,见 §6.5)
4. **EXPERIMENT4_FINAL_REPORT.md §6 训练 / §9.3 网络环境守卫包** — Exp4 训练 recipe 是 SA2 的 baseline 起点;包版本不动
5. **EXP4_PROPOSAL_v2.md §6 acceptance gates** — Exp4 训练验收范式参考(我在 §6 改写适配 Exp5)

---

## §2 训练设计

### §2.1 总体策略:warm-start + phased + early stop

```
                    Exp4 best ckpt (val_loss=0.7300, epoch=366)
                                    │
                                    │ load_state_dict(strict=False)
                                    │ — 4 类 missing keys (§5.2 list)
                                    │ — decoder.atom_latent_emb shape mismatch → silently skip
                                    ↓
                  Exp5 model (head + center_emb 随机初始化,backbone Exp4 weights)
                                    │
                          Phase 1 (head warmup, epoch 0-5)
                          freeze:  spectrum_encoder.* (除 center_emb)
                                   decoder.* (除 atom_latent_emb)
                          unfreeze: type_head.*
                                    spectrum_encoder.center_emb.*
                                    decoder.atom_latent_emb.*  ← shape changed,必须重训
                          lr: 1e-3 (head), 1e-3 (center_emb), 1e-3 (atom_latent_emb)
                                    │
                                    ↓
                          Phase 2 (joint fine-tune, epoch 6-end)
                          unfreeze 全部
                          lr (differential):
                            head + center_emb + atom_latent_emb : 1e-4
                            其它 backbone                        : 1e-5
                          early_stop on val_loss_total, patience=30
                                    │
                                    ↓
                        best.ckpt + last.ckpt → /home/tcat/diffcsp_exp5/checkpoints/
```

### §2.2 三 mode 决策:default `both`,**不要**在 SA2 主训练里切换

SA1 实测三 mode 都 forward 不 NaN,你训练 default `type_loss_mode=both`(yaml 已是 default)。**主训练只跑这一个 mode**。

如果训完 TypeAcc 不达标,**那是 SA2-续 的事**,不是 SA2:改 yaml 一行 `head_only` 重训(可以从 SA2 的 best 接着 fine-tune,不必从 Exp4 ckpt)。

### §2.3 关键超参(yaml 已默认,SA2 一般不动)

| 超参 | 值 | 来源 / 备注 |
|---|---|---|
| optimizer | AdamW | Exp4 沿用 |
| weight_decay | 0 | Exp4 沿用,扩散模型一般不加 |
| precision | **fp32** | MA4 决策 D1,继承不动;**不要试 bf16/amp**(SA1 §5.7 解释了 3 处 hardcoded fp32 site) |
| batch_size | 16 | Exp4 沿用;如显存吃紧降 8 |
| max_epochs | **400** | Exp4 训 489 epoch 收敛,fine-tune 应快得多;给 patience=30 自然停 |
| early_stop monitor | `val_loss_total` | 不是单独 coord 或 type;监控总目标 |
| early_stop patience | 30 | Exp4 沿用 |
| early_stop mode | min | val_loss 越小越好 |
| ckpt save_top_k | 3 | best-3 + last,够 SA3 / 反查用 |
| ckpt monitor | `val_loss_total` | 与 early_stop 同 |
| log_every_n_steps | 50 | Exp4 沿用 |
| val_check_interval | 1.0 | 每 epoch 跑一次 val |
| seed | 42 | Exp4 沿用,确保可复现 |

### §2.4 Phased training 实现选项

**SA2 自己选哪种,但任选一种必须做**(不做就是裸训,SA1 §5.1 已警告大梯度风险):

**选项 A:`on_train_epoch_start` 切换 `requires_grad`**(更 PL-style,推荐)

```python
def on_train_epoch_start(self):
    if self.current_epoch == 0:
        # Phase 1: freeze most, unfreeze new modules + reshape layer
        for name, p in self.named_parameters():
            p.requires_grad = (
                name.startswith("type_head.") or
                name.startswith("spectrum_encoder.center_emb.") or
                name.startswith("decoder.atom_latent_emb.")
            )
        self.log_phase_status(phase=1)
    elif self.current_epoch == 6:
        # Phase 2: unfreeze 全部
        for p in self.parameters():
            p.requires_grad = True
        # Differential lr 通过 optimizer param_groups 实现(见下)
        self.log_phase_status(phase=2)
```

注意 PL 不会因为 `requires_grad=False` 重建 optimizer。Differential lr 必须从 init 时就在 `configure_optimizers` 里分两个 param_group:

```python
def configure_optimizers(self):
    head_params, backbone_params = [], []
    for name, p in self.named_parameters():
        if (name.startswith("type_head.") or
            name.startswith("spectrum_encoder.center_emb.") or
            name.startswith("decoder.atom_latent_emb.")):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return torch.optim.AdamW([
        {"params": head_params,     "lr": 1e-3, "name": "head_lr"},      # phase 1 lr
        {"params": backbone_params, "lr": 0.0,  "name": "backbone_lr"},  # phase 1 frozen
    ])
```

然后在 phase 切换时手动改 `optimizer.param_groups[0]['lr']`、`[1]['lr']`。

**选项 B:两段式 `Trainer.fit()`**(更直观但要正确处理 ckpt 续接)

```python
# Phase 1
trainer1 = pl.Trainer(max_epochs=6, ...)
trainer1.fit(model, datamodule)
ckpt_phase1 = trainer1.checkpoint_callback.last_model_path

# Phase 2 (resume from phase 1, unfreeze 全部, 改 lr)
model = ExpFiveModel.load_from_checkpoint(ckpt_phase1)
for p in model.parameters(): p.requires_grad = True
# reconfigure optimizer with differential lr
trainer2 = pl.Trainer(max_epochs=400, ...)  # 接着训到 400
trainer2.fit(model, datamodule, ckpt_path=ckpt_phase1)
```

选项 A 更符合 PL 的 idiomatic 写法,我推荐 A。但 SA2 自己拍。

### §2.5 LR warmup 兜底(可选)

如果 SA2 觉得 phased training 实现复杂,**至少要做 lr warmup**:前 1000 steps lr 从 0 线性升到目标(1e-4 for head/atom_latent_emb, 1e-5 for backbone)。降低 head 随机初始化的初始大梯度对 backbone 的破坏。

但 phased training + lr warmup 都做最稳。

---

## §3 Sanity check(写代码前 + 训练前 + 训练 epoch 0 后)

### §3.1 写代码前(5 min)

```bash
# Pre-flight: 服务器磁盘 + GPU 状态
df -h /
free -h
nvidia-smi
ls -lh /home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py     # ~589 行,SA1 final
ls -lh /home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt
md5sum /home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt
# 期望 md5: dc9d2c9b371c78125f285a5a6478d404 (SA0 §1 已 verify)
```

### §3.2 训练前(SA2 训练入口脚本开头必跑)

**OQ-1 决议 — max(atom_types) 扫描**(SA1 留的 OPEN QUESTION 1):

```python
# 在 train.py 开头,DataModule init 后,Trainer.fit() 前
import torch
print("[OQ-1 sanity] Scanning train atom_types max...")
max_z = 0
for batch in datamodule.train_dataloader():
    max_z = max(max_z, int(batch.atom_types.max()))
    if max_z >= 100: break  # early exit if hit
print(f"[OQ-1 sanity] max(atom_types) over train = {max_z}")
assert max_z < 100, f"OQ-1 FAIL: max(atom_types)={max_z} ≥ MAX_ATOMIC_NUM=100. STOP and report MA."
print(f"[OQ-1 sanity] PASS (max < 100, n_elements=100 sufficient)")
```

如果 fail,**立刻停训上交 MA**——这意味着 yaml 的 `n_elements=100` 不够,要扩到 110 重训。

### §3.3 训练 epoch 0 后 5-10 step(关键存证)

SA2 训练 log **前 50 step 必看**:

| 信号 | 期望 | 异常处理 |
|---|---|---|
| `loss_coord` | epoch 0 ~ 0.7300(继承 Exp4)+ ε | 如 >0.85 不回落 → phased training 没做好(§5.1 诊断) |
| `loss_diffusion_type` | ~ 1.30(SA1 实测 1.3368) | 偏离大 → backbone 加载有问题 |
| `loss_type_ce_head` | epoch 0 ~ 4.60(随机初始化 ~ ln(100)=4.605),前 1000 step 应快速降到 ~3.5 | 降不下来 → head 没学 / center_emb 没接进去 |
| `loss_total` | ~ 4.80 起步,稳定下降 | NaN / 振荡 → lr 太高或 phased 切换出问题 |
| `grad_norm` | ~ 5-15 | >100 → 梯度爆炸,加 clip(默认 1.0) |

---

## §4 OQ 决议(MA 拍板,SA2 执行)

| OQ | SA1 留下的问题 | MA 决议 | SA2 怎么做 |
|---|---|---|---|
| **OQ-1** | max(atom_types) ≤ 100? | 训练入口加 §3.2 扫描 + assert | 必须做。fail 则停训上交 |
| **OQ-2** | handoff 笔误 (MSE vs CE 量级) | acknowledge,无影响 | 不动 |
| **OQ-3** | `cost_type` yaml 字段彻底删 vs 留 | **删** | 在 yaml 里删一行 `cost_type: 1.0`(代码不依赖,SA1 已实证)。理由:留着会让 SA2/SA3 看 yaml 时混淆 |

---

## §5 实施步骤

### §5.1 准备(磁盘清理 + cache 准备)

```bash
# §5.4 SA1 警告:磁盘 92% / swap 80% — SA2 训练前必须清理
ssh tcat@scsmlnprd02 "df -h /; du -sh /home/tcat/diffcsp_exp4/wandb/ 2>/dev/null; ls /tmp/diffcsp_cache/ 2>/dev/null"
# 清理 candidate(SA2 自己判断,Exp5 全程预算 30-50 GB):
#   - 旧 wandb runs (Exp4): 直接 rm
#   - /tmp/diffcsp_cache/ 旧 cache: rm 然后重建
#   - /home/tcat/diffcsp_exp4/checkpoints/ 中除 best-epoch366 外: 不动(Exp4 完结)
#   - /home/tcat/diffcsp_exp5/sa0/samples_raw_K10.pt: 2.7 MB 不动
# 目标:训练启动前 / 至少 200 GB 空闲

# 创建 cache(IO 加速,继承 Exp4 §6 做法)
mkdir -p /tmp/diffcsp_cache
cp -r /home/tcat/diffcsp_exp5/data/* /tmp/diffcsp_cache/  # ~650 MB,30 sec
```

### §5.2 训练入口脚本(SA2 新写,放 `/home/tcat/diffcsp_exp5/code/step4/step4_2_train.py`)

骨架(细节 SA2 实现):

```python
#!/usr/bin/env python
# step4_2_train.py — Exp5 SA2 主训练入口
import sys
# §5.6 PYTHONPATH 优先级(SA1 §5.6 carry-over,实测过的写法)
sys.path.insert(0, "/home/tcat/diffcsp_exp5/code/step3")
sys.path.insert(0, "/home/tcat/diffcsp_exp5/code/step2")
# Exp4 code 必须在末尾(diffcsp 子包)
sys.path.append("/home/tcat/diffcsp_exp4/code")

# 自检 import 路径(SA1 §5.6 强调)
import diffusion_w_type_xas
assert "/diffcsp_exp5/" in diffusion_w_type_xas.__file__, \
    f"WRONG IMPORT PATH: {diffusion_w_type_xas.__file__}"
print(f"[PYTHONPATH self-check] diffusion_w_type_xas: {diffusion_w_type_xas.__file__}")
import spectrum_encoder
assert "/diffcsp_exp5/" in spectrum_encoder.__file__
print(f"[PYTHONPATH self-check] spectrum_encoder: {spectrum_encoder.__file__}")

# ... [实例化 model + datamodule from yaml] ...

# OQ-1 sanity (§3.2)
# ... [scan max(atom_types)] ...

# Warm-start with strict=False (§5.2 期望 missing/unexpected 列表)
ckpt_path = "/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print(f"[CKPT WARM-START] missing keys ({len(missing)}):")
for k in missing: print(f"  - {k}")
print(f"[CKPT WARM-START] unexpected keys ({len(unexpected)}):")
for k in unexpected: print(f"  - {k}")

# 期望 missing(对照 SA1 §5.2 list,SA2 必须 visual diff):
#   spectrum_encoder.center_emb.weight                ← Exp5 新模块
#   type_head.fc.0.weight, .bias                      ← Exp5 新模块
#   type_head.fc.2.weight, .bias                      ← Exp5 新模块
#   decoder.atom_latent_emb.weight, .bias             ← shape (256, 528) vs Exp4 (256, 512)

# 期望 unexpected: 空列表
assert len(unexpected) == 0, f"OQ-checkpoint compatibility FAIL: unexpected keys = {unexpected}"

# 验证 missing key 与期望严格一致
expected_missing_prefixes = [
    "spectrum_encoder.center_emb.",
    "type_head.",
    "decoder.atom_latent_emb.",
]
for k in missing:
    assert any(k.startswith(p) for p in expected_missing_prefixes), \
        f"UNEXPECTED missing key: {k}"

# 训练
# ... [Trainer.fit() with phased callback] ...
```

### §5.3 改 yaml(执行 OQ-3 决议)

```bash
cd /home/tcat/diffcsp_exp5/code/step3/conf_xas/model
cp diffusion_xas.yaml diffusion_xas.yaml.bak_sa2  # 锚点,以防回滚
# 删 `cost_type: 1.0` 那一行(SA1 §3 决策 c 已说明代码不依赖)
sed -i '/^cost_type:/d' diffusion_xas.yaml
# verify
grep cost_type diffusion_xas.yaml || echo "✓ cost_type removed"
```

### §5.4 训练命令

```bash
cd /home/tcat/diffcsp_exp5/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/tmp/diffcsp_cache \
nohup /home/tcat/conda_envs/mlff/bin/python /home/tcat/diffcsp_exp5/code/step4/step4_2_train.py \
  > /home/tcat/diffcsp_exp5/logs/step2_train.log 2>&1 &
echo "Train PID: $!"
```

监控:

```bash
tail -F /home/tcat/diffcsp_exp5/logs/step2_train.log    # SA2 实时跟看,前 50 step + 每 epoch 跨入时
nvidia-smi -l 5                                          # GPU 利用率,期望 60-90%
```

### §5.5 训练中检查点(SA2 责任)

| 时间点 | 必查 |
|---|---|
| 启动 + 1 min | PYTHONPATH self-check 两条 print 都说 `/diffcsp_exp5/`;OQ-1 sanity PASS;ckpt 加载 missing/unexpected 与期望一致 |
| epoch 0 末 | val_loss_total 起点(应在 4.5-5.0 区间);val_coord_loss 应 ~ 0.73 |
| epoch 5 末(phase 1 → 2 切换) | val_loss_total 已下降(head warmup 应让 head_ce 从 4.6 降到 ~3.5,total 从 4.8 降到 ~4.0);log 出现 phase 2 切换 print |
| epoch 10 末 | val_loss_total 持续下降;val_coord_loss 不应 >0.85(否则 phased 失败,§5.1 诊断) |
| 每 ~5 epoch | 看下三个 loss 分量趋势,有无个别分量 stuck;ckpt 文件正常落地 |

---

## §6 验收闸门(SA2 完成的硬标准)

**8/8 必须通过才能交棒 SA3**:

1. ✅ **PYTHONPATH self-check 两条 print 都包含 `/diffcsp_exp5/`**(写进 train log 开头,SA1 §5.6 关键 carry-over)
2. ✅ **OQ-1 sanity scan PASS**:max(atom_types) < 100(SA2 train log 内打印)
3. ✅ **ckpt warm-start missing/unexpected 与期望严格一致**:unexpected = 空,missing 全部命中 §5.2 list 的 4 类前缀(SA2 train log 开头打印,**MA 验收时会 visual diff**)
4. ✅ **val_loss_total 收敛或 epoch 上限到达**(early stop 触发,或 max_epochs=400 跑完)
5. ✅ **val_coord_loss(end of training) ≤ 0.7500**(不能比 Exp4 起点 0.7300 显著差;ε=0.02 容忍)
6. ✅ **三个 loss 分量(coord / diffusion_type / type_ce_head)训练全程无 NaN/Inf**(SA2 grep log 确认)
7. ✅ **best.ckpt + last.ckpt + 完整 train log 在 `/home/tcat/diffcsp_exp5/checkpoints/` + `/home/tcat/diffcsp_exp5/logs/`**(SA3 即拿即用)
8. ✅ **写完 EXP5_STEP2_OUTPUT.md**,内容包含:
   - 8 个闸门各自的实测证据(数字 + log 引用)
   - 训练耗时 wall time / epoch 数 / 最终 val_loss_total / val_coord_loss / val_loss_type_ce_head
   - phased training 的实施细节 + 切换时点 log 摘要
   - ckpt 加载 missing/unexpected 完整 dump
   - **§Notes_for_SA3**(SA2 必 carry over,见 §8)
   - 任何 OPEN QUESTION

**注意**:**没有 TypeAcc 闸门**。TypeAcc 是 SA3 用 K=5 hungarian_fold 在 sample 后才能算出来的指标,**SA2 训练时 log 的 head_ce 不能直接换算 TypeAcc**。SA2 不要试图自己评估,那是 SA3 的事。

---

## §7 红线(绝对不能动)

- `holdout_samples_v2.csv` / `spectra_holdout.pkl`:**全程不读**(继承)
- `incompat_pool.csv`:**封存**(继承)
- `/home/tcat/diffcsp_exp4/`:**read-only**(包括 ckpt、code、data),不写不删
- `/home/tcat/diffcsp_exp5/sa0/`:**SA0 领地**,SA2 不写不读
- L=6, min-image, [-0.5, 0.5] coord, cost_lattice=0, N_NEIGHBORS=20:**全部继承,不动**
- silent drop + collate filter:**继承**
- precision=fp32:**继承不动,不试 bf16/AMP**(SA1 §5.7 解释了 3 处 hardcoded fp32 site)
- 网络环境守卫包(7 核心 + 18 子依赖):**不升级**(详见 Exp4 final report §9.3)
- SA1 写好的 7 个文件:**不改**(只读、import、训练)。如果 SA2 发现 SA1 代码有 bug,**停下来上交 MA**,不要自己修
- 三 mode flag 主训练只跑 `both`:不切换;切换是 SA2-续 的事

---

## §8 Notes for SA3(SA2 必须 carry over 进 EXP5_STEP2_OUTPUT.md)

这一节 SA2 不实现,但**必须原样写进 OUTPUT.md 的 §Notes_for_SA3 段落**,确保 SA3 看到。

### §8.1 评估标配 K=5 hungarian_fold(SA0 决议)

SA0(`/home/tcat/diffcsp_exp5/sa0/`)在 Exp4 ckpt 上验证:K=5 hungarian-with-fold 给 ΔRMSD −0.056 / ΔTypeAcc +0.040,wall time 5×。**SA3 评估必须同时报 K=1 + K=5 两组数字**,Exp4 和 Exp5 ckpt 都报,主线对比维度才一致。

K=5 hungarian_fold 实现已 production-ready 在 `/home/tcat/diffcsp_exp5/sa0/scripts/multisample_aggregate_v2.py`,SA3 只需替换 ckpt 路径用。

**绝对不要用 naive averaging**(SA0 §5 验证有 torus-bug);**不要用 medoid**(TypeAcc 反而掉)。

### §8.2 三 mode 的 ckpt 兼容性

SA2 default 训 `both` mode。SA3 评估 ckpt 时,yaml 仍设 `both`(同训练 mode);如果 SA3 想做 ablation(同 ckpt 在 `head_only` mode 下评估),只改 yaml 一行,代码不动 —— 因为 head 和 diffusion 内部 type prediction 都被训了,evaluator 只是选哪条路径出 type。

### §8.3 head_predict_types 接口

SA1 §3 决策中 `diffusion_w_type_xas.py` 暴露 `head_predict_types(latent, center_emb) → (B, 20)` 整数 tensor 方法。SA3 sample 路径里:
- coord 用 diffusion sample
- type 用 `head_predict_types` 的 argmax(替代 diffusion 内部的 type prediction)
- Hungarian 用坐标距离 cost matrix,与 SA1 EXP5_STEP1_HANDOFF §2.3 描述一致

### §8.4 best ckpt 的选择

SA2 save_top_k=3,落地 3 个 best ckpt。SA3 默认用 `best`(monitor=val_loss_total 最低)。如果想试更早的 ckpt(可能 val_coord_loss 更低但 val_loss_type_ce_head 更高,bias 不同),SA2 OUTPUT.md 应列出 top-3 各自的三个 loss 分量,SA3 自选。

---

## §9 时间预算

| 阶段 | 预估 wall time |
|---|---|
| 写训练入口 + phased callback | 2-4 h |
| 磁盘清理 + cache 重建 + smoke 5 step verify | 1 h |
| 完整训练(fp32, bs=16, 单 4090) | 12-24 h(fine-tune,远快于 Exp4 from-scratch 的 32h+;若 phased + early_stop 触发可能更短) |
| OUTPUT.md 撰写 | 1-2 h |
| **总计** | **~1.5-2 day** |

如果训练超过 36h 还未收敛 / early stop 未触发,**停下来上交 MA**——不正常。

如果 epoch 5 末 phase 1 → 2 切换后 val_loss_total 反而上升不回落(SA1 §5.1 诊断信号),**停下来上交 MA**——phased training 实施有问题,可能要回滚到 lr warmup 兜底方案。

---

## §10 与其它 SA 的关系

| SA | 状态 | SA2 关系 |
|---|---|---|
| SA0(K-averaging quick win) | ✅ 完成 | SA0 工件在独立 `/sa0/` 路径,不冲突。SA0 的 K=5 hungarian_fold 结论传给 SA3,SA2 只需知道 ckpt 后续要被这样评估 |
| SA1(架构改造) | ✅ 完成 | SA2 的 input。SA1 7 个改动文件已落位,SA2 只读不改。OQ-1/2/3 已由 MA 决议(§4) |
| SA2(本任务) | 🔄 待启动 | — |
| SA3(评估) | ⏳ 待 SA2 完成 | SA2 OUTPUT.md §Notes_for_SA3 把 §8 三条 carry over 给 SA3 |
| SA4(可视化 + 横向对比 + Phase B 决策) | ⏳ 待 SA3 完成 | — |

---

## §11 接力链工作哲学(从 Exp4 + SA1 经验继承)

- **诚实 > 流畅**:训练异常 / phased 切换出意外 / 数字偏离期望,在 OUTPUT.md 里写"我观察到 X,我假设 Y,我做了 Z"。MA 会判断
- **70% 上下文闸门**:context 用到 70% 时停下来交付现状,**不要硬撑**
- **不深 debug**:训练报错且 30 min 无头绪,贴 stacktrace + 诊断假设交回 MA
- **状态锚定**:`.bak_sa2` 锚点(yaml)+ `git status`(如 git init 了)+ ckpt 文件名时间戳

---

*Exp5 Main Agent 撰写,2026-04-29。SA2 接收后请在 24h 内回 ack 并报当日 §3.1 + §3.2 + §5.5 启动 + 1 min 三处 sanity check 结果。*
