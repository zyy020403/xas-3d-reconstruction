# EXP5_SA2_PRIME_LAUNCH_NOTE.md
# Exp5 v2 SA2' Launch — Training Sub-Agent

> **From**: MA5 (Exp5 v2 Main Agent)
> **To**: SA2' (Training sub-agent prime)
> **Date**: 2026-04-28
> **Status**: SA1' midterm reviewed and approved by MA5. SA2' GO.

---

## §0 你是谁,做什么,读什么

你是 Exp5 v2 SA2'。SA1' 已完成代码 surgery + 4 gates PASS + Exp4 baseline 锚定,你的任务是**启动 from-scratch 训练并守到 best ckpt 落地**(~32h)。

**必读 2 份**:
1. 本 note(MA5 review 决议 + Q1/Q3/Q4 + epoch 200 milestone)
2. `EXP5_STEP1_PRIME_OUTPUT.md`(SA1' 中期报告,**特别 §8** carry-over)

**不读**: proposal v2 全文 / Exp4 final report / v1 SA1 OUTPUT(SA1' 已吸收)。

**你的 4 个动作**(对应本 note §2-§5):
1. pre-flight check
2. nohup launch
3. 头 30 min 守屏
4. epoch 200 milestone protocol(到点 ping MA5,不停训练)

**红线**: 不动 holdout / 不改代码 / 不调 yaml / 不 warm-start / 不修 Phase 6.5。SA2' 是执行者,不是开发者。

---

## §1 MA5 Review 决议(三个 OPEN QUESTION)

SA1' OUTPUT §7 提了 5 个 OQ。MA5 决议如下:

### Q1 — 训练监控:**采纳 SA1' (c) val_loss only**

不加 EpochEndMetricsCallback。理由:
- val_loss 是 holistic 指标,反映训练健康度
- Tweedie shortcut 与 sample-time 不必然同向 → false-positive 风险
- 真 v2 vs Exp4 比较只能在 sample 后做(SA3 期)

**SA2' 监控就靠 4 个 sub-loss 的 PL 默认 epoch log**:
`val_loss / val_coord_loss / val_type_loss / val_density_loss`

### Q3 — Multiset F1 红绿灯:**SA1' 提案采纳,且修订 proposal v2**

| | proposal v2 §5.2 原值 | **MA5 采纳值(SA3 期使用)** |
|---|---|---|
| 🟢 主信号 | > 0.15(1.78× 改进) | **> 0.20(2.40× over Exp4 0.0843)** |
| ⚠️ 边际 | 0.10 - 0.15 | **0.10 - 0.20** |
| ❌ 失败 | < 0.10 | < 0.10 |

Set-Level TypeAcc secondary 阈值: **> 0.40**(Exp4 baseline 0.331 的 1.21×)。

SA2' 这条只是知情,不需要在训练时算。SA3 sample 后才用。

### Q4 — From-scratch epoch 200 警戒:**警戒采纳,fall-back 不是 warm-start**

warm-start 在 v2 实际不可行(decoder 528 vs 512 shape mismatch + MV-attention 是新的 + center_emb 是新的)。即使 partial load 可行,工程量比 from-scratch 还高,加速效果 < 50 epoch。

**真 fall-back 三选一**(epoch 200 时 MA5 决):
- (a) 趋势仍下降 → 继续训到 500
- (b) 平台化但 Multiset 已 > 0.15 → 提前接受当前 ckpt
- (c) 停滞或恶化 → 启动 Exp6 转向

详见 §5 milestone protocol。

### Q2 / Q5 — 不阻塞,知情即可

- Q2(Phase 6.4 loss 数没记录): SA2' 跑预启动 forward_test 时记录就行
- Q5(Phase 6.5 site 2/3 不修): 与 v1 SA1 一致,fp32 生产路径不受影响

---

## §2 Pre-flight Checklist(执行前必跑)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# 1. Verify CKPT_DIR is clean (NO Exp4 leftover ckpts — 528 vs 512 mismatch 会 crash)
ls -la /home/tcat/diffcsp_exp5/checkpoints/
# 期望: 空 或 只有 _smoke/ (smoke 残留, 若有则 rm -rf _smoke/)
# 如发现 Exp4 ckpt 残留: 必须 rm,这些不兼容

# 2. GPU + env
/home/tcat/conda_envs/mlff/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
# 期望: CUDA: True, devices: ≥ 1

# 3. v2 deliverables 在位
ls -la /home/tcat/diffcsp_exp5/code/step3/diffusion_w_type_xas.py \
       /home/tcat/diffcsp_exp5/code/step3/conf_xas/model/diffusion_xas.yaml \
       /home/tcat/diffcsp_exp5/code/step2/spectrum_encoder.py \
       /home/tcat/diffcsp_exp5/code/step4/step4_2_train.py
# 期望: 4 文件全在

# 4. 磁盘 + swap 状态(SA1' OUTPUT §5.4 警告过)
df -h / && free -h
# 期望: / 用量 < 95%(若 > 95%,清 /tmp/diffcsp_cache/ + 旧 wandb 再启动)

# 5. (可选)重跑一次 forward_test 抓 Phase 6.4 loss 实数(回应 Q2)
cd /home/tcat/diffcsp_exp5/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py 2>&1 | grep -E 'loss|PASS'
# 记录 Phase 6.4 loss 数,SA2' 中期报告附上
```

任一 fail → kill,贴 stderr 给 MA5。

---

## §3 Launch 命令

```bash
cd /home/tcat/diffcsp_exp5/code/step4

PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
nohup /home/tcat/conda_envs/mlff/bin/python step4_2_train.py \
    > /home/tcat/diffcsp_exp5/logs/step4_train_v2_stdout.log \
    2> /home/tcat/diffcsp_exp5/logs/step4_train_v2_stderr.log &

# Capture PID for later kill if needed
echo $! > /home/tcat/diffcsp_exp5/logs/step4_train_v2.pid
echo "PID: $(cat /home/tcat/diffcsp_exp5/logs/step4_train_v2.pid)"
```

---

## §4 头 30 min 守屏 Checklist

**必须坐在屏幕前 30 min**,看以下 5 项:

```bash
# 实时 tail stdout(前 30 min 不要离开)
tail -f /home/tcat/diffcsp_exp5/logs/step4_train_v2_stdout.log
```

| 检查项 | 期望 | 失败动作 |
|---|---|---|
| PYTHONPATH self-check 输出 `/diffcsp_exp5/...` | ✓ stdout 头部 | kill,环境问题 |
| Defensive assertions 全过(cost_lattice<1e-5 / cost_density==0.2 / num_heads==4 / decoder.latent_dim==528) | ✓ stdout 头部 | kill,yaml 不一致 |
| 第一个 epoch 启动,batch loss 出现 | ~30s 内 | kill,DataLoader 问题 |
| val_loss epoch 0 在 [2, 4](random init 区间) | epoch 1 末 PL 输出 | 若 > 5: 等 epoch 5,仍不回落 → kill |
| GPU utilization > 70%(`nvidia-smi`) | 持续 | 若 < 30%: pymatgen worker 卡,kill |

**特别注意 4 个 sub-loss 平衡**:
- `val_coord_loss`: 应在 ~ 1.0 - 1.5(diffusion 坐标 MSE)
- `val_type_loss`: 应在 ~ 1.3(diffusion type MSE,与 Exp4 同)
- `val_density_loss`: **可能比 Exp4 偏高**(因 cost_density 0.5→0.2,模型更允许它高)。乘 0.2 后总贡献应 ≤ 0.05
- `val_lattice_loss`: 任何值都行(cost_lattice=0,不进 total)

如 val_type_loss 起跳到 > 4 不回落 → kill,提前 abort 比硬撑 32h 强。

30 min 全过 → 关 ssh,等 ~ 32h。

---

## §5 Epoch 200 Milestone Protocol(MA5 新加)

v2 from-scratch,前 100-200 epoch 大概率不优于 Exp4(MV-attention 早期慢)。**SA2' 不做判断,只打 milestone signal**。

在 `step4_2_train.py` 的 PL `LightningModule.on_validation_epoch_end` 内或 `Callback.on_validation_end` 内,SA2' 加一条 print(只一处,~ 5 行,临 launch 前编辑):

```python
def on_validation_epoch_end(self):
    # ... existing code ...
    if self.current_epoch == 200:
        print("=" * 72, flush=True)
        print(f"EPOCH 200 MILESTONE — MA5 review needed", flush=True)
        print(f"  val_loss:         {self.trainer.callback_metrics.get('val_loss', 'N/A')}", flush=True)
        print(f"  val_coord_loss:   {self.trainer.callback_metrics.get('val_coord_loss', 'N/A')}", flush=True)
        print(f"  val_type_loss:    {self.trainer.callback_metrics.get('val_type_loss', 'N/A')}", flush=True)
        print(f"  val_density_loss: {self.trainer.callback_metrics.get('val_density_loss', 'N/A')}", flush=True)
        print("=" * 72, flush=True)
        # NOTE: do NOT stop training. MA5 reviews offline; SA2' continues.
```

**这条 print 不停训练**,只是给用户在 stdout log 里 grep 一行 marker。SA2' 继续训到 500 epoch 或 early stop。

用户看到 milestone 后会:
- 来 grep 4 sub-loss 趋势
- 决定 (a) 让继续(默认) / (b) 提前 sample / (c) kill 转 Exp6
- 把决议丢回 SA2' 或直接 ssh kill

如果 SA2' 上线时已经过了 epoch 200(不可能,但保险): 跳过这步,正常训。

---

## §6 训练完成后 Hand-back to MA5

best ckpt 落地 `/home/tcat/diffcsp_exp5/checkpoints/best-epoch*-val*.ckpt` 后:

写 `/home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md`,内容:

| 节 | 内容 |
|---|---|
| §1 训练参数确认 | yaml 关键字段 grep + 实际加载值 |
| §2 训练历程 | epoch / val_loss 折线(从 PL log 抓),含 epoch 200 milestone 的 4 sub-loss |
| §3 best ckpt 信息 | path / epoch / val_loss / 4 sub-loss / param count(应 = 3,650,388) |
| §4 wall time + GPU util | 实际耗时 / 平均利用率 |
| §5 异常事件 log | NaN / Inf / 卡住 / OOM 等 |
| §6 给 SA3 的 carry-over | best ckpt 路径 + sample 命令模板 + Exp4 baseline 对照锚点 |

**SA2' 不**: sample / 算 metrics / 跑 holdout。这些是 SA3 的工作。

---

## §7 速查红线

| | |
|---|---|
| ❌ 不改任何代码 / yaml(若需要,先 ping MA5) | |
| ❌ 不动 holdout(SA3 期才解禁) | |
| ❌ 不 warm-start from Exp4 ckpt(decoder shape mismatch,且 v2 是 from-scratch) | |
| ❌ 不修 Phase 6.5 hardcoded fp32(永久 SKIPPED-by-design) | |
| ❌ 不在 epoch 200 自己判断要不要停 — 只打 marker,MA5 决 | |
| ❌ 不在前 30 min 离开屏幕(必守) | |
| ✅ 任何 unexpected 立即 ping MA5,不自己改方案 | |

---

## §8 启动 checklist 你的第一条回复应

转给 SA2' 时,SA2' 第一条回复应:

1. 复述本 note §0 的 4 个动作(pre-flight / launch / 守屏 / milestone)
2. 复述 §7 的 5 条红线
3. 复述 Q1/Q3/Q4 决议各 1 句
4. 列出 pre-flight 命令的执行计划(顺序 1→5)
5. 估时(应 30 min pre-flight + 30 min 守屏 + ~32h 等待 + 0.5h 写 OUTPUT = 33h 总)

---

*MA5 撰写,2026-04-28。基于 SA1' OUTPUT 4 gates PASS 实证 + Exp4 baseline 实数(val Multiset 0.0843 / Set-Level 0.331)+ 用户拍板的 Q1c / Q3 0.20 阈值 / Q4 milestone protocol。*
