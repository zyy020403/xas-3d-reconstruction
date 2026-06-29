# EXP5_PRIME_STEP2_CONTINUE_HANDOFF.md
# SA-EXP5'-STEP2-TRAIN 续棒(STEP2-CONTINUE)launch note

> **From**: Exp5'-MA
> **To**: SA-EXP5'-STEP2-TRAIN(当前窗口续棒,不开新窗口)
> **日期**: 2026-05-04
> **任务范围**: 修 ckpt callback monitor bug + warm-start from last.ckpt + 续训到真 EarlyStop(~ 4-9h GPU)
> **预期 hand-back**: 续训完成 → 新 last.ckpt + 完整 epoch 趋势 → Exp5'-MA review → STEP3-SAMPLE 启动

---

## §0 一屏掌握

### 0.1 为什么续棒

STEP2-TRAIN 已完成的工作不动。但 V1 verify 揭示 epoch 152-154 三个指标**同时**发生阶跃改进:
- val_composite: 0.575 → 0.576
- val_loss: 76.80 → 75.60(降 1.5%)
- val_min_d: 1.580 → 1.590

这不是 plateau 末尾的噪声,是**还在缓慢爬升**的趋势。之前看似 0.575 长期平台是 PL prog_bar 的 3 位数值精度造成的视觉错觉(真实值可能是 0.5751→0.5759 缓爬),末尾突破精度门槛才显示出来。

EarlyStopping 是用错误的 `val_gate_pass_rate` 监控器触发的(详 errata 4 §2),**用对的 `val_composite_ckpt_score` 监控器,EarlyStop 不应该在 epoch 154 触发**。

### 0.2 你的 6 个子任务

| 步 | 任务 | 工程量 |
|---|---|---|
| K1 | 5 个 ckpt 永久档案 verify(防误删)+ 启动前 verify | 10 分钟 |
| K2 | 修 train.py:callback monitor 真注册 `val_composite_ckpt_score` | 0.3 天 |
| K3 | warm-start dry-run(2 epoch,验证 LR 接续 + monitor 生效) | 30 分钟 |
| K4 | 启动正式续训(warm-start from last.ckpt,真 EarlyStop monitor=composite) | 启动 30 分钟 |
| K5 | epoch 1-3 监控(LR 是否跳跃 + composite 是否真在爬) | 1-2h |
| K6 | 续训结束 hand-back(到真 EarlyStop 或 epoch 500) | 0.3 天 |

**总:1 天工程 + 4-9h GPU 无人值守**

### 0.3 Exp5'-MA 已拍板的 6 条不再讨论

1. **Warm-start from `/home/tcat/diffcsp_exp5_prime/checkpoints/last.ckpt`**(epoch 154,md5=9cd39421187df8d02951b9389266de36)— **不 from-scratch**,STEP2 已投入 7h 不浪费
2. **callback monitor 严格 `val_composite_ckpt_score`**(launch note §0.4 #1 拍板,errata 4 §2 揭示 bug,STEP2-CONTINUE 必修)
3. **patience = 30(EarlyStop)+ save_top_k = 3**(STEP2 用 1,真 best 丢失教训)
4. **MAX_EPOCHS 仍是 500**(LR scheduler T_max=500 跟随,不动 errata 4 §3.4)
5. **三件套 cost 不动**(1.0 / 0.5 / 0.2)
6. **batch=64 / workers=16 / PreCollatedDataset 不回退**(回退=重训,违背"已投入 7h 不浪费"原则)

---

## §1 Step K1 — 启动前 verify(10 分钟)

### 1.1 5 个 ckpt 永久档案 verify(防误删)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp5_prime/checkpoints/

ls -la
# 期望 5 个文件:
# - last.ckpt                                          (active,这次 warm-start 起点)
# - last.ckpt.frozen_step2_final                       (永久档案)
# - epoch=004-gate=0.5305.ckpt                         (lucky shot active)
# - epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot (永久档案)

md5sum last.ckpt
# 期望 9cd39421187df8d02951b9389266de36 (与 STEP2 hand-back 一致,确认未被改)
md5sum last.ckpt.frozen_step2_final
# 期望相同 md5(frozen 与 active 副本一致)
```

### 1.2 启动前环境 verify(沿用 STEP2 §1)

```bash
# (A) train.py 当前 md5 (STEP2 hand-back 后)
md5sum /home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py
# STEP2 时改完 md5 = 4b21cab63c775ee4647e46593d887a31 + 后续优化(SA hand-back 没记最终 md5,SA 报 Exp5'-MA 当前值)

# (B) STEP1-FIX-C 关键代码 md5(不动)
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py
# 期望 0bc6fc346e60b990e3a9fc25140000f0(STEP1-FIX-C 后)
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py
# 期望 94432ba56a7f3fd2ab0ce6281b66c5e6(STEP1-FIX-C 后)
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/conf_xas/model/diffusion_xas.yaml
# 期望 f73123a16166b220646af3537f7ece5b

# (C) cache + shell_boundaries
ls -la /home/tcat/diffcsp_exp5_prime/data/*.pt
cat /home/tcat/diffcsp_exp5_prime/data/cache_metadata.json  # L=20

# (D) 磁盘(续训 ~ 4-9h × 3 个新 ckpt + log,~ 200-500 MB)
df -h /home/tcat
du -sh /home/tcat/diffcsp_exp5_prime/

# (E) GPU 0 idle
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

### 1.3 PASS gate K1

- ✅ 5 个 ckpt 文件齐(last + epoch=4 各 2 份永久档案)
- ✅ last.ckpt md5 = `9cd39421187df8d02951b9389266de36`(未被改)
- ✅ STEP1-FIX-C 3 个关键文件 md5 不变
- ✅ cache L=20 完好
- ✅ 磁盘 ≥ 50 G avail
- ✅ GPU 0 idle

---

## §2 Step K2 — 修 train.py callback monitor bug(0.3 天)

### 2.1 cp 锚点

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step4
cp step4_2_train.py step4_2_train.py.bak_pre_step2_continue
md5sum step4_2_train.py.bak_pre_step2_continue
```

### 2.2 根因(errata 4 §2 揭示)

```
RuntimeError: Early stopping conditioned on metric `val_composite_ckpt_score`
which is not available. ...
```

LightningModule 的 `on_validation_epoch_end` 中 `self.log('val_composite_ckpt_score', ...)` 在第一个 validation epoch 之后才生效。PL Trainer 启动时 EarlyStopping 检查 monitor metric 注册表 → 不存在 → RuntimeError。

### 2.3 修复方案(选 B,SA 实施)

**A. 在 LightningModule `__init__` 末尾用 `self.log` 注册 dummy 值**(不可行,`self.log` 在 `__init__` 阶段没 trainer 上下文,会报错)

**B. ⭐ 改用 `strict=False` + `verbose=True`**(推荐,PL 原生支持)

```python
# 原代码(STEP2 SA 改的):
early_stop = EarlyStopping(
    monitor='val_composite_ckpt_score',  # → RuntimeError 因为还没 log 过
    mode='max',
    patience=30,
)
ckpt_cb = ModelCheckpoint(
    monitor='val_composite_ckpt_score',  # 同上
    mode='max',
    save_top_k=1,
)

# ⭐ 改为(STEP2-CONTINUE):
early_stop = EarlyStopping(
    monitor='val_composite_ckpt_score',
    mode='max',
    patience=30,
    strict=False,           # ⭐ 关键:metric 不存在时跳过(第一个 val epoch 前)而不是 raise
    verbose=True,           # ⭐ log 何时启用,便于 verify
    check_finite=True,
)
ckpt_cb = ModelCheckpoint(
    dirpath='/home/tcat/diffcsp_exp5_prime/checkpoints',
    filename='composite_best_epoch{epoch:03d}_score{val_composite_ckpt_score:.4f}',
    monitor='val_composite_ckpt_score',
    mode='max',
    save_top_k=3,           # ⭐ STEP2 用 1 导致真 best 丢失,改 3 留 ablation 余地
    save_last=True,
    auto_insert_metric_name=False,
)
```

**C. 备选:LightningModule 在 `setup` hook 里注册**(PL 2.5 推荐)

```python
# 在 LightningModule 类里加:
def setup(self, stage=None):
    super().setup(stage)
    # 预注册 metric 让 callback init 不报错
    if stage == 'fit':
        self.log('val_composite_ckpt_score', 0.0, on_step=False, on_epoch=True)
```

但 setup 阶段也未必能 log,SA 实施时 **优先 B**(strict=False)。**B 失败再尝试 C**。

### 2.4 命名规则修订(errata 4 §5.4)

ckpt filename 不写 "best" / "failed",只写事实:

```python
filename='composite_epoch{epoch:03d}_score{val_composite_ckpt_score:.4f}'
# composite_epoch200_score0.5821.ckpt — 描述事实
```

`save_last=True` 自动生成 `last.ckpt`,**会覆盖现有的 last.ckpt**。SA 启动续训前必须把现有 last.ckpt rename 防覆盖:

```bash
# K2 改完 train.py 后,启动续训前
cd /home/tcat/diffcsp_exp5_prime/checkpoints/
mv last.ckpt last.ckpt.from_step2_baseline    # rename 防被续训覆盖
ls -la
# 验证只有:
# - last.ckpt.from_step2_baseline (rename 后)
# - last.ckpt.frozen_step2_final (永久档案)
# - epoch=004-gate=0.5305.ckpt (留)
# - epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot (永久档案)
# 没有 active last.ckpt(等续训生成新的)
```

### 2.5 PASS gate K2

- ✅ train.py.bak_pre_step2_continue 锚点存在
- ✅ EarlyStopping `strict=False`、`verbose=True`、monitor=`val_composite_ckpt_score`
- ✅ ModelCheckpoint monitor=`val_composite_ckpt_score`、`save_top_k=3`、`save_last=True`
- ✅ filename 不含 "best"/"failed",只含 epoch + score
- ✅ STEP2 last.ckpt rename 为 `last.ckpt.from_step2_baseline`(防续训覆盖)
- ✅ trainer.fit() 调用 `ckpt_path='/home/tcat/diffcsp_exp5_prime/checkpoints/last.ckpt.from_step2_baseline'` 实施 warm-start

### 2.6 SA 报告新 train.py md5

```bash
md5sum /home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py
# 给 Exp5'-MA review,确认改动符合 §2.3 + §2.4
```

---

## §3 Step K3 — warm-start dry-run(30 分钟)

### 3.1 跑 2 epoch dry-run

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step4
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# 临时改 MAX_EPOCHS = 156(154 起 + 2 epoch dry-run)or 用 CLI --max-epochs 156(若支持)
# 推荐 sed 临时改:
sed -i.bak_dryrun 's/MAX_EPOCHS\s*=\s*500/MAX_EPOCHS = 156/g' step4_2_train.py
# 跑 dry-run
CUDA_VISIBLE_DEVICES=0 /home/tcat/conda_envs/mlff/bin/python step4_2_train.py 2>&1 | \
    tee /home/tcat/diffcsp_exp5_prime/logs/step2_continue_dryrun.log
# 完了恢复 500
sed -i 's/MAX_EPOCHS\s*=\s*156/MAX_EPOCHS = 500/g' step4_2_train.py
rm step4_2_train.py.bak_dryrun  # 删 sed 自动备份(因为我们已经有 .bak_pre_step2_continue)
```

### 3.2 dry-run 关键 verify(SA 必报)

| 项 | verify 方法 | 预期值 |
|---|---|---|
| Warm-start 加载成功 | 看 log "Restored ... from checkpoint" 字样 + epoch 从 154 而非 0 起算 | ✅ |
| LR 接续不跳跃 | 第一个续训 step 的 LR vs STEP2 epoch 154 末尾 LR | 差 < 5% |
| `val_composite_ckpt_score` 真被 monitor | 看 log `EarlyStopping verbose=True` 输出,确认 metric 已识别 | "EarlyStopping ... will use metric `val_composite_ckpt_score`" |
| 6 active loss 数量级 | 与 STEP2 epoch 154 端值一致 | 差 < 10% |
| 第一个 val epoch 后 ckpt 落盘 | `composite_epoch155_score*.ckpt` + `last.ckpt` | 2 个文件 |

### 3.3 dry-run 极重要的 LR check

```bash
# 提 dry-run 内 LR 数值
grep -oE "lr-Adam=[0-9.e+-]+" /home/tcat/diffcsp_exp5_prime/logs/step2_continue_dryrun.log | head -10
# vs STEP2 末尾 LR
grep -oE "lr-Adam=[0-9.e+-]+" /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log | tail -10
# 两段 LR 数值应连续(可能在 ~ 7e-5 量级,CosineAnneal 走到 epoch 154 衰减后)
```

**如 LR 跳跃 > 10%**:warm-start 加载 scheduler state 失败,SA 立即 stop,贴 log 给 Exp5'-MA(可能是 PL ckpt 加载兼容性问题,需调试)。

### 3.4 PASS gate K3

- ✅ 2 epoch dry-run 完成,无 RuntimeError(callback bug 修复成功的硬证)
- ✅ epoch 编号从 155 开始(不是 0)— warm-start 生效
- ✅ LR 续接,无跳跃
- ✅ `val_composite_ckpt_score` log "verbose=True" 确认 metric 识别
- ✅ 6 active loss 数量级与 STEP2 末尾一致
- ✅ 第一个 val epoch 后 ckpt 落盘 + filename 含 score
- ✅ MAX_EPOCHS 已恢复 500

如任一不过 → SA 立即 stop,贴日志,Exp5'-MA 决议(可能要换 §2.3 选项 C,或回退选项 A 完全 from-scratch — 但概率低,strict=False 是 PL 原生功能,几乎必生效)。

---

## §4 Step K4 — 启动正式续训(30 分钟启动)

### 4.1 启动前清理

```bash
# 删 dry-run 期间产生的 ckpt(不混淆正式 ckpt)
cd /home/tcat/diffcsp_exp5_prime/checkpoints/
rm -f composite_epoch155_score*.ckpt last.ckpt
ls -la
# 期望剩下:
# - last.ckpt.from_step2_baseline (warm-start 起点)
# - last.ckpt.frozen_step2_final
# - epoch=004-gate=0.5305.ckpt
# - epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot
```

### 4.2 启动正式续训(tmux + nohup)

```bash
# tmux session
tmux new -s exp5p_continue

cd /home/tcat/diffcsp_exp5_prime/code/step4
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# 启动续训(到真 EarlyStop 或 MAX_EPOCHS=500 之一)
CUDA_VISIBLE_DEVICES=0 nohup /home/tcat/conda_envs/mlff/bin/python step4_2_train.py \
    > /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_$(date +%Y%m%d_%H%M).log 2>&1 &

TRAIN_PID=$!
echo "Continue training PID: $TRAIN_PID"
echo $TRAIN_PID > /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue.pid

# Detach: Ctrl+b d
```

### 4.3 启动 5 分钟 health check

```bash
ps -p $(cat /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue.pid) && echo "alive"
tail -50 /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_*.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
# 期望:GPU 0 mem 5-15 GB,util > 70%,epoch 编号从 155 起(不是 0)
```

### 4.4 PASS gate K4

- ✅ 进程存活
- ✅ epoch 编号从 155 起
- ✅ GPU 利用率 > 70%
- ✅ tmux detach 成功

---

## §5 Step K5 — Epoch 155-160 监控(1-2h)

### 5.1 等 epoch 160 完成(~ 16 分钟,5 epoch × 2.7 min)

```bash
tail -f /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_*.log | grep "Epoch 160:"
# 看到 epoch 160 完成,Ctrl+C
```

### 5.2 提关键趋势(epoch 155-160)

```bash
# composite 是否还在爬
grep -oE "val_composite_ckpt_score=[0-9.]+" \
    /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_*.log | \
    awk -F= '{print $2}' | head -20

# val_loss 是否还在降
grep -oE "val_loss=[0-9.]+" \
    /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_*.log | \
    awk -F= '{print $2}' | head -20

# val_min_d 是否还在升
grep -oE "val_min_d_mean=[0-9.]+" \
    /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue_*.log | \
    awk -F= '{print $2}' | head -20
```

### 5.3 epoch 160 ping Exp5'-MA(关键决策点)

SA 把 §5.2 三段输出贴给 Exp5'-MA。**Exp5'-MA 决议**:

| 情景 | 决议 |
|---|---|
| composite 持续爬升(每 epoch +0.0005~0.001)| ✅ 续训生效,继续到真 EarlyStop |
| composite 卡 0.576 不动(5 epoch 内)| 真 plateau,SA 主动终止训练,进 K6 hand-back |
| composite 反向(下降 > 0.005)| ⚠️ warm-start 副作用,Exp5'-MA 紧急决议(可能是 LR scheduler state 加载有问题,要 stop + 调试)|
| 6 active loss 任一 NaN/Inf | 立即 stop,ping |

### 5.4 PASS gate K5

- ✅ epoch 155-160 完整完成
- ✅ 6 active loss 全 finite
- ✅ §5.2 三段趋势 grep 输出齐
- ✅ Exp5'-MA 决议:继续 / 终止

---

## §6 Step K6 — 续训结束 hand-back

### 6.1 三种结束情景

| 情景 | 触发 | 行动 |
|---|---|---|
| 真 EarlyStop | composite 30 epoch 不升 | hand-back 给 Exp5'-MA |
| 跑满 500 epoch | 不太可能(LR 已低)| hand-back |
| K5 Exp5'-MA 决议终止 | composite 真 plateau | SA 主动 stop + hand-back |
| GPU/系统崩溃 | crash | 立即 ping,贴 last 100 行 log |

### 6.2 训练结束后立即操作

```bash
# 进程退出
ps -p $(cat /home/tcat/diffcsp_exp5_prime/logs/train_step2_continue.pid) || echo "exited"

# ckpt 列表(save_top_k=3 期望 3 个 composite_epoch*.ckpt + 1 个 last.ckpt)
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/

# 提 best 3 ckpt 的 epoch + score
ls /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch*.ckpt | \
    grep -oE "epoch[0-9]+_score[0-9.]+"

# 立即 frozen 备份所有新 ckpt(防误删)
cd /home/tcat/diffcsp_exp5_prime/checkpoints/
for ckpt in composite_epoch*.ckpt last.ckpt; do
    [ -f "$ckpt" ] && cp "$ckpt" "${ckpt}.frozen_step2_continue_final"
done
```

### 6.3 hand-back 必报

写 `EXP5_PRIME_STEP2_CONTINUE_HANDBACK.md` 落服务器根目录:

```
# §0 状态
- 续训完成 epoch X / 500
- 结束原因(真 EarlyStop / Exp5'-MA 决议终止 / 跑满 / 崩溃)
- 续训耗时 X h
- 启动 epoch 155 → 结束 epoch X(增加 Y epoch)

# §1 ckpt evidence
| ckpt | md5 | epoch | composite | gate | min_d |
|---|---|---|---|---|---|
| composite_epoch180_score0.582.ckpt | ... | 180 | 0.582 | ... | ... |
| (top-3) |
| last.ckpt | ... | X | ... | ... | ... |
| 永久档案 .frozen_step2_continue_final 列表 |

# §2 趋势对比表(STEP2 vs STEP2-CONTINUE 全程)
| 指标 | STEP2 epoch 154 | STEP2-CONTINUE 末尾 | Δ |
|---|---|---|---|
| val_composite | 0.576 | ? | ? |
| val_loss | 75.60 | ? | ? |
| val_min_d_mean | 1.590 | ? | ? |
| val_gate_pass_rate | 0.455 | ? | ? |

# §3 续训 verdict
| 指标 | 实测 | 阈值 | 状态 |
|---|---|---|---|
| val_composite | ? | ≥ 0.40 GREEN | ? |
| val_gate_pass_rate | ? | ≥ 0.80 GREEN / 0.60 AMBER | ? |
| val_min_d_mean | ? | ≥ 1.5 Å | ? |

# §4 训练 log 关键节点
- 续训第一 epoch (155) 6 active loss
- 中期 epoch (~ 200 / 250)
- 末尾 epoch
- LR 全程趋势(grep lr-Adam)

# §5 STEP3-SAMPLE 准备
- 推荐 ckpt path:best composite ckpt(具体)/ last.ckpt(具体)
- Exp5'-MA 决议路径

# §6 OPEN 问题
```

### 6.4 PASS gate K6

- ✅ 训练正常结束
- ✅ ckpt 列表 + frozen 备份
- ✅ hand-back 完整
- ✅ Exp5'-MA review → 启动 STEP3-SAMPLE(用 STEP3 launch note 已写好)

---

## §7 红线(SA-EXP5'-STEP2-CONTINUE 全程不动)

| 红线 | 说明 |
|---|---|
| ❌ 不动 holdout / 7 守卫包 / Exp4 backbone | |
| ❌ 不动 Exp5 v2 ckpt 永久档案 | |
| ❌ 不动 STEP1-FIX-C 11 文件代码 md5 | |
| ❌ 不动 cache .pt(L=20)| |
| ❌ 不动 shell_boundaries.pkl | |
| ❌ **不动 STEP2 末尾 4 个永久档案 ckpt**(last.frozen_step2_final + epoch=4 frozen + last.from_step2_baseline + epoch=4 active)| |
| ❌ 不动 MAX_EPOCHS=500 / LR T_max=500 / batch=64 / lr=1e-4 / fp32 | 已成事实 |
| ❌ 不擅自调三件套 cost(1.0 / 0.5 / 0.2)| |
| ❌ 不擅自 cost_density(0.2)| |
| ❌ **不 from-scratch 重训**(必 warm-start from last.ckpt.from_step2_baseline)| |
| ❌ 不擅自调 patience=30 / save_top_k=3 | Exp5'-MA 拍板 |
| ❌ 不擅自删 STEP2-CONTINUE 期间生成的 ckpt | |
| ❌ 任何不确定 → ping,不擅自 fix | |

---

## §8 Watch-only 项(SA 报告 Exp5'-MA 决议)

1. **LR 接续平滑性**:K3 dry-run 必报 LR 续接 < 5% 跳跃。如 K4 正式续训发现 LR 跳跃,贴日志
2. **continued epoch 编号**:必从 155 起(不是 0)。如发现从 0 起,warm-start 失败,立即 stop
3. **save_top_k=3 ckpt 命名**:filename 含 score(`composite_epoch180_score0.582.ckpt`),不写 best/failed
4. **磁盘趋势**:save_top_k=3 + last + 永久档案 ~ 7 个 ckpt × 44 MB ≈ 300 MB,可接受
5. **`loss_shell_count`**:STEP2 epoch 154 端值 398,续训如继续保持高位是 expected,降到 < 30 是惊喜不是要求

---

## §9 OPEN QUESTIONS(SA 不答,贴给 Exp5'-MA)

### Q1 — strict=False 是否真生效(仍报 RuntimeError)

K3 dry-run 第一个 val epoch 前若仍报 RuntimeError,说明 strict=False 不够。SA 立即 stop,Exp5'-MA 决议改用 §2.3 选项 C(LightningModule setup hook 预注册)。

### Q2 — warm-start 后 PreCollatedDataset state 是否一致

PreCollatedDataset 在 datamodule.setup() 时构建。warm-start 重新启动 trainer 会重新调用 setup(),内存预加载应一致。但**若发现 6 active loss 数量级与 STEP2 末尾不一致 > 10%**,可能是数据 shuffle 种子不同,SA 报告。

### Q3 — 真 EarlyStop 在 epoch X 触发时,是否还要继续

如真 EarlyStop 在 epoch ~ 180-200 触发,composite 改进 < 0.005,**ROI 临界**。SA 不决议,Exp5'-MA 决议是否接受这个改进进 STEP3,或 patience 加大重启。

---

## §10 你不做的事

- **STEP3 sample**(Exp5'-MA 已写好 STEP3-SAMPLE launch note,等续训完后启动)
- **修改 LR / batch / scheduler 任何超参**
- **修改三件套 cost**
- **改回 batch=16 / num_workers=0**
- **写新 errata**(若续训完后有新发现,Exp5'-MA 决议是否更新 errata 4)

---

## §11 工作哲学红线

1. 任何技术判断先列证据
2. 任何不确定 → 贴日志
3. K3 dry-run + K4 启动 + K5 epoch 160 三个 ping 点都强制
4. 70% 上下文闸门(SA 当前窗口已有 STEP2 + 续棒,接近闸门要主动 hand-back)
5. **续训中途 ping Exp5'-MA 是好事,不是失败**

---

*Exp5'-MA 撰写,2026-05-04,基于 STEP2 hand-back final + V1 grep 揭示 epoch 152-154 仍有改进 + errata 4 §2 callback bug 根因。SA-EXP5'-STEP2-TRAIN 当前窗口续棒做 STEP2-CONTINUE。续训完后 Exp5'-MA 启动 STEP3-SAMPLE(launch note 已落 outputs)。*
