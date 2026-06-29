# ✅ MA5 决议收到 — SA2' 执行最后一项 hand-back

3 件事处理:

1. **Milestone grep 无输出** → 走 (b) 路径,§5.3 写 lessons learned。你的 read 完全正确:`check_val_every_n_epoch=5` 配 `current_epoch == 200` 是 cosmetic timing miss(epoch 200 不是 val 周期,callback 在 `on_validation_epoch_end` 钩子上没机会触发)。**不阻塞,不返工**,记一笔给未来 ExpN。
2. **Q-SA3-1 (I)** 收到,§6.4 按你 phrasing 改。
3. **OUTPUT 写盘** 授权收到,heredoc 在下面。

---

## 📋 BLOCK H — 写盘 EXP5_SA2_PRIME_OUTPUT.md

operator 整段 paste:

```bash
cat > /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md << 'OUTPUT_EOF'
# EXP5_SA2_PRIME_OUTPUT.md
# Exp5 v2 SA2' Hand-back — Training Complete

> **From**: SA2' (Training sub-agent prime)
> **To**: MA5 (for review + SA3' launch decision)
> **Date**: 2026-04-29 21:21+12:00
> **Status**: Training complete, best ckpt landed, MA5 reviewed and decided (I) 接受 epoch 484 启动 SA3'

---

## §1 训练参数确认(yaml + 实际加载)

实际加载值(均匹配 SA1' yaml,defensive assertions launch 时全过):
- `cost_lattice = 0.0` (lattice frozen,不进 total)
- `cost_coord = 1.0`
- `cost_type = 1.0`
- `cost_density = 0.2`(v1 漏字段,v2 SA1' 显式落地)
- `latent_dim = 272` (= spectrum 256 + center_emb 16)
- `decoder.latent_dim = 528` (= time_dim 256 + spectrum 272)
- `mv_attention.num_heads = 4`
- `mv_attention.residual_alpha = 0.5`(float, 不是 nn.Parameter)
- `n_center_elements = 95`, `center_emb_dim = 16`
- 模型参数量: **3,650,388**
- precision: fp32 (MA4 D1)
- batch_size: 16
- LR: 1e-4, Cosine T_max=500, eta_min=1e-6
- max_epochs: 500, early_stop patience: 30, save_top_k: 1, check_val_every_n_epoch: 5

## §2 训练历程(从 PL log 抓)

| Epoch | val_loss | val_coord | val_type | val_density |
|---|---|---|---|---|
| 5 (first val) | 0.912 | 0.859 | 0.036 | 0.0854 |
| 8 (epoch-avg) | — | 0.852 | 0.0245 | 0.0854 |
| 394 (early best) | 0.7074 | — | — | — |
| 484 (final best) | 0.7065 | 0.684 | 0.00593 | 0.0854 |
| 500 (训练终点,自然结束) | (未优于 484) | — | — | — |

**关键观察**:
- val_type_loss 从 epoch 5 的 0.036 降到 epoch 484 的 0.00593(~6×),说明 MV-attention + center_emb 学到了显著结构信号
- val_density_loss 从 epoch 5 起即在 ~0.085 plateau,× cost_density=0.2 后微贡献 0.017 — 与 Exp4 同行为
- val_coord_loss 从 forward_test 锚 1.325 → epoch 484 的 0.684,稳定下降
- 最后 100 epoch 仍缓慢改进(394→484: 0.7074→0.7065,Δ=0.0009 / 90 epoch),未真正 plateau,但 LR 已到 eta_min

## §3 best ckpt 信息

```
路径:        /home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt
last ckpt:   /home/tcat/diffcsp_exp5/checkpoints/last.ckpt
size:        44 MB each
epoch:       484 (max_epochs=500,自然完结,non-early-stop)
val_loss:    0.7065
val_coord:   0.684
val_type:    0.00593
val_density: 0.0854
val_lattice: 1.360 (× cost_lattice=0,不计入)
param count: 3,650,388 ✓ (与 SA1' 期望一致)
path file:   /home/tcat/diffcsp_exp5/best_checkpoint_path.txt 已写入
```

## §4 wall time + GPU util

- 起: 2026-04-28 17:39+12:00
- 终: 2026-04-29 21:21+12:00
- 总 wall time: ~28h(略低于 LAUNCH_NOTE 32h 估)
- 平均 epoch 时长: ~3.4 min/epoch
- GPU util: launch 后 30s 检查 9-15%(GPU 0,4090),GPU 1 idle 全程
- 显存峰值: ~865 MiB / 24 GiB(大量 headroom)
- ⚠️ GPU util 偏低(non-blocking observation): 可能 batch_size=16 + DataLoader bottleneck,
  SA2' 不动(yaml 红线),记录给未来 ExpN tuning 参考

## §5 异常事件 log

### 5.1 Launch 阶段 — line 219 AttributeError + α' patch
- 首次 launch 在 datamodule.setup() 后 line 219 crash:
  `AttributeError: 'XasLocalDataModuleV2' object has no attribute 'train_dataset'`
- Root cause: SA1' fork Exp4 step4c_2_train.py 模板,带过来 V1 datamodule API (`.train_dataset`),
  但 v2 datamodule (`xas_local_datamodule_v2.py`) 实际命名是 `.train_ds`。命名 contract 不一致。
- DIAG E 定位 bug 范围:line 219-220 两行,`train_size`/`val_size` 仅被 line 221 logger.info 消费(DIAG F)
- MA5 决议 patch 路径 (γ): 授权 SA2' apply,scope 限 line 219-220 两行 rename(α' 选项)
- Patch 应用方式: Python in-place + occurrence count assertion(替代 sed,防 multi-match silent error)
- Patch 结果: 2 substitutions, 0 leftover, ast.parse OK, line 221 logger.info 在重 launch 顺利通过
- 备份保留: `step4_2_train.py.bak_pre_milestone`(milestone-only baseline) +
            `step4_2_train.py.bak_pre_alpha`(milestone+α' 前 baseline)
- Wall-time 影响: ~1h (诊断 + patch + verify + relaunch),不阻塞 32h 训练总时长

### 5.2 训练循环阶段 — 0 异常
- 无 NaN / Inf / OOM / 卡住事件
- stderr 仅 PL slurm warning,无 traceback
- 28h 训练全程稳定
- nohup 在 ssh 断开后正常存活到训练结束

### 5.3 Epoch 200 milestone marker — cosmetic miss(b 路径)
- `grep "EPOCH 200 MILESTONE" .../step4_train_v2_stdout.log` 无输出
- Root cause(MA5 + SA2' 共识):PL Callback `on_validation_epoch_end` 钩子 +
  `current_epoch == 200` 触发条件 + `check_val_every_n_epoch=5` 配置三者交互
  导致 epoch 200 不是 val 周期,Callback 没机会执行
- 影响:cosmetic 监控信号丢失,无功能影响 — 训练已完成,best ckpt 已落地
- LAUNCH_NOTE §5 milestone review 实质通过 MA5 周期性 ssh check 替代完成
- **Lessons learned 给未来 ExpN**:milestone Callback 应在
  `on_train_epoch_end`(每 epoch 都触发)上挂,触发条件用 `current_epoch == 200`,
  避免依赖 val 周期。或者改用 `on_validation_epoch_end` + 触发条件
  `current_epoch >= 200 and not self._fired`(一次性 latch)以容错 val 周期错位。

## §6 给 SA3' 的 carry-over

### 6.1 Best ckpt 路径
```
BEST_CKPT=/home/tcat/diffcsp_exp5/checkpoints/epoch=484-val_loss=0.7065.ckpt
```
也可读 `/home/tcat/diffcsp_exp5/best_checkpoint_path.txt`。

### 6.2 Sample 命令模板(SA1' OUTPUT §8.5 引)

```bash
PY=/home/tcat/conda_envs/mlff/bin/python
cd /home/tcat/diffcsp_exp5/code/step5
PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
$PY step5_1_sample.py --split val  --ckpt $BEST_CKPT
$PY step5_1_sample.py --split test --ckpt $BEST_CKPT
# DO NOT run on holdout — SA3' 红线 (LAUNCH_NOTE §7)

$PY step5_2_compute_metrics.py --split val
$PY step5_2_compute_metrics.py --split test
```

注:`step5_1_sample.py` 是否存在 SA2' 未 verify(OOS),SA3' 上线时 pre-flight 检查;
若不存在,SA3' 写。

### 6.3 Exp4 baseline 对照锚点(SA3' verdict 用)

| 指标 | Exp4 val | Exp4 test | v2 success threshold (Q3) |
|---|---|---|---|
| RMSD (Å) | 1.4849 | 1.4852 | < 1.5 (no Geometry regression) |
| pred_in_cutoff | 18.93/20 | 18.93/20 | > 18 |
| Set-Level TypeAcc | 0.3309 | 0.3330 | > 0.40 |
| **Multiset Macro-F1** | **0.0843** | **0.0846** | **🟢 > 0.20 (2.40×) / ⚠️ 0.10-0.20 / ❌ < 0.10** |
| Collapse Ratio | 0.0% | 0.0% | < 5% |

Exp4 baseline 文件位置:
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt`
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_val_per_sample.csv` (7,621 rows)
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_test_metrics.txt`
- `/home/tcat/diffcsp_exp5/logs/exp4_baseline_test_per_sample.csv` (4,481 rows)

### 6.4 Train-to-convergence 决议(MA5 拍板)

**MA5 决议: (I) 接受 epoch 484 ckpt 启动 SA3'**

理由(MA5 phrasing):
- 先看 SA3 verdict,如改进再续训 — 先低成本验证、再决定加投入的正确序列
- val_loss 第 N 位小数 ≠ Multiset F1,先 sample 看真信号比再压 val_loss 划算
- Cosine LR 已到 eta_min=1e-6,再训 100-200 epoch 是对噪声地板做优化
- LAUNCH_NOTE §5 milestone 三选一中的 (b) "提前接受当前 ckpt"精神支持 (I)

**续训作为 SA3 verdict 后的条件后续选项**(F1 在 0.10-0.20 amber 区时触发):
- 🟢 F1 > 0.20: 直接 SA4' figure,不续训(成功就成功了,再压无意义)
- ⚠️ F1 ∈ [0.10, 0.20]: 续训触发条件 — MV-attention 学到了东西但不充分,
  续 100-200 epoch 可能跨过 0.20 阈值
- ❌ F1 < 0.10: 不续训,转 Exp6 方向(MV-attention 这条路不通)

续训机制(条件触发时):PL 自带 `Trainer.fit(ckpt_path=last.ckpt)`,改 yaml
`max_epochs: 500 → 700` 即可,~0.5h 工程量。SA3 verdict 出后再考虑。

### 6.5 SA3' 红线(从 SA2' 沿用)
- ❌ 不动 holdout(SA2' 阶段全程未碰,SA3' 仅 sample val + test,**不**碰 holdout)
- ❌ 不改训练代码 / yaml(SA2' 阶段除 α' patch + milestone Callback 外 0 改动)
- ❌ 不 warm-start 任何东西(已无意义,训练已完成)
- ❌ 不删 / 不改 best ckpt 与 last ckpt(SA3' 只读)
- ❌ 不擅自决定续训(amber 区触发条件由 MA5 ratify 后再启动)

---

## §7 SA2' Hand-back 总结

SA2' 4 项任务全部完成:
1. ✅ Pre-flight 5/5 PASS(含 forward_test Phase 6.4 loss=2.6899 实数,回应 Q2)
2. ✅ Launch(经历 1 次 line 219 crash + α' patch 后,真 launch 进训练循环)
3. ✅ 头 30 min 守屏(epoch 0 启动 + 第一次 val=0.912 + GPU on + stderr clean)
4. ✅ Epoch 200 milestone protocol(marker cosmetic miss,实质 review 由 MA5 周期 check 替代完成)

红线全程未越:核心模型代码(`diffusion_w_type_xas.py`)未触;yaml 未触;
holdout 未触;Phase 6.5 fp32 sites 未触;warm-start 未尝试。

代码改动 surface(全部在 SA2' 授权范围内,均贴 diff 给 MA5 ack):
- `step4_2_train.py` Callback 块(milestone marker,MA5 (A)-modified 决议)
- `step4_2_train.py` line 219-220 rename(α' patch,MA5 (γ) 决议)

best ckpt 落地,MA5 决议 (I) 接受当前 ckpt,启动 SA3'。

— SA2' (sub-agent prime), 2026-04-29 21:21+12:00
OUTPUT_EOF

# Verify
echo "=== Write OK ==="
ls -la /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md
echo ""
echo "=== Line count ==="
wc -l /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md
echo ""
echo "=== Head + Tail sanity ==="
head -5 /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md
echo "..."
tail -5 /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md
```

---

## 🎬 SA2' 任务完结声明

执行完 BLOCK H 后,operator 把 `cat /home/tcat/diffcsp_exp5/EXP5_SA2_PRIME_OUTPUT.md` 完整输出贴回这里,**这是 SA2' 最后一次 deliverable**。

之后:
1. **MA5 review** SA2' OUTPUT(§5.1 patch 历史 + §3 ckpt 信息 + §6 carry-over 完整性)
2. **MA5 写 SA3' launch note**(~200 行,paste-ready),引用 SA2' OUTPUT
3. **你启动 SA3'**,跑 sample + metrics + 出 Multiset F1 verdict
4. SA3' 出数后 verdict 三分支(🟢 > 0.20 → SA4' figure / ⚠️ 0.10-0.20 → 续训 / ❌ < 0.10 → Exp6 转向)

预计 SA3' 总耗时 ~9h sample + 0.5h metrics + 0.5h 报告。

---

**SA2' summary,一句话**: 28h 训练 0 异常完成,1 个 contract bug 干净修掉,核心模型 0 触动,所有红线守住,best ckpt val_loss=0.7065 (val_type_loss=0.00593,vs Exp4 v1 SA1 ~200×),MA5 决议 (I) 启动 SA3'。

Operator 跑 BLOCK H + 贴回 OUTPUT 完整内容 → SA2' offline。