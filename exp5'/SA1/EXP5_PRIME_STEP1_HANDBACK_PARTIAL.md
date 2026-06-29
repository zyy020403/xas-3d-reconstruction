# EXP5_PRIME_STEP1_HANDBACK_PARTIAL.md
# SA-EXP5'-STEP1 → SA-EXP5'-STEP1-续 中期移交 launch note

> **撰写者**: SA-EXP5'-STEP1 (本人)
> **日期**: 2026-05-02
> **范围**: STEP1 §1.1-§1.4 完成 (dataset / datamodule / model / yaml),
>          §1.5-§1.7 (train.py / forward_test / smoke) 待续棒接
> **触发**: 上下文窗口 ~ 70%,launch note §15 红线 4 主动 hand-back
> **接棒人**: SA-EXP5'-STEP1-续

---

## §0 上下文 (10 秒读懂)

Exp5' 是 Exp5 v2 的 fork,fix v2 epoch 529 ckpt SA-METRICS-V3 dry-run 暴露的两个灾难:
- min_d gate pass rate 5-11% (95% 样本两两重合 < 1.5 Å)
- shell-1 distance score 0.0000

修复手段:从 shell_boundaries.pkl (387 MB,Exp4 Step 2.5 产出,md5 cf2050e4...) 抽 5 字段 inject 进 dataset,加 3 个新物理 loss 攻击两个灾难:
- `_pairwise_min_distance_penalty` (cost=1.0,主线攻 min_d gate)
- `_shell_distance_loss` (cost=0.5,主线攻 shell-1 distance score)
- `_shell_count_loss` (cost=0.2,辅助攻配位数)

3 个 loss 末尾全部 isfinite guard (launch note §0.4 #2 强制)。

整个 STEP1 是工程 surgery + smoke verify,**不启动正式 ~32-40h 训练**(那是 SA-EXP5'-train 的事)。

---

## §1 已完成 (§1.1-§1.4)

### §1.0 mkdir + cp -r + ln -s ✓ PASS
- `/home/tcat/diffcsp_exp5_prime/` 建好 (19 MB)
- `code/{step2,step3,step4,step5,step6}` 全部 cp 自 exp5/code
- `data` 软链接 → exp4/data (shell_boundaries.pkl md5 cf2050e4... 验证)
- `checkpoints/` `logs/` 空,from-scratch
- 8 项 startup verify 全过 (7 守卫包 / GPU 0+1 / 磁盘 65G avail / pkl schema 9 字段对齐)

### §1.1 + §1.2 dataset_v2 + datamodule_v2 改动 ✓ PASS
**改动文件** (均 `.bak_pre_exp5_prime` 锚点已建):
- `step3/xas_local_dataset_v2.py` (374 → 463 行)
- `step3/xas_local_datamodule_v2.py` (257 → 292 行)

**改动内容**:
- dataset 加 module-level helper `_extract_shell_fields(shell_meta)` 提取 5 字段
- dataset `__init__` 加 100-sample sanity check (≥ 95/100,launch note §0.4 #4 强制)
- dataset 两条 path (cache + slow) return dict 末尾 `**_extract_shell_fields(shell_meta)`
- datamodule `_dict_to_pyg_data` wrapper 加 5 个 `(1,)` graph-level tensor inject

**Smoke verify 跑过** (`step1_dm_smoke.log`):
- sanity 100/100 hits (train + val 双过)
- 5 字段 shape (4,) + dtype 全对 (float32×2 / bool / long×2)
- PyG Batch construction succeeded
- 物理 sanity bonus: shell2_d > shell1_d 在 4/4 样本成立
- Cache 99.99% / 99.96% hit
- Benchmark 全在阈值内

### §1.3 + §1.4 model + yaml 改动 ✓ PASS
**改动文件** (均 `.bak_pre_exp5_prime` 锚点已建):
- `step3/diffusion_w_type_xas.py` (411 → 721 行)
- `step3/conf_xas/model/diffusion_xas.yaml` (~80 → ~95 行)

**改动内容**:
- model 加 4 个 staticmethod (`_tweedie_x0_hat` / 3 个 loss / `_compute_min_d_metrics`)
- `__init__` 加 3 个 cost 字段 (默认 0.0,defensive 兼容 Exp4-era yaml)
- `forward()`:
  - Tweedie x0_hat 推导 pred_frac_coords (gradient-flowing,**不 detach**,docstring "Design note" 标注给 Exp5'-MA review)
  - 3 个新 loss + 2 个 monitor metric (`val_min_d_mean` / `val_gate_pass_rate`)
  - total_loss 7 项 (cost_lattice=0 + 4 老 + 3 新)
- `training_step` / `compute_stats` 加 5 字段 log
- yaml 加 3 行 cost (1.0 / 0.5 / 0.2,proposal §2.4 锁定)

**Sanity test 6 项全过** (Test 1 random / Test 2 collapsed / Test 3 NaN /
Test 4 no-shell2 / Test 5 healthy backward / Test 6 NaN backward),关键:
- Test 2 `loss_pairwise_min = 2.2457` 与理论 `relu(1.5)² ≈ 2.25` 误差 < 0.005
- Test 3 三个 loss 全 `value=0.0 finite=True requires_grad=True` (isfinite guard 验证)
- Test 5 `total.backward()` 成功 + `pred_frac.grad` finite + nonzero
- Test 6 NaN 输入 + backward `pred_frac.grad` finite + zero (NaN 不污染梯度)

---

## §2 ⚠️ isfinite guard 实现踩坑 — 续棒必读

我前两版 guard 实现都错,折腾 3 轮才对。**根因**:`_pairwise` 跟 `_shell_*` 在 NaN 输入下走的 forward 路径不同:
- `_pairwise`:NaN 通过 `relu` / `mean` 传播到 total_loss → 触发 if 分支
- `_shell_*`:NaN 在 `gaps > threshold` 短路成 False → boundaries 空 → 完全跳过累加 → total_loss 保持 init 0 (finite!) → **走 else 分支** → grad tether `0.0 * NaN.sum()` 把 0 污染回 NaN

**最终方案**(3 处 staticmethod 统一):
```python
loss = total_loss / max(n_active, 1)
if not torch.isfinite(loss):
    loss = pred_frac_coords.new_zeros(())
# Unconditional grad tether via sanitized input
sanitized = torch.nan_to_num(pred_frac_coords, nan=0.0, posinf=0.0, neginf=0.0)
loss = loss + 0.0 * sanitized.sum()
return loss
```

**给续棒的提醒**:任何后续 loss 函数(forward_test 或 SA-EXP5'-train 阶段加的)如果加 isfinite guard,**必须本地 unit test 覆盖 NaN 输入 + backward 全链路**,不能只测 forward 数值。我犯过的错你不要重犯。

---

## §3 待做 (§1.5-§1.7)

### §1.5 train.py PL 原生 ModelCheckpoint 切换 ⭐ 关键
**文件**:`step3/step4_2_train.py`(launch note 写的是 `code/step{2,3,4,5}` 但实际 train script 在 step3 还是别处需 verify;或在 `code/step4/`)。

**改动**(launch note §0.4 #1 拍板,§5):
1. **删除** 当前自定义 callback / 硬编码的 `last_ckpt = ...epoch=484-...`(SA2'' 时代遗留,launch note §10 红线第 7 条要删)
2. **加** PL 原生 `ModelCheckpoint`:
   ```python
   from pytorch_lightning.callbacks import ModelCheckpoint
   ckpt_cb = ModelCheckpoint(
       monitor='val_composite_ckpt_score',
       mode='max',
       save_top_k=1,
       save_last=True,
       filename='epoch={epoch:03d}-score={val_composite_ckpt_score:.4f}',
       dirpath='/home/tcat/diffcsp_exp5_prime/checkpoints/',
   )
   ```
3. **加** EarlyStopping 切换到 composite score (launch note §0.4 #1):
   ```python
   from pytorch_lightning.callbacks import EarlyStopping
   es_cb = EarlyStopping(monitor='val_composite_ckpt_score', mode='max', patience=...)
   ```
4. **加** composite score 计算到 LightningModule (改 `diffusion_w_type_xas.py`):
   - `on_validation_epoch_end()` 末尾计算 + log:
     ```python
     score = (0.2 * (1 - min(val_loss, 1))
              + 0.5 * val_gate_pass_rate
              + 0.3 * (1 - min(val_loss_pairwise_min, 1)))
     self.log('val_composite_ckpt_score', score, prog_bar=True)
     ```
   - launch note §3.3 公式来源是 proposal §3.3
5. **保留** MAX_EPOCHS=500 (`step4_2_train.py` line 83 写死,launch note §0.4 #5 + final_report v2 §5.2)

### §1.6 forward_test.py Phase 6.7 sanity ⭐ 关键
**文件**:`step3/forward_test.py` (22931 bytes,~ 500 行)

**Phase 6.7 改动** (launch note §6):
- 加 5 字段 batch verify (类似我在 datamodule smoke test 里写的,但 batch_size=2/4/8 多档测)
- 加 composite_score 计算 path 测试 (validation_epoch_end → log → ckpt_cb.on_validation_end)
- 加 **Phase 6.7.g 主动 NaN injection 测 isfinite guard** (launch note §0.4 #2 强制):
  - 直接灌 `frac_coords = torch.full(..., nan)` 进 forward,verify 3 个 loss 都 finite + grad-bearing,total_loss.backward() 后 grad finite。**这部分我已经在 SA-1 阶段独立 sanity 测过,但没集成进 forward_test.py**,你需要把 sanity test 内容融进 Phase 6.7.g
- 测试 ckpt 加载/保存 round-trip (避免 v2 时代 callback 改动留下的 schema 漂移)

### §1.7 step4_1_smoke_test.py 改 + 跑
**文件**:`step3/step4_1_smoke_test.py` 或 `code/step4/` (位置需 verify)

**改动** (launch note §7):
- 加 5 字段 batch verify (跟 forward_test 重叠但更短)
- 跑 1-2 个 epoch dry-run 看 loss curve 是否合理:
  - 7 项 loss 全 finite ✓
  - `val_gate_pass_rate` 不为 0 (健康 baseline)
  - `val_composite_ckpt_score` 计算出真实数字
  - ckpt 文件落盘 (验证 ModelCheckpoint 配置)
- **不跑正式训练**,smoke 完贴 log 给 Exp5'-MA review,然后 hand-back 到 SA-EXP5'-train

---

## §4 给续棒的踩坑预警 (除 §2 外)

1. **梯度流决议** (model docstring "Design note — pred_frac_coords source" 显式标注):我让 `pred_x` **不 detach**,3 个新 loss 有梯度推 decoder。如 Exp5'-MA review 后觉得应 detach,改 `_tweedie_x0_hat` 1 行即可。**SA 不擅自决定**,留 OPEN 给 MA。

2. **`_density_loss` vs `_shell_distance_loss` 方向冲突信号** (errata 2 §1):cost_density=0.2 沿用,launch note §10 红线明文锁,SA-EXP5'-STEP1-续 + SA-EXP5'-train 都**不动**。Exp5'-MA 监控期决议。

3. **shell_dist 真值/预测 inconsistency** (launch note §11 watch-only #2):真值用全邻居(~ 200+ atoms 在 cutoff 内)mean,预测用 N=20 truncated frac。**已知设计,proposal §2.2 接受**。Watch-only,不修。

4. **train.py 文件位置不确定**:我没看具体 path,launch note §5 说 `step4_2_train.py` line 83 写死 MAX_EPOCHS=500,但 step3/step4 实际位置需要续棒第一步先 `find /home/tcat/diffcsp_exp5_prime -name "step4_2_train*"` 确认。

5. **PL 2.5 validation hook API** (launch note §12 Q4):新 PL 用 `on_validation_epoch_end()` + `self.validation_step_outputs` list。我看了下当前 model 的 `validation_step` 直接 return loss,没用 outputs list。**续棒需要看 v2 现状**(可能要小 surgery 加 list,可能不需要),不擅自改,有疑问报 Exp5'-MA。

6. **composite score log 时机**:必须在 `on_validation_epoch_end` log,**不能在 `validation_step`**。前者是 epoch-level,ModelCheckpoint 才能拿到正确值;后者是 batch-level,会触发 PL 警告 + ckpt 拿到的是最后一个 batch 的值。

---

## §5 文件 md5 + 路径 (起点锚点)

```
/home/tcat/diffcsp_exp5_prime/code/step3/
  xas_local_dataset_v2.py            (改动版)
  xas_local_dataset_v2.py.bak_pre_exp5_prime
  xas_local_datamodule_v2.py         (改动版)
  xas_local_datamodule_v2.py.bak_pre_exp5_prime
  diffusion_w_type_xas.py            (改动版,721 行)
  diffusion_w_type_xas.py.bak_pre_exp5_prime
  conf_xas/model/diffusion_xas.yaml  (改动版)
  conf_xas/model/diffusion_xas.yaml.bak_pre_exp5_prime
  forward_test.py                    (UNCHANGED, 22931 bytes)
  
/home/tcat/diffcsp_exp5_prime/checkpoints/  (空,from-scratch)
/home/tcat/diffcsp_exp5_prime/logs/
  step1_dm_smoke.log                 (datamodule smoke 日志)
  
/home/tcat/diffcsp_exp5/checkpoints/  (永久档案,不动)
  sa2_baseline_epoch484_val0.7065.ckpt.frozen   md5=155a58c9b64c0ed2749597c69f3e6f86
  sa2pp_resume_epoch529_val0.7003.ckpt.frozen   md5=72ad4275153b86a65a1399e4ab357d85
```

---

## §6 启动顺序 (给续棒 SA-EXP5'-STEP1-续)

1. 读本 hand-back + launch note §5 + §6 + §7 (即 §1.5-§1.7 所属章节)
2. ssh 上服务器跑 verify:
   ```bash
   ls -la /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py*
   md5sum /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py  # 应是 f6a65ea0...
   find /home/tcat/diffcsp_exp5_prime -name "step4_2_train*" -o -name "step4_1_smoke*"
   ```
3. 按 §1.5 → §1.6 → §1.7 顺序推进
4. 每个 step PASS 后跟 Exp5'-MA 确认再进下一步
5. §1.7 smoke 跑完后写 SA-EXP5'-STEP1 final hand-back,移交 SA-EXP5'-train

---

## §7 我的诚实自评

**做得好**:
- 100/100 sanity 通过 + 5 字段 schema 对齐 + 物理 sanity bonus (shell2 > shell1)
- model 改动逻辑清晰,Tweedie 推导符合 `_density_loss` 同源
- yaml 简洁,3 行 cost 锁定
- Test 2 collapsed 数值 (2.2457 vs 理论 2.25) 验证 loss 函数数学正确

**做得不够好**:
- isfinite guard 改 3 轮才对,前两版没本地验证就发服务器跑。**根因**:对 PyTorch 自动微分 + IEEE 754 NaN 传播在不同 forward 路径下的交互不够熟。续棒做 §1.6 Phase 6.7.g 时如要扩展 guard,**先本地端到端跑** (NaN 输入 + backward + grad finite check) 再发。
- §1.0 PASS 那轮花了过多 token 在逐项核对,被用户提醒后才精简。续棒注意节奏。

**未完成**:
- §1.5-§1.7 全部留给续棒,~ 50% 工程量。我估算续棒大约需要 1 个完整窗口 (40-60k token) 完成,如再不够再 hand-back 一次到 SA-EXP5'-STEP1-续² 完成 §1.7 smoke。

---

*SA-EXP5'-STEP1, 2026-05-02 hand-back, 全部 STEP1 §1.1-§1.4 完成且 sanity 验证通过。*
