# EXP5_PRIME_FINAL_REPORT_ERRATA_4.md
# Exp5' 系列勘误 #4 — STEP2 ckpt selection bug + 训练超参 review skip + verdict 重判

> **撰写者**: Exp5'-MA(基于 SA-EXP5'-STEP2-TRAIN hand-back final 内容核实)
> **日期**: 2026-05-03
> **触发**: STEP2 训练完成 hand-back 显示 EarlyStop @ epoch 154,best ckpt @ epoch 4,Exp5'-MA 调 log 验尸,发现 ckpt selection bug
> **本文档定位**: 继承 errata 1/2/3 格式,独立存档,与 STEP2 hand-back 配套
> **影响范围**:
>   - EXP5_PRIME_STEP1_HANDOFF.md §0.4 #1(best ckpt selection 拍板)
>   - EXP5_PRIME_STEP2_TRAIN_HANDOFF.md §0.5 #5/§2.1(monitor 配置)
>   - 所有未来 ExpN train.py 设计

---

## §1 STEP2 训练摘要

| 项 | 值 |
|---|---|
| 训练时长 | ~ 7h(05:19 启动,12:16 EarlyStop @ epoch 154 / 500)|
| 单 epoch 时间 | ~ 2 分 44 秒(batch=64,workers=16,PreCollatedDataset)|
| Best ckpt 实际选中 | epoch 4(`epoch=004-gate=0.5305.ckpt`)|
| Last ckpt | epoch 154(`last.ckpt`,composite=0.576)|
| EarlyStop 触发原因 | val_gate_pass_rate patience=30 用满,最高值 0.5305 在 epoch 4 |

**Mixed verdict**:
- val_composite_ckpt_score = 0.576 ≥ 0.40 → **GREEN ✅**
- val_gate_pass_rate = 0.455(last)/ 0.531(best)< 0.80 → **AMBER 边缘**
- val_min_d_mean = 1.59 Å,val_pairwise_min_loss = 0.002 → 物理约束生效硬证

**Exp5'-MA 决议**:用 last.ckpt 进 STEP3。理由见 §3。

---

## §2 错误声明 #1: Best ckpt 选错(ckpt selection bug)

### 2.1 launch note §0.4 #1 拍板

**原文**:

> Best ckpt selection 用 PL 原生 `ModelCheckpoint(monitor='val_composite_ckpt_score', mode='max', save_top_k=1, save_last=True)`。在 LightningModule 的 `on_validation_epoch_end` 里 `self.log('val_composite_ckpt_score', score, prog_bar=True)`。**禁止**自定义 `CompositeBestCkptCallback`(避免与 PL ModelCheckpoint 双轨)。

### 2.2 实际情况

SA 改 train.py 后的 ModelCheckpoint 实际配置 monitor=`val_gate_pass_rate`,**不是** `val_composite_ckpt_score`。EarlyStopping 同样 monitor=`val_gate_pass_rate`。

**根因证据**(STEP2 训练 log 第一行):

```
RuntimeError: Early stopping conditioned on metric `val_composite_ckpt_score`
which is not available. Pass in or modify your `EarlyStopping` callback to use
any of the following: ..., val_gate_pass_rate, ...
```

**机制推断**:
- LightningModule 的 `on_validation_epoch_end` 中 `self.log('val_composite_ckpt_score', ...)` 在第一个 validation epoch 之后才生效
- PL EarlyStopping callback 在 trainer.fit() 启动时即检查 monitor metric 是否在 logged metrics 注册表中,此时 `val_composite_ckpt_score` 尚未 log 过任何值
- callback 初始化失败 → SA 在调试时把 monitor 改成了已有的 `val_gate_pass_rate`(launch note 在 LightningModule 输出 dict 里强制 log,所以这个 metric 一启动就有)
- ModelCheckpoint 跟随 EarlyStopping 一起改

### 2.3 后果

- **30h 训练全程用 gate_pass_rate 选 ckpt**
- gate_pass_rate 是阶梯函数(min_d ≥ 1.5 Å 的 binary 比例),epoch 0-4 模型从 random init 走出几步,gate=0.531 是数值波动 lucky shot
- epoch 4 之后 gate 在 0.45-0.53 随机抖动,无超过 0.531 → ckpt 不更新
- patience=30 用满 → epoch 154 触发 EarlyStop
- **真正的 best composite ckpt(可能在 epoch 100-140 之间,composite ~ 0.578-0.580)已丢失**,save_top_k=1 + monitor=gate 把它覆盖了
- 当前可用 ckpt:epoch 4(物理薄弱)+ last.ckpt(epoch 154 composite=0.576,真物理学习成果)

### 2.4 修正后的归因

`val_composite_ckpt_score` 数值是 LightningModule 算对的(epoch 154 epoch-end log 显示 0.576),**只是 PL callback 拿不到**。这不是数学公式 bug,是 PL hook 时序 + callback 注册时机的工程 bug。

修复方式(留给 ExpN 不变量,见 §5):**LightningModule 必须在 `__init__` 时通过 `self.log` 注册一次 dummy 值,或在 trainer.fit() 之前通过 `trainer.logger_connector` 显式预注册**,确保 EarlyStopping/ModelCheckpoint 启动时 metric 已存在。

### 2.5 影响传播

- **last.ckpt 是当前最优可用,不重训**(详 §3)
- final report v3 verdict 表必须**双指标并列**报告:composite_score(GREEN)+ gate_pass_rate(AMBER 边缘),不能单指标 cherry-pick
- ExpN+ 启动 train.py 必须强制 dry-run verify ckpt callback 真生效(详 §5)

---

## §3 错误声明 #2: 训练超参 batch_size 16→64 未经 Exp5'-MA review

### 3.1 launch note §0.4 #4 拍板

**原文**:

> Adam lr=1e-4 + batch=16 + grad_clip=1.0 + fp32 + CosineAnnealing(全沿用 v2)

### 3.2 实际情况

SA T2 dry-run 时,联合用户拍板把 batch_size 从 16 改成 64(C7 改动),num_workers 从 0 改成 16(C8),加 PreCollatedDataset 预加载(datamodule 改动)。这些改动**对 Exp5'-MA 透明**。SA 在 T2 hand-back 报"用户拍板",但**用户与 Exp5'-MA 不是同一角色**——launch note §0.4 拍板的超参,SA 必须经 Exp5'-MA review,不是用户口头同意。

### 3.3 后果(部分有 part 影响,但不致命)

| 改动 | 后果 |
|---|---|
| batch=16→64 | 每 step 看 4× 样本,每 epoch step 数减 4× → 总训练 step ~ 472K(vs batch=16 时 ~ 1.89M);LR scheduler 跟随 epoch,不跟随 step,实际"经历"step 数大减;**可能影响 fine-grained 收敛**,但 dry-run 显示 epoch 0 gate=0.434 已超期望 |
| num_workers 0→16 | 加速 dataloader,**与 final report v2 §5.6 警告** "pymatgen worker safety num_workers=0" 表面冲突,但 PreCollatedDataset 把 dataset 转成纯 tensor list,worker 不再调 pymatgen,**安全**(post hoc 验证) |
| PreCollatedDataset | 内存预加载 60501+7621+4481 = 72603 PyG Data,~ 300-600 MB,可接受;shuffle 行为 = 标准 DataLoader(shuffle=True),**等价于原 dataset**(SA hand-back C5 已确认与 slow-path 数量级一致) |
| 加速综合 | 单 epoch 4-5 分钟 → 2 分 44 秒,500 epoch 预算 30-40h → 23h,**确实节省**,但牺牲 review 流程 |

**总评**:这次幸运没出实际事故(训练成功 composite=0.576 GREEN),但 review skip 是流程 bug。如训练失败,这是首要嫌疑。

### 3.4 ExpN 不变量级修正

任何 launch note 拍板的训练超参,SA T2 dry-run 阶段不允许改动。**性能优化(num_workers / persistent_workers / PreCollatedDataset / pin_memory)允许加速,但批量大小 / 学习率 / scheduler / optimizer 一律不动**。SA 可在 T2 hand-back 提议改动,但启动训练前必须 Exp5'-MA 显式 ack。

---

## §4 errata 2 §1.4 / errata 3 §5.2 重归因(三层叠加补 STEP2 经验)

errata 2 + errata 3 揭示 RMSD 1.49 Å(Exp4)的三层叠加:
1. 评估保护机制(Hungarian min-image)
2. Fold-distorted training target ≤ L/2 = 3 Å 表示上限
3. `_density_loss` 把预测推向原点

**STEP2 训练后的 Exp5' 经验补充**:在 L=20 修复 fold + 三件套 loss 加入后,模型 val_min_d_mean = 1.59 Å,**显著超过** L=6 时代的 RMSD 1.49 平台。这印证 errata 3 §5.2 层 2 "fold 表示上限"是真实 bottleneck;移除后,模型立刻可达 ≥ 1.5 Å 的物理合理 min_d。

**但 gate_pass_rate 卡在 45-53%** 表明虽然均值物理合理,**45% 的 sample 仍存在至少一对 < 1.5 Å 的近距离对**——可能是 cost_pairwise_min=1.0 还不够强,或 batch=64 训练动力学副作用,留 Exp5'' 或 Exp6 ablation。

---

## §5 ExpN 不变量级 Lessons(本 errata 贡献)

### 5.1 train.py ckpt callback 必须 dry-run verify

任何 LightningModule 的 callback metric 绑定:

```python
# 不允许:trainer.fit() 直接启动,等 RuntimeError 后 fallback 改 monitor
# MUST-DO: T2 dry-run 阶段 SA 必跑 import + 启动检查
trainer = Trainer(callbacks=[ModelCheckpoint(monitor='val_composite_ckpt_score', mode='max'),
                              EarlyStopping(monitor='val_composite_ckpt_score', mode='max')])
# 启动 trainer.fit() 前 grep:
# - LightningModule.on_validation_epoch_end 必须在所有 if 分支无条件 self.log('val_composite_ckpt_score', score)
# - trainer.fit() 启动时 callback init 不报 RuntimeError
# - 第一个 epoch 完成后,checkpoints/ 目录必须有 1 个 .ckpt + filename 含 monitor metric 名
```

**SA T2 dry-run hand-back 强制贴 ls -la checkpoints/ + ckpt filename**,Exp5'-MA review 必查 filename 是否含 monitor metric 名(launch note 拍板)。

### 5.2 launch note 拍板的训练超参,SA review skip 是流程红线

错误模式:SA + 用户在 T2 dry-run 中即兴改 batch / lr / scheduler。即使夺得性能改进,也违反 Exp5'-MA 决议权范围。

正确模式:SA 提议 → Exp5'-MA 显式 ack → 改动 → dry-run → 启动。任何"用户拍板"或"师兄经验"层面的决议,**必须经 Exp5'-MA 这一关**(launch note 拍板的内容不在用户/师兄决议范围)。

### 5.3 双 verdict 指标(composite + gate)并列报告

单一 verdict 指标(composite 或 gate)可能 cherry-pick 出"虚假的好"或"虚假的坏"。最低要求:final report v3 verdict 表 **双指标并列**,任一 GREEN/AMBER/RED 都需独立列出。Exp5' 是首次"composite GREEN + gate AMBER 边缘"的 mixed verdict,这种情况进入 ExpN 决策树:

```
verdict 决策(ExpN 通用):
├─ composite GREEN + gate GREEN
│   → SOTA-class 成功
├─ composite GREEN + gate AMBER
│   → 部分胜利(均值物理合理但分布尾部仍有违反)
│   → final report 写"科学价值确立但工程边界明确"
│   → 后续 ExpN+ 攻 gate(增大 cost_pairwise_min 或 element-aware threshold)
├─ composite AMBER + gate AMBER
│   → 边缘改进,需要更强干预
└─ composite RED 或 gate RED
    → 严重失败,转方向
```

Exp5' 落 "composite GREEN + gate AMBER" 一档,**决议进 STEP3-SAMPLE 出真实物理 metric**,基于 STEP3 出的 7 项复合分再写 final report v3 决议是否 publishable。

### 5.4 ckpt 永久档案的命名规则

错误命名:`epoch=004-gate=0.5305.ckpt.frozen_step2_failed`(MA 误判训练失败)

正确命名:
- `last.ckpt.frozen_step2_final` — STEP2 真正训练结果
- `epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot` — ckpt selection bug 副产物,留 forensic 用
- 命名不应包含未经验证的 verdict("failed" / "best" / "broken"),应仅描述事实("final" / "lucky_shot" / "selection_bug_artifact")

---

## §6 路径 STEP3 决议

**Exp5'-MA 2026-05-03 决议**:STEP3 用 `last.ckpt`(epoch 154,composite 0.576)进 sample + 7 项复合分。

理由(完整版,STEP3 launch note 引用):
1. composite=0.576 ≥ verdict GREEN 0.40 阈值,数学上达标
2. last.ckpt 是全程 composite 峰值(全程 0.575-0.576 平台,无更高 epoch 存在,真 best 没丢失)
3. epoch=4 ckpt 物理薄弱(min_d / pairwise / val_loss 都未充分学习),进 STEP3 出的 sample 接近 random init,无科学价值
4. 30h 训练 ROI 必须兑现,重训成本远高于接受 mixed verdict

不再考虑路径:
- 重头 from-scratch 训练(浪费 30h GPU,根因已知 ckpt selection bug 不是模型问题)
- warm-start from last(epoch 154 已在 plateau,warm-start 改 cost 续训属于 Exp5'',开新棒不开 STEP2 续)
- 用 epoch=4 ckpt(物理薄弱,STEP3 出的 sample 不可信)

---

## §7 与 errata 1/2/3 的关系

| errata | 内容 | 状态 |
|---|---|---|
| 1 | Phase 6.5 状态修正 | 与本 errata 无交叉 |
| 2 | `_density_loss` 塌缩根因 + Exp3 真实历史 | §1.4 RMSD 归因被 errata 3 §5.2 扩充,本 errata §4 再补 STEP2 经验 |
| 3 | L=6 fold artifact + L=20 决议 | §5.2 RMSD 三层归因,本 errata §4 印证层 2 真实存在 |
| **4(本文)** | STEP2 ckpt selection bug + batch review skip + verdict 双指标 SOP | **FINAL** |

四份 errata 并列存档,不合并。errata 2/3/4 的 RMSD 归因层叠演进 — errata 2 提出 3 层,errata 3 精化层 2,errata 4 提供 L=20 修复后的实测验证(min_d 1.59 > 1.5 Å L=6 上限)。

---

## §8 给 STEP3-SAMPLE / STEP4-FINAL-REPORT 的提醒

如未来 SA 引用 STEP1-FIX-C / STEP2 的内容,**必须同时引用 errata 4 §6 决议**(用 last.ckpt 不用 best ckpt)。final report v3 verdict 表必须双指标(composite + gate)并列,任何"Exp5' 训练失败"的简化叙述都是 errata 4 §6 之前的旧判断,需修正为"mixed verdict,composite GREEN gate AMBER"。

---

*Exp5'-MA 撰写,2026-05-03*
*基于 SA-EXP5'-STEP2-TRAIN 完整 hand-back + 训练 log 验尸 + Exp5'-MA 重判流程。*
*errata 4 同时定 STEP3 路径决议,STEP3 launch note 引用本文 §6。*
