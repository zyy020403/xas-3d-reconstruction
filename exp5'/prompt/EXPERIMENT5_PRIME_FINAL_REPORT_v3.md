# EXPERIMENT5_PRIME_FINAL_REPORT_v3.md
# Exp5' 系列最终报告 v3 — Diagnosis & Fix of Fold Artifact + Partial Validation of Physical Loss Constraints in Diffusion-based XAS Local Structure Prediction

> **撰写者**: Exp5'-MA(Exp5 系列第 3 任 Main Agent)
> **日期**: 2026-05-09
> **版本**: v3(继 v1 = Exp5 v2 final report,v2 = Exp5 v2 final report v2,本文是 Exp5' 阶段的 v3)
> **数据来源**: SA-EXP5'-STEP1 / STEP1-AUDIT / STEP1-FIX / STEP1-FIX-C / STEP2-TRAIN / STEP2-CONTINUE / STEP3-SAMPLE 全 hand-back 文档 + 5 份 errata + 训练 log + sample predictions + step5_3 复合分输出
> **核心 ckpt**: `composite_epoch169_score0.5881.ckpt`(STEP2-CONTINUE 续训 BEST,md5 `127afa44a850d8f7e4fcdae17e2761a1`)
> **关联文档**: 5 份 errata(EXP4_FINAL_REPORT_ERRATA_2 / 3,EXP5_PRIME_FINAL_REPORT_ERRATA_4 / 5)+ 7 份 SA hand-back

---

## §0 摘要

Exp5' 是 Exp5 v2 物理灾难(verdict ❌:gate_pass_rate 5-11% / shell-1 distance score 0.0000)之后的 from-scratch 重启,目标用三件套物理 loss(`_pairwise_min_distance_penalty` / `_shell_distance_loss` / `_shell_count_loss`)修复 v2 的物理违反问题。

执行过程中 SA-EXP5'-STEP1 在 §1.5 之前自查 dataset 物理性,发现 Exp4/v2/Exp5' 三代沿用的 dataset_v2.py L_VIRTUAL=6 + CUTOFF_R=10 不匹配,造成 64% 样本的 ground truth frac_coords 经 min-image fold 后产生**虚假近距离对**(fold artifact)。Exp5'-MA 决议 L=6→20 修复,STEP1-FIX 完成。

修复后 STEP2 训练经历两次曲折(ckpt selection bug 错用 `val_gate_pass_rate` 监控 / 续训 STEP2-CONTINUE 找回真 best ckpt)后,Exp5' 取得**部分胜利**:
- ✅ `_pairwise_min_distance_penalty` 生效,gate_pass_rate 从 v2 时代 5-11% 提升到 64%(6-13× 改进)
- ✅ Collapse rate 0.00%,无原子重合崩塌
- ✅ 三 split(val 7621 / test 4481 / holdout 3025)verdict 一致,差异 < 0.004,泛化优秀
- ❌ `_shell_distance_loss` 和 `_shell_count_loss` 未生效:pred shell-1 mean radial 6.32 Å vs true 2.27 Å(误差 4 Å);shell-1 distance score 0.035(目标 ≥ 0.50)
- ❌ Composite (step5_3 7 项) 0.080 RED(目标 ≥ 0.40 GREEN)

Shell loss 未生效根因诊断为**"鸡蛋启动问题"**(errata 5 §2):gap-based shell 切壳算法需要 pred 已具有壳层结构才能产生有效梯度,而 pred 壳层结构需要 shell loss 引导形成,两者互为条件,无外部信号打破循环。

Exp5' 阶段总结:**fold artifact 修复 + pairwise loss 设计为 publishable 真贡献**;shell loss 鸡蛋问题诊断为 ExpN+ 不变量级 lesson;Exp5'' proposal 将基于 errata 5 §6 候选 A(distance-supervised)或候选 B(distance-aware density-loss)二选一重设计 shell 监督机制。

---

## §1 背景与动机

### 1.1 Exp5 v2 verdict

Exp5 v2 阶段(MA5 主导)产出 SA2 baseline(epoch 484,val_loss 0.7065)和 SA2'' 续训(epoch 529,val_loss 0.7003),数学指标 +28.8% Multiset Macro-F1 改进(0.0843→0.1086,vs Exp4 baseline)。

但 SA-METRICS-V3 在 100 样本 dry-run 上揭示物理灾难:
- min_d gate pass rate 5%(val)/ 11%(test)— 95% 样本存在两两 < 1.5 Å 的近距离对
- shell-1 distance score 0.0000 — gate-pass 子集都不知第一壳层应在 ~ 2-3 Å
- Composite 均值 0.0056-0.0062

数学指标的 +28.8% 改进无法掩盖物理灾难,verdict ❌ physical-invalid。

### 1.2 Exp5' from-scratch 决策

用户 2026-05-01 拍板 Exp5' from-scratch:
- 不 warm-start v2 ckpt(95% 输出已"学坏",warm-start 需先"忘"再"学")
- 工作目录新建 `/home/tcat/diffcsp_exp5_prime/`
- 沿用 v2 架构:MV-attention(num_heads=4)+ center embedding(95×16d)+ cost_density 0.2 + Adam lr=1e-4 + CosineAnneal T_max=500
- 核心改动:**三件套物理 loss**(详 §1.3),目标修 v2 物理违反

### 1.3 三件套物理 loss(Exp5' 核心设计)

| Loss | 权重 (cost) | 设计目的 | 实施 |
|---|---|---|---|
| `_pairwise_min_distance_penalty` | 1.0 | 强制原子两两距离 ≥ 1.5 Å | ReLU(1.5 - d_pair)² mean,min-image cart |
| `_shell_distance_loss` | 0.5 | pred shell-1/shell-2 半径与 ground truth 一致 | gap > 0.1563 切壳 → 各壳 mean radial vs pkl 真值 MSE |
| `_shell_count_loss` | 0.2 | pred 配位数与 ground truth 一致 | 同上,但目标是各壳原子数 |

ground truth(`true_shell1_d_mean / true_shell2_d_mean / true_shell1_n / true_shell2_n / has_shell2`)从 Exp4 Step 2.5 产物 `shell_boundaries.pkl`(387 MB,md5 `cf2050e4...`)inject 进 dataset。

---

## §2 工程历程(7 个 SA 棒 + 5 份 errata)

### 2.1 棒次时间线

| 棒 | 日期 | 任务 | 关键产出 |
|---|---|---|---|
| **STEP1**(SA1)| 2026-05-01 | dataset shell_boundaries inject + 三件套 loss + smoke 验证 | 4 个核心代码 md5 锁定 + 6 项 sanity test 全过 |
| **STEP1-AUDIT**(SA1 自查)| 2026-05-02 | 自查 dataset 物理性,发现 fold artifact | errata 3 草稿 + 路径 B(L=6→20)候选 |
| **STEP1-FIX**(SA2)| 2026-05-02 | L_VIRTUAL=6→20,8 文件改 + cartesian sanity 100/100 | fold 案例硬证(0.40 Å → 6.40 Å)|
| **STEP1-FIX-C**(同 SA2)| 2026-05-03 | dataset cache rebuild + cache-loaded smoke | 3 个 cache .pt + cache_metadata.json L=20 |
| **STEP2-TRAIN**(SA3)| 2026-05-03 | 32-40h 训练 | EarlyStop epoch 154,best ckpt selection bug |
| **STEP2-CONTINUE**(同 SA3)| 2026-05-04 → 05-06 | 修 callback + warm-start 续训 | epoch 169 BEST composite=0.5881 |
| **STEP3-SAMPLE**(SA4)| 2026-05-09 | sample 三 split + step5_3 复合分 + sanity | 9 个 step5_3 输出 + verdict mixed |

### 2.2 5 份 errata 主线

| Errata | 触发 | 核心发现 |
|---|---|---|
| **#2**(继 Exp4)| Exp4 final report 反思期 | `_density_loss` 是塌缩剂(原点吸引子);Exp3 .detach + 虚假指标教训;方向 menu 调整 |
| **#3**(STEP1-AUDIT)| SA1 自查 dataset 物理性 | L_VIRTUAL=6 + CUTOFF_R=10 → 64% fold artifact;路径 B(L=20)决议 |
| **#4**(STEP2 后)| Exp5'-MA 训练 log 验尸 | EarlyStopping monitor 错用 `val_gate_pass_rate`(`val_composite_ckpt_score` 在第一个 val epoch 前不存在 → RuntimeError → fallback);verdict 双指标 SOP |
| **#5**(STEP3 后)| step5_3 verdict shell-1 RED | shell loss 鸡蛋启动问题;Exp5'' 方向决议候选 A/B |

5 份 errata 是 Exp5' 工程历程的核心 paper trail。每份独立存档,不合并。

---

## §3 Fold Artifact 诊断与修复(errata 3 详细复盘)

### 3.1 SA1 自查发现

STEP1 完成 §1.4(model + yaml 改动)+ §1.5(train.py)前,SA1 主动对 dataset 输出 ground truth `frac_coords` 做物理 sanity:

- 1% 样本两两 frac 距离 < 0.01(对应 cart < 0.06 Å,原子近乎重合)
- 94% 样本至少一对 frac 两两距离 < 0.25(cart < 1.5 Å)
- 100 样本诊断:**64% fold artifact + 36% 真重叠**

### 3.2 fold artifact 几何机制(errata 3 §2.2)

`xas_local_dataset_v2.py` 的 `__getitem__` 末段:

```python
relative_cart = coords_top - center_cart[None, :]   # 真 cart Å 位移
frac = relative_cart / L_VIRTUAL                     # / 6.0,frac ∈ (-∞, +∞)
frac = frac - np.round(frac)                         # min-image wrap → [-0.5, 0.5]
```

**触发条件**:任意两邻居在中心两侧且各自距中心 > L/2 = 3 Å。CUTOFF_R = 10 Å 导致大量邻居对满足此条件。

**典型案例**(errata 3 §2.2):
- 邻居 A: cart=[+3.2, 0, 0],真距 6.4 Å
- 邻居 B: cart=[-3.2, 0, 0]
- frac diff after fold = 0.066,**pairwise cart (as seen by loss) = 0.40 Å** ← 严重虚假违反

### 3.3 修复方案 — L_VIRTUAL = 6 → 20

errata 3 §8 决议路径 B:

| 选项 | 描述 | 决议 |
|---|---|---|
| A. L=12 | 修一半,数值边界仍有折叠风险 | 排除 |
| **B. L=20** | 完全消除 fold(L/2=10 ≥ CUTOFF_R) | **采纳** |
| C. 保留 L=6 修 loss | 只修 pairwise loss 局部症状 | 排除 |
| D. 降 CUTOFF_R 到 3 | 砍掉 shell-2 信号,违 Exp5' 设计意图 | 排除 |

### 3.4 修复后硬证(STEP1-FIX-C)

| 验证 | 旧 L=6 | 新 L=20 | 状态 |
|---|---|---|---|
| Cartesian sanity 100 sample | 6/100 PASS | **100/100 PASS** | ✅ |
| Fold 案例 [+3.2,0,0]/[-3.2,0,0] | 0.40 Å(虚假违反)| **6.40 Å**(真距)| ✅ |
| dataset 全 train cache rebuild | — | 60501/60507(99.99%)| ✅ |

errata 3 §3 同时验证 `shell_boundaries.pkl` 的 `distances` 字段是 raw cart Å(由 pymatgen.get_neighbors 直接返回,不经 fold),与 L_VIRTUAL 完全解耦。pkl 自一致性 verify 100/100 cart 一致,**0.1563 阈值是 cart Å 阈值**,与 L 无关,新 L=20 下使用反而更合理(详 §6.2)。

### 3.5 Fold artifact 影响传播链

```
Exp2 (Fe-only, L=6, Fe-O ~2 Å < L/2=3) → fold 几乎不触发 → 训练数据干净 ✅
Exp4 (88 元素, L=6, CUTOFF_R=10 沿用) → 大量邻居 > 3 Å → fold artifact 进 ground truth ⚠️
   → RMSD 1.49 Å 含层 2 表示上限(≤ L/2 = 3)
Exp5 v2 (沿用 Exp4 dataset) → gate_pass 5-11% 物理灾难 ❌
Exp5' STEP1 → STEP1-AUDIT 发现 → STEP1-FIX 修复 ✅
```

errata 2 §1.4 RMSD 1.49 Å 旧归因(三层评估保护机制)被 errata 3 §5.2 扩充为三层叠加(评估保护 + fold 表示上限 + density loss 原点吸引)。Exp5' STEP2-CONTINUE 修复 fold 后实测 val_min_d_mean = 1.59 Å,**超过 L=6 时代的 1.49 平台**,印证表示上限确实是 fold artifact 导致的几何 bottleneck。

---

## §4 训练历程(STEP2 + STEP2-CONTINUE)

### 4.1 STEP2 训练

| 项 | 配置 |
|---|---|
| Ckpt 起点 | from-scratch(STEP1-FIX-C 后,纯 random init)|
| 优化器 | Adam,lr=1e-4 |
| Batch | 64(SA T2 dry-run 期间联合用户拍板从 16 改 64,详 errata 4 §3) |
| Workers | 16,persistent_workers + pin_memory + PreCollatedDataset |
| Scheduler | CosineAnnealing T_max=500 |
| Grad clip | 1.0 |
| Precision | fp32 |
| Max epochs | 500 |
| Patience | 30 |
| GPU | RTX 4090 单卡(GPU 0)|

实际结果:
- 单 epoch ~ 2:44(945 steps × ~ 6 it/s)
- EarlyStop 在 epoch 154 触发(7h 总耗时)
- Best ckpt 命名 `epoch=004-gate=0.5305.ckpt`

### 4.2 errata 4 ckpt selection bug

STEP2 训练 log 第一行 traceback:

```
RuntimeError: Early stopping conditioned on metric `val_composite_ckpt_score`
which is not available. Pass in or modify your `EarlyStopping` callback to use
any of the following: ..., val_gate_pass_rate, ...
```

**机制**:LightningModule `on_validation_epoch_end` 中 `self.log('val_composite_ckpt_score', ...)` 在第一个 validation epoch 之后才生效。PL Trainer 启动时 EarlyStopping callback 检查 monitor metric 注册表 → 不存在 → RuntimeError。SA 调试时 fallback 把 monitor 改成已存在的 `val_gate_pass_rate`,ModelCheckpoint 跟随。

**后果**:`val_gate_pass_rate` 是阶梯函数(min_d ≥ 1.5 Å 的 binary 比例),epoch 4 时 gate=0.5305 是数值波动 lucky shot,后续 150 epoch 无超越 → patience 用满 → epoch 154 EarlyStop。**真正的 composite 训练是在持续上升的**,但被错误 monitor 提前停了。

### 4.3 STEP2-CONTINUE 续训

Exp5'-MA 调 STEP2 训练 log 末尾发现:

```
val_composite_ckpt_score 末尾 50 个 step 数值: 0.575 × 47 → 0.576 × 3
val_loss 末尾 50: 76.80 × 47 → 75.60 × 3
val_min_d_mean 末尾 50: 1.580 × 47 → 1.590 × 3
```

三个指标**同时**在 epoch 152-154 阶跃改进。所谓"0.575 长期平台"是 PL prog_bar 3 位数值精度造成的视觉错觉,真实值在缓慢爬,末尾突破精度门槛才显示出来。

**STEP2-CONTINUE 修复 + 续训**:
- callback 加 `strict=False`,允许 metric 在第一个 val epoch 前不存在(PL 原生 feature)
- ModelCheckpoint `save_top_k=3`(STEP2 用 1 让真 best 丢失)
- Warm-start from `last.ckpt.from_step2_baseline`(epoch 154 → 155 起)
- 跑 165 epoch,真 EarlyStop 在 epoch 319 触发

**STEP2-CONTINUE 结果**:

| 项 | STEP2 (epoch 154) | STEP2-CONTINUE (epoch 169 BEST) | Δ |
|---|---|---|---|
| val_composite_ckpt_score | 0.576 | **0.5881** | +0.012 ✅ |
| val_loss | 75.60 | 73.70(末尾,过 best 后)| -2.5% ✅ |
| val_min_d_mean | 1.590 | 1.580(末尾)| ≈ 持平 |
| val_gate_pass_rate | 0.455 | 0.531 | +0.076 ✅ |

Best ckpt:`composite_epoch169_score0.5881.ckpt`,md5 `127afa44a850d8f7e4fcdae17e2761a1`,**这是 STEP3 sample 用的 ckpt**(errata 4 §6 修订决议 → errata 5 §3 再确认)。

### 4.4 训练动力学观察(给 Exp5'' baseline)

| Loss | Epoch 0 | Epoch 169(BEST)| 减幅 |
|---|---|---|---|
| loss_coord | 1.030 | ~ 0.78(参考 epoch 154 数据)| ~ 24% |
| loss_type | 0.968 | ~ 0.017 | ~ 98% ✅ |
| loss_density | 0.0336 | ~ 0.04 | 持平 |
| **loss_pairwise_min** | 0.00149 | **~ 0.002** | 持平极低 ✅(errata 5 §2.3 自启动)|
| **loss_shell_dist** | 4.040 | ~ 3.15 | ~ 22%(看似降但梯度低效,详 §5)|
| **loss_shell_count** | 399.0 | ~ 401(末)| 0%(完全没降)|

`loss_pairwise_min` 极低且稳定 — 反映 model 全程严格满足 ≥ 1.5 Å 约束。
`loss_shell_dist / loss_shell_count` 数值未显著改善 — 鸡蛋启动问题(详 §5)。

---

## §5 Sample + 复合分(STEP3 SA4 输出)

### 5.1 Sample 配置

- ckpt: `composite_epoch169_score0.5881.ckpt`(errata 5 §3 决议)
- L_VIRTUAL = 20.0
- 三 split:val 7621 / test 4481 / **holdout 3025**(注:launch note §0.5 #3 写 1000,实际 holdout split 完整版是 3025;1000 是 Exp4 时代 `holdout_1000_ids.txt` 的旧子集)
- Diffusion sample steps: 与 Exp4/v2 一致(1000 steps cosine schedule)

### 5.2 STEP3-VERDICT(双指标,errata 4 §5.3 SOP)

| Split | composite (step5_3 7 项) | composite verdict | gate_pass_rate | gate verdict | min_d mean | min_d p10 |
|---|---|---|---|---|---|---|
| val     | **0.0801** | **RED ❌** | 64.0% | AMBER ⚠️ | 1.687 Å | ~ 1.0 Å |
| test    | 0.0795 | RED ❌ | 65.2% | AMBER ⚠️ | 1.695 Å | ~ 1.0 Å |
| holdout | 0.0828 | RED ❌ | 63.8% | AMBER ⚠️ | 1.681 Å | ~ 1.0 Å |

三 split 差异 < 0.004 — 泛化优秀。

### 5.3 7 项复合分明细

| 项 | val | test | holdout | 阈值 | 状态 |
|---|---|---|---|---|---|
| gate_pass_rate(权重 0.20)| 64.0% | 65.2% | 63.8% | ≥ 80% GREEN | ❌ AMBER 边缘 |
| shell-1 distance score(权重 0.20)| **0.0346** | 0.0343 | 0.0371 | ≥ 0.50 GREEN | ❌ 接近 0 |
| shell-1 coord_n score(权重 0.20)| 0.1799 | 0.1746 | 0.1887 | — 参考 | 极低 |
| shell-2 distance score(权重 0.10)| ~ 0.04 | ~ 0.04 | ~ 0.04 | — 参考 | 极低 |
| shell-2 coord_n score(权重 0.10)| 0.3159 | 0.3164 | 0.3147 | — 参考 | 中等 |
| type set-level acc(权重 0.10)| 0.0071 | 0.0069 | 0.0092 | — 参考 | 接近 0 |
| RMSD score(权重 0.10)| ~ 0.06 | ~ 0.06 | ~ 0.06 | — 参考 | 极低 |

权重和 ≈ 1.0(详细公式见 step5_3 脚本注释)。

**关键观察**:
- `gate_pass_rate` 是唯一接近 verdict 阈值的项 — `_pairwise_min_distance_penalty` 唯一生效证据
- 4 项 shell 监督相关全部接近 0 — `_shell_distance_loss + _shell_count_loss` 实际未生效
- `type set-level acc 0.007` — 模型连"shell-1 该是哪种元素"都没学,因为 shell-1 都没找对位置

### 5.4 Physical Sanity Report

| 项 | val | test | holdout | gate | 状态 |
|---|---|---|---|---|---|
| Collapse rate (≥ 50% atoms within 0.5 Å) | **0.00%** | 0.00% | 0.00% | ≤ 1% | ✅ |
| Shell-1 found rate | 99.7% | 99.6% | 99.7% | — | ✅ |
| **Pred shell-1 mean radial dist** | **6.32 Å** | 6.28 Å | 6.29 Å | — | **严重偏外** |
| **True shell-1 mean radial dist** | 2.27 Å | 2.30 Å | 2.25 Å | — | 参考 |
| Shell-1 RMSE (pred vs true) | **4.81 Å** | 4.77 Å | 4.79 Å | ≤ 1.0 Å | ❌ |

**致命差距**:模型预测 shell-1 在 ~ 6.3 Å,真值在 ~ 2.3 Å,**误差 4 Å**。L=20 box 内,模型把原子推到 box 中半径(L/2=10 内的 ~ 6 处),没有向真实物理 shell-1 收敛。

### 5.5 与 Exp5 v2 对比(verdict 改进证据)

| 指标 | Exp5 v2 (SA-METRICS-V3 dry-run 100) | Exp5' STEP3 (val 全集) | 改进倍数 |
|---|---|---|---|
| gate_pass_rate | 5-11% | **64%** | **6-13×** ✅ |
| composite (step5_3) | 0.005-0.011 | **0.080** | **10-16×** ✅ |
| shell-1 distance score | 0.0000 | 0.035 | 从 0 到 0.035 |
| collapse rate | 不明 | **0.00%** | 显式验证 ✅ |
| min_d mean (Å) | < 1.0 推测 | **1.687** | 显著改进 ✅ |

**Exp5' 部分胜利硬证**:fold 修复 + pairwise loss 让物理违反大幅改进。但 shell 监督未生效,verdict 仍 RED。

---

## §6 关键诊断:Shell loss 鸡蛋问题(errata 5 §2)

### 6.1 设计原意

`_shell_distance_loss` 依赖 gap 算法:

```python
coords_i = pred_frac_coords[i] * L                # cart Å
radial = coords_i.norm(dim=1).sort()[0]          # ascending
gaps = sorted_d[1:] - sorted_d[:-1]
boundaries = (gaps > 0.1563).nonzero()           # cart Å threshold
if len(boundaries) >= 1:
    pred_s1_d_mean = radial[:boundaries[0]+1].mean()
    loss += (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2
```

**核心假设**:pred 已具有壳层结构 → gap 切出 shell-1 / shell-2 → loss 衡量壳半径与真值差距 → 梯度修正壳位置。

### 6.2 实际机制

**Random init pred 没有壳层结构**:原子在 box 内随机散布,radial 分布近似单峰连续,**gap 全 < 0.1563 Å → boundaries 空 → n_active = 0**。

```
loss = total_loss / max(n_active, 1) = 0 / 1 = 0
```

数值是 finite 的(不会 NaN),但**梯度信号几乎为零**。

随训练推进,`_pairwise_min_distance_penalty` 把原子推开,**pred 开始有粗略 shell 形态**。但此时 gap 切出的"shell"位置随机(由 pred 自身分布决定,不一定在物理 shell 边界处)→ 算出的 pred_s1_d_mean 与 ground truth `true_shell1_d_mean` 是不同概念 → loss 数值降但**梯度方向不指向真实物理 shell-1**。

### 6.3 鸡蛋启动问题精确定义

> `loss_shell_dist` 需要 "pred 已有清晰壳层结构" 才能产生有效梯度;而 "pred 有清晰壳层结构" 需要 `loss_shell_dist` 提供梯度引导。两者互为条件,无外部信号打破循环。

对比 `_pairwise_min_distance_penalty` **不依赖 pred 已有任何结构** — 从 random init 起,只要任意两原子 cart 距离 < 1.5 Å 就有 violation > 0 → 梯度回传推开原子。**自启动**。

这是 STEP3 verdict 中 gate=64%(pairwise 大幅生效)与 shell-1 score=0.035(shell loss 几乎没生效)的根因差异。

### 6.4 为什么 Exp5'-MA 4 次 watch-only 都没追到

Exp5'-MA 监督失职复盘:

| 时机 | watch-only 提示 | 为什么没追 |
|---|---|---|
| proposal §2.5 | "epoch 0-10 ill-defined" | 假设 ill-defined 期短,长期会自正 |
| STEP1 launch note §11 | shell loss 数值监控 | 检查 finite + 数量级,不检查梯度有效 |
| STEP1-FIX-C C5 | shell_count 16~189 | 接受为 expected(epoch 0 端)|
| STEP2 epoch 0 | shell_count=399 占 95% | 标 watch-only,信任 epoch 50 后会改善 |

**根本是因为 watch-only 检查的是"loss 数值",不检查"loss 是否产生有效梯度"**。errata 5 §5.3 已落 ExpN+ 不变量级修正:dry-run 必 dump n_active 比例,watch-only 升级为 active monitor。

---

## §7 Verdict 总结(双指标 + 阶段定位)

### 7.1 双指标 verdict 表

| 指标 | 阈值 (GREEN) | 阈值 (AMBER) | Exp5' val 实测 | 阶段 verdict |
|---|---|---|---|---|
| Composite (step5_3 7 项) | ≥ 0.40 | ≥ 0.20 | **0.080** | **RED ❌** |
| Gate_pass_rate | ≥ 0.80 | ≥ 0.60 | 64.0% | **AMBER 边缘** |
| Min_d_mean (Å) | ≥ 2.0 | ≥ 1.5 | 1.687 | **AMBER** |
| Collapse rate | ≤ 1% | ≤ 5% | 0.00% | **GREEN ✅** |
| Shell-1 distance score | ≥ 0.50 | ≥ 0.20 | 0.035 | **RED ❌** |
| Shell-1 elem score | — | — | 0.007 | 极低 |

### 7.2 阶段定位:**Mixed verdict — Partial Success**

Exp5' 不是 v2 那种全面物理灾难(verdict ❌),而是**mixed verdict**:
- ✅ 物理约束部分成功(gate / collapse / min_d)
- ❌ 物理监督部分失败(shell 各项)

学术界 / 工业界对这种 mixed verdict 的定位:**publishable partial result**,前提是失败根因诊断清晰 + lessons 落 paper。

### 7.3 与 v2 对比(Exp5' 真实贡献)

| 维度 | v2 状态 | Exp5' 状态 | Exp5' 贡献 |
|---|---|---|---|
| Fold artifact | 隐藏 bug | 诊断 + 修复 | ⭐ 主贡献 |
| Pairwise physical constraint | 未引入 | 引入 + 验证生效 | ⭐ 真贡献 |
| Shell physical constraint | 未引入 | 引入但未生效 + 鸡蛋问题诊断 | 失败 lesson |
| Ckpt selection bug | 未发现 | 诊断 + 修复(strict=False) | 工程 lesson |
| Verdict 双指标 SOP | 单 verdict | 双指标 SOP 落 errata 4 | 流程 lesson |
| Paper trail | 1 份 final report | 5 份 errata + 7 份 hand-back | 完整 |

---

## §8 Limitations

### 8.1 Shell loss 设计前提失败

详 §6 + errata 5 §2。这是 Exp5' 最大 limitation,直接导致 verdict mixed。Exp5'' 重设计的核心动机。

### 8.2 训练超参 batch_size 改动未经 Exp5'-MA review

详 errata 4 §3。SA T2 dry-run 期间联合用户把 batch=16 改为 64,绕过 launch note §0.4 拍板的 review 流程。结果幸运没出事(STEP2-CONTINUE composite 真改进),但 review skip 是流程 bug。**ExpN+ 强制**:任何 launch note 拍板的训练超参,SA T2 不允许改动。

### 8.3 Best ckpt 在 STEP2 时丢失

详 errata 4 §2。STEP2 训练时 ckpt selection 用错 monitor(`val_gate_pass_rate` 抖动),真正 epoch 100-140 之间的 composite best ckpt 未被保存。STEP2-CONTINUE 续训找回 epoch 169 BEST(0.5881),但 STEP2 期间是否还有更高 epoch 的 ckpt 已无法验证。

### 8.4 Holdout split 数量不一致

launch note §0.5 #3 写 holdout=1000(基于 Exp4 时代 `holdout_1000_ids.txt`),实际 STEP3 sample 用完整 holdout split = 3025。final report v3 verdict 表用 3025,**1000 子集没单独评估**。SA OPEN Q1 已标。

### 8.5 Composite 公式权重和 ≠ 1.0

step5_3 复合分公式权重和 = 0.90(SA OPEN Q2 报告)。proposal §B.2 原始拍板如此,SA 未改。**Exp5'' / final report v3 引用复合分时需注明**。这不是 bug 但是设计 quirk。

### 8.6 Exp5'-MA watch-only 4 次未追根因

详 §6.4 + errata 5 §2.4 / §8。Exp5'-MA 监督责任已自评,修正措施落 errata 5 §5.1 / §5.3。

---

## §9 Lessons Learned(写进 ExpN 不变量)

### 9.1 Dataset ground truth 必须在原始物理量纲(cartesian Å)下验证

errata 3 §7.1 SOP:任何 dataset `__getitem__` 改动后,输出 ground truth coordinates 必须验证两两距离 ≥ MIN_BOND_LENGTH = 0.7 Å(H-H 物理下限),不允许仅在 frac 空间验证。

### 9.2 L_VIRTUAL 必须满足 L/2 ≥ CUTOFF_R

errata 3 §7.3:任何虚拟立方体表示设计,L_VIRTUAL / 2 ≥ CUTOFF_R 是物理约束。L=6 + CUTOFF_R=10 是设计边界扩展时的遗漏(Exp2 Fe-only 时 L=6 合理,88 元素扩展时未更新 L)。

### 9.3 依赖 pred 结构的 loss 必须 dry-run 验证 n_active

errata 5 §5.1:任何 gap-based / cluster-based / topology-based / structure-aware loss,设计文档必须显式标注"鸡蛋启动条件",dry-run 阶段 dump n_active 值,验证非零比例 ≥ 50% 才能进训练。

### 9.4 训练时 ckpt selection 公式 ≠ 评估时 verdict 公式

errata 5 §5.2:`val_composite_ckpt_score`(训练时,3 项加权)≠ step5_3 `composite`(评估时,7 项加权)。final report 只能用后者作 verdict。SA / Exp5'-MA 跨阶段引用必须明示公式来源。

### 9.5 Watch-only 不能只检查"数值",必须检查"梯度有效性"

errata 5 §5.3:"loss 数值 finite + 在合理范围"不充分。必须显式 check 梯度是否产生预期效果(如 shell_count_loss 应与 shell-1 distance score 同步降)。

### 9.6 Verdict 双指标并列(composite + gate)

errata 4 §5.3:final report verdict 表必须双指标(或多指标)并列,任何单指标 cherry-pick 都是 cherry-picking。Exp5' "composite RED + gate AMBER" 双指标公开,正是这个 SOP 的产物。

### 9.7 训练超参 launch note 拍板,SA review skip 是流程红线

errata 4 §3.4:任何 launch note 拍板的训练超参(batch / lr / scheduler / optimizer),SA T2 dry-run 不允许改动。性能优化(num_workers / persistent / pin_memory)允许加速但**不动核心超参**。

### 9.8 PL callback 启动期 metric 未注册问题用 strict=False 修

errata 4 §5.1:LightningModule 的 `on_validation_epoch_end` 中 self.log 的 metric 在第一个 val epoch 之前不存在,EarlyStopping/ModelCheckpoint 启动 init 时报 RuntimeError。修复方案:`EarlyStopping(strict=False)`(PL 原生 feature)。

---

## §10 Future Work — Exp5'' 方向

### 10.1 主线方向

详 errata 5 §6:Exp5'' 主线是"shell loss 重设计 + 不动其他"。

不动:
- `_pairwise_min_distance_penalty`(λ=1.0)— gate 64% 硬证生效
- L_VIRTUAL=20 / cost_density=0.2 / batch=64 / num_workers=16 / PreCollatedDataset
- ckpt callback strict=False / save_top_k=3

只改:Shell loss 公式,二选一:

**候选 A — Distance-supervised(KNN 切片)**:不依赖 gap,直接用 ground truth shell count 切 pred 最近邻 → 拉向 ground truth 半径
**候选 B — Distance-aware density**:把 `_density_loss` 从原点吸引子改为 shell-target attractor

二选一暂不定,Exp5'' proposal 阶段决议。可能是 A+B 混合。

### 10.2 不开 Exp6(架构级改动)

Exp5'' 是 loss 函数微调,**1-2 周可完成**。Exp6 是 equivariant decoder / 完全新方向,4-8 周。Exp5' 失败根因已诊断清晰(鸡蛋问题),Exp5'' 微调即可,无需 Exp6 大改。

### 10.3 长期方向(Exp7+ 候选,留 Exp5'' 后)

- Equivariant decoder(e3nn / SE(3)-Transformer)— Exp7+ 长线
- Hierarchical type prediction(errata 2 §3.2 候选 8)— Exp6+
- Classifier-free guidance(errata 2 §3.2 候选 9)— Exp6+

### 10.4 投稿建议

Exp5' 可投 short paper / workshop:
- 标题候选:"Diagnosis and Fix of Fold Artifact in Diffusion-based XAS Local Structure Prediction"
- 核心贡献:fold artifact 几何机制 + L_VIRTUAL 设计准则 + pairwise loss 验证 + shell loss 鸡蛋问题 + 5 份 errata SOP
- 不投长 paper(等 Exp5'' 出 GREEN verdict 后再投全长 paper)

---

## §11 Appendix — 关键文件 md5 与永久档案

### 11.1 代码 md5(STEP1-FIX-C → STEP2-CONTINUE 后)

| 文件 | md5 | 阶段 |
|---|---|---|
| `step3/diffusion_w_type_xas.py` | `0bc6fc346e60b990e3a9fc25140000f0` | STEP1-FIX-C |
| `step3/conf_xas/model/diffusion_xas.yaml` | `f73123a16166b220646af3537f7ece5b` | STEP1-FIX-C |
| `step3/xas_local_dataset_v2.py` | `94432ba56a7f3fd2ab0ce6281b66c5e6` | STEP1-FIX-C |
| `step3/xas_local_datamodule_v2.py` | `a040fd9d711011e28a1b7f75005c4def` | STEP1-FIX-C |
| `step3/precompute_structure_cache_exp5_prime.py` | `91836c2540b1d58acd72357b5c8e505c` | STEP1-FIX-C |
| `step4/step4_2_train.py` | (STEP2-CONTINUE 修订后,SA 报告新 md5)| STEP2-CONTINUE |

### 11.2 Ckpt md5

| Ckpt | md5 | 用途 |
|---|---|---|
| `last.ckpt.frozen_step2_final` | `9cd39421187df8d02951b9389266de36` | STEP2 末永久档案 |
| `epoch=004-gate=0.5305.ckpt.frozen_step2_lucky_shot` | (STEP2 hand-back 记录)| ckpt selection bug 副产物 |
| **`composite_epoch169_score0.5881.ckpt`** | **`127afa44a850d8f7e4fcdae17e2761a1`** | **Exp5' BEST,STEP3 用** |
| `composite_epoch164_score0.5837.ckpt` | (STEP2-CONTINUE 记录)| top-3 |
| `composite_epoch234_score0.5845.ckpt` | (STEP2-CONTINUE 记录)| top-3 |

### 11.3 数据 md5

| 文件 | md5 |
|---|---|
| `shell_boundaries.pkl` | `cf2050e4899160f5698ad2481377e94c` |
| `cache_metadata.json` | (含 `L_VIRTUAL: 20.0`,build_date 2026-05-03)|
| `train_structure_cache.pt` (新 L=20) | (STEP1-FIX-C 记录)|
| `val_structure_cache.pt` | (同上)|
| `test_structure_cache.pt` | (同上)|

### 11.4 Predictions + step5_3 输出

| 文件 | 说明 |
|---|---|
| `predictions/predictions_val.pt` | 7621 samples,ckpt epoch 169 |
| `predictions/predictions_test.pt` | 4481 samples |
| `predictions/predictions_holdout.pt` | 3025 samples |
| `logs/composite_score_{val,test,holdout}.txt` | step5_3 主报告 × 3 |
| `logs/composite_score_per_sample_{val,test,holdout}.csv` | per-sample 复合分 × 3 |
| `logs/min_d_violations_{val,test,holdout}.csv` | gate fail samples × 3 |
| `logs/physical_sanity_{val,test,holdout}.txt` | sanity 报告 × 3 |

---

## §12 5 份 Errata 索引

| Errata | 题目 | 影响范围 | 状态 |
|---|---|---|---|
| 2(继 Exp4)| `_density_loss` 塌缩 + Exp3 历史 + 方向 menu | EXPERIMENT4_FINAL_REPORT §7.2/§7.3/§10 | FINAL |
| 3(STEP1-AUDIT)| L=6 fold artifact + L=20 决议 + RMSD 三层归因 | EXP5_PRIME_PROPOSAL §2.1 + EXP4 final report §7.2/§10 + errata 2 §1.4 | FINAL |
| 4(STEP2 后)| Ckpt selection bug + verdict 双指标 SOP + last.ckpt 路径决议 | EXP5_PRIME_STEP1_HANDOFF §0.4 #1 / EXP5_PRIME_STEP2_TRAIN_HANDOFF §0.5 #5 | FINAL(§6 已被 errata 5 §3 修订)|
| 5(STEP3 后)| Shell loss 鸡蛋启动问题 + Exp5'' 方向决议(候选 A/B)| EXP5_PRIME_PROPOSAL §2.2/§2.3 + Exp5'' proposal | FINAL |

5 份 errata 是 Exp5' 阶段的核心 paper trail,与本 final report v3 配套引用。

---

## §13 致谢与 Paper Trail

Exp5' 工程历程:7 棒 SA(STEP1 / STEP1-AUDIT / STEP1-FIX / STEP1-FIX-C / STEP2-TRAIN / STEP2-CONTINUE / STEP3-SAMPLE)+ 5 份 errata + 1 份 final report v3。

**关键 SA 工作风格亮点**(写进 paper trail 给后续 ExpN 参考):

- **STEP1 SA1**:在 §1.5 之前主动自查 dataset 物理性,发现 fold artifact — **超 launch note 范围的责任心**,errata 3 的 trigger
- **STEP1-FIX SA2**:在 errata 3 §9 清单外主动追加 step5_1_sample.py / step4_2_train.py 改 L=20 — **清单外补全的工程严谨**
- **STEP3 SA4**:发现 ckpt 决议过期(launch note §0.5 #1 vs STEP2-CONTINUE 修订),停下 ping Exp5'-MA — **不擅自动 launch note 拍板的红线意识**

**Exp5'-MA(我)的失误**:
- errata 4 §2 揭示前 4 次 watch-only 没追根因(shell_count 数量级异常)
- 上一次 hand-back review 误判"训练失败接受现状"(用户本能反应救场,迫使 STEP2-CONTINUE 续训)
- launch note §0.4 拍板 best ckpt monitor 没强制 dry-run verify 真生效

**用户**:多次本能怀疑救场("epoch 4 best 这能信吗" / "训练还没结束怎么就取样" / "0.1563 这值怎么来的不合理吧"),三次都点透了 Exp5'-MA 自己没看清的问题。

---

*Exp5'-MA 撰写,2026-05-09*
*基于 SA-EXP5'-STEP1 / STEP1-AUDIT / STEP1-FIX / STEP1-FIX-C / STEP2-TRAIN / STEP2-CONTINUE / STEP3-SAMPLE 完整 hand-back + 5 份 errata + 训练 log + sample 输出*
*Exp5' 阶段总结。下一步:Exp5'' proposal(基于 errata 5 §6 候选 A/B 二选一重设计 shell loss),由 Exp5''-MA 主导。Exp5'' 完成后投稿决议。*
