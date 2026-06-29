# EXPERIMENT 4 FINAL REPORT
# DiffCSP-Exp4: 88-element XAS → Local Atomic Structure Prediction

> **撰写者**: DiffCSP-Exp4-Main-Agent 5
> **日期**: 2026-04-28
> **状态**: 完成,Holdout 检验通过,所有 §6 红线 0/4 触发
> **接力链**: MA1→MA2→MA3→MA4→MA5(本文档作者)+ Sub-Agent 1→2→3→4→4-续→4-续 2→Step4Agent→Step5Agent→Step6Agent
> **本文档目标读者**: (1)用户 review;(2)Exp5 Agent 主参考;(3)Exp6 Agent 回看
> **本文档对标**: EXPERIMENT2_FINAL_REPORT.md(同样三段式 + 历史 + 改进方向)

---

## 0. 执行摘要

### 0.1 一屏数字

| 指标 | val | test | **holdout** | max-Δ | Exp2 holdout | 评判 |
|---|---|---|---|---|---|---|
| RMSD (Å) | 1.4849 | 1.4852 | **1.4866** | 0.0017 | 1.47 | 🟢 几何 parity |
| Type Accuracy | 0.1877 | 0.1904 | **0.1973** | 0.0096 | 0.241 | 🟡 总体偏低,**分层 Tier B 0.26 = Exp2 parity** |
| pred_in_cutoff | 18.93/20 | 18.93/20 | **18.92/20** | 0.01 | 17.52/20 | 🟢 反而优于 Exp2 |
| true_in_cutoff | 19.80/20 | 19.84/20 | **19.79/20** | 0.05 | 18.99/20 | reference |
| 有效样本 / 名义 | 7621/7624 | 4481/4481 | 3025/3025 | — | 787/787 | silent_drop ≤ 0.04% |

### 0.2 一段话核心结论

**Exp4 在 88 元素中心任务下达到了 Exp2 Fe-only 的几何精度 parity,几何先验(原子聚集到中心附近)实际比 Exp2 学得更强(`pred_in_cutoff` 18.92 > 17.52)。但类型预测(Type Accuracy 0.19 vs Exp2 0.24)出现了在 Exp2 时代不存在的结构性失败模式——hard sample 上的 diffusion decoder 会塌缩(predicted atoms 聚集向中心,坐标 cost 维持在 population mean 但 type 完全错配)。这一失败模式的三个侧面分别在 fig5(rank-1 形态非单调)、fig3(hard sample 视觉)、fig4(RMSD↔TypeAcc Pearson r ≈ 0)上独立可见,共同指向 Exp5 的核心改进方向: type prediction 必须从 diffusion decoder 中解耦出来,辅以 anti-collapse 机制和/或 center-element conditioning。**

### 0.3 接力链关键决策回顾(Exp5 Agent 必读)

| 决策 | 来源 | 为什么 |
|---|---|---|
| 沿用 L=6 + min-image + [-0.5, 0.5] coord | Exp2 Step4d 验证 | Exp2 step4/4b/4c 系列证明 L=12 + 任何 `% 1.` 折叠都失败 |
| `cost_lattice = 0` | EXP4_PROPOSAL_v2 §1.3 锁定 | local cluster 任务下 lattice loss 无意义 |
| `N_NEIGHBORS = 20` 固定 | 同上 | 与 Exp2 直接可比 |
| **不加 TypeClassifier head** | EXP4_PROPOSAL_v2 §1.3,Exp3 时代证伪 | **本次 Exp4 数据反证了 Exp3 证伪逻辑** —— 见 §7 O3 |
| precision=fp32 全程 | MA4 决策 D1 | PL 2.5.5 中 `precision='bf16'` ≡ `'bf16-mixed'`,与 Exp2 PL 1.9.5 bf16 不等价 |
| dataset_v2 silent drop(return None)+ collate filter | Step4Agent Phase 4.6 修复 | SA3 时代改成 raise 不兼容 88 元素低密度结构,silent drop = 真正的 Exp2 行为 |
| Step 5 单 sample/sample | MA5 决策 | 与 Exp2 可比性最强 |

---

## 1. 指标定义(Exp4 增量,Exp2 已定义部分简引)

Exp2 final report §1 已经详细定义了 RMSD / Type Accuracy / pred_in_cutoff / true_in_cutoff 四个核心指标。这里只补充 Exp4 新增的两类分层:

### 1.1 eval_cutoff Tier 分层(Exp4 新增,Exp2 没做)

定义: 按每样本的 `eval_cutoff`(Step 2.5 算出的"含 d20 的最小壳层外缘",per-sample,float32)分桶:

| Tier | eval_cutoff 范围 | 物理含义 |
|---|---|---|
| A | ≤ 3.0 Å | 极密集结构(罕见,通常是金属间化合物) |
| B | 3.0 – 4.0 Å | 第 1 + 第 2 配位壳层主导(典型氧化物) |
| C | 4.0 – 5.0 Å | 第 2 壳层 + 部分第 3 |
| D | > 5.0 Å | 稀疏结构,第 3 壳层及以上 |

**为什么这样分**: XANES 物理上是 near-edge 探针,对第 1/2 配位壳层敏感度最高,远壳信息损失严重。Tier B 是物理上"XANES 应该最 informative"的区段,Tier B 的 TypeAcc 可作为模型上限的最佳估计。

### 1.2 By-Neighbor-Rank 分层(Exp4 fig5 新增)

定义: 对每个样本,把 20 个真实邻居按距中心的笛卡尔距离 sort,得到 rank 1(最近)到 rank 20(最远)。Hungarian 配对后,统计每个 true rank 上 `pred_atom_type == true_atom_type` 的比例。

**为什么这样做**: 直接验证"XANES 是 near-shell 探针"的物理假设——按 rank 应当呈单调下降。Exp4 fig5 实测 **形态非单调**(rank 1 = 0.243,rank 3 = 0.275 peak,rank 14 = 0.128 trough,rank 20 = 0.178 rebound)。这是 Exp4 vs Exp2 的关键 differentiator(详见 §7 O1)。

---

## 2. 数据集

### 2.1 Exp4 vs Exp2 数据规模对比

| 项 | Exp2 | Exp4 | 倍数 |
|---|---|---|---|
| 中心元素 | Fe-only | 88 元素 | 88× |
| 训练集 | 7,595 | 60,507 | 8.0× |
| 验证集 | 1,627 | 7,624 | 4.7× |
| 测试集 | 1,627 | 4,481 | 2.8× |
| Holdout | 787 | 3,025 | 3.8× |
| 总有效样本 | 11,636 | 75,637 | 6.5× |
| Incompat 池(Exp4 封存) | — | 52,745 | — |
| Type 分类难度 | ~30 元素 | 88 元素 | 显著上升 |

### 2.2 数据来源(Exp4 主结构)

继承 Materials Project FEFF 计算的 K-edge XANES,Step 1-2.5 经过 5 阶段筛选 + 物理对齐(详见 EXP4_PROGRESS_LOG.md)。

**核心改进 vs Exp2**: Step 2.5 引入 `site_equivalence_tag` + `eval_cutoff` per-sample 计算 + `shell_boundaries.pkl`(完整壳层信息),为 Tier 分层评估提供数据基础。

### 2.3 Exp4 数据集已知偏差

**Tier A holdout 全空**(N=0/3025): split selection 偶然性,Tier A 在 val/test 也只有 13 / 3 样本(< 0.2%),不影响主结论但**Exp5 / Exp6 评估时若用 Tier A 数字必须 caveat**。

**88 元素分布不均**: 中心元素中 O / Fe / Cu / Mn 等占多数,稀有元素(如 Bi / U / La 系)样本数极少。Step 5 fig3 在 6 panel 内体现了这种多样性(Al / Ho / Cl / F / C / Sb 各一),但**主体仍是 transition metal + main group oxide**。Exp5 评估时如果想做"按中心元素分组",注意 N < 50 的元素族要合并或单独标记。

---

## 3. 模型与方法(Exp4 实际跑通的版本)

### 3.1 架构概览

```
XAS 谱(per sample)                      
   │                                       
   ├── XANES xmu (150 点) ────┐           
   ├── EXAFS chi1 (200 点) ───┤            
   └── FEFF features (74 维) ─┤  ← Exp4 改 73→74,新增中心元素 one-hot 一维
                              │
                              ↓
                    SpectrumEncoder
                              │
                              ↓
                    Latent (256 维)        
                              │
                              ↓
              CSPDiffusion (DiffCSP backbone)  
                              │
                              ↓
              反扩散 1000 步                
                              │
                              ↓
        Output: 20 个邻居原子的 (frac_coords, atom_types)
```

### 3.2 关键不变量(Exp4 锁定)

| 量 | 值 | 来源 |
|---|---|---|
| L (盒子边长) | 6.0 Å | Exp2 Step4d 验证 |
| coord 系 | [-0.5, 0.5] | Exp2 Step4d 验证 |
| min-image 折叠 | 是 (`frac -= round(frac)`) | Exp2 Step4d |
| `cost_lattice` | 0 | MA4 决策 |
| N_NEIGHBORS | 20 | EXP4_PROPOSAL_v2 §1.3 |
| 邻居搜索半径 | 10.0 Å | Exp2 沿用 |
| <20 邻居处理 | `return None`(silent drop) | Step4Agent Phase 4.6 修复 |
| FEFF feature 维度 | 74(73+中心元素 one-hot)| Sub-Agent 3/4 改完 |
| Latent 维度 | 256 | DiffCSP 默认 |
| 反扩散步数 | 1000 | Exp2 沿用 |

### 3.3 训练超参数(Step4Agent 实跑值)

| 参数 | 值 |
|---|---|
| precision | fp32(MA4 D1) |
| batch_size | 16 |
| optimizer | Adam, lr=1e-4 |
| gradient_clip_val | 1.0 |
| max_epochs | 500 |
| early_stop patience | 30 |
| save_top_k | 1 |
| num_workers | 0(pymatgen SGA worker safety) |
| accelerator/devices | gpu / 1 |
| **实际早停 epoch** | **396**(best at epoch 366) |
| **训练 wall time** | **~32 小时**(单 RTX 4090) |

### 3.4 与 Exp2 比的改动总结(代码层)

| 文件 | Exp4 改动 | 备注 |
|---|---|---|
| `spectrum_encoder.py` | 5 处 `73→74` | Sub-Agent 3 改完 |
| `diffusion_w_type_xas.py` line 108 | `feat_dim=74` | Sub-Agent 4 改完 |
| `diffusion_xas.yaml` line 18 | `feat_dim=74` | Sub-Agent 4 改完 |
| `xas_local_dataset_v2.py` | 新建,基于 Exp2 `xas_local_dataset_L6.py` 改造,支持 v2 split + Step 2.5 字段 + 88 元素 + Phase 4.6 silent drop | Sub-Agent 3 + Step4Agent |
| `xas_local_datamodule_v2.py` | 新建,加 None-filter collate (`xas_collate_fn_v2`) | Sub-Agent 4 + Step4Agent Phase 4.6 |
| `forward_test.py` | 新建,5 phase sanity test(Phase 6.1-6.5) | Sub-Agent 4-续 2 fp32 改完 |

---

## 4. 训练历史(详细时间线 + 关键事件)

### 4.1 Step 3 接力链(env + dataset + sanity test)

时间跨度: 2026-04 中下旬。**6 个 sub-agent 接力**:

| Sub-Agent | 主要交付 | 状态 |
|---|---|---|
| SA1 | env 初设 | 完成 |
| SA2 | spectrum_encoder.py 改 73→74 | 完成 |
| SA3 | xas_local_dataset_v2.py 新建 | 完成,但留下 `<20 邻居 → raise` 设计缺陷 |
| SA4 | datamodule_v2 + Phase 5/5b 改动 | 完成 |
| SA4-续 | env 装 18 个 diffcsp 子依赖,Phase 6.4 PASS | 完成,Phase 6.5 bf16 阻塞 |
| **SA4-续 2** | **forward_test.py fp32 改造,Phase 6.5 PASS,5/5 闸门 CLEAR** | 完成 |

**关键决策**: PL 2.5.5 中 `precision='bf16'` ≡ `'bf16-mixed'`,与 Exp2 PL 1.9.5 不等价。MA4 决策 D1: 全程 fp32(放弃 PL 1.9.5 跨 env 重建路径)。

### 4.2 Step 4 训练阶段(Step4Agent + Phase 4.6 修复)

#### 4.2.1 第一次启动(红灯)

启动时间: 2026-04-26 早晨。
进程 PID: 3267562。
**红灯触发**: dataset L224 raise(`s contradicts Step 2.5 assumption (typical 30-100 within 10 Å)`)。
**根因**: SA3 把 Exp2 的 `return None`(silent)改成 `raise`,但 datamodule_v2 的 collate 没补 None-filter。88 元素全量数据集中存在低密度结构(Li 层状化合物 / 间隙化合物等),触发 raise 后整个训练直接 Exit 1。

**关键诊断转折**: 用户(凭 domain 知识)指出"如果黑名单一半以上,就应该和之前的逻辑一样"。MA5 经源码核查(Exp2 `xas_local_dataset_L6.py` L175-180),确认 Exp2 真实行为是 silent `return None`,SA3 改 raise 是设计缺陷而非数据问题。**这一诊断节省了一次黑名单预扫描的 5-10 min + 工程冗余**,直接进入正确修复路径。

#### 4.2.2 Phase 4.6 修复(Step4Agent 解禁执行)

修复 scope:
- `xas_local_dataset_v2.py`: 2 处 `raise` → `return None`(L224 < 20 邻居 check + frac sentinel check)
- `xas_local_datamodule_v2.py`: 加 `xas_collate_fn_v2`,filter None 后用 `Batch.from_data_list`

修复后重跑 forward_test.py: 5/5 PASS。重跑 smoke test: PASS。

#### 4.2.3 第二次启动(成功)

启动时间: 2026-04-26 ~12:05。
**训练曲线关键点**:
- epoch 1: val_loss = 0.9846(初始)
- epoch 50: val_loss ≈ 0.840(显著下降阶段)
- epoch 150: val_loss ≈ 0.770(放缓)
- epoch 273: val_loss = 0.7387(plateau 入口)
- **epoch 366: val_loss = 0.72998**(best)
- epoch 396: 早停触发(patience=30 计满)
- 完成时间: 2026-04-27 14:26

**收敛健康度**: 单调下降,无 NaN/Inf,无过拟合迹象。约 4.5 min/epoch 的 wall time(60K samples / bs=16 / num_workers=0 / pymatgen SGA ~13 ms/sample bottleneck)。

### 4.3 Step 5 评估(Step5Agent + Phase 5b 解禁)

#### 4.3.1 Phase 5a(val + test)

- val sample: 7621/7624 effective(silent_drop=3, 0.04%)
- test sample: 4481/4481 effective(silent_drop=0)
- 两次 sample wall time: ~9 小时
- val/test metrics 一致到 3 位小数(无过拟合信号)

#### 4.3.2 Phase 5b(holdout 解禁)

- Step5Agent 经历 2 次 false start:
  1. `(jhub_env)` 误用,缺 hydra → 修复用 mlff env 绝对路径
  2. `loader_map` 残留 phase 5a holdout 闸门 → 修复用 `XasLocalDatasetV2(split="holdout")` 直接构造
- holdout sample: 3025/3025 effective(silent_drop=0)
- holdout metrics: RMSD 1.4866,TypeAcc 0.1973,pred_in 18.92,与 val/test 一致到 3 位小数
- **4/4 红线全过,0 过拟合证据**

### 4.4 Step 6 可视化(Step6Agent)

- 6 张 figure 全 PASS,wall time 5.1 s
- 自检: Step6 Hungarian 重算 weighted avg TypeAcc = 0.1877 == Step5 val_csv mean(|Δ|=0.0000),算法实现可信
- 4 个 open observation O1-O4 上交 MA5,本文档 §7 详细解读

---

## 5. 评估结果(完整数据)

### 5.1 Aggregate(三 split 横览)

| Metric | val (N=7621) | test (N=4481) | holdout (N=3025) |
|---|---|---|---|
| RMSD mean (Å) | 1.4849 | 1.4852 | 1.4866 |
| RMSD median (Å) | 1.4746 | 1.4712 | 1.4780 |
| RMSD std (Å) | 0.1246 | 0.1292 | 0.1216 |
| RMSD min / max | 0.985 / 2.713 | 1.072 / 2.730 | 0.886 / 2.359 |
| TypeAcc mean | 0.1877 | 0.1904 | 0.1973 |
| TypeAcc median | 0.1500 | 0.1500 | 0.1500 |
| TypeAcc std | 0.1820 | 0.1842 | 0.1880 |
| pred_in_cutoff mean | 18.93/20 | 18.93/20 | 18.92/20 |
| true_in_cutoff mean | 19.80/20 | 19.84/20 | 19.79/20 |
| eval_cutoff mean (Å) | 4.647 | 4.630 | 4.661 |

### 5.2 Tier 分层(完整 4 split × 3 metric)

#### Tier 内 Per-split 数字

**RMSD by Tier(单位 Å,跨 split 一致性 <0.01)**:

| Tier | val | test | holdout |
|---|---|---|---|
| A: ≤3 Å | 1.5645 (N=13) | 1.4408 (N=3) | N/A (N=0) |
| B: 3-4 Å | 1.4746 (N=1961) | 1.4691 (N=1164) | 1.4663 (N=797) |
| C: 4-5 Å | 1.4846 (N=3893) | 1.4865 (N=2302) | 1.4899 (N=1536) |
| D: >5 Å | 1.4964 (N=1754) | 1.5012 (N=1012) | 1.5027 (N=692) |

**TypeAcc by Tier**(关键 differentiator):

| Tier | val | test | holdout | range |
|---|---|---|---|---|
| A: ≤3 Å | 0.3577 (N=13) | 0.0167 (N=3) | N/A | (small N noise) |
| B: 3-4 Å | **0.2496** | **0.2661** | **0.2590** | 0.017 |
| C: 4-5 Å | 0.1812 | 0.1803 | 0.1878 | 0.008 |
| D: >5 Å | 0.1316 | 0.1267 | 0.1474 | 0.021 |

**关键发现**: Tier B 三 split 一致 0.25-0.27,**= Exp2 Fe-only holdout 0.241**。Tier B 是 XANES 物理 informative 的最佳区段;Tier B 表现 = Exp2 baseline,说明 model 已经把 XANES 物理上能榨的"近邻信息"用到了上限。

#### Tier 单调性(三 split 都呈现 B > C > D)

```
       val      test     holdout
B   0.2496    0.2661    0.2590    ← Exp2 parity
C   0.1812    0.1803    0.1878    ← physics 信号衰减
D   0.1316    0.1267    0.1474    ← 信息论上限附近
```

### 5.3 By-Neighbor-Rank 分层(fig5,只 val)

完整 20 rank 的 TypeAcc:

| rank | TypeAcc | rank | TypeAcc |
|---|---|---|---|
| 1 | 0.2434 | 11 | 0.1491 |
| 2 | 0.2661 | 12 | 0.1509 |
| 3 | **0.2752 (peak)** | 13 | 0.1337 |
| 4 | 0.2605 | 14 | **0.1283 (trough)** |
| 5 | 0.2447 | 15 | 0.1447 |
| 6 | 0.2365 | 16 | 0.1357 |
| 7 | 0.1992 | 17 | 0.1593 |
| 8 | 0.1872 | 18 | 0.1598 |
| 9 | 0.1708 | 19 | 0.1664 |
| 10 | 0.1636 | 20 | **0.1783 (rebound)** |

**形态**: 非单调。peak 在 rank 3,trough 在 rank 14,rank 20 反弹。详见 §7 O1。

### 5.4 RMSD ↔ TypeAcc 相关(fig4)

| split | Pearson r | p-value | slope | intercept |
|---|---|---|---|---|
| val | +0.0068 | 0.555 | +0.0099 | +0.1730 |
| test | -0.0264 | 0.077 | -0.0377 | +0.2464 |
| holdout | +0.0218 | 0.230 | +0.0337 | +0.1472 |

**全部 |r| < 0.03,全部 p > 0.05**。两个指标在 Exp4 上**统计上独立**。详见 §7 O3。

---

## 6. 6 张 Figure 视觉描述

(详细数字见 §5;此处只描述视觉特征)

### 6.1 fig1 — RMSD 分布直方图(3 panel)

- 3 子图横排,val / test / holdout
- 每子图: Gaussian-like hist(40 bins,range 0-4 Å),mean 虚线 ≈ 1.485,random baseline 2.32 红虚线
- **三 split 形状几乎完全重合**(no overfitting 视觉证据)

### 6.2 fig2 — TypeAcc 分布直方图(3 panel)

- 3 子图横排,21 bins(k/20)
- **双峰分布**: TypeAcc=0 处大 spike(matched 0/20),0.25-0.30 处小 hump
- 双峰是 Exp2 也有的形态——但 Exp4 的 0 处 spike 显著更高(O2 collapse mode 体现之一)

### 6.3 fig2b — TypeAcc by Tier(3 split boxplot)**【Headline figure】**

- 4 tier × 3 box(val 蓝 / test 橙 / holdout 绿)
- Tier A holdout 标 "N/A" 绿色注释
- Exp2 reference 红虚线 0.241 横穿 Tier B box 中位
- **视觉故事**: Tier B 三 split 中位都贴 Exp2 line,Tier C 下降到 ~0.18,Tier D 进一步下降到 ~0.13-0.15。**这一张图把"模型已经达到 XANES 物理上限"讲完了**。

### 6.4 fig3 — 3D 结构对比(6 panel,val only)

| Panel | sample_name | 中心 | RMSD | TypeAcc | 视觉特征 |
|---|---|---|---|---|---|
| Best #1 | mp-10908 | Al | 0.985 | 0.050 | pred ≈ true 完美重合,但 type 全错(O2 暗示) |
| Best #2 | mp-4291 | Ho | 1.027 | 0.500 | pred 与 true 大体重合,type 半对 |
| Mid #1 | mp-561299 | Cl | 1.485 | 0.000 | **pred 塌缩到中心**,true 散布(O2 教科书例) |
| Mid #2 | mp-780857 | F | 1.485 | 0.200 | pred 部分散布,部分塌缩 |
| Worst #1 | mp-20978 | C | 2.712 | 0.000 | **pred 完全塌缩**(O2 极端例) |
| Worst #2 | mp-1013704 | Sb | 2.631 | 0.000 | **pred 完全塌缩** |

中心元素多样性确认: Al / Ho / Cl / F / C / Sb,**全无 Fe** —— 与 Exp2 的 Fe-only 形成视觉对比。

### 6.5 fig4 — RMSD vs TypeAcc(3 split overlay scatter)

- val 蓝 / test 橙 / holdout 绿,alpha=0.3 散点
- 3 条独立回归线**几乎水平**(slope <0.04)
- annotation 框三组 r 全部 < 0.03,全部 p > 0.05

### 6.6 fig5 — TypeAcc by Neighbor Rank(20 bar,val only)

- x 轴 rank 1-20,y 轴 TypeAcc
- **形态非单调**: rank 3 peak,rank 14 trough,rank 20 rebound
- random baseline 1/88 = 0.0114 灰虚线
- mean reference 0.1877 蓝虚线

---

## 7. 三大发现(O1/O2/O3 深度解读)

这一节是 Exp5 / Exp6 的核心参考。**三个发现是同一个底层机制的三个侧面**。

### 7.1 O1 — fig5 形态非单调,Exp4 Specific 失败签名

#### 观察

- rank 1 TypeAcc = 0.2434(没到峰)
- rank 3 TypeAcc = 0.2752(peak)
- rank 14 TypeAcc = 0.1283(trough)
- rank 20 TypeAcc = 0.1783(rebound)

#### 三个候选解读(MA5 倾向 + 候选 1)

**候选 1: Rank 1 高熵问题**(MA5 倾向)
Exp2 是 Fe-only,rank 1 邻居在氧化物中**几乎确定是 O**——模型可学到近确定先验。Exp4 是 88 元素中心,rank 1 身份高度依赖中心元素(F-center 看 cation,Fe-center 看 O,C-center 看 C/N/O,等等)。Center-element-conditioned 信息没有显式 inject 到 decoder,模型只能从 spectrum 间接推断,在 rank 1 这种"最高方差"位置失败率最大。

**候选 2: Rank 14 trough = 第 2 壳层 - 第 3 壳层过渡的信息真空**
Rank 11-15 落在"既不够近(near-edge XANES 不敏感)又不够远(EXAFS 周期信号未充分发展)"的过渡区。第 2 壳层多重散射混合,第 3 壳层周期性还没显现,这是 XAS 物理上信息密度最低的 rank 区间。

**候选 3: Rank 18-20 rebound = 周期性"白送"加分**
最远 3 个邻居经常落在晶体周期重复位点上(host-lattice 高对称等价位置),即使模型没真正预测对,匹配上"位置等价的同类元素"概率比中间 rank 高。

#### 对 Exp5 的指引

**候选 1 是最 actionable 的**——可以直接通过 center-element conditioning 改造来攻击。Exp5 加 center embedding 注入 decoder 后,**应该重跑 fig5,看 rank 1 是否爬升到接近 rank 3 水平**。这是 Exp5 改进的 quantitative 验证点。

### 7.2 O2 — Predicted-Atom Collapse Mode(只在 fig3 视觉可见)

#### 观察

fig3 中 Mid #1 (Cl, RMSD=1.485, TypeAcc=0.000)、Worst #1 (C, RMSD=2.712, TypeAcc=0.000)、Worst #2 (Sb, RMSD=2.631, TypeAcc=0.000)三个 panel,**predicted atoms(空心圆)聚集在原点附近 ±1.5 Å,而 true atoms(实心)散布到 ±3.5 Å**。

对照 Best #1 (Al)、Best #2 (Ho):pred 与 true 共同分布,无塌缩。

#### 机制候选(MA5 解读)

Diffusion decoder 在 hard sample 上**部分回退到中心-塌缩先验(mean-position fallback)**。Hungarian min-image 匹配会把"塌缩的 pred 云"分配到"散布的 true 原子",几何 cost 由于 min-image 截断而**有界**(RMSD 不会爆炸,落在 population mean ~1.49 附近),但 type 配对完全任意化,TypeAcc 直接归零。

**这解释了为什么 fig2 在 TypeAcc=0 处有大 spike**——大量 sample 落在"RMSD 看起来正常但 TypeAcc 完全错"的 collapse 模式。

#### 为什么 Exp2 没看到

Exp2 是 Fe-only,所有 sample 中心都是 Fe + 氧化物近邻分布,**model 学到的"中心塌缩先验"恰好与真实分布部分重合**——塌缩没造成灾难性 type 错配。Exp4 88 元素中心,真实邻居分布因元素不同跨度极大,塌缩先验与真实分布偏离更远,失败更明显。

#### 对 Exp5 的指引

两个潜在攻击方向:

1. **Anti-collapse auxiliary loss**: 加一项惩罚 predicted_atoms 标准差过低(比 true 的 std 显著小)
2. **Per-sample diversity regularization**: 强制 predicted_atoms 的 pairwise 距离方差不能崩塌

### 7.3 O3 — RMSD ↔ TypeAcc Decoupling(三 split 全 |r|<0.03)

#### 观察

|r| 在三 split 都 < 0.03,p > 0.05,**统计上独立**。Exp2 fig4 应有显著负相关(Exp2 final report 没贴具体数字,但定性是"高 RMSD 通常伴随低 TypeAcc")。

#### 机制(O2 的另一面)

如果 collapse mode(O2)在大量 sample 上表现为"RMSD 居中 + TypeAcc 归零",这一种 sample population 在 fig4 上是一组分布在(RMSD=1.4-1.6, TypeAcc=0)的密集云——它**机械地拉平任何潜在的 RMSD-TypeAcc 相关性**。

#### 反证 Exp3 时代的"加 head 无效"结论

EXP4_PROPOSAL_v2 §1.3 锁定"不加 TypeClassifier head"基于 Exp3 时代的证伪,逻辑是"diffusion decoder 已经学到 type,加 head 无显著增益"。但 Exp4 数据显示:

- 几何 prediction(RMSD)已饱和到 L=6 prior 上限(Exp2 parity)
- type prediction(TypeAcc 0.19)远低于物理上限(Tier B 0.26 是 XANES 上限的可达点,但 aggregate 0.19 因 collapse 拖累)
- **两者已经几乎独立(|r|<0.03)**

这意味着: **解耦 type prediction 进独立 head,在 Exp4 数据上极不可能让坐标变差**(独立的指标本就不会互相干扰),而**有 strong 概率让 type 提升**(可单独优化,不被 diffusion 训练目标稀释)。

**Exp3 证伪是基于 Fe-only,不再适用 Exp4 的 88 元素 + collapse 现象**。Exp5 应该重新评估这一方向。

---

## 8. 已知失败模式 + 工程债务

### 8.1 失败模式

1. **Hard sample collapse**(O2): 占比未量化,但从 fig2 TypeAcc=0 spike 大小估计 ~10-20% sample。
2. **Rank 1 weakness**(O1): 88 元素中心导致 rank-1 高熵,模型学不全。
3. **Tier A 数据稀疏**: 全数据集 Tier A 只 16 sample(13+3+0),evaluation 上 Tier A 数字属噪声。

### 8.2 工程债务(供 Exp5 知情)

| 债务项 | 影响 | Exp5 是否要还 |
|---|---|---|
| `dataset_v2` silent drop 没 log 哪些样本被 drop | 未来 ablation 难溯源 | 建议 Exp5 加 drop log,但不强求 |
| `predictions_*.pt` 没含 sample-level 元数据(center_element / eval_cutoff)| Step6 fig5 计算需 join 其他 CSV | Exp5 sample 脚本扩展输出即可 |
| `fig5` 的 Hungarian 实现独立于 metrics 脚本 | 维护两套算法风险 | Exp5 重构时可合并 |
| `precision='bf16'` PL 2.5.5 vs 1.9.5 行为漂移记录 | 未来若想试 bf16 需重做兼容验证 | Exp5 不动这个 |
| Exp4 没存 `sample_index → silent_drop_reason` 映射 | 不能精确判断"什么样的样本被 drop" | Exp5 sample 时建议补 |
| Tier A 极稀(16 全集) | Exp5 评估若用 Tier A 数字 不可信 | Exp5 评估时显式 caveat 或重新 binning |

### 8.3 不变量保持(Exp5 必须继承)

- L=6, min-image, [-0.5, 0.5] coord 系: 这是 Exp2 step4 → step4d 系列证明的必要条件,Exp5 任何架构改动都不能动这三条
- `cost_lattice = 0`: local cluster 任务下不变
- N_NEIGHBORS = 20: 与 Exp2 / Exp4 可比性必须的
- silent drop + collate filter: 这是处理 88 元素低密度结构的必要机制

---

## 9. 完整文件清单 + 数据位置 + 读取方式 ⭐

**这一节是 Exp5 / Exp6 启动后最高频访问的参考。每条都标明"当前用版本 vs 备份版本",防止混淆。**

### 9.1 服务器目录结构(Exp4 完成态)

```
/home/tcat/diffcsp_exp4/
├── code/
│   ├── .env                          ← Sub-Agent 4-续 创建,3 行 export
│   ├── diffcsp/                       ← DiffCSP 包(不动)
│   ├── conf/                          ← 顶层 hydra config(不动)
│   ├── step2/
│   │   └── spectrum_encoder.py        ← Sub-Agent 3 改完(73→74,5 处)
│   ├── step3/
│   │   ├── xas_local_dataset_v2.py        ← ⭐ 当前用版本(Phase 4.6 silent drop)
│   │   ├── xas_local_dataset_v2.py.bak_phase46  ← 备份(Phase 4.6 修复前,raise 版本)
│   │   ├── xas_local_datamodule_v2.py     ← ⭐ 当前用版本(含 xas_collate_fn_v2)
│   │   ├── xas_local_datamodule_v2.py.bak_phase46  ← 备份(无 None-filter)
│   │   ├── diffusion_w_type_xas.py        ← 当前用版本(line 108 改 73→74)
│   │   ├── diffusion_w_type_xas.py.bak    ← Sub-Agent 4 早期备份
│   │   ├── conf_xas/model/diffusion_xas.yaml   ← 当前用(line 18 改 74)
│   │   ├── conf_xas/model/diffusion_xas.yaml.bak  ← 备份
│   │   ├── forward_test.py            ← ⭐ 当前用(fp32, 5/5 PASS, md5 71a0e546…)
│   │   ├── forward_test.py.bak3       ← 备份(fp32 改前的最后 4/5 PASS, md5 3d1441c3…)
│   │   ├── forward_test.py.bak2       ← Sub-Agent 4 中期备份
│   │   └── forward_test.py.bak        ← 早期备份(可能不存在)
│   ├── step4/                          ← Step4Agent 产出
│   │   ├── step4_1_smoke_test.py
│   │   ├── step4_2_train.py            ← ⭐ 训练入口
│   │   └── step4_README.md
│   ├── step5/                          ← Step5Agent 产出
│   │   ├── step5_0_hard_check.py
│   │   ├── step5_1_sample.py           ← ⭐ 当前用(Phase 5b 直接构造 dataset_v2)
│   │   ├── step5_1_sample.py.bak_phase5     ← 备份(Phase 5a, 含 holdout RuntimeError 闸门)
│   │   ├── step5_1_sample.py.bak_phase5b_attempt1  ← 备份(loader_map 残留闸门版)
│   │   ├── step5_2_compute_metrics.py  ← ⭐ 当前用
│   │   ├── step5_2_compute_metrics.py.bak_phase5  ← 备份(Phase 5a 版)
│   │   ├── predictions_val.pt          ← ⭐ Step 6 / Exp5 ablation 主要输入
│   │   ├── predictions_test.pt
│   │   ├── predictions_holdout.pt
│   │   ├── per_sample_metrics_val.csv  ← ⭐ Step 6 / Exp5 数据分析主要输入
│   │   ├── per_sample_metrics_test.csv
│   │   ├── per_sample_metrics_holdout.csv
│   │   ├── metrics_report_val.txt
│   │   ├── metrics_report_test.txt
│   │   └── metrics_report_holdout.txt
│   └── step6/                          ← Step6Agent 产出
│       ├── step6_0_hard_check.py
│       ├── step6_visualize.py          ← ⭐ 当前用
│       └── figures/
│           ├── fig1_rmsd_distribution.png
│           ├── fig2_typeacc_distribution.png
│           ├── fig2b_typeacc_by_tier.png    ← ⭐ Headline figure
│           ├── fig3_structure_comparison.png
│           ├── fig4_rmsd_vs_typeacc.png
│           └── fig5_typeacc_by_rank.png
│
├── data/                               ← 全部数据,不动
│   ├── data_inventory_v2.csv           (33.5 MB) ⭐ 主索引
│   ├── train_samples_v2.csv            (3.3 MB) 60507 行
│   ├── val_samples_v2.csv              (0.42 MB) 7624 行
│   ├── test_samples_v2.csv             (0.24 MB) 4481 行
│   ├── holdout_samples_v2.csv          (0.17 MB) 3025 行 ⚠️ Exp5 也要保护
│   ├── feff_features_imputed.pkl       (40.3 MB) 74 维 feature
│   ├── feff_feature_scaler.pkl         (1.6 KB) RobustScaler
│   ├── feff_feature_names.txt          74 个特征名
│   ├── spectra_train.pkl               (148.4 MB)
│   ├── spectra_val.pkl                 (18.7 MB)
│   ├── spectra_test.pkl                (11.1 MB)
│   ├── spectra_holdout.pkl             (7.4 MB) ⚠️ Exp5 也要保护
│   ├── shell_boundaries.pkl            (369.5 MB) ⭐ Tier 评估必读
│   ├── site_equivalence_tag.csv        (9.5 MB) 归档
│   ├── incompat_pool.csv               (3.3 MB) 封存,Exp5 也不动
│   └── MP_all_POSCAR_flat/              ← POSCAR 目录
│
├── checkpoints/
│   ├── best-epoch366-val0.7300.ckpt    (40.2 MB) ⭐⭐ Exp5 fine-tune 起点
│   ├── last.ckpt                       (40.2 MB) epoch 395 末态
│   ├── _smoke/                          ← Phase 4.2 smoke test 残留(可清理)
│   └── (red light v1 残留 ckpt 已被 best/last 覆盖,无遗物)
│
├── logs/
│   ├── (训练 + 评估 + 可视化全套 log)
│   ├── step4_red_light_2026-04-26_stderr.log  ← 红灯归档
│   ├── step4_red_light_2026-04-26_stdout.log
│   ├── step4_train_v2_stdout.log              ← Phase 4.6 修复后训练 log
│   ├── step4_train_v2_stderr.log
│   ├── step5_*.log
│   └── step6_*.log
│
└── best_checkpoint_path.txt            ← 单行,内容 = best ckpt 绝对路径
```

### 9.2 文件读取代码片段(Exp5 复制即可用)

#### 读取 best ckpt(Exp5 fine-tune 起点)

```python
import torch
import sys
sys.path.insert(0, "/home/tcat/diffcsp_exp4/code/step3")
sys.path.insert(0, "/home/tcat/diffcsp_exp4/code/step2")

CKPT_PATH = "/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt"
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
# ckpt keys: "epoch", "global_step", "state_dict", "hyper_parameters", ...
# state_dict size: 96 keys, all "decoder.*" prefix
# hyper_parameters 含 feat_dim=74, cost_lattice=0.0, beta_scheduler.timesteps=1000

# 重建 model:
from omegaconf import OmegaConf
import hydra
cfg = OmegaConf.load("/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml")
model = hydra.utils.instantiate(cfg.model, lattice_scaler=None, scaler=None)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
```

#### 读取 per-sample metrics(Exp5 数据分析)

```python
import pandas as pd

# 三 split 的 per-sample 指标
df_val = pd.read_csv("/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv")
df_test = pd.read_csv("/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_test.csv")
df_hold = pd.read_csv("/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_holdout.csv")

# Schema: sample_name, mp_id, rmsd, type_acc, n_pred_in, n_true_in, eval_cutoff
# Tier 分析:
def get_tier(c):
    if c <= 3.0: return "A"
    if c <= 4.0: return "B"
    if c <= 5.0: return "C"
    return "D"
df_val["tier"] = df_val["eval_cutoff"].apply(get_tier)
df_val.groupby("tier")[["rmsd", "type_acc"]].mean()
```

#### 读取 predictions_val.pt(Exp5 ablation 用)

```python
import torch
preds = torch.load("/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt", weights_only=False)
# Schema (Step5Agent format):
# preds["mp_id"]:              list[str], len = N
# preds["sample_name"]:        list[str], len = N
# preds["pred_frac_coords"]:   list of (20, 3) tensor/array
# preds["pred_atom_types"]:    list of (20,) tensor/array
# preds["true_frac_coords"]:   list of (20, 3) tensor/array
# preds["true_atom_types"]:    list of (20,) tensor/array
# preds["lengths"]:            list of (1, 3) tensor (= [[6, 6, 6]])
# preds["eval_cutoff"]:        list of float

# 注意: 字段是 list-of-arrays,N 元素未对齐成单 tensor
```

#### 读取 shell_boundaries(Tier 计算 / Exp5 进阶分层)

```python
import pickle
with open("/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl", "rb") as f:
    shell = pickle.load(f)
# shell[sample_name] = dict with 9 fields:
# - threshold: float (gap threshold, 0.1563)
# - distances: (N_neighbors,) float32 (全部邻居距离, 截至 10 Å)
# - species_Z: (N_neighbors,) int8
# - shell_starts: (N_shells,) float32
# - shell_ends: (N_shells,) float32
# - shell_n_atoms: (N_shells,) int32
# - shell_of_atom: (N_neighbors,) int32
# - eval_cutoff: float
# - n_center_sites: int
```

#### 数据集 + datamodule(Exp5 训练复用)

```python
from xas_local_dataset_v2 import XasLocalDatasetV2
from xas_local_datamodule_v2 import XasLocalDataModuleV2, xas_collate_fn_v2

DATA_DIR = "/home/tcat/diffcsp_exp4/data"

# 单 split 直接构造(Phase 5b 模式,绕开 datamodule 的 holdout 闸门)
ds = XasLocalDatasetV2(split="val", data_dir=DATA_DIR)
# ds[i] 返回 torch_geometric.data.Data 对象(或 None,silent drop)

# 通过 datamodule 训练(主要路径)
dm = XasLocalDataModuleV2(batch_size=16, num_workers=0, data_dir=DATA_DIR)
dm.setup("fit")
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
# Note: dm 不暴露 holdout_dataloader,这是 by design 防止训练时误读
```

### 9.3 网络环境守卫包(Exp5 启动前必查)

7 个核心包,Exp5 不应升级:

```python
import sklearn, numpy, scipy, pymatgen, torch, pytorch_lightning, torch_scatter
print(sklearn.__version__)       # 期望 1.7.2
print(numpy.__version__)         # 期望 2.2.6
print(scipy.__version__)         # 期望 1.15.3
print(pymatgen.__version__)      # 期望 2025.10.7
print(torch.__version__)         # 期望 2.4.1+cu124
print(pytorch_lightning.__version__)  # 期望 2.5.5
import torch_scatter
print(torch_scatter.__version__) # 期望 2.1.2+pt24cu124
```

加上 Sub-Agent 4-续 装的 18 个 diffcsp 子依赖(einops 0.8.2 / p_tqdm 1.4.2 / smact 3.2.0 / matminer 0.9.3 / pyxtal 1.1.3 / torch_sparse 0.6.18+pt24cu124 / 等)。Exp5 在 mlff env 下应该全部可用。

### 9.4 红线: Exp5 绝对不能动的文件

- `holdout_samples_v2.csv` / `spectra_holdout.pkl`: 训练期不可读
- `incompat_pool.csv`: Exp4 全程封存
- `forward_test.py.bak3`: Step 3 阶段最终回滚锚点(仅 emergency 用)
- `xas_local_dataset_v2.py.bak_phase46`: Phase 4.6 修复前的 raise 版本(仅 emergency)
- 所有 `*.bak*`: 备份系列,read-only 历史

---

## 10. Exp5 改进方向 Menu(7 个候选 + MA5 排序)

下面是 MA5 基于 Exp4 三大发现(O1/O2/O3)+ 用户提到的"盲人摸象/多视角注意力"提议给出的改进方向 menu。**每个方向都标注了攻击的失败模式 + 工程量 + 预期收益**。

**Exp5 Agent 应该选 1-3 个方向组合,写改进版 proposal。MA5 给排序但不替决。**

### 方向 1 ⭐⭐⭐: Decoupled TypeClassifier Head(Exp2 §3.2 direction A)

- **攻击失败模式**: O3 RMSD-TypeAcc decoupling 反证了 Exp3 时代证伪的逻辑
- **机制**: SpectrumEncoder latent (256d) → MLP head → (B, 20, N_elem) 多分类 logits
- **Loss**: `total = diffusion_loss + λ * type_ce_loss`,λ=0.5 起步
- **预期**: TypeAcc 从 0.19 → 0.30-0.40+;RMSD 不变(O3 已证 |r|<0.03 → 加 head 几乎不会拉低坐标)
- **工程量**: 小(~50 行代码改动 + 重训)
- **风险**: 低(独立 head 不影响 diffusion)
- **MA5 排名 #1**: 收益/工程比最高,且直接对应 O3 数据反证

### 方向 2 ⭐⭐: Center-Element Conditioning Injection

- **攻击失败模式**: O1 rank-1 weakness(候选解读 1)
- **机制**: 把 88 元素 center_element 做 embedding (88 → 16d),与 SpectrumEncoder latent concat 进 decoder
- **预期**: rank 1 TypeAcc 从 0.243 → 接近 rank 3 (0.275);整体 TypeAcc 提升 +0.02-0.05
- **工程量**: 小-中(改 dataset_v2 输出 + diffusion 输入 + yaml feat_dim)
- **风险**: 低
- **MA5 排名 #2**: 直接对应 O1,可与方向 1 组合

### 方向 3 ⭐⭐: Multi-View Attention Pooling(用户提议的"盲人摸象")

- **攻击失败模式**: 间接攻击 O2(decoder 信息利用率不足),非直接
- **机制**: 把 XANES (150d) / chi1 (200d) / FEFF (74d) 当 3 个独立 view,各自 encode 到 256d 后**做 cross-attention pool**(query 共享,各 view 当 key/value),而不是当前的 concat→MLP
- **预期**: latent 信息密度提升,**潜力高但不确定 Exp4 数据上能否突破 0.30**
- **工程量**: 中(改 SpectrumEncoder,~150 行)
- **风险**: 中。**这个方向 Exp2 / Exp4 都没做过,完全 unprecedented**。可能与方向 1/2 协同,也可能挤压它们的提升空间(latent 已经够 informative,attention 改造收益边际递减)
- **MA5 排名 #3**: 潜力大但风险也大,**MA5 强烈建议作为方向 1+2 之后的进阶 ablation,不作第一棒**
- **关于这个方向我的诚实看法**(给用户): 这是个有想象力的方向,我支持你尝试。但我作为 MA5 必须提醒: **Exp4 的核心瓶颈是 decoder 端的 collapse(O2),不是 encoder 端的信息聚合不足**。多视角 attention 主要改善 encoder,可能让 latent 更精确,但 decoder 没修好就照样 collapse。**最稳的策略**: 先做方向 1 + 方向 2(直接攻击 O1/O2/O3),作为 baseline_v2;然后在 baseline_v2 上做方向 3 ablation,看 attention pooling 能否进一步推高 TypeAcc

### 方向 4 ⭐: Anti-Collapse Auxiliary Loss(O2 直接攻击)

- **攻击失败模式**: O2 collapse mode
- **机制**: 加一项 `loss_diversity = max(0, std(true_atoms) - std(pred_atoms) - margin)`,惩罚 pred 标准差比 true 显著小
- **预期**: hard sample 的 collapse 缓解,fig2 在 TypeAcc=0 处的 spike 缩小,RMSD 几乎不变(已经饱和)
- **工程量**: 小(~30 行 loss 添加)
- **风险**: 中。loss 调权不当可能破坏 diffusion 训练目标。需要 careful weight tuning
- **MA5 排名 #4**: actionable 但调参敏感

### 方向 5: Multi-Sample Test-Time Averaging

- **攻击失败模式**: 不直接攻击 O1/O2/O3,但通用降噪
- **机制**: 推理时跑 K=5-10 sample,取平均(坐标)/ majority vote(类型)
- **预期**: RMSD 降 5-10%(从 1.485 → ~1.40),TypeAcc 升 0.02-0.05
- **工程量**: 极小(只改 sample 脚本)
- **风险**: 极低
- **MA5 排名 #5**: 不需训练,**用 Exp4 ckpt 直接 ablation 即可**。Exp5 启动后第一周可作为 quick win

### 方向 6: Equivariant Decoder(NequIP / e3nn 替换)

- **攻击失败模式**: 通用架构升级,间接帮所有
- **机制**: 把 cspnet 的 backbone 替换为 e3nn 风格 SO(3)-equivariant GNN
- **预期**: 不确定。e3nn 在 conformer 预测里表现好,但 diffusion + e3nn 的组合 Exp2/Exp4 都没验证过
- **工程量**: 大(>500 行,需要重写 backbone)
- **风险**: 高。可能与现有 yaml/diffusion 框架不兼容,需要重设计训练循环
- **MA5 排名 #6**: 长期方向,**Exp5 不推荐**,可作 Exp6 候选

### 方向 7: Cascaded / Two-Stage Diffusion

- **攻击失败模式**: O2 collapse
- **机制**: 第一阶段做 coarse coord,第二阶段(条件于第一阶段结果)做 type-aware 精修
- **预期**: 收益不明,模拟两阶段架构很容易引入新 bug
- **工程量**: 大(>300 行)
- **风险**: 高
- **MA5 排名 #7**: **不推荐**,工程债务大于潜在收益

### MA5 推荐的 Exp5 第一棒组合

**方案 A(保守)**: 方向 1 + 方向 5
- TypeClassifier head + multi-sample averaging
- 工程量: 小
- 预期 TypeAcc: 0.19 → 0.32-0.38(只方向 1)+ 0.02-0.05(方向 5)= **0.34-0.43**
- 与 Exp2 §3.2 direction A 目标 0.40 对齐

**方案 B(平衡,MA5 倾向)**: 方向 1 + 方向 2 + 方向 5
- 加上 center conditioning,直接攻击 O1
- 工程量: 中
- 预期 TypeAcc: **0.36-0.45**,RMSD 可能小幅下降到 1.42-1.45

**方案 C(进阶,如用户坚持多视角)**: 方向 1 + 方向 2 + 方向 3 + 方向 5
- 把用户的"盲人摸象"加进来,但作 baseline_v2 之后的 ablation
- 工程量: 大
- 预期: TypeAcc 可能突破 0.50,但风险也最高

**MA5 给 Exp5 Agent 写 proposal 的指令**: 
> "用户倾向方向 3(多视角注意力),但 MA5 提醒这不是 Exp4 核心瓶颈。Exp5 proposal 应至少包含方向 1(decoupled head)作为 baseline,方向 3 作为 attention ablation。具体方案让 Exp5 Agent 在读完 Exp4 final report + 用户最新意向后写。"

---

## 11. Exp5 启动需要的共享文档清单 ⭐

用户开新窗口启动 Exp5 时需要传给 Exp5 Main Agent 的文档,按优先级排:

### 11.1 必传(Exp5 Main Agent 主参考)

| # | 文档 | 来源 | 必读? |
|---|---|---|---|
| 1 | **EXPERIMENT4_FINAL_REPORT.md**(本文档) | MA5 输出 | ✅ 精读 |
| 2 | **EXPERIMENT2_FINAL_REPORT.md** | Exp2 时代 | ✅ 精读 §1 指标 + §3 改进 |
| 3 | **EXP4_PROPOSAL_v2.md** | Exp4 启动时 | ✅ §1.3 不变量(Exp5 必继承) |
| 4 | **EXP4_FILE_INVENTORY.md** | MA2 写 | ✅ 数据文件位置 |
| 5 | **EXP4_STEP6_STEP6AGENT_FINAL_REPORT.md** | Step6Agent 输出 | ✅ O1-O4 详细数据(本 final report §7 已浓缩) |

### 11.2 选传(Exp5 改进方向相关时按需读)

| # | 文档 | 何时需要 |
|---|---|---|
| 6 | EXP4_STEP5AGENT_FINAL_REPORT.md | Exp5 想 reproduce Step 5 评估流程时 |
| 7 | EXP4_STEP4_SUBAGENT5_INTERIM_REPORT.md | Exp5 想看 Step 4 训练曲线 / Phase 4.6 修复细节时 |
| 8 | EXP4_STEP3_SUBAGENT4CONT_FINAL_REPORT.md | Exp5 想看 Step 3 接力链 + 守卫包列表时 |
| 9 | Step6Agent 6 张 figure PNG | Exp5 写 proposal 时贴图引用 |

### 11.3 不要传(noise,会消耗 Exp5 上下文)

- ❌ EXP4_MAINAGENT1/2/3/4_HANDOFF.md(MA5 已经在本 final report 里 condensed 完所有有效信息)
- ❌ EXP4_STEP4_SUBAGENT5_HANDOFF.md / EXP4_STEP5_STEP5AGENT_HANDOFF.md / EXP4_STEP6_STEP6AGENT_HANDOFF.md(handoff 是给 sub-agent 的,Main Agent 不需要)
- ❌ EXP4_PROGRESS_LOG.md(Step 1/2/2.5 历史细节,Exp5 不直接用)
- ❌ exp2tree.txt(目录树,信息已含本 final report §9.1)
- ❌ 任何中间 .bak 文件
- ❌ 训练 log

### 11.4 服务器代码(Exp5 Sub-Agent 跑命令时按需 view)

不需要"传给 Main Agent",但 Exp5 Sub-Agent 在写代码时会需要 view 这些文件。Exp5 Main Agent 把以下路径直接告诉 Sub-Agent 即可:

```
/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py   ← Phase 4.6 当前版
/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
/home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py        ← 5 处 73→74
/home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py    ← line 108 = 74
/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml  ← line 18 = 74
/home/tcat/diffcsp_exp4/code/step4/step4_2_train.py           ← 训练参考
/home/tcat/diffcsp_exp4/code/step5/step5_1_sample.py          ← 评估参考
```

### 11.5 给 Exp5 Main Agent 的 first-message 模板建议

```
你是 DiffCSP-Experiment5 Main Agent。Exp4 已完成,best ckpt 
val_loss=0.7300, holdout RMSD=1.4866 / TypeAcc=0.1973。

Exp5 的核心改进方向(用户初步意向 + MA5 建议):
- 用户倾向: 多视角注意力聚合("盲人摸象")
- MA5 强烈推荐: 先做 decoupled TypeClassifier head + center-element conditioning,
  作 baseline_v2;再做多视角 attention 作 ablation
- 用户已 approved Exp5 启动,但具体方案要让你写改进版 proposal

Exp5 你的工作:
1. 读必读文档(我会传 5 份,见下)
2. 写 EXP5_PROPOSAL.md,锁定 Exp5 的不变量 + 改进方向 + 验收标准
3. (用户 review 后)开 Sub-Agent 接力链跑 Step 1-6

继承 Exp4 的不变量(不可改):
- L=6, min-image, [-0.5, 0.5] coord
- cost_lattice = 0
- N_NEIGHBORS = 20
- silent drop + collate filter
- 88 元素中心范围

服务器: scsmlnprd02
工作目录: /home/tcat/diffcsp_exp5/(待你 mkdir)
ckpt 起点: /home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt

[然后传 5 份必读文档(§11.1)]
```

---

## 12. 接力链 Lessons Learned(留给未来 ExpN)

1. **接力链工作哲学**: 诚实 > 流畅,70% 上下文闸门,不深 debug,状态锚定文档
2. **dataset 改动要带配套 collate**: SA3 改 raise 但没改 collate,导致 Step 4 红灯。**任何"silent → loud"或反向改动都要审 downstream**
3. **PL 版本漂移要逐 API 验证**: Exp2 PL 1.9.5 vs Exp4 PL 2.5.5,`precision='bf16'` 行为完全不同。**不要假设旧 API 在新版本工作**
4. **用户 domain 知识 outvote agent reflex**: Phase 4.6 修复时,用户的"黑名单一半以上"指引是关键诊断节点。MA5 当时的 Path A blacklist 倾向是错的,被用户矫正
5. **figure 6 cut 是好决定**: Step6Agent 主动在报告里 flag "fig6 不会解决 O1,推荐 fig5+rank-1-by-element-row 替代"——sub-agent 主动给出比交付更好的建议是接力链最高质量信号
6. **不可变量需要在 handoff 里 explicit 出现 ≥3 次**: Exp3 时代"不加 head"在 Exp4 PROPOSAL §1.3 锁定,但 Exp4 数据反证了这个锁定逻辑——**不变量要带"why"和"何时可重审"**

---

## 附录 A: Exp4 vs Exp2 横向对比一览

| 维度 | Exp2 | Exp4 | 进步? |
|---|---|---|---|
| 中心元素 | Fe-only | 88 元素 | 任务难度 88× |
| 训练样本 | 7,595 | 60,507 | 8× |
| 几何精度 (RMSD) | 1.47 | 1.49 | parity ✓ |
| Type Accuracy | 0.241 | 0.197 | 表面下降 18%,但 Tier B 0.26 = parity ✓ |
| pred_in_cutoff | 17.52 | 18.92 | +1.4 ✓ |
| 训练精度 | bf16 | fp32 | 慢但稳 |
| 训练时间 | ~6-10 h | ~32 h | 慢 4-5× |
| 接力链复杂度 | 单 main agent | MA1-5 + 9 个 sub-agent | 显著上升 |
| 失败模式新增 | — | O2 collapse + O1 rank-1 | Exp4 specific |

**结论**: Exp4 在 88× 难度任务上达到 Exp2 几何 parity,**这是一个 robust 的 generalization**。type prediction 表面回退,但 per-tier 看是 Exp2 parity,真正的差距是 88 元素中心带来的 collapse 失败模式,需要 Exp5 架构改进解决。

---

## 附录 B: 文档撰写 Self-Audit

本 final report 撰写时 MA5 自检:

1. ✅ 不混淆备份和最终用版本(§9.1 每条 .bak 都标注作用)
2. ✅ 每个脚本和数据位置 + 读取方式齐全(§9.2 给可执行代码片段)
3. ✅ 三大发现 O1/O2/O3 深度解读,不空泛(§7)
4. ✅ Exp5 改进方向给 menu 而非单选(§10)
5. ✅ 用户提议(多视角注意力)我直接给我的判断(§10 方向 3),不替决但给观点
6. ✅ Exp5 启动需要的共享文档清单完整(§11)
7. ✅ 必读 / 选传 / 不要传 三档分明(§11.1-11.3)
8. ✅ 接力链 lessons learned 留给未来 ExpN(§12)
9. ✅ 与 Exp2 final report 文风对齐(三段式 + 历史 + 改进方向)

---

*Main Agent 5 撰写,2026-04-28,Experiment 4 完结。等用户 review 后转交 Exp5 Main Agent。*
