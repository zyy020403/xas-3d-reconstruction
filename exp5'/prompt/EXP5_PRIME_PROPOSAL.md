# EXP5_PRIME_PROPOSAL.md
# Exp5' Proposal — Physical Constraint Extension (from-scratch)

> **撰写者**: MA5(Exp5 v2 Main Agent,移交 Exp5'-MA 前最后一份 proposal)
> **日期**: 2026-05-01
> **状态**: 用户拍板 from-scratch 重训
> **接班**: Exp5'-MA(下一棒 Main Agent,以下也称 Exp5' MA)
> **取代**: EXP5_PROPOSAL_v2_AMENDED.md(归档)+ EXPERIMENT5_FINAL_REPORT_v1.md(归档)
> **位置**: 本 proposal + EXPERIMENT5_FINAL_REPORT_v2.md + EXP5_FILE_GUIDE_v2.md + EXP5_PRIME_MA_HANDOFF.md 是 Exp5' 启动 4 件套

---

## §0 执行摘要

### 0.1 Exp5 v2 状态(MA5 移交时)

- **训练**: SA2 epoch 484 + SA2'' 续训 epoch 529,best val_loss 0.7003
- **数学评估**: Multiset F1 0.1086 vs Exp4 0.0843(+28.8%)
- **物理评估(SA-METRICS-V3 新发现)**: **min_d gate pass rate 5-11%**(95% 样本预测原子两两距离 < 1.5 Å)
- **更深问题(SA-METRICS-V3 §1.3)**: 即使 gate-pass 5-11% 子集,shell-1 distance score = 0(模型不知第一壳层应在 ~ 2-3 Å)

### 0.2 Exp5' 任务

**from-scratch 重训**(用户决策,不 warm-start),纳入三件套物理 loss + Step 2.5 ground truth 真正进训练:
1. **min pairwise distance** ≥ 1.5 Å penalty(攻 95% 物理违反)
2. **shell distance loss**(攻 shell-1 distance = 0 问题,Exp4 的 ground truth 终于进训练)
3. **shell coordination count loss**(辅助,让 shell-1 配位数预测对)

**架构沿用 Exp5 v2**: MV-attention encoder + center embedding + cost_density 0.2(三件保留,不 ablation)

**估时**: ~32-40h from-scratch 训练 + ~3.5h sample + ~2h metrics + ~2h figure + ~3h final report ≈ **40-50h 总,跨 6-8 个对话窗口**

### 0.3 success criterion(锁定)

| 指标 | Exp4 | Exp5 v2 | **Exp5' 目标** | 强度 |
|---|---|---|---|---|
| min_d gate pass rate | (未测,推测低)| **5-11%** | **≥ 80%** | 🟢 必须 |
| shell-1 distance score | (未测)| **0.0000** | **≥ 0.50** | 🟢 必须 |
| 复合总分 | (未测)| **0.005-0.011** | **≥ 0.40** | 🟢 必须 |
| Multiset F1(辅助)| 0.0843 | 0.1086 | ≥ 0.10(不退步)| ⚠️ 软指标 |
| RMSD(辅助)| 1.4849 | 1.4954 | ≤ 1.55(不大幅退步)| ⚠️ 软指标 |

任一 🟢 必须项不达标 → Exp5' 失败,转 Exp6 大改架构。

---

## §1 不变量(Exp5' 完全沿用,不动)

### 1.1 数据相关

| 项 | 值 | 来源 |
|---|---|---|
| split | 60507/7624/4481/3025 | Exp4 |
| 中心元素 | 88 元素(Z ∈ [2, 94] 实测) | Exp4 |
| L | 6.0 Å | Exp2 step4d |
| coord 系 | [-0.5, 0.5] + min-image | Exp2 step4d |
| N_NEIGHBORS | 20 | Exp4 |
| 邻居搜索半径 | 10.0 Å | Exp2 |
| FEFF feature 维度 | 74 | Exp4 |
| **shell_boundaries.pkl** | **`/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl`(387 MB,md5 cf2050e4...)** | **Exp4 Step 2.5,Exp5' 训练首次 inject 进 batch** |

### 1.2 架构相关(Exp5 v2 沿用)

| 项 | 值 |
|---|---|
| MV-attention | num_heads=4, residual_alpha=0.5 固定 |
| Center embedding | n_center_elements=95, center_emb_dim=16 |
| latent_dim(top-level) | 272 |
| decoder.latent_dim | 528 |
| `cost_density` | **0.2** 沿用(不再调) |
| `cost_lattice` | 0.0 沿用 |
| `cost_coord` | 1.0 沿用 |
| `cost_type` | 1.0 沿用 |
| precision | fp32 |

### 1.3 训练超参数(Exp5 v2 沿用)

| 项 | 值 |
|---|---|
| optimizer | Adam, lr=1e-4 |
| batch_size | 16 |
| gradient_clip_val | 1.0 |
| max_epochs | 500(若不收敛 Exp5' MA 决议续训)|
| early_stop patience | 30 |
| save_top_k | 1(用复合 best ckpt criterion 不只是 val_loss,详 §3.4)|
| num_workers | 0(pymatgen SGA worker safety) |
| LR scheduler | CosineAnnealingLR T_max=500, eta_min=1e-6(沿用 v2 SA1' 设计)|

### 1.4 红线(Exp5' 全程不动)

| 红线 | 来源 |
|---|---|
| ❌ 不动 holdout(永久封存) | Exp4 |
| ❌ 不动 incompat_pool.csv | Exp4 |
| ❌ 不升级 7 守卫包 | Exp5 全程 |
| ❌ 不动 cspnet.py 等 Exp4 backbone | Exp5 全程 |
| ❌ 不修 Phase 6.5 hardcoded fp32(永久 SKIPPED-by-design)| v1 SA1 OUTPUT §5.7 |
| ❌ 不动 Exp5 v2 的 .frozen ckpt(2 个永久 safety net)| MA5 设 |

---

## §2 三件套物理 loss(Exp5' 核心改动)

### 2.1 主线 — `cost_pairwise_min` (新加)

**机制**: 惩罚同一 sample 内 20 原子两两 cartesian 距离 < 1.5 Å 的 pair。

**精确公式**:

```python
def _pairwise_min_distance_penalty(pred_frac_coords, num_atoms, L=6.0, threshold=1.5):
    """
    pred_frac_coords: (Σ N_i, 3) flat batch frac coords
    num_atoms: (B,) 每 sample 原子数(常 = 20)
    """
    total_loss = 0.0
    n_samples = 0
    start = 0
    for ni in num_atoms:
        ni = int(ni)
        coords_i = pred_frac_coords[start:start+ni] * L  # (ni, 3) cartesian
        # min-image: 因为 frac in [-0.5, 0.5],cartesian 已经 in [-L/2, L/2],
        # 但 box 是周期的,两两距离要按 min-image 取最近镜像。
        # 实际操作:先算 frac 差,套 % 1.0 - 0.5,再乘 L
        frac_i = pred_frac_coords[start:start+ni]
        diff_frac = frac_i.unsqueeze(0) - frac_i.unsqueeze(1)  # (ni, ni, 3)
        diff_frac = diff_frac - diff_frac.round()   # min-image to [-0.5, 0.5]
        diff_cart = diff_frac * L                    # (ni, ni, 3)
        d = diff_cart.norm(dim=-1)                   # (ni, ni)
        # mask 自身 + upper triangle
        mask = torch.triu(torch.ones_like(d), diagonal=1).bool()
        d_pairs = d[mask]                            # (ni*(ni-1)/2,)
        violation = torch.relu(threshold - d_pairs)  # (m,)
        total_loss = total_loss + (violation ** 2).mean()
        n_samples += 1
        start += ni
    return total_loss / max(n_samples, 1)
```

**起步 λ**(用户拍板 SA-METRICS-V3 §4.1 的精化建议):

```yaml
cost_pairwise_min: 1.0   # Exp5' 起步,前 10 epoch 监控调整
```

**调度**(Exp5'-MA 监控 epoch 0-50):
- epoch 0-2: λ=1.0(初始尝试)
- epoch 3-5: 监控 violation rate
  - violation 单调下降到 < 50%: λ 维持 1.0
  - violation 卡在 > 70%: λ ramp 到 2.0(完全重合 sample 需要更强 push)
  - RMSD 飙升 > SA2 baseline + 10%: λ 减半,重启
- epoch 5+ violation < 30%: λ 可降回 0.5

### 2.2 主线 — `cost_shell_dist` (新加)

**机制**: 让模型预测的 shell-1 / shell-2 mean radial distance 接近 Exp4 Step 2.5 ground truth。

**前置工作**: 把 `shell_boundaries.pkl` 的 per-sample 字段 inject 进 dataset/datamodule 输出的 batch。具体:
- `xas_local_dataset_v2.py` `__getitem__` 中加一段:加载 `shell_boundaries.pkl` 的 sample_name 对应记录,提取 true_shell1_distances / true_shell2_distances / true_shell1_n / true_shell2_n / true_shell1_species / true_shell2_species,塞进 Data 对象
- `xas_local_datamodule_v2.py` collate 加 LongTensor / FloatTensor 字段

**精确公式**:

```python
def _shell_distance_loss(pred_frac_coords, num_atoms,
                         true_shell1_d_mean, true_shell2_d_mean, has_shell2,
                         L=6.0, threshold_gap=0.1563):
    """
    用 Step 2.5 gap 算法对 pred 切壳 → 算 pred shell-1 mean distance →
    对比 true_shell1_d_mean(从 shell_boundaries.pkl 读)→ MSE。
    shell-2 同理,但加 has_shell2 mask(true 没 shell-2 的 sample 不算)。
    """
    total_loss = 0.0
    n_active = 0
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        coords_i = pred_frac_coords[start:start+ni] * L
        radial = coords_i.norm(dim=1)
        sorted_d, _ = radial.sort()

        # gap-based shell split
        if len(sorted_d) >= 2:
            gaps = sorted_d[1:] - sorted_d[:-1]
            boundaries = (gaps > threshold_gap).nonzero(as_tuple=True)[0]
            if len(boundaries) >= 1:
                shell1_end = int(boundaries[0].item()) + 1
                pred_s1_d_mean = sorted_d[:shell1_end].mean()
                # shell-1 distance loss
                total_loss = total_loss + (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2

                if len(boundaries) >= 2 and bool(has_shell2[i]):
                    shell2_end = int(boundaries[1].item()) + 1
                    pred_s2_d_mean = sorted_d[shell1_end:shell2_end].mean()
                    total_loss = total_loss + (pred_s2_d_mean - true_shell2_d_mean[i]) ** 2

                n_active += 1
        start += ni
    return total_loss / max(n_active, 1)
```

**起步 λ**:

```yaml
cost_shell_dist: 0.5   # Exp5' 起步
```

如 epoch 30+ shell-1 distance score 仍 < 0.30,Exp5' MA 监控决策 ramp 到 1.0。

### 2.3 辅助线 — `cost_shell_count` (新加)

**机制**: 让模型预测的 shell-1 / shell-2 配位数(原子数)接近 ground truth。

**注**: 配位数是离散整数,用 MSE soft 监督即可(模型输出连续 pred shell count via gap 算法,做 floor / round 比较麻烦 → 直接 MSE on float)。

**精确公式**:

```python
def _shell_count_loss(pred_frac_coords, num_atoms,
                     true_shell1_n, true_shell2_n, has_shell2,
                     L=6.0, threshold_gap=0.1563):
    """
    用 gap 算法切 pred shell → 算 shell-1 / shell-2 count → MSE。
    """
    total_loss = 0.0
    n_active = 0
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        coords_i = pred_frac_coords[start:start+ni] * L
        radial = coords_i.norm(dim=1)
        sorted_d, _ = radial.sort()

        if len(sorted_d) >= 2:
            gaps = sorted_d[1:] - sorted_d[:-1]
            boundaries = (gaps > threshold_gap).nonzero(as_tuple=True)[0]
            if len(boundaries) >= 1:
                pred_s1_n = float(int(boundaries[0].item()) + 1)
                total_loss = total_loss + (pred_s1_n - float(true_shell1_n[i])) ** 2

                if len(boundaries) >= 2 and bool(has_shell2[i]):
                    pred_s2_n = float(int(boundaries[1].item()) - int(boundaries[0].item()))
                    total_loss = total_loss + (pred_s2_n - float(true_shell2_n[i])) ** 2

                n_active += 1
        start += ni
    return total_loss / max(n_active, 1)
```

**起步 λ**:

```yaml
cost_shell_count: 0.2   # Exp5' 起步,辅助
```

### 2.4 总 loss

```python
loss = (cost_lattice * loss_lattice              # 0.0
      + cost_coord   * loss_coord                # 1.0,沿用
      + cost_type    * loss_type                 # 1.0,沿用
      + cost_density * loss_density              # 0.2,沿用 v2
      + cost_pairwise_min * loss_pairwise_min    # 1.0 起步,新
      + cost_shell_dist   * loss_shell_dist      # 0.5 起步,新
      + cost_shell_count  * loss_shell_count)    # 0.2 起步,新
```

### 2.5 已知 trade-off / 风险

1. **gap 算法在重合数据上 ill-defined** — 训练初期(epoch 0-10)pred 大量重合,gap 切壳输出 garbage shell-1 distance。**这没关系**,只要 pairwise penalty 把原子先分开,gap 切壳就稳定。但 epoch 0-5 的 shell_dist / shell_count loss 数值可能很大或很小,Exp5' MA 不要慌张。

2. **shell-2 缺失样本** — 部分 sample 真值只 1 个壳(`has_shell2=False`)。loss 函数已 mask,不算 shell-2 loss。

3. **shell_boundaries.pkl inject 进 dataset 增加 I/O 开销** — 387 MB pkl 一次性 load(SA-EXP5'-train 在 datamodule.setup() 加载到内存 dict),per-sample lookup 只是 dict access,~ 0 开销。

---

## §3 Exp5' 评估改动

### 3.1 主指标 — 复合分(沿用 SA-METRICS-V3)

`step5_3_composite_score.py` 已写 + dry-run 验证,Exp5' 训完直接重跑,**不改算法**。

主信号锁定阈值(用户拍板):
- 🟢 GREEN: gate pass rate ≥ 80% + 复合分均值 ≥ 0.40 + shell-1 distance score ≥ 0.50
- ⚠️ AMBER: 60-80% gate pass + 复合分 0.20-0.40
- ❌ RED: < 60% gate pass(Exp5' 失败,转 Exp6)

### 3.2 监控指标(Exp5'-MA 训练时 epoch-level log)

```
[exp5_prime] epoch=N
  val_loss        = X.XX
  val_coord_loss  = X.XX
  val_type_loss   = X.XX
  val_density_loss= X.XX
  val_pairwise_min_loss = X.XX  (target: → 0)
  val_shell_dist_loss   = X.XX  (target: → 0)
  val_shell_count_loss  = X.XX  (target: → 0)
  val_min_d_mean        = X.XX  (target: increasing toward > 2.0)
  val_min_d_p10         = X.XX  (target: > 1.5 by epoch 50)
  val_gate_pass_rate    = XX.X% (target: > 80% by end)
  val_overlap_rate (min_d<0.1) = XX.X%  (target: → 0% by epoch 20)
  λ_pairwise = X.X
```

注: training-time monitor 用简化版 metrics(不跑 Hungarian / 不算 multiset),只算几个简单数(min_d / gate_pass / pairwise loss),~ < 1s overhead per epoch。完整 7 项复合分由 step5_3 在 sample 后跑。

### 3.3 best ckpt selection criterion(用户拍板)

不能单看 val_loss(SA2 已证明 val_loss 0.7003 时 95% 物理违反)。Exp5' best ckpt 综合 3 项:

```python
score_for_ckpt = (
    α * (1.0 - val_loss / 1.0)           # α = 0.2
  + β * val_gate_pass_rate                # β = 0.5  ← 物理可用,最高
  + γ * (1.0 - val_pairwise_min_loss)    # γ = 0.3
)
# 选最大 score 的 ckpt 作 best。可同时保 last.ckpt(训练结束的)。
```

**实施方式**: PL 自定义 `ModelCheckpoint` callback,monitor 一个新 metric `val_composite_ckpt_score`(由 Exp5'-MA 在 LightningModule 算并 log)。

---

## §4 Exp5' 实施步骤(给 Exp5'-MA 写 SA handoff 用)

### 4.1 Step 4.1 — Dataset / Datamodule inject shell_boundaries.pkl

**新文件**: `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` (fork from v2,加 shell_boundaries 字段)

**改动概要**:
- `__init__`: 加 `shell_boundaries_path` 参数,load 387 MB pkl 到 self
- `__getitem__`: 加从 self.sb[sample_name] 提取 5 字段 → Data 对象

**新文件**: `xas_local_datamodule_v2.py` (fork,加 collate 5 字段)

### 4.2 Step 4.2 — diffusion_w_type_xas.py 加 3 个 loss 函数

**新文件**: `/home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py`(fork from v2)

**改动**:
- 加 `_pairwise_min_distance_penalty` static method(§2.1)
- 加 `_shell_distance_loss` method(§2.2)
- 加 `_shell_count_loss` method(§2.3)
- forward() 内调用 + 加进 total_loss
- 总 loss output dict 加 5 字段(loss_pairwise_min / loss_shell_dist / loss_shell_count / val_gate_pass_rate / val_min_d_mean)

### 4.3 Step 4.3 — yaml 加 3 字段

```yaml
# 加在 step3/conf_xas/model/diffusion_xas.yaml
cost_pairwise_min: 1.0   # Exp5' 起步
cost_shell_dist:   0.5   # Exp5' 起步
cost_shell_count:  0.2   # Exp5' 起步
```

### 4.4 Step 4.4 — train.py 加 ckpt selection callback

新自定义 callback 算 `val_composite_ckpt_score`,monitor 这个 metric 而不是 val_loss。

### 4.5 Step 4.5 — forward_test 改 Phase 6.7 测三个新 loss

**新 phase**: Phase 6.7 — Exp5' 三个新 loss 项 sanity:
- pairwise_min_loss 在 dummy batch 上有意义(min_d=2.0 dummy → loss=0;min_d=0.5 dummy → loss > 0)
- shell_dist_loss 在 dummy batch 上 finite
- shell_count_loss finite
- yaml cost_pairwise_min/shell_dist/shell_count 加载到模型

### 4.6 Step 4.6 — smoke test

跑 2 epoch × 10 batch,确认 6 个 loss(原 4 + 新 3 减 lattice = 6 active)都有数 + 无 NaN/Inf。

### 4.7 Step 4.7 — 启动 ~32-40h from-scratch 训练

Exp5'-MA 写 SA-EXP5'-train handoff(类比 SA2'),启动训练。

### 4.8 Step 4.8 — Sample + step5_3 复合分

Exp5'-MA 写 SA-EXP5'-sample handoff,sample val + test → 跑 step5_3 → verdict。

### 4.9 Step 4.9 — Exp5' final report

verdict 成功 → SA-EXP5'-figure 6 张(Exp5' vs Exp5 vs Exp4 三方对照)+ Exp5' final report。
verdict 失败 → 转 Exp6 proposal。

---

## §5 Exp5' 工作目录决策

**新建独立目录**: `/home/tcat/diffcsp_exp5_prime/`

理由:
- Exp5 v2 全部 .frozen ckpt + log + predictions 永久保留作历史 baseline,不混入 Exp5'
- Exp5' 改动 surface(dataset / datamodule / model / yaml)与 v2 不同,fork 进新目录避免误覆盖
- Exp5' 失败时,Exp5 v2 资产完整,可回溯做 Exp6 baseline

**不沿用 v1 那套 "重命名 + .bak_exp4 锚点"**:Exp5' 是从干净起点的 fork,不 surgery,直接 cp Exp5 v2 整个 code 树到新目录改。

```
/home/tcat/diffcsp_exp5_prime/
├── code/
│   ├── step2/spectrum_encoder.py       (cp from Exp5 v2,不改)
│   ├── step3/
│   │   ├── diffusion_w_type_xas.py     (cp from Exp5 v2 + 加 3 loss 函数)
│   │   ├── xas_local_dataset_v2.py     (cp from Exp5 v2 + 加 shell inject)
│   │   ├── xas_local_datamodule_v2.py  (cp from Exp5 v2 + 加 5 字段 collate)
│   │   ├── conf_xas/model/diffusion_xas.yaml  (cp + 加 3 cost 字段)
│   │   └── forward_test.py             (cp + 加 Phase 6.7)
│   ├── step4/
│   │   ├── step4_1_smoke_test.py       (cp + 加 6 loss 字段验证)
│   │   └── step4_2_train.py            (cp + ckpt selection callback,from-scratch 不 ckpt_path)
│   └── step5/
│       ├── step5_1_sample.py           (cp from Exp5 v2,不改)
│       ├── step5_2_compute_metrics.py  (cp,作历史对照)
│       └── step5_3_composite_score.py  (cp from Exp5 v2,不改)
├── checkpoints/   (空,等 Exp5' 训练)
├── data/          (软链接到 /home/tcat/diffcsp_exp4/data/)
└── logs/          (空,等 Exp5' 训练)
```

---

## §6 与 Exp5 v2 的关系

**Exp5' 与 Exp5 v2 是平行实验,不是替代**:
- Exp5 v2 在历史档案(MV-attention 架构 + 28.8% Multiset F1 改进 + 物理灾难 95% 违反这些事实永久保留)
- Exp5' 在 v2 基础上加物理 loss 重训,验证"加物理约束能否同时保住几何/类型 + 修物理"
- Exp5'' / Exp6(后续)可能做 architecture ablation(去 MV-attention 单看 center_emb + 物理 loss 效果)

Exp5' final report v3 完成后,Exp5 系列可正式 close 或继续开 Exp5''。MA5 不做这个决定,留给 Exp5'-MA / 用户。

---

## §7 给 Exp5'-MA 的 OPEN QUESTIONS

### Q1 — `_shell_distance_loss` / `_shell_count_loss` 在重合数据上的 numerical 行为

epoch 0-10 pred 大量重合时 gap 算法切壳输出 garbage,shell_dist_loss 数值可能很大或 0(取决于 gap 是否 > 0.1563 阈值)。Exp5'-MA 在写代码时考虑 numerical safety:
- 加 `eps=1e-6` 防 sqrt(0)
- gradient clipping=1.0 已有(沿用 v2)
- 监控前 5 epoch loss curve 防 NaN

### Q2 — shell ground truth inject 的 collate 兼容性

`xas_local_datamodule_v2.py` 的 collate 是 PyG `Batch.from_data_list`。新加的 5 字段(`true_shell1_d_mean: float`, `true_shell2_d_mean: float`, `has_shell2: bool`, `true_shell1_n: int`, `true_shell2_n: int`)是 per-sample 标量,collate 时拼成 `(B,)` 张量即可,不像 atom_types / frac_coords 那样 ragged。Exp5'-MA 测试时确认 collate 不报错。

### Q3 — Step 6 picker 如何处理(用户提到的"挑 78 / 7621 样本"问题)

SA-METRICS-V3 §2.2 提到 step6 picker "1% 选样率" 实质上是 cherry-pick。Exp5' final report 应明确记录全 7621 / 4481 的 gate pass 率 + 复合分,**不允许 picker subset 当 verdict**。step6 picker 脚本 `compare_spectra_v3.py` / `pick_samples_for_feff.py` 在 Exp5' 阶段不动,但 final report 不依赖其输出。

### Q4 — Exp5' 训练后,SA2 v2 ckpt 是否在 step5_3 上重 sample 一遍

SA-METRICS-V3 dry-run 是 100 样本,虽对决策足够,但 Exp5' final report 需要"v2 vs Exp5' 全量物理对照"才能说服 reviewer。**建议 Exp5'-MA 在 SA-EXP5'-sample 阶段顺便用 v2 epoch 529 ckpt 重 sample → 跑 step5_3 全量**(~ 3.5h 额外)。这样 final report 可以列出真正的 v2 全量 vs Exp5' 全量对照表。

### Q5 — Exp5' verdict failure 转 Exp6 的具体方向

如果 Exp5' 复合分 < 0.20 / gate pass < 60%(双 fail),说明加物理 loss 不足以救 v2 架构,Exp6 选项(从 Exp4 errata 2 §3.2 menu):
- 方向 4: distance-aware loss(把 shell distance 直接进 diffusion 噪声 schedule)
- 方向 9: Classifier-Free Guidance
- 新候选: 去 MV-attention(Exp5'' ablation)
- 新候选: Equivariant decoder(e3nn,长线)

具体由 Exp5'-MA 在 verdict 后写 Exp6 proposal,本 proposal 不 pre-commit。

---

## §8 风险声明

Exp5' 不保证成功。具体 risk:

1. **三件套 loss 平衡风险**: pairwise_min(λ=1.0)+ shell_dist(λ=0.5)+ shell_count(λ=0.2)三个新 loss 与原 4 个 loss(coord/type/density,total weight 2.2)总和 weight 1.7 vs 2.2,占比 43%。如训练中 RMSD/TypeAcc 大幅退步,需 λ 重调。**Exp5'-MA 要有 λ 调度心理准备**。

2. **shell ground truth inject 工程风险**: 新加 5 字段进 collate 是中等工程复杂度,如 dataset/datamodule 改动出问题,smoke test 应 catch。SA-EXP5'-train 启动前必跑 smoke + Phase 6.7 forward_test。

3. **gap 算法在 pred 上的不稳定性**: 训练初期 pred 重合,gap 切壳输出 garbage,shell loss 可能突然飙升或归零。保留 gradient_clip=1.0 + 监控前 5 epoch 数值。

4. **best ckpt selection criterion α/β/γ 调参风险**: 0.2/0.5/0.3 是 SA-METRICS-V3 推荐,但实际训练中可能某项 dominate。Exp5'-MA 监控 selection score 曲线,必要时调 weight。

如 Exp5'-MA 在训练前 5 epoch 看到任何上述异常,**停训,贴日志给用户 review**,不擅自改方案。

---

## §9 Exp5' 与之前的接力链关系

```
Exp4(MA1-MA5)→ Exp5 v1(被 kill)→ Exp5 v2(MA5 + SA1'-SA3' + SA-METRICS-V3)→ Exp5'(Exp5'-MA)
                                                                                        ↓
                                                                          Exp6 / Exp5'' / Exp5' final
```

Exp5'-MA 是 Exp5 系列第 3 任 Main Agent(v1 MA → v2 MA5 → Exp5'-MA)。

---

*MA5 撰写,2026-05-01,基于 SA-METRICS-V3 95% 物理违反诊断 + 用户 from-scratch 决策 + Exp4 Step 2.5 ground truth 真正进训练设计。Exp5'-MA 接手见 EXP5_PRIME_MA_HANDOFF.md。*
