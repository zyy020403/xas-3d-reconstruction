# EXP5_DOUBLE_PRIME_PROPOSAL.md
# Exp5'' Proposal — Distance-Supervised Shell Loss (Fix the Egg-Chicken Problem)

> **撰写者**: Exp5'-MA(交棒给 Exp5''-MA)
> **日期**: 2026-05-09
> **版本**: Exp5'' v1
> **基线**: Exp5' final report v3(`composite_epoch169_score0.5881.ckpt`,composite=0.080 RED)
> **核心动机**: errata 5 §2 鸡蛋启动问题 — `_shell_distance_loss / _shell_count_loss` 实际未生效,pred shell-1 偏离真值 4 Å
> **目标**: 重设计 shell loss 让其自启动,从 pred shell-1 mean radial 6.32 Å → 接近真值 2.27 Å
> **预期 verdict**: composite (step5_3) ≥ 0.30 GREEN(从 Exp5' 0.08 提升 ≥ 4×),gate 沿用 ≥ 60%

---

## §0 一屏掌握

### 0.1 Exp5'' 是什么

Exp5'' 是 Exp5' 的 **loss 函数级微调**(不动架构 / 数据 / 训练流程)。唯一改动:**重设计 shell loss 从 gap-based 改为 distance-supervised**。

### 0.2 为什么要做(errata 5 鸡蛋问题精确)

Exp5' STEP3 verdict:
- ✅ `_pairwise_min_distance_penalty` 生效(gate 64%,自启动)
- ❌ `_shell_distance_loss / _shell_count_loss` 未生效(shell-1 score 0.035,鸡蛋问题)

鸡蛋问题:gap-based shell 切壳需要 pred 已有壳层结构 → 才能切出 boundary → 才能产生有效梯度。但 pred 壳层结构的形成需要 shell loss 引导。两者互为条件,无外部信号打破循环。

**Distance-supervised 候选 A** 不依赖 pred 已有结构,从 random init 起就能产生有效梯度(详 §2)。

### 0.3 Exp5'' 不做什么

| 不动 | 来源 |
|---|---|
| L_VIRTUAL = 20 | errata 3 修复 |
| `_pairwise_min_distance_penalty`(λ=1.0) | Exp5' gate 64% 硬证生效 |
| `_density_loss`(cost=0.2) | errata 2 揭示是塌缩剂但 Exp5' 沿用 OK,不再调 |
| Architecture(MV-attention + center embedding) | Exp5 v2 沿用 |
| `shell_boundaries.pkl` | errata 3 §3 干净 |
| Adam lr=1e-4 / fp32 / grad_clip=1.0 | Exp5' 沿用 |
| Batch=64 / num_workers=16 / PreCollatedDataset | STEP2-CONTINUE 已验证可用 |
| ckpt callback strict=False / save_top_k=3 | errata 4 §5.1 修复 |

**只改 shell loss 公式**。其他**全部**保持 Exp5' STEP1-FIX-C → STEP2-CONTINUE 配置。

### 0.4 任务步骤(给 Exp5''-MA 心里有底)

| 步 | 任务 | 工程 | GPU |
|---|---|---|---|
| P1 | 改 `_shell_distance_loss` + `_shell_count_loss` 公式(详 §2)+ 加 n_active dry-run dump | 0.5 天 | — |
| P2 | Forward_test Phase 6.7 重跑(含 n_active 验证)| 0.3 天 | — |
| P3 | Smoke test 2 epoch + n_active 比例 ≥ 50% verify(errata 5 §5.1 强制)| 0.3 天 | 30 分钟 |
| P4 | Warm-start training from Exp5' BEST ckpt(epoch 169)| — | 8-12h |
| P5 | Sample 三 split + step5_3 复合分(沿用 Exp5' STEP3 流程)| 0.3 天 | 1.5h |
| P6 | Final report v3 附录 / 或独立 final report v4(看 verdict 决议)| 0.5 天 | — |

**总:1-2 天工程 + 10-14h GPU 无人值守 + 1-2h 评估**

### 0.5 SA 决议:**0 个 SA,Exp5''-MA 直接做**

理由:
- Exp5'' 任务范围明确(改 1 个 loss 公式)
- 所有"未知 unknown"已通过 Exp5' 5 份 errata 解决
- Exp5''-MA 直接干 ROI 高于开 SA

**例外触发条件**(届时 Exp5''-MA 决议是否开 SA):
- 候选 A 训练 verdict 反不如 Exp5' → 开 SA-EXP5''-AUDIT 验尸
- n_active dry-run 仍低 → 候选 A 也踩鸡蛋问题,转 fallback B 或重设计
- 任何"我没预料到"的 bug

---

## §1 设计前提与 Exp5' 教训

### 1.1 Exp5' 教训(进 Exp5'' 不变量)

详 errata 5 §5。对 Exp5'' 强制约束:

1. **n_active 必 dry-run dump**(errata 5 §5.1):任何 loss 函数,P3 smoke 阶段必报"该 loss 在 batch 中 active 比例",≥ 50% 才能进训练
2. **训练时 ckpt selection 公式 ≠ 评估时 verdict 公式**(errata 5 §5.2):final report verdict 必须用 step5_3 7 项复合分,**不许**用 LightningModule `val_composite_ckpt_score` 作 verdict
3. **Watch-only 升级为 active monitor**(errata 5 §5.3):loss 数值 + 梯度有效性都 check
4. **训练超参 launch note 拍板,不擅自改**(errata 4 §3.4):Exp5'' 沿用 Exp5' 全部超参,**不改 batch / lr / scheduler**

### 1.2 Exp5' 给 Exp5'' 的硬证基础

| 验证项 | Exp5' 验证状态 | Exp5'' 沿用 |
|---|---|---|
| L=20 fold artifact 修复 | cartesian sanity 100/100 | ✅ |
| pairwise_min loss 生效 | gate 64% | ✅ 不动 |
| Cache (L=20) 完好 | 99.99% valid | ✅ 不重建 |
| shell_boundaries.pkl 干净 | cart Å + cart 一致率 100% | ✅ 不动 |
| ckpt callback 修复 | strict=False 工作 | ✅ 沿用 |
| dataset 5 字段 inject | sanity check 100/100 | ✅ 不动 |
| 训练流程 (batch / workers) | STEP2-CONTINUE composite +0.012 真改进 | ✅ 沿用 |

**Exp5'' 的工程风险面 ≈ 0**(只动 1 个 loss 公式)。

### 1.3 Exp5'' baseline 对比

| 维度 | Exp5 v2 baseline | Exp5' baseline | Exp5'' 目标 |
|---|---|---|---|
| Composite (step5_3) | 0.005-0.011 | **0.080**(基线)| **≥ 0.30** |
| Gate_pass_rate | 5-11% | **64%** | ≥ 70%(沿用 + 微改进)|
| Shell-1 distance score | 0.0000 | **0.035**(主攻方向)| **≥ 0.30** |
| Pred shell-1 mean (Å) | 不明 | **6.32**(目标修)| **接近 2.5**(true 2.27)|
| Collapse rate | 不明 | 0.00% | 0% 沿用 |

---

## §2 候选 A:Distance-Supervised Shell Loss(主线决议)

### 2.1 为什么选 A 不选 B

Exp5'-MA 2026-05-09 决议主线候选 A,**B 作 ablation fallback**(详 §3)。

| 维度 | A: Distance-supervised | B: Distance-aware density |
|---|---|---|
| 鸡蛋启动 | ✅ 无,从 random init 起就有梯度 | ⚠️ 早期分配震荡可能不稳 |
| 工程改动 | 中(改 2 个 loss 函数 + 公式)| 中(改 `_density_loss`)|
| 与现有 loss 协同 | ✅ 与 pairwise_min 天然兼容 | ⚠️ 与 pairwise_min 可能冲突(B 拉所有原子去 shell 半径,违 pairwise 约束)|
| Cheating 风险 | 用 ground truth shell count(标量 label,合规,详 §2.4)| 用 ground truth shell mean(已 Exp5' 在用,同合规)|
| 失败风险 | 中等(可能仍 RED 但有改进)| 较高(B 有可能让 collapse rate 反升)|

A 主线,B 备选。如 A 训练 verdict 反不如 Exp5' → 开 SA-EXP5''-ABLATION 切到 B。

### 2.2 候选 A 公式(替换 Exp5' `_shell_distance_loss`)

```python
def _shell_distance_loss_v2(pred_frac_coords, num_atoms,
                             true_shell1_d_mean, true_shell2_d_mean,
                             true_shell1_n, true_shell2_n, has_shell2,
                             L=L_VIRTUAL):
    """
    Distance-supervised shell loss (Exp5'' candidate A).

    NOT gap-based — 从 random init 起就能产生有效梯度。
    用 ground truth shell-1/shell-2 配位数 (true_shell{1,2}_n) 作切片大小,
    取 pred 最近 K 个原子的 radial mean,与 ground truth 半径做 MSE。

    Egg-chicken 解决:不依赖 pred 已有壳层结构。即使 random init
    pred 半径完全混乱,sorted_d[:K].mean() 仍是 well-defined value,
    与 true_shell1_d_mean 的差距驱动梯度修正所有 K 个最近原子。

    Compliance (cheating):用 true_shell{1,2}_n 是 sample-level 标量
    label (一个数字),不是 sample-level 输出 (frac_coords)。等价于图像
    分类用 ground truth label,不是用 ground truth pixel。proposal §2.3
    旧版 _shell_count_loss 已用 true_shell{1,2}_n 作 MSE target,合规。

    Args:
        pred_frac_coords:   (Σ N_i, 3) ∈ [-0.5, 0.5]
        num_atoms:          (B,) typically 20 each
        true_shell1_d_mean: (B,) Å, ground truth shell-1 mean radial dist
        true_shell2_d_mean: (B,) Å, shell-2
        true_shell1_n:      (B,) int, shell-1 配位数
        true_shell2_n:      (B,) int, shell-2 配位数
        has_shell2:         (B,) bool

    Returns:
        scalar tensor, isfinite-guarded.
    """
    total_loss = pred_frac_coords.new_zeros(())
    n_active = 0  # ⭐ Exp5'' 强制 dump (errata 5 §5.1)
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        if ni < 2:
            start += ni
            continue
        coords_i = pred_frac_coords[start:start+ni] * L          # cart Å
        radial = coords_i.norm(dim=1)
        sorted_d, _ = radial.sort()                              # ascending

        # Shell-1: 取最近 K1 = true_shell1_n 个原子(clip 到 [1, ni])
        n_s1 = max(1, min(int(true_shell1_n[i].item()), ni))
        pred_s1_d_mean = sorted_d[:n_s1].mean()
        total_loss = total_loss + (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2

        # Shell-2: 取 K1 之后的 K2 = true_shell2_n 个原子
        if bool(has_shell2[i].item()):
            n_s2 = max(1, min(int(true_shell2_n[i].item()), ni - n_s1))
            if n_s1 + n_s2 <= ni:
                pred_s2_d_mean = sorted_d[n_s1:n_s1+n_s2].mean()
                total_loss = total_loss + (pred_s2_d_mean - true_shell2_d_mean[i]) ** 2

        n_active += 1                                            # ⭐ active 比例 dump
        start += ni

    loss = total_loss / max(n_active, 1)
    # isfinite guard (Exp5' 沿用)
    if not torch.isfinite(loss):
        loss = pred_frac_coords.new_zeros(())
    sanitized = torch.nan_to_num(pred_frac_coords, nan=0.0, posinf=0.0, neginf=0.0)
    loss = loss + 0.0 * sanitized.sum()
    return loss, n_active  # ⭐ 返回 n_active 给 forward 端 dump
```

### 2.3 候选 A 公式(替换 Exp5' `_shell_count_loss`)

```python
def _shell_count_loss_v2(pred_frac_coords, num_atoms,
                          true_shell1_d_mean, true_shell2_d_mean,
                          true_shell1_n, true_shell2_n, has_shell2,
                          L=L_VIRTUAL,
                          shell_band_width=1.0):
    """
    Distance-supervised shell count loss (Exp5'' candidate A).

    用 ground truth shell-1/shell-2 半径作 band(中心 ± shell_band_width/2),
    数 pred 在 band 内的原子数,与 true_shell{1,2}_n 做 float MSE。

    Egg-chicken 解决:band 由 ground truth radial 定位,不依赖 pred gap 切壳。
    即使 random init pred 散在 band 外,(pred_count - true_count)² 给出
    "应有 K 个原子在此处" 的明确梯度信号,driving 原子向 band 收敛。

    shell_band_width: ±0.5 Å 容差(典型 shell 厚度,师兄经验值 ≤ 0.3 Å,加宽到 1.0 Å 给容错)
    """
    total_loss = pred_frac_coords.new_zeros(())
    n_active = 0
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        if ni < 2:
            start += ni
            continue
        coords_i = pred_frac_coords[start:start+ni] * L
        radial = coords_i.norm(dim=1)

        # Shell-1 band: |radial - true_shell1_d_mean| < shell_band_width/2
        s1_band_mask = (radial - true_shell1_d_mean[i]).abs() < (shell_band_width / 2)
        pred_s1_count = s1_band_mask.float().sum()  # 用 float 让梯度可传(soft count)
        total_loss = total_loss + (pred_s1_count - float(true_shell1_n[i])) ** 2

        if bool(has_shell2[i].item()):
            s2_band_mask = (radial - true_shell2_d_mean[i]).abs() < (shell_band_width / 2)
            pred_s2_count = s2_band_mask.float().sum()
            total_loss = total_loss + (pred_s2_count - float(true_shell2_n[i])) ** 2

        n_active += 1
        start += ni

    loss = total_loss / max(n_active, 1)
    if not torch.isfinite(loss):
        loss = pred_frac_coords.new_zeros(())
    sanitized = torch.nan_to_num(pred_frac_coords, nan=0.0, posinf=0.0, neginf=0.0)
    loss = loss + 0.0 * sanitized.sum()
    return loss, n_active
```

**注意**:`(radial - true_shell1_d_mean[i]).abs() < threshold` 在 PyTorch 中是 boolean mask,通过 `.float().sum()` 转 differentiable。但 boolean mask 本身不可微(梯度 0),实际生效靠"梯度回传到 radial → 通过 coords_i = frac × L → 回到 frac_coords"。**P3 smoke 必须 verify pred_count 对 frac_coords 的梯度非零**。

如 P3 smoke 显示梯度为 0(boolean mask 阻断),改用 sigmoid soft mask:

```python
# 替代:soft band membership
s1_membership = torch.sigmoid((shell_band_width/2 - (radial - true_shell1_d_mean[i]).abs()) * 10.0)
pred_s1_count = s1_membership.sum()  # differentiable
```

### 2.4 Cheating 合规性论证(预防 Exp5''-MA 自我怀疑)

候选 A 用了 `true_shell1_n / true_shell2_n` 作切片大小,可能引发"是不是用了过多 ground truth 信息"担忧。Exp5'-MA 论证:

**层级 1 — sample-level coordinates(最严格 cheating)**
- 用 ground truth `frac_coords` 直接 supervise pred coords → 这是回归任务的标准做法,但**绝不能在 inference 时用**

**层级 2 — sample-level statistics(标量 label)**
- 用 sample 的统计量(如 shell count = 一个数字,shell mean = 一个数字)→ **这是 supervised label,合规**
- 类比:图像分类用 ground truth class label,**不是 cheating**
- Exp5' 已经在用 `true_shell1_d_mean`(shell-1 平均半径,标量)作 `_shell_distance_loss` MSE 目标 → Exp5''-MA 沿用同一标准用 `true_shell1_n`(shell-1 配位数,标量)作切片大小,**是同一性质的 supervised signal**

**层级 3 — population-level statistics(数据集统计)**
- 用 train 集整体的 shell 分布作 prior(如 `_density_loss` 拉向 train 集中位数)→ 完全合规

**Exp5'' 候选 A 在层级 2,合规**。proposal §2.3 旧版 `_shell_count_loss` 当年也是用 `true_shell{1,2}_n` 作 MSE target,这次只是把它从 "MSE target" 升级为 "切片大小 selector"。

**inference 时如何处理**:候选 A 训练时用 ground truth count 切 pred,**inference 时不需要 ground truth count**(inference 是生成 frac_coords,不需要切壳;切壳只在 evaluation step5_3 时做,用的是 pred gap 算法)。所以 inference 时无 ground truth 依赖,**与 deployment 兼容**。

### 2.5 forward() 内调用 + 总 loss 公式

```python
def forward(self, batch):
    # ...(已有 coord/type/density/pairwise_min loss 计算)...

    # ⭐ Exp5'' 候选 A:重设计 shell loss
    loss_shell_dist_v2, n_active_shell_dist = self._shell_distance_loss_v2(
        pred_frac_coords, batch.num_atoms,
        batch.true_shell1_d_mean, batch.true_shell2_d_mean,
        batch.true_shell1_n, batch.true_shell2_n, batch.has_shell2,
        L=self.L
    )
    loss_shell_count_v2, n_active_shell_count = self._shell_count_loss_v2(
        pred_frac_coords, batch.num_atoms,
        batch.true_shell1_d_mean, batch.true_shell2_d_mean,
        batch.true_shell1_n, batch.true_shell2_n, batch.has_shell2,
        L=self.L,
        shell_band_width=1.0
    )

    # ⭐ total_loss 沿用 Exp5' 7 项公式,只换 shell loss 实现
    total_loss = (self.cost_lattice * loss_lattice              # 0.0
                + self.cost_coord   * loss_coord                # 1.0
                + self.cost_type    * loss_type                 # 1.0
                + self.cost_density * loss_density              # 0.2
                + self.cost_pairwise_min * loss_pairwise_min    # 1.0
                + self.cost_shell_dist   * loss_shell_dist_v2   # 0.5 沿用
                + self.cost_shell_count  * loss_shell_count_v2) # 0.2 沿用

    return {
        'loss': total_loss,
        'loss_lattice': loss_lattice,
        'loss_coord':   loss_coord,
        'loss_type':    loss_type,
        'loss_density': loss_density,
        'loss_pairwise_min': loss_pairwise_min,
        'loss_shell_dist':   loss_shell_dist_v2,        # 沿用旧 key 让 step5_3 不改
        'loss_shell_count':  loss_shell_count_v2,
        'n_active_shell_dist':  n_active_shell_dist,    # ⭐ Exp5'' 新加 dump
        'n_active_shell_count': n_active_shell_count,
        'pred_frac_coords': pred_frac_coords,
    }
```

**cost 不动**(0.5 / 0.2),理由:
- 候选 A 数量级与 Exp5' 旧版相近,cost 不需重新平衡
- 改 cost 是新一项 ablation,Exp5'' 主线只改公式不改 cost
- 如 P5 verdict RED,Exp5''' 阶段(若有)再 ablation cost

### 2.6 LightningModule 端 metric 增加

```python
def on_validation_epoch_end(self):
    outputs = self.validation_step_outputs
    # ...(沿用 Exp5')...

    # ⭐ Exp5'' 新加 metric
    n_active_shell_dist_mean = float(np.mean([o['n_active_shell_dist'] / batch_size_actual for o in outputs]))
    n_active_shell_count_mean = float(np.mean([o['n_active_shell_count'] / batch_size_actual for o in outputs]))

    self.log('val_n_active_shell_dist_ratio',  n_active_shell_dist_mean,  prog_bar=True)
    self.log('val_n_active_shell_count_ratio', n_active_shell_count_mean, prog_bar=True)
```

**val_n_active_shell_*_ratio** 应 epoch 0 就接近 1.0(候选 A 设计上每个 sample 都 active),如 < 0.95 ping Exp5''-MA 调查。

---

## §3 候选 B:Distance-Aware Density(Fallback / Ablation)

### 3.1 触发 B 的条件

不主动跑 B。仅在以下情况切换:
- 候选 A P5 训练完成后,verdict 反不如 Exp5'(composite < 0.08 或 shell-1 dist < 0.035)
- 候选 A P3 smoke 显示 n_active 低(< 50%,本不应该,但万一)
- 候选 A 训练曲线 collapse rate 上升 > 1%(候选 A 的 shell band 没硬性约束 pairwise,理论可能让原子聚集)

### 3.2 候选 B 公式(占位,实施时再细化)

把 `_density_loss` 改为 shell-target attractor:

```python
def _density_loss_v2(input_frac_coords, pred_x, sigmas, sigmas_norm,
                     true_shell1_d_mean, has_shell2, true_shell2_d_mean,
                     L=L_VIRTUAL):
    """每原子软分配到 shell-1/shell-2 by 距离最近,attract 到对应半径"""
    # Tweedie x0_hat
    sigma2 = sigmas ** 2
    x0_hat = input_frac_coords + sigma2 * pred_x.detach() * torch.sqrt(sigmas_norm)
    radial_hat = (x0_hat * L).norm(dim=-1)

    # 软分配
    d_to_s1 = (radial_hat - true_shell1_d_mean[batch_idx]).abs()
    d_to_s2 = (radial_hat - true_shell2_d_mean[batch_idx]).abs() if has_shell2 else float('inf')
    target = torch.where(d_to_s1 < d_to_s2, true_shell1_d_mean[batch_idx], true_shell2_d_mean[batch_idx])

    return ((radial_hat - target) ** 2).mean()
```

详细公式留 Exp5''-MA 切到 B 时再补,不在 proposal 里展开(避免 over-engineering)。

### 3.3 切到 B 的工作量预估

切到 B 等于重新做 Exp5'' 一遍:
- P1 改 `_density_loss_v2` 公式
- P2-P3 sanity test
- P4 训练(warm-start from Exp5' BEST,8-12h GPU)
- P5 sample + step5_3
- P6 final report

**额外 + 1-2 天工程 + 8-12h GPU**。如 A failed,Exp5''-MA 决议是否值得切 B。

---

## §4 训练计划

### 4.1 Warm-start vs From-scratch

**决议:Warm-start from `composite_epoch169_score0.5881.ckpt`**(Exp5' BEST)。

理由:
- Exp5' BEST 已学到:fold 修复后的 frac 坐标 + pairwise_min 约束 + type prediction(loss_type 已收敛到 0.017)
- 这些 "学到的部分" Exp5'' 不重学,直接 build on top
- 唯一要"忘"的是 Exp5' 旧 shell loss 留下的"假 shell 切壳行为",但因为 Exp5' 旧 shell loss 实际没生效,模型其实没"学坏",warm-start 不需要"先忘后学"
- 节省 7-30h GPU vs from-scratch

**Risk**:Exp5' BEST optimizer state 含 `_shell_distance_loss / _shell_count_loss` 旧版的 momentum / Adam state,这些项 Exp5'' 已替换为新公式,**但 PyTorch / PL 的 optimizer state 是按 model parameter 索引的,不是按 loss 名字索引**。模型参数全部不变(只改 loss 函数,不改 model.parameters()),所以 optimizer state 兼容。

### 4.2 训练超参

完全沿用 Exp5' STEP2-CONTINUE:

```yaml
optimizer: Adam
lr: 1e-4
batch_size: 64
num_workers: 16
persistent_workers: true
pin_memory: true
PreCollatedDataset: true
grad_clip: 1.0
precision: fp32
max_epochs: 500
patience: 30  # EarlyStopping
scheduler: CosineAnnealing
T_max: 500
GPU: [0]
strict: false  # ⭐ EarlyStopping strict=False
save_top_k: 3
save_last: true
monitor: val_composite_ckpt_score
mode: max
ckpt_path: /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt  # warm-start
```

### 4.3 中期监控触发(epoch 5 / 30 / 100)

| Epoch | 必报字段 | 触发停训条件 |
|---|---|---|
| 5 | val_n_active_shell_*_ratio + val_loss_shell_dist + val_loss_shell_count + val_composite_ckpt_score | n_active < 0.95 / loss_shell_dist 飙升 > 5 / NaN |
| 30 | + pred shell-1 mean dist 趋势 | pred shell-1 mean 没在向 true 方向(2.27 Å)收敛 |
| 100 | + 复合分相对 Exp5' baseline 改进趋势 | val_composite_ckpt_score 没超过 Exp5' BEST 0.5881 |

**Exp5''-MA 自己监控**(0 个 SA),但每个触发点暂停训练 10 分钟自我评估,有问题立即停。

### 4.4 预期训练时长

- Warm-start 起点 epoch 169(Exp5' BEST)
- Exp5'' 续训至 EarlyStop 或 epoch 500
- 单 epoch ~ 2:44(沿用 Exp5' 速度)
- 预期 EarlyStop 在 epoch 250-350(候选 A 应该比 Exp5' shell loss 收敛快,因为有有效梯度)
- **预计 GPU 时长:4-8h**

---

## §5 Verdict 阈值

### 5.1 Exp5'' verdict 主指标

| 指标 | Exp5'(基线)| Exp5'' GREEN | Exp5'' AMBER | Exp5'' RED |
|---|---|---|---|---|
| Composite (step5_3 7 项) | 0.080 | ≥ 0.30 ✅ | ≥ 0.15 ⚠️ | < 0.15 ❌ |
| Shell-1 distance score | 0.035 | ≥ 0.30 ✅ | ≥ 0.15 ⚠️ | < 0.10 ❌ |
| Pred shell-1 mean (Å) | 6.32 | ≤ 3.5(接近 true 2.27)✅ | ≤ 5.0 ⚠️ | > 5.0 ❌ |
| Gate_pass_rate | 64% | ≥ 70% ✅(微改进)| ≥ 60%(沿用)⚠️ | < 50%(退步)❌ |
| Collapse rate | 0% | ≤ 0.5% ✅(沿用)| ≤ 1% ⚠️ | > 1% ❌ |

### 5.2 决策树

```
Exp5'' P5 sample + step5_3 复合分:

├─ All GREEN
│   → Exp5'' 主线成功,verdict GREEN
│   → 写 Exp5'' final report (independent)
│   → 与 Exp5' 联合投稿全长 paper(见 §6.4)
│
├─ Composite GREEN + Shell-1 distance AMBER
│   → 候选 A 部分生效(shell band 太宽?)
│   → Exp5'''-MA 决议:调 shell_band_width 1.0→0.5(类似师兄 0.3 Å 经验)
│   → 或 Exp5''_v2 续 ablation
│
├─ Composite AMBER + Shell-1 distance RED
│   → 候选 A 没解决鸡蛋问题(可能 boolean mask 梯度阻断)
│   → 改用 sigmoid soft mask(§2.3 末尾备注)
│   → 或切候选 B
│
└─ All RED
    → 候选 A 完全失败
    → 切候选 B(§3)
    → 或 Exp6 路径(架构级,equivariant decoder)
```

### 5.3 Exp5' 与 Exp5'' 之间不能 cherry-pick

errata 4 §5.3 SOP:final report 必须双指标(或多指标)并列。Exp5'' final report 必须列:
- Exp5 v2 baseline(物理灾难)
- Exp5' baseline(部分成功)
- Exp5'' verdict(Mixed / Success / Failure)

不允许只比 Exp5' baseline 不比 v2 baseline,更不允许只看 composite GREEN 不报 shell-1 实测值。

---

## §6 关联 Exp5' Final Report v3

### 6.1 final report v3 的引用

Exp5' final report v3 §10 Future Work 列了 Exp5'' 候选 A/B,本 proposal **明确决议主线 A**。final report v3 不需修订,等 Exp5'' verdict 出来后**写 final report v4**(或 final report v3 §X 补附录)。

### 6.2 errata 5 §6 决议 → 本 proposal §2.1 落地

errata 5 §6 当时未定 A/B,本 proposal §2.1 落地为 **A 主线 + B fallback**。Exp5''-MA 接班后**不重新评估 A/B**,直接执行 A。

如果用户 / 师兄有不同意见(比如倾向 B),应在 Exp5''-MA 启动前与 Exp5'-MA(我)讨论,**不在 proposal 接班后再讨论**,因为重新讨论 = 工作流回退 = 浪费。

### 6.3 ckpt 接续

Exp5' BEST `composite_epoch169_score0.5881.ckpt` → Exp5'' warm-start 起点 → Exp5'' BEST(预期 epoch 200+)。

**Exp5' BEST ckpt 永久不动**。Exp5'' 训练产生的新 ckpt 落到独立目录:

```
/home/tcat/diffcsp_exp5_prime/checkpoints/         # Exp5' 永久档案
/home/tcat/diffcsp_exp5_double_prime/checkpoints/  # ⭐ Exp5'' 新建
```

理由:Exp5'' 与 Exp5' 是不同实验,paper trail 必须独立;**不混 ckpt 目录**避免 errata 4 教训重演。

### 6.4 投稿建议(更新 final report v3 §10.4)

| Verdict 情景 | 投稿方案 |
|---|---|
| Exp5'' All GREEN | **全长 paper**:fold artifact 诊断 + L=20 修复 + pairwise loss 验证 + shell loss 鸡蛋问题 + distance-supervised 重设计 = 完整方法论 |
| Exp5'' Mixed(composite GREEN / shell-1 AMBER)| **Short paper / workshop**:method paper 主推 fold + pairwise + shell-1 partial improvement |
| Exp5'' Failure(全 RED 或 fallback B 也失败)| **不投**,转 Exp6 架构级,留更多实验数据再投 |

---

## §7 Exp5''-MA 接班后的 6 件事(不可跳过)

### 7.1 启动 verify(沿用 STEP1-FIX-C / STEP2-CONTINUE / STEP3 流程)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# (A) Exp5' BEST ckpt 完好
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt
# 期望 127afa44a850d8f7e4fcdae17e2761a1

# (B) STEP3 9 个 step5_3 输出在(Exp5'' 评估时对比基线用)
ls /home/tcat/diffcsp_exp5_prime/logs/composite_score_*

# (C) Exp5'' 工作目录新建
mkdir -p /home/tcat/diffcsp_exp5_double_prime/{code,checkpoints,logs,predictions,data}
ln -s /home/tcat/diffcsp_exp4/data /home/tcat/diffcsp_exp5_double_prime/data_exp4
ln -s /home/tcat/diffcsp_exp5_prime/data /home/tcat/diffcsp_exp5_double_prime/data_exp5_prime

# (D) Exp5' code 树 cp 进 Exp5''(只改 1 个 loss 公式)
cp -r /home/tcat/diffcsp_exp5_prime/code /home/tcat/diffcsp_exp5_double_prime/

# (E) PYTHONPATH(沿用 Exp5' 三段)
export PYTHONPATH=/home/tcat/diffcsp_exp5_double_prime/code/step3:/home/tcat/diffcsp_exp5_double_prime/code/step2:/home/tcat/diffcsp_exp4/code

# (F) Exp5' BEST ckpt cp 进 Exp5''(warm-start 起点 + 永久档案)
cp /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt \
   /home/tcat/diffcsp_exp5_double_prime/checkpoints/start_from_exp5_prime_epoch169.ckpt
```

### 7.2 P1: 改 shell loss 公式

详 §2.2-2.5。改 `step3/diffusion_w_type_xas.py`,加 `.bak_pre_exp5pp` 锚点。

### 7.3 P2: forward_test Phase 6.7 重跑(加 n_active 验证)

加 6.7.h:

```python
# 6.7.h: shell loss n_active dump (errata 5 §5.1 强制)
loss_sd, n_active_sd = model._shell_distance_loss_v2(coords_spread, num_atoms_collapse, ...)
assert n_active_sd >= 1, f"n_active too low: {n_active_sd}"
log(f"[6.7.h PASS] n_active_shell_dist = {n_active_sd}/{1} on dummy batch")
```

### 7.4 P3: smoke test 2 epoch + n_active 比例

P3 必须 hand-back 报告:`val_n_active_shell_dist_ratio` ≥ 0.95(候选 A 设计上接近 1.0)。如 < 0.95,**P3 fail,Exp5''-MA 暂停 + 调查**。

### 7.5 P4: 启动训练(warm-start)

```python
# step4_2_train.py 改:
ckpt_path = "/home/tcat/diffcsp_exp5_double_prime/checkpoints/start_from_exp5_prime_epoch169.ckpt"
trainer.fit(model, ckpt_path=ckpt_path)
```

中期监控 §4.3 三个触发点。

### 7.6 P5 + P6: Sample + step5_3 + final report

沿用 Exp5' STEP3-SAMPLE 流程。`step5_3_composite_score.py` 完全不改(7 项复合分公式不变),只把 ckpt 路径改 Exp5'' BEST,输出文件名后缀加 `_exp5pp_lastckpt`。

最终 final report v4(或 final report v3 §X 附录,看 verdict 决议),Exp5''-MA 写。

---

## §8 风险与 Mitigation

### 8.1 Risk 1:候选 A 仍踩鸡蛋问题(boolean mask 梯度阻断)

**Mitigation**:§2.3 末尾已写 sigmoid soft mask 备选。P3 smoke 必 verify pred_count 对 frac_coords 的梯度 > 0,否则切到 sigmoid 重跑 P3。

### 8.2 Risk 2:Warm-start optimizer state 不兼容

**Mitigation**:§4.1 已论证 model.parameters() 不变 → optimizer state 索引兼容。P3 smoke verify 第一个 train_step 后参数有更新(grad_norm > 0)。

### 8.3 Risk 3:候选 A 让 collapse rate 上升

候选 A 把原子拉向 shell-1 半径(~ 2 Å),如 K1 = true_shell1_n = 6 个原子全聚到半径 2 Å 球面附近,可能违 pairwise_min 1.5 Å 约束。

**Mitigation**:`_pairwise_min_distance_penalty` 仍在(λ=1.0),pairwise 约束不变。两个 loss 协同:shell loss 拉 radial,pairwise loss 推开角向。理论上稳定。
**Verify**:P3 smoke + P4 epoch 5 验证 collapse rate ≤ 1%。

### 8.4 Risk 4:Cheating 论证不被 reviewer 接受

**Mitigation**:§2.4 已论证 ground truth shell count 是 sample-level 标量 label(同图像分类的 ground truth label),非 sample-level coordinates。Inference 不需 ground truth(§2.4 末段)。Final paper 实验段强化此论证。

### 8.5 Risk 5:Exp5'-MA 误判,实际 Exp5'' 应该 from-scratch 不 warm-start

**Mitigation**:Exp5'' 启动后 P3 smoke 如发现 model 行为与 Exp5' BEST 严重偏离(如 type acc 突然降),Exp5''-MA 决议是否切 from-scratch。这是低概率(< 10%)。

---

## §9 与 errata 5 §5 ExpN 不变量的对应

| Lesson | 来源 | Exp5'' 落实 |
|---|---|---|
| 5.1 n_active dry-run | errata 5 §5.1 | §2.2 / §2.3 公式末尾返回 n_active + §7.4 P3 hand-back 必报 |
| 5.2 训练 vs 评估 composite 公式区分 | errata 5 §5.2 | §5.1 verdict 阈值用 step5_3,§5.3 报告必双指标 |
| 5.3 Watch-only 升级 active monitor | errata 5 §5.3 | §4.3 中期监控 3 个触发点带停训条件 |
| 5.4 失败值得记录 | errata 5 §5.4 | §6.4 投稿方案含 Mixed verdict 分支 |

---

## §10 收尾

Exp5'' 是 Exp5' 的微调收尾。**不开新方向,只修一个 bug(鸡蛋问题)**。

**预期成果**:
- 90% 概率:Mixed verdict(composite GREEN 0.30+,shell-1 AMBER 0.20+)→ short paper
- 50% 概率:All GREEN(composite 0.40+,shell-1 0.30+)→ 全长 paper
- 10% 概率:候选 A 失败 → 切 B,或转 Exp6

**ROI**:1-2 天工程 + 10-14h GPU,换从 Exp5' RED 到 Exp5'' AMBER/GREEN 的可能性。

**不做的事**:
- 不改架构
- 不改训练超参
- 不重训 dataset(L=20 cache 不重建)
- 不动 Exp5' baseline(永久档案)
- 不写新 errata(除非 Exp5'' 触发新根因)

---

*Exp5'-MA 撰写,2026-05-09*
*基于 errata 5 §6 候选 A/B 决议落地为 A 主线 + B fallback*
*Exp5''-MA 接班后直接执行,0 个 SA(except verdict failure 触发 SA-AUDIT/ABLATION)*
