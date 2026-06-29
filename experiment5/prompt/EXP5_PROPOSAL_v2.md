# EXP5_PROPOSAL_v2.md
# DiffCSP-Experiment5: Multi-View Attention Encoder + Density-Loss De-anchoring

> **撰写者**: DiffCSP-Exp5-Main-Agent (= MA5 of Exp4 续作)
> **日期**: 2026-04-28
> **本文档对标**: EXP4_PROPOSAL_v2.md
> **背景**: Exp5 v1(独立 MA + SA1+SA2,head collapse epoch 36 kill)→ MA5 接管重写 v1 → v2(本文档)在 v1 基础上加入 fig agent 诊断的 `_density_loss` 塌缩根因发现
> **核心改动**: SpectrumEncoder 改 MV-attention(用户原意盲人摸象)+ `cost_density 0.5 → 0.2`(Exp4 → Exp5 v2 减弱塌缩剂)
> **不做的事**: TypeClassifier head(Exp3 + Exp5 v1 双重证伪)、from-Exp4-ckpt warm-start、multi-sample averaging

---

## §0 摘要(给 SA 一屏掌握)

**主线 1**: SpectrumEncoder 改 MV-attention,xmu/chi1/feff 三视角各 256d → cross-attention pool → 256d → 拼 center_emb (16d) → 进 diffusion decoder。

**主线 2**: `cost_density: 0.5 → 0.2`。Exp4 隐藏的塌缩根因被 fig agent 诊断 + MA5 conversation_search 联合定位为 `_density_loss`(Tweedie x0_hat → min-image → L2 → 0)在 88 元素分布下过强,详 §1.4。Exp5 v2 减弱 60% 看是否解放 hard sample 的 shell-shaped 输出。

**主验收**: holdout RMSD ≤ 1.40 (Exp4 1.4866,目标 -5.5%)+ collapse 比例 < Exp4(SA4 统计)。
**Bonus**: Set-Level TypeAcc / Multiset F1 上升、fig3 hard sample 视觉改善、SA3 投影 ablation 投影前后 RMSD 差 < 0.1(证明输出本身已合理,而非评估救场)。

**5 步,~5 天**: SA1' (改架构 + smoke) 2-3 天 → SA2' (训练) ~32h → SA3 (评估 + 投影 ablation) 0.5 天 → SA4 (figure + collapse 统计 + Exp6 决议) 0.5 天。

---

## §1 Exp4 + Exp3 真实历史(防止 Exp5 SA 重蹈)

### 1.1 Exp4 不重述

参考 EXPERIMENT4_FINAL_REPORT.md。Holdout RMSD 1.4866 / TypeAcc 0.1973 / pred_in_cutoff 18.92 / val/test/holdout 一致到 3 位小数。三大发现 O1/O2/O3。

### 1.2 Exp3 真实历史(纠正 EXP4 §10 简化叙事)

EXP4 final report §10 把 Exp3 简化为"证伪 TypeClassifier head 无效"。**这是错的**。真实 Exp3 教训:

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| Exp3 Step4e (v1) | TypeClassifier 仅以 spectrum_cond.detach() (256d) 为输入 | val_type_acc=0.21,**比 Exp2 baseline 0.249 还低** |
| Exp3 Step4f (v2) | 把 feff_features (73d) 拼到 head 输入(变 329d) | **val_type_acc 飙到 0.601**,但用户警觉"O 主导先验" |
| Exp3 总结 | Set-Level TypeAcc + Multiset F1 才是真信号 | position-by-position TypeAcc 在 Fe-only 数据上**可被全猜 O 虚报到 0.60** |

**Exp3 硬规则(写进 Exp5 不变量)**:

1. 任何附加 head 接触 latent 必须 `.detach()` —— Step4e 不 detach 时坐标 loss 爆到 9200 万
2. Position-by-position TypeAcc 是虚假指标 —— Set-Level + Multiset 才可信
3. 元素分类在当前框架下是病态问题(邻居顺序扩散后随机 + majority class 主导 + 配对错误)

Exp3 总结建议: **放弃元素分类提升,专注 RMSD**。Exp5 v2 采纳。

### 1.3 Exp5 v1 失败回顾

| 错点 | 来源 |
|---|---|
| 把用户 MV-attention 主意降级为 Phase B | Exp5 v1 MA self-audit 问题 1 |
| 没用 conversation_search 调 Exp3 | 问题 3 |
| 复刻 Exp3 v1 (spectrum-only) head 设计,没用 v2 (feff 直拼) | 问题 5 |
| 没读 §7.2 collapse 真因 → 误以为加 head 可救 type | 问题 6 |

SA2 现场: val head_ce = 2.20 > majority baseline 1.39,**conditioned majority collapse**(spectrum-blind 元素混合分布,可能 conditioned on center_Z),collapse 推断 epoch 10-15 完成。Exp5 v2 不加 head,从根上避开。

### 1.4 ⭐ fig agent 诊断 + `_density_loss` 塌缩根因(本节是 v1 → v2 的最大新增)

#### 1.4.1 fig agent 关键诊断

Exp4 Step6Agent(fig 端)被用户问"明明做了第一/二壳层物理约束,为什么键长还是不合理"时,给出三层机制分析:

1. **扩散模型的物理约束几乎总是软的不是硬的** —— SpectrumEncoder latent 软引导 denoising 轨迹,管不住生成端输出值落在哪。Hard sample 上 latent 不够 informative,decoder fall back 到训练分布平均特征 = "原子向中心密集"
2. **L=6 box + min-image + Hungarian 三层机械救场** 把 RMSD 撑在 1.49 Å,即使预测全塌中心,RMSD 数学上不超过 ~3 Å。fig4 三 split |r| < 0.03 + p > 0.05 强证: **RMSD 1.49 不是被物理顶住的,是被评估保护机制顶住的**
3. **88 元素让 distance prior 稀释到 1/88** —— Exp2 Fe-only 模型可学 near-deterministic Fe-O~2.0Å 先验;Exp4 88 中心散在 88 套距离尺度,prior 失效

#### 1.4.2 MA5 conversation_search 验证后的精确定位

fig agent 假设"约束按 shell 范围分段施加"——**与 Exp4 实际不符**。Exp4 物理约束实际在三层:

| 层 | 实现 | 是否塌缩剂 |
|---|---|---|
| Dataset 端 | `xas_local_dataset_v2.py` Phase 4.6 silent drop(<20 邻居 / frac 越界 → return None) | 否,只是过滤 |
| **Loss 端** ⭐ | `diffusion_w_type_xas.py::_density_loss`,Tweedie 估算 x0_hat → min-image → L2 均值 → 0 | **是,核心塌缩剂** |
| Evaluation 端 | `eval_cutoff` per-sample,`pred_in_cutoff` 指标 | 否,只是评估 |

**`_density_loss` 关键代码(Exp2 step4 时代加,Exp3/Exp4 继承,默认 cost_density=0.5)**:

```python
@staticmethod
def _density_loss(input_frac_coords, pred_x, sigmas_per_atom, sigmas_norm_per_atom):
    """
    用 Tweedie 公式从 (x_t, score) 估算去噪后的 x0_hat,
    再用最小镜像将其映射到 [-0.5, 0.5],计算 L2 均值。
    这迫使模型预测的"干净位置"集中于原点(中心元素)。
    """
    sigma2 = sigmas_per_atom ** 2
    sqrt_norm = torch.sqrt(sigmas_norm_per_atom)
    x0_hat = input_frac_coords + sigma2 * pred_x.detach() * sqrt_norm
    x0_hat_mi = x0_hat % 1.0
    x0_hat_mi = x0_hat_mi - (x0_hat_mi > 0.5).float()
    return (x0_hat_mi ** 2).mean()  # ← 全局 L2 → 0,无 shell 区分!
```

#### 1.4.3 重新归因 Exp4 collapse mode

`_density_loss` **是塌缩主犯**。它训练目标就是"x0_hat 朝原点压",不区分 shell。Exp2 Fe-only 下 OK(Fe-O ~2 Å 窄分布,"靠近原点"和"shell-shaped"在 L=6 box 几乎等价);**Exp4 88 元素下,中心 → 邻居距离尺度 1.2-3.5 Å 跨度大,"靠近原点"这个 prior 就错了**。

机制:
- Easy sample(spectrum signal 强): model 顶住 density loss 输出散布的 shell → fig3 Best #1/#2
- Hard sample(spectrum signal 弱): model 顺从 density loss 输出聚集云 → fig3 Mid #1 / Worst #1/#2 collapse 教科书例

**Exp4 hard sample collapse 不是 bug,是 training objective 的 feature 副作用**。Exp4 final report §7.2 把 collapse 归因为"diffusion decoder mean-position fallback",方向对但浅;真正根因是 `_density_loss` 这个训练目标本身。

详细 errata: 见 **EXP4_FINAL_REPORT_ERRATA_2.md**(MA5 同步交付)。

---

## §2 不变量(Exp5 不可改)

| 项 | 值 | 来源 |
|---|---|---|
| 中心元素 | 88 元素(Z ∈ [2,94]) | Exp4 |
| split | 60507/7624/4481/3025 | Exp4 |
| L | 6.0 Å | Exp2 step4d |
| coord 系 | [-0.5, 0.5] + min-image | Exp2 step4d |
| `cost_lattice` | 0 | MA4 决策 |
| `cost_coord` | 1.0 | Exp4 默认 |
| `cost_type` | 1.0 | Exp4 默认 |
| **`cost_density`** | **0.2**(Exp4 是 0.5) | **本 proposal §3.5,fig agent 诊断 + MA5 conversation_search 联合定位的塌缩根因减弱** |
| N_NEIGHBORS | 20 | EXP4_PROPOSAL_v2 §1.3 |
| 邻居搜索半径 | 10.0 Å | Exp2 |
| <20 邻居 / frac 越界 | `return None` + collate filter | Exp4 Phase 4.6 |
| FEFF feature 维度 | 74 | Exp4 |
| Latent 维度 | 256 | DiffCSP 默认 |
| Center embedding | 16d | Exp5 v1 SA1 |
| 反扩散步数 | 1000 | Exp2 |
| precision | fp32 | MA4 D1 |
| 任何附加 head 接 latent 必须 .detach() | 是 | Exp3 Step4e 教训 |
| **Position-by-position TypeAcc 仅作历史对照** | **Set-Level + Multiset 是真指标** | **Exp3 总结教训** |

**红线**:
- holdout 训练期不可读
- incompat_pool.csv 全程封存
- 不动 `forward_test.py.bak3` / 各 `.bak*`
- 不升级 7 守卫包(scikit-learn 1.7.2 / numpy 2.2.6 / scipy 1.15.3 / pymatgen 2025.10.7 / torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / torch-scatter 2.1.2+pt24cu124)
- **不加 TypeClassifier head**(Exp3 + Exp5 v1 双重证伪)
- **不做 multi-sample averaging**(独立任务)
- **不 fine-tune from Exp4 ckpt**(encoder 大改 shape mismatch)
- **cost_density 不许调到 0**(完全删除塌缩剂在 88 元素 + L=6 下风险高,可能 RMSD 退化到 2-3Å。0.2 是减弱,不是删除)

---

## §3 改动设计

### 3.1 SpectrumEncoder MV-attention 改造(主线 1)

**Exp4 现状**:
```
xmu (B, 150) → Conv1d-Pool-Linear → (B, 256)
chi1 (B, 200) → Conv1d-Pool-Linear → (B, 128)
feff (B, 74) → MLP → (B, 64)
                  ↓
        cat (B, 448) → Linear(448→256) → Linear(256→256) → (B, 256)
```

**Exp5 v1 SA1 改的(继承)**:
```
center_Z (B,) → Embedding(95, 16) → (B, 16)
final = cat(latent (B, 256), center_emb (B, 16)) → (B, 272)
```

**Exp5 v2 MV-attention 替换 fusion 层**:
```
xmu (B, 150) → Conv1d-Pool-Linear → (B, 256)            view_xmu
chi1 (B, 200) → Conv1d-Pool-Linear → (B, 256)  ★ 升 128→256  view_chi
feff (B, 74) → MLP-up → (B, 256)               ★ 升 64→256   view_feff
                  ↓
        stack → (B, 3, 256)                              views

        learnable query (1, 256)                         q
                  ↓
        Multi-head cross-attention(num_heads=4):
            Q = expand(q, B, 1, 256)
            K = views,  V = views
            attn_out = MHA(Q, K, V)  → (B, 1, 256)
                  ↓
        squeeze → (B, 256)                               fused
                  ↓
        + LayerNorm + residual: fused + 0.5 * mean(views)
                  ↓
        Linear(256, 256)  → (B, 256)                     latent

center_Z (B,) → Embedding(95, 16) → (B, 16)              center_emb
        ↓
final = cat(latent, center_emb) → (B, 272)
```

**关键设计**:

1. 三 view 升维到 256d 平衡(Exp4 256/128/64 不平衡)
2. Learnable query: `nn.Parameter(torch.randn(1, 256))`,broadcast,代表"decoder 想抓取什么"——不依赖 spectrum
3. `num_heads=4`(每 head 64d)
4. Residual `fused + 0.5 * mean(views)`: 防 attention 早期 noise 拉飞 latent。0.5 固定,不可学
5. LayerNorm 必须有

**预期机制**:
- 三 view 各保持完整 256d 表征(Exp4 chi/feff 压成 128/64,信息损失)
- Attention 显式建模"哪个 view 在哪个 sample 上更 informative"——hard sample 若 spectrum 信噪比低,attention 应更 weight feff 物理先验
- 核心假设: Exp4 cat→MLP fusion 是**静态加权**,所有样本同公式;MV-attention 是**动态加权**,逐样本调

### 3.2 不改的(继承 Exp5 v1 SA1)

| 文件 | Exp5 v1 已改 | 处理 |
|---|---|---|
| `xas_local_dataset_v2.py` | 加了 `center_element_Z` 字段 | 保留 |
| `xas_local_datamodule_v2.py` | 复用 Exp4 Phase 4.6 版,无改 | 保留 |
| `forward_test.py` | Phase 6.5 skip-with-note | 保留 |

### 3.3 撤销的(Exp5 v1 SA1 加的 head 部分)

| 文件 | v1 加的 | v2 处理 |
|---|---|---|
| `diffusion_w_type_xas_exp5.py` | `TypeClassifierHead` 类 + forward 内 type_ce_loss + total_loss = diffusion + λ × type_ce | **删除 head 类 + 还原 forward 和 loss 到 Exp4 形态** |
| `diffusion_xas_exp5.yaml` | head + center_emb + λ_type 字段 | **删除 head 字段,保留 center_emb 字段** |
| `step4_2_train_exp5.py` | type_loss_mode flag + phased training 钩子 | **删除 type_loss_mode**(from-scratch 不需 phased) |

### 3.4 新增评估指标(Exp3 教训)

`step5_2_compute_metrics.py` 在 Exp4 已有指标基础上加:

- **Set-Level TypeAcc**: 把 20 pred 元素 + 20 true 元素当 multiset,用类型距离做 Hungarian(不依赖 coord 配对),得匹配率
- **Multiset F1**: 元素分布层面 F1(set 角度,完全不依赖位置)
- 保留 position-by-position TypeAcc 作历史对照

**训练监控**(SA2): val_loss + val_set_level_typeacc + val_multiset_f1。Position-by-position 不监控。

### 3.5 ⭐ cost_density 减弱(主线 2,本 v2 新增)

**Exp4**: `cost_density: 0.5`(`_density_loss` 全局 L2 → 0,把所有 x0_hat 朝原点压)。
**Exp5 v2**: `cost_density: 0.2`(减弱 60%)。

**理由**: §1.4 已论证 `_density_loss` 是 88 元素 collapse 主犯。彻底删 (=0) 风险高(可能 RMSD 退化到 2-3Å,Exp2 step4c 时代就是因为没 density 才 RMSD 4.2 → 加上 → 1.47);保留不动 (=0.5) 与 Exp4 完全无差,白训。**0.2 是工程折中**: 减弱塌缩剂同时保留"原子应靠近原点"的弱 prior,给 MV-attention 改进的 latent 留发力空间。

**SA1' yaml 改动**: `conf_xas/model/diffusion_xas_exp5.yaml` 内 `cost_density: 0.5` → `0.2`。一行改动。

**风险**: 0.2 也可能错。SA3 评估时 `投影 ablation` 直接验证(详 §5.4)——如果投影后 RMSD 较 Exp5 v2 输出 RMSD 显著降(>0.1Å),说明输出仍塌缩,需进一步减弱;如果 ≤0.05,说明 0.2 足够。不靠预设,靠数据决议。

### 3.6 Exp5 v2 vs Exp4 直接对照

| 维度 | Exp4 | Exp5 v2 | 差异性质 |
|---|---|---|---|
| SpectrumEncoder | cat → MLP fusion | MV-attention | 主架构改 |
| Latent 维度(出 encoder) | 256 | 272 (256 + 16 center_emb) | 架构改 |
| cost_density | 0.5 | **0.2** | loss 减弱 |
| TypeClassifier head | 无 | 无 | 无变 |
| precision / batch / lr / optimizer | fp32 / 16 / 1e-4 / Adam | fp32 / 16 / 1e-4 / Adam | 无变 |
| max_epochs / patience | 500 / 30 | 500 / 30 | 无变 |
| dataset 行为 | Phase 4.6 silent drop | 同(继承) | 无变 |
| 训练时长 | ~32h | ~32h | 同 |

**两个改动同跑**(MV-attention + density 减弱),v2 验收阶段 SA3/SA4 通过 ablation 隔离贡献:
- 投影前 vs 投影后 RMSD: 若投影后改善 >0.1 → 说明仍塌缩 → density 减弱不够,问题主要在 decoder
- collapse 比例 Exp5 vs Exp4: 若显著降 → MV-attention + density 减弱组合有效

---

## §4 SA 编排

| SA | 任务 | 工程量 |
|---|---|---|
| **Exp5-SA1'** | 撤 head + MV-attention 重写 + cost_density yaml 改 0.2 + Set-Level TypeAcc 监控 + forward_test 5/5 PASS + smoke test | 中(2-3 天) |
| Exp5-SA2' | from-scratch 训练 | 后台 ~32h |
| Exp5-SA3 | sample val/test → 解禁 holdout → 算 Set-Level / Multiset 指标 + **投影 ablation** | 0.5 天 |
| Exp5-SA4 | 重画 6 figure(Exp5 vs Exp4 overlay)+ **collapse 比例统计** + Exp6 决议 | 0.5 天 |

### 4.1 SA1' 工作明细(防 SA1 现场再跑偏)

1. **6.1**: 撤销 `diffusion_w_type_xas_exp5.py` 的 `TypeClassifierHead` 类 + forward type_ce_loss + total_loss 还原(diff Exp5 v1 vs Exp4 此文件,删除 head 部分)
2. **6.2**: 撤销 yaml 的 `head:` 字段 + λ_type
3. **6.3**: ★ **yaml 改 `cost_density: 0.5 → 0.2`**(一行,关键)
4. **6.4**: 重写 `spectrum_encoder_exp5.py`(基于 Exp5 v1 SA1 已写的 + center embedding)把 fusion 替换成 MV-attention(§3.1 详细规格)
5. **6.5**: yaml 加 MV-attention 字段(`mv_attention.num_heads=4`、`mv_attention.residual_alpha=0.5`)
6. **6.6**: `step4_2_train_exp5.py` 删 type_loss_mode + phased training,改 from-scratch + max_epochs=500 + early_stop=30
7. **6.7**: `step5_2_compute_metrics.py` 加 Set-Level TypeAcc + Multiset F1 函数
8. **6.8**: forward_test.py 跑 6.1-6.4 + 6.5 skipped-by-design,5/5 PASS
9. **6.9**: smoke test(2 epoch × 10 batch)PASS + ckpt 落地
10. **6.10**: 中期报告交回 MA5(SA1' 不启动正式训练,等 MA5 review smoke test 后开 SA2')

### 4.2 SA3 工作明细

1. 同 Exp4 Step5Agent: sample val/test (~9h) → metrics → red-light gate → 解禁 holdout → 同
2. **新增 Phase 5b.5 投影 ablation**(详 §5.4): 跑完 sample 后,把 pred_frac_coords 内**距原点笛卡尔距离 > 5.5 Å** 的原子(已经在 box 边界外或 collapse 出格的离群)投影回 5.5 Å shell,重算 RMSD + Set-Level TypeAcc。**不重训,只后处理**。
3. 新增指标在 metrics_report 内分两栏报: 投影前 / 投影后

### 4.3 SA4 工作明细

1. 同 Exp4 Step6Agent: 6 figure(改 Exp5 vs Exp4 三 split overlay)
2. **新增 collapse 比例统计**: 对 val/test/holdout 三 split,统计 `pred std (across 20 atoms) < 0.5 × true std` 的样本数 + 占比。Exp5 vs Exp4 直接对比。
3. **figure 7 新增**: Exp5 vs Exp4 RMSD/TypeAcc by Tier overlay(2x2 panel,直接看 MV-attention + density 减弱组合 effect)

---

## §5 验收标准

### 5.1 主验收(必须达)

| 指标 | Exp4 holdout baseline | Exp5 v2 验收阈值 |
|---|---|---|
| RMSD (Å) | 1.4866 | ≤ 1.40(-5.5%) |
| pred_in_cutoff (/20) | 18.92 | ≥ 18.5(不退化) |
| val/test/holdout RMSD Δ | ≤ 0.001 | ≤ 0.005 |
| silent_drop 率 | ≤ 0.04% | ≤ 0.1% |

### 5.2 Bonus 验收(达成 = Exp6 进取方向调整)

| 指标 | Exp4 baseline | Bonus 阈值 |
|---|---|---|
| Set-Level TypeAcc | 待 SA3 重算 Exp4 ckpt 同指标作 baseline | 比 Exp4 baseline 高 ≥ 0.02 |
| Multiset F1 | 同 | 同 |
| Collapse 比例(val) | 待 SA4 算 Exp4 baseline | < Exp4 - 5% |
| fig3 hard-sample collapse | 6 panel 中 3 panel collapse | ≤ 1 panel |

### 5.3 红线(任一触发立刻停 + 报 MA5)

- RMSD > 2.0 Å: encoder 改造 + cost_density 减弱组合可能破坏了 condition signal
- val/holdout Δ RMSD > 0.1: 过拟合
- 训练 NaN/Inf
- Smoke test FAIL
- forward_test 5/5 不 PASS

### 5.4 ⭐ 投影 ablation(诊断验收,本 v2 新增)

**目的**: 区分"输出本身合理"和"输出塌缩但被评估机制顶住"。

**方法**:
```python
# SA3 跑完 sample 后,对每个样本的 pred_frac_coords 做后处理投影
# 投影规则: 笛卡尔距原点 > R_max 的原子,投影到 R_max shell 表面
#   R_max 取训练集真实距离 99 percentile (从 shell_boundaries.pkl 估,SA3 自己计算)
# 投影后重算 RMSD + Set-Level TypeAcc + Multiset F1
```

**判据**:
- 投影前 vs 投影后 RMSD Δ < 0.05 Å → **输出本身合理**,Exp5 v2 改进有效
- 投影前 vs 投影后 RMSD Δ ≥ 0.1 Å → **输出仍塌缩**,density 减弱 0.5 → 0.2 不够,需 Exp6 进一步处理
- 投影前 vs 投影后 RMSD Δ 0.05-0.1 Å → 中间区,进 Exp6 ablation

### 5.5 ⭐ Collapse 比例统计(SA4)

**目的**: 直接量化 fig3 6 panel 显示的失败模式在整体上的比例。

**方法**:
```python
# 对每个样本计算
pred_xyz_std = np.std(pred_frac * L, axis=0)  # (3,) per-axis std
true_xyz_std = np.std(true_frac * L, axis=0)  # (3,)
is_collapsed = (pred_xyz_std.mean() < 0.5 * true_xyz_std.mean())
collapse_pct = collapsed_count / total_count
```

**Exp5 vs Exp4 baseline 比对**: SA4 必须用 Exp4 best ckpt 跑同样诊断作 baseline,不能只报 Exp5 数。

---

## §6 风险评估(诚实)

### 6.1 MV-attention 真有用吗?

没人在 atomic structure diffusion 上验证过。Exp4 几何已 parity,encoder 改进收益边际可能递减。
**对策**: §5.1 RMSD ≤ 1.40 是激进目标,做到 1.42-1.45 也是负结果(MV-attention 在该任务无显著增益,Exp6 转方向)。

### 6.2 cost_density 0.2 选择不对

可能 0.2 还是过强(collapse 不消)、或太弱(RMSD 退化到 1.6+)。
**对策**: §5.4 投影 ablation 直接诊断,Exp5 v2 是"减弱方向"探索,不是终极配置。Exp6 可基于 SA3 数据决议 0.1 / 0.05 / 0 哪个值。

### 6.3 from-scratch ~32h 太长

后台 nohup + 30 min 早期监控 + 关窗口。同 Exp4 流程。

### 6.4 SA1' 又跑偏

(1) handoff §6.1-6.10 子任务一个不漏 + 每步 PASS gate;(2) 中期报告 review 后才开 SA2';(3) **SA1' 禁止启动正式训练**(明文)。

### 6.5 ⭐ MV-attention 攻击的是 encoder,塌缩根因可能在 decoder

fig agent 机制分析: latent 不够 informative 时,decoder 退到 mean-position prior。MV-attention **间接**缓解(让 latent 更 informative 让 decoder 不退),**不直接修 decoder**。如果 decoder mean-position fallback 是真根因,MV-attention 单独收益有限,需要 cost_density 减弱配合。**因此 v2 把两个改动同跑,不是单独做**。

如果 SA3/SA4 显示 v2 提升有限,Exp6 候选是 anti-collapse loss(直接攻 decoder)而不是继续 encoder 改造。

### 6.6 SA0 multi-sample 没做会不会丢东西?

Exp5 完后,如果想知道 Exp4/Exp5 ckpt multi-sample 能到多少,我帮单开一棒,~12-15h,与 Exp5/Exp6 都不冲突。

---

## §7 与 Exp4 final report errata 的关系

EXP4_FINAL_REPORT_ERRATA_2.md(MA5 同步交付)单独记录:
- §7.2 collapse 归因更新: 从"diffusion decoder mean-position fallback"细化为"`_density_loss` cost_density=0.5 在 88 元素分布下过强"
- §10 方向 menu 更新: 方向 4 anti-collapse loss 优先级提升到 ⭐⭐⭐(原 ⭐)
- §7.3 解耦反证 Exp3 那段更新: 真实 Exp3 教训不是"加 head 无效",而是"position-by-position 是虚假指标 + .detach 必须有"

---

## §8 文件清单(Exp5 工作目录)

```
/home/tcat/diffcsp_exp5/                       ← SA1' 创建
├── code/
│   ├── .env
│   ├── step3/
│   │   ├── xas_local_dataset_v2_exp5.py        ← v1 SA1 写的,保留
│   │   ├── xas_local_datamodule_v2.py          ← 复用 Exp4
│   │   ├── spectrum_encoder_exp5.py            ★ SA1' 重写 MV-attention
│   │   ├── diffusion_w_type_xas_exp5.py        ★ SA1' 撤 head 还原 Exp4 形态
│   │   ├── conf_xas/model/diffusion_xas_exp5.yaml   ★ SA1' 撤 head + 加 MV-attention + cost_density 0.2
│   │   └── forward_test.py                     ← 复用 v1 (Phase 6.5 skip-with-note)
│   ├── step4/
│   │   ├── step4_1_smoke_test.py               ← 复用 Exp4 模板
│   │   └── step4_2_train.py                    ★ SA1' 改 from-scratch
│   ├── step5/
│   │   ├── step5_1_sample.py                   ← 复用 Exp4
│   │   ├── step5_2_compute_metrics.py          ★ SA1' 加 Set-Level + Multiset
│   │   └── step5_3_projection_ablation.py      ★ SA3 新增(投影 ablation)
│   └── step6/
│       └── step6_visualize.py                  ← SA4 fork Exp4 改
├── data/                                       ← 软链接到 Exp4 data
├── checkpoints/
└── logs/
```

**前置 setup 命令**(SA1' 第一棒跑):

```bash
mkdir -p /home/tcat/diffcsp_exp5/{code/step3,code/step4,code/step5,code/step6,data,checkpoints,logs}
mkdir -p /home/tcat/diffcsp_exp5/code/step3/conf_xas/model

# 软链接数据(节省 650MB + 防误改)
cd /home/tcat/diffcsp_exp5/data
for f in /home/tcat/diffcsp_exp4/data/*; do ln -s "$f" .; done

# 复制 .env (改路径)
sed 's|diffcsp_exp4|diffcsp_exp5|g' /home/tcat/diffcsp_exp4/code/.env > /home/tcat/diffcsp_exp5/code/.env

# 取 Exp5 v1 SA1 已改好的文件作起点(具体路径用户确认)
# cp /home/tcat/diffcsp_exp5_v1/code/step3/{forward_test.py,xas_local_dataset_v2_exp5.py} \
#    /home/tcat/diffcsp_exp5/code/step3/
```

如 Exp5 v1 工作目录路径与上面假设不同,SA1' 跑前问用户确认。

---

## §9 给 Exp5-SA1' 的共享文档清单

| # | 文档 | 必读? |
|---|---|---|
| 1 | **本 proposal v2**(EXP5_PROPOSAL_v2.md) | ✅ 精 |
| 2 | EXP4_FILE_GUIDE.md | ✅ 精(取用 Exp4 服务器代码) |
| 3 | EXPERIMENT4_FINAL_REPORT.md | ✅ 速(数字 baseline) |
| 4 | EXP4_ERRATA_2026-04-28.md | ✅ 速(Phase 6.5 真实状态) |
| 5 | **EXP4_FINAL_REPORT_ERRATA_2.md** | ✅ 精(`_density_loss` 塌缩根因) |
| 6 | Exp5 v1 SA1 改的 5 个文件(spectrum_encoder + dataset + diffusion + yaml + train) | ✅ SA1' 必看,作起点基线 |

**不传**: EXP4_PROPOSAL_v2.md / 所有 EXP4 中间 handoff / Exp5 v1 MA self-audit。

---

## §10 Lessons Learned(给后续 ExpN)

1. **不要让"上一棒 MA 推荐"override 用户原意** —— MA 的 menu 也是需要 critique 的对象
2. **conversation_search 在 ExpN 启动时是默认动作**,不是补救
3. **指标体系比模型架构更优先验证** —— position-by-position TypeAcc 在 Fe-only 数据上虚假这件事,如果 Exp4 / Exp5 v1 一开始就用 Set-Level,Exp3 Step4f 的 0.601 误判不会传播两轮
4. **Type prediction 在当前框架下是病态问题** —— Exp3 总结建议放弃元素分类,Exp5 v2 采纳
5. ⭐ **诊断 vs 评估指标必须分离** —— Exp4 RMSD 1.49 看起来是物理约束顶住的(评估视角),实际是 L=6 box + min-image + Hungarian 三层机械救场顶住的(诊断视角)。**任何 RMSD 卡在 box-half (L/2) 附近的实验,都需要用投影 ablation 验证**——投影前后差异 > 0.1 Å 即警示。这条规则进 Exp6+ 不变量。
6. ⭐ **训练目标本身可能是失败模式的来源** —— Exp4 `_density_loss` (cost_density=0.5) 在 Fe-only 设计时合理,在 88 元素扩展时变成塌缩剂。**任何辅助 loss 跨实验复用前,要重审其在新数据分布下的隐式偏置**。
7. ⭐ **fig agent 这种"读图反推机制"的价值很高** —— Exp4 final report §7 我自己写的 collapse 归因(MA5 视角)只到"decoder fallback"层,fig agent 推到"评估机制顶住"层。让独立 sub-agent 看 figure + 反推,与 main agent 写 final report 是互补的诊断动作,不应只让 main agent 一人解读

---

*Main Agent 5 撰写,2026-04-28,Exp5 v2 重启 proposal(含 fig agent 诊断 + density loss 根因发现)。*
