# EXP4_FINAL_REPORT_ERRATA_2.md
# Exp4 final report 勘误 #2 — `_density_loss` 塌缩根因 + Exp3 真实历史 + 方向 menu 调整

> **撰写者**: Main Agent 5
> **日期**: 2026-04-28(Exp5 v2 proposal 撰写期间发现)
> **本文档定位**: 继承 EXP4_ERRATA_2026-04-28.md 格式(Phase 6.5 状态修正)
> **触发**: Exp4 Step6Agent(fig 端)被用户问"为什么物理约束失效",给出三层机制诊断;MA5 conversation_search 验证后精确定位塌缩根因
> **影响范围**: EXPERIMENT4_FINAL_REPORT.md §7.2 / §7.3 / §10
> **不重写原文档**,作为独立 errata 存档,后续读者引用 final report 时附带阅读

---

## §1 错误声明 #1: §7.2 collapse 归因不够深

### 1.1 原文

EXPERIMENT4_FINAL_REPORT.md §7.2 "O2 — Predicted-atom collapse mode":

> "Diffusion decoder 在 hard sample 上**部分回退到中心-塌缩先验(mean-position fallback)**。"

### 1.2 实际情况

**fig agent 三层诊断**(Exp4 Step6Agent 给用户的回复):

1. 扩散模型物理约束几乎总是软的不是硬的 —— SpectrumEncoder latent 软引导 denoising 轨迹,管不住生成端输出值落在哪
2. **L=6 box + min-image + Hungarian 三层机械救场把 RMSD 撑在 1.49 Å** —— 即使预测全塌中心,RMSD 数学上不超过 ~3 Å。fig4 三 split |r| < 0.03 / p > 0.05 是强证: **RMSD 1.49 不是物理顶住,是评估保护机制顶住的**
3. 88 元素让 distance prior 稀释到 1/88 —— Exp2 Fe-only 模型可学 near-deterministic Fe-O ~2.0Å 先验,Exp4 88 中心散在 88 套距离尺度,prior 失效

**MA5 conversation_search 后的精确定位**:

fig agent 假设"约束按 shell 范围分段施加"——**与 Exp4 实际不符**。Exp4 物理约束实际三层:

| 层 | 实现 | 是否塌缩剂 |
|---|---|---|
| Dataset 端 | `xas_local_dataset_v2.py` Phase 4.6 silent drop | 否,只是过滤 |
| **Loss 端** ⭐ | `diffusion_w_type_xas.py::_density_loss`,Tweedie x0_hat → min-image → L2 → 0,cost_density=0.5 | **是,核心塌缩剂** |
| Evaluation 端 | `eval_cutoff` per-sample,`pred_in_cutoff` | 否,只是评估 |

**`_density_loss` 关键代码**(Exp2 step4 时代加,Exp3/Exp4 继承):

```python
@staticmethod
def _density_loss(input_frac_coords, pred_x, sigmas_per_atom, sigmas_norm_per_atom):
    """用 Tweedie 公式从 (x_t, score) 估算去噪后的 x0_hat,
    再用最小镜像将其映射到 [-0.5, 0.5],计算 L2 均值。
    这迫使模型预测的"干净位置"集中于原点(中心元素)。"""
    sigma2 = sigmas_per_atom ** 2
    sqrt_norm = torch.sqrt(sigmas_norm_per_atom)
    x0_hat = input_frac_coords + sigma2 * pred_x.detach() * sqrt_norm
    x0_hat_mi = x0_hat % 1.0
    x0_hat_mi = x0_hat_mi - (x0_hat_mi > 0.5).float()
    return (x0_hat_mi ** 2).mean()  # ← 全局 L2 → 0,无 shell 区分
```

### 1.3 修正后的归因

**`_density_loss` 是塌缩主犯**。它训练目标就是"x0_hat 朝原点压",**全局 L2,不区分 shell**。

- Exp2 Fe-only: OK(Fe-O ~2 Å 窄分布,"靠近原点"和"shell-shaped"在 L=6 box 几乎等价)
- **Exp4 88 元素: 错**(中心 → 邻居距离 1.2-3.5 Å 跨度大,"靠近原点"这个 prior 错)

机制:
- Easy sample(spectrum signal 强): model 顶住 density loss 输出散布的 shell → fig3 Best #1/#2
- Hard sample(spectrum signal 弱): model 顺从 density loss 输出聚集云 → fig3 Mid #1 / Worst #1/#2 collapse 教科书例

**Exp4 hard sample collapse 不是 bug,是 training objective 的 feature 副作用**。Exp4 final report §7.2 把 collapse 归因为"diffusion decoder mean-position fallback",方向对但浅;真正根因是 `_density_loss` 这个训练目标本身。

### 1.4 影响传播

- **Exp4 训练结果不变**(0.7300 ckpt 完全可信)
- **Exp4 评估结果不变**(数字仍是 holdout RMSD 1.4866 / TypeAcc 0.1973 / pred_in_cutoff 18.92)
- **改变的是对结果的解读**: RMSD 1.49 不是几何精度极限,是评估保护机制 + 训练目标合谋的结果。fig4 RMSD↔TypeAcc |r|<0.03 现在有了根因解释——RMSD 不是"被物理顶住",是被三层机械救场顶住,所以与 type 难度自然解耦
- **Exp5 v2 直接受影响**: cost_density 0.5 → 0.2(EXP5_PROPOSAL_v2.md §3.5),减弱塌缩剂作主线 2 改动

---

## §2 错误声明 #2: §7.3 反证 Exp3 那段叙事不准确

### 2.1 原文

EXPERIMENT4_FINAL_REPORT.md §7.3:

> "EXP4_PROPOSAL_v2 §1.3 锁定'不加 TypeClassifier head'基于 Exp3 时代的证伪,逻辑是'diffusion decoder 已经学到 type,加 head 无显著增益'。但 Exp4 数据显示...这意味着: 解耦 type prediction 进独立 head,在 Exp4 数据上极不可能让坐标变差..."

### 2.2 实际情况

Exp3 真实历史不是"加 head 无效",而是更复杂的两段:

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| Exp3 Step4e (v1) | TypeClassifier 仅以 spectrum_cond.detach() (256d) 为输入 | val_type_acc=0.21,**比 Exp2 baseline 0.249 还低**(.detach 切断梯度,encoder 没被要求编码 type) |
| Exp3 Step4f (v2) | feff_features (73d) 拼到 head 输入(变 329d) | **val_type_acc 飙到 0.601**,但用户警觉"O 主导先验" |

**Exp3 总结报告原话**(用户提供):

> "Exp3 主要目标(TypeClassifier 提升元素预测)**从根本上失败**,原因不在工程实现,而在**问题本身的统计结构**: Fe氧化物邻居元素分布高度偏斜(第一壳层 O 占比 > 60%),任何以交叉熵为目标的分类器都会收敛到'多数类'解,这是贝叶斯最优策略,不是过拟合,也不是实现缺陷。"
>
> "**Position-by-position TypeAcc 是可被'全猜多数类'游走的虚假指标**,Set-Level TypeAcc + Multiset F1 才是真信号。"

### 2.3 修正后的叙事

**Exp3 真正的硬规则(进 Exp5 不变量)**:

1. 任何附加 head 接触 latent 必须 `.detach()`(Step4e 不 detach 时坐标 loss 直接爆到 9200 万)
2. **Position-by-position TypeAcc 是虚假指标**,Set-Level + Multiset 才可信
3. 元素分类在当前框架下是病态问题(邻居顺序扩散后随机化 + majority class 主导 + 坐标 RMSD 1.5Å 让配对本身大量错误)
4. Exp3 总结建议: **专注 RMSD,放弃 type 提升**

### 2.4 影响传播

- **Exp4 final report §7.3 + §10 方向 1 (decoupled head, ⭐⭐⭐)** 这一线索全错位—— "加 head 反证可行" 这个逻辑不再成立,因为 Exp4 0.197 是用虚假指标 (position-by-position) 算的,真实 type prediction 上限可能远更低或更高,需要重新算 Set-Level/Multiset 才能定
- **Exp5 v2 直接受影响**: 不加 TypeClassifier head(EXP5_PROPOSAL_v2.md §1.2、§2、§3.3)
- **Exp5 v1 SA1+SA2 失败的部分根因**: 没看 Exp3 总结的 .detach + 虚假指标教训,复刻了 Exp3 v1 spectrum-only head 设计

---

## §3 影响 §10 方向 menu 排序

EXPERIMENT4_FINAL_REPORT.md §10 给 Exp5/Exp6 列了 7 方向 menu。基于 errata 2 重新排序:

### 3.1 原排序

| 方向 | MA5 原排名 | 注释 |
|---|---|---|
| 1. Decoupled TypeClassifier head | ⭐⭐⭐ | 基于 Exp4 |r|<0.03 反证 |
| 2. Center-element conditioning | ⭐⭐ | 攻 O1 rank-1 weakness |
| 3. Multi-view attention | ⭐⭐ | 用户意向但 MA5 排第 3 |
| 4. Anti-collapse loss | ⭐ | 攻 O2 |
| 5. Multi-sample averaging | — | 0 训练成本 quick win |
| 6. Equivariant decoder | — | Exp6+ 长线 |
| 7. Cascaded diffusion | — | 不推荐 |

### 3.2 修正后排序

| 方向 | 新排名 | 修正理由 |
|---|---|---|
| ~~1. Decoupled head~~ | **退** | Exp3 双重证伪(v1 detach 失败 + v2 虚假指标),Exp5 v1 SA2 三重证伪。**Exp5/6 不再考虑** |
| **2. Center-element conditioning** | **保留 ⭐⭐** | Exp5 v1 SA1 已实现 (95×16d Embedding),Exp5 v2 继承 |
| **3. MV-attention** | **升 ⭐⭐⭐** | 用户原意,Exp5 v2 主线。**fig agent 诊断的 latent 不够 informative 可由 MV-attention 改进** |
| **4. Anti-collapse loss** | **升 ⭐⭐⭐** | errata 2 §1 揭示 `_density_loss` 是塌缩主犯。**直接攻击 decoder collapse 的方向**。Exp5 v2 已经做的减弱版是 cost_density 0.5→0.2,**Exp6 候选是改造为 distance-aware loss(shell-target L2 而非 origin-attractor L2)** |
| 5. Multi-sample averaging | 保留 | 任何时候单独跑都行,0 训练成本,与 ExpN 都不冲突 |
| 6. e3nn equivariant | 保留 长线 | Exp7+ |
| 7. Cascaded | 不推荐 | 同原 |
| **新增 8. Hierarchical type prediction** | ⭐⭐ | Exp4 final report §10 已列(新候选 a),不变 |
| **新增 9. Classifier-free guidance** | ⭐⭐ | Exp4 final report §10 已列(新候选 b),不变 |

### 3.3 Exp6 方向决策树更新

errata 2 把 §1 的 decision tree 大幅简化:

```
Exp5 v2 验收(主验收 RMSD ≤ 1.40 + 投影 ablation):

├─ RMSD < 1.40 + 投影前后 Δ < 0.05
│   → MV-attention + density 0.2 组合成功
│   → Exp6 进取: 方向 4 改造 distance-aware (shell-target) loss
│             或 新候选 8 hierarchical type
│
├─ RMSD 1.40-1.50 + 投影前后 Δ 0.05-0.10
│   → MV-attention 部分有效,density 减弱不够
│   → Exp6 攻坚: 方向 4 强化(cost_density 进一步减或重设计)
│             或 新候选 9 CFG
│
└─ RMSD ≥ 1.50 或 投影前后 Δ > 0.1
    → MV-attention 在该任务无显著增益(负结果有价值)
    → Exp6 完全转方向: 方向 4 distance-aware + 新候选 9 CFG
       不再做 encoder 改造
```

---

## §4 Lessons Learned(写进 ExpN 不变量级别)

### 4.1 RMSD 卡在 L/2 附近时,投影 ablation 是必做诊断

任何 RMSD 卡在 box-half (L/2) 附近的实验,都需要用投影 ablation 验证——投影前后差异 > 0.1 Å 即警示输出被评估机制顶住而非物理约束顶住。这条规则进 Exp6+ 不变量。

### 4.2 训练目标本身可能是失败模式来源

`_density_loss` 在 Fe-only(Exp2)设计时合理,在 88 元素(Exp4)扩展时变成塌缩剂。**任何辅助 loss 跨实验复用前,要重审其在新数据分布下的隐式偏置**。

### 4.3 Fig agent 类的"读图反推机制" 价值高

Exp4 final report §7 我自己写的 collapse 归因(MA5 视角)只到"decoder fallback"层,fig agent 推到"评估机制顶住"层。让独立 sub-agent 看 figure + 反推,与 main agent 写 final report 是互补的诊断动作。**未来 ExpN 的 figure agent 完成产出后,应固定加一个步骤: "用户主动问 figure agent 至少一个反推问题"**。

### 4.4 Conversation_search 必须早做

如果 MA5 在写 EXP4 final report §10 方向 menu 时就用 conversation_search 调 Exp3 历史,§7.3 反证那段不会写错,§10 方向 1 不会标 ⭐⭐⭐,Exp5 v1 不会跑偏。**ExpN 启动时调用 conversation_search 是默认动作,不是补救**。

---

## §5 给后续 ExpN 的提醒

如果未来 ExpN(N ≥ 5)的 Sub-Agent 看到任何引用 EXP4 final report §7.2 / §7.3 / §10 的下游文档,应同时引用本 errata 2。

**已知 propagate 错误的下游**:
- EXP4_PROPOSAL_v2 §1.3 "不加 TypeClassifier head"理由 —— 表面结论对(确实不加),但理由是 Exp3 简化叙事,真理由是 Exp3 .detach 教训 + 虚假指标 + 病态问题
- 任何引用 "Exp4 §10 方向 1 ⭐⭐⭐" 的文档 —— 应改为 "MV-attention 主线 + anti-collapse loss 候选"

---

## §6 与 EXP4_ERRATA_2026-04-28.md 的关系

errata 1: Phase 6.5 实际状态修正(SA4-续 2 报 5/5 PASS 未独立验证)
errata 2(本文档): `_density_loss` 塌缩根因 + Exp3 真实历史 + 方向 menu 调整

两份 errata **并列存档,不合并**——错误的传播路径可追溯,Lessons Learned 可累积。

---

*MA5 撰写,2026-04-28,Exp5 v2 proposal 撰写期间因 fig agent 诊断 + conversation_search 联合发现的根因 errata。*
