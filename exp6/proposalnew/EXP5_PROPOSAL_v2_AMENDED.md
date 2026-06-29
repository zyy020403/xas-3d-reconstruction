# EXP5_PROPOSAL_v2_AMENDED.md
# Exp5 v2 Proposal — Amended for Composite Scoring + Pairwise Distance Constraint Discovery

> **Original**: EXP5_PROPOSAL_v2.md (2026-04-28 by MA5 时代)
> **Amended by**: MA5 (final amendment before transition to Exp5 MA2/MA-EXP5-v2-extension)
> **Date**: 2026-05-01
> **Trigger**: 用户在 SA2'' 续训完成后统计预测结构发现大量原子间距 < 1.5 Å,
>   暴露 v2 评估体系第四层盲区(min pairwise distance)
> **Scope**: 不重写原 proposal,append 修订内容到原 proposal 末尾;
>   修订内容主要在: 评估体系 §3.4 / 验收标准 §5 / Exp6 路径 → 新增 Exp5' 分支

---

## §A 物理约束发现(2026-05-01,MA5 + 用户)

### A.1 现象

用户在 SA2'' 续训完成后,统计 `predictions_v2_*.pt` 的预测原子结构,发现:
**大量样本中存在原子间距 < 1.5 Å 的"原子重合"现象,这些预测结构在物理上不合理(无法跑 FEFF 计算)。**

### A.2 评估盲区根因

Exp5 v2 设计的 6 项 metric 体系(RMSD / pred_in_cutoff / Set-Level TypeAcc / Multiset Macro-F1 /
Collapse Ratio / Projection Ablation Δ RMSD)**全部未检测原子两两之间的距离**:

- RMSD: per-atom 距 ground truth 的 Hungarian 距离(不看 pred 内部结构)
- pred_in_cutoff: 每个 pred 原子距原点距离(不看两两距离)
- Set-Level / Multiset: 元素 multiset 比较(完全 ignore 几何)
- Collapse Ratio: pred std vs true std(20 原子全挤在 1 Å 球内 std=0.5,接近 true std,触发不了)
- Projection Ablation: 每个 pred 距原点(同 pred_in_cutoff,不看两两)

**这是一个独立的 collapse 模式**,数学层面 RMSD/Collapse 都看不到,但物理上不合理。

### A.3 责任划分

- SA1' 写 metrics 时未设计 min pairwise distance 检测
- SA3' 跑投影 ablation 时报告 "0 原子需投影 / Δ=0",我以为排除了"评估保护机制顶住"风险,
  实际只排除了 R_max 球壳外的塌缩,**球壳内两两重合是另一类塌缩,投影 ablation 看不到**
- MA5 review SA1' 设计时未质疑指标完备性

这是 Main Agent 责任。Lessons learned 写进本 proposal §C 给 Exp5 MA2+。

---

## §B Exp5 v2 评估体系修订 — 7 项复合评分(取代单一 Multiset F1 主信号)

### B.1 背景

Exp5 v2 verdict 框架原用 Multiset Macro-F1 作主信号(>0.20 GREEN / 0.10-0.20 AMBER / <0.10 RED),
SA3' 测得 SA2 baseline val=0.1086 落 AMBER。续训后 SA2'' epoch 529 val_loss 0.7003(改进 0.88%),
**但用户提出**:抽象 multiset 指标不如 6+ 项物理 motivated 评分 closer to 化学家判断"结构对不对"。

### B.2 7 项复合评分定义(用户 2026-05-01 拍板)

**前置物理 gate**:

| Gate | 定义 | 触发后果 |
|---|---|---|
| **min pairwise distance ≥ 1.5 Å** | 样本预测的 20 原子两两 cartesian 距离最小值 ≥ 1.5 Å | **不通过 → 该样本总分 = 0,其余 6 项不算** |

理由: 1.5 Å 是 EXAFS / FEFF 物理上"原子间最小化学键长度"的下限,违反则结构无法跑 FEFF,等价于物理无效。

**6 项加权评分**(总和为 1.0,gate 未通过则不参与计算):

| # | 项 | 权重 | 容错 | 评分函数 |
|---|---|---|---|---|
| 1 | 第一壳层配位原子数 | 0.20 | ±1.5 个 | 1 if \|n_pred - n_true\| ≤ 1.5 else max(0, 1 - (\|...\|-1.5)/3.0) |
| 2 | 第一壳层距离 | 0.20 | ±0.2 Å | 1 if \|d_pred - d_true\| ≤ 0.2 else max(0, 1 - (\|...\|-0.2)/0.5) |
| 3 | 第一壳层元素种类(CNO 等价) | 0.20 | C/N/O 视为同类 | Multiset 交集 / 总数(替换 C/N/O 为合并 token) |
| 4 | 第二壳层配位原子数 | 0.10 | ±3 个 | 同 #1 容错 ±3 |
| 5 | 第二壳层距离 | 0.10 | ±0.2 Å | 同 #2 |
| 6 | 第二壳层元素种类(CNO 等价) | 0.10 | 同 #3 | 同 #3 |

**总分**: gate 通过 → `Σ w_i × score_i` ∈ [0, 1];gate 未通过 → 0。

**CNO 等价规则**: 实验发现 C(Z=6) / N(Z=7) / O(Z=8) 在 EXAFS 振幅上极接近(散射 amplitude 差 <10%),
随机替换得到的谱图几乎无法区分,因此评分时 C/N/O 合并为同一 token。

### B.3 第一壳层 / 第二壳层边界来源

**用 Exp4 Step 2.5 已有产出**(MA2 时代设计 + 用户拍板):
- 文件: `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl`
- 大小: 387 MB
- md5: `cf2050e4899160f5698ad2481377e94c`
- 算法: 基于训练集 6.28M 邻接距离 gap 的 p10 阈值 = **0.1563 Å**(MA2 拍板)
- Schema (per-sample dict): 9 字段
  ```
  {sample_name (e.g. "mp-555067__mp-...-EXAFS-As-K"): {
    "threshold":      float (= 0.1563 全样本一致),
    "distances":      array of float (该样本所有邻居距中心距离, 升序),
    "species_Z":      array of int (对应原子 Z, 与 distances 同 index),
    "shell_starts":   array of int (各 shell 起始 index),
    "shell_ends":     array of int (各 shell 结束 index, exclusive),
    "shell_n_atoms":  array of int (各 shell 原子数),
    "shell_of_atom":  array of int (每原子的 shell index),
    "eval_cutoff":    float (评估半径,常 ≈ shell-2 上限),
    "n_center_sites": int
  }}
  ```

⚠️ **SA1' fallback bug**: SA1' 写 `step5_2_compute_metrics.py` 投影 ablation helper 时
未读对 per-sample dict 的 lookup,fallback 到 R_max=5.5 Å 全局值。
**SA-METRICS-V3 必修 bug** —— 详见 §B.4。

### B.4 实施细节

**新文件**: `/home/tcat/diffcsp_exp5/code/step5/step5_3_composite_score.py`(SA-METRICS-V3 创建)

**输入**:
- predictions_v2_{val,test}.pt(SA3' 已 sample,from SA2 epoch 484 ckpt)
- predictions_v2_resumed_{val,test}.pt(可选,from SA2'' epoch 529 ckpt,若 Exp5 MA2 决定 SA3'' 重 sample)
- shell_boundaries.pkl(per-sample, 由 sample_name 索引)

**输出**:
- 每 split 一个 `composite_score_{split}.txt`(主报告,含 gate 通过率 + 7 项均分)
- 每 split 一个 `composite_score_per_sample_{split}.csv`(7 项分 + min_d 实际值)
- `min_d_violations_{split}.csv`(min_d < 1.5 Å 样本清单 + 实际 min_d 值,Exp5' 定 lambda 用)

**不动**: step5_2_compute_metrics.py 的 4 个 v2 算法函数(Set-Level / Multiset / Collapse / Projection 保留作历史对照)。
SA-METRICS-V3 写**新**脚本,不改老脚本核心算法。

### B.5 verdict 框架修订(取代原 §5.2)

**原 verdict**(Multiset F1 主信号):
- 🟢 GREEN: > 0.20
- ⚠️ AMBER: 0.10 - 0.20
- ❌ RED: < 0.10

**修订 verdict**(复合分主信号 + min_d gate 通过率副信号):

| 信号 | 阈值 | 含义 |
|---|---|---|
| **复合总分均值(per-sample 平均)** | 主信号 | 7 项加权评分 |
| **min_d gate 通过率**(整 dataset 比例) | 副信号 | 物理有效性 |

| 复合分均值 | min_d gate 通过率 | 档位 | MA 决议方向 |
|---|---|---|---|
| > 0.50 | > 80% | 🟢 GREEN | Exp5 v2 真正成功,SA4' figure 启动 |
| > 0.50 | 60-80% | 🟡 GREEN-but-physical-warn | 数学评分高但物理违反多,Exp5' 物理约束必修 |
| 0.30-0.50 | > 80% | ⚠️ AMBER | 续训或转 Exp5' 决议 |
| 0.30-0.50 | < 80% | ⚠️ AMBER-physical | Exp5' 必修 |
| < 0.30 | any | ❌ RED | 转 Exp6 方向(MV-attention 不通) |

具体阈值 0.50 / 0.30 是 MA5 临时估,**SA-METRICS-V3 跑出 SA2 baseline 复合分后,Exp5 MA2 重新校准阈值**。

---

## §C Exp5'(physical constraint extension)— 新增分支

### C.1 任务

针对 §A 物理约束发现,Exp5' 在 Exp5 v2 SA2'' epoch 529 ckpt 基础上加入物理 loss 重训。

### C.2 设计方向(Exp5 MA2 / Exp5 MA2 设计 proposal)

**Loss 加项**:
```python
# 在 diffusion_w_type_xas.py forward 内新增
def _pairwise_min_distance_penalty(pred_xyz, threshold=1.5):
    """
    pred_xyz: (B*N, 3) cartesian coords (frac × L)
    Penalize pairs with d_ij < threshold within same sample.
    Returns scalar loss.
    """
    # 实现伪码:
    # 1. 按 batch_idx split → 每样本 (N=20, 3)
    # 2. 计算 cdist (N, N) → take upper triangle
    # 3. mask d < threshold
    # 4. loss = relu(threshold - d).pow(2).mean()
    pass

# yaml 加新字段:
# cost_pairwise_min: 0.X  # 待 lambda 调
```

**Lambda 调度初步建议**(SA-METRICS-V3 数据出来后调):
- SA2 baseline min_d 违反率 < 10%: 起步 λ=0.1
- 10-30%: 起步 λ=0.5
- > 30%: 起步 λ=1.0
- 训练前 10 epoch 监控:违反率应单调下降,若 RMSD 同时上升 > 5% 则 λ 减半

**Warm-start**:
- 起点 ckpt: `/home/tcat/diffcsp_exp5/checkpoints/sa2pp_resume_epoch529_val0.7003.ckpt.frozen`
- weights 完全兼容(Exp5' 不改架构,只加 loss 项)
- max_epochs: 200(在 epoch 529 基础上续训到 729),early stop patience=30

**估时**: ~10-15h 训练 + ~3.5h sample(SA-METRICS-V3 把 sample 脚本兼容性已打通)+ 0.5h metrics + 0.5h 报告 ≈ 14-19h 总。

### C.3 Exp5' verdict

**主目标**: min_d gate 通过率 ≥ 90%(物理有效性)+ 复合总分 ≥ SA2 baseline + 0.05(不退步)。

**额外检查**: RMSD ≤ SA2 baseline + 0.05(避免为压 pairwise 距离破坏 RMSD)。

### C.4 与 Exp5 v2 的关系

- Exp5 v2 收尾时不做 §C 工作(等 Exp5 MA2 接手)
- Exp5 v2 final report 写到 SA3'' 续训完成 + SA-METRICS-V3 复合分数据 + Exp5' 待启动状态
- Exp5' 是 Exp5 v2 的 extension,共享 v2 的 codebase / data / ckpt,**不重启 Exp 编号**

---

## §D 关于其他 Exp6 候选(errata 2 §3.2 列的 4 个方向)

Exp5' 完成后再视复合分 verdict 决定:

| Exp5' 后情况 | Exp5 MA2 后续方向 |
|---|---|
| Exp5' 复合分 > 0.50 + min_d > 90% | Exp5/Exp5' 阶段任务完成,SA4' figure + Exp6 proposal 自由开题 |
| Exp5' 复合分 0.30-0.50 + min_d > 90% | 物理 OK 但精度不够,Exp6 候选: distance-aware loss(errata 2 方向 4)/ CFG(方向 9) |
| Exp5' 复合分 < 0.30 或 min_d 仍 < 90% | MV-attention + 物理约束组合不通,Exp6 大改架构: hierarchical type / equivariant decoder |

---

## §E 给 Exp5 MA2 的 Lessons Learned(写进 ExpN 不变量级)

1. **min pairwise distance 检测必须进 ExpN 不变量** — 任何使用 diffusion 生成原子坐标的 Exp,
   评估体系**必须**包含 min pairwise distance 检测,1.5 Å 作为物理硬下限。
   写进 EXP4_FINAL_REPORT_ERRATA_2.md §4 Lessons Learned 第 5 条(我无 push 权限,Exp5 MA2 接手时补)。

2. **复合评分 vs 抽象 multiset metrics** — Set-Level / Multiset Macro-F1 是数学完备的解耦指标,
   但**不是物理评分**。物理 motivated 6+ 项评分(数量 / 距离 / 元素 / shell-1 / shell-2)
   closer to 化学家判断,Exp6+ 应默认采用复合体系作主信号,multiset 仅供历史对照。

3. **shell_boundaries.pkl 是 Exp4 已有 ground truth** — 任何后续 ExpN 评估**禁止**用 fallback
   常数(如 5.5 Å)替代 per-sample 边界。SA1' 这次就是 fallback bug 隐藏到 SA3' 投影 ablation 才暴露。

4. **投影 ablation 不是物理完备性证明** — Δ=0 + 0 原子需投影 **只**证明 pred 在 R_max 球壳内,
   **不**证明 pred 内部物理合理。物理完备性需要 min pairwise distance 检测补足。

5. **MA review SA 设计时,要主动质疑指标 menu 的完备性** — 我 review SA1' 4 个 metric 函数时
   只检查算法实现是否正确,没问"这 4 项加 Exp4 已有的 RMSD/TypeAcc 共 6 项,够不够覆盖所有 collapse 模式"。
   Exp5 MA2+ review 类似设计应主动补这一问。

---

## §F 修订后的 Exp5 v2 proposal 不变量(给 Exp5 MA2 速查)

继承自原 proposal v2 §2,加 §B 评估修订 + §C Exp5' 路径:

| 项 | 值 / 状态 |
|---|---|
| 主线 1 架构 | MV-attention encoder(num_heads=4, residual_alpha=0.5,完成,SA1') |
| 主线 2 loss | cost_density 0.5 → 0.2(完成,SA1') |
| 不加 head | ✓(完成,SA1' 撤销 v1 head) |
| Center embedding | ✓(继承 v1 SA1, n_center_elements=95, center_emb_dim=16) |
| precision | fp32 |
| from-scratch | ✓ SA2' epoch 484 best val=0.7065 |
| 续训 from best | ✓ SA2'' epoch 529 val=0.7003(改进 0.88%) |
| **NEW: 评估主信号** | **复合总分(7 项加权 + min_d 1.5 Å gate)** |
| **NEW: 主验收阈值** | **复合分均值 > 0.50 + min_d gate > 80%** |
| **NEW: Exp5' 触发** | **复合分 < 0.50 OR min_d < 80% → 加 pairwise penalty 重训** |

---

*MA5 撰写,2026-05-01。基于 SA1'/SA2'/SA3'/SA2'' 全程产出 + 用户 2026-05-01 物理约束发现 +
shell_boundaries.pkl(md5 cf2050e4...)Exp4 Step 2.5 ground truth 锚定。Exp5 MA2 接手见 EXP5_FILE_GUIDE_FINAL.md + EXP5_MA2_HANDOFF.md。*
