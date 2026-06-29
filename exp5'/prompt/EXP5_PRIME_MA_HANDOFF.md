# EXP5_PRIME_MA_HANDOFF.md
# Exp5' MA 接班 Handoff(MA5 → Exp5'-MA)

> **From**: MA5(Exp5 v2 Main Agent,即将上下文 70% 闸门 transition)
> **To**: Exp5'-MA(下一棒 Main Agent,执行 Exp5' from-scratch 重训 + 三件套物理 loss)
> **Date**: 2026-05-01
> **Status**: Exp5 v2 verdict ❌ physical-invalid,Exp5' from-scratch 已锁定方向
> **本文档定位**: Exp5'-MA 一文上手,读完 + 4 份 critical 文件,立即可写第一棒 SA handoff

---

## §0 一屏掌握

### 0.1 你是谁,做什么

你是 Exp5'-MA(Exp5 系列第 3 任 Main Agent,前任: v1-MA → v2-MA5 → 你)。

Exp5 v2 已完成训练 + 评估改造,**verdict 是物理灾难**(95% 样本预测原子两两距离 < 1.5 Å,即原子重合无法跑 FEFF)。用户拍板**from-scratch 重训**(不 warm-start),加三件套物理 loss + 把 Exp4 Step 2.5 已有的 shell ground truth 真正 inject 进训练。

你的任务:
1. 写 **Exp5' Step 1 SA handoff**(架构 surgery + 物理 loss 实现 + smoke + forward_test,~ 1-2 天)
2. 之后顺序:**SA-EXP5'-train**(~ 32-40h from-scratch 训练)→ **SA-EXP5'-sample**(~ 7h sample val/test + Exp4 物理对照)→ **SA-EXP5'-figure-final**(figure + final report v3)
3. 全程跨 6-8 个对话窗口,你可能也要 transition 到下一棒 MA。

### 0.2 必读 4 份(按顺序)

1. **EXP5_PRIME_PROPOSAL.md** — Exp5' 详细 proposal,三件套 loss 公式 + verdict 阈值 + 工作目录决策。**你所有工作的锚点**。
2. **EXPERIMENT5_FINAL_REPORT_v2.md** — Exp5 v2 final 历史档案,§0-§4 必看(verdict / 接力链 / 数据处理沿用清单)
3. **EXP5_FILE_GUIDE_v2.md** — 完整服务器/本地索引,**§9 verify 块照着跑一遍**作启动 sanity
4. **EXP4_FINAL_REPORT_ERRATA_2.md** — `_density_loss` 塌缩根因 + Exp3 真实历史(MA5 的根基,你也要内化)

**不读**(浪费时间): SA1' / SA2' / SA3' 各自全文 OUTPUT(摘要在 final report v2 §1-§2)、EXP5_PROPOSAL_v2.md 原版(被 v2 final report 取代)、SA-METRICS-V3 EARLY_HANDBACK 全文(关键数据已在 final report v2 §0)。

### 0.3 启动后第一条回复请按以下格式

```
我已读完 4 份必读文档。复述 Exp5' 当前状态 + 任务方向:
[列 6 条:
 - Exp5 v2 verdict + 关键数字
 - Exp5' from-scratch 决策 + 不 warm-start
 - 三件套物理 loss(pairwise + shell_dist + shell_count)
 - shell_boundaries.pkl 真正进训练
 - 沿用 MV-attention + center_emb + cost_density 0.2
 - 工作目录新建 /home/tcat/diffcsp_exp5_prime/]

我注意到 4 个最容易出错的点:
[列 4 条,例如:
 - shell_boundaries inject 进 dataset 工程复杂(新 5 字段 collate)
 - gap 算法在 pred 重合数据上 numerical 不稳(epoch 0-10 注意 NaN)
 - best ckpt selection 不能只看 val_loss(必须用复合 score)
 - PYTHONPATH 必须 exp5_prime 不是 exp5(避免误用 v2 .py)]

我下一步:
[列 SA Step 1 handoff 撰写计划 + 第 1 件让用户做的 verify 命令]
```

---

## §1 Exp5' 当前状态(给你速查)

### 1.1 已完成(MA5 移交时锁定)

| 阶段 | 产出 |
|---|---|
| Exp5 v2 训练(SA2' + SA2'')| epoch 529 best val_loss 0.7003,4 个 ckpt(2 active + 2 frozen)|
| Exp5 v2 sample + 数学评估(SA3')| Multiset F1 0.1086(val)/ 0.1096(test),predictions_v2_*.pt 在服务器 |
| Exp5 v2 物理评估(SA-METRICS-V3 dry-run)| **95% 样本物理违反 + 复合分 0.005-0.011**,verdict ❌ |
| Exp5' 启动 4 件套(MA5 临走)| 本文件 + proposal + final report v2 + file guide v2 |

### 1.2 待启动(你的任务)

| 阶段 | 估时 | 说明 |
|---|---|---|
| **Exp5' Step 1: 架构 + loss 实现 + smoke**(本文 §3 Step 1.0-1.7) | 1-2 天 | 你写 SA handoff,SA 实施 |
| Exp5' Step 2: from-scratch 训练 | ~ 32-40h | SA-EXP5'-train |
| Exp5' Step 3: sample + 复合分 + Exp4 对照 | ~ 7h sample + 0.5h metrics | SA-EXP5'-sample |
| Exp5' Step 4: 6 figure + final report v3 | ~ 5h | SA-EXP5'-figure-final |

总 ~ 50h 跨 6-8 对话窗口,你大概率也 transition 到下一棒 MA。

### 1.3 服务器 active 资产(给你速查)

```
/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl   ⭐ 387 MB Step 2.5 ground truth
/home/tcat/diffcsp_exp5/                            ⭐ Exp5 v2 历史档案,fork 起点
/home/tcat/diffcsp_exp5_prime/                      ⏳ 你建,空目录起 from-scratch
```

---

## §2 第一棒 SA-EXP5'-STEP1 任务规格(你写 launch note 用)

### 2.1 任务范围(1-2 天工程)

新建 `/home/tcat/diffcsp_exp5_prime/` + cp Exp5 v2 code + 改造 6 处:

1. **dataset / datamodule** — shell_boundaries.pkl inject 进 batch(5 字段)
2. **model** — `diffusion_w_type_xas.py` 加 3 个新 loss 函数(pairwise_min / shell_dist / shell_count)+ forward 调用 + 5 个新 output 字段
3. **yaml** — 加 3 个 cost_* 字段
4. **train.py** — 去 last_ckpt 硬编码(from-scratch);加 ckpt selection callback(α/β/γ = 0.2/0.5/0.3)
5. **forward_test** — 加 Phase 6.7 测三新 loss
6. **smoke test** — 6 active loss 字段验证

详细规格在 EXP5_PRIME_PROPOSAL.md §2(三件套 loss 精确公式)+ §4(实施步骤 Step 4.1-4.6)。

### 2.2 7 项任务清单(对应 Exp5' Step 1 子任务)

| 步 | 任务 | 工程量 |
|---|---|---|
| 1.0 | 服务器 mkdir + cp Exp5 v2 code 树 + symlink data | 10 分钟 |
| 1.1 | dataset_v2.py 加 shell_boundaries inject | 0.5 天 |
| 1.2 | datamodule_v2.py collate 加 5 字段 | 0.3 天 |
| 1.3 | diffusion_w_type_xas.py 加 3 loss 函数 + forward 调用 | 0.5 天 |
| 1.4 | yaml 加 3 cost_* 字段 | 5 分钟 |
| 1.5 | train.py ckpt selection callback + from-scratch | 0.3 天 |
| 1.6 | forward_test.py 加 Phase 6.7 + 6.4 loss range 调整 | 0.3 天 |
| 1.7 | smoke test 改写 + 跑 2 epoch × 10 batch PASS | 0.3 天 |
| 1.8 | SA-EXP5'-STEP1 中期报告 → 你 review → 启动 SA-EXP5'-train | 0.2 天 |

### 2.3 PASS gates(SA-EXP5'-STEP1 hand-back 必过)

- forward_test: 5/5 PASS + 1 SKIPPED + Phase 6.7 PASS(三新 loss 在 dummy batch 上 finite + cost_* yaml 加载)
- smoke test: 2 epoch × 10 batch,6 active loss 全 finite,无 NaN/Inf,best ckpt selection callback 触发
- shell_boundaries inject 验证: dataset 第 1 个 batch 包含 5 个新字段,collate 输出 (B,) shape 正确

### 2.4 红线(SA-EXP5'-STEP1 全程不动)

| | |
|---|---|
| ❌ 不动 holdout | |
| ❌ 不升级 7 守卫包 | |
| ❌ 不动 Exp5 v2 ckpt(.frozen 永久,active 历史档案) | |
| ❌ 不动 Exp4 backbone(`/home/tcat/diffcsp_exp4/code/diffcsp/`)| |
| ❌ 不修 Phase 6.5 hardcoded fp32 | |
| ❌ 不动 step5_2_compute_metrics.py(留作 v2 历史档案) | |
| ❌ 不动 step5_3_composite_score.py(SA-METRICS-V3 产出,Exp5' 沿用) | |
| ❌ 不启动正式训练(SA-EXP5'-STEP1 只跑 smoke + forward_test) | |
| ❌ 不擅自调三件套 λ(proposal §2.1-2.3 起步值锁定: 1.0 / 0.5 / 0.2) | |

---

## §3 Step 1 详细子任务(给你写 launch note 模板)

### 3.1 Step 1.0 — 服务器目录建立

EXP5_FILE_GUIDE_v2.md §6 已给完整 mkdir + cp + symlink 命令,直接复用。

### 3.2 Step 1.1 — dataset_v2.py inject

**改动文件**: `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py`

**改动逻辑**:

```python
# __init__ 加:
SHELL_BOUNDARIES_PATH = "/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl"

class XasLocalDatasetV2:
    def __init__(self, ...):
        ...
        # Exp5': load shell_boundaries.pkl 一次到内存(387 MB)
        import pickle
        with open(SHELL_BOUNDARIES_PATH, 'rb') as f:
            self._shell_boundaries = pickle.load(f)
        logger.info(f"  loaded shell_boundaries: {len(self._shell_boundaries)} samples")

    def __getitem__(self, idx):
        ...
        sample_name = ...  # 已有逻辑
        sb_i = self._shell_boundaries.get(sample_name, None)
        if sb_i is None:
            return None  # 接 silent_drop 逻辑

        # 提取 5 字段
        shell_n = sb_i['shell_n_atoms']
        shell_of_atom = sb_i['shell_of_atom']
        distances = sb_i['distances']

        # shell-1: shell_of_atom == 0
        shell1_mask = (shell_of_atom == 0)
        shell1_distances = distances[shell1_mask]

        true_shell1_d_mean = float(shell1_distances.mean()) if len(shell1_distances) > 0 else 0.0
        true_shell1_n      = int(shell_n[0]) if len(shell_n) > 0 else 0

        # shell-2: shell_of_atom == 1
        if len(shell_n) > 1 and shell_n[1] > 0:
            shell2_mask = (shell_of_atom == 1)
            shell2_distances = distances[shell2_mask]
            true_shell2_d_mean = float(shell2_distances.mean())
            true_shell2_n      = int(shell_n[1])
            has_shell2         = True
        else:
            true_shell2_d_mean = 0.0
            true_shell2_n      = 0
            has_shell2         = False

        # 塞进 Data 对象
        data.true_shell1_d_mean = torch.tensor(true_shell1_d_mean, dtype=torch.float32)
        data.true_shell2_d_mean = torch.tensor(true_shell2_d_mean, dtype=torch.float32)
        data.has_shell2         = torch.tensor(has_shell2, dtype=torch.bool)
        data.true_shell1_n      = torch.tensor(true_shell1_n, dtype=torch.long)
        data.true_shell2_n      = torch.tensor(true_shell2_n, dtype=torch.long)

        return data
```

### 3.3 Step 1.2 — datamodule collate

PyG `Batch.from_data_list` 自动处理张量字段的 batching。SA-EXP5'-STEP1 验证 collate 输出 5 个新字段 shape = (B,)。

### 3.4 Step 1.3 — model 加 3 loss 函数

EXP5_PRIME_PROPOSAL.md §2 已给完整 pseudocode,SA-EXP5'-STEP1 实施时:

- 三函数都加在 `class CSPDiffusion` 内(static method 或 instance method 均可)
- forward() 内调用三函数后加进 total_loss
- output dict 加 5 字段: `loss_pairwise_min / loss_shell_dist / loss_shell_count / val_min_d_mean(epoch-level)/ val_gate_pass_rate(epoch-level)`

### 3.5 Step 1.4 — yaml 改

```yaml
# 在 step3/conf_xas/model/diffusion_xas.yaml 末尾加:
cost_pairwise_min: 1.0   # Exp5' 起步
cost_shell_dist:   0.5   # Exp5' 起步
cost_shell_count:  0.2   # Exp5' 起步
```

### 3.6 Step 1.5 — train.py 改

(a) 删除 SA2'' 时 MA5 加的 `last_ckpt = .../epoch=484-...` 硬编码,改回 `ckpt_path = None`(from-scratch)

(b) 加 `CompositeBestCkptCallback` 类:

```python
class CompositeBestCkptCallback(pl.Callback):
    """Exp5' best ckpt selection: weighted combination of 3 metrics."""
    def __init__(self, alpha=0.2, beta=0.5, gamma=0.3, dirpath='/home/tcat/diffcsp_exp5_prime/checkpoints'):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.best_score = -1e9
        self.dirpath = dirpath

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        val_loss = float(m.get('val_loss', 1.0))
        gate_pass = float(m.get('val_gate_pass_rate', 0.0))
        pairwise = float(m.get('val_pairwise_min_loss', 1.0))

        score = (self.alpha * (1.0 - val_loss / 1.0)
               + self.beta  * gate_pass
               + self.gamma * (1.0 - pairwise))

        if score > self.best_score:
            self.best_score = score
            ckpt_path = f'{self.dirpath}/composite_best_score{score:.4f}_epoch{trainer.current_epoch}.ckpt'
            trainer.save_checkpoint(ckpt_path)
            print(f"[CompositeBest] new best score={score:.4f} @ epoch {trainer.current_epoch}")
```

(c) trainer.callbacks 列表加 `CompositeBestCkptCallback()` 实例

### 3.7 Step 1.6 — forward_test Phase 6.7

新加 Phase 6.7 — Exp5' 三 loss + yaml cost_* 验证:

```python
def phase_67(batch_cpu):
    log("Phase 6.7 — Exp5' three new physical loss functions")

    # 6.7.a: 三函数存在
    model = _instantiate_model()
    assert hasattr(model, '_pairwise_min_distance_penalty')
    assert hasattr(model, '_shell_distance_loss')
    assert hasattr(model, '_shell_count_loss')

    # 6.7.b: dummy batch min_d=0.5 → pairwise_min_loss > 0
    # construct dummy batch with intentional overlap
    ...
    loss = model._pairwise_min_distance_penalty(...)
    assert loss.item() > 0

    # 6.7.c: dummy batch min_d=2.0 → pairwise_min_loss = 0
    ...
    assert loss.item() < 1e-6

    # 6.7.d: yaml cost_pairwise_min=1.0 / cost_shell_dist=0.5 / cost_shell_count=0.2
    # 加载到 self.cost_pairwise_min / etc
    assert abs(model.cost_pairwise_min - 1.0) < 1e-6
    assert abs(model.cost_shell_dist - 0.5) < 1e-6
    assert abs(model.cost_shell_count - 0.2) < 1e-6
    log("[Phase 6.7 PASS]")
```

### 3.8 Step 1.7 — smoke test

跑 2 epoch × 10 batch,确认 6 个 active loss(coord / type / density / pairwise_min / shell_dist / shell_count)全 finite,best ckpt callback 触发。

---

## §4 Lambda 调度(给 SA-EXP5'-train 准备)

EXP5_PRIME_PROPOSAL.md §2.1 已锁定起步 + 调度规则。SA-EXP5'-train 训练中**Exp5'-MA epoch-level 监控**(不让 SA 自己调 λ,Exp5'-MA 决议):

- epoch 0-2: λ_pairwise=1.0(不动)
- epoch 3-5: 你看 violation rate
  - 单调下降到 < 50%: 维持 1.0
  - 卡 > 70%: ramp 到 2.0(用户 ack)
  - RMSD 飙升 > Exp5 v2 + 10%: 减半重启
- epoch 5+: violation < 30%: 降回 0.5 平衡其他 loss

**SA-EXP5'-train 不擅自调 λ,有任何异常 ping 你**。

---

## §5 红线汇总(全程)

| 红线 | 说明 |
|---|---|
| ❌ 不动 holdout | 永久封存 |
| ❌ 不升级 7 守卫包 | 全程 |
| ❌ 不动 Exp5 v2 ckpt | 永久档案 |
| ❌ 不修 Phase 6.5 fp32 | 永久 SKIPPED |
| ❌ 不擅自调 λ | SA 不动,你监控决议 |
| ❌ 不动 step5_3_composite_score.py | SA-METRICS-V3 产出,Exp5' 沿用 |
| ❌ 不动 holdout(SA-EXP5'-sample 仅 sample val + test + Exp4 对照,不动 holdout)| |
| ❌ 你写完 SA handoff 必先用户 review | MA 工作哲学 |
| ❌ 70% 上下文闸门是硬线 | 接近时主动 transition |
| ❌ 任何不确定的事 → 写脚本让用户 confirm,不靠记忆 | 用户原话 |
| ❌ 任何技术判断先 conversation_search + 列证据 | MA 工作哲学 |

---

## §6 给你的 8 条 Lessons(MA5 移交)

1. **数学完备 ≠ 物理完备** — Multiset F1 等数学指标不是物理评分。Exp5' 主指标用 step5_3 复合分。
2. **Min pairwise distance 是 ExpN 不变量** — 任何 diffusion 生成原子坐标的 Exp,评估必含 1.5 Å gate。
3. **Step 2.5 ground truth 应进训练,不只评估** — Exp5' 是这个 lesson 的 first implementation。
4. **MA review SA 设计时主动质疑指标完备性** — 不止 verify 算法正确性。
5. **训练目标没要求的事,模型不会自己学** — 设计 loss 时先列"模型应学什么物理性质",再 invert 到 loss 项。
6. **小补丁也要 MA ack + diff** — 任何 surgery 都贴 diff,scope 严守。
7. **用户的物理统计是 ground truth** — 算法/metric 都不是。
8. **Step6 picker subset 不是 verdict** — Exp5' final report 必须报告全 7621/4481 样本的 gate pass 率。

---

## §7 当前 transition 状态

| 项 | 状态 |
|---|---|
| MA5 上下文 | 接近 70% 闸门,主动 transition |
| Exp5' 决策 | 全部锁定(from-scratch + 三件套 loss + 沿用 v2 架构) |
| 4 件套交接文档 | 全部 落 outputs/(用户上传给你) |
| Exp5' 工作目录 | 待你 mkdir(EXP5_FILE_GUIDE_v2.md §6 命令) |
| Exp5'-MA 立刻动作 | 让用户跑 EXP5_FILE_GUIDE_v2.md §9 verify → 看输出 → 写 SA-EXP5'-STEP1 launch note |

---

## §8 final 移交宣告

**MA5 移交,Exp5'-MA 接手**。

预计 Exp5'-MA 一棒到 Exp5' final report v3:
- Step 1(架构 + loss 实现): 1-2 天
- Step 2(训练): 32-40h
- Step 3(sample + metrics): 7h
- Step 4(figure + report): 5h
- 总 ~ 50h 跨 6-8 对话窗口

Exp5' 真正完成后,Exp5 系列正式 close 或继续开 Exp5''(architecture ablation,sea Exp5'-MA 决议)。

**MA5 离场。祝好。**

---

*MA5 撰写,2026-05-01,移交 Exp5'-MA 前最后一份 deliverable。
基于 SA1'/SA2'/SA3'/SA2''/SA-METRICS-V3 全程产出 + 用户 2026-05-01 物理统计发现 + from-scratch 重训决策 + 三件套 loss 设计。*
