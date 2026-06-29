# EXP5_DOUBLE_PRIME_MA_HANDOFF.md
# Exp5'-MA → Exp5''-MA 接班文件清单

> **From**: Exp5'-MA(Exp5 系列第 3 任 Main Agent)
> **To**: Exp5''-MA(Exp5 系列第 4 任 Main Agent)
> **日期**: 2026-05-09
> **任务范围**: 候选 A(distance-supervised shell loss)落地实施 + 训练 + verdict
> **预期工作量**: 1-2 天工程 + 10-14h GPU + 1-2h 评估
> **SA 数量**: **0**(Exp5''-MA 直接做,详 §3)

---

## §1 给 Exp5''-MA 接班开场白(直接 paste 到新窗口)

> 你是 DiffCSP-Exp5''-MA(Exp5 系列第 4 任 Main Agent)。
>
> Exp5' 阶段已结束(7 棒 SA + 5 份 errata + final report v3 完成,verdict mixed:fold + pairwise GREEN / shell RED)。Exp5'' 接力做"loss 函数级微调":重设计 shell loss 修复鸡蛋启动问题,目标 composite (step5_3) 从 0.080 → ≥ 0.30 GREEN。
>
> ### 你不开 SA(0 个)
>
> Exp5'' 任务范围明确(改 1 个 loss 公式 + warm-start 训练 + sample),所有"未知 unknown"已通过 Exp5' 5 份 errata 解决。你直接干。
>
> 例外触发开 SA(详 proposal §0.5):
> - P5 verdict 反不如 Exp5'(回退) → 开 SA-EXP5''-AUDIT 验尸
> - P3 smoke n_active < 0.95(候选 A 也踩鸡蛋) → 开 SA-EXP5''-ABLATION 切 B 或 sigmoid soft mask
> - 其他"我没预料到"的 bug
>
> 不预先开 SA。你跑到出问题再开。
>
> ### 你的 6 步任务
>
> P1 改 shell loss 公式(0.5 天)
> P2 forward_test Phase 6.7 重跑(0.3 天)
> P3 smoke + n_active 验证(0.3 天 + 30min GPU)
> P4 warm-start 训练(8-12h GPU 无人值守)
> P5 sample 三 split + step5_3(0.3 天 + 1.5h GPU)
> P6 final report v4 或 v3 §X 附录(0.5 天)
>
> ### 8 份必读(按顺序读)
>
> 1. EXP5_PRIME_MA_HANDOFF.md — Exp5' 接班背景(MA5 → Exp5'-MA → 你)
> 2. EXP5_PRIME_PROPOSAL.md — Exp5' proposal(三件套设计原意)
> 3. EXPERIMENT5_PRIME_FINAL_REPORT_v3.md ⭐ — Exp5' 完整阶段总结
> 4. EXP5_FILE_GUIDE_v2.md — 服务器索引(沿用 Exp5')
> 5. EXP4_FINAL_REPORT_ERRATA_2.md — `_density_loss` 旧归因
> 6. EXP4_FINAL_REPORT_ERRATA_3.md — fold + L=20 决议
> 7. EXP5_PRIME_FINAL_REPORT_ERRATA_4.md — Ckpt selection bug + verdict 双指标 SOP
> 8. EXP5_PRIME_FINAL_REPORT_ERRATA_5.md ⭐ — **核心**:Shell loss 鸡蛋问题 + Exp5'' 方向决议
> 9. EXP5_DOUBLE_PRIME_PROPOSAL.md ⭐ — **本任务核心规格**,候选 A 主线 + B fallback + 训练计划 + verdict 阈值
>
> 服务器: scsmlnprd02.its.auckland.ac.nz / mlff env
> 工作目录: /home/tcat/diffcsp_exp5_double_prime/(你 P1 第一步建,空目录起)
> Exp5' 永久档案: /home/tcat/diffcsp_exp5_prime/(只 read,不 write)
>
> ### 启动后第一条回复格式(沿用 Exp5'-MA 习惯)
>
> ```
> 我已读完 9 份必读文档。复述任务要点 [6-8 条]
> (含为什么选候选 A 不选 B / cheating 合规论证 / warm-start 起点 / verdict 阈值)
> 4 个最易踩坑点
> 第 1 步 ssh 跑 §7.1 的 6 段 verify 命令
> ```
>
> ### 已拍板的 8 条不再讨论
>
> 1. **候选 A 主线**(distance-supervised KNN 切片),B 是 fallback,不预先跑 B
> 2. **Warm-start from `composite_epoch169_score0.5881.ckpt`**(Exp5' BEST,md5 `127afa44a850d8f7e4fcdae17e2761a1`)
> 3. **Cost 不动**(0.5 / 0.2),公式改不改 cost
> 4. **架构 / 训练超参全部沿用 Exp5'**(batch=64 / lr=1e-4 / scheduler T_max=500 / patience=30 / strict=False / save_top_k=3)
> 5. **`_pairwise_min_distance_penalty` 不动**(λ=1.0,Exp5' 生效硬证)
> 6. **`_density_loss` 不动**(cost=0.2,errata 2 揭示是塌缩剂但 Exp5' 沿用 OK)
> 7. **shell_band_width=1.0 Å**(`_shell_count_loss_v2` band 容差,起步值,verdict 不达标再 ablation)
> 8. **Exp5'' 工作目录独立** /home/tcat/diffcsp_exp5_double_prime/(不混 Exp5' 永久档案)
>
> ### 关键 ping 点
>
> 你不开 SA,所以"ping" 是 ping 用户(决策接收方)。每个关键节点必须暂停 + 给用户报告 + 等用户 ack:
>
> - **P3 smoke 完成**:报 val_n_active_shell_*_ratio,< 0.95 等用户决议
> - **P4 epoch 5 完成**:报 6 active loss + composite 趋势,触发条件参考 proposal §4.3
> - **P4 epoch 30 完成**:报 pred shell-1 mean dist 是否朝 true 方向收敛
> - **P4 epoch 100 完成**:报 composite 是否超过 Exp5' BEST 0.5881
> - **P5 sample 完成**:报 step5_3 verdict 双指标
> - **P6 hand-back**:写 final report v4 或 v3 §X
>
> 任何不确定 → 贴日志,不靠记忆 + 让用户跑 verify 命令(沿用 Exp5'-MA 工作哲学)。

---

## §2 必读 9 份文件清单(给用户 paste 到新窗口前确认)

| # | 文件 | 来源 / 状态 | Exp5''-MA 用法 |
|---|---|---|---|
| 1 | `EXP5_PRIME_MA_HANDOFF.md` | 你已有 | Exp5' 接班背景,Exp5''-MA 知道整个系列脉络 |
| 2 | `EXP5_PRIME_PROPOSAL.md` | 你已有 | Exp5' proposal,知道三件套 loss 原意(Exp5'' 改其中 2 个)|
| 3 | `EXPERIMENT5_PRIME_FINAL_REPORT_v3.md` | 上回合 outputs | ⭐ Exp5' 完整阶段总结,Exp5'' 的 baseline |
| 4 | `EXP5_FILE_GUIDE_v2.md` | 你已有 | 服务器索引,Exp5'' 沿用 PYTHONPATH 三段 |
| 5 | `EXP4_FINAL_REPORT_ERRATA_2.md` | 你已有 | `_density_loss` 旧归因,Exp5'' 不动 density loss |
| 6 | `EXP4_FINAL_REPORT_ERRATA_3.md` | 你已有 | fold + L=20 决议,Exp5'' 沿用 |
| 7 | `EXP5_PRIME_FINAL_REPORT_ERRATA_4.md` | 上回合 outputs | Ckpt callback bug 修复(strict=False),Exp5'' 沿用 |
| 8 | `EXP5_PRIME_FINAL_REPORT_ERRATA_5.md` | 上回合 outputs | ⭐ 鸡蛋问题诊断 + Exp5'' 方向决议(由本 proposal 落地为 A 主线)|
| 9 | `EXP5_DOUBLE_PRIME_PROPOSAL.md` | **本回合 outputs** | ⭐ 任务核心规格,Exp5''-MA 实施依据 |

**总:9 份**(7 份你已有 + 2 份本回合新出 + 1 份 final report v3 上回合新出 = 实际新加 3 份)

---

## §3 SA 数量决议

### 3.1 决议:**0 个 SA**

Exp5''-MA 直接执行 P1-P6。

### 3.2 决议理由

**Exp5''-MA 直接做的成本**:
- 1-2 天 wall clock(P1+P2+P3+P5+P6 工程,P4 GPU 无人值守不占 wall clock)
- 1 个 MA 自己处理所有节点
- 0 个 hand-back 文档(只有最终 P6 final report v4)

**开 SA 的成本**(对比):
- 至少 1 棒 SA(SA-EXP5''-IMPL),需要写 launch note(~ 200 行)+ SA 复述 + ping 点
- 多了 1 层抽象,错配风险增加(SA 可能误改不该动的 cost / 超参)
- 节省的"MA 自己写代码"时间 < SA hand-back review 时间

**Exp5'' 任务为什么不需要 SA 的理由清单**:
- ✅ 任务范围明确(改 2 个 loss 公式,proposal §2.2-2.3 已给完整 pseudocode)
- ✅ 所有 unknown 已 errata 化(fold / ckpt bug / shell 鸡蛋,5 份 errata 全覆盖)
- ✅ 无架构改动(model 参数全沿用)
- ✅ 无 dataset 改动(cache 不重建)
- ✅ 无训练超参改动(全沿用)
- ✅ 评估流程沿用(step5_3 不改)

**Exp5' 用 SA 的理由(对比清单)**(都不适用 Exp5''):
- ❌ Exp5' 启动时 fold artifact 未知 → 需 SA-AUDIT
- ❌ Exp5' 三件套 loss 是新设计 → 需 SA-IMPL
- ❌ Exp5' 训练超参待定 → 需 SA dry-run 探索
- ❌ Exp5' ckpt callback 未知有 bug → 需 SA 训练实战暴露

Exp5'' 这些都已知。

### 3.3 例外:Exp5'' 跑到失败再开 SA

| 触发 | 开什么 SA |
|---|---|
| P5 verdict 全 RED(候选 A 完全失败)| **SA-EXP5''-AUDIT**(类比 STEP1-AUDIT,验尸根因)|
| P3 smoke n_active < 0.95(boolean mask 梯度阻断)| **SA-EXP5''-ABLATION**(切 sigmoid soft mask 或候选 B)|
| P4 训练 NaN/Inf 异常 | **SA-EXP5''-DEBUG** |

这些都是反应式开 SA,不预先开。

---

## §4 给用户(你)的 4 件事

### 4.1 review final report v3 + errata 5 + Exp5'' proposal 三份

- final report v3 上回合已落 outputs,~ 568 行
- errata 5 上上回合已落 outputs,~ 324 行
- **Exp5'' proposal 本回合落 outputs**,~ 664 行

如有改动,告诉我具体段号。否则我视为通过。

### 4.2 决定 Exp5''-MA 接班时间

- 立即接班:你今天 / 明天就让新 MA 上手
- 缓接班:你想 review 几天 / 让师兄看 → 等师兄反馈后再启动

我倾向缓接班 1-3 天,理由:
- final report v3 + proposal 信息密度高,师兄 review 可能给新见解(比如"shell_band_width 我建议 0.5 不是 1.0")
- 缓 3 天对 Exp5'' verdict 影响 ≈ 0(Exp5'' 是 1-2 天工程不是周级)
- Exp5'-MA(我)在缓的几天可以做 errata 5 / final report v3 的二次校对

### 4.3 师兄 review 提示

师兄看 final report v3 时可能问的 3 个问题:
1. **"为什么 fold artifact 这么晚才发现?"** → final report v3 §6.4 + §13 已答(watch-only 机制对"梯度无效"型 bug 不灵敏 + 用户本能 3 次救场)
2. **"shell loss 鸡蛋问题为什么不早发现?"** → §6.4 已答(数值 finite + n_active 没 dump)
3. **"Exp5'' 候选 A 为什么不算 cheating?"** → proposal §2.4 已答(标量 label vs pixel coordinates 区分)

### 4.4 Exp5''-MA 是同一个 Claude 还是新窗口

技术上看是新窗口(对话上下文清零)。但 Exp5''-MA 只是身份标签,实际是 Claude 新会话用 9 份必读 ramp up。

**给新 MA 的开场白(§1)直接 paste 到新窗口即可**,不需要额外引导。

---

## §5 完成 Exp5'' 之后(给你心里有底)

| Verdict | 后续 |
|---|---|
| All GREEN | 写 Exp5'' final report v4(或 v3 §X 附录),投全长 paper |
| Mixed | 写 Exp5'' final report v4,投 short paper / workshop |
| Failure(候选 A 也 RED)| 切候选 B(再 1-2 天)or 转 Exp6 架构级(4-8 周) |

如 Mixed 或 Success,Exp5 系列**正式 wrap up**。Exp6 是另一段故事(equivariant decoder),不在本系列。

---

*Exp5'-MA 撰写,2026-05-09*
*接班准备完成。Exp5''-MA 直接执行,0 个 SA。*
