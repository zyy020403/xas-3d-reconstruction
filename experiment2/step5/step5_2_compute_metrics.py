# step5_2_compute_metrics.py
# Step 5.2 — Holdout 集评估（盲测，禁止干预结果）
# ============================================================
# 改自 step4b_4_compute_metrics.py (v5)，修改两处：
#   1. 输入路径 → experiment2/step5/predictions_holdout.pt
#   2. 输出路径 → experiment2/step5/metrics_holdout.txt
#   3. L=6.0（与 Step4d / xas_local_dataset_L6.py 一致）
#
# 评估方法：最小镜像匈牙利匹配（与 Step4b v5 完全相同）
# 严格禁止：查看结果后反向调整模型——完整汇报所有数字
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP5_DIR    = os.path.join(EXP2_ROOT, "step5")
L = 6.0


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=6.0):
    """
    全部 20 个预测原子 vs 20 个真实原子做匈牙利匹配。

    坐标约定（Step4d / xas_local_dataset_L6.py）：
      pred_frac, true_frac 均在 [-0.5, 0.5]（min-image 折叠后）

    匹配和 RMSD：使用最小镜像距离（周期边界条件）
      delta = p - t;  delta -= round(delta);  dist = ||delta * L||
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)

    n = pred_frac.shape[0]  # 固定为 20

    # ── 构建 20×20 最小镜像距离矩阵 ──────────────────────────────────────
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac   # (20, 3)
        delta -= np.round(delta)            # 最小镜像
        cost_matrix[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── RMSD：用匹配后的最小镜像距离 ─────────────────────────────────────
    matched_dists_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_dists_sq.append(np.sum((delta * L) ** 2))

    rmsd     = float(np.sqrt(np.mean(matched_dists_sq)))
    type_acc = float((pred_types[row_ind] == true_types[col_ind]).mean())

    # ── 子群统计：eval_cutoff 范围内有多少原子（最小镜像距离到 Fe 原点）──
    pred_mi = pred_frac.copy(); pred_mi[pred_mi > 0.5] -= 1.0
    true_mi = true_frac.copy(); true_mi[true_mi > 0.5] -= 1.0
    pred_dists = np.linalg.norm(pred_mi * L, axis=1)
    true_dists = np.linalg.norm(true_mi * L, axis=1)
    n_pred_in  = int((pred_dists <= eval_cutoff).sum())
    n_true_in  = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd':        rmsd,
        'type_acc':    type_acc,
        'n_pred_in':   n_pred_in,
        'n_true_in':   n_true_in,
        'eval_cutoff': eval_cutoff,
    }


def subgroup_stats(results, key, bins):
    import numpy as np
    groups = []
    for lo, hi, label in bins:
        sub = [r for r in results
               if (lo is None or r[key] >= lo) and (hi is None or r[key] < hi)]
        if not sub:
            groups.append((label, 0, float('nan'), float('nan')))
            continue
        groups.append((label, len(sub),
                       float(np.mean([r['rmsd'] for r in sub])),
                       float(np.mean([r['type_acc'] for r in sub]))))
    return groups


def compute_metrics(pred_path, split_name, report_lines):
    import numpy as np, torch
    logger = logging.getLogger(__name__)

    if not os.path.exists(pred_path):
        logger.error(f"❌ 找不到 {pred_path}，请先运行 step5_1_sample.py")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])
    logger.info(f"\n计算 {split_name} 集指标（{n} 个样本，L={L}，最小镜像匈牙利匹配）...")

    results, skipped = [], 0
    for i in range(n):
        pf = preds['pred_frac_coords'][i].numpy()
        pt = preds['pred_atom_types'][i].numpy()
        tf = preds['true_frac_coords'][i].numpy()
        tt = preds['true_atom_types'][i].numpy()
        ec = float(preds['eval_cutoff'][i])

        if pf.shape[0] != 20 or tf.shape[0] != 20:
            skipped += 1
            continue

        r = evaluate_sample(pf, pt, tf, tt, ec, L=L)
        results.append(r)

    logger.info(f"  有效样本：{len(results)}/{n}（跳过 {skipped}）")

    if not results:
        logger.error("❌ 无有效样本，终止评估")
        return

    rmsds     = np.array([r['rmsd']      for r in results])
    type_accs = np.array([r['type_acc']  for r in results])
    n_pred_in = np.array([r['n_pred_in'] for r in results])
    n_true_in = np.array([r['n_true_in'] for r in results])

    rb = L / 2 * (3 / 5) ** 0.5   # ≈ 2.32 Å（L=6，[-3,3] 均匀分布随机基线）

    lines = [
        "",
        f"=== {split_name} Set Metrics (Step5 Holdout, min-image Hungarian matching) ===",
        f"N_samples        : {len(results)}",
        f"RMSD (Å)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"Type Accuracy    : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}",
        f"",
        f"原子密度参考（eval_cutoff 内，最小镜像到 Fe 原点）：",
        f"  pred_in_cutoff : mean={n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff : mean={n_true_in.mean():.2f} / 20",
        f"",
        f"参考（Step4d val/test）：RMSD=1.47Å  TypeAcc=0.249  pred_in_cutoff≈17.47/20",
    ]

    lines += ["", "── 子群：eval_cutoff ──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 Å"), (3.0, 4.0, "3.0–4.0 Å")]):
        lines.append(f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += ["", "── 子群：n_true_in（eval_cutoff 内真实原子数）──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9,   "≤ 8（第一壳层）"),
             (9,   15,   "9–14（第二壳层）"),
             (15,  None, "15–20（第三壳层）")]):
        lines.append(f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    # ── 验收判断（STEP5_HANDOFF.md 验收标准）─────────────────────────────
    lines += ["", "── Holdout 验收判断（STEP5_HANDOFF.md 标准）──"]
    rmsd_ok   = 1.4 <= rmsds.mean() <= 2.0
    typeacc_ok = type_accs.mean() >= 0.20
    cutoff_ok  = n_pred_in.mean() >= 15.0

    lines.append(f"  RMSD         : {rmsds.mean():.4f} Å  "
                 f"{'✅ 正常 [1.4-2.0]' if rmsd_ok else ('⚠️ 偏高' if rmsds.mean() <= 2.5 else '❌ 异常 > 2.5 需汇报')}")
    lines.append(f"  Type Accuracy: {type_accs.mean():.4f}  "
                 f"{'✅ 正常 [0.22-0.28]' if type_accs.mean() >= 0.22 else ('⚠️ 略低' if typeacc_ok else '❌ 异常 < 0.20 需汇报')}")
    lines.append(f"  pred_in_cutoff: {n_pred_in.mean():.2f}/20  "
                 f"{'✅ 正常 [15-20]' if cutoff_ok else '❌ 异常 < 10 需汇报'}")

    lines += [
        "",
        f"随机基线 RMSD ≈ {rb:.2f} Å（L={L}，[-L/2,L/2] 均匀分布期望）",
    ]
    if rmsds.mean() < 2.0:
        lines.append("✅✅ RMSD < 2.0 Å，Holdout 验收通过！模型泛化性良好。")
    elif rmsds.mean() < 2.5:
        lines.append("⚠️  RMSD 在 2.0–2.5 Å 之间，略高于 val/test，需汇报 Main Agent 2。")
    else:
        lines.append("❌ RMSD > 2.5 Å，属于异常，需汇报 Main Agent 2。")

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step5_2  Holdout 集评估（v5，最小镜像匈牙利匹配，L=6）")
    logger.info("=" * 60)

    os.makedirs(STEP5_DIR, exist_ok=True)

    report_lines = [
        "Step5 Holdout Metrics Report",
        f"坐标系：L=6.0，min-image [-0.5,0.5]，匹配/RMSD 使用最小镜像周期边界距离",
        f"checkpoint：experiment2/step4d/checkpoints/epoch=249-val_loss=0.8554.ckpt",
        "=" * 60,
    ]

    pred_path = os.path.join(STEP5_DIR, "predictions_holdout.pt")
    compute_metrics(pred_path, "Holdout", report_lines)

    # ── 生成汇报模板 ─────────────────────────────────────────────────────
    import torch, numpy as np

    if os.path.exists(pred_path):
        preds = torch.load(pred_path, map_location="cpu")
        n_samples = len(preds['mp_id'])
    else:
        n_samples = "N/A"

    # 尝试读取结果数字以填写汇报模板
    try:
        preds = torch.load(pred_path, map_location="cpu")
        from scipy.optimize import linear_sum_assignment
        _results = []
        for i in range(len(preds['mp_id'])):
            pf = preds['pred_frac_coords'][i].numpy()
            pt = preds['pred_atom_types'][i].numpy()
            tf = preds['true_frac_coords'][i].numpy()
            tt = preds['true_atom_types'][i].numpy()
            ec = float(preds['eval_cutoff'][i])
            if pf.shape[0] == 20 and tf.shape[0] == 20:
                _results.append(evaluate_sample(pf, pt, tf, tt, ec, L=L))

        _rmsds     = np.array([r['rmsd']      for r in _results])
        _type_accs = np.array([r['type_acc']  for r in _results])
        _pred_in   = np.array([r['n_pred_in'] for r in _results])
        _true_in   = np.array([r['n_true_in'] for r in _results])

        report_template = f"""
## Step5 完成报告

**执行内容**：Holdout 盲测检验

**Holdout 样本数**：{len(_results)}（原始 IDs：{n_samples}）

**评估结果**：
  RMSD（holdout）：{_rmsds.mean():.4f} Å
  Type Accuracy（holdout）：{_type_accs.mean():.4f}
  pred_in_cutoff（holdout）：{_pred_in.mean():.2f} / 20
  true_in_cutoff（holdout）：{_true_in.mean():.2f} / 20

**与 val/test 对比**：
  RMSD：val=1.47Å，test=1.47Å，holdout={_rmsds.mean():.4f}Å
  Type Acc：val=0.249，holdout={_type_accs.mean():.4f}

**输出文件**：
  predictions_holdout.pt ✅
  metrics_holdout.txt ✅

**异常/发现**：
  {"无异常，Holdout 结果与 val/test 接近，泛化性良好。" if _rmsds.mean() <= 2.0 else "⚠️ RMSD 超出预期范围，需 Main Agent 2 决策。"}

**需要 Main Agent 2 决策的问题**：
  {"无。" if _rmsds.mean() <= 2.5 and _type_accs.mean() >= 0.20 else "RMSD 或 Type Accuracy 超出异常阈值，请 Main Agent 2 查阅 metrics_holdout.txt。"}
"""
        report_lines.append("\n" + "=" * 60)
        report_lines.append(report_template)
        logger.info(report_template)

    except Exception as e:
        logger.warning(f"无法生成汇报模板（{e}），请手动填写。")

    report_path = os.path.join(STEP5_DIR, "metrics_holdout.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 60)
    logger.info("Step5_2 完成")
    logger.info("请将 metrics_holdout.txt 内容发送给 Main Agent 2。")
