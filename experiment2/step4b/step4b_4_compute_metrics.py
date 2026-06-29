# step4b_4_compute_metrics.py  (v5 — Step4b，最小镜像用于匹配/RMSD，不做全局坐标折叠)
# ============================================================
# v4 → v5 修正说明：
#
#   v4 删除了所有最小镜像操作，导致 RMSD=11.6 Å（远超随机基线）。
#   原因：v6 坐标系下 frac ∈ [0,1]，邻居原子因 % 1.0 折叠，
#         分布在 frac≈0 AND frac≈1 两端（物理上都靠近 Fe 原点）。
#         若直接用 frac*L 的笛卡尔距离匹配，frac=0.02 与 frac=0.98
#         被算成 11.5 Å，但实际距离只有 0.24 Å → 匈牙利匹配完全错乱。
#
#   正确做法：最小镜像用于匈牙利匹配的 cost matrix 和 RMSD 计算，
#             但 NOT 用于全局坐标折叠（那个 -0.5 预处理是 v3 的旧 bug）。
#
#   最小镜像距离（分数坐标，周期边界）：
#     delta = pred_frac_i - true_frac_j
#     delta -= np.round(delta)          ← 最小镜像
#     dist = ||delta * L||
#
#   这与 v3 的 "先折叠到 [-0.5,0.5] 再算欧氏距离" 等价，
#   但不依赖坐标系是 [-0.5,0.5] 还是 [0,1]，对 v6 通用。
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP4b_DIR   = os.path.join(EXP2_ROOT, "step4b")
L = 12.0


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=12.0):
    """
    全部 20 个预测原子 vs 20 个真实原子做匈牙利匹配。

    坐标约定（v6 / Step4b）：
      pred_frac, true_frac 均在 [0, 1]
      Fe 原点对应 frac=0（邻居原子因 %1.0 折叠，分布在 frac≈0 和 frac≈1 两端）

    匹配和 RMSD：使用最小镜像距离（周期边界条件）
      delta = p - t;  delta -= round(delta);  dist = ||delta * L||
    这样 frac=0.02 与 frac=0.98 的距离 = ||(-0.04)*12|| = 0.48 Å（正确）
    而非直接欧氏距离 = ||(0.02-0.98)*12|| = 11.52 Å（错误）
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)

    n = pred_frac.shape[0]   # 固定为 20

    # ── 构建 20×20 最小镜像距离矩阵 ──────────────────────────────────────
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac          # (20, 3)
        delta -= np.round(delta)                   # 最小镜像
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

    # ── 子群统计：eval_cutoff 范围内有多少原子（均用最小镜像距离到 Fe 原点）──
    # Fe 原点在 frac=0，最小镜像距离 = min(frac, 1-frac) * L（逐轴）
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
        logger.error(f"❌ 找不到 {pred_path}，请先运行采样脚本")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])
    logger.info(f"\n计算 {split_name} 集指标（{n} 个样本，最小镜像匈牙利匹配）...")

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

    rmsds     = np.array([r['rmsd'] for r in results])
    type_accs = np.array([r['type_acc'] for r in results])
    n_pred_in = np.array([r['n_pred_in'] for r in results])
    n_true_in = np.array([r['n_true_in'] for r in results])

    rb = L / 2 * (3 / 5) ** 0.5   # ≈ 4.65 Å，[-L/2,L/2] 均匀分布随机基线

    lines = [
        "",
        f"=== {split_name} Set Metrics (Step4b v5, min-image Hungarian matching) ===",
        f"N_samples        : {len(results)}",
        f"RMSD (Å)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"Type Accuracy    : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}",
        f"",
        f"原子密度参考（eval_cutoff 内，最小镜像到 Fe 原点）：",
        f"  pred_in_cutoff : mean={n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff : mean={n_true_in.mean():.2f} / 20",
        f"  （验收要求：pred ≈ true ≈ 17/20）",
    ]

    lines += ["", "── 子群：eval_cutoff ──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 Å"), (3.0, 4.0, "3.0–4.0 Å")]):
        lines.append(f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += ["", "── 子群：n_true_in（eval_cutoff 内真实原子数）──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9,  "≤ 8（第一壳层）"),
             (9,   15,  "9–14（第二壳层）"),
             (15,  None,"15–20（第三壳层）")]):
        lines.append(f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += [
        "",
        f"随机基线 RMSD ≈ {rb:.2f} Å（[-L/2,L/2] 均匀分布期望）",
        f"Step4b 验收标准：RMSD < 2.0 Å，pred_in_cutoff ≈ true_in_cutoff ≈ 17/20",
    ]
    if rmsds.mean() < 2.0:
        lines.append("✅✅ RMSD < 2.0 Å，达到 Step4b 验收标准！")
    elif rmsds.mean() < rb * 0.7:
        lines.append("✅ 模型显著优于随机基线（< 70%），未达 2.0 Å 目标")
    elif rmsds.mean() < rb * 0.9:
        lines.append("⚠️  模型略优于随机基线（< 90%），需进一步分析")
    else:
        lines.append("❌ 模型未有效优于随机基线，需汇报 Main Agent 2")

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step4b_4  评估指标（v5，最小镜像匈牙利匹配）")
    logger.info("=" * 60)

    report_lines = [
        "Step4b Metrics Report (v5)",
        "坐标系：Dataset v6 [0,1]，匹配/RMSD 使用最小镜像周期边界距离",
        "=" * 60,
    ]

    compute_metrics(
        os.path.join(STEP4b_DIR, "predictions_val.pt"),  "Val",  report_lines)
    compute_metrics(
        os.path.join(STEP4b_DIR, "predictions_test.pt"), "Test", report_lines)

    report_path = os.path.join(STEP4b_DIR, "metrics_report_v5.txt")
    os.makedirs(STEP4b_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 60)
    logger.info("Step4b_4 完成")