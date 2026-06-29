# step4_4_compute_metrics.py  (v3 — full 20-atom Hungarian matching)
# ============================================================
# 关键修正：
#   预测目标就是固定 20 个邻居，true 也是固定 20 个。
#   不做 eval_cutoff 过滤，直接对全部 20 vs 20 做匈牙利匹配。
#   eval_cutoff 仅作为子群分组依据，不参与过滤。
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP4_DIR    = os.path.join(EXP2_ROOT, "step4")
L = 12.0


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=12.0):
    """
    直接对全部 20 个预测原子 vs 20 个真实原子做匈牙利匹配。
    不做 eval_cutoff 截断过滤（eval_cutoff 仅用于子群分析）。

    坐标约定：
      pred_frac：sample() 输出，经 % 1. 在 [0,1]
      true_frac：dataset 存储，在 [-0.5, 0.5]
      → 对 pred_frac 做最小镜像折叠回 [-0.5, 0.5] 再转笛卡尔
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # 最小镜像：pred [0,1] → [-0.5, 0.5]，与 true 对齐
    pred_frac = pred_frac.copy()
    pred_frac[pred_frac > 0.5] -= 1.0

    pred_cart = pred_frac * L   # (20, 3)，以 Fe 为原点，单位 Å
    true_cart = true_frac * L   # (20, 3)

    n = pred_cart.shape[0]   # 固定为 20

    # 构建 20×20 距离矩阵
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(pred_cart[i] - true_cart[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pred  = pred_cart[row_ind]
    matched_true  = true_cart[col_ind]
    matched_pt    = pred_types[row_ind]
    matched_tt    = true_types[col_ind]

    rmsd     = float(np.sqrt(((matched_pred - matched_true) ** 2).sum(axis=1).mean()))
    type_acc = float((matched_pt == matched_tt).mean())

    # 额外指标：eval_cutoff 范围内有多少预测原子
    pred_dists  = np.linalg.norm(pred_cart, axis=1)
    n_pred_in   = int((pred_dists <= eval_cutoff).sum())
    true_dists  = np.linalg.norm(true_cart, axis=1)
    n_true_in   = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd':         rmsd,
        'type_acc':     type_acc,
        'n_pred_in':    n_pred_in,    # eval_cutoff 内预测原子数
        'n_true_in':    n_true_in,    # eval_cutoff 内真实原子数
        'eval_cutoff':  eval_cutoff,
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
        logger.error(f"❌ 找不到 {pred_path}，请先运行 step4_3_sample.py")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])
    logger.info(f"\n计算 {split_name} 集指标（{n} 个样本，全 20 原子匹配）...")

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

    lines = [
        "",
        f"=== {split_name} Set Metrics (full 20-atom matching) ===",
        f"N_samples        : {len(results)}",
        f"RMSD (Å)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"Type Accuracy    : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}",
        f"",
        f"原子密度参考（eval_cutoff 内）：",
        f"  pred_in_cutoff : mean={n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff : mean={n_true_in.mean():.2f} / 20",
        f"  （两者接近说明模型原子密度分布合理）",
    ]

    lines += ["", "── 子群：eval_cutoff ──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 Å"), (3.0, 4.0, "3.0–4.0 Å")]):
        lines.append(f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += ["", "── 子群：n_true_in（eval_cutoff 内真实原子数）──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9, "≤ 8（第一壳层）"),
             (9, 15, "9–14（第二壳层）"),
             (15, None, "15–20（第三壳层）")]):
        lines.append(f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    rb = L / 2 * (3 / 5) ** 0.5
    lines += [
        "",
        f"参考：随机基线 RMSD ≈ {rb:.2f} Å（L=12 均匀分布期望）",
    ]
    if rmsds.mean() < rb * 0.5:
        lines.append("✅✅ 模型大幅优于随机基线（< 50%）")
    elif rmsds.mean() < rb * 0.7:
        lines.append("✅ 模型显著优于随机基线（< 70%）")
    else:
        lines.append("⚠️  模型仍未达到有效阈值")

    for line in lines: logger.info(line)
    report_lines.extend(lines)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step 4.4  评估指标计算（v3，全 20 原子匹配）")
    logger.info("=" * 60)

    report_lines = []

    # 读取 v1 原始采样结果（不需要重新采样）
    compute_metrics(
        os.path.join(STEP4_DIR, "predictions_val.pt"),  "Val",  report_lines)
    compute_metrics(
        os.path.join(STEP4_DIR, "predictions_test.pt"), "Test", report_lines)

    report_path = os.path.join(STEP4_DIR, "metrics_report_v3.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 60)
    logger.info("Step 4.4 完成")