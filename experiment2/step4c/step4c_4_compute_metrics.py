# step4c_4_compute_metrics.py
# Step4c formal metrics (identical logic to step4b_4_compute_metrics.py v5)
# ============================================================
# Only difference from step4b version:
#   - reads from step4c/predictions_val.pt and predictions_test.pt
#   - writes report to step4c/metrics_report.txt
#   - header updated to reflect Step4c coordinate system
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP4c_DIR   = os.path.join(EXP2_ROOT, "step4c")
L = 12.0


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=12.0):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)
    n = pred_frac.shape[0]

    cost_matrix = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)
        cost_matrix[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_dists_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_dists_sq.append(np.sum((delta * L) ** 2))

    rmsd     = float(np.sqrt(np.mean(matched_dists_sq)))
    type_acc = float((pred_types[row_ind] == true_types[col_ind]).mean())

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
        logger.error(f"Not found: {pred_path}. Run sampling first.")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])
    logger.info(f"\nComputing {split_name} metrics ({n} samples)...")

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

    logger.info(f"  Valid: {len(results)}/{n} (skipped {skipped})")

    rmsds     = np.array([r['rmsd']     for r in results])
    type_accs = np.array([r['type_acc'] for r in results])
    n_pred_in = np.array([r['n_pred_in'] for r in results])
    n_true_in = np.array([r['n_true_in'] for r in results])

    rb = L / 2 * (3 / 5) ** 0.5   # ~4.65 A

    lines = [
        "",
        f"=== {split_name} Set Metrics (Step4c, [-0.5,0.5] coordinate system) ===",
        f"N_samples        : {len(results)}",
        f"RMSD (A)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"Type Accuracy    : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}",
        f"",
        f"pred_in_cutoff   : mean={n_pred_in.mean():.2f} / 20",
        f"true_in_cutoff   : mean={n_true_in.mean():.2f} / 20",
    ]

    lines += ["", "-- Subgroup: eval_cutoff --"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 A"), (3.0, 4.0, "3.0-4.0 A")]):
        lines.append(f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += ["", "-- Subgroup: n_true_in --"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9, "<=8"), (9, 15, "9-14"), (15, None, "15-20")]):
        lines.append(f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += [
        "",
        f"Random baseline RMSD ~ {rb:.2f} A",
        f"Target: RMSD < 2.0 A, pred_in_cutoff ~ true_in_cutoff",
    ]
    if rmsds.mean() < 2.0:
        lines.append("PASS: RMSD < 2.0 A")
    elif rmsds.mean() < rb * 0.7:
        lines.append("Model significantly better than random (< 70% baseline)")
    elif rmsds.mean() < rb * 0.9:
        lines.append("Model slightly better than random (< 90% baseline)")
    else:
        lines.append("FAIL: Model not meaningfully better than random baseline")

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step4c metrics (min-image Hungarian matching)")
    logger.info("=" * 60)

    report_lines = [
        "Step4c Metrics Report",
        "Coordinate system: [-0.5, 0.5] (dataset v5 + diffusion v3)",
        "Matching: min-image Hungarian",
        "=" * 60,
    ]

    compute_metrics(
        os.path.join(STEP4c_DIR, "predictions_val.pt"),  "Val",  report_lines)
    compute_metrics(
        os.path.join(STEP4c_DIR, "predictions_test.pt"), "Test", report_lines)

    report_path = os.path.join(STEP4c_DIR, "metrics_report.txt")
    os.makedirs(STEP4c_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\nReport -> {report_path}")
    logger.info("=" * 60)
