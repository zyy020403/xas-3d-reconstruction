#!/usr/bin/env python
"""
step5_2_compute_metrics.py
========================================================================
DiffCSP-Exp4 Step 5.2 — metrics for one split (val | test)

Adapted from Exp2 step5_2_compute_metrics.py with these changes:
  - argparse --split  (one at a time, cleaner output)
  - 4-tier eval_cutoff stratification (Exp4 differentiator vs Exp2)
  - per_sample_metrics_{split}.csv output (Step6Agent input)
  - §6 thresholds (handoff): RMSD 1.2-2.0 / TypeAcc 0.20-0.35 / pred_in 14-19
  - effective/nominal dual reporting (Phase 4.6 silent-drop caveat)
  - Reference baseline: Exp2 holdout (1.47 / 0.241 / 17.52)
    with 88-element caveat (Exp4 strictly not directly comparable)

evaluate_sample() algorithm verbatim from Exp2 (proven correct).

Usage:
  cd /home/tcat/diffcsp_exp4/code/step5
  PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
    python step5_2_compute_metrics.py --split val 2>&1 | \
    tee /home/tcat/diffcsp_exp4/logs/step5_metrics_val.log

  python step5_2_compute_metrics.py --split test 2>&1 | \
    tee /home/tcat/diffcsp_exp4/logs/step5_metrics_test.log
"""

import argparse, os, sys, logging, warnings, csv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DIFFCSP_ROOT = "/home/tcat/diffcsp_exp4"
OUT_DIR      = f"{DIFFCSP_ROOT}/code/step5"
L            = 6.0
N_NEIGHBORS  = 20


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=6.0):
    """
    20×20 Hungarian matching with min-image distance.
    (Exp2 algorithm verbatim, proven correct by Exp2 final report.)

    Coordinates: [-0.5, 0.5] (min-image folded by dataset_v2 line 245-ish).
    Distance:    delta = p - t;  delta -= round(delta);  ||delta * L||
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)
    n = pred_frac.shape[0]  # always 20

    # 20×20 min-image cost matrix
    cost = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)
        cost[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost)

    # RMSD via matched min-image distance
    matched_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_sq.append(np.sum((delta * L) ** 2))
    rmsd = float(np.sqrt(np.mean(matched_sq)))
    type_acc = float((pred_types[row_ind] == true_types[col_ind]).mean())

    # in-cutoff stats: distance from atom to Fe origin (min-image)
    pred_mi = pred_frac - np.round(pred_frac)
    true_mi = true_frac - np.round(true_frac)
    pred_dists = np.linalg.norm(pred_mi * L, axis=1)
    true_dists = np.linalg.norm(true_mi * L, axis=1)
    n_pred_in = int((pred_dists <= eval_cutoff).sum())
    n_true_in = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd': rmsd,
        'type_acc': type_acc,
        'n_pred_in': n_pred_in,
        'n_true_in': n_true_in,
        'eval_cutoff': eval_cutoff,
    }


def subgroup(results, key, bins):
    """Return list of (label, n, mean_rmsd, mean_typeacc, mean_pred_in)."""
    import numpy as np
    out = []
    for lo, hi, label in bins:
        sub = [r for r in results
               if (lo is None or r[key] >= lo) and (hi is None or r[key] < hi)]
        if not sub:
            out.append((label, 0, float('nan'), float('nan'), float('nan')))
            continue
        out.append((label, len(sub),
                    float(np.mean([r['rmsd']      for r in sub])),
                    float(np.mean([r['type_acc']  for r in sub])),
                    float(np.mean([r['n_pred_in'] for r in sub]))))
    return out


def verdict_per_metric(rmsd, typeacc, pred_in):
    """STEP5_HANDOFF §6 thresholds.

    green = within MA5-defined acceptable band
    amber = somewhat outside but not red-line
    red   = handoff red-light (must report to MA5)
    """
    rmsd_ok    = 1.2 <= rmsd <= 2.0
    type_ok    = 0.20 <= typeacc <= 0.35
    cutoff_ok  = 14 <= pred_in <= 19

    rmsd_red    = (rmsd > 3.0) or (rmsd < 0.5)         # handoff §6 hard red lines
    type_red    = typeacc > 0.6                         # handoff §6 hard red line
    cutoff_red  = pred_in < 5                           # handoff §6 hard red line

    rmsd_flag   = ("✅ green [1.2-2.0]"   if rmsd_ok   else
                   "❌ RED — report MA5"  if rmsd_red   else
                   "⚠️  amber")
    type_flag   = ("✅ green [0.20-0.35]" if type_ok   else
                   "❌ RED — report MA5"  if type_red   else
                   "⚠️  amber")
    cutoff_flag = ("✅ green [14-19]"     if cutoff_ok else
                   "❌ RED — report MA5"  if cutoff_red else
                   "⚠️  amber")
    return rmsd_flag, type_flag, cutoff_flag


def compute_metrics(split, args):
    import numpy as np, torch

    pred_path = os.path.join(args.out_dir, f"predictions_{split}.pt")
    if not os.path.exists(pred_path):
        logger.error(f"❌ {pred_path} missing. Run step5_1_sample.py first.")
        return

    preds = torch.load(pred_path, map_location="cpu", weights_only=False)
    n_total = len(preds['sample_name'])
    n_nominal = preds.get('n_nominal', n_total)
    logger.info(f"\n[{split}]  Computing metrics  "
                f"(loaded={n_total}, nominal={n_nominal}, L={L})")

    results, skipped = [], 0
    for i in range(n_total):
        pf = preds['pred_frac_coords'][i].numpy()
        pt = preds['pred_atom_types'][i].numpy()
        tf = preds['true_frac_coords'][i].numpy()
        tt = preds['true_atom_types'][i].numpy()
        ec = float(preds['eval_cutoff'][i])
        if pf.shape[0] != N_NEIGHBORS or tf.shape[0] != N_NEIGHBORS:
            skipped += 1
            continue
        r = evaluate_sample(pf, pt, tf, tt, ec, L=L)
        r['sample_name'] = preds['sample_name'][i]
        r['mp_id']       = preds['mp_id'][i]
        results.append(r)

    n_eff = len(results)
    logger.info(f"  effective evaluated: {n_eff} / {n_nominal} "
                f"(skipped shape-mismatch: {skipped})")

    if not results:
        logger.error("❌ no valid results, abort")
        return

    rmsds     = np.array([r['rmsd']      for r in results])
    type_accs = np.array([r['type_acc']  for r in results])
    n_pred_in = np.array([r['n_pred_in'] for r in results])
    n_true_in = np.array([r['n_true_in'] for r in results])
    eval_cuts = np.array([r['eval_cutoff'] for r in results])

    rb = L / 2 * (3/5)**0.5  # ≈ 2.32 Å, uniform-prior random baseline

    # ── Build report ──────────────────────────────────────────────────
    lines = [
        "=" * 72,
        f"  Step 5.2  {split.upper()} Set Metrics  (DiffCSP-Exp4)",
        "=" * 72,
        f"  Coordinate system : L={L}, [-0.5, 0.5], min-image Hungarian",
        f"  Checkpoint        : {preds.get('checkpoint', 'N/A')}",
        f"  Samples           : effective={n_eff}  /  nominal={n_nominal}  "
            f"(silent-drop dataset={n_nominal - n_total}, +shape-mismatch={skipped})",
        f"  Random baseline   : RMSD ≈ {rb:.2f} Å  (uniform [-L/2, L/2])",
        "",
        "── Aggregate metrics ──",
        f"  RMSD (Å)          : mean={rmsds.mean():.4f}  median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"  Type Accuracy     : mean={type_accs.mean():.4f}  median={np.median(type_accs):.4f}  std={type_accs.std():.4f}",
        f"  pred_in_cutoff    : mean={n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff    : mean={n_true_in.mean():.2f} / 20",
        f"  eval_cutoff (Å)   : mean={eval_cuts.mean():.3f}  median={np.median(eval_cuts):.3f}",
        "",
        "── Reference: Exp2 holdout (Fe-only, ⚠️ NOT directly comparable) ──",
        "  Exp2 RMSD=1.47 Å  TypeAcc=0.241  pred_in_cutoff=17.52/20  true_in_cutoff=18.99/20",
        "  Caveat: Exp2 = Fe K-edge only.  Exp4 = 88 elements, harder type prediction.",
        "          Per MA4 D1, Exp4 = fp32 (Exp2 = bf16); ±5% numerical caveat applies.",
        "",
        "── Stratified by eval_cutoff (4-tier, Exp4 differentiator) ──",
        f"  {'Tier':28s}  {'N':>5s}  {'RMSD':>7s}  {'TypeAcc':>8s}  {'pred_in':>8s}",
    ]

    eval_bins = [
        (None, 3.0,  "A: ≤ 3.0 Å (dense)"),
        (3.0,  4.0,  "B: 3.0 – 4.0 Å"),
        (4.0,  5.0,  "C: 4.0 – 5.0 Å"),
        (5.0,  None, "D: > 5.0 Å (sparse)"),
    ]
    for label, n, mr, mt, mp in subgroup(results, 'eval_cutoff', eval_bins):
        lines.append(f"  {label:28s}  {n:>5d}  {mr:>7.4f}  {mt:>8.4f}  {mp:>8.2f}")

    lines += [
        "",
        "── Stratified by n_true_in (shell density, comparable to Exp2) ──",
        f"  {'Bin':28s}  {'N':>5s}  {'RMSD':>7s}  {'TypeAcc':>8s}  {'pred_in':>8s}",
    ]
    true_bins = [
        (None, 9,    "≤ 8 (1st-shell only)"),
        (9,    15,   "9 – 14 (mid-shell)"),
        (15,   None, "15 – 20 (full shell)"),
    ]
    for label, n, mr, mt, mp in subgroup(results, 'n_true_in', true_bins):
        lines.append(f"  {label:28s}  {n:>5d}  {mr:>7.4f}  {mt:>8.4f}  {mp:>8.2f}")

    # ── §6 verdict ────────────────────────────────────────────────────
    rmsd_flag, type_flag, cutoff_flag = verdict_per_metric(
        rmsds.mean(), type_accs.mean(), n_pred_in.mean())
    lines += [
        "",
        "── Verdict (STEP5_HANDOFF §6 thresholds) ──",
        f"  RMSD            {rmsds.mean():.4f}     {rmsd_flag}",
        f"  Type Accuracy   {type_accs.mean():.4f}     {type_flag}",
        f"  pred_in_cutoff  {n_pred_in.mean():.2f}/20    {cutoff_flag}",
        "",
        "  Caveat: §6 verdict is preliminary signal-only.  MA5 makes go/no-go call.",
        "          Step5Agent does not decide fine-tune / re-train / phase 5b.",
        "=" * 72,
    ]

    for ln in lines:
        logger.info(ln)

    # ── Write text report ─────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, f"metrics_report_{split}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"\n  written: {report_path}")

    # ── Write per-sample CSV (Step6Agent input) ───────────────────────
    csv_path = os.path.join(args.out_dir, f"per_sample_metrics_{split}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_name", "mp_id", "rmsd", "type_acc",
                    "n_pred_in", "n_true_in", "eval_cutoff"])
        for r in results:
            w.writerow([r['sample_name'], r['mp_id'],
                        f"{r['rmsd']:.6f}", f"{r['type_acc']:.6f}",
                        r['n_pred_in'], r['n_true_in'],
                        f"{r['eval_cutoff']:.6f}"])
    logger.info(f"  written: {csv_path}  ({n_eff} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "holdout"], required=True)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.split == "holdout":
        raise RuntimeError(
            "❌ holdout metrics requires MA5 phase 5b approval. "
            "Step5Agent first leg = val + test only. STOP."
        )

    logger.info("=" * 60)
    logger.info(f"Step 5.2  metrics  split={args.split}  L={L}")
    logger.info("=" * 60)
    compute_metrics(args.split, args)


if __name__ == "__main__":
    main()
