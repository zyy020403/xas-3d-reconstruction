#!/usr/bin/env python
"""
step5_2_compute_metrics.py  —  Exp5 v2 SA1' metrics for one split (val | test)
============================================================================
改造自 Exp4 step5_2_compute_metrics.py, 加 4 个新函数:
  - compute_set_level_typeacc        (per-sample multiset intersection / N)
  - compute_multiset_f1_macro        (dataset-level Macro-F1 across element classes)
  - compute_collapse_ratio           (per-sample pred std vs true std collapse detection)
  - compute_projection_ablation_rmsd (helper for SA3, projection beyond R_max → recompute RMSD)

输出:
  - metrics_report_<split>.txt      Exp5 v2 主面板 (Set-Level/Multiset/Collapse) +
                                     历史对照 (position-by-position TypeAcc) +
                                     4-tier eval_cutoff stratification (Exp4 carry-over)
  - per_sample_metrics_<split>.csv  per-sample for Step6Agent

§6 thresholds (handoff carry-over):
  RMSD 1.2-2.0 / TypeAcc 0.20-0.35 / pred_in 14-19 — 这是 Exp4 标尺,
  Exp5 v2 主信号是 Set-Level + Multiset Macro-F1, 红绿灯由 MA5/SA3 制定。

Reference baseline: Exp2 holdout (Fe-only, ⚠️ NOT directly comparable to Exp5 88-element).

evaluate_sample() algorithm verbatim from Exp2 (proven correct).

Usage:
  cd /home/tcat/diffcsp_exp5/code/step5
  PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
    python step5_2_compute_metrics.py --split val 2>&1 | \
    tee /home/tcat/diffcsp_exp5/logs/step5_metrics_val.log

Dry-run Exp4 baseline (SA1' 跑, 只用 val + test, NOT holdout):
  python step5_2_compute_metrics.py --split val \
      --predictions /home/tcat/diffcsp_exp4/code/step5/predictions_val.pt \
      --output /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_metrics.txt
  python step5_2_compute_metrics.py --split test \
      --predictions /home/tcat/diffcsp_exp4/code/step5/predictions_test.pt \
      --output /home/tcat/diffcsp_exp5/logs/exp4_baseline_test_metrics.txt
"""

import argparse, os, sys, logging, warnings, csv
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DIFFCSP_ROOT = "/home/tcat/diffcsp_exp5"
OUT_DIR      = f"{DIFFCSP_ROOT}/code/step5"
L            = 6.0
N_NEIGHBORS  = 20

# Element symbols for Multiset F1 reporting (Z → symbol)
_PERIODIC_SYMBOLS = [
    "_", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm",
]

def _z_symbol(z: int) -> str:
    if 1 <= z < len(_PERIODIC_SYMBOLS):
        return _PERIODIC_SYMBOLS[z]
    return f"Z{z}"


# ──────────────────────────────────────────────────────────────────────────
# Exp4 carry-over — Hungarian min-image evaluate_sample (verbatim)
# ──────────────────────────────────────────────────────────────────────────

def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=6.0):
    """
    20×20 Hungarian matching with min-image distance.
    Coordinates: [-0.5, 0.5] (min-image folded by dataset_v2).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)
    n = pred_frac.shape[0]

    cost = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)
        cost[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_sq.append(np.sum((delta * L) ** 2))
    rmsd = float(np.sqrt(np.mean(matched_sq)))
    type_acc = float((pred_types[row_ind] == true_types[col_ind]).mean())

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
    """STEP5_HANDOFF §6 thresholds (Exp4 carry-over)."""
    rmsd_ok    = 1.2 <= rmsd <= 2.0
    type_ok    = 0.20 <= typeacc <= 0.35
    cutoff_ok  = 14 <= pred_in <= 19

    rmsd_red    = (rmsd > 3.0) or (rmsd < 0.5)
    type_red    = typeacc > 0.6
    cutoff_red  = pred_in < 5

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


# ──────────────────────────────────────────────────────────────────────────
# Exp5 v2 SA1' new metrics — handoff §6 algorithm definitions
# ──────────────────────────────────────────────────────────────────────────

def compute_set_level_typeacc(pred_types, true_types) -> float:
    """
    Per-sample Set-Level TypeAcc = sum_c min(pred_count_c, true_count_c) / N

    Mathematical equivalent: multiset intersection size / N.

    Parameters
    ----------
    pred_types : (N,) int array  — predicted Z values
    true_types : (N,) int array  — ground-truth Z values

    Returns
    -------
    float in [0, 1], higher is better
    """
    pred_cnt = Counter(pred_types.tolist())
    true_cnt = Counter(true_types.tolist())
    intersection = sum(
        min(pred_cnt[c], true_cnt[c])
        for c in (set(pred_cnt) | set(true_cnt))
    )
    return intersection / max(len(true_types), 1)


def compute_set_level_typeacc_dataset(all_pred_types, all_true_types) -> dict:
    """Dataset-level: per-sample Set-Level TypeAcc 的均值 + std."""
    import numpy as np
    vals = [
        compute_set_level_typeacc(p, t)
        for p, t in zip(all_pred_types, all_true_types)
    ]
    return {
        'set_level_typeacc_mean': float(np.mean(vals)) if vals else 0.0,
        'set_level_typeacc_std':  float(np.std(vals)) if vals else 0.0,
        'n_samples':              len(vals),
        'per_sample_values':      vals,
    }


def compute_multiset_f1_macro(all_pred_types, all_true_types,
                               eps: float = 1e-9) -> dict:
    """
    Dataset-level Multiset Macro-F1 across element classes.

    For each class c that appears in any true sample:
        TP_c = sum_samples min(pred_count_c, true_count_c)
        FP_c = sum_samples max(0, pred_count_c - true_count_c)
        FN_c = sum_samples max(0, true_count_c - pred_count_c)
        precision_c = TP_c / (TP_c + FP_c + eps)
        recall_c    = TP_c / (TP_c + FN_c + eps)
        F1_c        = 2 * P * R / (P + R + eps)

    Macro-F1 = mean over classes appearing in any true sample.

    This exposes majority-class bias (e.g., pred always = O):
      For O:  precision~0.6, recall=1.0, F1=0.75
      For 30 other classes: F1=0
      Macro-F1 ≈ 0.75/30 = 0.025  → 极低,  暴露塌缩

    Returns
    -------
    dict with keys: multiset_macro_f1, per_class_f1, per_class_precision,
                    per_class_recall, per_class_support_true,
                    per_class_support_pred, n_classes_evaluated
    """
    import numpy as np

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support_true = defaultdict(int)
    support_pred = defaultdict(int)
    classes_in_true = set()

    for p, t in zip(all_pred_types, all_true_types):
        p_cnt = Counter(p.tolist())
        t_cnt = Counter(t.tolist())
        all_classes = set(p_cnt) | set(t_cnt)
        for c in all_classes:
            pc = p_cnt[c]
            tc = t_cnt[c]
            tp[c] += min(pc, tc)
            fp[c] += max(0, pc - tc)
            fn[c] += max(0, tc - pc)
            support_true[c] += tc
            support_pred[c] += pc
            if tc > 0:
                classes_in_true.add(c)

    per_class_precision = {}
    per_class_recall    = {}
    per_class_f1        = {}
    for c in classes_in_true:
        prec = tp[c] / (tp[c] + fp[c] + eps)
        rec  = tp[c] / (tp[c] + fn[c] + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        per_class_precision[c] = prec
        per_class_recall[c]    = rec
        per_class_f1[c]        = f1

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

    return {
        'multiset_macro_f1':      macro_f1,
        'per_class_f1':           per_class_f1,
        'per_class_precision':    per_class_precision,
        'per_class_recall':       per_class_recall,
        'per_class_support_true': dict(support_true),
        'per_class_support_pred': dict(support_pred),
        'n_classes_evaluated':    len(classes_in_true),
    }


def compute_collapse_ratio(all_pred_frac, all_true_frac,
                            L: float = 6.0, threshold: float = 0.5) -> dict:
    """
    Per-sample collapse detection (proposal v2 §5.5).

    is_collapsed if pred std (avg over 3 axes, in cartesian Å)
                 < threshold * true std

    Parameters
    ----------
    all_pred_frac : list of (20, 3) frac arrays (∈ [-0.5, 0.5])
    all_true_frac : list of (20, 3) frac arrays
    L         : box edge in Å (= 6.0)
    threshold : default 0.5

    Returns
    -------
    dict: collapse_ratio / n_collapsed / n_total / pred_std_dist / true_std_dist
    """
    import numpy as np
    n_total = len(all_pred_frac)
    n_collapsed = 0
    pred_std_dist = []
    true_std_dist = []

    for p, t in zip(all_pred_frac, all_true_frac):
        pred_xyz_std = float(np.std(np.asarray(p) * L, axis=0).mean())
        true_xyz_std = float(np.std(np.asarray(t) * L, axis=0).mean())
        pred_std_dist.append(pred_xyz_std)
        true_std_dist.append(true_xyz_std)
        if pred_xyz_std < threshold * true_xyz_std:
            n_collapsed += 1

    return {
        'collapse_ratio': n_collapsed / max(n_total, 1),
        'n_collapsed':    n_collapsed,
        'n_total':        n_total,
        'pred_std_dist':  pred_std_dist,
        'true_std_dist':  true_std_dist,
    }


def compute_projection_ablation_rmsd(all_pred_frac, all_true_frac,
                                      R_max: float, L: float = 6.0) -> dict:
    """
    Diagnostic ablation (proposal v2 §5.4) for SA3.

    Project prediction atoms whose cartesian distance from origin > R_max
    onto the R_max sphere shell (preserving direction). Recompute Hungarian
    min-image RMSD vs true.

    Parameters
    ----------
    all_pred_frac : list of (20, 3) frac arrays
    all_true_frac : list of (20, 3) frac arrays
    R_max         : float, projection sphere radius in Å
                    (SA3: read from shell_boundaries.pkl 99th percentile,
                     ≈ 5.5 Å for L=6 box)
    L             : float = 6.0

    Returns
    -------
    dict: rmsd_before / rmsd_after / rmsd_delta / n_atoms_projected_avg
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def _hungarian_rmsd(p_frac, t_frac):
        n = p_frac.shape[0]
        cost = np.zeros((n, n))
        for i in range(n):
            delta = p_frac[i] - t_frac
            delta -= np.round(delta)
            cost[i] = np.linalg.norm(delta * L, axis=1)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_sq = []
        for ri, ci in zip(row_ind, col_ind):
            delta = p_frac[ri] - t_frac[ci]
            delta -= np.round(delta)
            matched_sq.append(np.sum((delta * L) ** 2))
        return float(np.sqrt(np.mean(matched_sq)))

    rmsd_before_list = []
    rmsd_after_list  = []
    n_proj_list      = []

    for p, t in zip(all_pred_frac, all_true_frac):
        p = np.asarray(p, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)

        rmsd_before = _hungarian_rmsd(p, t)
        rmsd_before_list.append(rmsd_before)

        # Project xyz beyond R_max onto sphere
        p_xyz = p * L
        norms = np.linalg.norm(p_xyz, axis=-1)
        mask = norms > R_max
        n_proj = int(mask.sum())
        n_proj_list.append(n_proj)
        if n_proj > 0:
            p_xyz_proj = p_xyz.copy()
            p_xyz_proj[mask] = p_xyz[mask] * (R_max / norms[mask][:, None])
            p_proj_frac = p_xyz_proj / L
        else:
            p_proj_frac = p

        rmsd_after = _hungarian_rmsd(p_proj_frac, t)
        rmsd_after_list.append(rmsd_after)

    rmsd_before_mean = float(np.mean(rmsd_before_list)) if rmsd_before_list else 0.0
    rmsd_after_mean  = float(np.mean(rmsd_after_list))  if rmsd_after_list  else 0.0
    return {
        'rmsd_before':            rmsd_before_mean,
        'rmsd_after':             rmsd_after_mean,
        'rmsd_delta':             rmsd_before_mean - rmsd_after_mean,
        'n_atoms_projected_avg':  float(np.mean(n_proj_list)) if n_proj_list else 0.0,
        'R_max':                  R_max,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main metrics computation
# ──────────────────────────────────────────────────────────────────────────

def compute_metrics(split, args):
    import numpy as np, torch

    pred_path = args.predictions or os.path.join(args.out_dir, f"predictions_{split}.pt")
    if not os.path.exists(pred_path):
        logger.error(f"❌ {pred_path} missing. Run step5_1_sample.py first.")
        return

    preds = torch.load(pred_path, map_location="cpu", weights_only=False)
    n_total = len(preds['sample_name'])
    n_nominal = preds.get('n_nominal', n_total)
    logger.info(f"\n[{split}]  Computing metrics  "
                f"(loaded={n_total}, nominal={n_nominal}, L={L})")
    logger.info(f"  predictions: {pred_path}")

    # Per-sample collection
    results = []
    all_pred_types_arr = []
    all_true_types_arr = []
    all_pred_frac_arr  = []
    all_true_frac_arr  = []
    skipped = 0

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

        all_pred_types_arr.append(pt)
        all_true_types_arr.append(tt)
        all_pred_frac_arr.append(pf)
        all_true_frac_arr.append(tf)

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

    rb = L / 2 * (3/5)**0.5  # ≈ 1.16 Å for L=6, uniform-prior random baseline

    # ── Exp5 v2 main panel computations ────────────────────────────────
    set_level = compute_set_level_typeacc_dataset(all_pred_types_arr, all_true_types_arr)
    multiset  = compute_multiset_f1_macro(all_pred_types_arr, all_true_types_arr)
    collapse  = compute_collapse_ratio(all_pred_frac_arr, all_true_frac_arr, L=L)

    # ── Build report ──────────────────────────────────────────────────
    lines = [
        "=" * 72,
        f"  EXP5 V2 METRICS REPORT — split: {split}",
        "=" * 72,
        f"  Coordinate system : L={L}, [-0.5, 0.5], min-image Hungarian",
        f"  Checkpoint        : {preds.get('checkpoint', 'N/A')}",
        f"  Total samples     : {n_nominal} (nominal)",
        f"  Effective samples : {n_eff} "
            f"(silent_drop count: {n_nominal - n_total}, +shape-mismatch: {skipped})",
        f"  Random baseline   : RMSD ≈ {rb:.2f} Å  (uniform [-L/2, L/2])",
        "",
        "── Geometry (主面板, 与 Exp4 对比) ──",
        f"  RMSD (Å)          : {rmsds.mean():.4f} ± {rmsds.std():.4f}  "
            f"median={np.median(rmsds):.4f}",
        f"  pred_in_cutoff    : {n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff    : {n_true_in.mean():.2f} / 20  (reference)",
        f"  eval_cutoff (Å)   : {eval_cuts.mean():.3f}  (median {np.median(eval_cuts):.3f})",
        "",
        "── Type metrics (Exp5 v2 主面板, 真信号) ──",
        f"  Set-Level TypeAcc : {set_level['set_level_typeacc_mean']:.4f} "
            f"± {set_level['set_level_typeacc_std']:.4f}   "
            f"(per-sample multiset intersection / 20)",
        f"  Multiset Macro-F1 : {multiset['multiset_macro_f1']:.4f}            "
            f"(dataset-level, across {multiset['n_classes_evaluated']} element classes)",
        f"  Collapse Ratio    : {collapse['collapse_ratio']*100:.1f}%             "
            f"({collapse['n_collapsed']}/{collapse['n_total']} at threshold=0.5)",
        "",
        "── Type metrics (历史对照, Exp3 已证为虚假指标, 仅供回溯) ──",
        f"  Position-by-position TypeAcc: {type_accs.mean():.4f}    "
            f"[VIRTUAL METRIC — DO NOT USE for Exp5 v2 decisions]",
        "",
    ]

    # Top-K per-class F1 detail (sorted by support_true)
    pcs_true = multiset['per_class_support_true']
    pcs_pred = multiset['per_class_support_pred']
    pcf1     = multiset['per_class_f1']
    sorted_classes = sorted(pcs_true.items(), key=lambda kv: -kv[1])
    topk = min(10, len(sorted_classes))
    if topk > 0:
        lines += [
            f"── Top-{topk} element classes by support_true (Multiset F1 detail) ──",
            f"  {'Z':<4s} {'Sym':<4s}  {'F1':>7s}   {'support_true':>13s}  "
                f"{'support_pred':>13s}",
        ]
        for z, sup_t in sorted_classes[:topk]:
            sym = _z_symbol(int(z))
            f1  = pcf1.get(z, 0.0)
            sup_p = pcs_pred.get(z, 0)
            lines.append(f"  {z:<4d} {sym:<4s}  {f1:>7.4f}   {sup_t:>13d}  {sup_p:>13d}")
        lines.append("")

    # Stratified by eval_cutoff (Exp4 carry-over diagnostic)
    lines += [
        "── Stratified by eval_cutoff (4-tier, Exp4 diagnostic carry-over) ──",
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

    # §6 verdict (Exp4 thresholds, preliminary signal only)
    rmsd_flag, type_flag, cutoff_flag = verdict_per_metric(
        rmsds.mean(), type_accs.mean(), n_pred_in.mean())
    lines += [
        "",
        "── Verdict (STEP5_HANDOFF §6 thresholds, Exp4 signal — MA5 makes go/no-go) ──",
        f"  RMSD (Geometry)             {rmsds.mean():.4f}     {rmsd_flag}",
        f"  Position TypeAcc (历史)     {type_accs.mean():.4f}     {type_flag}",
        f"  pred_in_cutoff (Geometry)   {n_pred_in.mean():.2f}/20    {cutoff_flag}",
        "",
        "  Caveat: Exp5 v2 真信号是 Set-Level / Multiset / Collapse 三件套.",
        "          §6 Exp4 verdict 仅作 Geometry 通道的预诊断. MA5 决定整体 go/no-go.",
        "=" * 72,
    ]

    for ln in lines:
        logger.info(ln)

    # ── Write text report ─────────────────────────────────────────────
    report_path = (args.output if args.output
                   else os.path.join(args.out_dir, f"metrics_report_{split}.txt"))
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"\n  written: {report_path}")

    # ── Write per-sample CSV (Step6Agent input) ───────────────────────
    csv_path = (args.csv_output if args.csv_output
                else os.path.join(args.out_dir, f"per_sample_metrics_{split}.csv"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_name", "mp_id", "rmsd", "type_acc",
                    "set_level_typeacc",
                    "n_pred_in", "n_true_in", "eval_cutoff",
                    "pred_xyz_std_A", "true_xyz_std_A"])
        per_sample_set_level = set_level['per_sample_values']
        pred_std_dist = collapse['pred_std_dist']
        true_std_dist = collapse['true_std_dist']
        for idx, r in enumerate(results):
            w.writerow([
                r['sample_name'], r['mp_id'],
                f"{r['rmsd']:.6f}", f"{r['type_acc']:.6f}",
                f"{per_sample_set_level[idx]:.6f}",
                r['n_pred_in'], r['n_true_in'],
                f"{r['eval_cutoff']:.6f}",
                f"{pred_std_dist[idx]:.6f}",
                f"{true_std_dist[idx]:.6f}",
            ])
    logger.info(f"  written: {csv_path}  ({n_eff} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "holdout"], required=True)
    ap.add_argument("--out_dir", default=OUT_DIR,
                    help="default output dir for report + CSV")
    ap.add_argument("--predictions", default=None,
                    help="override predictions_{split}.pt path "
                         "(e.g. for Exp4 baseline dry-run)")
    ap.add_argument("--output", default=None,
                    help="override metrics_report_{split}.txt path")
    ap.add_argument("--csv_output", default=None,
                    help="override per_sample_metrics_{split}.csv path")
    args = ap.parse_args()

    if args.split == "holdout":
        logger.warning("⚠️  HOLDOUT METRICS — SA3 phase only; SA1' should NOT call holdout.")

    logger.info("=" * 60)
    logger.info(f"Step 5.2  metrics  split={args.split}  L={L}  (Exp5 v2 SA1')")
    logger.info("=" * 60)
    compute_metrics(args.split, args)


if __name__ == "__main__":
    main()
