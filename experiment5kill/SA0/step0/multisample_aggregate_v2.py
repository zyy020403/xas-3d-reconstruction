#!/usr/bin/env python
"""
multisample_aggregate_v2.py
========================================================================
Exp5 SA0 — bugfix re-aggregation for the K-averaging quick win.

The first aggregate run showed RMSD getting WORSE with K (naive K=10
RMSD=2.35 vs K=1=1.49). Investigation revealed two implementation bugs
and motivates two extra baselines:

  Strategy A  hungarian_fold      — ★ THE BUG FIX
      Old code aligned slots via Hungarian then naively averaged
      raw frac coords. That breaks across the [-0.5, 0.5] boundary
      because frac coords live on a torus. Fix: after Hungarian
      alignment, fold each aligned k to the anchor's local neighborhood
      via min-image (subtract round((aligned - anchor))) BEFORE mean,
      then wrap result back to [-0.5, 0.5].

  Strategy B  hungarian_fold_bestanchor
      Old code fixed anchor=k=0. New: try each k as anchor, pick the
      one minimizing post-aggregation RMSD-vs-truth (still cheating?
      no — it's intra-sample selection, not test-set leakage).

  Strategy C  medoid
      Skip averaging entirely. Pick the k that minimizes total
      pairwise RMSD to other K-1. Robust to torus-averaging issues.

  Strategy D  oracle_best   (UPPER BOUND, cheats with ground truth)
      Of the K candidates, pick the one with lowest RMSD vs truth.
      Tells us how much "good sample" exists in the K — if oracle
      RMSD << K=1 mean, aggregation has headroom; if oracle ≈ K=1,
      averaging strategies are doomed regardless.

Reuses verbatim Exp4 step5_2 evaluate_sample for fair comparison.

Inputs:  /home/tcat/diffcsp_exp5/sa0/results/samples_raw_K10.pt
         /home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv

Outputs: /home/tcat/diffcsp_exp5/sa0/results/multisample_v2_raw.csv
         /home/tcat/diffcsp_exp5/sa0/results/multisample_v2_results.md
         /home/tcat/diffcsp_exp5/sa0/results/multisample_v2_K_curves.png

Wall: ~3 min CPU-only on the 500-subset × 4 strategies × {1,5,10}.

Usage:
  cd /home/tcat/diffcsp_exp5/sa0/scripts
  python multisample_aggregate_v2.py \
      --samples_pt /home/tcat/diffcsp_exp5/sa0/results/samples_raw_K10.pt \
      --out_dir    /home/tcat/diffcsp_exp5/sa0/results
"""
import argparse, os, csv, logging, sys
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

L = 6.0
EXP4_K1_FULL_VAL = {"RMSD": 1.4849, "TypeAcc": 0.1877, "pred_in": 18.93}


# ── Exp4 step5_2's evaluate_sample (verbatim) ────────────────────────
def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=L):
    from scipy.optimize import linear_sum_assignment
    pred_frac = np.asarray(pred_frac, dtype=np.float64)
    true_frac = np.asarray(true_frac, dtype=np.float64)
    pred_types = np.asarray(pred_types).flatten()
    true_types = np.asarray(true_types).flatten()
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
    pred_dists = np.linalg.norm(pred_mi * L, axis=1)
    n_pred_in = int((pred_dists <= eval_cutoff).sum())
    return {"rmsd": rmsd, "type_acc": type_acc, "n_pred_in": n_pred_in}


def _to_np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


# ── Strategy implementations ─────────────────────────────────────────
def aggregate_naive(frac_K, types_K):
    """Per-slot mean (no fold) + per-slot mode. Buggy on torus — kept
    here for direct reproduction of the v1 numbers."""
    from scipy.stats import mode
    fc = _to_np(frac_K)
    at = _to_np(types_K)
    return fc.mean(axis=0), mode(at, axis=0, keepdims=False).mode


def _hungarian_align_one(fc_k, fc_anchor, at_k, L=L):
    """Slot-align fc_k to fc_anchor via min-image Hungarian on coords.
    Returns aligned coords (NOT yet folded) and aligned types in anchor's slot order."""
    from scipy.optimize import linear_sum_assignment
    N = fc_k.shape[0]
    cost = np.zeros((N, N))
    for i in range(N):
        delta = fc_k[i] - fc_anchor
        delta -= np.round(delta)
        cost[i] = np.linalg.norm(delta * L, axis=1)
    row, col = linear_sum_assignment(cost)
    aligned_c = np.zeros_like(fc_anchor)
    aligned_t = np.zeros_like(at_k)
    aligned_c[col] = fc_k[row]
    aligned_t[col] = at_k[row]
    return aligned_c, aligned_t


def aggregate_hungarian_fold(frac_K, types_K, anchor_k=0, L=L):
    """★ THE BUG FIX: Hungarian-align all to anchor, FOLD aligned to
    anchor's neighborhood via min-image, then mean + wrap to [-0.5, 0.5]."""
    from scipy.stats import mode
    fc = _to_np(frac_K)
    at = _to_np(types_K)
    K = fc.shape[0]
    if K == 1:
        return fc[0], at[0]
    anchor_coords = fc[anchor_k]
    anchor_types  = at[anchor_k]
    aligned_c = [anchor_coords]
    aligned_t = [anchor_types]
    for k in range(K):
        if k == anchor_k:
            continue
        ac, at_a = _hungarian_align_one(fc[k], anchor_coords, at[k], L=L)
        # Fold into anchor's neighborhood: subtract round(diff) so each slot
        # sits within ±0.5 of the anchor's coord component-wise
        ac = ac - np.round(ac - anchor_coords)
        aligned_c.append(ac)
        aligned_t.append(at_a)
    aligned_c = np.stack(aligned_c)
    aligned_t = np.stack(aligned_t)
    mean_c = aligned_c.mean(axis=0)
    # Wrap result back to canonical [-0.5, 0.5] range
    mean_c = mean_c - np.round(mean_c)
    return mean_c, mode(aligned_t, axis=0, keepdims=False).mode


def aggregate_hungarian_fold_bestanchor(frac_K, types_K, true_frac, true_types, ec, L=L):
    """Try each k as anchor, pick anchor giving lowest RMSD-vs-truth.
    Note: 'best' is selected via ground truth, so this is a near-oracle
    for anchor choice — but the aggregation itself is still legitimate."""
    K = _to_np(frac_K).shape[0]
    best = None
    best_anchor = None
    for k in range(K):
        ac, at_ = aggregate_hungarian_fold(frac_K, types_K, anchor_k=k, L=L)
        m = evaluate_sample(ac, at_, true_frac, true_types, ec, L=L)
        if best is None or m["rmsd"] < best["rmsd"]:
            best = m
            best_anchor = k
            best_c, best_t = ac, at_
    return best_c, best_t, {"best_anchor": best_anchor}


def aggregate_medoid(frac_K, types_K, L=L):
    """Pick the single k minimizing sum of pairwise RMSD to other K-1.
    No averaging — just picks the most central candidate."""
    fc = _to_np(frac_K)
    at = _to_np(types_K)
    K = fc.shape[0]
    if K == 1:
        return fc[0], at[0], {"medoid_k": 0}
    pair_rmsd = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            # RMSD between two K-candidates with min-image, no Hungarian
            # (slot order is consistent within a single sample's K outputs
            # because dataset feeds the same atoms in the same order each sweep)
            delta = fc[i] - fc[j]
            delta -= np.round(delta)
            r = float(np.sqrt(np.mean(np.sum((delta * L) ** 2, axis=1))))
            pair_rmsd[i, j] = pair_rmsd[j, i] = r
    medoid_k = int(np.argmin(pair_rmsd.sum(axis=1)))
    return fc[medoid_k], at[medoid_k], {"medoid_k": medoid_k}


def aggregate_oracle_best(frac_K, types_K, true_frac, true_types, ec, L=L):
    """UPPER BOUND: pick k minimizing RMSD-vs-truth. Cheats with truth.
    Tells us how much 'good sample' exists in the K candidates."""
    K = _to_np(frac_K).shape[0]
    best = None
    best_k = None
    fc = _to_np(frac_K)
    at = _to_np(types_K)
    for k in range(K):
        m = evaluate_sample(fc[k], at[k], true_frac, true_types, ec, L=L)
        if best is None or m["rmsd"] < best["rmsd"]:
            best = m
            best_k = k
            best_c, best_t = fc[k], at[k]
    return best_c, best_t, {"oracle_k": best_k}


def _agg_stats(stats):
    rmsd = np.array([s["rmsd"]      for s in stats])
    ta   = np.array([s["type_acc"]  for s in stats])
    pi   = np.array([s["n_pred_in"] for s in stats])
    n = len(stats)
    return {
        "n":         n,
        "rmsd_mean": float(rmsd.mean()), "rmsd_std": float(rmsd.std(ddof=1)),
        "rmsd_se":   float(rmsd.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
        "ta_mean":   float(ta.mean()),
        "ta_se":     float(ta.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
        "pi_mean":   float(pi.mean()),
        "pi_se":     float(pi.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_pt",   required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--K_values",     type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--exp4_psm_csv", default="/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv")
    args = ap.parse_args()

    STRATEGIES = [
        "naive",
        "hungarian_fold",            # bug-fix
        "hungarian_fold_bestanchor", # bug-fix + best-anchor selection
        "medoid",                    # alternative: no averaging
        "oracle_best",               # upper bound (cheats)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Exp5 SA0  multisample_aggregate_v2  (bugfix + extras)")
    logger.info(f"  samples_pt : {args.samples_pt}")
    logger.info(f"  out_dir    : {args.out_dir}")
    logger.info(f"  K_values   : {args.K_values}")
    logger.info(f"  strategies : {STRATEGIES}")
    logger.info("=" * 60)

    data = torch.load(args.samples_pt, map_location="cpu", weights_only=False)
    K_max = data["K"]
    n_samples = len(data["sample_names"])
    logger.info(f"\n  loaded:  K_max={K_max}  n_kept={n_samples}  "
                f"ckpt_md5={data['ckpt_md5_full'][:12]}...")
    if max(args.K_values) > K_max:
        logger.error(f"❌ K={max(args.K_values)} > K_max={K_max}")
        sys.exit(1)

    # Subset Exp4 K=1 reference
    name2exp4 = {}
    if os.path.isfile(args.exp4_psm_csv):
        with open(args.exp4_psm_csv) as f:
            for r in csv.DictReader(f):
                name2exp4[r["sample_name"]] = {
                    "rmsd":     float(r["rmsd"]),
                    "type_acc": float(r["type_acc"]),
                    "n_pred_in":int(r["n_pred_in"]),
                }
    sub = [name2exp4[n] for n in data["sample_names"] if n in name2exp4]
    exp4_K1_subset = None
    if len(sub) == n_samples:
        rmsd_arr = np.array([x["rmsd"]     for x in sub])
        ta_arr   = np.array([x["type_acc"] for x in sub])
        pi_arr   = np.array([x["n_pred_in"]for x in sub])
        n = n_samples
        exp4_K1_subset = {
            "RMSD":      float(rmsd_arr.mean()),
            "RMSD_se":   float(rmsd_arr.std(ddof=1)/np.sqrt(n)),
            "TypeAcc":   float(ta_arr.mean()),
            "TypeAcc_se":float(ta_arr.std(ddof=1)/np.sqrt(n)),
            "pred_in":   float(pi_arr.mean()),
            "pred_in_se":float(pi_arr.std(ddof=1)/np.sqrt(n)),
        }
        logger.info(f"\n  Exp4 K=1 subset: RMSD={exp4_K1_subset['RMSD']:.4f}  "
                    f"TypeAcc={exp4_K1_subset['TypeAcc']:.4f}  pred_in={exp4_K1_subset['pred_in']:.2f}")

    # Eval loop
    logger.info("\n  evaluating ...")
    rows = []
    summary = {}
    per_tier = {}
    for K_eval in args.K_values:
        for strat in STRATEGIES:
            summary[(K_eval, strat)] = []
            per_tier[(K_eval, strat)] = {"A": [], "B": [], "C": [], "D": []}

    import time
    t0 = time.time()
    last_log = t0
    for i, name in enumerate(data["sample_names"]):
        fc_full = data["pred_frac_coords_K"][i]
        at_full = data["pred_atom_types_K"][i]
        tf      = _to_np(data["true_frac_coords"][i])
        tt      = _to_np(data["true_atom_types"][i])
        ec      = data["eval_cutoffs"][i]
        tier    = data["tiers"][i]
        for K_eval in args.K_values:
            fc_K = fc_full[:K_eval]
            at_K = at_full[:K_eval]
            for strat in STRATEGIES:
                meta = {}
                if strat == "naive":
                    agg_c, agg_t = aggregate_naive(fc_K, at_K)
                elif strat == "hungarian_fold":
                    agg_c, agg_t = aggregate_hungarian_fold(fc_K, at_K, anchor_k=0, L=L)
                elif strat == "hungarian_fold_bestanchor":
                    agg_c, agg_t, meta = aggregate_hungarian_fold_bestanchor(fc_K, at_K, tf, tt, ec, L=L)
                elif strat == "medoid":
                    agg_c, agg_t, meta = aggregate_medoid(fc_K, at_K, L=L)
                elif strat == "oracle_best":
                    agg_c, agg_t, meta = aggregate_oracle_best(fc_K, at_K, tf, tt, ec, L=L)
                else:
                    raise ValueError(strat)
                m = evaluate_sample(agg_c, agg_t, tf, tt, ec, L=L)
                m.update({"sample_name": name, "tier": tier, "K": K_eval,
                          "strategy": strat, "eval_cutoff": ec, **meta})
                rows.append(m)
                summary[(K_eval, strat)].append(m)
                per_tier[(K_eval, strat)][tier].append(m)
        if time.time() - last_log > 30:
            logger.info(f"    progress: {i+1}/{n_samples}  ({(time.time()-t0):.0f}s elapsed)")
            last_log = time.time()
    logger.info(f"  eval done: {len(rows)} rows  wall={time.time()-t0:.0f}s")

    # Raw CSV
    raw_csv = os.path.join(args.out_dir, "multisample_v2_raw.csv")
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_name","tier","eval_cutoff","K","strategy",
                    "rmsd","type_acc","n_pred_in","extra"])
        for r in rows:
            extra = ",".join(f"{k}={v}" for k,v in r.items()
                             if k in ("best_anchor","medoid_k","oracle_k"))
            w.writerow([r["sample_name"], r["tier"], f"{r['eval_cutoff']:.4f}",
                        r["K"], r["strategy"], f"{r['rmsd']:.6f}",
                        f"{r['type_acc']:.6f}", r["n_pred_in"], extra])
    logger.info(f"  wrote: {raw_csv}")

    overall = {k: _agg_stats(v) for k, v in summary.items()}
    per_tier_agg = {k: {t: _agg_stats(v) for t, v in d.items() if v}
                    for k, d in per_tier.items()}

    # Plot
    plot_path = os.path.join(args.out_dir, "multisample_v2_K_curves.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ks_sorted = sorted(set(args.K_values))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {
            "naive":                     "tab:red",
            "hungarian_fold":            "tab:blue",
            "hungarian_fold_bestanchor": "tab:cyan",
            "medoid":                    "tab:green",
            "oracle_best":               "tab:purple",
        }
        styles = {
            "oracle_best":               "--",   # dashed: cheating
            "hungarian_fold_bestanchor": "--",   # dashed: also semi-cheating (uses truth for anchor)
        }
        for strat in STRATEGIES:
            rmsd_y = [overall[(K, strat)]["rmsd_mean"] for K in Ks_sorted]
            rmsd_e = [overall[(K, strat)]["rmsd_se"]   for K in Ks_sorted]
            ta_y   = [overall[(K, strat)]["ta_mean"]   for K in Ks_sorted]
            ta_e   = [overall[(K, strat)]["ta_se"]     for K in Ks_sorted]
            ls = styles.get(strat, "-")
            axes[0].errorbar(Ks_sorted, rmsd_y, yerr=rmsd_e, marker="o",
                             label=strat, color=colors.get(strat), capsize=3, ls=ls)
            axes[1].errorbar(Ks_sorted, ta_y, yerr=ta_e, marker="o",
                             label=strat, color=colors.get(strat), capsize=3, ls=ls)
        if exp4_K1_subset is not None:
            axes[0].axhline(exp4_K1_subset["RMSD"], color="k", ls=":", lw=0.8,
                            label="Exp4 K=1 subset")
            axes[1].axhline(exp4_K1_subset["TypeAcc"], color="k", ls=":", lw=0.8,
                            label="Exp4 K=1 subset")
        axes[0].set_xlabel("K"); axes[0].set_ylabel("Mean RMSD (Å) ↓")
        axes[0].set_title(f"RMSD vs K  (n={n_samples}; dashed = cheats with truth)")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_xticks(Ks_sorted)
        axes[1].set_xlabel("K"); axes[1].set_ylabel("Mean TypeAcc ↑")
        axes[1].set_title(f"TypeAcc vs K  (n={n_samples})")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); axes[1].set_xticks(Ks_sorted)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"  wrote: {plot_path}")
    except Exception as e:
        logger.warning(f"  plot failed: {e}")
        plot_path = None

    # K=1 sanity check (any non-cheating strategy gives the same K=1 number)
    sa0_K1 = overall[(1, "naive")]
    sanity_pass = True
    sanity_lines = []
    if exp4_K1_subset is not None:
        for label, sa0_key, ref_key in [
            ("RMSD",    "rmsd_mean", "RMSD"),
            ("TypeAcc", "ta_mean",   "TypeAcc"),
            ("pred_in", "pi_mean",   "pred_in"),
        ]:
            sa0_v  = sa0_K1[sa0_key]
            ref_v  = exp4_K1_subset[ref_key]
            ref_se = exp4_K1_subset[f"{ref_key}_se"]
            lo, hi = ref_v - 2 * ref_se, ref_v + 2 * ref_se
            ok = lo <= sa0_v <= hi
            sanity_pass = sanity_pass and ok
            sanity_lines.append(
                f"  {label:8s}  SA0_K1={sa0_v:.4f}  Exp4_K1_subset={ref_v:.4f}  "
                f"±2SE_band=[{lo:.4f}, {hi:.4f}]  {'✓ PASS' if ok else '❌ FAIL'}"
            )

    # Markdown report
    md_path = os.path.join(args.out_dir, "multisample_v2_results.md")
    tier_count = {}
    for t in data["tiers"]:
        tier_count[t] = tier_count.get(t, 0) + 1

    with open(md_path, "w") as f:
        f.write("# Exp5 SA0 — Multi-Sample Averaging  (v2: bug-fix + baselines)\n\n")
        f.write("## What changed from v1\n\n")
        f.write("v1's `hungarian` strategy aligned slots correctly but then averaged "
                "raw frac coords across the [-0.5, 0.5] toroidal boundary — folding "
                "atom positions toward 0 by accident. This v2 adds:\n\n")
        f.write("- **`hungarian_fold`** (bug fix): same Hungarian alignment, but "
                "fold each aligned k into the anchor's neighborhood via min-image "
                "before averaging, then wrap result back to [-0.5, 0.5].\n")
        f.write("- **`hungarian_fold_bestanchor`** (semi-oracle): try each k as the "
                "anchor, keep the one with lowest RMSD vs truth. Uses ground truth for "
                "anchor choice → lower bound for the bug-fixed strategy in deployment.\n")
        f.write("- **`medoid`**: pick the single k minimizing sum of pairwise RMSD to the "
                "other K-1. Skips averaging entirely. Pure deployment-safe.\n")
        f.write("- **`oracle_best`** (upper bound; cheats): pick the k of K with lowest "
                "RMSD vs truth. Tells us how much aggregation has to give up.\n\n")

        f.write("## Subset & sanity (unchanged from v1)\n\n")
        f.write(f"- {n_samples} val samples (B={tier_count.get('B',0)} "
                f"C={tier_count.get('C',0)} D={tier_count.get('D',0)})\n")
        f.write(f"- ckpt md5: `{data['ckpt_md5_full']}`  epoch=366  K_max={K_max}\n")
        f.write(f"- torch {data.get('torch_version','?')} on {data['device']}; "
                f"sample wall: 86.3 min for K={K_max} on RTX 4090 (8.6 min/sweep)\n")
        if exp4_K1_subset is not None:
            f.write("\n```\n")
            for ln in sanity_lines:
                f.write(ln + "\n")
            f.write("```\n\n")
            f.write(f"Sanity overall: **{'✓ PASS' if sanity_pass else '❌ FAIL'}**\n\n")

        f.write("## Main results — RMSD by K and strategy\n\n")
        f.write("(rows marked † use ground truth; not deployable as-is)\n\n")
        f.write("| K  | Strategy                        | RMSD   | ΔvsK1   | TypeAcc | pred_in |\n")
        f.write("|---:|---------------------------------|-------:|--------:|--------:|--------:|\n")
        ref_rmsd = sa0_K1["rmsd_mean"]
        ref_ta   = sa0_K1["ta_mean"]
        for K_eval in sorted(args.K_values):
            for strat in STRATEGIES:
                if (K_eval, strat) not in overall:
                    continue
                s = overall[(K_eval, strat)]
                drm = s["rmsd_mean"] - ref_rmsd
                marker = "†" if strat in ("hungarian_fold_bestanchor","oracle_best") else " "
                lbl = "(any)" if K_eval == 1 else strat
                f.write(f"| {K_eval} | {lbl + marker:32s} | "
                        f"{s['rmsd_mean']:.4f} | {drm:+.4f} | "
                        f"{s['ta_mean']:.4f} | {s['pi_mean']:.2f} |\n")
                if K_eval == 1:
                    break  # all strategies identical at K=1

        f.write("\n## Per-Tier breakdown — Tier B (n={})\n\n".format(tier_count.get('B', 0)))
        f.write("| K  | Strategy                        | RMSD   | TypeAcc | pred_in |\n")
        f.write("|---:|---------------------------------|-------:|--------:|--------:|\n")
        for K_eval in sorted(args.K_values):
            for strat in STRATEGIES:
                if (K_eval, strat) not in per_tier_agg or "B" not in per_tier_agg[(K_eval, strat)]:
                    continue
                s = per_tier_agg[(K_eval, strat)]["B"]
                lbl = "(any)" if K_eval == 1 else strat
                f.write(f"| {K_eval} | {lbl:32s} | {s['rmsd_mean']:.4f} | "
                        f"{s['ta_mean']:.4f} | {s['pi_mean']:.2f} |\n")
                if K_eval == 1:
                    break

        f.write("\n## Per-Tier breakdown — Tier C (n={})\n\n".format(tier_count.get('C', 0)))
        f.write("| K  | Strategy                        | RMSD   | TypeAcc | pred_in |\n")
        f.write("|---:|---------------------------------|-------:|--------:|--------:|\n")
        for K_eval in sorted(args.K_values):
            for strat in STRATEGIES:
                if (K_eval, strat) not in per_tier_agg or "C" not in per_tier_agg[(K_eval, strat)]:
                    continue
                s = per_tier_agg[(K_eval, strat)]["C"]
                lbl = "(any)" if K_eval == 1 else strat
                f.write(f"| {K_eval} | {lbl:32s} | {s['rmsd_mean']:.4f} | "
                        f"{s['ta_mean']:.4f} | {s['pi_mean']:.2f} |\n")
                if K_eval == 1:
                    break

        f.write("\n## Per-Tier breakdown — Tier D (n={})\n\n".format(tier_count.get('D', 0)))
        f.write("| K  | Strategy                        | RMSD   | TypeAcc | pred_in |\n")
        f.write("|---:|---------------------------------|-------:|--------:|--------:|\n")
        for K_eval in sorted(args.K_values):
            for strat in STRATEGIES:
                if (K_eval, strat) not in per_tier_agg or "D" not in per_tier_agg[(K_eval, strat)]:
                    continue
                s = per_tier_agg[(K_eval, strat)]["D"]
                lbl = "(any)" if K_eval == 1 else strat
                f.write(f"| {K_eval} | {lbl:32s} | {s['rmsd_mean']:.4f} | "
                        f"{s['ta_mean']:.4f} | {s['pi_mean']:.2f} |\n")
                if K_eval == 1:
                    break

        if plot_path:
            f.write(f"\n## Curves\n\n![K vs metrics]({os.path.basename(plot_path)})\n\n")
            f.write("Solid = deployment-safe; dashed = cheats with ground truth.\n")

    logger.info(f"  wrote: {md_path}")
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE.  out_dir={args.out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
