#!/usr/bin/env python
"""
multisample_aggregate.py
========================================================================
Exp5 SA0 — aggregate K-sample TTA results, evaluate, compare strategies.

Reads:  samples_raw_K{K}.pt  (output of multisample.py)
Writes: multisample_raw.csv          (per-sample × K' × strategy)
        multisample_results.md       (summary with sanity gate + tables)
        multisample_K_curves.png     (K' vs RMSD/TypeAcc, both strategies)

Strategies:
  - naive: per-slot mean coords + per-slot mode types (assumes slot stability)
  - hungarian: align k=1..K-1 to k=0 via min-image Hungarian, then naive on aligned

Eval: Exp4 step5_2's evaluate_sample() — 20×20 Hungarian + min-image RMSD.

Sanity gate: SA0 K=1 subset mean must lie in Exp4 K=1 (same subset) ±2·SE.

Usage:
  python multisample_aggregate.py \
    --samples_pt /home/tcat/diffcsp_exp5/sa0/results/samples_raw_K10.pt \
    --out_dir    /home/tcat/diffcsp_exp5/sa0/results \
    --K_values 1 5 10 \
    --strategies naive hungarian
"""

import argparse, os, csv, logging, sys
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

L = 6.0

# Exp4 §5.1 full-val K=1 baseline (per_sample_metrics_val.csv mean over 7621 samples)
EXP4_K1_FULL_VAL = {"RMSD": 1.4849, "TypeAcc": 0.1877, "pred_in": 18.93}


# ── eval (verbatim from Exp4 step5_2) ────────────────────────────────────
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


# ── aggregation strategies ───────────────────────────────────────────────
def _to_np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def aggregate_naive(frac_K, types_K):
    """Plain per-slot mean (coords) + per-slot mode (types).
    Naive in the handoff §2.3 sense — does NOT do min-image folding."""
    from scipy.stats import mode
    fc = _to_np(frac_K)   # (K, 20, 3)
    at = _to_np(types_K)  # (K, 20)
    agg_coords = fc.mean(axis=0)
    agg_types  = mode(at, axis=0, keepdims=False).mode
    return agg_coords, agg_types


def aggregate_hungarian(frac_K, types_K, L=L):
    """Anchor=k=0, align k=1..K-1 via min-image Hungarian on coords, then naive
    on aligned. Per handoff §2.3 strategy Y."""
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import mode
    fc = _to_np(frac_K)
    at = _to_np(types_K)
    K, N, _ = fc.shape
    if K == 1:
        return fc[0], at[0]
    anchor_coords = fc[0]
    aligned_c = [anchor_coords]
    aligned_t = [at[0]]
    for k in range(1, K):
        cost = np.zeros((N, N))
        for i in range(N):
            delta = fc[k][i] - anchor_coords
            delta -= np.round(delta)
            cost[i] = np.linalg.norm(delta * L, axis=1)
        row, col = linear_sum_assignment(cost)
        rc = np.zeros_like(anchor_coords)
        rt = np.zeros_like(at[0])
        rc[col] = fc[k][row]
        rt[col] = at[k][row]
        aligned_c.append(rc)
        aligned_t.append(rt)
    aligned_c = np.stack(aligned_c)
    aligned_t = np.stack(aligned_t)
    return aligned_c.mean(axis=0), mode(aligned_t, axis=0, keepdims=False).mode


def _agg_stats(stats):
    rmsd = np.array([s["rmsd"]      for s in stats])
    ta   = np.array([s["type_acc"]  for s in stats])
    pi   = np.array([s["n_pred_in"] for s in stats])
    n = len(stats)
    return {
        "n":         n,
        "rmsd_mean": float(rmsd.mean()), "rmsd_std": float(rmsd.std(ddof=1)),
        "rmsd_se":   float(rmsd.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
        "ta_mean":   float(ta.mean()),   "ta_std":   float(ta.std(ddof=1)),
        "ta_se":     float(ta.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
        "pi_mean":   float(pi.mean()),   "pi_std":   float(pi.std(ddof=1)),
        "pi_se":     float(pi.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0,
    }


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_pt",   required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--K_values",     type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--strategies",   nargs="+", default=["naive", "hungarian"])
    ap.add_argument("--exp4_psm_csv", default="/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Exp5 SA0  multisample_aggregate")
    logger.info(f"  samples_pt : {args.samples_pt}")
    logger.info(f"  out_dir    : {args.out_dir}")
    logger.info(f"  K_values   : {args.K_values}")
    logger.info(f"  strategies : {args.strategies}")
    logger.info("=" * 60)

    data = torch.load(args.samples_pt, map_location="cpu", weights_only=False)
    K_max = data["K"]
    n_samples = len(data["sample_names"])
    logger.info(f"\n  loaded:  K_max={K_max}  n_kept={n_samples}  "
                f"ckpt_md5={data['ckpt_md5_full'][:12]}...  "
                f"wall_total={sum(data['wall_per_K'])/60:.1f} min")
    if max(args.K_values) > K_max:
        logger.error(f"❌ requested K={max(args.K_values)} > K_max in samples_pt={K_max}")
        sys.exit(1)

    # ── Subset Exp4 K=1 reference ───────────────────────────────────
    name2exp4 = {}
    if os.path.isfile(args.exp4_psm_csv):
        with open(args.exp4_psm_csv) as f:
            for r in csv.DictReader(f):
                name2exp4[r["sample_name"]] = {
                    "rmsd":      float(r["rmsd"]),
                    "type_acc":  float(r["type_acc"]),
                    "n_pred_in": int(r["n_pred_in"]),
                }
    sub = [name2exp4[n] for n in data["sample_names"] if n in name2exp4]
    exp4_K1_subset = None
    if len(sub) == n_samples:
        rmsd_arr = np.array([x["rmsd"]      for x in sub])
        ta_arr   = np.array([x["type_acc"]  for x in sub])
        pi_arr   = np.array([x["n_pred_in"] for x in sub])
        n = n_samples
        exp4_K1_subset = {
            "RMSD":         float(rmsd_arr.mean()),
            "RMSD_se":      float(rmsd_arr.std(ddof=1)/np.sqrt(n)),
            "TypeAcc":      float(ta_arr.mean()),
            "TypeAcc_se":   float(ta_arr.std(ddof=1)/np.sqrt(n)),
            "pred_in":      float(pi_arr.mean()),
            "pred_in_se":   float(pi_arr.std(ddof=1)/np.sqrt(n)),
        }
        logger.info(f"\n  Exp4 K=1 subset reference (n={n}):")
        logger.info(f"    RMSD     mean={exp4_K1_subset['RMSD']:.4f}  ±2SE=[{exp4_K1_subset['RMSD']-2*exp4_K1_subset['RMSD_se']:.4f}, {exp4_K1_subset['RMSD']+2*exp4_K1_subset['RMSD_se']:.4f}]")
        logger.info(f"    TypeAcc  mean={exp4_K1_subset['TypeAcc']:.4f}  ±2SE=[{exp4_K1_subset['TypeAcc']-2*exp4_K1_subset['TypeAcc_se']:.4f}, {exp4_K1_subset['TypeAcc']+2*exp4_K1_subset['TypeAcc_se']:.4f}]")
        logger.info(f"    pred_in  mean={exp4_K1_subset['pred_in']:.2f}  ±2SE=[{exp4_K1_subset['pred_in']-2*exp4_K1_subset['pred_in_se']:.2f}, {exp4_K1_subset['pred_in']+2*exp4_K1_subset['pred_in_se']:.2f}]")
    else:
        logger.warning(f"  only {len(sub)}/{n_samples} subset names found in PSM csv; sanity will be skipped")

    # ── Per-sample × K × strategy eval ──────────────────────────────
    logger.info("\n  evaluating ...")
    rows = []
    summary = {}    # (K_eval, strat) → list of stat dicts (overall)
    per_tier = {}   # (K_eval, strat) → {tier: list}
    for K_eval in args.K_values:
        for strat in args.strategies:
            key = (K_eval, strat)
            summary[key] = []
            per_tier[key] = {"A": [], "B": [], "C": [], "D": []}

    for i, name in enumerate(data["sample_names"]):
        fc_full = data["pred_frac_coords_K"][i]   # (K_max, 20, 3) tensor
        at_full = data["pred_atom_types_K"][i]    # (K_max, 20)    tensor
        tf      = _to_np(data["true_frac_coords"][i])
        tt      = _to_np(data["true_atom_types"][i])
        ec      = data["eval_cutoffs"][i]
        tier    = data["tiers"][i]
        for K_eval in args.K_values:
            fc_K = fc_full[:K_eval]
            at_K = at_full[:K_eval]
            for strat in args.strategies:
                if strat == "naive":
                    agg_c, agg_t = aggregate_naive(fc_K, at_K)
                elif strat == "hungarian":
                    agg_c, agg_t = aggregate_hungarian(fc_K, at_K, L=L)
                else:
                    raise ValueError(strat)
                m = evaluate_sample(agg_c, agg_t, tf, tt, ec, L=L)
                m.update({"sample_name": name, "tier": tier, "K": K_eval,
                          "strategy": strat, "eval_cutoff": ec})
                rows.append(m)
                summary[(K_eval, strat)].append(m)
                per_tier[(K_eval, strat)][tier].append(m)

    # ── Write per-sample raw CSV ────────────────────────────────────
    raw_csv = os.path.join(args.out_dir, "multisample_raw.csv")
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_name", "tier", "eval_cutoff", "K", "strategy",
                    "rmsd", "type_acc", "n_pred_in"])
        for r in rows:
            w.writerow([r["sample_name"], r["tier"], f"{r['eval_cutoff']:.4f}",
                        r["K"], r["strategy"],
                        f"{r['rmsd']:.6f}", f"{r['type_acc']:.6f}", r["n_pred_in"]])
    logger.info(f"  wrote: {raw_csv}")

    # ── Aggregate stats ─────────────────────────────────────────────
    overall = {k: _agg_stats(v) for k, v in summary.items()}
    per_tier_agg = {k: {t: _agg_stats(v) for t, v in d.items() if v}
                    for k, d in per_tier.items()}

    sa0_K1 = None
    for strat in args.strategies:
        if (1, strat) in overall:
            sa0_K1 = overall[(1, strat)]
            break

    # ── Sanity gate ─────────────────────────────────────────────────
    sanity_lines = []
    sanity_pass = True
    if exp4_K1_subset is not None and sa0_K1 is not None:
        for label, sa0_key, ref_key in [
            ("RMSD",    "rmsd_mean", "RMSD"),
            ("TypeAcc", "ta_mean",   "TypeAcc"),
            ("pred_in", "pi_mean",   "pred_in"),
        ]:
            sa0_v = sa0_K1[sa0_key]
            ref_v = exp4_K1_subset[ref_key]
            ref_se = exp4_K1_subset[f"{ref_key}_se"]
            lo, hi = ref_v - 2 * ref_se, ref_v + 2 * ref_se
            ok = lo <= sa0_v <= hi
            sanity_pass = sanity_pass and ok
            sanity_lines.append(
                f"  {label:8s}  SA0_K1={sa0_v:.4f}  Exp4_K1_subset={ref_v:.4f}  "
                f"±2SE_band=[{lo:.4f}, {hi:.4f}]  {'✓ PASS' if ok else '❌ FAIL'}"
            )
        logger.info("\n  sanity:")
        for ln in sanity_lines:
            logger.info(ln)
        logger.info(f"  overall: {'✓ PASS' if sanity_pass else '❌ FAIL'}")

    # ── Plot ────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, "multisample_K_curves.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ks_sorted = sorted(set(args.K_values))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = {"naive": "tab:blue", "hungarian": "tab:orange"}
        for strat in args.strategies:
            rmsd_y = [overall[(K, strat)]["rmsd_mean"] for K in Ks_sorted]
            rmsd_e = [overall[(K, strat)]["rmsd_se"]   for K in Ks_sorted]
            ta_y   = [overall[(K, strat)]["ta_mean"]   for K in Ks_sorted]
            ta_e   = [overall[(K, strat)]["ta_se"]     for K in Ks_sorted]
            axes[0].errorbar(Ks_sorted, rmsd_y, yerr=rmsd_e, marker="o",
                             label=strat, color=colors.get(strat), capsize=3)
            axes[1].errorbar(Ks_sorted, ta_y, yerr=ta_e, marker="o",
                             label=strat, color=colors.get(strat), capsize=3)
        # K=1 reference line from Exp4 subset
        if exp4_K1_subset is not None:
            axes[0].axhline(exp4_K1_subset["RMSD"], color="k", ls="--", lw=0.8,
                            label=f"Exp4 K=1 subset")
            axes[1].axhline(exp4_K1_subset["TypeAcc"], color="k", ls="--", lw=0.8,
                            label=f"Exp4 K=1 subset")
        axes[0].set_xlabel("K (samples averaged)")
        axes[0].set_ylabel("Mean RMSD (Å)  ↓ better")
        axes[0].set_title(f"RMSD vs K  (n={n_samples} val subset)")
        axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xticks(Ks_sorted)
        axes[1].set_xlabel("K (samples averaged)")
        axes[1].set_ylabel("Mean TypeAcc  ↑ better")
        axes[1].set_title(f"TypeAcc vs K  (n={n_samples} val subset)")
        axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xticks(Ks_sorted)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"  wrote: {plot_path}")
    except Exception as e:
        logger.warning(f"  plot failed: {e}")
        plot_path = None

    # ── Markdown report ─────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, "multisample_results.md")
    tier_count = {}
    for t in data["tiers"]:
        tier_count[t] = tier_count.get(t, 0) + 1

    with open(md_path, "w") as f:
        f.write("# Exp5 SA0 — Multi-Sample Averaging Quick Win\n\n")
        f.write("## Subset\n\n")
        f.write(f"- {n_samples} val samples kept (target {data.get('n_target', n_samples)}, "
                f"dropped {data.get('n_dropped', 0)})\n")
        f.write(f"- Tier breakdown: " +
                ", ".join(f"{t}={tier_count.get(t,0)}" for t in "ABCD") + "\n")
        f.write(f"- Stratified seed = 0; K-sweep seeds = "
                f"{data['seed_base']} + k for k in 0..{K_max-1}\n")
        f.write(f"- Checkpoint: `{data['checkpoint']}`\n")
        f.write(f"- ckpt md5 (full): `{data['ckpt_md5_full']}`\n")
        f.write(f"- torch {data.get('torch_version','?')} (CUDA build {data.get('torch_cuda_build','?')}) on `{data['device']}`\n")
        f.write(f"- Wall: {sum(data['wall_per_K'])/60:.1f} min for K={K_max} "
                f"(per-K: {', '.join(f'{w:.0f}s' for w in data['wall_per_K'])})\n\n")

        f.write("## Sanity check: SA0 K=1 vs Exp4 §5.1\n\n")
        f.write(f"**Reference (Exp4 K=1 full val, n=7621):** "
                f"RMSD={EXP4_K1_FULL_VAL['RMSD']:.4f}, "
                f"TypeAcc={EXP4_K1_FULL_VAL['TypeAcc']:.4f}, "
                f"pred_in={EXP4_K1_FULL_VAL['pred_in']:.2f}\n\n")
        if exp4_K1_subset is not None:
            f.write(f"**Reference restricted to same {n_samples} subset:** "
                    f"RMSD={exp4_K1_subset['RMSD']:.4f} (SE={exp4_K1_subset['RMSD_se']:.4f}), "
                    f"TypeAcc={exp4_K1_subset['TypeAcc']:.4f} (SE={exp4_K1_subset['TypeAcc_se']:.4f}), "
                    f"pred_in={exp4_K1_subset['pred_in']:.2f} (SE={exp4_K1_subset['pred_in_se']:.2f})\n\n")
            f.write("Sanity criterion: SA0 K=1 mean ∈ Exp4 K=1 subset ±2·SE.\n\n")
            f.write("```\n")
            for ln in sanity_lines:
                f.write(ln + "\n")
            f.write("```\n\n")
            f.write(f"**Sanity overall: {'✓ PASS' if sanity_pass else '❌ FAIL — investigate before trusting K>1'}**\n\n")
        else:
            f.write("(sanity skipped — could not match all subset names in PSM csv)\n\n")

        f.write("## Main results\n\n")
        f.write("| K  | Strategy  | n   | RMSD   | ΔRMSD vs K=1 | TypeAcc | ΔTypeAcc vs K=1 | pred_in/20 |\n")
        f.write("|---:|-----------|----:|-------:|-------------:|--------:|----------------:|-----------:|\n")
        for K_eval in sorted(args.K_values):
            for strat in args.strategies:
                if (K_eval, strat) not in overall:
                    continue
                s = overall[(K_eval, strat)]
                if K_eval == 1:
                    drm, dta = 0.0, 0.0
                else:
                    drm = s["rmsd_mean"] - sa0_K1["rmsd_mean"]
                    dta = s["ta_mean"]   - sa0_K1["ta_mean"]
                lbl = "(n/a)" if K_eval == 1 else strat
                f.write(f"| {K_eval} | {lbl} | {s['n']} | {s['rmsd_mean']:.4f} | "
                        f"{drm:+.4f} | {s['ta_mean']:.4f} | {dta:+.4f} | {s['pi_mean']:.2f} |\n")
                if K_eval == 1:
                    break  # naive ≡ hungarian for K=1

        f.write("\n## Per-Tier breakdown\n\n")
        for tier in "BCD":
            n_t = tier_count.get(tier, 0)
            if n_t == 0:
                continue
            f.write(f"### Tier {tier}  (n={n_t})\n\n")
            f.write("| K  | Strategy  | RMSD   | TypeAcc | pred_in/20 |\n")
            f.write("|---:|-----------|-------:|--------:|-----------:|\n")
            for K_eval in sorted(args.K_values):
                for strat in args.strategies:
                    if (K_eval, strat) not in per_tier_agg or tier not in per_tier_agg[(K_eval, strat)]:
                        continue
                    s = per_tier_agg[(K_eval, strat)][tier]
                    lbl = "(n/a)" if K_eval == 1 else strat
                    f.write(f"| {K_eval} | {lbl} | {s['rmsd_mean']:.4f} | "
                            f"{s['ta_mean']:.4f} | {s['pi_mean']:.2f} |\n")
                    if K_eval == 1:
                        break
            f.write("\n")

        if plot_path:
            f.write(f"## Curves\n\n![K vs metrics]({os.path.basename(plot_path)})\n\n")

        f.write("## Conclusion\n\n*To be filled by Main Agent based on numbers above:*\n\n")
        f.write("- naive vs Hungarian — which wins?\n")
        f.write("- K=5 vs K=10 marginal gain — plateau or still climbing?\n")
        f.write("- ROI for Exp5: should K-averaging be standard for SA3?\n\n")
        f.write("## Open questions for Main Agent\n\n*To be filled.*\n")

    logger.info(f"  wrote: {md_path}")
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE.  sanity {'PASS' if sanity_pass else 'FAIL'}  "
                f"out_dir={args.out_dir}")
    logger.info("=" * 60)
    sys.exit(0 if sanity_pass else 2)


if __name__ == "__main__":
    main()
