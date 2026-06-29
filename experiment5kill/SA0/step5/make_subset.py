#!/usr/bin/env python
"""
make_subset.py
========================================================================
Exp5 SA0 — build stratified 500-sample subset from Exp4 val (seed=0).

Tier B/C/D matched to val proportions (Tier A skipped per handoff §3,
only 13 samples in full val, statistical noise too high).

Val Tier counts (from pre-flight): B=1961 C=3893 D=1754  (total non-A=7608)
Target 500 split:
  B: round(1961/7608 × 500) = 129
  C: round(3893/7608 × 500) = 256
  D:                     500 − 129 − 256 = 115
Sum = 500 ✓

Reads:  /home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv
Writes: /home/tcat/diffcsp_exp5/sa0/results/sa0_subset_500.csv

The output CSV also carries Exp4's K=1 metrics for each chosen sample,
so the aggregate step can compute the K=1 subset reference baseline
without re-reading the original PSM file.

Usage:
  python make_subset.py
"""
import argparse, os, csv
import numpy as np

DEFAULT_PSM = "/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv"
DEFAULT_OUT = "/home/tcat/diffcsp_exp5/sa0/results/sa0_subset_500.csv"
DEFAULT_SEED = 0
TARGET = {"B": 129, "C": 256, "D": 115}   # sums to 500


def assign_tier(ec):
    if ec < 3.0:  return "A"
    if ec < 4.0:  return "B"
    if ec < 5.0:  return "C"
    return "D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm_csv", default=DEFAULT_PSM)
    ap.add_argument("--out_csv", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    print("=" * 60)
    print("Exp5 SA0 — make_subset")
    print("=" * 60)
    print(f"  psm_csv : {args.psm_csv}")
    print(f"  out_csv : {args.out_csv}")
    print(f"  seed    : {args.seed}")
    print(f"  target  : {TARGET}  (sum={sum(TARGET.values())})")

    rows = []
    with open(args.psm_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"\n  read {len(rows)} rows from PSM csv")

    by_tier = {"A": [], "B": [], "C": [], "D": []}
    for r in rows:
        by_tier[assign_tier(float(r["eval_cutoff"]))].append(r)
    print("\n  Val tier population:")
    for t in "ABCD":
        print(f"    {t}: {len(by_tier[t])}")

    rng = np.random.default_rng(args.seed)
    selected = []
    for tier, n_target in TARGET.items():
        pool = by_tier[tier]
        if len(pool) < n_target:
            print(f"  ⚠️  pool {tier} ({len(pool)}) < target ({n_target}); taking all")
            chosen = pool
        else:
            idx = sorted(rng.choice(len(pool), size=n_target, replace=False).tolist())
            chosen = [pool[i] for i in idx]
        for r in chosen:
            r["tier"] = tier
            selected.append(r)

    print(f"\n  Selected: {len(selected)}")
    print("    breakdown:", {t: sum(1 for r in selected if r["tier"] == t) for t in "BCD"})

    selected.sort(key=lambda r: r["sample_name"])  # stable order

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_name", "mp_id", "tier", "eval_cutoff",
                    "exp4_K1_rmsd", "exp4_K1_type_acc",
                    "exp4_K1_n_pred_in", "exp4_K1_n_true_in"])
        for r in selected:
            w.writerow([r["sample_name"], r["mp_id"], r["tier"], r["eval_cutoff"],
                        r["rmsd"], r["type_acc"],
                        r["n_pred_in"], r["n_true_in"]])
    print(f"\n  written: {args.out_csv}")

    # Subset reference K=1 baseline (informational)
    rmsd_arr = np.array([float(r["rmsd"])      for r in selected])
    ta_arr   = np.array([float(r["type_acc"])  for r in selected])
    pi_arr   = np.array([float(r["n_pred_in"]) for r in selected])
    n = len(selected)
    print(f"\n  Exp4 K=1 (restricted to selected {n} samples — SA0 K=1 sanity reference):")
    print(f"    RMSD     mean={rmsd_arr.mean():.4f}  std={rmsd_arr.std(ddof=1):.4f}  "
          f"SE={rmsd_arr.std(ddof=1)/np.sqrt(n):.4f}  ±2SE band="
          f"[{rmsd_arr.mean()-2*rmsd_arr.std(ddof=1)/np.sqrt(n):.4f}, "
          f"{rmsd_arr.mean()+2*rmsd_arr.std(ddof=1)/np.sqrt(n):.4f}]")
    print(f"    TypeAcc  mean={ta_arr.mean():.4f}  std={ta_arr.std(ddof=1):.4f}  "
          f"SE={ta_arr.std(ddof=1)/np.sqrt(n):.4f}")
    print(f"    pred_in  mean={pi_arr.mean():.2f}  std={pi_arr.std(ddof=1):.2f}  "
          f"SE={pi_arr.std(ddof=1)/np.sqrt(n):.2f}")


if __name__ == "__main__":
    main()
