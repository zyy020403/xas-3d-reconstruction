"""
Step 2.5 Phase C — Multi-site equivalence diagnostic
====================================================

Question: in multi-site samples (n_center_sites >= 2), do all equivalent
center sites have the SAME shell-1 structure? If yes, "pick first site" is
cheap. If no, Step 3 Dataset needs a smarter strategy.

Sampling (per MA):
  * 5 samples each at n_center_sites ∈ {2, 4, 8, 16}  → 20 samples total
  * train split only
  * random_state = 42
  * bail out with warning if any stratum has fewer than 5 candidates

Per sample:
  * re-load POSCAR → primitive
  * find all indices of center_element
  * for each index i:
      - get_neighbors(prim[i], r=4.0)
      - sort by distance
      - cut shell-1 using threshold = 0.1563 Å (first gap > threshold breaks)
      - build signature_i = tuple(sorted([(Z, round(d, 2))]))
      - record shell1 distances (sorted)
  * compare signatures:
      - shell1_all_equal: bool
      - n_unique_shell1_signatures: int
      - max_shell1_distance_MAE: max pairwise MAE across all pairs (i,j)
          * if shell1 length differs between i and j -> SENTINEL 999.0
          * otherwise mean |d_i - d_j| for sorted arrays

Also attempt (optional, best-effort) MP-API metadata lookup for FEFF protocol.

Outputs
-------
* step2_5c_multisite_diagnostic.csv  (22 rows: 20 samples + header)
* step2_5c_summary.txt                (human-readable answer to Q1, Q2)
* step2_5c_diagnose.log

Run
---
python step2_5c_diagnose_multisite.py
"""
import io
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EXP4_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR     = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR    = os.path.join(EXP4_ROOT, "step2_5")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")

NEIGHBOR_PKL_PATH = os.path.join(STEP25_DIR, "step2_5_neighbor_distances.pkl")

CSV_PATH     = os.path.join(STEP25_DIR, "step2_5c_multisite_diagnostic.csv")
SUMMARY_PATH = os.path.join(STEP25_DIR, "step2_5c_summary.txt")
LOG_PATH     = os.path.join(STEP25_DIR, "step2_5c_diagnose.log")

THRESHOLD   = 0.1563   # p10, matches Phase B
TARGET_NCS  = [2, 4, 8, 16]
N_PER_BUCKET = 5
RANDOM_STATE = 42
MAE_SENTINEL_INCOMPATIBLE = 999.0


# -----------------------------------------------------------------------------
# Shell-1 extraction for one site
# -----------------------------------------------------------------------------
def cut_shell1(sorted_distances, threshold):
    """Return end index (inclusive) of shell 1 given sorted distances."""
    if len(sorted_distances) == 0:
        return -1
    if len(sorted_distances) == 1:
        return 0
    gaps = sorted_distances[1:] - sorted_distances[:-1]
    brks = np.where(gaps > threshold)[0]
    if len(brks) == 0:
        return len(sorted_distances) - 1
    return int(brks[0])


def shell1_for_site(prim, center_idx, r_cutoff=4.0, threshold=THRESHOLD):
    """Return (distances_sorted, Z_sorted) for shell-1 neighbors of center_idx."""
    nbrs = prim.get_neighbors(prim[center_idx], r=r_cutoff)
    if len(nbrs) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int8)
    pairs = sorted([(nbr.nn_distance, nbr.specie.Z) for nbr in nbrs], key=lambda x: x[0])
    d = np.array([p[0] for p in pairs], dtype=np.float32)
    Z = np.array([p[1] for p in pairs], dtype=np.int8)
    end = cut_shell1(d, threshold)
    return d[:end + 1], Z[:end + 1]


def signature(d, Z):
    """Canonical shell-1 signature: sorted tuple of (Z, round(d, 2))."""
    return tuple(sorted([(int(z), round(float(x), 2)) for z, x in zip(Z, d)]))


def pairwise_max_mae(distances_per_site):
    """Max pairwise mean |dᵢ - dⱼ| over all site pairs. SENTINEL on length mismatch."""
    max_mae = 0.0
    incompatible_hit = False
    n = len(distances_per_site)
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = distances_per_site[i], distances_per_site[j]
            if len(di) != len(dj):
                incompatible_hit = True
                continue
            if len(di) == 0:
                continue
            mae = float(np.mean(np.abs(di - dj)))
            if mae > max_mae:
                max_mae = mae
    return (MAE_SENTINEL_INCOMPATIBLE if incompatible_hit else max_mae), incompatible_hit


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_rows(inv, neighbors):
    """Return list of dict rows. Adds n_center_sites from neighbors pickle."""
    train = inv[inv["split"] == "train"].copy()
    train["n_center_sites"] = train["sample_name"].map(
        lambda s: neighbors.get(s, {}).get("n_center_sites", 0) if neighbors.get(s) else 0
    )

    rng = np.random.default_rng(RANDOM_STATE)
    picked = []
    for target in TARGET_NCS:
        cand = train[train["n_center_sites"] == target]
        if len(cand) < N_PER_BUCKET:
            print(f"  ⚠ bucket n_center_sites=={target} has only {len(cand)} < {N_PER_BUCKET}, using all")
            picked_rows = cand
        else:
            idx = rng.choice(len(cand), size=N_PER_BUCKET, replace=False)
            picked_rows = cand.iloc[idx]
        for _, r in picked_rows.iterrows():
            picked.append(r.to_dict())
    return picked


# -----------------------------------------------------------------------------
# MP-API best-effort lookup (may fail silently — that's OK)
# -----------------------------------------------------------------------------
def mp_api_lookup_notes():
    """
    Sub-Agent cannot query mp-api without an API key.
    Emit manual-check instructions for the user.
    """
    return (
        "MP EXAFS protocol lookup — MANUAL CHECK NEEDED:\n"
        "  Sub-Agent has no mp-api credentials in this environment.\n"
        "  To resolve site-specific vs site-averaged semantics, please check one of:\n"
        "    (a) Materials Project docs:  https://docs.materialsproject.org/\n"
        "        → search for 'XAS' or 'EXAFS' workflow docs\n"
        "    (b) FEFF calculation page of a known sample, e.g.:\n"
        "        https://next-gen.materialsproject.org/materials/mp-18658\n"
        "        → XAS tab → look for 'Absorbing Atom' or 'Averaged Sites' tag\n"
        "    (c) The source paper: Mathew et al., Scientific Data 5, 180151 (2018)\n"
        "        'High-throughput computational X-ray absorption spectroscopy'\n"
        "  Key term to find: 'site-averaged' vs 'site-specific' FEFF spectra."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    t0 = time.time()
    print("Step 2.5 Phase C — multi-site diagnostic")
    print(f"  threshold: {THRESHOLD} Å, r_cutoff: 4.0 Å")
    print(f"  sampling: {N_PER_BUCKET} per bucket × {len(TARGET_NCS)} buckets = {N_PER_BUCKET * len(TARGET_NCS)} samples")
    print(f"  buckets (n_center_sites): {TARGET_NCS}")
    print(f"  random_state: {RANDOM_STATE}")

    inv = pd.read_csv(INVENTORY_CSV)
    print(f"\n  loading neighbors pickle ...")
    with open(NEIGHBOR_PKL_PATH, "rb") as f:
        neighbors = pickle.load(f)

    rows = sample_rows(inv, neighbors)
    print(f"  picked: {len(rows)} samples")

    # Sanity: confirm n_center_sites per picked
    for r in rows:
        print(f"    - {r['sample_name']:<55s}  center={r['center_element']:<4s}  "
              f"n_sites={r['n_center_sites']}  mp_id={r['mp_id']}")

    # ---- Diagnose each sample ----
    print(f"\n  diagnosing ...")
    results = []
    for r in rows:
        t_sample = time.time()
        try:
            s_super = Structure.from_file(r["poscar_path"])
            prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()
        except Exception as e:
            print(f"    ✗ {r['sample_name']}: primitive_error {type(e).__name__}")
            results.append({
                "sample_name": r["sample_name"],
                "mp_id": r["mp_id"],
                "center_element": r["center_element"],
                "n_center_sites_prev": r["n_center_sites"],
                "n_center_sites_redo": 0,
                "shell1_all_equal": None,
                "n_unique_shell1_signatures": None,
                "max_shell1_distance_MAE": None,
                "any_incompatible_length": None,
                "shell1_sizes": None,
                "error": f"primitive_error:{type(e).__name__}",
            })
            continue

        center_indices = [i for i, site in enumerate(prim)
                          if site.specie.symbol == r["center_element"]]
        n_redo = len(center_indices)

        sigs = []
        d_per_site = []
        shell1_sizes = []
        for ci in center_indices:
            d, Z = shell1_for_site(prim, ci)
            if len(d) == 0:
                sigs.append(("empty",))
                d_per_site.append(np.array([], dtype=np.float32))
                shell1_sizes.append(0)
                continue
            sigs.append(signature(d, Z))
            d_per_site.append(d)
            shell1_sizes.append(len(d))

        n_unique_sigs = len(set(sigs))
        all_equal = (n_unique_sigs == 1)
        max_mae, incompat = pairwise_max_mae(d_per_site)

        print(f"    {r['sample_name']:<55s}  "
              f"sites={n_redo:>3d}  unique_sigs={n_unique_sigs:>2d}  "
              f"MAE={max_mae:.3f}  {'INCOMPAT' if incompat else '        '}  "
              f"shell1_sizes={shell1_sizes}  "
              f"({time.time() - t_sample:.1f}s)")

        results.append({
            "sample_name": r["sample_name"],
            "mp_id": r["mp_id"],
            "center_element": r["center_element"],
            "n_center_sites_prev": r["n_center_sites"],
            "n_center_sites_redo": n_redo,
            "shell1_all_equal": all_equal,
            "n_unique_shell1_signatures": n_unique_sigs,
            "max_shell1_distance_MAE": round(float(max_mae), 4),
            "any_incompatible_length": bool(incompat),
            "shell1_sizes": ",".join(map(str, shell1_sizes)),
            "error": "",
        })

    # ---- Save CSV ----
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  saved {CSV_PATH}")
    print(df.to_string(index=False))

    # ---- Answer Q1, Q2 ----
    valid = df[df["error"] == ""]
    n_valid = len(valid)
    n_all_equal = int(valid["shell1_all_equal"].sum()) if n_valid else 0
    pct_all_equal = 100.0 * n_all_equal / n_valid if n_valid else 0.0

    mae_vals = valid[valid["max_shell1_distance_MAE"] < MAE_SENTINEL_INCOMPATIBLE]["max_shell1_distance_MAE"].values
    mae_stats_str = "N/A"
    if len(mae_vals) > 0:
        mae_stats_str = (f"n={len(mae_vals)}  "
                         f"median={np.median(mae_vals):.4f}  "
                         f"p90={np.percentile(mae_vals, 90):.4f}  "
                         f"max={np.max(mae_vals):.4f}")
    n_incompat = int(valid["any_incompatible_length"].sum()) if n_valid else 0

    out = []
    out.append("=" * 64)
    out.append("Step 2.5 Phase C — Multi-site diagnostic summary")
    out.append("=" * 64)
    out.append(f"Samples processed:  {n_valid} / {len(df)} (errors: {len(df) - n_valid})")
    out.append(f"Threshold used:     {THRESHOLD} Å  (matches Phase B)")
    out.append("")
    out.append("Q1. 所有等价位点 shell-1 完全一致的样本比例:")
    out.append(f"   {n_all_equal} / {n_valid} = {pct_all_equal:.1f}%")
    out.append("")
    out.append("Q2. max_shell1_distance_MAE 分布 (excluding length-incompat):")
    out.append(f"   {mae_stats_str}")
    out.append(f"   samples with length-incompat between sites: {n_incompat}/{n_valid}")
    out.append("")
    out.append("Decision guide (per MA):")
    out.append(f"   {pct_all_equal:.1f}% ≥ 70% → Option A (keep 'first site'), tag site_equivalence")
    out.append(f"   {pct_all_equal:.1f}% <  70% → Option B variant: Step 3 Dataset randomly picks one equivalent site per __getitem__")
    out.append("")
    out.append("Per-sample results:")
    out.append(df.to_string(index=False))
    out.append("")
    out.append(mp_api_lookup_notes())

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n  saved {SUMMARY_PATH}")

    print(f"\n=== Q1: {pct_all_equal:.1f}% of samples have fully-equivalent shell-1 across all sites ===")
    print(f"=== Q2: shell1-distance MAE: {mae_stats_str} ===")
    print(f"\n  wall-clock: {time.time() - t0:.1f}s")


class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for t in self.s: t.write(x)
    def flush(self):
        for t in self.s: t.flush()


if __name__ == "__main__":
    log_buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, log_buf)
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"\nLog: {LOG_PATH}")
