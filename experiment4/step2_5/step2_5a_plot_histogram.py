"""
Step 2.5 Phase A.2 — Gap distribution analysis + candidate thresholds.
======================================================================

Reads
-----
* step2_5_neighbor_distances.pkl  (from phase A.1)
* data_inventory.csv              (for split labels)

Produces
--------
* step2_5_gap_histogram.png            — global + top-5 element histograms
* step2_5_gap_stats.csv                — global gap stats
* step2_5_gap_stats_by_element.csv     — by-element gap stats
* step2_5_candidate_thresholds.csv     — 5 candidates + simulation results
* step2_5a_summary.txt                 — human-readable summary
* step2_5a_plot.log                    — execution log

Notes on spec (🔒 LOCKED)
-------------------------
* Only TRAIN samples contribute gaps (handoff §5.3: avoid info leakage)
* Only gaps where d[i] <= 6.0 Å are counted (handoff §5.3)
* 5 candidates: valley / p10-cut / p15-cut / p20-cut / empirical 0.3
* "p10" means "cut the top 10% of gaps" = 90th percentile of gap distribution
* Simulation evaluates each candidate on train: mean n_shells, shell-1 size,
  shell-1/2 outer radii, isolated-single% (shell with 1 atom), over-merged%
  (shell-1 span >0.5 Å → likely merged clusters)

Run
---
python step2_5a_plot_histogram.py
"""
import io
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
EXP4_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR     = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR    = os.path.join(EXP4_ROOT, "step2_5")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")

NEIGHBOR_PKL_PATH = os.path.join(STEP25_DIR, "step2_5_neighbor_distances.pkl")

HIST_PATH           = os.path.join(STEP25_DIR, "step2_5_gap_histogram.png")
GAP_STATS_CSV       = os.path.join(STEP25_DIR, "step2_5_gap_stats.csv")
GAP_STATS_ELEM_CSV  = os.path.join(STEP25_DIR, "step2_5_gap_stats_by_element.csv")
THRESH_CSV          = os.path.join(STEP25_DIR, "step2_5_candidate_thresholds.csv")
SUMMARY_PATH        = os.path.join(STEP25_DIR, "step2_5a_summary.txt")
LOG_PATH            = os.path.join(STEP25_DIR, "step2_5a_plot.log")

INNER_CUTOFF_GAPS   = 6.0     # 🔒 LOCKED (handoff §5.3)
EMPIRICAL_THRESHOLD = 0.30    # 🔒 LOCKED
SIM_CUTOFF          = 6.0     # for shell simulation reporting ("in_6A")
OVER_MERGED_SPAN    = 0.5     # heuristic: shell-1 span > 0.5 Å → merged


# -----------------------------------------------------------------------------
# Gap collection (train only, d<=6 Å mask)
# -----------------------------------------------------------------------------
def collect_gaps(pkl_data, inv):
    train_names = set(inv[inv["split"] == "train"]["sample_name"])

    all_gaps = []
    all_gaps_by_element = defaultdict(list)
    n_ok, n_skipped = 0, 0

    for sname in tqdm(train_names, desc="collecting gaps"):
        rec = pkl_data.get(sname)
        if rec is None or rec["status"] != "ok":
            n_skipped += 1
            continue
        d = rec["distances"]
        if len(d) < 2:
            n_skipped += 1
            continue
        gaps = d[1:] - d[:-1]
        mask = d[:-1] <= INNER_CUTOFF_GAPS
        masked = gaps[mask]
        all_gaps.extend(masked.tolist())
        all_gaps_by_element[rec["center_element"]].extend(masked.tolist())
        n_ok += 1

    return np.array(all_gaps, dtype=np.float32), all_gaps_by_element, n_ok, n_skipped


# -----------------------------------------------------------------------------
# Stats & valley detection
# -----------------------------------------------------------------------------
def compute_gap_stats(gaps):
    return {
        "n_gaps": int(len(gaps)),
        "mean":   float(np.mean(gaps)),
        "median": float(np.median(gaps)),
        "std":    float(np.std(gaps)),
        "min":    float(np.min(gaps)),
        "max":    float(np.max(gaps)),
        "p25":    float(np.percentile(gaps, 25)),
        "p50":    float(np.percentile(gaps, 50)),
        "p75":    float(np.percentile(gaps, 75)),
        "p80":    float(np.percentile(gaps, 80)),
        "p85":    float(np.percentile(gaps, 85)),
        "p90":    float(np.percentile(gaps, 90)),
        "p95":    float(np.percentile(gaps, 95)),
        "p99":    float(np.percentile(gaps, 99)),
    }


def find_valley(gaps, search_lo=0.15, search_hi=0.50,
                bin_width=0.02, smooth_win=5):
    """Smoothed-histogram local minimum in [search_lo, search_hi]."""
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    counts, edges = np.histogram(gaps, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.ones(smooth_win) / smooth_win
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")

    in_range = np.where((centers >= search_lo) & (centers <= search_hi))[0]
    if len(in_range) < 3:
        return None, counts, smooth, centers

    minima = []
    for i in in_range[1:-1]:
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
            minima.append((float(centers[i]), float(smooth[i])))
    if not minima:
        return None, counts, smooth, centers
    # deepest valley
    valley = min(minima, key=lambda x: x[1])[0]
    return valley, counts, smooth, centers


# -----------------------------------------------------------------------------
# Shell-cutting + effect simulation
# -----------------------------------------------------------------------------
def compute_shells(distances, threshold):
    """Return list of (start_idx, end_idx) for each shell (both inclusive)."""
    n = len(distances)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    gaps = distances[1:] - distances[:-1]
    breaks = np.where(gaps > threshold)[0]  # shell ends at break index i
    out, start = [], 0
    for b in breaks:
        out.append((start, int(b)))
        start = int(b) + 1
    out.append((start, n - 1))
    return out


def simulate_threshold(pkl_data, train_names, threshold):
    """
    Simulate shell cutting on train and aggregate metrics.
    Metrics mirror handoff §5.5.
    """
    n_shells_in_cutoff_list = []
    shell1_n_atoms = []
    shell1_outer   = []
    shell2_outer   = []
    isolated_single_count = 0
    total_shells_in_cutoff = 0
    over_merged_count = 0
    n_samples = 0

    for sname in train_names:
        rec = pkl_data.get(sname)
        if rec is None or rec["status"] != "ok":
            continue
        d = rec["distances"]
        if len(d) < 2:
            continue
        n_samples += 1

        shells = compute_shells(d, threshold)

        # shells "in 6 Å" = those whose FIRST atom is at d <= SIM_CUTOFF
        n_in_cutoff = 0
        for (s, e) in shells:
            if d[s] <= SIM_CUTOFF:
                n_in_cutoff += 1
                total_shells_in_cutoff += 1
                if e == s:   # single-atom shell
                    isolated_single_count += 1
        n_shells_in_cutoff_list.append(n_in_cutoff)

        # shell 1
        s1_s, s1_e = shells[0]
        s1_n    = s1_e - s1_s + 1
        s1_span = float(d[s1_e] - d[s1_s])
        shell1_n_atoms.append(s1_n)
        shell1_outer.append(float(d[s1_e]))
        if s1_span > OVER_MERGED_SPAN:
            over_merged_count += 1

        # shell 2
        if len(shells) >= 2:
            _, s2_e = shells[1]
            shell2_outer.append(float(d[s2_e]))

    if n_samples == 0:
        return None

    return {
        "threshold":                 round(float(threshold), 4),
        "n_samples":                 n_samples,
        "mean_n_shells_in_6A":       float(np.mean(n_shells_in_cutoff_list)),
        "median_n_shells_in_6A":     float(np.median(n_shells_in_cutoff_list)),
        "mean_shell1_n_atoms":       float(np.mean(shell1_n_atoms)),
        "median_shell1_n_atoms":     float(np.median(shell1_n_atoms)),
        "mean_shell1_outer":         float(np.mean(shell1_outer)),
        "mean_shell2_outer":         float(np.mean(shell2_outer)) if shell2_outer else None,
        "isolated_single_pct":       (100.0 * isolated_single_count / total_shells_in_cutoff
                                       if total_shells_in_cutoff else 0.0),
        "over_merged_pct":           100.0 * over_merged_count / n_samples,
    }


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_histogram(gaps, candidates, gaps_by_element, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    bins = np.arange(0, 1.5, 0.02)

    # ---- Top: global ----
    ax = axes[0]
    ax.hist(gaps, bins=bins, color="steelblue", alpha=0.75,
            edgecolor="black", linewidth=0.3)
    color_map = {
        "valley":          "red",
        "p10 (p90 cut)":   "orange",
        "p15 (p85 cut)":   "green",
        "p20 (p80 cut)":   "purple",
        "empirical 0.30":  "black",
    }
    for name, val in candidates.items():
        if val is None:
            continue
        ax.axvline(val, color=color_map.get(name, "gray"),
                   linestyle="--", linewidth=1.8,
                   label=f"{name} = {val:.3f}")
    ax.set_xlabel("Adjacent distance gap (Å)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Global adjacent-distance gap distribution "
        f"(train only, n_gaps = {len(gaps):,}, d ≤ 6 Å)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1.5)
    ax.grid(alpha=0.3)

    # ---- Bottom: top-5 elements ----
    ax2 = axes[1]
    top5 = sorted(gaps_by_element.items(), key=lambda kv: -len(kv[1]))[:5]
    cmap = plt.get_cmap("tab10")
    for i, (elem, g_list) in enumerate(top5):
        g = np.asarray(g_list, dtype=np.float32)
        ax2.hist(g, bins=bins, alpha=0.40,
                 label=f"{elem} (n={len(g):,})", color=cmap(i))
    ax2.set_xlabel("Adjacent distance gap (Å)")
    ax2.set_ylabel("Count")
    ax2.set_title("Gap distribution by top-5 center elements")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(0, 1.5)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Step 2.5 Phase A.2 — gap analysis + candidate thresholds")

    print(f"\nLoading pickle: {NEIGHBOR_PKL_PATH}")
    with open(NEIGHBOR_PKL_PATH, "rb") as f:
        pkl = pickle.load(f)
    print(f"  entries: {len(pkl):,}")

    inv = pd.read_csv(INVENTORY_CSV)

    # ---- Collect gaps ----
    print(f"\nCollecting gaps (train only, d <= {INNER_CUTOFF_GAPS} Å) ...")
    gaps, gaps_by_elem, n_ok, n_skip = collect_gaps(pkl, inv)
    print(f"  train ok: {n_ok:,}, skipped: {n_skip}")
    print(f"  total gaps: {len(gaps):,}")
    assert len(gaps) > 0, "No gaps collected — abort"

    # ---- Global stats ----
    stats = compute_gap_stats(gaps)
    print(f"\nGlobal gap stats:")
    for k, v in stats.items():
        print(f"  {k:<8s}: {v:.5f}" if isinstance(v, float) else f"  {k:<8s}: {v:,}")
    pd.DataFrame([stats]).to_csv(GAP_STATS_CSV, index=False)
    print(f"  saved {GAP_STATS_CSV}")

    # ---- Per-element stats ----
    elem_rows = []
    for elem, g_list in gaps_by_elem.items():
        g = np.asarray(g_list, dtype=np.float32)
        if len(g) == 0:
            continue
        elem_rows.append({
            "center_element": elem,
            "n_gaps": int(len(g)),
            "mean":   float(np.mean(g)),
            "median": float(np.median(g)),
            "p80":    float(np.percentile(g, 80)),
            "p85":    float(np.percentile(g, 85)),
            "p90":    float(np.percentile(g, 90)),
            "p95":    float(np.percentile(g, 95)),
        })
    elem_df = (pd.DataFrame(elem_rows)
                 .sort_values("n_gaps", ascending=False)
                 .reset_index(drop=True))
    elem_df.to_csv(GAP_STATS_ELEM_CSV, index=False)
    print(f"  saved {GAP_STATS_ELEM_CSV}")
    print(f"\nTop-10 elements by gap count:")
    print(elem_df.head(10).to_string(index=False))

    # ---- Candidates ----
    print(f"\nComputing candidate thresholds ...")
    valley, _hist_counts, _smooth, _centers = find_valley(gaps)
    p90 = float(np.percentile(gaps, 90))   # p10 candidate (cut top 10%)
    p85 = float(np.percentile(gaps, 85))   # p15 candidate
    p80 = float(np.percentile(gaps, 80))   # p20 candidate
    print(f"  valley            : {'NOT FOUND' if valley is None else f'{valley:.4f} Å'}")
    print(f"  p10 (p90-gap cut) : {p90:.4f} Å")
    print(f"  p15 (p85-gap cut) : {p85:.4f} Å")
    print(f"  p20 (p80-gap cut) : {p80:.4f} Å")
    print(f"  empirical         : {EMPIRICAL_THRESHOLD:.4f} Å")

    candidates = {
        "valley":         valley,
        "p10 (p90 cut)":  p90,
        "p15 (p85 cut)":  p85,
        "p20 (p80 cut)":  p80,
        "empirical 0.30": EMPIRICAL_THRESHOLD,
    }

    # ---- Simulate each candidate ----
    print(f"\nSimulating shell-cutting for each candidate (on train) ...")
    train_names = set(inv[inv["split"] == "train"]["sample_name"])
    sim_rows = []
    for name, val in candidates.items():
        if val is None:
            print(f"  {name:<20s} : SKIP (no valley found)")
            continue
        sim = simulate_threshold(pkl, train_names, val)
        sim["candidate"] = name
        sim_rows.append(sim)
        print(f"  {name:<20s} = {val:.4f} : "
              f"n_shells_6A={sim['mean_n_shells_in_6A']:.2f}, "
              f"shell1_n={sim['mean_shell1_n_atoms']:.1f}, "
              f"shell1_outer={sim['mean_shell1_outer']:.2f}Å, "
              f"shell2_outer={sim['mean_shell2_outer']:.2f}Å, "
              f"iso_single={sim['isolated_single_pct']:.1f}%, "
              f"merged={sim['over_merged_pct']:.1f}%")

    sim_df_cols = ["candidate", "threshold", "n_samples",
                   "mean_n_shells_in_6A", "median_n_shells_in_6A",
                   "mean_shell1_n_atoms", "median_shell1_n_atoms",
                   "mean_shell1_outer", "mean_shell2_outer",
                   "isolated_single_pct", "over_merged_pct"]
    sim_df = pd.DataFrame(sim_rows)[sim_df_cols]
    sim_df.to_csv(THRESH_CSV, index=False)
    print(f"  saved {THRESH_CSV}")

    # ---- Histogram ----
    print(f"\nGenerating histogram ...")
    plot_histogram(gaps, candidates, gaps_by_elem, HIST_PATH)
    print(f"  saved {HIST_PATH}")

    # ---- Summary ----
    print(f"\nGenerating summary ...")
    ok_records = [r for r in pkl.values() if r["status"] == "ok"]
    status_counts = Counter(r["status"] for r in pkl.values())
    ncs_counts = Counter(r["n_center_sites"] for r in ok_records)

    out = []
    out.append("=" * 64)
    out.append("Step 2.5 Phase A — Summary")
    out.append("=" * 64)
    out.append(f"Total samples (pickle):  {len(pkl):,}")
    out.append(f"Status 'ok':             {status_counts['ok']:,}")
    out.append(f"Train samples used:      {n_ok:,}")
    out.append(f"Total gaps (d ≤ 6 Å):    {len(gaps):,}")
    out.append("")
    out.append("Status distribution:")
    for s, c in status_counts.most_common():
        out.append(f"  {s:<30s}  {c:>7,}")
    out.append("")
    out.append("n_center_sites distribution (ok only):")
    multi_cnt = 0
    for k in sorted(ncs_counts.keys()):
        v = ncs_counts[k]
        pct = 100 * v / len(ok_records)
        out.append(f"  {k:>3d} sites:  {v:>7,}  ({pct:5.2f}%)")
        if k >= 2:
            multi_cnt += v
    multi_pct = 100 * multi_cnt / len(ok_records)
    flag = "   ⚠ FLAG (>20%)" if multi_pct > 20 else ""
    out.append(f"  >=2 sites:  {multi_cnt:,}  ({multi_pct:.2f}%){flag}")
    out.append("")
    out.append("Candidate thresholds (Å):")
    for name, v in candidates.items():
        out.append(f"  {name:<20s}  {'N/A' if v is None else f'{v:.4f}'}")
    out.append("")
    out.append("Simulation results (see candidate_thresholds.csv for full table):")
    for _, r in sim_df.iterrows():
        out.append(f"  {r['candidate']:<20s}  "
                   f"shells_6A={r['mean_n_shells_in_6A']:.2f}  "
                   f"shell1_n={r['mean_shell1_n_atoms']:.2f}  "
                   f"shell1_outer={r['mean_shell1_outer']:.2f}Å  "
                   f"iso_single%={r['isolated_single_pct']:.2f}  "
                   f"merged%={r['over_merged_pct']:.2f}")
    out.append("")
    out.append("Global gap stats:")
    for k, v in stats.items():
        out.append(f"  {k:<8s}: {v:.5f}" if isinstance(v, float)
                   else f"  {k:<8s}: {v:,}")

    summary_text = "\n".join(out)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  saved {SUMMARY_PATH}")

    print(f"\nTotal wall-clock: {time.time() - t0:.1f}s")


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
