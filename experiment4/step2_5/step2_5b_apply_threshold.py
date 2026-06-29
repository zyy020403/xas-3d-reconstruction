"""
Step 2.5 Phase B — Apply threshold = 0.1563 Å and produce shell_boundaries.pkl
==============================================================================

Per MA decision: threshold = p10 = 0.1563 Å.

Schema of shell_boundaries.pkl (handoff §5.7):
  dict[sample_name] -> {
      "threshold":      float (0.1563 for all),
      "distances":      np.ndarray (N,)  float32,
      "species_Z":      np.ndarray (N,)  int8,
      "shell_starts":   np.ndarray (S,)  float32,
      "shell_ends":     np.ndarray (S,)  float32,
      "shell_n_atoms":  np.ndarray (S,)  int32,
      "shell_of_atom":  np.ndarray (N,)  int32,
      "eval_cutoff":    float,   # shell_ends[shell_of_atom[19]] if N>=20 else shell_ends[-1]
      "n_center_sites": int,
  }

eval_cutoff rule: take the outer edge of the shell that CONTAINS the 20th
nearest neighbor — not min(d20, 4.0). Preserves full-shell semantics.

Run
---
python step2_5b_apply_threshold.py
  (or: python step2_5b_apply_threshold.py --threshold 0.1563  if overriding)
"""
import argparse
import io
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EXP4_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR     = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR    = os.path.join(EXP4_ROOT, "step2_5")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")

NEIGHBOR_PKL_PATH    = os.path.join(STEP25_DIR, "step2_5_neighbor_distances.pkl")
SHELL_BOUND_PATH     = os.path.join(STEP25_DIR, "shell_boundaries.pkl")
STATS_SPLIT_PATH     = os.path.join(STEP25_DIR, "shell_stats_by_split.csv")
STATS_ELEMENT_PATH   = os.path.join(STEP25_DIR, "shell_stats_by_element.csv")
SUMMARY_PATH         = os.path.join(STEP25_DIR, "step2_5b_summary.txt")
LOG_PATH             = os.path.join(STEP25_DIR, "step2_5b_apply.log")


# -----------------------------------------------------------------------------
# Core: gap-based shell cutting
# -----------------------------------------------------------------------------
def compute_shells(distances, threshold):
    """
    Split sorted distances into shells where adjacent gap > threshold.
    Returns list of (start_idx, end_idx) inclusive.
    """
    n = len(distances)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    gaps = distances[1:] - distances[:-1]
    breaks = np.where(gaps > threshold)[0]
    out, start = [], 0
    for b in breaks:
        out.append((start, int(b)))
        start = int(b) + 1
    out.append((start, n - 1))
    return out


def build_shell_record(rec, threshold):
    """Return the §5.7-schema dict for one sample, or None if skipped."""
    if rec["status"] != "ok":
        return None

    d = rec["distances"]
    Z = rec["species_Z"]
    if d is None or len(d) == 0:
        return None

    shells = compute_shells(d, threshold)
    n_shells = len(shells)

    shell_starts  = np.array([d[s]       for (s, _) in shells], dtype=np.float32)
    shell_ends    = np.array([d[e]       for (_, e) in shells], dtype=np.float32)
    shell_n_atoms = np.array([e - s + 1  for (s, e) in shells], dtype=np.int32)

    shell_of_atom = np.empty(len(d), dtype=np.int32)
    for idx, (s, e) in enumerate(shells):
        shell_of_atom[s:e + 1] = idx

    # eval_cutoff = outer edge of the shell containing the 20th nearest neighbor
    if len(d) >= 20:
        eval_cutoff = float(shell_ends[shell_of_atom[19]])
    else:
        eval_cutoff = float(shell_ends[-1])

    return {
        "threshold":       float(threshold),
        "distances":       d,
        "species_Z":       Z,
        "shell_starts":    shell_starts,
        "shell_ends":      shell_ends,
        "shell_n_atoms":   shell_n_atoms,
        "shell_of_atom":   shell_of_atom,
        "eval_cutoff":     eval_cutoff,
        "n_center_sites":  int(rec["n_center_sites"]),
    }


# -----------------------------------------------------------------------------
# Aggregation helpers for summary CSVs
# -----------------------------------------------------------------------------
def aggregate_records(recs_iterable):
    """
    Takes an iterable of (sample_name, record) pairs and returns aggregate stats.
    """
    ev = []
    n1 = []
    s1_out = []
    s2_out = []
    n_shells = []
    n_samples = 0
    for _, rec in recs_iterable:
        if rec is None:
            continue
        n_samples += 1
        ev.append(rec["eval_cutoff"])
        n1.append(int(rec["shell_n_atoms"][0]))
        s1_out.append(float(rec["shell_ends"][0]))
        if len(rec["shell_ends"]) >= 2:
            s2_out.append(float(rec["shell_ends"][1]))
        n_shells.append(len(rec["shell_ends"]))

    if n_samples == 0:
        return None

    return {
        "n_samples":         n_samples,
        "mean_eval_cutoff":  float(np.mean(ev)),
        "median_eval_cutoff": float(np.median(ev)),
        "p5_eval_cutoff":    float(np.percentile(ev, 5)),
        "p95_eval_cutoff":   float(np.percentile(ev, 95)),
        "mean_shell1_n":     float(np.mean(n1)),
        "mean_shell1_outer": float(np.mean(s1_out)),
        "mean_shell2_outer": float(np.mean(s2_out)) if s2_out else None,
        "mean_n_shells":     float(np.mean(n_shells)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.1563,
                        help="Shell-cutting threshold in Å (MA default 0.1563 = p10)")
    args = parser.parse_args()
    threshold = args.threshold

    t0 = time.time()
    print(f"Step 2.5 Phase B — shell boundary generation")
    print(f"  threshold: {threshold:.4f} Å")

    # ---- Load inputs ----
    print(f"  loading neighbors: {NEIGHBOR_PKL_PATH}")
    with open(NEIGHBOR_PKL_PATH, "rb") as f:
        neighbors = pickle.load(f)
    print(f"    entries: {len(neighbors):,}")

    inv = pd.read_csv(INVENTORY_CSV)
    sample_to_split = dict(zip(inv["sample_name"], inv["split"]))
    sample_to_element = dict(zip(inv["sample_name"], inv["center_element"]))

    # ---- Build shell records ----
    print(f"  computing shell boundaries ...")
    shell_boundaries = {}
    n_skipped = 0
    for sname, rec in tqdm(neighbors.items(), total=len(neighbors)):
        out = build_shell_record(rec, threshold)
        if out is None:
            n_skipped += 1
            continue
        shell_boundaries[sname] = out

    print(f"    built:   {len(shell_boundaries):,}")
    print(f"    skipped: {n_skipped:,}")

    # ---- Sanity ----
    sample = next(iter(shell_boundaries.values()))
    print(f"\n  sanity check (first record):")
    print(f"    threshold:     {sample['threshold']}")
    print(f"    distances:     shape={sample['distances'].shape}, dtype={sample['distances'].dtype}")
    print(f"    species_Z:     shape={sample['species_Z'].shape}, dtype={sample['species_Z'].dtype}")
    print(f"    shell_starts:  {sample['shell_starts'][:5].tolist()}")
    print(f"    shell_ends:    {sample['shell_ends'][:5].tolist()}")
    print(f"    shell_n_atoms: {sample['shell_n_atoms'][:5].tolist()}")
    print(f"    eval_cutoff:   {sample['eval_cutoff']:.4f}")
    print(f"    n_center_sites: {sample['n_center_sites']}")

    # Invariant checks on a sample of 100
    rng = np.random.default_rng(0)
    keys = list(shell_boundaries.keys())
    for k in rng.choice(keys, size=100, replace=False):
        r = shell_boundaries[k]
        assert len(r["distances"]) == len(r["shell_of_atom"])
        assert len(r["shell_starts"]) == len(r["shell_ends"]) == len(r["shell_n_atoms"])
        assert int(r["shell_n_atoms"].sum()) == len(r["distances"])
        assert np.all(r["shell_ends"] >= r["shell_starts"])
        assert r["eval_cutoff"] >= r["shell_starts"][0]
    print(f"    ✓ invariants hold on 100 random samples")

    # ---- Serialize ----
    print(f"\n  saving pickle ...")
    with open(SHELL_BOUND_PATH, "wb") as f:
        pickle.dump(shell_boundaries, f, protocol=4)
    size_mb = os.path.getsize(SHELL_BOUND_PATH) / 1024 / 1024
    print(f"    {SHELL_BOUND_PATH}  ({size_mb:.1f} MB)")

    # ---- Aggregate by split ----
    print(f"\n  aggregating stats by split ...")
    by_split = defaultdict(list)
    for sname, rec in shell_boundaries.items():
        split = sample_to_split.get(sname, "unknown")
        by_split[split].append((sname, rec))
    split_rows = []
    for split in ["train", "val", "test", "holdout"]:
        agg = aggregate_records(by_split[split])
        if agg is None:
            continue
        agg["split"] = split
        split_rows.append(agg)
    split_df = pd.DataFrame(split_rows)
    cols = ["split"] + [c for c in split_df.columns if c != "split"]
    split_df = split_df[cols]
    split_df.to_csv(STATS_SPLIT_PATH, index=False)
    print(f"    {STATS_SPLIT_PATH}")
    print(split_df.to_string(index=False))

    # ---- Aggregate by element ----
    print(f"\n  aggregating stats by element ...")
    by_elem = defaultdict(list)
    for sname, rec in shell_boundaries.items():
        elem = sample_to_element.get(sname, "unknown")
        by_elem[elem].append((sname, rec))
    elem_rows = []
    for elem, recs in by_elem.items():
        agg = aggregate_records(recs)
        if agg is None:
            continue
        agg["center_element"] = elem
        elem_rows.append(agg)
    elem_df = (pd.DataFrame(elem_rows)
                 .sort_values("n_samples", ascending=False)
                 .reset_index(drop=True))
    cols = ["center_element"] + [c for c in elem_df.columns if c != "center_element"]
    elem_df = elem_df[cols]
    elem_df.to_csv(STATS_ELEMENT_PATH, index=False)
    print(f"    {STATS_ELEMENT_PATH}  ({len(elem_df)} rows)")
    print(f"  top-10 by sample count:")
    print(elem_df.head(10).to_string(index=False))

    # ---- Summary ----
    out = []
    out.append("=" * 64)
    out.append("Step 2.5 Phase B — Summary")
    out.append("=" * 64)
    out.append(f"Threshold:                {threshold:.4f} Å (p10, MA-selected)")
    out.append(f"Records built:            {len(shell_boundaries):,}")
    out.append(f"Skipped:                  {n_skipped:,}")
    out.append(f"pickle size:              {size_mb:.1f} MB")
    out.append(f"wall-clock:               {time.time() - t0:.1f}s")
    out.append("")
    out.append("By split:")
    out.append(split_df.to_string(index=False))
    out.append("")
    out.append("By element (top-10):")
    out.append(elem_df.head(10).to_string(index=False))
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n  wrote {SUMMARY_PATH}")
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
