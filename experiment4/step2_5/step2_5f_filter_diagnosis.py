"""
Step 2.5 Phase F — Filter diagnosis (Option D pre-flight)
==========================================================

Reads site_equivalence_tag.csv. Computes statistics about what would
remain if we dropped:
  - tag == 'incompat'
  - tag == 'near_equivalent' AND n_unique_shell1_multisets > 1  (the 999
    "multiset mismatch" samples that MA wants reclassified to incompat)

Pure statistics. NO data writing, NO new id files, NO new pkl.

MA uses these numbers to decide whether to actually go with Option D.

Run
---
python step2_5f_filter_diagnosis.py
"""
import os
import sys
import io

import pandas as pd

EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP25_DIR = os.path.join(EXP4_ROOT, "step2_5")
TAG_CSV_PATH = os.path.join(STEP25_DIR, "site_equivalence_tag.csv")

OUT_TXT_PATH = os.path.join(STEP25_DIR, "step2_5f_filter_diagnosis.txt")
LOG_PATH     = os.path.join(STEP25_DIR, "step2_5f_filter_diagnosis.log")

# Reference numbers from Step 1 / Phase D for comparison
ORIG_TOTAL    = 128382
ORIG_MPID     = 41431
ORIG_ELEMENTS = 88
ORIG_SPLIT_COUNTS = {"train": 102660, "val": 12912, "test": 7696, "holdout": 5114}
ORIG_SPLIT_MPIDS  = {"train": 33147,  "val": 4142,   "test": 2485,  "holdout": 1657}
TARGET_SPLIT_PCT  = {"train": 80.0,   "val": 10.0,   "test": 6.0,   "holdout": 4.0}


def fmt_pct(num, denom, places=2):
    if denom == 0:
        return "  N/A"
    return f"{100.0 * num / denom:>5.{places}f}%"


def main():
    print("=" * 78)
    print("Step 2.5 Phase F — Filter diagnosis (Option D pre-flight)")
    print("=" * 78)

    df = pd.read_csv(TAG_CSV_PATH)
    print(f"\n  loaded site_equivalence_tag.csv: shape={df.shape}")
    assert df.shape[0] == ORIG_TOTAL, f"row count mismatch: {df.shape[0]}"

    # Build drop mask
    is_incompat = df["tag"] == "incompat"
    is_neareq_mismatch = (df["tag"] == "near_equivalent") & (df["n_unique_shell1_multisets"] > 1)
    drop_mask = is_incompat | is_neareq_mismatch

    df_keep = df[~drop_mask].copy()
    df_drop = df[drop_mask].copy()

    n_drop_incompat = int(is_incompat.sum())
    n_drop_neareq   = int(is_neareq_mismatch.sum())
    n_drop_total    = int(drop_mask.sum())
    n_keep_total    = int((~drop_mask).sum())

    # ---- (a) totals --------------------------------------------------------
    print()
    print("─" * 78)
    print("(a) TOTAL SAMPLE COUNTS")
    print("─" * 78)
    print(f"  Original:                  {ORIG_TOTAL:>7,}")
    print(f"  Drop (tag=incompat):       {n_drop_incompat:>7,}  ({fmt_pct(n_drop_incompat, ORIG_TOTAL)})")
    print(f"  Drop (near_eq mismatch):   {n_drop_neareq:>7,}  ({fmt_pct(n_drop_neareq, ORIG_TOTAL)})")
    print(f"  Drop total:                {n_drop_total:>7,}  ({fmt_pct(n_drop_total, ORIG_TOTAL)})")
    print(f"  KEEP:                      {n_keep_total:>7,}  ({fmt_pct(n_keep_total, ORIG_TOTAL)})")
    print(f"  MA's prediction was 75,637 — match? "
          f"{'✓' if n_keep_total == 75637 else f'✗ (got {n_keep_total})'}")

    # ---- (b) center elements ----------------------------------------------
    print()
    print("─" * 78)
    print("(b) UNIQUE CENTER_ELEMENT")
    print("─" * 78)
    elems_orig = set(df["center_element"].unique())
    elems_keep = set(df_keep["center_element"].unique())
    elems_lost = sorted(elems_orig - elems_keep)
    print(f"  Original unique elements:  {len(elems_orig):>3d}  (Step 1 reported 88)")
    print(f"  After-drop unique:         {len(elems_keep):>3d}")
    print(f"  Completely lost elements:  {len(elems_lost)}")
    if elems_lost:
        print(f"    {elems_lost}")

    # ---- (c) split distribution -------------------------------------------
    print()
    print("─" * 78)
    print("(c) SPLIT DISTRIBUTION")
    print("─" * 78)
    print(f"  {'split':<10s}{'orig':>10s}{'keep':>10s}{'orig_pct':>11s}{'keep_pct':>11s}{'target':>10s}")
    print(f"  {'-'*10}{'-'*10}{'-'*10}{'-'*11}{'-'*11}{'-'*10}")
    keep_split_counts = df_keep["split"].value_counts().to_dict()
    for split in ["train", "val", "test", "holdout"]:
        o = ORIG_SPLIT_COUNTS[split]
        k = int(keep_split_counts.get(split, 0))
        opct = 100.0 * o / ORIG_TOTAL
        kpct = 100.0 * k / n_keep_total if n_keep_total else 0
        tgt = TARGET_SPLIT_PCT[split]
        print(f"  {split:<10s}{o:>10,}{k:>10,}{opct:>10.2f}%{kpct:>10.2f}%{tgt:>9.0f}%")

    # ---- (d) Top 20 elements pre/post -------------------------------------
    print()
    print("─" * 78)
    print("(d) TOP-20 CENTER_ELEMENT — DROP IMPACT")
    print("─" * 78)
    elem_pre  = df.groupby("center_element").size().rename("n_orig")
    elem_post = df_keep.groupby("center_element").size().rename("n_keep")
    elem_table = pd.concat([elem_pre, elem_post], axis=1).fillna(0).astype(int)
    elem_table["n_drop"]   = elem_table["n_orig"] - elem_table["n_keep"]
    elem_table["drop_pct"] = (100.0 * elem_table["n_drop"] / elem_table["n_orig"]).round(2)
    elem_table = elem_table.sort_values("n_orig", ascending=False)

    print(f"  {'rank':>4s}  {'elem':<5s} {'n_orig':>8s} {'n_keep':>8s} {'n_drop':>8s} {'drop%':>8s}")
    print(f"  {'-'*4}  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, (elem, row) in enumerate(elem_table.head(20).iterrows(), 1):
        print(f"  {i:>4d}  {elem:<5s} {row['n_orig']:>8,} {row['n_keep']:>8,} "
              f"{row['n_drop']:>8,} {row['drop_pct']:>7.2f}%")

    # ---- (e) elements with low post-drop counts ---------------------------
    print()
    print("─" * 78)
    print("(e) ELEMENTS WITH < 200 SAMPLES AFTER DROP (statistical concern)")
    print("─" * 78)
    low = elem_table[elem_table["n_keep"] < 200].sort_values("n_keep")
    if len(low) == 0:
        print("  None.")
    else:
        print(f"  Total elements with n_keep < 200: {len(low)}")
        print(f"  Of these, n_keep == 0 (completely lost): "
              f"{int((elem_table['n_keep'] == 0).sum())}")
        print()
        print(f"  {'elem':<5s} {'n_orig':>8s} {'n_keep':>8s} {'drop%':>8s}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for elem, row in low.iterrows():
            print(f"  {elem:<5s} {row['n_orig']:>8,} {row['n_keep']:>8,} {row['drop_pct']:>7.2f}%")

    # ---- (f) split × element coverage -------------------------------------
    print()
    print("─" * 78)
    print("(f) ELEMENT COVERAGE PER SPLIT")
    print("─" * 78)
    cov_orig = df.groupby("split")["center_element"].nunique().to_dict()
    cov_keep = df_keep.groupby("split")["center_element"].nunique().to_dict()
    print(f"  {'split':<10s}{'orig_elems':>14s}{'keep_elems':>14s}{'lost':>8s}")
    print(f"  {'-'*10}{'-'*14}{'-'*14}{'-'*8}")
    for split in ["train", "val", "test", "holdout"]:
        o = cov_orig.get(split, 0)
        k = cov_keep.get(split, 0)
        print(f"  {split:<10s}{o:>14d}{k:>14d}{o-k:>8d}")

    # ---- (g) mp_id totals -------------------------------------------------
    print()
    print("─" * 78)
    print("(g) MP_ID TOTALS")
    print("─" * 78)
    mpid_orig_set = set(df["mp_id"].unique())
    mpid_keep_set = set(df_keep["mp_id"].unique())
    mpid_lost = mpid_orig_set - mpid_keep_set
    print(f"  Original unique mp_ids:                            {len(mpid_orig_set):>7,}")
    print(f"  After-drop unique mp_ids (≥1 valid spectrum left): {len(mpid_keep_set):>7,}")
    print(f"  Completely-lost mp_ids (all spectra incompat):     {len(mpid_lost):>7,}")
    print(f"  Drop rate of mp_ids:                               "
          f"{fmt_pct(len(mpid_lost), len(mpid_orig_set))}")

    # ---- (h) mp_id per split ----------------------------------------------
    print()
    print("─" * 78)
    print("(h) MP_ID COUNT PER SPLIT")
    print("─" * 78)
    print(f"  {'split':<10s}{'orig_mpids':>13s}{'keep_mpids':>13s}{'lost':>8s}{'lost%':>9s}")
    print(f"  {'-'*10}{'-'*13}{'-'*13}{'-'*8}{'-'*9}")
    mpid_per_split_orig = df.groupby("split")["mp_id"].nunique().to_dict()
    mpid_per_split_keep = df_keep.groupby("split")["mp_id"].nunique().to_dict()
    for split in ["train", "val", "test", "holdout"]:
        o = mpid_per_split_orig.get(split, 0)
        k = mpid_per_split_keep.get(split, 0)
        lost = o - k
        pct = 100.0 * lost / o if o else 0
        print(f"  {split:<10s}{o:>13,}{k:>13,}{lost:>8,}{pct:>8.2f}%")

    # ---- (i) per-mp_id at-least-one-valid invariant ----------------------
    print()
    print("─" * 78)
    print("(i) INVARIANT: every retained mp_id has ≥1 non-incompat sample")
    print("─" * 78)
    counts_per_mpid_in_keep = df_keep.groupby("mp_id").size()
    bad_count = int((counts_per_mpid_in_keep < 1).sum())
    print(f"  Retained mp_id count:                  {len(counts_per_mpid_in_keep):,}")
    print(f"  Of those with 0 valid samples:         {bad_count}  "
          f"(must be 0 by construction — sanity check)")
    print(f"  Min valid-samples per retained mp_id:  {counts_per_mpid_in_keep.min()}")
    print(f"  Median:                                {int(counts_per_mpid_in_keep.median())}")
    print(f"  Max:                                   {counts_per_mpid_in_keep.max()}")

    # mp_ids that lost some but not all of their spectra
    counts_per_mpid_in_orig = df.groupby("mp_id").size()
    surviving = counts_per_mpid_in_keep.reindex(counts_per_mpid_in_orig.index, fill_value=0)
    partially_dropped = ((surviving > 0) & (surviving < counts_per_mpid_in_orig)).sum()
    fully_dropped     = (surviving == 0).sum()
    untouched         = (surviving == counts_per_mpid_in_orig).sum()
    print()
    print(f"  By mp_id 'damage state':")
    print(f"    Untouched (no spectrum dropped):       {untouched:>7,}  "
          f"({fmt_pct(untouched, len(mpid_orig_set))})")
    print(f"    Partially dropped (some spectra lost): {partially_dropped:>7,}  "
          f"({fmt_pct(partially_dropped, len(mpid_orig_set))})")
    print(f"    Fully dropped (mp_id removed):         {fully_dropped:>7,}  "
          f"({fmt_pct(fully_dropped, len(mpid_orig_set))})")

    print()
    print("=" * 78)
    print("DONE.")
    print("=" * 78)


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
        text = log_buf.getvalue()
        with open(OUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nWrote: {OUT_TXT_PATH}")
        print(f"       {LOG_PATH}")
