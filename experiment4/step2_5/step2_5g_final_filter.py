"""
step2_5g_final_filter.py — Step 2.5 finalization (Option D execution)

Per MA's confirmed Option D decision:
  - Drop tag == 'incompat'
  - Drop tag == 'near_equivalent' AND n_unique_shell1_multisets > 1 (the 999)

Outputs (all in experiment4/step2_5/):
  - data_inventory_v2.csv     : 75,637 rows, v1 cols + site_equivalence_tag
  - train_samples_v2.csv      : 60,507 rows
  - val_samples_v2.csv        :  7,624 rows
  - test_samples_v2.csv       :  4,481 rows
  - holdout_samples_v2.csv    :  3,025 rows
  - incompat_pool.csv         : 52,745 rows (sealed for Exp5)
  - step2_5g_summary.txt      : Exp4 dataset name card
  - step2_5g_filter.log

Run
---
python step2_5g_final_filter.py
"""
import io
import os
import sys
from itertools import combinations

import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EXP4_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR  = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR = os.path.join(EXP4_ROOT, "step2_5")

INV_PATH = os.path.join(STEP1_DIR, "data_inventory.csv")
TAG_PATH = os.path.join(STEP25_DIR, "site_equivalence_tag.csv")

INV_V2_PATH        = os.path.join(STEP25_DIR, "data_inventory_v2.csv")
TRAIN_V2_PATH      = os.path.join(STEP25_DIR, "train_samples_v2.csv")
VAL_V2_PATH        = os.path.join(STEP25_DIR, "val_samples_v2.csv")
TEST_V2_PATH       = os.path.join(STEP25_DIR, "test_samples_v2.csv")
HOLDOUT_V2_PATH    = os.path.join(STEP25_DIR, "holdout_samples_v2.csv")
INCOMPAT_POOL_PATH = os.path.join(STEP25_DIR, "incompat_pool.csv")
SUMMARY_PATH       = os.path.join(STEP25_DIR, "step2_5g_summary.txt")
LOG_PATH           = os.path.join(STEP25_DIR, "step2_5g_filter.log")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Step 2.5 finalization — Option D filter + Exp4 dataset name card")
    print("=" * 72)

    # ---- Load ----
    print("\n  loading inputs ...")
    inv = pd.read_csv(INV_PATH)
    tag = pd.read_csv(TAG_PATH)
    print(f"    data_inventory.csv:        shape={inv.shape}, columns={len(inv.columns)}")
    print(f"    site_equivalence_tag.csv:  shape={tag.shape}")
    assert inv.shape[0] == 128382, f"inv rows = {inv.shape[0]}, expected 128382"
    assert tag.shape[0] == 128382, f"tag rows = {tag.shape[0]}, expected 128382"

    v1_cols = list(inv.columns)
    print(f"    v1 inventory columns: {v1_cols}")

    # ---- Merge (only the 3 fields we need from tag, to avoid _x/_y suffixes) ----
    print("\n  merging tag info into inventory ...")
    tag_subset = tag[["sample_name", "tag",
                      "n_unique_shell1_multisets", "max_shell1_MAE"]]
    merged = inv.merge(tag_subset, on="sample_name", how="left")
    assert merged.shape[0] == 128382, f"merge fan-out: {merged.shape[0]}"
    n_missing_tag = int(merged["tag"].isna().sum())
    assert n_missing_tag == 0, f"{n_missing_tag} samples in inv have no tag"
    print(f"    merged shape: {merged.shape}  ✓ no fan-out, no missing tags")

    # ---- Apply filter (Option D) ----
    print("\n  applying Option D filter ...")
    is_incompat = merged["tag"] == "incompat"
    is_neareq_mismatch = ((merged["tag"] == "near_equivalent")
                          & (merged["n_unique_shell1_multisets"] > 1))
    drop_mask = is_incompat | is_neareq_mismatch

    n_drop_incompat = int(is_incompat.sum())
    n_drop_neareq   = int(is_neareq_mismatch.sum())
    n_drop          = int(drop_mask.sum())
    n_keep          = int((~drop_mask).sum())

    print(f"    drop (tag=incompat):              {n_drop_incompat:>7,}")
    print(f"    drop (near_eq with multiset >1):  {n_drop_neareq:>7,}")
    print(f"    drop total:                       {n_drop:>7,}")
    print(f"    keep:                             {n_keep:>7,}")

    kept    = merged[~drop_mask].copy()
    dropped = merged[drop_mask].copy()

    # ---- Build data_inventory_v2.csv ----
    print(f"\n  writing data_inventory_v2.csv ...")
    inv_v2 = (kept.rename(columns={"tag": "site_equivalence_tag"})
                  [v1_cols + ["site_equivalence_tag"]])
    inv_v2.to_csv(INV_V2_PATH, index=False)
    size_mb = os.path.getsize(INV_V2_PATH) / 1024 / 1024
    print(f"    saved: {os.path.basename(INV_V2_PATH)}  "
          f"({size_mb:.2f} MB, {len(inv_v2):,} rows, {len(inv_v2.columns)} cols)")

    # ---- Build 4 split sample CSVs ----
    print(f"\n  writing 4 split sample CSVs ...")
    sample_cols = ["mp_id", "center_element", "sample_name", "site_equivalence_tag"]
    split_files = {
        "train":   TRAIN_V2_PATH,
        "val":     VAL_V2_PATH,
        "test":    TEST_V2_PATH,
        "holdout": HOLDOUT_V2_PATH,
    }
    split_counts = {}
    for split, path in split_files.items():
        df_split = inv_v2[inv_v2["split"] == split][sample_cols].copy()
        df_split.to_csv(path, index=False)
        split_counts[split] = len(df_split)
        size_kb = os.path.getsize(path) / 1024
        print(f"    {split:<10s}  {len(df_split):>7,}  →  "
              f"{os.path.basename(path)}  ({size_kb:.1f} KB)")

    # ---- Build incompat_pool.csv (with comment header) ----
    print(f"\n  writing incompat_pool.csv (with comment header) ...")
    pool = (dropped[["sample_name", "mp_id", "center_element", "split",
                     "tag", "max_shell1_MAE"]]
            .rename(columns={"split": "original_split"})
            .copy())

    with open(INCOMPAT_POOL_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("# purpose: exp5_training_pool\n")
        f.write("# These samples have site_equivalence_tag in "
                "{incompat, near_equivalent_multiset_mismatch}.\n")
        f.write("# Excluded from Exp4 training. Reserved for Exp5 if a "
                "site-averaging strategy is implemented.\n")
        f.write("# See Step 2.5 Phase D / F report for rationale.\n")
        pool.to_csv(f, index=False)
    pool_mb = os.path.getsize(INCOMPAT_POOL_PATH) / 1024 / 1024
    print(f"    saved: {os.path.basename(INCOMPAT_POOL_PATH)}  "
          f"({pool_mb:.2f} MB, {len(pool):,} rows)")
    print(f"    note: read with pd.read_csv(path, comment='#')")

    # ---- Assertions ----
    print(f"\n  running 6 final assertions ...")

    assert len(inv_v2) == 75637, f"inv_v2 has {len(inv_v2)} rows"
    print(f"    ✓ len(data_inventory_v2)              == 75,637")

    total_split = sum(split_counts.values())
    assert total_split == 75637, f"sum of split files = {total_split}"
    print(f"    ✓ sum(4 split v2 files)               == 75,637  "
          f"({split_counts['train']:,} + {split_counts['val']:,} + "
          f"{split_counts['test']:,} + {split_counts['holdout']:,})")

    assert len(pool) == 52745, f"pool has {len(pool)} rows"
    print(f"    ✓ len(incompat_pool)                  == 52,745")

    n_mpid = inv_v2["mp_id"].nunique()
    assert n_mpid == 35445, f"v2 has {n_mpid} mp_ids"
    print(f"    ✓ unique mp_ids in v2                 == 35,445")

    n_elem = inv_v2["center_element"].nunique()
    assert n_elem == 88, f"v2 has {n_elem} elements"
    print(f"    ✓ unique elements in v2               == 88")

    splits = ["train", "val", "test", "holdout"]
    mpids_per_split = {
        s: set(inv_v2[inv_v2["split"] == s]["mp_id"]) for s in splits
    }
    overlaps = []
    for s1, s2 in combinations(splits, 2):
        n_overlap = len(mpids_per_split[s1] & mpids_per_split[s2])
        if n_overlap > 0:
            overlaps.append(f"{s1}×{s2}={n_overlap}")
    assert len(overlaps) == 0, f"intersecting mp_ids: {overlaps}"
    print(f"    ✓ pairwise mp_id intersection (6 pairs) == 0")

    # ---- Build summary ("Exp4 dataset name card") ----
    print(f"\n  writing Exp4 dataset name card ...")

    tag_counts = inv_v2["site_equivalence_tag"].value_counts().to_dict()

    out = []
    out.append("=" * 64)
    out.append("Exp4 Dataset — Final Name Card")
    out.append("=" * 64)
    out.append("")
    out.append(f"Total samples:       {len(inv_v2):>8,}")
    out.append(f"Total mp_ids:        {n_mpid:>8,}")
    out.append(f"Element coverage:    {n_elem:>8d}  (Step 1 reported 88)")
    out.append("")
    out.append("Split distribution:")
    for s in splits:
        n = split_counts[s]
        pct = 100.0 * n / len(inv_v2)
        out.append(f"  {s:<10s}  {n:>7,}  ({pct:5.2f}%)")
    out.append("")
    out.append("Tag distribution (kept samples):")
    for t in ["single_site", "equivalent", "near_equivalent"]:
        n = int(tag_counts.get(t, 0))
        pct = 100.0 * n / len(inv_v2)
        out.append(f"  {t:<22s}  {n:>7,}  ({pct:5.2f}%)")
    out.append("")
    out.append("Comparison with Exp2:")
    out.append(f"  Exp2: ~11,636 samples (Fe-oxide focus, single element)")
    out.append(f"  Exp4: {len(inv_v2):,} samples ({n_mpid:,} mp_ids, {n_elem} elements)")
    out.append(f"        → 6.5× more samples, 88× more elements")
    out.append("")
    out.append("Sealed (incompat_pool.csv, reserved for Exp5):")
    out.append(f"  {len(pool):,} samples (40.31% of original 128,382)")
    out.append(f"    {n_drop_incompat:,} tagged 'incompat'")
    out.append(f"    {n_drop_neareq:,} tagged 'near_equivalent' with multiset mismatch")
    out.append("")
    out.append("Output files in this drop:")
    out.append(f"  data_inventory_v2.csv     ({len(inv_v2):,} rows)")
    out.append(f"  train_samples_v2.csv      ({split_counts['train']:,} rows)")
    out.append(f"  val_samples_v2.csv        ({split_counts['val']:,} rows)")
    out.append(f"  test_samples_v2.csv       ({split_counts['test']:,} rows)")
    out.append(f"  holdout_samples_v2.csv    ({split_counts['holdout']:,} rows)")
    out.append(f"  incompat_pool.csv         ({len(pool):,} rows, sealed)")

    summary_text = "\n".join(out)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"    saved: {os.path.basename(SUMMARY_PATH)}")

    print()
    print(summary_text)
    print()
    print("=" * 72)
    print("Step 2.5 finalization complete.")
    print("=" * 72)


# -----------------------------------------------------------------------------
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
