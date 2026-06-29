# step1_4_split_stratified.py
# ------------------------------------------------------------
# Exp4 Step 1.4
#   mp_id-level 4-way stratified split (spec §5.6):
#     - rare element: global sample-level count < 20
#     - mp_id containing any rare element -> force entire mp_id to train
#     - For remaining mp_ids, primary_element = argmin count(E in elem_set)
#     - Stratified 3-step split on primary_element:
#         step A: 4% holdout   (test_size=0.04)
#         step B: 6% test      (test_size=0.0625  -> 6/96  of remainder)
#         step C: 10% val      (test_size=0.1111  -> 10/90 of remainder)
#       => final train:val:test:holdout = 0.80:0.10:0.06:0.04
#     - primary groups too small to stratify are moved to train
#       (escalating: drop smallest class(es) until sklearn accepts)
#   Outputs (per spec §6):
#     data_inventory.csv              (final spec-shape inventory)
#     train_ids.txt / val_ids.txt / test_ids.txt / holdout_ids.txt
#     train_samples.csv / val_samples.csv / test_samples.csv / holdout_samples.csv
#     step1_4_full_inventory.pkl      (for step1_5 scaler fitting)
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"

RANDOM_STATE = 42
RARE_THRESHOLD = 20

# derived from target proportions 0.80 : 0.10 : 0.06 : 0.04
TEST_SIZE_HOLDOUT = 0.04
TEST_SIZE_TEST    = 0.0625     # 0.06 / (1 - 0.04)
TEST_SIZE_VAL     = 0.1111111  # 0.10 / (1 - 0.04 - 0.06)


def safe_stratified_split(df, stratify_col, test_size, random_state):
    """
    Stratified split on df[stratify_col]. Classes too small for stratify are
    iteratively dropped from the split pool (and appended to train output).
    Returns (train_df, test_df, moved_to_train_classes).
    """
    moved_classes = []
    counts = df[stratify_col].value_counts()

    # Singletons must be moved up-front
    singletons = counts[counts < 2].index.tolist()
    forced = df[df[stratify_col].isin(singletons)].copy()
    pool   = df[~df[stratify_col].isin(singletons)].copy()
    moved_classes.extend(singletons)

    while True:
        if len(pool) == 0:
            return forced, df.iloc[0:0].copy(), moved_classes
        try:
            tr, te = train_test_split(
                pool, test_size=test_size, random_state=random_state,
                stratify=pool[stratify_col])
            tr = pd.concat([tr, forced], ignore_index=True)
            return tr, te, moved_classes
        except ValueError as e:
            # Drop the smallest class(es) and retry
            cur = pool[stratify_col].value_counts()
            smallest = cur[cur == cur.min()].index.tolist()
            print(f"  [stratify] retry: drop class(es) {smallest}  (err: {str(e)[:80]})")
            moved_classes.extend(smallest)
            extra = pool[pool[stratify_col].isin(smallest)].copy()
            forced = pd.concat([forced, extra], ignore_index=True)
            pool = pool[~pool[stratify_col].isin(smallest)].copy()


def tier_for_natoms(n):
    if n <= 30:  return "A"
    if n <= 80:  return "B"
    return "C"


def main():
    inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_3_imputed_inventory.pkl"))
    print(f"[Load] step1_3_imputed_inventory: {inv.shape}")

    # ---------- Global element counts (sample-level) ----------
    elem_counts = inv["center_element"].value_counts()
    print(f"\n[Elements] distinct: {len(elem_counts)}")
    print(f"  top 5:\n{elem_counts.head().to_string()}")
    rare_elems = set(elem_counts[elem_counts < RARE_THRESHOLD].index)
    print(f"\n[Rare] count<{RARE_THRESHOLD}: {len(rare_elems)} elements")
    print(f"  {sorted(rare_elems)}")

    # ---------- Per-mp_id classification ----------
    mp_groups = inv.groupby("mp_id")["center_element"].agg(lambda x: sorted(set(x))).reset_index()
    mp_groups.columns = ["mp_id", "elem_set"]

    def classify(row):
        eset = set(row["elem_set"])
        has_rare = bool(eset & rare_elems)
        if has_rare:
            return "rare_mpid", None
        nonrare = eset - rare_elems
        if not nonrare:
            return "rare_mpid", None
        primary = min(nonrare, key=lambda e: elem_counts[e])
        return "nonrare", primary

    mp_groups[["class", "primary_element"]] = mp_groups.apply(
        lambda r: pd.Series(classify(r)), axis=1)

    n_rare_mp = int((mp_groups["class"] == "rare_mpid").sum())
    n_nonrare_mp = int((mp_groups["class"] == "nonrare").sum())
    print(f"\n[mp_id classes] rare_mpid={n_rare_mp}  nonrare={n_nonrare_mp}")

    rare_mps = mp_groups[mp_groups["class"] == "rare_mpid"]["mp_id"].tolist()
    nonrare  = mp_groups[mp_groups["class"] == "nonrare"][["mp_id", "primary_element"]].copy()

    # ---------- 3-step stratified split ----------
    print(f"\n[Split A] holdout (target 4%)")
    rest1, holdout_df, mvA = safe_stratified_split(
        nonrare, "primary_element", TEST_SIZE_HOLDOUT, RANDOM_STATE)
    print(f"  holdout mp_ids={len(holdout_df)}, rest={len(rest1)}, moved-to-train={len(mvA)}")

    print(f"\n[Split B] test (target 6% total)")
    rest2, test_df, mvB = safe_stratified_split(
        rest1, "primary_element", TEST_SIZE_TEST, RANDOM_STATE)
    print(f"  test mp_ids={len(test_df)}, rest={len(rest2)}, moved-to-train={len(mvB)}")

    print(f"\n[Split C] val (target 10% total)")
    train_nr, val_df, mvC = safe_stratified_split(
        rest2, "primary_element", TEST_SIZE_VAL, RANDOM_STATE)
    print(f"  val mp_ids={len(val_df)}, train(nonrare)={len(train_nr)}, moved-to-train={len(mvC)}")

    moved_all = sorted(set(mvA) | set(mvB) | set(mvC))
    if moved_all:
        print(f"\n[Split] primary classes forced into train due to small size:")
        print(f"  {moved_all}")

    # ---------- Build final mp_id sets ----------
    train_ids   = set(train_nr["mp_id"]) | set(rare_mps)
    val_ids     = set(val_df["mp_id"])
    test_ids    = set(test_df["mp_id"])
    holdout_ids = set(holdout_df["mp_id"])

    print(f"\n[Assert] zero intersection between splits")
    pairs = [("train", train_ids, "val", val_ids),
             ("train", train_ids, "test", test_ids),
             ("train", train_ids, "holdout", holdout_ids),
             ("val",   val_ids,   "test", test_ids),
             ("val",   val_ids,   "holdout", holdout_ids),
             ("test",  test_ids,  "holdout", holdout_ids)]
    all_pass = True
    for a, sa, b, sb in pairs:
        ix = sa & sb
        mark = "OK" if not ix else "FAIL"
        print(f"  {a:8s} ∩ {b:8s} = {len(ix):4d}  [{mark}]")
        if ix:
            all_pass = False
            print(f"    examples: {list(ix)[:5]}")
    assert all_pass, "split intersection assertion FAILED"

    # ---------- Assign split + quality_tier to inv ----------
    def split_of(mp):
        if mp in train_ids:   return "train"
        if mp in val_ids:     return "val"
        if mp in test_ids:    return "test"
        if mp in holdout_ids: return "holdout"
        return "UNASSIGNED"

    inv["split"] = inv["mp_id"].map(split_of)
    assert (inv["split"] == "UNASSIGNED").sum() == 0
    inv["quality_tier"] = inv["prim_n_atoms"].apply(tier_for_natoms)
    inv["is_iqr_outlier"] = False  # kept samples all passed IQR

    # ---------- Save data_inventory.csv (spec §6.1) ----------
    inventory_cols = [
        "sample_name", "mp_id", "center_element",
        "chi_path", "xmu_path", "poscar_path",
        "prim_n_atoms", "has_pre_edge",
        "chi_valid", "xmu_valid", "poscar_valid",
        "is_iqr_outlier", "split", "quality_tier",
    ]
    di = inv[inventory_cols].copy()
    di.to_csv(os.path.join(STEP1_DIR, "data_inventory.csv"), index=False)
    print(f"\n[Save] data_inventory.csv  shape={di.shape}")

    # ---------- Save full inventory with feff features for step1_5 ----------
    inv.to_pickle(os.path.join(STEP1_DIR, "step1_4_full_inventory.pkl"))
    print(f"[Save] step1_4_full_inventory.pkl (with feff features, for step1_5)")

    # ---------- Save id txts ----------
    def write_ids(name, ids):
        with open(os.path.join(STEP1_DIR, name), "w", encoding="utf-8") as f:
            for mp in sorted(ids):
                f.write(mp + "\n")
    write_ids("train_ids.txt",   train_ids)
    write_ids("val_ids.txt",     val_ids)
    write_ids("test_ids.txt",    test_ids)
    write_ids("holdout_ids.txt", holdout_ids)

    # ---------- Save sample csvs ----------
    scols = ["mp_id", "center_element", "sample_name"]
    for split in ("train", "val", "test", "holdout"):
        sub = inv[inv["split"] == split][scols]
        sub.to_csv(os.path.join(STEP1_DIR, f"{split}_samples.csv"), index=False)

    print(f"\n----- Step 1.4 Summary -----")
    for split in ("train", "val", "test", "holdout"):
        sub = inv[inv["split"] == split]
        print(f"  {split:8s}  mp_ids={sub['mp_id'].nunique():6d}  "
              f"samples={len(sub):7d}  ({len(sub)/len(inv):.2%})")
    if moved_all:
        print(f"  small-primary classes forced to train ({len(moved_all)}): {moved_all}")


if __name__ == "__main__":
    main()
