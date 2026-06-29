# step1_3_feff_imputation.py
# ------------------------------------------------------------
# Exp4 Step 1.3
#   Order is strict (spec §5.5):
#     1. has_pre_edge = (pre_peak_I.notna()).astype(int)      <-- BEFORE any fillna
#     2. fill_zero cols      -> .fillna(0.0)
#     3. fill_group_median cols (7 *_E) -> groupby(center_element).transform(
#            lambda x: x.fillna(x.median())),
#        then fallback to global median if a group's median is still NaN
#     4. any remaining NaN in the 12 "uncovered" metadata cols
#        (xmu_E*, E0, mu_at_E0, dmu_max, flag_*_valid, chi_k*) is
#        handled with the same group-median+global-median strategy
#        (safety net — spec is silent on these; normally all filled)
#   Final feff feature dim = 73 + 1(has_pre_edge) = 74
#   Save:
#     step1_3_imputed_inventory.pkl
#     step1_3_feff_feature_names.txt  (74 names, order fixed)
#     step1_3_n_nan_before_impute.csv (per-col NaN count BEFORE impute)
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"

FILL_ZERO_EXPLICIT = [
    "pre_peak_I", "white_line_I", "post_peak1_I", "d1_pre_I", "d1_post_I",
    "area_pre", "area_edge", "area_white_line", "area_post1",
    "pre_white_ratio", "post_white_ratio",
]
FILL_ZERO_PREFIXES = ("k2chi_", "k3chi_", "R1_", "R2_")

FILL_GROUP_MEDIAN = [
    "pre_peak_E", "white_line_E", "post_peak1_E", "d1_pre_E", "d1_post_E",
    "pre_centroid_E", "white_centroid_E",
]

META_COLS = ["sample_dir", "sample_name", "feature_version"]
NON_FEAT_COLS = META_COLS + [
    "mp_id", "center_element",
    "chi_path", "xmu_path", "poscar_path",
    "poscar_valid", "prim_n_atoms", "poscar_reason",
    "chi_valid", "chi_reason",
    "xmu_valid", "xmu_reason",
]


def main():
    inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_2_filtered_inventory.pkl"))
    print(f"[Load] step1_2_filtered_inventory: {inv.shape}")

    numeric_cols = [c for c in inv.columns if c not in NON_FEAT_COLS]
    assert len(numeric_cols) == 73, f"expected 73 numeric cols, got {len(numeric_cols)}"

    # ---------- Step 1: has_pre_edge BEFORE any fillna ----------
    print(f"\n[Step 1] has_pre_edge (from raw pre_peak_I NaN status)")
    assert "pre_peak_I" in inv.columns
    inv["has_pre_edge"] = inv["pre_peak_I"].notna().astype(int)
    n_has = int(inv["has_pre_edge"].sum())
    print(f"  has_pre_edge=1: {n_has} / {len(inv)}  ({n_has/len(inv):.3%})")

    # ---------- Record n_nan_before_impute ----------
    n_nan_before = inv[numeric_cols].isna().sum()
    n_nan_before.to_csv(os.path.join(STEP1_DIR, "step1_3_n_nan_before_impute.csv"),
                         header=["n_nan"])
    print(f"  saved n_nan_before_impute for {len(numeric_cols)} cols")
    print(f"  total NaN cells pre-impute: {int(n_nan_before.sum())}")

    # ---------- Classify cols ----------
    fill_zero_set = set(FILL_ZERO_EXPLICIT)
    for c in numeric_cols:
        if c.startswith(FILL_ZERO_PREFIXES):
            fill_zero_set.add(c)
    fill_zero_cols   = [c for c in numeric_cols if c in fill_zero_set]
    fill_median_cols = [c for c in FILL_GROUP_MEDIAN if c in numeric_cols]
    covered = set(fill_zero_cols) | set(fill_median_cols)
    uncovered = [c for c in numeric_cols if c not in covered]

    print(f"\n[Classify]")
    print(f"  fill_zero      cols: {len(fill_zero_cols)}")
    print(f"  group_median   cols: {len(fill_median_cols)}  {fill_median_cols}")
    print(f"  uncovered      cols: {len(uncovered)}  {uncovered}")

    # ---------- Step 2: fill 0 ----------
    print(f"\n[Step 2] Fill 0 in {len(fill_zero_cols)} cols")
    for c in fill_zero_cols:
        inv[c] = inv[c].fillna(0.0)

    # ---------- Step 3: group median for *_E + safety net for uncovered ----------
    # Uncovered cols: if any NaN left, fall back to same group-median strategy
    uncovered_with_nan = [c for c in uncovered if inv[c].isna().any()]
    if uncovered_with_nan:
        print(f"\n[Safety] {len(uncovered_with_nan)} 'uncovered' col(s) have residual NaN; "
              f"applying group-median fallback:")
        print(f"         {uncovered_with_nan}")
    all_median_cols = fill_median_cols + uncovered_with_nan

    print(f"\n[Step 3] Group-median fill for {len(all_median_cols)} cols")
    for c in all_median_cols:
        # per-group median
        inv[c] = inv.groupby("center_element")[c].transform(
            lambda x: x.fillna(x.median()))
        # global fallback
        remain = int(inv[c].isna().sum())
        if remain > 0:
            gm = inv[c].median()
            if pd.isna(gm):
                print(f"  [ERROR] col {c}: even global median is NaN -> fallback 0.0")
                inv[c] = inv[c].fillna(0.0)
            else:
                print(f"  [Fallback] col {c}: {remain} NaN -> global median = {gm:.6g}")
                inv[c] = inv[c].fillna(gm)

    # ---------- Final assert: no NaN in 73 numeric cols ----------
    residual = int(inv[numeric_cols].isna().sum().sum())
    assert residual == 0, (
        f"Post-impute NaN count = {residual} (should be 0). "
        f"Columns with NaN: "
        f"{inv[numeric_cols].isna().sum().pipe(lambda s: s[s>0]).to_dict()}"
    )
    print(f"\n[Assert] post-impute NaN count in 73 numeric cols: 0 ✓")

    # ---------- Final feature list (74) ----------
    feat_names = numeric_cols + ["has_pre_edge"]
    assert len(feat_names) == 74
    print(f"[Dim] feff feature dim = {len(feat_names)}  ✓")

    # ---------- Save ----------
    inv.to_pickle(os.path.join(STEP1_DIR, "step1_3_imputed_inventory.pkl"))
    with open(os.path.join(STEP1_DIR, "step1_3_feff_feature_names.txt"),
              "w", encoding="utf-8") as f:
        for c in feat_names:
            f.write(c + "\n")
    print(f"\n[Save] step1_3_imputed_inventory.pkl")
    print(f"[Save] step1_3_feff_feature_names.txt  ({len(feat_names)} names)")

    print(f"\n----- Step 1.3 Summary -----")
    print(f"  kept rows:                 {len(inv)}")
    print(f"  feff dim (final):          74")
    print(f"  has_pre_edge value_counts: {inv['has_pre_edge'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
