# step1_2_filter_outliers.py
# ------------------------------------------------------------
# Exp4 Step 1.2
#   Filter in the following order (spec §5.4):
#     (0) drop invalid chi / xmu / poscar (from step1_1 flags)
#         - poscar_reason == "file_missing"       -> reason="missing_poscar"
#         - poscar_reason == parse/symmetry err  -> reason="poscar_invalid"
#         - chi_valid  == False                   -> reason="chi_invalid"
#         - xmu_valid  == False                   -> reason="xmu_invalid"
#     (a) drop center_element == "H"              -> reason="H_element"
#     (c) IQR*50 per (center_element, col) outlier -> reason="iqr_outlier"
#         * done ON 73 feff numeric cols
#         * skip col for group if group size < 5, or IQR == 0 / NaN
#         * any-column outlier => whole row dropped
#   Save:
#     step1_2_filtered_inventory.pkl  (kept samples, still with NaN)
#     step1_excluded_log.csv          (all reasons, including parse_fail from 1.1)
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"

IQR_MULTIPLIER = 50.0
MIN_GROUP_SIZE = 5

META_COLS    = ["sample_dir", "sample_name", "feature_version"]
NON_FEAT_COLS = META_COLS + [
    "mp_id", "center_element",
    "chi_path", "xmu_path", "poscar_path",
    "poscar_valid", "prim_n_atoms", "poscar_reason",
    "chi_valid", "chi_reason",
    "xmu_valid", "xmu_reason",
]


def main():
    inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_1_raw_inventory.pkl"))
    print(f"[Load] step1_1_raw_inventory: {inv.shape}")

    numeric_cols = [c for c in inv.columns if c not in NON_FEAT_COLS]
    print(f"[Info] numeric feff cols: {len(numeric_cols)} (expect 73)")
    assert len(numeric_cols) == 73, f"expected 73 numeric cols, got {len(numeric_cols)}"

    excluded = []  # list of dicts {sample_name, mp_id, center_element, reason}

    def dump(mask, reason):
        """Record masked rows to excluded log and drop them from inv."""
        nonlocal inv
        if not mask.any():
            print(f"[Filter] {reason:18s} drop=0  remaining={len(inv)}")
            return
        sub = inv[mask]
        for _, r in sub.iterrows():
            excluded.append(dict(
                sample_name=r["sample_name"],
                mp_id=r["mp_id"],
                center_element=r["center_element"],
                reason=reason,
            ))
        inv = inv[~mask].reset_index(drop=True)
        print(f"[Filter] {reason:18s} drop={int(mask.sum()):6d}  remaining={len(inv)}")

    # ---- (0) invalid chi ----
    dump(~inv["chi_valid"], "chi_invalid")

    # ---- (0) invalid xmu ----
    dump(~inv["xmu_valid"], "xmu_invalid")

    # ---- (0) invalid POSCAR: separate missing vs parse-fail ----
    m_missing = (~inv["poscar_valid"]) & (inv["poscar_reason"] == "file_missing")
    dump(m_missing, "missing_poscar")
    m_bad = (~inv["poscar_valid"])  # remaining ones are parse/symmetry errors
    dump(m_bad, "poscar_invalid")

    # ---- (a) drop H ----
    dump(inv["center_element"] == "H", "H_element")

    # ---- (c) IQR × 50 per (element, col) ----
    print(f"\n[Filter] IQR*{IQR_MULTIPLIER} outlier detection on {len(numeric_cols)} cols "
          f"grouped by center_element (skip if group<{MIN_GROUP_SIZE} or IQR==0/NaN)")
    is_outlier = pd.Series(False, index=inv.index)
    for col in tqdm(numeric_cols, desc="IQR cols"):
        grp = inv.groupby("center_element")[col]
        med = grp.transform("median")
        q1  = grp.transform(lambda x: x.quantile(0.25))
        q3  = grp.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        sz  = grp.transform("size")
        valid = (sz >= MIN_GROUP_SIZE) & iqr.notna() & (iqr > 0)
        col_mask = valid & ((inv[col] - med).abs() > IQR_MULTIPLIER * iqr)
        col_mask = col_mask.fillna(False).astype(bool)
        is_outlier = is_outlier | col_mask

    n_out = int(is_outlier.sum())
    print(f"  IQR outlier rows: {n_out}")
    dump(is_outlier, "iqr_outlier")

    # ---- Save filtered inventory ----
    out = os.path.join(STEP1_DIR, "step1_2_filtered_inventory.pkl")
    inv.to_pickle(out)
    print(f"\n[Save] {out}  shape={inv.shape}")

    # ---- Save excluded log (including parse_fail from step1_1) ----
    exc_df = pd.DataFrame(excluded, columns=["sample_name", "mp_id",
                                              "center_element", "reason"])
    pf_path = os.path.join(STEP1_DIR, "step1_1_parse_fail.csv")
    if os.path.isfile(pf_path):
        pf = pd.read_csv(pf_path)
        exc_df = pd.concat([pf, exc_df], ignore_index=True, sort=False)
    exc_out = os.path.join(STEP1_DIR, "step1_excluded_log.csv")
    exc_df.to_csv(exc_out, index=False)

    print(f"\n----- Step 1.2 Summary -----")
    print(f"  final kept samples: {len(inv)}")
    print(f"  excluded total:     {len(exc_df)}")
    print(f"  excluded by reason:")
    print(exc_df["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
