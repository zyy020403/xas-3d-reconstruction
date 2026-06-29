# step1_diag_check_outliers.py
# ------------------------------------------------------------
# Exp4 Step 1 post-hoc diagnostic
#   Audit the 74-dim imputed feff feature matrix to verify nothing
#   absurd slipped through.
#   - Per-column: min / max / mean / std / count(<0) / count(|x|>1e6)
#   - Per-column: top-5 smallest & top-5 largest with sample_name + center_element
#   - Energy-column special audit: negative or zero-energy samples
#   - Within-element outlier scan: |x - group_median| / group_iqr > 50 survivors
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"


def main():
    # ---------- Load ----------
    feff = pd.read_pickle(os.path.join(STEP1_DIR, "feff_features_imputed.pkl"))
    inv  = pd.read_csv(os.path.join(STEP1_DIR, "data_inventory.csv"),
                        usecols=["sample_name", "mp_id", "center_element"])
    inv = inv.set_index("sample_name")
    print(f"[Load] feff: {feff.shape}, inv: {inv.shape}")

    feat_names = list(feff.columns)
    assert len(feat_names) == 74

    # Merge element for per-element analysis
    df = feff.join(inv[["center_element", "mp_id"]])
    assert df["center_element"].notna().all()

    # ==================================================================
    # A. Per-column global stats
    # ==================================================================
    print("\n" + "=" * 78)
    print("A. Per-column global stats (74 cols)")
    print("=" * 78)
    rows = []
    for c in feat_names:
        x = feff[c].values
        rows.append({
            "col": c,
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "n_neg": int((x < 0).sum()),
            "n_abs_gt_1e6": int((np.abs(x) > 1e6).sum()),
            "n_abs_gt_1e4": int((np.abs(x) > 1e4).sum()),
        })
    stats = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4g}".format)
    print(stats.to_string(index=False))

    # Save for reference
    stats.to_csv(os.path.join(STEP1_DIR, "step1_diag_col_stats.csv"), index=False)

    # ==================================================================
    # B. Flag suspicious columns
    # ==================================================================
    print("\n" + "=" * 78)
    print("B. Suspicious columns (any of: n_neg>0 / |x|>1e6 / impossible)")
    print("=" * 78)

    susp = stats[(stats["n_abs_gt_1e6"] > 0) | (stats["min"] < -1e4) |
                  (stats["max"] > 1e6)]
    if len(susp) == 0:
        print("  (none — max |x| <= 1e6 and min > -1e4 across all 74 cols)")
    else:
        print(susp.to_string(index=False))

    # ==================================================================
    # C. Per-column: top-5 min & top-5 max samples
    # ==================================================================
    print("\n" + "=" * 78)
    print("C. Per-column top-5 MIN and top-5 MAX samples")
    print("=" * 78)
    for c in feat_names:
        s = df[c]
        bot = s.nsmallest(5)
        top = s.nlargest(5)
        print(f"\n--- {c} ---")
        print(f"  top-5 MIN:")
        for sn, v in bot.items():
            print(f"    {v:14.6g}  {sn}  ({df.loc[sn,'center_element']})")
        print(f"  top-5 MAX:")
        for sn, v in top.items():
            print(f"    {v:14.6g}  {sn}  ({df.loc[sn,'center_element']})")

    # ==================================================================
    # D. Energy-column special audit
    # ==================================================================
    print("\n" + "=" * 78)
    print("D. Energy-column special audit")
    print("=" * 78)
    energy_cols = [c for c in feat_names
                   if ("_E" in c) or c in ("E0",) or c.startswith("xmu_E")]
    print(f"  energy-like cols ({len(energy_cols)}): {energy_cols}")
    for c in energy_cols:
        x = feff[c]
        n_neg  = int((x < 0).sum())
        n_zero = int((x == 0).sum())
        n_huge = int((x > 1.5e5).sum())  # >150 keV is suspicious (no K-edge there)
        print(f"\n  [{c}]  n(<0)={n_neg}  n(==0)={n_zero}  n(>150keV)={n_huge}")
        if n_neg:
            sub = df[x < 0]
            print(f"    negative examples (first 10):")
            for sn, row in sub.head(10).iterrows():
                print(f"      {row[c]:14.6g}  {sn}  ({row['center_element']})")
        if n_huge:
            sub = df[x > 1.5e5]
            print(f"    >150 keV examples (first 10):")
            for sn, row in sub.head(10).iterrows():
                print(f"      {row[c]:14.6g}  {sn}  ({row['center_element']})")

    # ==================================================================
    # E. Within-element IQR*50 residual outlier scan
    # (samples that survived step1_2 but look element-weird anyway)
    # ==================================================================
    print("\n" + "=" * 78)
    print("E. Within-element IQR*50 residual check on final kept samples")
    print("   (sanity: should be 0 — they all passed step1_2's IQR*50 gate)")
    print("=" * 78)
    hits = 0
    for c in feat_names:
        grp = df.groupby("center_element")[c]
        med = grp.transform("median")
        q1  = grp.transform(lambda x: x.quantile(0.25))
        q3  = grp.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        sz  = grp.transform("size")
        ok  = (sz >= 5) & iqr.notna() & (iqr > 0)
        mask = ok & ((df[c] - med).abs() > 50.0 * iqr)
        n = int(mask.fillna(False).sum())
        if n > 0:
            hits += n
            print(f"  {c}: {n} residual outliers")
    print(f"\n  total residual outlier hits across cols: {hits}")
    print(f"  (if >0, these are samples that were borderline and squeaked through)")

    # ==================================================================
    # F. Known physical sanity (Actinide K-edges)
    # ==================================================================
    print("\n" + "=" * 78)
    print("F. Known physical sanity: elements with K-edge > 100 keV")
    print("=" * 78)
    hi_e = {
        "U":  115.606,  # in keV
        "Np": 118.669,
        "Pu": 121.818,
        "Am": 125.027,
    }
    for e, kev in hi_e.items():
        sub = df[df["center_element"] == e]
        if len(sub) == 0:
            print(f"  {e} ({kev} keV): no samples (OK if none in dataset)")
            continue
        emin = sub["xmu_Emin"].median() if "xmu_Emin" in sub.columns else np.nan
        e0   = sub["E0"].median()        if "E0"        in sub.columns else np.nan
        print(f"  {e} ({kev} keV): n={len(sub)}  "
              f"median xmu_Emin={emin:.1f} eV  median E0={e0:.1f} eV  "
              f"(both should be in ~100-130 keV range if physical)")


if __name__ == "__main__":
    main()
