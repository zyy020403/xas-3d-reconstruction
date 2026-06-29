# step1_5_fit_scaler.py
# ------------------------------------------------------------
# Exp4 Step 1.5
#   - Load step1_4_full_inventory.pkl (has split + 74 feff features)
#   - Filter to split == "train"
#   - Fit RobustScaler on (N_train, 74)  — GLOBAL fit (not per-element)
#   - Do NOT transform the data; scaler is consumed by Step 2/3 loaders
#   - Save:
#       feff_feature_scaler.pkl
#       feff_feature_stats.csv
#       (with columns feature_name, median, iqr, q1, q3, min, max, n_nan_before_impute)
# ------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"


def main():
    inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_4_full_inventory.pkl"))
    print(f"[Load] step1_4_full_inventory: {inv.shape}")

    with open(os.path.join(STEP1_DIR, "step1_3_feff_feature_names.txt"),
              encoding="utf-8") as f:
        feat_names = [l.strip() for l in f if l.strip()]
    assert len(feat_names) == 74, f"expected 74 feat names, got {len(feat_names)}"
    print(f"[Feat] 74 feature names loaded")

    train = inv[inv["split"] == "train"].copy()
    print(f"[Train] {len(train)} samples")

    X = train[feat_names].values.astype(np.float64)
    assert not np.isnan(X).any(), "X_train contains NaN"
    print(f"[X_train] shape={X.shape}  dtype={X.dtype}")

    # ---------- Fit ----------
    scaler = RobustScaler()
    scaler.fit(X)
    print(f"[Fit] RobustScaler done")
    print(f"  center_[0:5]: {scaler.center_[:5]}")
    print(f"  scale_[0:5] : {scaler.scale_[:5]}")

    # ---------- Save pkl ----------
    pkl_out = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")
    joblib.dump(scaler, pkl_out)
    print(f"[Save] {pkl_out}")

    # ---------- Reload sanity check ----------
    reloaded = joblib.load(pkl_out)
    z = reloaded.transform(X[:5])
    assert z.shape == (5, 74), f"unexpected transform shape: {z.shape}"
    print(f"[Sanity] reloaded scaler.transform(X[:5]) -> {z.shape}  OK")

    # ---------- Build stats table ----------
    n_nan_path = os.path.join(STEP1_DIR, "step1_3_n_nan_before_impute.csv")
    n_nan_df = pd.read_csv(n_nan_path, index_col=0, header=0)
    n_nan_map = n_nan_df.iloc[:, 0].to_dict()

    rows = []
    for i, name in enumerate(feat_names):
        col = X[:, i]
        q1, med, q3 = np.percentile(col, [25, 50, 75])
        rows.append(dict(
            feature_name=name,
            median=float(med),
            iqr=float(q3 - q1),
            q1=float(q1),
            q3=float(q3),
            min=float(col.min()),
            max=float(col.max()),
            n_nan_before_impute=int(n_nan_map.get(name, 0)),
        ))
    stats = pd.DataFrame(rows)
    stats_out = os.path.join(STEP1_DIR, "feff_feature_stats.csv")
    stats.to_csv(stats_out, index=False)
    print(f"[Save] {stats_out}  rows={len(stats)}")

    # Cross-check center_ vs median
    max_diff = float(np.max(np.abs(scaler.center_ - stats["median"].values)))
    print(f"[Sanity] max |scaler.center_ - stats.median| = {max_diff:.6g}  (expect ~0)")

    print(f"\n----- Step 1.5 Summary -----")
    print(f"  scaler fit: {X.shape[0]} train samples × 74 feats")
    print(f"  center_[0] = {scaler.center_[0]:.6g}")
    print(f"  scale_[0]  = {scaler.scale_[0]:.6g}")


if __name__ == "__main__":
    main()
