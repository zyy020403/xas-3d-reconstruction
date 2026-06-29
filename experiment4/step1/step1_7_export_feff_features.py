# step1_7_export_feff_features.py
# ------------------------------------------------------------
# Exp4 Step 1.7 (post-hoc export for Step 2/3 consumption)
#   Load step1_4_full_inventory.pkl (has 74 feff + metadata)
#   Produce a SLIM, Step-2-friendly artifact:
#     feff_features_imputed.pkl
#       - pandas DataFrame
#       - index = sample_name (str, unique)
#       - 74 columns = feature names in step1_3_feff_feature_names.txt order
#       - dtype = float32
#   Also emit feff_features_imputed_head.csv for human inspection.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd

STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1"


def main():
    inv = pd.read_pickle(os.path.join(STEP1_DIR, "step1_4_full_inventory.pkl"))
    print(f"[Load] step1_4_full_inventory: {inv.shape}")

    with open(os.path.join(STEP1_DIR, "step1_3_feff_feature_names.txt"),
              encoding="utf-8") as f:
        feat_names = [l.strip() for l in f if l.strip()]
    assert len(feat_names) == 74

    # Build slim DF
    df = inv.set_index("sample_name")[feat_names].astype(np.float32)
    assert df.index.is_unique, "sample_name not unique!"
    assert df.shape[1] == 74
    assert not df.isna().any().any(), "residual NaN in feff features!"

    out_pkl = os.path.join(STEP1_DIR, "feff_features_imputed.pkl")
    df.to_pickle(out_pkl)
    size_mb = os.path.getsize(out_pkl) / 1024 / 1024
    print(f"[Save] {out_pkl}  shape={df.shape}  size={size_mb:.1f} MB  dtype=float32")

    # Preview csv
    df.head(5).to_csv(os.path.join(STEP1_DIR, "feff_features_imputed_head.csv"))
    print(f"[Save] feff_features_imputed_head.csv (first 5 rows)")

    # Quick sanity
    print(f"\n[Sanity]")
    print(f"  n rows:  {len(df)}")
    print(f"  n cols:  {df.shape[1]}")
    print(f"  dtype:   {df.dtypes.unique()}")
    print(f"  index:   unique={df.index.is_unique}, name={df.index.name}")
    print(f"  min/max of col 0 ({feat_names[0]}): "
          f"{df[feat_names[0]].min():.4g} / {df[feat_names[0]].max():.4g}")


if __name__ == "__main__":
    main()
