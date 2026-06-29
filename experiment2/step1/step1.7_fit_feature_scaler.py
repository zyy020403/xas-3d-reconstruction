# step1.7_fit_feature_scaler.py
# 输入：data_inventory.csv + feff_features CSV
# 输出：feff_feature_scaler.pkl, feff_feature_stats.csv

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

STEP1_DIR     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
SCALER_PKL    = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")
STATS_CSV     = os.path.join(STEP1_DIR, "feff_feature_stats.csv")

def parse_sample_name(s):
    try:
        parts = s.split("_")
        return parts[0]+"_"+parts[1], parts[-1].zfill(2)
    except:
        return None, None

def main():
    df_inv  = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    df_feat = pd.read_csv(FEFF_FEAT_CSV, encoding="utf-8-sig")

    parsed = df_feat["sample_name"].apply(parse_sample_name)
    df_feat["mp_id"]   = [x[0] for x in parsed]
    df_feat["site_nn"] = [x[1] for x in parsed]
    df_feat = df_feat.drop_duplicates(subset=["mp_id","site_nn"])

    feat_cols = df_feat.columns[3:76].tolist()
    print(f"特征列数 : {len(feat_cols)}")

    # 只用 train 集拟合 scaler
    train_ids = df_inv[df_inv["split"] == "train"]["mp_id"].tolist()
    print(f"train 样本数 : {len(train_ids)}")

    df_inv["site_nn"] = df_inv["site_nn"].astype(str).str.zfill(2)
    df_feat["site_nn"] = df_feat["site_nn"].astype(str).str.zfill(2)

    df_train = df_inv[df_inv["split"] == "train"][["mp_id","site_nn"]].merge(
        df_feat[["mp_id","site_nn"] + feat_cols],
        on=["mp_id","site_nn"], how="left"
    )

    print(f"merge 后 train 行数 : {len(df_train)}")

    # 统计各列均值/方差/NaN比例（供 Step3 填充 NaN 用）
    stats = pd.DataFrame({
        "feature":   feat_cols,
        "mean":      df_train[feat_cols].mean().values,
        "std":       df_train[feat_cols].std().values,
        "nan_ratio": df_train[feat_cols].isna().mean().values,
    })
    stats.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")
    print(f"特征统计已保存 → {STATS_CSV}")

    # 用均值填充 NaN 后拟合 scaler
    X = df_train[feat_cols].copy()
    for c in feat_cols:
        X[c] = X[c].fillna(X[c].mean())

    scaler = StandardScaler()
    scaler.fit(X.values)

    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler 已保存 → {SCALER_PKL}")

    # 验证
    X_scaled = scaler.transform(X.values)
    print(f"\n验证 scaler（train集）：")
    print(f"  scaled mean ≈ 0 : {np.abs(X_scaled.mean(axis=0)).max():.6f}")
    print(f"  scaled std  ≈ 1 : {np.abs(X_scaled.std(axis=0) - 1).max():.6f}")
    print(f"\nStep 1.7 完成")

if __name__ == "__main__":
    main()