# step1.5_match_feff_features.py  【修正版】

import os
import pandas as pd

STEP1_DIR     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
SITE_MAP_CSV  = os.path.join(STEP1_DIR, "selected_site_map.csv")
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
OUTPUT_CSV    = os.path.join(STEP1_DIR, "data_inventory.csv")

def parse_sample_name(sample_name: str):
    try:
        parts = sample_name.split("_")
        site_nn = parts[-1].zfill(2)
        mp_id = parts[0] + "_" + parts[1]
        return mp_id, site_nn
    except Exception:
        return None, None

def main():
    df_site = pd.read_csv(SITE_MAP_CSV, encoding="utf-8-sig")
    print(f"selected_site_map 行数      : {len(df_site)}")

    df_feat = pd.read_csv(FEFF_FEAT_CSV, encoding="utf-8-sig")
    print(f"feff_features 原始行数      : {len(df_feat)}")

    # 解析 mp_id 和 site_nn
    parsed = df_feat["sample_name"].apply(parse_sample_name)
    df_feat["mp_id"]   = [x[0] for x in parsed]
    df_feat["site_nn"] = [x[1] for x in parsed]

    # 去除完全重复行
    df_feat = df_feat.drop_duplicates(subset=["mp_id", "site_nn"])
    print(f"feff_features 去重后行数    : {len(df_feat)}")

    feat_cols = df_feat.columns[3:76].tolist()
    print(f"特征列数                    : {len(feat_cols)}")

    df_feat["has_nan_features"] = df_feat[feat_cols].isna().any(axis=1)
    print(f"含 NaN 特征的行数           : {df_feat['has_nan_features'].sum()}")

    # JOIN
    df_site["site_nn"] = df_site["site_nn"].astype(str).str.zfill(2)
    df_feat["site_nn"] = df_feat["site_nn"].astype(str).str.zfill(2)

    df_merged = df_site.merge(
        df_feat[["mp_id", "site_nn"] + feat_cols + ["has_nan_features"]],
        on=["mp_id", "site_nn"],
        how="left"
    )

    # 确认行数没有膨胀
    assert len(df_merged) == len(df_site), \
        f"merge 后行数异常！{len(df_merged)} != {len(df_site)}"
    print(f"\nmerge 后行数（应={len(df_site)}）: {len(df_merged)} ✓")

    df_merged["has_feff_feat"] = ~df_merged[feat_cols[0]].isna()
    n_matched = int(df_merged["has_feff_feat"].sum())
    n_missing = int((~df_merged["has_feff_feat"]).sum())
    print(f"匹配成功                    : {n_matched}")
    print(f"匹配失败（缺失）            : {n_missing}")

    if n_missing > 0:
        miss = df_merged[~df_merged["has_feff_feat"]][["mp_id","site_nn","formula"]].head(10)
        print("缺失样本示例：")
        print(miss.to_string(index=False))

    # 只保留匹配成功的
    df_out = df_merged[df_merged["has_feff_feat"]].copy()
    df_out["split"] = "unassigned"

    print(f"\n最终 data_inventory 行数    : {len(df_out)}")
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"输出文件 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()