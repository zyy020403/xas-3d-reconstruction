# step1.4_lvsi_site_selection.py
# 输入：step1_poscar_check.csv + prim_natoms_all.csv
# 任务：按 mp_id 选最小有效 site（LVSI），过滤 prim_n_atoms > 80
# 输出：selected_site_map.csv

import os
import pandas as pd

STEP1_DIR      = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
POSCAR_CSV     = os.path.join(STEP1_DIR, "step1_poscar_check.csv")
NATOMS_CSV     = os.path.join(STEP1_DIR, "prim_natoms_all.csv")
OUTPUT_CSV     = os.path.join(STEP1_DIR, "selected_site_map.csv")

MAX_PRIM_ATOMS = 80

def main():
    df_scan  = pd.read_csv(POSCAR_CSV,  encoding="utf-8-sig")
    df_atoms = pd.read_csv(NATOMS_CSV,  encoding="utf-8-sig")

    # 合并原胞原子数
    df = df_scan.merge(
        df_atoms[["folder_name", "prim_n_atoms", "prim_reason"]],
        on="folder_name", how="left"
    )

    print(f"合并后总行数       : {len(df)}")
    print(f"unique mp_id       : {df['mp_id'].nunique()}")

    # 只保留三项全有效的记录
    df = df[df["chi1_valid"] & df["xmu_valid"] & df["poscar_valid"]].copy()
    print(f"三项全有效行数     : {len(df)}")

    # 每个 mp_id 统计总 site 数 和 有效 site 数
    total_sites = df.groupby("mp_id")["site_nn"].count().rename("total_sites")
    valid_sites = df.groupby("mp_id")["site_nn"].count().rename("valid_sites")

    # LVSI：每个 mp_id 取 site_nn 最小的那条
    df["site_nn_int"] = df["site_nn"].astype(int)
    df_lvsi = (
        df.sort_values("site_nn_int")
          .groupby("mp_id")
          .first()
          .reset_index()
    )
    df_lvsi = df_lvsi.merge(total_sites, on="mp_id").merge(valid_sites, on="mp_id")

    print(f"\nLVSI 后化合物数    : {len(df_lvsi)}")

    # 过滤 prim_n_atoms > MAX_PRIM_ATOMS
    mask_large = df_lvsi["prim_n_atoms"] > MAX_PRIM_ATOMS
    n_excluded = int(mask_large.sum())
    df_lvsi["excluded_large"] = mask_large

    df_keep = df_lvsi[~mask_large].copy()
    print(f"过滤 >{MAX_PRIM_ATOMS} 原子        : {n_excluded} 个化合物 excluded")
    print(f"最终保留化合物数   : {len(df_keep)}")

    # 单 site vs 多 site 分布
    single = int((df_keep["total_sites"] == 1).sum())
    multi  = int((df_keep["total_sites"] >  1).sum())
    print(f"\n单 site 化合物     : {single}")
    print(f"多 site 化合物     : {multi}")

    # 原子数分布
    arr = df_keep["prim_n_atoms"].values
    import numpy as np
    bins   = [0, 10, 20, 30, 40, 60, 80]
    labels = ["1-10","11-20","21-30","31-40","41-60","61-80"]
    print(f"\n{'区间':>8}  {'数量':>6}  {'占比':>6}")
    print("-" * 28)
    for i, label in enumerate(labels):
        n = int(((arr > bins[i]) & (arr <= bins[i+1])).sum())
        print(f"{label:>8}  {n:>6}  {n/len(arr)*100:>5.1f}%")
    print(f"{'合计':>8}  {len(arr):>6}  100.0%")

    # 输出
    out_cols = ["mp_id", "formula", "folder_name", "folder_path",
                "site_nn", "total_sites", "valid_sites",
                "prim_n_atoms", "prim_reason"]
    df_keep[out_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n输出文件 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()