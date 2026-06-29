# check_prim_distribution_v2.py
# 按 mp_id 去重后统计（每个化合物只算一次，取site_nn最小的那条）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
NATOMS_CSV = os.path.join(STEP1_DIR, "prim_natoms_all.csv")
SCAN_CSV   = os.path.join(STEP1_DIR, "step1_poscar_check.csv")
OUTPUT_PNG = os.path.join(STEP1_DIR, "prim_natoms_distribution_v2.png")

def main():
    df_atoms = pd.read_csv(NATOMS_CSV, encoding="utf-8-sig")
    df_scan  = pd.read_csv(SCAN_CSV,   encoding="utf-8-sig")

    # 合并，拿到 mp_id 和 site_nn
    df = df_scan[["mp_id", "folder_name", "site_nn"]].merge(
        df_atoms[["folder_name", "prim_n_atoms"]], on="folder_name", how="inner"
    )

    # 每个 mp_id 只保留 site_nn 最小的那条（LVSI）
    df["site_nn_int"] = df["site_nn"].astype(int)
    df_lvsi = df.sort_values("site_nn_int").groupby("mp_id").first().reset_index()

    print(f"去重后化合物总数：{len(df_lvsi)}")

    arr = df_lvsi["prim_n_atoms"].values
    arr = arr[arr > 0]

    print(f"\n======= 按化合物去重后原胞原子数统计 =======")
    print(f"化合物总数 : {len(arr)}")
    print(f"mean       : {arr.mean():.1f}")
    print(f"median     : {np.median(arr):.1f}")
    print(f"min        : {arr.min()}")
    print(f"max        : {arr.max()}")

    # 分段统计（这次是真正的分段，加起来=总数）
    bins = [0, 10, 20, 30, 40, 60, 80, 999]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-60", "61-80", "81+"]
    total = len(arr)
    print(f"\n{'区间':>8}  {'数量':>6}  {'占比':>6}")
    print("-" * 28)
    for i, label in enumerate(labels):
        n = int(((arr > bins[i]) & (arr <= bins[i+1])).sum())
        print(f"{label:>8}  {n:>6}  {n/total*100:>5.1f}%")
    print(f"{'合计':>8}  {total:>6}  100.0%")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(arr, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].axvline(20, color="orange", linewidth=1.5, linestyle="--", label="DiffCSP original (20)")
    axes[0].axvline(80, color="red",    linewidth=1.5, linestyle="--", label="our cutoff (80)")
    axes[0].set_xlabel("primitive cell num_atoms")
    axes[0].set_ylabel("num compounds")
    axes[0].set_title("Full distribution (per compound)")
    axes[0].legend()

    arr_zoom = arr[arr <= 80]
    axes[1].hist(arr_zoom, bins=60, color="steelblue", edgecolor="none", alpha=0.85)
    axes[1].axvline(20, color="orange", linewidth=1.5, linestyle="--", label="DiffCSP original (20)")
    axes[1].set_xlabel("primitive cell num_atoms")
    axes[1].set_ylabel("num compounds")
    axes[1].set_title("Zoomed: num_atoms <= 80 (per compound)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"\n分布图 → {OUTPUT_PNG}")

if __name__ == "__main__":
    main()