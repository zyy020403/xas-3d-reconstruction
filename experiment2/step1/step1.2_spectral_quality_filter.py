# step1.2_spectral_quality_filter.py  【最终修正版 v2】

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

STEP1_DIR   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
INPUT_CSV   = os.path.join(STEP1_DIR, "step1_raw_scan.csv")
OUTPUT_CSV  = os.path.join(STEP1_DIR, "step1_quality_filter.csv")
OUTPUT_PNG  = os.path.join(STEP1_DIR, "chi_std_distribution.png")

def read_chi1_std(folder_path: str):
    fpath = Path(folder_path) / "chi1.dat"
    try:
        data = np.loadtxt(fpath, comments="#")
        if data.ndim < 2 or data.shape[0] < 30:
            return 0.0, False, f"行数不足({data.shape[0] if data.ndim>=1 else 0})"
        chi_col = data[:, 1]
        if np.any(~np.isfinite(chi_col)):
            return 0.0, False, "含NaN/Inf"
        std = float(np.std(chi_col))
        if std < 1e-6:
            return std, False, "全平/死谱(std<1e-6)"
        return std, True, "ok"
    except Exception as e:
        return 0.0, False, f"读取失败:{e}"

def read_xmu_valid(folder_path: str):
    fpath = Path(folder_path) / "xmu.dat"
    try:
        data = np.loadtxt(fpath, comments="#")
        if data.ndim < 2 or data.shape[0] < 50:
            return False, f"行数不足({data.shape[0] if data.ndim>=1 else 0})"
        if data.shape[1] < 4:
            return False, f"列数不足({data.shape[1]}列)"
        mu_col = data[:, 3]   # omega / e / k / mu / mu0 / chi
        if np.any(~np.isfinite(mu_col)):
            return False, "含NaN/Inf"
        if np.all(mu_col == 0):
            return False, "全零"
        if np.std(mu_col) == 0:
            return False, "无变化(std=0)"
        return True, "ok"
    except Exception as e:
        return False, f"读取失败:{e}"

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    mask_ok = (df["mp_id"] != "PARSE_FAIL") & \
              df["has_chi1"] & df["has_xmu"] & df["has_poscar"]
    df_work = df[mask_ok].copy()
    print(f"输入记录数（三文件齐全）：{len(df_work)}")

    chi_stds    = []
    chi_valids  = []
    chi_reasons = []
    xmu_valids  = []
    xmu_reasons = []

    for _, row in tqdm(df_work.iterrows(), total=len(df_work), desc="质量检查"):
        std, cv, cr = read_chi1_std(row["folder_path"])
        chi_stds.append(std)
        chi_valids.append(cv)
        chi_reasons.append(cr)

        xv, xr = read_xmu_valid(row["folder_path"])
        xmu_valids.append(xv)
        xmu_reasons.append(xr)

    df_work["chi_std"]     = chi_stds
    df_work["chi1_valid"]  = chi_valids
    df_work["chi1_reason"] = chi_reasons
    df_work["xmu_valid"]   = xmu_valids
    df_work["xmu_reason"]  = xmu_reasons

    n_total     = len(df_work)
    n_chi_bad   = int((~df_work["chi1_valid"]).sum())
    n_xmu_bad   = int((~df_work["xmu_valid"]).sum())
    n_both_good = int((df_work["chi1_valid"] & df_work["xmu_valid"]).sum())

    print("\n======= Step 1.2 质量过滤结果 =======")
    print(f"过滤前（三文件齐全）  : {n_total}")
    print(f"chi1 无效             : {n_chi_bad}")
    print(f"xmu  无效             : {n_xmu_bad}")
    print(f"两者均有效（过滤后）  : {n_both_good}")

    df_work.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n输出文件 → {OUTPUT_CSV}")

    # 分布图（英文标签，避免中文字体警告）
    all_stds = df_work.loc[df_work["chi_std"] > 0, "chi_std"].values
    plt.figure(figsize=(8, 4))
    plt.hist(all_stds, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    plt.axvline(1e-6, color="red", linewidth=1.5, label="threshold=1e-6")
    plt.xlabel("chi1 std")
    plt.ylabel("count")
    plt.title("chi1.dat column-2 std distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"distribution plot -> {OUTPUT_PNG}")

    bad_chi = df_work[~df_work["chi1_valid"]]["chi1_reason"].value_counts()
    bad_xmu = df_work[~df_work["xmu_valid"]]["xmu_reason"].value_counts()
    if len(bad_chi):
        print("\nchi1 invalid reasons:")
        print(bad_chi.to_string())
    if len(bad_xmu):
        print("\nxmu invalid reasons:")
        print(bad_xmu.to_string())

if __name__ == "__main__":
    main()