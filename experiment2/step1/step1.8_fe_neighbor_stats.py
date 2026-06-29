# step1.8_fe_neighbor_stats.py
# 对 200 个抽样结构，统计 Fe 中心周围近邻原子数量和距离分布
# 用于确定局部结构截取的 N 值

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

STEP1_DIR   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
SITE_MAP    = os.path.join(STEP1_DIR, "selected_site_map.csv")
OUTPUT_PNG  = os.path.join(STEP1_DIR, "fe_neighbor_distribution.png")
OUTPUT_CSV  = os.path.join(STEP1_DIR, "fe_neighbor_stats.csv")

SAMPLE_N    = 200
RANDOM_SEED = 42
MAX_CUTOFF  = 4.0    # Å，统计范围上限
N_CANDIDATE = [12, 14, 16, 18, 20, 22, 24]  # 候选 N 值

def get_fe_site_index(structure, site_nn: str):
    """
    从超胞中找到 LVSI 对应的 Fe 位点。
    site_nn 是文件夹名里的序号（如 "02" 表示第2个 Fe 位点，1-indexed）。
    """
    fe_indices = [i for i, s in enumerate(structure) if s.specie.symbol == "Fe"]
    idx = int(site_nn) - 1   # 转 0-indexed
    if idx < len(fe_indices):
        return fe_indices[idx]
    return fe_indices[0]     # fallback：取第一个 Fe

def analyze_neighbors(folder_path: str, site_nn: str, cutoff: float = MAX_CUTOFF):
    """
    返回 Fe 中心到各近邻的距离列表（Å），以及各距离壳层的累计原子数
    """
    fpath = Path(folder_path) / "POSCAR_supercell_fixed"
    try:
        structure = Structure.from_file(str(fpath))
        fe_idx = get_fe_site_index(structure, site_nn)
        fe_site = structure[fe_idx]

        # 获取 cutoff 范围内所有近邻（不含自身）
        neighbors = structure.get_neighbors(fe_site, r=cutoff)
        distances = sorted([n.nn_distance for n in neighbors])
        return distances, None
    except Exception as e:
        return None, str(e)[:80]

def main():
    df = pd.read_csv(SITE_MAP, encoding="utf-8-sig")
    print(f"selected_site_map 行数 : {len(df)}")

    random.seed(RANDOM_SEED)
    sample_idx = random.sample(range(len(df)), min(SAMPLE_N, len(df)))
    df_sample  = df.iloc[sample_idx].reset_index(drop=True)

    all_distances  = []   # 所有样本的近邻距离（拍平）
    cumcount_at_N  = {n: [] for n in N_CANDIDATE}  # 每个样本在截断 N 处的实际距离
    n_neighbors_list = []  # 每个样本 4Å 内的近邻总数
    fail_count = 0

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="近邻统计"):
        dists, err = analyze_neighbors(row["folder_path"], str(row["site_nn"]))
        if dists is None:
            fail_count += 1
            continue

        all_distances.extend(dists)
        n_neighbors_list.append(len(dists))

        # 对每个候选 N：第 N 个近邻的距离（截断距离）
        for n in N_CANDIDATE:
            if len(dists) >= n:
                cumcount_at_N[n].append(dists[n-1])   # 第 n 个近邻的距离
            # 若近邻数不足 N，记录 None（后续统计缺失率）

    print(f"\n统计成功：{len(n_neighbors_list)}  |  失败：{fail_count}")

    # ── 8Å 内近邻总数统计 ──
    arr_total = np.array(n_neighbors_list)
    print(f"\n=== {MAX_CUTOFF}Å 内近邻原子数统计 ===")
    print(f"mean   : {arr_total.mean():.1f}")
    print(f"median : {np.median(arr_total):.1f}")
    print(f"min    : {arr_total.min()}")
    print(f"max    : {arr_total.max()}")
    print(f"< 18 的样本数 : {(arr_total < 18).sum()} ({(arr_total<18).mean()*100:.1f}%)")

    # ── 各候选 N 的截断距离统计 ──
    print(f"\n=== 各候选 N 对应的截断距离（第N个近邻的距离）===")
    print(f"{'N':>4}  {'可用样本':>8}  {'缺失率':>6}  {'mean(Å)':>8}  {'median(Å)':>9}  {'max(Å)':>7}")
    print("-" * 55)
    rows = []
    for n in N_CANDIDATE:
        dlist = cumcount_at_N[n]
        n_valid = len(dlist)
        n_missing = len(n_neighbors_list) - n_valid
        miss_rate = n_missing / len(n_neighbors_list) * 100
        if dlist:
            arr = np.array(dlist)
            print(f"{n:>4}  {n_valid:>8}  {miss_rate:>5.1f}%  {arr.mean():>8.3f}  {np.median(arr):>9.3f}  {arr.max():>7.3f}")
            rows.append({"N": n, "n_valid": n_valid, "miss_rate_pct": miss_rate,
                         "cutoff_mean_A": arr.mean(), "cutoff_median_A": np.median(arr),
                         "cutoff_max_A": arr.max()})
        else:
            print(f"{n:>4}  {'N/A':>8}")

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n统计结果 → {OUTPUT_CSV}")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左：近邻距离整体分布（径向分布函数）
    arr_all = np.array(all_distances)
    axes[0].hist(arr_all, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].set_xlabel("Fe-neighbor distance (A)")
    axes[0].set_ylabel("count")
    axes[0].set_title("All neighbor distances (0-6A)")
    for cutoff_d in [3.0, 4.0, 5.0]:
        axes[0].axvline(cutoff_d, color="red", linewidth=1, linestyle="--", alpha=0.6)

    # 中：4Å 内近邻总数分布
    axes[1].hist(arr_total, bins=40, color="seagreen", edgecolor="none", alpha=0.85)
    for n in [18, 20, 24]:
        axes[1].axvline(n, color="red", linewidth=1.5, linestyle="--", label=f"N={n}")
    axes[1].set_xlabel("num neighbors within 6A")
    axes[1].set_ylabel("num compounds")
    axes[1].set_title("Neighbor count distribution")
    axes[1].legend(fontsize=8)

    # 右：各候选 N 的 median 截断距离
    ns     = [r["N"] for r in rows]
    meds   = [r["cutoff_median_A"] for r in rows]
    misses = [r["miss_rate_pct"] for r in rows]
    ax2 = axes[2]
    ax2.plot(ns, meds, "o-", color="steelblue", label="median cutoff (A)")
    ax2.axhline(5.0, color="gray", linewidth=1, linestyle="--", label="5A reference")
    ax2.set_xlabel("N (num neighbors)")
    ax2.set_ylabel("median cutoff distance (A)", color="steelblue")
    ax2.set_title("Cutoff distance vs N")
    ax3 = ax2.twinx()
    ax3.bar(ns, misses, alpha=0.3, color="orange", width=1.2, label="miss rate (%)")
    ax3.set_ylabel("miss rate (%)", color="orange")
    ax2.legend(loc="upper left", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"分布图 → {OUTPUT_PNG}")

if __name__ == "__main__":
    main()