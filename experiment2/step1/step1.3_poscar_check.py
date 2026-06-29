# step1.3_poscar_check.py
# 输入：step1_quality_filter.csv（17464 条有效记录）
# 任务：全量 POSCAR 可解析性检查 + 抽样 300 个做原胞转换统计
# 输出：step1_poscar_check.csv

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── 路径常量 ──────────────────────────────────────────────────────────────────
STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
INPUT_CSV  = os.path.join(STEP1_DIR, "step1_quality_filter.csv")
OUTPUT_CSV = os.path.join(STEP1_DIR, "step1_poscar_check.csv")

SAMPLE_N   = 300
RANDOM_SEED = 42

# ── POSCAR 可解析性检查（不做原胞转换，只确认能读）──────────────────────────
def check_poscar_readable(folder_path: str):
    """返回 (valid: bool, n_atoms: int, reason: str)"""
    fpath = Path(folder_path) / "POSCAR_supercell_fixed"
    try:
        s = Structure.from_file(str(fpath))
        return True, s.num_sites, "ok"
    except Exception as e:
        return False, -1, str(e)[:80]

# ── 原胞转换（仅抽样用）────────────────────────────────────────────────────
def try_get_primitive(folder_path: str, symprec: float = 0.1):
    """返回 (prim_n_atoms: int, reason: str)"""
    fpath = Path(folder_path) / "POSCAR_supercell_fixed"
    try:
        structure = Structure.from_file(str(fpath))
        analyzer  = SpacegroupAnalyzer(structure, symprec=symprec)
        primitive = analyzer.get_primitive_standard_structure()
        return primitive.num_sites, "ok"
    except Exception as e:
        return -1, str(e)[:80]

# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # 只处理 chi1+xmu 均有效的记录
    mask = df["chi1_valid"] & df["xmu_valid"]
    df_work = df[mask].copy()
    print(f"输入有效记录数：{len(df_work)}")

    # ── 全量 POSCAR 可读性检查 ──
    poscar_valids  = []
    poscar_natoms  = []
    poscar_reasons = []

    for _, row in tqdm(df_work.iterrows(), total=len(df_work), desc="POSCAR 可读性"):
        v, n, r = check_poscar_readable(row["folder_path"])
        poscar_valids.append(v)
        poscar_natoms.append(n)
        poscar_reasons.append(r)

    df_work["poscar_valid"]        = poscar_valids
    df_work["supercell_n_atoms"]   = poscar_natoms
    df_work["poscar_reason"]       = poscar_reasons

    n_poscar_bad = int((~df_work["poscar_valid"]).sum())
    n_poscar_ok  = int(df_work["poscar_valid"].sum())
    print(f"\nPOSCAR 可读：{n_poscar_ok}  |  不可读：{n_poscar_bad}")

    # ── 抽样 300 个做原胞转换 ──
    df_valid = df_work[df_work["poscar_valid"]].copy()
    sample_n = min(SAMPLE_N, len(df_valid))

    random.seed(RANDOM_SEED)
    sample_idx = random.sample(list(df_valid.index), sample_n)
    df_sample  = df_valid.loc[sample_idx]

    prim_atoms_list = []
    prim_fail_count = 0

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="原胞转换抽样"):
        n_prim, reason = try_get_primitive(row["folder_path"])
        prim_atoms_list.append(n_prim)
        if n_prim == -1:
            prim_fail_count += 1

    # 统计（排除转换失败的 -1）
    valid_prim = [x for x in prim_atoms_list if x > 0]

    print(f"\n======= Step 1.3 原胞转换抽样统计（N={sample_n}）=======")
    print(f"转换成功：{len(valid_prim)}  |  失败：{prim_fail_count}")
    if valid_prim:
        arr = np.array(valid_prim)
        print(f"原胞原子数  mean   : {arr.mean():.1f}")
        print(f"原胞原子数  median : {np.median(arr):.1f}")
        print(f"原胞原子数  min    : {arr.min()}")
        print(f"原胞原子数  max    : {arr.max()}")
        print(f"原胞原子数  95th pct : {np.percentile(arr, 95):.1f}")
        if np.percentile(arr, 95) > 40:
            print("\n⚠️  警告：95th percentile > 40，建议 Main Agent 考虑调整 symprec")
        else:
            print("\n✓  95th percentile ≤ 40，symprec=0.1 合适")

        # 原子数分布简表
        from collections import Counter
        cnt = Counter(valid_prim)
        print("\n原胞原子数分布（前15个最常见值）：")
        for n_at, freq in sorted(cnt.items(), key=lambda x: -x[1])[:15]:
            print(f"  {n_at:3d} 个原子 : {freq:4d} 个样本")

    # ── 保存 CSV ──
    # 全量记录加上 poscar 检查结果，抽样原胞数不写回（只统计用）
    df_work.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n输出文件 → {OUTPUT_CSV}")

    # 最终三项全有效数量
    n_all_valid = int((df_work["chi1_valid"] & df_work["xmu_valid"] & df_work["poscar_valid"]).sum())
    print(f"\n三项全有效（chi1 ∧ xmu ∧ poscar）：{n_all_valid}")

if __name__ == "__main__":
    main()