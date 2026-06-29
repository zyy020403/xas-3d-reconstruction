# =============================================================================
# 脚本编号: step4.2
# 脚本名称: step4.2_compute_metrics.py
# 输入:
#   - experiment/step4/predictions_val.pt
#   - experiment/step4/predictions_test.pt
#   - experiment/step1/data_inventory.csv  (含 quality_tier, is_ionic, n_sites 等)
#   - experiment/step1/bond_length_constraints.json
# 输出:
#   - experiment/step4/metrics_report.txt  (人类可读报告)
#   - experiment/step4/metrics_detail.csv  (每个 compound 的详细指标，供后续分析)
# 说明:
#   复用 compute_metrics.py 中的 Crystal 类和 RecEval 类进行结构匹配。
#   额外计算键长违规率（新增物理合理性指标）。
#   按 is_ionic / n_sites / quality_tier / 元素种类 四维度进行子群分析。
# =============================================================================

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP4_DIR  = os.path.join(EXPERIMENT_DIR, "step4")
STEP1_DIR  = os.path.join(EXPERIMENT_DIR, "step1")
STEP3_DIR  = os.path.join(EXPERIMENT_DIR, "step3")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STEP3_DIR)
sys.path.insert(0, SCRIPTS_DIR)

# ─── 导入 compute_metrics 中可复用的类 ──────────────────────────────────────
from compute_metrics import Crystal, RecEval

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

# ─── 路径 ────────────────────────────────────────────────────────────────────
PRED_VAL_PATH   = os.path.join(STEP4_DIR, "predictions_val.pt")
PRED_TEST_PATH  = os.path.join(STEP4_DIR, "predictions_test.pt")
INVENTORY_CSV   = os.path.join(STEP1_DIR, "data_inventory.csv")
BOND_JSON       = os.path.join(STEP1_DIR, "bond_length_constraints.json")
REPORT_PATH     = os.path.join(STEP4_DIR, "metrics_report.txt")
DETAIL_CSV_PATH = os.path.join(STEP4_DIR, "metrics_detail.csv")


# ─── 辅助：predictions dict → Crystal 对象列表 ──────────────────────────────
def preds_to_crystal_arrays(predictions: dict):
    """
    将 predictions dict 拆为两个有序列表：
    - mp_ids: list of str
    - pred_arrays: list of dict (frac_coords, atom_types, lengths, angles)
    - gt_arrays:   list of dict
    """
    mp_ids, pred_arrays, gt_arrays = [], [], []
    for mp_id, entry in predictions.items():
        mp_ids.append(mp_id)
        pred_arrays.append({
            "frac_coords": entry["pred_frac_coords"].numpy(),
            "atom_types":  entry["pred_atom_types"].numpy(),
            "lengths":     entry["pred_lengths"].numpy(),
            "angles":      entry["pred_angles"].numpy(),
        })
        gt_arrays.append({
            "frac_coords": entry["gt_frac_coords"].numpy(),
            "atom_types":  entry["gt_atom_types"].numpy(),
            "lengths":     entry["gt_lengths"].numpy(),
            "angles":      entry["gt_angles"].numpy(),
        })
    return mp_ids, pred_arrays, gt_arrays


# ─── 辅助：键长违规率 ─────────────────────────────────────────────────────────
def load_bond_constraints(json_path: str) -> dict:
    """
    加载键长约束字典。
    键格式: "Fe-O"，值: [min_Å, max_Å]
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] bond_length_constraints.json not found: {json_path}")
        return {}
    with open(json_path, "r") as f:
        raw = json.load(f)
    # 双向建表：Fe-O 和 O-Fe 都能查到
    constraints = {}
    for pair, bounds in raw.items():
        constraints[pair] = bounds
        parts = pair.split("-")
        if len(parts) == 2:
            rev = f"{parts[1]}-{parts[0]}"
            if rev not in constraints:
                constraints[rev] = bounds
    print(f"[Constraints] Loaded {len(raw)} pairs ({len(constraints)} with reverse).")
    return constraints


# chemical_symbols 用于原子序数 → 元素符号
chemical_symbols = [
    'X','H','He','Li','Be','B','C','N','O','F','Ne',
    'Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy',
    'Ho','Er','Tm','Yb','Lu',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
    'Po','At','Rn',
    'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk',
    'Cf','Es','Fm','Md','No','Lr',
]


def compute_bond_violation_rate(crystal_array: dict, constraints: dict) -> float:
    """
    对单个预测结构计算键长违规率。
    返回: 违规对数 / 所有近邻对数  (0.0 ~ 1.0)
    如果无约束数据或结构无效，返回 NaN。
    """
    if not constraints:
        return float("nan")

    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(crystal_array["lengths"].tolist() + crystal_array["angles"].tolist())
            ),
            species=crystal_array["atom_types"].tolist(),
            coords=crystal_array["frac_coords"],
            coords_are_cartesian=False,
        )
    except Exception:
        return float("nan")

    # 用 cutoff=4.0 Å 获取所有近邻对
    cutoff = 4.0
    try:
        all_neighbors = structure.get_all_neighbors(cutoff)
    except Exception:
        return float("nan")

    total_pairs   = 0
    violated_pairs = 0

    for site_idx, neighbors in enumerate(all_neighbors):
        center_elem = chemical_symbols[structure[site_idx].specie.Z]
        for nbr in neighbors:
            nbr_elem = chemical_symbols[nbr.specie.Z]
            dist     = nbr.nn_distance

            pair_key     = f"{center_elem}-{nbr_elem}"
            pair_key_rev = f"{nbr_elem}-{center_elem}"

            # 查约束（双向）
            bounds = constraints.get(pair_key) or constraints.get(pair_key_rev)
            if bounds is None:
                continue  # 无约束则跳过此对

            total_pairs += 1
            lo, hi = bounds[0], bounds[1]
            if dist < lo or dist > hi:
                violated_pairs += 1

    if total_pairs == 0:
        return float("nan")
    return violated_pairs / total_pairs


# ─── 辅助：子群统计 ───────────────────────────────────────────────────────────
def subgroup_stats(detail_df: pd.DataFrame, mask: pd.Series, label: str) -> str:
    sub = detail_df[mask]
    n   = len(sub)
    if n == 0:
        return f"  {label}: N=0 (无数据)"
    matched = sub["rms_dist"].notna()
    match_rate = matched.sum() / n
    mean_rms   = sub.loc[matched, "rms_dist"].mean() if matched.sum() > 0 else float("nan")
    viol_valid = sub["bond_violation_rate"].dropna()
    mean_viol  = viol_valid.mean() if len(viol_valid) > 0 else float("nan")
    return (
        f"  {label}: Match Rate = {match_rate:.1%}  "
        f"RMSE = {mean_rms:.4f} Å  "
        f"BondViol = {mean_viol:.1%}  "
        f"(N={n})"
    )


# ─── 核心评估函数 ─────────────────────────────────────────────────────────────
def evaluate_split(split_name: str, predictions: dict,
                   inventory_df: pd.DataFrame,
                   bond_constraints: dict) -> pd.DataFrame:
    """
    对一个 split（val 或 test）完整评估，返回 per-compound 详细 DataFrame。
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {split_name}  ({len(predictions)} compounds)")
    print(f"{'='*60}")

    mp_ids, pred_arrays, gt_arrays = preds_to_crystal_arrays(predictions)

    # ── 构建 Crystal 对象 ─────────────────────────────────────────────────────
    print("[1/4] Building Crystal objects ...")
    pred_crystals, gt_crystals = [], []
    build_errors = 0
    for i, (pa, ga) in enumerate(tqdm(
            zip(pred_arrays, gt_arrays), total=len(mp_ids), desc="Building")):
        try:
            pred_crystals.append(Crystal(pa))
        except Exception as e:
            print(f"  [ERROR] pred Crystal build failed for {mp_ids[i]}: {e}")
            pred_crystals.append(None)
            build_errors += 1
        try:
            gt_crystals.append(Crystal(ga))
        except Exception as e:
            print(f"  [ERROR] gt Crystal build failed for {mp_ids[i]}: {e}")
            gt_crystals.append(None)
            build_errors += 1
    if build_errors:
        print(f"  WARNING: {build_errors} Crystal build errors (will be marked as None).")

    # ── StructureMatcher 计算 rms_dist ────────────────────────────────────────
    print("[2/4] Running StructureMatcher ...")
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    rms_dists = []
    for pc, gc in tqdm(zip(pred_crystals, gt_crystals),
                       total=len(mp_ids), desc="Matching"):
        if pc is None or gc is None:
            rms_dists.append(None)
            continue
        if not (pc.constructed and gc.constructed):
            rms_dists.append(None)
            continue
        try:
            rms = matcher.get_rms_dist(pc.structure, gc.structure)
            rms_dists.append(rms[0] if rms is not None else None)
        except Exception:
            rms_dists.append(None)

    # ── 键长违规率 ─────────────────────────────────────────────────────────────
    print("[3/4] Computing bond length violation rates ...")
    violation_rates = []
    for pa in tqdm(pred_arrays, desc="BondViol"):
        violation_rates.append(compute_bond_violation_rate(pa, bond_constraints))

    # ── 整合 per-compound 数据表 ───────────────────────────────────────────────
    print("[4/4] Merging with inventory ...")
    detail_rows = []
    for i, mp_id in enumerate(mp_ids):
        row = {
            "mp_id":               mp_id,
            "split":               split_name,
            "rms_dist":            rms_dists[i],
            "matched":             rms_dists[i] is not None,
            "bond_violation_rate": violation_rates[i],
            "n_atoms":             pred_arrays[i]["frac_coords"].shape[0],
        }
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    detail_df["mp_id"] = detail_df["mp_id"].astype(str)

    # join inventory
    if inventory_df is not None and len(inventory_df) > 0:
        inv_cols = ["mp_id"]
        for col in ["is_ionic", "quality_tier", "n_sites", "element"]:
            if col in inventory_df.columns:
                inv_cols.append(col)
        inv_sub = inventory_df[inv_cols].drop_duplicates("mp_id")
        inv_sub["mp_id"] = inv_sub["mp_id"].astype(str)
        detail_df = detail_df.merge(inv_sub, on="mp_id", how="left")
        print(f"  Inventory join: {detail_df['mp_id'].notna().sum()} rows matched.")
    else:
        print("  WARNING: inventory_df is empty or None, skipping join.")

    return detail_df


# ─── 报告生成 ─────────────────────────────────────────────────────────────────
def generate_report(val_df: pd.DataFrame,
                    test_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("  Step 4 评估指标报告")
    lines.append("=" * 65)

    for split_name, df in [("Val", val_df), ("Test", test_df)]:
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {split_name} 集")
        lines.append(f"{'─'*60}")
        n = len(df)
        matched = df["rms_dist"].notna()
        match_rate = matched.sum() / n
        mean_rms   = df.loc[matched, "rms_dist"].mean() if matched.sum() > 0 else float("nan")
        viol_vals  = df["bond_violation_rate"].dropna()
        mean_viol  = viol_vals.mean() if len(viol_vals) > 0 else float("nan")

        lines.append(f"总样本数              : {n}")
        lines.append(f"Match Rate            : {match_rate:.1%}  ({matched.sum()}/{n})")
        lines.append(f"Mean RMSE (matched)   : {mean_rms:.4f} Å")
        lines.append(f"Bond Viol Rate (mean) : {mean_viol:.1%}")

        # ── is_ionic ──────────────────────────────────────────────────────────
        if "is_ionic" in df.columns:
            lines.append("\n  [is_ionic 分析]")
            lines.append(subgroup_stats(df, df["is_ionic"] == False, "纯共价 (is_ionic=False)"))
            lines.append(subgroup_stats(df, df["is_ionic"] == True,  "含 ionic (is_ionic=True)"))
        else:
            lines.append("\n  [is_ionic] 列不可用（inventory 未 join 成功）")

        # ── n_sites ───────────────────────────────────────────────────────────
        if "n_sites" in df.columns:
            lines.append("\n  [位点数分析]")
            lines.append(subgroup_stats(df, df["n_sites"] == 1,               "1 个位点"))
            lines.append(subgroup_stats(df, df["n_sites"].between(2, 3),      "2–3 个位点"))
            lines.append(subgroup_stats(df, df["n_sites"] >= 4,               "4+ 个位点"))
        else:
            lines.append("\n  [n_sites] 列不可用")

        # ── quality_tier ──────────────────────────────────────────────────────
        if "quality_tier" in df.columns:
            lines.append("\n  [质量分级分析]")
            for tier in ["A", "B", "C"]:
                lines.append(subgroup_stats(df, df["quality_tier"] == tier,
                                            f"Quality {tier}"))
        else:
            lines.append("\n  [quality_tier] 列不可用")

        # ── 元素种类 ──────────────────────────────────────────────────────────
        if "is_ionic" in df.columns and "element" in df.columns:
            lines.append("\n  [元素种类分析]")
            # 仅含 Fe，且 is_ionic=False
            fe_only_mask = (
                df["element"].str.upper().str.contains("FE", na=False) &
                (df["is_ionic"] == False)
            )
            lines.append(subgroup_stats(df, fe_only_mask, "含 Fe 且纯共价"))
            mixed_mask = (
                df["element"].str.upper().str.contains("FE", na=False) &
                (df["is_ionic"] == True)
            )
            lines.append(subgroup_stats(df, mixed_mask, "含 Fe 且含 ionic"))

        # ── 异常检测 ──────────────────────────────────────────────────────────
        lines.append("\n  [预测结构健康检查]")
        if "pred_lengths" in df.columns:
            pass  # 已在 sanity check 阶段处理
        # 检查 rms_dist 中极端异常值
        if matched.sum() > 0:
            max_rms = df.loc[matched, "rms_dist"].max()
            n_extreme = (df.loc[matched, "rms_dist"] > 1.0).sum()
            lines.append(f"  max RMSE (matched)         : {max_rms:.4f} Å")
            lines.append(f"  RMSE > 1.0 Å (matched)    : {n_extreme}")
        # 键长违规 > 50%
        n_high_viol = (df["bond_violation_rate"] > 0.5).sum()
        lines.append(f"  Bond Viol > 50% compounds : {n_high_viol}")

    lines.append("\n" + "=" * 65)
    lines.append("  报告生成完毕")
    lines.append("=" * 65)
    return "\n".join(lines)


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    # 加载键长约束
    bond_constraints = load_bond_constraints(BOND_JSON)

    # 加载 inventory
    print(f"[Inventory] Loading {INVENTORY_CSV} ...")
    if os.path.exists(INVENTORY_CSV):
        inventory_df = pd.read_csv(INVENTORY_CSV)
        print(f"  Loaded {len(inventory_df)} rows, columns: {list(inventory_df.columns)}")
    else:
        print(f"  WARNING: {INVENTORY_CSV} not found, subgroup analysis will be skipped.")
        inventory_df = None

    # 加载预测
    print(f"\n[Load] Loading predictions ...")
    val_preds  = torch.load(PRED_VAL_PATH,  map_location="cpu",
                            weights_only=False)
    test_preds = torch.load(PRED_TEST_PATH, map_location="cpu",
                            weights_only=False)
    print(f"  Val: {len(val_preds)} compounds")
    print(f"  Test: {len(test_preds)} compounds")

    # 评估
    val_df  = evaluate_split("val",  val_preds,  inventory_df, bond_constraints)
    test_df = evaluate_split("test", test_preds, inventory_df, bond_constraints)

    # 合并 detail CSV
    all_detail = pd.concat([val_df, test_df], ignore_index=True)
    all_detail.to_csv(DETAIL_CSV_PATH, index=False)
    print(f"\n[Saved] Detail CSV -> {DETAIL_CSV_PATH}")

    # 生成文字报告
    report = generate_report(val_df, test_df)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Saved] Report -> {REPORT_PATH}")

    # 快速控制台摘要
    for split_name, df in [("Val", val_df), ("Test", test_df)]:
        n = len(df)
        matched = df["rms_dist"].notna().sum()
        print(f"[SUMMARY] {split_name}: Match Rate = {matched/n:.1%}  "
              f"({matched}/{n})")


if __name__ == "__main__":
    main()