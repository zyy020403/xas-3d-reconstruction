# pick_samples_for_feff_v2.py
# Exp5 v2 — 为师兄 FEFF 验证挑选样本(无元素限制 + POSCAR 输出版)
# ============================================================================
# 改动 vs pick_samples_for_feff.py(v1):
#   1. 取消元素白名单(全 88 元素 pool 都参与挑选)
#   2. 输出 POSCAR(VASP 5.x 格式,师兄要的格式)+ 保留 .xyz 作 cross-check
#   3. POSCAR 用虚拟正交 8×8×8 Å box,中心原子在 (4, 4, 4),邻居围绕
#      —— 防 box 边界效应,FEFF 直接读 cartesian 部分即可
#
# 用法:
#   /home/tcat/conda_envs/mlff/bin/python pick_samples_for_feff_v2.py
#   /home/tcat/conda_envs/mlff/bin/python pick_samples_for_feff_v2.py --n 20 --min-dist 1.5
# ============================================================================

import argparse
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch

from pymatgen.core import Element

warnings.filterwarnings("ignore")


# ─── Paths ───────────────────────────────────────────────────────────────────
EXP5_ROOT  = "/home/tcat/diffcsp_exp5"
EXP4_ROOT  = "/home/tcat/diffcsp_exp4"
STEP5_DIR  = os.path.join(EXP5_ROOT, "code", "step5")
STEP6_DIR  = os.path.join(EXP5_ROOT, "code", "step6")
OUT_DIR    = os.path.join(STEP6_DIR, "feff_candidates")
os.makedirs(OUT_DIR, exist_ok=True)

PT_VAL      = os.path.join(STEP5_DIR, "predictions_v2_val.pt")
PT_TEST     = os.path.join(STEP5_DIR, "predictions_v2_test.pt")
INVENTORY   = os.path.join(EXP4_ROOT, "data", "data_inventory_v2.csv")


# ─── Constants ───────────────────────────────────────────────────────────────
L_DATASET = 6.0   # dataset 用的虚拟分数坐标 box(从 frac→cart 用)
L_POSCAR  = 8.0   # POSCAR 输出用的虚拟正交 box(留 buffer 防边界)
CENTER_OFFSET = L_POSCAR / 2  # 中心原子在 (4, 4, 4)


def z_to_symbol(z):
    try:
        return Element.from_Z(int(z)).symbol
    except Exception:
        return "X"


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def min_pairwise_distance(frac_coords: np.ndarray, L: float = L_DATASET) -> float:
    """Min pairwise distance in Å, including center at origin."""
    f = frac_coords - np.round(frac_coords)
    cart = f * L  # (N, 3)
    all_pts = np.vstack([np.zeros((1, 3)), cart])  # (N+1, 3) including center

    n = all_pts.shape[0]
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(all_pts[i] - all_pts[j])
            if d < min_d:
                min_d = d
    return min_d


def write_xyz(path: str, center_elem: str,
              neighbor_frac: np.ndarray, neighbor_types: np.ndarray,
              comment: str = ""):
    """XYZ output (center at origin)."""
    f = neighbor_frac - np.round(neighbor_frac)
    cart = f * L_DATASET  # (N, 3) in Å

    n_total = 1 + cart.shape[0]
    with open(path, "w") as fo:
        fo.write(f"{n_total}\n")
        fo.write(f"{comment}\n")
        fo.write(f"{center_elem:<4s}  {0.0:12.6f}  {0.0:12.6f}  {0.0:12.6f}\n")
        for i in range(cart.shape[0]):
            sym = z_to_symbol(int(neighbor_types[i]))
            fo.write(f"{sym:<4s}  {cart[i, 0]:12.6f}  {cart[i, 1]:12.6f}  {cart[i, 2]:12.6f}\n")


def write_poscar(path: str, center_elem: str,
                 neighbor_frac: np.ndarray, neighbor_types: np.ndarray,
                 comment: str = ""):
    """
    Write POSCAR (VASP 5.x format).

    Layout:
      line 1: comment
      line 2: scale 1.0
      line 3-5: lattice vectors (orthogonal 8×8×8 Å box)
      line 6:   element symbols (sorted, with center listed first)
      line 7:   atom counts per element
      line 8:   "Cartesian"
      line 9+:  cartesian coords (Å, center at (4,4,4), neighbors offset)
    """
    # Get cartesian coords for neighbors (center at origin reference)
    f = neighbor_frac - np.round(neighbor_frac)
    nbr_cart = f * L_DATASET  # (N, 3)

    # Shift everything so center is at (4, 4, 4) in the 8×8×8 box
    center_cart = np.array([CENTER_OFFSET, CENTER_OFFSET, CENTER_OFFSET])
    nbr_cart_shifted = nbr_cart + center_cart  # neighbors around (4,4,4)

    # Group atoms by element symbol — VASP requires same-element atoms together
    nbr_symbols = [z_to_symbol(int(z)) for z in neighbor_types]

    # All atoms including center
    all_symbols = [center_elem] + nbr_symbols
    all_coords  = np.vstack([center_cart[None, :], nbr_cart_shifted])

    # Group by symbol, preserving an order: center element first, then by Z ascending
    # (VASP doesn't care about order, but consistent ordering helps readability)
    counts = Counter(all_symbols)
    # Order: center element first, then others by Z
    other_symbols = sorted([s for s in counts if s != center_elem],
                           key=lambda s: Element(s).Z)
    elem_order = [center_elem] + other_symbols

    # Indices grouped by element
    indices_by_elem = {sym: [i for i, s in enumerate(all_symbols) if s == sym]
                       for sym in elem_order}

    # Write
    with open(path, "w") as fo:
        fo.write(f"{comment}\n")
        fo.write("1.0\n")
        # Orthogonal box
        fo.write(f"  {L_POSCAR:12.6f}    0.000000    0.000000\n")
        fo.write(f"     0.000000  {L_POSCAR:12.6f}    0.000000\n")
        fo.write(f"     0.000000    0.000000  {L_POSCAR:12.6f}\n")
        # Element labels
        fo.write("  " + "  ".join(elem_order) + "\n")
        # Counts
        fo.write("  " + "  ".join(str(counts[s]) for s in elem_order) + "\n")
        # Coordinate type
        fo.write("Cartesian\n")
        # Coordinates, grouped by element
        for sym in elem_order:
            for idx in indices_by_elem[sym]:
                c = all_coords[idx]
                fo.write(f"  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-dist", type=float, default=1.5)
    ap.add_argument("--use-test", action="store_true", default=True,
                    help="use both val + test (default: True for v2)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"FEFF candidate picker v2 — NO element restriction + POSCAR output")
    print(f"  target N        : {args.n}")
    print(f"  min pairwise    : {args.min_dist} Å")
    print(f"  splits          : val + test")
    print(f"  POSCAR box      : {L_POSCAR}×{L_POSCAR}×{L_POSCAR} Å (center at "
          f"({CENTER_OFFSET}, {CENTER_OFFSET}, {CENTER_OFFSET}))")
    print(f"  output dir      : {OUT_DIR}")
    print("=" * 78)

    print("\n[1/4] Loading inventory ...")
    inv = pd.read_csv(INVENTORY, usecols=["sample_name", "center_element"])
    print(f"  inventory rows: {len(inv)}")

    print("\n[2/4] Loading v2 predictions ...")
    all_records = []
    for split, path in [("val", PT_VAL), ("test", PT_TEST)]:
        if not os.path.exists(path):
            continue
        preds = torch.load(path, map_location="cpu", weights_only=False)
        n = len(preds["sample_name"])
        print(f"  loaded {split}: N={n}")
        for i in range(n):
            sn = preds["sample_name"][i]
            pf = _to_np(preds["pred_frac_coords"][i])
            pt = _to_np(preds["pred_atom_types"][i])
            tf = _to_np(preds["true_frac_coords"][i])
            tt = _to_np(preds["true_atom_types"][i])
            if pf.shape[0] != 20 or tf.shape[0] != 20:
                continue
            all_records.append({
                "sample_name": sn,
                "mp_id":       preds["mp_id"][i],
                "split":       split,
                "pred_frac":   pf,
                "pred_types":  pt,
                "true_frac":   tf,
                "true_types":  tt,
            })
    print(f"  total candidate samples: {len(all_records)}")

    inv_map = inv.set_index("sample_name")["center_element"].to_dict()
    for rec in all_records:
        rec["center_element"] = inv_map.get(rec["sample_name"], "?")

    print(f"\n[3/4] Computing min pairwise distance + RMSD per record ...")
    from scipy.optimize import linear_sum_assignment

    for k, rec in enumerate(all_records):
        rec["min_pair_dist"] = min_pairwise_distance(rec["pred_frac"], L_DATASET)
        n = 20
        cost = np.zeros((n, n))
        for i in range(n):
            d = rec["pred_frac"][i] - rec["true_frac"]
            d -= np.round(d)
            cost[i] = np.linalg.norm(d * L_DATASET, axis=1)
        row, col = linear_sum_assignment(cost)
        sq = []
        for ri, ci in zip(row, col):
            d = rec["pred_frac"][ri] - rec["true_frac"][ci]
            d -= np.round(d)
            sq.append(np.sum((d * L_DATASET) ** 2))
        rec["rmsd"] = float(np.sqrt(np.mean(sq)))

        if (k + 1) % 1000 == 0:
            print(f"    [{k+1}/{len(all_records)}]")

    valid = [r for r in all_records if r["min_pair_dist"] >= args.min_dist]
    print(f"  after min-dist (≥{args.min_dist} Å) filter: "
          f"{len(valid)} / {len(all_records)}")

    if len(valid) < args.n:
        print(f"\n  ⚠️  Only {len(valid)} samples meet min-dist threshold "
              f"(asked for {args.n}). Will pick all {len(valid)}.")

    # Sort by RMSD ascending (best predictions first)
    valid_sorted = sorted(valid, key=lambda r: r["rmsd"])

    # Element diversity bonus: try to pick spread across center elements
    picked = []
    used_centers = Counter()
    target_per_element = max(1, args.n // 8)  # roughly 8 different elements

    # First pass: take 1-2 of each element until n reached
    for rec in valid_sorted:
        ce = rec["center_element"]
        if used_centers[ce] < target_per_element and len(picked) < args.n:
            picked.append(rec)
            used_centers[ce] += 1

    # Second pass: fill remaining with best RMSD regardless
    if len(picked) < args.n:
        already = set(r["sample_name"] for r in picked)
        for rec in valid_sorted:
            if rec["sample_name"] not in already and len(picked) < args.n:
                picked.append(rec)

    print(f"\n  picked {len(picked)} samples:")
    for rec in picked:
        print(f"    [{rec['split']:4s}] {rec['sample_name']:42s} "
              f"center={rec['center_element']:3s}  "
              f"RMSD={rec['rmsd']:.3f}  "
              f"min_pair_dist={rec['min_pair_dist']:.3f} Å")

    # ─── Write POSCAR + xyz + manifest ──────────────────────────────────
    print(f"\n[4/4] Writing POSCAR + xyz + manifest to {OUT_DIR} ...")
    manifest_rows = []
    for rec in picked:
        sn = rec["sample_name"].replace("/", "_").replace(":", "_")
        ce = rec["center_element"]
        center_z = Element(ce).Z

        pred_xyz_path    = os.path.join(OUT_DIR, f"{sn}_pred.xyz")
        true_xyz_path    = os.path.join(OUT_DIR, f"{sn}_true.xyz")
        pred_poscar_path = os.path.join(OUT_DIR, f"{sn}_pred_POSCAR")
        true_poscar_path = os.path.join(OUT_DIR, f"{sn}_true_POSCAR")

        comment_pred = (f"Exp5_v2_PREDICTED  sample={rec['sample_name']}  "
                        f"center={ce}  RMSD={rec['rmsd']:.3f}A  "
                        f"min_pair={rec['min_pair_dist']:.3f}A")
        comment_true = (f"Exp5_v2_TRUE  sample={rec['sample_name']}  center={ce}")

        write_xyz(pred_xyz_path, ce, rec["pred_frac"], rec["pred_types"],
                  comment=comment_pred)
        write_xyz(true_xyz_path, ce, rec["true_frac"], rec["true_types"],
                  comment=comment_true)
        write_poscar(pred_poscar_path, ce, rec["pred_frac"], rec["pred_types"],
                     comment=comment_pred)
        write_poscar(true_poscar_path, ce, rec["true_frac"], rec["true_types"],
                     comment=comment_true)

        manifest_rows.append({
            "sample_name":     rec["sample_name"],
            "mp_id":           rec["mp_id"],
            "split":           rec["split"],
            "center_element":  ce,
            "center_z":        center_z,
            "rmsd":            rec["rmsd"],
            "min_pair_dist":   rec["min_pair_dist"],
            "pred_POSCAR":     pred_poscar_path,
            "true_POSCAR":     true_poscar_path,
            "pred_xyz":        pred_xyz_path,
            "true_xyz":        true_xyz_path,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\n  manifest:    {manifest_path}")
    print(f"  POSCAR pairs: {len(picked)} (pred + true = {len(picked)*2} files)")
    print(f"  xyz pairs:    {len(picked)} (pred + true = {len(picked)*2} files)")

    print(f"\n  picked by element:")
    print(manifest_df["center_element"].value_counts().to_string())

    print("\n" + "=" * 78)
    print(f"DONE. Send {OUT_DIR} to your senior for FEFF computation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
