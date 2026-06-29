# pick_samples_for_feff.py
# Exp5 v2 — 为师兄 FEFF 验证挑选 20 个样本 + 导出 .xyz
# ============================================================================
# 筛选规则:
#   1. 中心元素 ∈ {Fe, Co, Ni, Zn, Cu}(师兄推荐的第三周期过渡金属)
#   2. 任意两个预测原子间距 > 1.0 Å(物理合理性,防止重合)
#   3. 优先挑 RMSD 较低的(预测可信度高的)
#   4. 中心元素分布尽量均匀(每种元素 ~ 4 个,共 20 个)
#
# 输出:
#   /home/tcat/diffcsp_exp5/code/step6/feff_candidates/
#     ├── manifest.csv              (20 个样本元信息)
#     ├── {sample_name}_pred.xyz    (预测结构,师兄算 FEFF 用)
#     └── {sample_name}_true.xyz    (真实结构,作为对照参考)
#
# 用法:
#   /home/tcat/conda_envs/mlff/bin/python pick_samples_for_feff.py
#   /home/tcat/conda_envs/mlff/bin/python pick_samples_for_feff.py --n 30 --min-dist 1.0
# ============================================================================

import argparse
import os
import sys
import warnings

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
L = 6.0
TARGET_ELEMENTS = ["Fe", "Co", "Ni", "Zn", "Cu"]


def z_to_symbol(z):
    try:
        return Element.from_Z(int(z)).symbol
    except Exception:
        return "X"


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def min_pairwise_distance(frac_coords: np.ndarray, L: float = L) -> float:
    """Return min pairwise distance (Å) between any two atoms in the structure
    (with center, pre-min-image)."""
    # fold to [-0.5, 0.5]
    f = frac_coords - np.round(frac_coords)
    cart = f * L  # (N, 3)

    # Include the center atom (origin)
    all_pts = np.vstack([np.zeros((1, 3)), cart])  # (N+1, 3)

    n = all_pts.shape[0]
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(all_pts[i] - all_pts[j])
            if d < min_d:
                min_d = d
    return min_d


def write_xyz(path: str, center_z: int, center_elem: str,
              neighbor_frac: np.ndarray, neighbor_types: np.ndarray,
              comment: str = ""):
    """
    Write XYZ format:
      line 1: total atom count (1 center + N neighbors)
      line 2: comment
      line 3+: element  x  y  z   (Å, with center at origin)

    Center atom written first (at origin), then 20 neighbors.
    """
    f = neighbor_frac - np.round(neighbor_frac)
    cart = f * L  # (N, 3) in Å

    n_total = 1 + cart.shape[0]
    with open(path, "w") as fo:
        fo.write(f"{n_total}\n")
        fo.write(f"{comment}\n")
        # Center atom at origin
        fo.write(f"{center_elem:<4s}  {0.0:12.6f}  {0.0:12.6f}  {0.0:12.6f}\n")
        # Neighbors
        for i in range(cart.shape[0]):
            sym = z_to_symbol(int(neighbor_types[i]))
            fo.write(f"{sym:<4s}  {cart[i, 0]:12.6f}  {cart[i, 1]:12.6f}  {cart[i, 2]:12.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="total samples to pick (default: 20)")
    ap.add_argument("--min-dist", type=float, default=1.0,
                    help="minimum pairwise atom distance (Å,默认 1.0)")
    ap.add_argument("--per-element", type=int, default=4,
                    help="target samples per center element (default: 4 = 20/5)")
    ap.add_argument("--use-test", action="store_true",
                    help="also include test split (default: val only)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"FEFF candidate picker (Exp5 v2)")
    print(f"  target N        : {args.n}")
    print(f"  min pairwise    : {args.min_dist} Å")
    print(f"  per-element     : {args.per_element}")
    print(f"  target elements : {TARGET_ELEMENTS}")
    print(f"  splits          : {'val + test' if args.use_test else 'val only'}")
    print(f"  output dir      : {OUT_DIR}")
    print("=" * 78)

    # ─── Load inventory ─────────────────────────────────────────────────
    print("\n[1/4] Loading inventory ...")
    inv = pd.read_csv(INVENTORY, usecols=["sample_name", "center_element"])
    print(f"  inventory rows: {len(inv)}")

    # ─── Load predictions ───────────────────────────────────────────────
    print("\n[2/4] Loading v2 predictions ...")
    splits_to_use = ["val"]
    if args.use_test:
        splits_to_use.append("test")

    all_records = []
    for split in splits_to_use:
        path = PT_VAL if split == "val" else PT_TEST
        if not os.path.exists(path):
            print(f"  WARN: {path} missing")
            continue
        preds = torch.load(path, map_location="cpu", weights_only=False)
        n = len(preds["sample_name"])
        print(f"  loaded {split}: N={n}")

        for i in range(n):
            sn = preds["sample_name"][i]
            mpid = preds["mp_id"][i]
            pf = _to_np(preds["pred_frac_coords"][i])
            pt = _to_np(preds["pred_atom_types"][i])
            tf = _to_np(preds["true_frac_coords"][i])
            tt = _to_np(preds["true_atom_types"][i])
            if pf.shape[0] != 20 or tf.shape[0] != 20:
                continue
            all_records.append({
                "sample_name": sn,
                "mp_id":       mpid,
                "split":       split,
                "pred_frac":   pf,
                "pred_types":  pt,
                "true_frac":   tf,
                "true_types":  tt,
            })

    print(f"  total candidate samples: {len(all_records)}")

    # Map sample_name → center_element
    inv_map = inv.set_index("sample_name")["center_element"].to_dict()

    # ─── Filter by center element ───────────────────────────────────────
    print(f"\n[3/4] Filtering by center element ∈ {TARGET_ELEMENTS} ...")
    filtered = []
    for rec in all_records:
        ce = inv_map.get(rec["sample_name"])
        if ce in TARGET_ELEMENTS:
            rec["center_element"] = ce
            filtered.append(rec)
    print(f"  after element filter: {len(filtered)}")

    # ─── Compute min pairwise distance + RMSD per record ────────────────
    print("\n[3/4] Computing min pairwise distance + RMSD per record ...")
    from scipy.optimize import linear_sum_assignment

    for k, rec in enumerate(filtered):
        # Min pairwise distance (predicted structure)
        rec["min_pair_dist"] = min_pairwise_distance(rec["pred_frac"], L)

        # RMSD via Hungarian min-image
        n = 20
        cost = np.zeros((n, n))
        for i in range(n):
            d = rec["pred_frac"][i] - rec["true_frac"]
            d -= np.round(d)
            cost[i] = np.linalg.norm(d * L, axis=1)
        row, col = linear_sum_assignment(cost)
        sq = []
        for ri, ci in zip(row, col):
            d = rec["pred_frac"][ri] - rec["true_frac"][ci]
            d -= np.round(d)
            sq.append(np.sum((d * L) ** 2))
        rec["rmsd"] = float(np.sqrt(np.mean(sq)))

        if (k + 1) % 200 == 0:
            print(f"    [{k+1}/{len(filtered)}]")

    # ─── Apply min-dist filter + sort by RMSD per element ───────────────
    valid = [r for r in filtered if r["min_pair_dist"] >= args.min_dist]
    print(f"  after min-dist (≥{args.min_dist} Å) filter: {len(valid)} / {len(filtered)}")

    # Sort by RMSD ascending per center element + take per-element top N
    by_element = {}
    for rec in valid:
        ce = rec["center_element"]
        by_element.setdefault(ce, []).append(rec)

    print(f"\n  per-element availability:")
    for ce in TARGET_ELEMENTS:
        n_e = len(by_element.get(ce, []))
        print(f"    {ce}: {n_e} candidates")

    # Pick: try args.per_element per element, fill from the largest pool if shortfall
    picked = []
    for ce in TARGET_ELEMENTS:
        pool = sorted(by_element.get(ce, []), key=lambda r: r["rmsd"])
        picked.extend(pool[: args.per_element])

    # If short, fill from any remaining sorted by RMSD
    if len(picked) < args.n:
        already = set(r["sample_name"] for r in picked)
        rest = sorted([r for r in valid if r["sample_name"] not in already],
                      key=lambda r: r["rmsd"])
        need = args.n - len(picked)
        picked.extend(rest[:need])

    # If oversized, trim to args.n
    picked = picked[: args.n]

    print(f"\n  picked {len(picked)} samples:")
    for rec in picked:
        print(f"    [{rec['split']:4s}] {rec['sample_name']:42s} "
              f"center={rec['center_element']:3s}  "
              f"RMSD={rec['rmsd']:.3f}  "
              f"min_pair_dist={rec['min_pair_dist']:.3f} Å")

    # ─── Write outputs ──────────────────────────────────────────────────
    print(f"\n[4/4] Writing manifest + .xyz files to {OUT_DIR} ...")
    manifest_rows = []
    for rec in picked:
        sn = rec["sample_name"].replace("/", "_").replace(":", "_")
        ce = rec["center_element"]
        center_z = Element(ce).Z

        pred_path = os.path.join(OUT_DIR, f"{sn}_pred.xyz")
        true_path = os.path.join(OUT_DIR, f"{sn}_true.xyz")

        write_xyz(pred_path, center_z, ce,
                  rec["pred_frac"], rec["pred_types"],
                  comment=(f"Exp5 v2 PREDICTED structure  "
                           f"sample={rec['sample_name']}  "
                           f"mp_id={rec['mp_id']}  "
                           f"center={ce}  RMSD={rec['rmsd']:.3f}"))
        write_xyz(true_path, center_z, ce,
                  rec["true_frac"], rec["true_types"],
                  comment=(f"Exp5 v2 TRUE structure (reference)  "
                           f"sample={rec['sample_name']}  "
                           f"mp_id={rec['mp_id']}  "
                           f"center={ce}"))
        manifest_rows.append({
            "sample_name":    rec["sample_name"],
            "mp_id":          rec["mp_id"],
            "split":          rec["split"],
            "center_element": ce,
            "center_z":       center_z,
            "rmsd":           rec["rmsd"],
            "min_pair_dist":  rec["min_pair_dist"],
            "pred_xyz":       pred_path,
            "true_xyz":       true_path,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\n  manifest:  {manifest_path}")
    print(f"  xyz files: {len(picked)*2} files in {OUT_DIR}")

    # Per-element count summary
    print(f"\n  picked by element:")
    print(manifest_df["center_element"].value_counts().to_string())

    print("\n" + "=" * 78)
    print(f"DONE. Send {OUT_DIR} to your senior for FEFF computation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
