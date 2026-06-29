"""
Step 2.5 Phase D — Brute-force neighbor finder sanity test
============================================================

Goal: verify a pure-numpy periodic neighbor finder gives results matching
pymatgen's get_neighbors at r=10 (Phase A's working path), so we can use
it as a drop-in replacement at smaller r where pymatgen's Cython path is
hitting a Windows buffer-dtype bug.

Tests on 5 multi-site samples:
  (a) pymatgen.get_neighbors(r=10.0)        — Phase A used this, should work
  (b) brute_force(r=10.0)
  (c) compare (a) vs (b): same count + same sorted distances within tol
  (d) pymatgen.get_neighbors(r=shell1_outer+0.05)  — expected to fail
  (e) brute_force(r=shell1_outer+0.05)

If (c) passes 5/5 → brute-force is correct
If (d) fails 5/5  → confirms the env-specific bug, justifying brute-force fallback

Run
---
python step2_5d_test_brute.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
INV_PATH  = os.path.join(EXP4_ROOT, "step1", "data_inventory.csv")
NBR_PATH  = os.path.join(EXP4_ROOT, "step2_5", "step2_5_neighbor_distances.pkl")
SHL_PATH  = os.path.join(EXP4_ROOT, "step2_5", "shell_boundaries.pkl")


# ---------------------------------------------------------------------------
# Pure-numpy periodic neighbor finder
# ---------------------------------------------------------------------------
def find_neighbors_brute(cart_coords, species_Z, lattice_matrix,
                         center_idx, r):
    """
    Pure-numpy periodic neighbor finder.

    Returns sorted list of (Z, distance) tuples for atoms within radius r
    of cart_coords[center_idx], excluding the center atom itself, including
    periodic images out to the necessary number of cells.

    cart_coords    : (n_sites, 3) float
    species_Z      : (n_sites,)  int
    lattice_matrix : (3, 3) float, rows = lattice vectors a, b, c
    center_idx     : int, index into cart_coords
    r              : float, cutoff radius
    """
    a = lattice_matrix[0]
    b = lattice_matrix[1]
    c = lattice_matrix[2]

    # Perpendicular distance from origin to each face plane.
    # = volume / face_area.  Tells us how thick one cell is along each axis.
    vol = abs(np.dot(a, np.cross(b, c)))
    perp_a = vol / np.linalg.norm(np.cross(b, c))
    perp_b = vol / np.linalg.norm(np.cross(a, c))
    perp_c = vol / np.linalg.norm(np.cross(a, b))

    # +1 safety margin to handle numerical edge cases at boundaries
    n_a = int(np.ceil(r / perp_a)) + 1
    n_b = int(np.ceil(r / perp_b)) + 1
    n_c = int(np.ceil(r / perp_c)) + 1

    site_coord = cart_coords[center_idx]

    ia = np.arange(-n_a, n_a + 1)
    ib = np.arange(-n_b, n_b + 1)
    ic = np.arange(-n_c, n_c + 1)
    aa, bb, cc = np.meshgrid(ia, ib, ic, indexing="ij")
    aa = aa.ravel().astype(np.float64)
    bb = bb.ravel().astype(np.float64)
    cc = cc.ravel().astype(np.float64)

    # offsets: (n_images, 3)
    offsets = (aa[:, None] * a[None, :]
               + bb[:, None] * b[None, :]
               + cc[:, None] * c[None, :])

    # shifted - center: (n_images, n_sites, 3)
    diffs = cart_coords[None, :, :] + offsets[:, None, :] - site_coord[None, None, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))    # (n_images, n_sites)

    mask = (dists > 1e-8) & (dists <= r)

    # Pull matching site indices and distances
    out_dists = dists[mask]
    n_sites = cart_coords.shape[0]
    site_idx_grid = np.broadcast_to(np.arange(n_sites)[None, :], dists.shape)
    out_site_idx = site_idx_grid[mask]
    out_Zs = species_Z[out_site_idx]

    order = np.argsort(out_dists)
    return list(zip(out_Zs[order].tolist(), out_dists[order].tolist()))


# ---------------------------------------------------------------------------
# Helpers to extract lattice/coord arrays from pymatgen Structure
# ---------------------------------------------------------------------------
def extract_arrays(prim):
    cart = np.ascontiguousarray(prim.cart_coords, dtype=np.float64)
    Zs   = np.array([s.specie.Z for s in prim], dtype=np.int64)
    M    = np.ascontiguousarray(prim.lattice.matrix, dtype=np.float64)
    return cart, Zs, M


def pymatgen_neighbors_pairs(prim, center_idx, r):
    """Try pymatgen's get_neighbors and return sorted (Z, distance) pairs.
    Returns None if the call fails (caller decides what to do)."""
    try:
        nbrs = prim.get_neighbors(prim[center_idx], r=r)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    pairs = sorted([(int(n.specie.Z), float(n.nn_distance)) for n in nbrs],
                   key=lambda x: x[1])
    return pairs, None


def compare_pairs(pairs_a, pairs_b, tol=1e-3):
    """Two pair-lists 'match' if same length and pairwise same Z + close distance."""
    if len(pairs_a) != len(pairs_b):
        return False, f"length mismatch: {len(pairs_a)} vs {len(pairs_b)}"
    for i, ((Za, da), (Zb, db)) in enumerate(zip(pairs_a, pairs_b)):
        if Za != Zb:
            return False, f"Z mismatch at idx {i}: {Za} vs {Zb}"
        if abs(da - db) > tol:
            return False, f"distance mismatch at idx {i}: {da:.5f} vs {db:.5f}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Brute-force neighbor finder — sanity test")
    print("=" * 72)

    # Versions
    try:
        from importlib.metadata import version as _ver
        print(f"  numpy:    {_ver('numpy')}")
        print(f"  pymatgen: {_ver('pymatgen')}")
    except Exception as e:
        print(f"  version probe failed: {e}")

    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    inv = pd.read_csv(INV_PATH)
    with open(NBR_PATH, "rb") as f:
        neighbors = pickle.load(f)
    with open(SHL_PATH, "rb") as f:
        shells = pickle.load(f)

    # 5 multi-site samples (one each at n=2,4,8,16 + first multi-site O)
    inv["n_center_sites"] = inv["sample_name"].map(
        lambda s: neighbors.get(s, {}).get("n_center_sites", 0)
    )
    targets = []
    for n in [2, 4, 8, 16]:
        c = inv[inv["n_center_sites"] == n]
        if len(c) > 0:
            targets.append(c.iloc[0].to_dict())
    co = inv[(inv["center_element"] == "O") & (inv["n_center_sites"] >= 2)]
    if len(co) > 0:
        targets.append(co.iloc[0].to_dict())

    print(f"\nTargets: {len(targets)}")
    for t in targets:
        print(f"  {t['sample_name']:<55s} center={t['center_element']:<3s} sites={t['n_center_sites']}")
    print()

    R_VALIDATE = 10.0
    summary = {"r10_match": 0, "r10_total": 0,
               "rsmall_pmg_fail": 0, "rsmall_total": 0,
               "rsmall_brute_ok": 0}

    for t in targets:
        sname = t["sample_name"]
        center_el = t["center_element"]
        print("-" * 72)
        print(f"SAMPLE: {sname}  center={center_el}")
        print("-" * 72)

        try:
            s_super = Structure.from_file(t["poscar_path"])
            prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()
        except Exception as e:
            print(f"  primitive failed: {e}")
            continue

        cart, Zs, M = extract_arrays(prim)
        print(f"  primitive: {len(prim)} sites; lattice |a|={np.linalg.norm(M[0]):.3f}, "
              f"|b|={np.linalg.norm(M[1]):.3f}, |c|={np.linalg.norm(M[2]):.3f}")

        center_indices = [i for i, site in enumerate(prim)
                          if site.specie.symbol == center_el]
        ci = center_indices[0]
        print(f"  center_indices: {center_indices}, using ci={ci}")

        # ------ test (a) pymatgen at r=10 ------
        pmg_pairs_10, pmg_err_10 = pymatgen_neighbors_pairs(prim, ci, R_VALIDATE)
        if pmg_err_10:
            print(f"  pymatgen r={R_VALIDATE}: FAILED  ({pmg_err_10})")
        else:
            print(f"  pymatgen r={R_VALIDATE}: {len(pmg_pairs_10)} neighbors, "
                  f"first dist={pmg_pairs_10[0][1]:.4f}, last={pmg_pairs_10[-1][1]:.4f}")

        # ------ test (b) brute at r=10 ------
        brute_pairs_10 = find_neighbors_brute(cart, Zs, M, ci, R_VALIDATE)
        print(f"  brute    r={R_VALIDATE}: {len(brute_pairs_10)} neighbors, "
              f"first dist={brute_pairs_10[0][1]:.4f}, last={brute_pairs_10[-1][1]:.4f}")

        # ------ test (c) compare ------
        if pmg_err_10:
            print(f"  comparison: SKIP (pymatgen r=10 failed)")
        else:
            ok, msg = compare_pairs(pmg_pairs_10, brute_pairs_10, tol=1e-3)
            summary["r10_total"] += 1
            if ok:
                summary["r10_match"] += 1
                print(f"  comparison r={R_VALIDATE}: ✓ MATCH ({len(pmg_pairs_10)} pairs)")
            else:
                print(f"  comparison r={R_VALIDATE}: ✗ MISMATCH — {msg}")
                # diagnostic
                print(f"    pmg first 5:   {pmg_pairs_10[:5]}")
                print(f"    brute first 5: {brute_pairs_10[:5]}")

        # ------ test (d) pymatgen at small r ------
        shell1_outer = float(shells[sname]["shell_ends"][0])
        r_small = shell1_outer + 0.05
        pmg_pairs_s, pmg_err_s = pymatgen_neighbors_pairs(prim, ci, r_small)
        summary["rsmall_total"] += 1
        if pmg_err_s:
            summary["rsmall_pmg_fail"] += 1
            print(f"  pymatgen r={r_small:.4f}: FAILED  ({pmg_err_s[:80]})")
        else:
            print(f"  pymatgen r={r_small:.4f}: OK with {len(pmg_pairs_s)} neighbors")

        # ------ test (e) brute at small r ------
        brute_pairs_s = find_neighbors_brute(cart, Zs, M, ci, r_small)
        summary["rsmall_brute_ok"] += 1
        print(f"  brute    r={r_small:.4f}: {len(brute_pairs_s)} neighbors  "
              f"{brute_pairs_s[:6] if brute_pairs_s else '(empty)'}")
        print()

    # ---------- Summary ----------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  r=10 pymatgen-vs-brute match: "
          f"{summary['r10_match']} / {summary['r10_total']}")
    print(f"  r=small pymatgen failures:    "
          f"{summary['rsmall_pmg_fail']} / {summary['rsmall_total']}")
    print(f"  r=small brute success:        "
          f"{summary['rsmall_brute_ok']} / {summary['rsmall_total']}")
    print()
    if summary["r10_match"] == summary["r10_total"] and summary["r10_total"] > 0:
        print("  ✓ Brute-force implementation verified against pymatgen at r=10")
        if summary["rsmall_pmg_fail"] > 0:
            print("  ✓ Confirms env-specific bug at small r — proceed with v2 (brute fallback)")
        else:
            print("  ⚠ pymatgen at small r unexpectedly OK — investigate before v2")
    else:
        print("  ✗ Brute-force does NOT match pymatgen at r=10 — DO NOT proceed with v2")
        print("    Likely an off-by-one in the image-count formula or coordinate convention.")


if __name__ == "__main__":
    main()
