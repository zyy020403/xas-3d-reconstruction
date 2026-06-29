"""
Step 2.5 Phase D — Quick diagnostic
====================================

Goal: figure out the actual exception that caused 100% of multi-site
samples to be tagged 'neighbor_error' in step2_5d_full_multisite_tag.py.

Tests 5 multi-site samples (one each at n_center_sites = 2, 4, 8, 16,
plus an O sample) with full traceback logging.

Run
---
python step2_5d_diagnose.py
"""
import os
import pickle
import traceback

import pandas as pd

EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
INV_PATH  = os.path.join(EXP4_ROOT, "step1", "data_inventory.csv")
NBR_PATH  = os.path.join(EXP4_ROOT, "step2_5", "step2_5_neighbor_distances.pkl")
SHL_PATH  = os.path.join(EXP4_ROOT, "step2_5", "shell_boundaries.pkl")


def main():
    print("=" * 72)
    print("Phase D diagnostic — exposing the true exception")
    print("=" * 72)

    # Load
    inv = pd.read_csv(INV_PATH)
    with open(NBR_PATH, "rb") as f:
        neighbors = pickle.load(f)
    with open(SHL_PATH, "rb") as f:
        shells = pickle.load(f)

    # Build a lookup of n_center_sites
    inv["n_center_sites"] = inv["sample_name"].map(
        lambda s: neighbors.get(s, {}).get("n_center_sites", 0)
    )

    # Pick one sample per bucket (deterministic: first match)
    targets = []
    for n in [2, 4, 8, 16]:
        cand = inv[inv["n_center_sites"] == n]
        if len(cand) > 0:
            targets.append(cand.iloc[0].to_dict())
    # Plus first multi-site O sample
    cand_o = inv[(inv["center_element"] == "O") & (inv["n_center_sites"] >= 2)]
    if len(cand_o) > 0:
        targets.append(cand_o.iloc[0].to_dict())

    print(f"\nTesting {len(targets)} samples ...")
    for t in targets:
        print(f"  - {t['sample_name']:<55s} center={t['center_element']:<3s} sites={t['n_center_sites']}")
    print()

    # Now do the actual Phase D logic step-by-step on each, with full trace
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        from importlib.metadata import version as _ver
        print(f"pymatgen version: {_ver('pymatgen')}")
    except Exception as _e:
        print(f"pymatgen version: <unknown> ({type(_e).__name__})")
    print()

    for t in targets:
        sname  = t["sample_name"]
        center = t["center_element"]
        n_exp  = t["n_center_sites"]

        print("-" * 72)
        print(f"SAMPLE: {sname}  (center={center}, n_sites_expected={n_exp})")
        print("-" * 72)

        # 1) Load primitive
        try:
            s_super = Structure.from_file(t["poscar_path"])
            prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()
            print(f"  [1] primitive built: {len(prim)} sites")
        except Exception as e:
            print(f"  [1] primitive FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # 2) Find center indices
        center_indices = [i for i, site in enumerate(prim)
                          if site.specie.symbol == center]
        print(f"  [2] center_indices: {center_indices}  (count={len(center_indices)})")
        if len(center_indices) != n_exp:
            print(f"      ⚠ mismatch with Phase A ({n_exp})")

        # 3) Get shell1_outer from Phase B
        if sname not in shells:
            print(f"  [3] sample missing from shell_boundaries — SKIP")
            continue
        shell1_outer = float(shells[sname]["shell_ends"][0])
        r_query = shell1_outer + 0.05
        print(f"  [3] shell1_outer = {shell1_outer:.5f} Å, r_query = {r_query:.5f} Å")
        print(f"      type(shell1_outer)={type(shell1_outer).__name__}, type(r_query)={type(r_query).__name__}")

        # 4) Try get_neighbors on each site
        for ci in center_indices:
            try:
                site_obj = prim[ci]
                print(f"  [4] site {ci}: type={type(site_obj).__name__}, specie={site_obj.specie.symbol}")
            except Exception as e:
                print(f"  [4] site {ci}: prim[ci] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                break

            try:
                nbrs = prim.get_neighbors(site_obj, r=r_query)
                print(f"  [5] get_neighbors(site, r={r_query:.4f}): {len(nbrs)} returned")
                if len(nbrs) > 0:
                    n0 = nbrs[0]
                    print(f"      first nbr: type={type(n0).__name__}, "
                          f"dist={n0.nn_distance:.4f}, Z={n0.specie.Z}")
            except Exception as e:
                print(f"  [5] get_neighbors FAILED: {type(e).__name__}: {e}")
                print(f"      ↓↓↓ FULL TRACEBACK ↓↓↓")
                traceback.print_exc()
                print(f"      ↑↑↑")
                break

            # Try the list-comp + numpy block too
            try:
                from collections import Counter
                import numpy as np
                EPS_INCLUDE = 1e-4
                valid = [(n.nn_distance, int(n.specie.Z)) for n in nbrs
                         if n.nn_distance <= shell1_outer + EPS_INCLUDE]
                valid.sort(key=lambda x: x[0])
                ds = np.array([v[0] for v in valid], dtype=np.float32)
                Zs = [v[1] for v in valid]
                ms = frozenset(Counter(Zs).items())
                print(f"  [6] processed: {len(valid)} valid, ds={ds.tolist() if len(ds)<=10 else f'{len(ds)} elems'}, Zs={Zs}")
                print(f"      multiset = {ms}")
            except Exception as e:
                print(f"  [6] processing FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                break

            # only test first 2 sites to keep output short
            if center_indices.index(ci) >= 1:
                print(f"  ... (skipping remaining sites for brevity)")
                break

        print()


if __name__ == "__main__":
    main()
