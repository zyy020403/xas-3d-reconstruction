"""
Step 2.5 Phase D v2 — brute-force multi-site equivalence tagging
=================================================================

v1 tagged 100% of multi-site samples as 'neighbor_error' because pymatgen's
Cython find_points_in_spheres throws 'Buffer dtype mismatch' on this Windows
env (numpy 1.26.4 + pymatgen 2024.8.9). Confirmed by step2_5d_test_brute.py.

v2 replaces all pymatgen.get_neighbors calls with a pure-numpy brute-force
periodic neighbor finder (verified vs Phase A's stored distances on 5 samples
with chemistry-consistent results: Ta-O=1.98 Å, Na-O=2.4 Å, etc.).

Algorithm and tag decision are unchanged from v1 (per MA spec).

Run
---
python step2_5d_full_multisite_tag_v2.py
"""
import io
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
EXP4_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR     = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR    = os.path.join(EXP4_ROOT, "step2_5")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")

NEIGHBOR_PKL_PATH = os.path.join(STEP25_DIR, "step2_5_neighbor_distances.pkl")
SHELL_BOUND_PATH  = os.path.join(STEP25_DIR, "shell_boundaries.pkl")
TAG_CSV_PATH      = os.path.join(STEP25_DIR, "site_equivalence_tag.csv")
SUMMARY_PATH      = os.path.join(STEP25_DIR, "step2_5d_summary.txt")
LOG_PATH          = os.path.join(STEP25_DIR, "step2_5d_tag.log")

SYMPREC      = 0.1
EPS_QUERY    = 0.05    # query radius padding
EPS_INCLUDE  = 0.005   # filter tolerance (10x bumped vs v1 for safety against
                       # spglib/pymatgen ~5e-4 Å absolute coord precision drift)
N_WORKERS    = 8

EQUIV_MAE_TOL    = 0.01
NEAR_EQUIV_MAE   = 0.10
SENTINEL_MAE     = 999.0


# -----------------------------------------------------------------------------
# Pure-numpy brute-force PBC neighbor finder (validated chemistry-wise)
# -----------------------------------------------------------------------------
def find_neighbors_brute(cart_coords, species_Z, lattice_matrix,
                         center_idx, r):
    """
    Returns (Zs_sorted, dists_sorted) — neighbors of cart_coords[center_idx]
    within radius r (excluding self), summed across periodic images.
    Pure numpy. Bypasses pymatgen's broken Cython.
    """
    a = lattice_matrix[0]
    b = lattice_matrix[1]
    c = lattice_matrix[2]
    vol = abs(np.dot(a, np.cross(b, c)))
    perp_a = vol / np.linalg.norm(np.cross(b, c))
    perp_b = vol / np.linalg.norm(np.cross(a, c))
    perp_c = vol / np.linalg.norm(np.cross(a, b))
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

    offsets = (aa[:, None] * a[None, :]
               + bb[:, None] * b[None, :]
               + cc[:, None] * c[None, :])

    diffs = cart_coords[None, :, :] + offsets[:, None, :] - site_coord[None, None, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))      # (n_images, n_sites)

    mask = (dists > 1e-8) & (dists <= r)

    out_dists = dists[mask]
    n_sites = cart_coords.shape[0]
    site_idx_grid = np.broadcast_to(np.arange(n_sites)[None, :], dists.shape)
    out_site_idx = site_idx_grid[mask]
    out_Zs = species_Z[out_site_idx]

    order = np.argsort(out_dists)
    return out_Zs[order], out_dists[order].astype(np.float32)


# -----------------------------------------------------------------------------
# Worker: process all rows of one mp_id group
# -----------------------------------------------------------------------------
def process_mp_id_group(args):
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    mp_id, rows = args
    results = []

    needs_primitive = any(r["n_center_sites"] >= 2 for r in rows)
    cart = None
    Zs_arr = None
    M = None
    elem_to_indices = None
    primitive_error = None

    if needs_primitive:
        try:
            s_super = Structure.from_file(rows[0]["poscar_path"])
            prim = SpacegroupAnalyzer(s_super, symprec=SYMPREC).get_primitive_standard_structure()
            cart   = np.ascontiguousarray(prim.cart_coords, dtype=np.float64)
            Zs_arr = np.array([s.specie.Z for s in prim], dtype=np.int64)
            M      = np.ascontiguousarray(prim.lattice.matrix, dtype=np.float64)
            elem_to_indices = defaultdict(list)
            for i, site in enumerate(prim):
                elem_to_indices[site.specie.symbol].append(i)
        except Exception as e:
            primitive_error = f"primitive_error:{type(e).__name__}"

    for r in rows:
        n_sites = r["n_center_sites"]

        if n_sites == 1:
            results.append({
                "sample_name": r["sample_name"],
                "n_center_sites": 1,
                "tag": "single_site",
                "max_shell1_MAE": 0.0,
                "n_unique_shell1_multisets": 1,
            })
            continue

        if primitive_error is not None:
            results.append({
                "sample_name": r["sample_name"],
                "n_center_sites": n_sites,
                "tag": "primitive_error",
                "max_shell1_MAE": SENTINEL_MAE,
                "n_unique_shell1_multisets": -1,
            })
            continue

        center_indices = elem_to_indices.get(r["center_element"], [])
        if len(center_indices) != n_sites:
            results.append({
                "sample_name": r["sample_name"],
                "n_center_sites": n_sites,
                "tag": "phase_a_mismatch",
                "max_shell1_MAE": SENTINEL_MAE,
                "n_unique_shell1_multisets": -1,
            })
            continue

        shell1_outer = float(r["shell1_outer"])
        r_query = shell1_outer + EPS_QUERY
        per_site_distances = []
        per_site_multisets = []
        ok_geometry = True

        for ci in center_indices:
            try:
                Zs_nbr, ds_nbr = find_neighbors_brute(cart, Zs_arr, M, ci, r_query)
            except Exception:
                ok_geometry = False
                break
            keep = ds_nbr <= shell1_outer + EPS_INCLUDE
            ds_in = ds_nbr[keep]
            Zs_in = Zs_nbr[keep].astype(int).tolist()
            per_site_distances.append(ds_in)
            per_site_multisets.append(frozenset(Counter(Zs_in).items()))

        if not ok_geometry:
            results.append({
                "sample_name": r["sample_name"],
                "n_center_sites": n_sites,
                "tag": "neighbor_error",
                "max_shell1_MAE": SENTINEL_MAE,
                "n_unique_shell1_multisets": -1,
            })
            continue

        n_unique_multisets = len(set(per_site_multisets))
        all_multisets_equal = (n_unique_multisets == 1)
        sizes = {len(d) for d in per_site_distances}
        all_counts_equal = (len(sizes) == 1)

        if all_counts_equal:
            n = len(per_site_distances)
            max_mae = 0.0
            if n >= 2 and len(per_site_distances[0]) > 0:
                for i in range(n):
                    di = per_site_distances[i]
                    for j in range(i + 1, n):
                        dj = per_site_distances[j]
                        mae = float(np.mean(np.abs(di - dj)))
                        if mae > max_mae:
                            max_mae = mae
            max_mae_recorded = max_mae
        else:
            max_mae = float("inf")
            max_mae_recorded = SENTINEL_MAE

        if all_multisets_equal and max_mae < EQUIV_MAE_TOL:
            tag = "equivalent"
        elif all_counts_equal and max_mae < NEAR_EQUIV_MAE:
            tag = "near_equivalent"
        else:
            tag = "incompat"

        results.append({
            "sample_name": r["sample_name"],
            "n_center_sites": n_sites,
            "tag": tag,
            "max_shell1_MAE": round(max_mae_recorded, 5),
            "n_unique_shell1_multisets": n_unique_multisets,
        })

    return results


# -----------------------------------------------------------------------------
# Pre-flight sanity check vs Phase A's saved distances
# -----------------------------------------------------------------------------
def sanity_check(inv, neighbors, n_test=5):
    """
    Compare brute-force at r=10 vs Phase A's saved distances on n_test samples.
    Pass criteria:
      - count: exact match
      - species sequence: exact match
      - distance MAE: <= 0.01 Å (tolerates spglib coord precision drift)
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    print("\n  Pre-flight sanity check (brute-force vs Phase A's saved arrays):")

    # Pick first multi-site sample at each n_center_sites
    inv_n = inv.copy()
    inv_n["n_center_sites"] = inv_n["sample_name"].map(
        lambda s: neighbors.get(s, {}).get("n_center_sites", 0)
    )

    targets = []
    for n in [2, 4, 8, 16, 12]:
        c = inv_n[inv_n["n_center_sites"] == n]
        for _, row in c.head(3).iterrows():
            sname = row["sample_name"]
            if sname in neighbors and neighbors[sname]["status"] == "ok":
                targets.append(row.to_dict())
                break
        if len(targets) >= n_test:
            break

    n_ok = 0
    n_total = len(targets[:n_test])
    for t in targets[:n_test]:
        sname = t["sample_name"]
        try:
            s_super = Structure.from_file(t["poscar_path"])
            prim = SpacegroupAnalyzer(s_super, symprec=SYMPREC).get_primitive_standard_structure()
            cart   = np.ascontiguousarray(prim.cart_coords, dtype=np.float64)
            Zs_arr = np.array([s.specie.Z for s in prim], dtype=np.int64)
            M      = np.ascontiguousarray(prim.lattice.matrix, dtype=np.float64)

            center_indices = [i for i, site in enumerate(prim)
                              if site.specie.symbol == t["center_element"]]
            ci = center_indices[0]   # match Phase A's "first site" choice

            Zs_b, ds_b = find_neighbors_brute(cart, Zs_arr, M, ci, 10.0)

            saved = neighbors[sname]
            saved_d = saved["distances"]
            saved_Z = saved["species_Z"]

            if len(ds_b) != len(saved_d):
                print(f"    ✗ {sname}: count mismatch {len(ds_b)} vs {len(saved_d)}")
                continue
            d_mae = float(np.mean(np.abs(ds_b - saved_d)))
            d_max = float(np.max(np.abs(ds_b - saved_d)))
            Z_match = np.array_equal(Zs_b.astype(np.int8), saved_Z)
            if d_max > 0.01 or not Z_match:
                print(f"    ✗ {sname}: max d diff = {d_max:.5f} Å, Z match = {Z_match}")
                continue
            print(f"    ✓ {sname}: {len(ds_b)} nbrs, MAE = {d_mae:.5f} Å, "
                  f"max diff = {d_max:.5f} Å")
            n_ok += 1
        except Exception as e:
            print(f"    ✗ {sname}: {type(e).__name__}: {e}")

    return n_ok, n_total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"Step 2.5 Phase D v2 — brute-force multi-site tagging")
    print(f"  detected CPU: {cpu_count()}, using N_WORKERS = {N_WORKERS}")

    print(f"\n  loading inventory ...")
    inv = pd.read_csv(INVENTORY_CSV)

    print(f"  loading neighbors pickle ...")
    with open(NEIGHBOR_PKL_PATH, "rb") as f:
        neighbors = pickle.load(f)

    sample_to_n_sites = {s: int(rec["n_center_sites"]) for s, rec in neighbors.items()}

    print(f"  loading shell_boundaries pickle ...")
    with open(SHELL_BOUND_PATH, "rb") as f:
        shell_bound = pickle.load(f)
    sample_to_shell1_outer = {s: float(rec["shell_ends"][0]) for s, rec in shell_bound.items()}
    del shell_bound

    # Sanity check first — abort if brute-force doesn't match Phase A
    n_ok, n_total = sanity_check(inv, neighbors, n_test=5)
    if n_ok != n_total or n_total == 0:
        print(f"\n  ✗✗ sanity check FAILED ({n_ok}/{n_total}) — abort, do not run full pass")
        sys.exit(1)
    print(f"  ✓ sanity check passed: {n_ok}/{n_total}")
    del neighbors

    # Build groups
    print(f"\n  building mp_id groups ...")
    groups = []
    missing = 0
    for mp_id, gdf in inv.groupby("mp_id"):
        rows = []
        for _, ir in gdf.iterrows():
            sname = ir["sample_name"]
            if sname not in sample_to_n_sites or sname not in sample_to_shell1_outer:
                missing += 1
                continue
            rows.append({
                "sample_name": sname,
                "center_element": ir["center_element"],
                "poscar_path": ir["poscar_path"],
                "n_center_sites": sample_to_n_sites[sname],
                "shell1_outer": sample_to_shell1_outer[sname],
            })
        if rows:
            groups.append((mp_id, rows))
    if missing:
        print(f"    ⚠ samples missing: {missing}")
    n_groups = len(groups)
    groups.sort(key=lambda g: -len(g[1]))

    n_single = sum(1 for _, rs in groups for r in rs if r["n_center_sites"] == 1)
    n_multi  = sum(1 for _, rs in groups for r in rs if r["n_center_sites"] >= 2)
    print(f"    n_groups: {n_groups:,}, single: {n_single:,}, multi: {n_multi:,}")

    # Parallel
    print(f"\n  starting Pool({N_WORKERS}) ...")
    t_compute = time.time()
    all_results = []
    with Pool(processes=N_WORKERS) as pool:
        for grp_results in tqdm(pool.imap_unordered(process_mp_id_group, groups, chunksize=1),
                                total=n_groups, desc="mp_id groups"):
            all_results.extend(grp_results)
    compute_time = time.time() - t_compute
    print(f"\n  compute wall-clock: {compute_time:.1f}s ({compute_time/60:.2f} min)")

    # ---- Tag distribution ----
    tag_counts = Counter(r["tag"] for r in all_results)
    print(f"\n  tag distribution:")
    for t in ["single_site", "equivalent", "near_equivalent", "incompat",
              "primitive_error", "neighbor_error", "phase_a_mismatch"]:
        if tag_counts.get(t, 0):
            pct = 100 * tag_counts[t] / len(all_results)
            print(f"    {t:<22s} {tag_counts[t]:>7,}  ({pct:5.2f}%)")

    # ---- Build dataframe + merge with split + element ----
    df = pd.DataFrame(all_results)
    df = df.merge(inv[["sample_name", "split", "center_element", "mp_id"]],
                  on="sample_name", how="left")
    cols = ["sample_name", "mp_id", "center_element", "split",
            "n_center_sites", "tag", "max_shell1_MAE", "n_unique_shell1_multisets"]
    df = df[cols]
    df.to_csv(TAG_CSV_PATH, index=False)
    print(f"\n  saved {TAG_CSV_PATH}  ({len(df):,} rows)")

    # ---- Aggregations ----
    tag_split_table = pd.crosstab(df["tag"], df["split"], margins=True, margins_name="ALL")
    col_order = [c for c in ["train", "val", "test", "holdout", "ALL"] if c in tag_split_table.columns]
    tag_split_table = tag_split_table[col_order]
    row_order = [r for r in ["single_site", "equivalent", "near_equivalent", "incompat",
                              "primitive_error", "neighbor_error", "phase_a_mismatch", "ALL"]
                 if r in tag_split_table.index]
    tag_split_table = tag_split_table.loc[row_order]
    tag_split_pct = (tag_split_table.div(tag_split_table.loc["ALL"], axis=1) * 100.0).round(2)

    incompat_df = df[df["tag"] == "incompat"]
    incompat_by_elem = incompat_df.groupby("center_element").size().reset_index(name="n_incompat")
    total_by_elem    = df.groupby("center_element").size().reset_index(name="n_total")
    elem_table = incompat_by_elem.merge(total_by_elem, on="center_element")
    elem_table["incompat_pct"] = (100.0 * elem_table["n_incompat"] / elem_table["n_total"]).round(2)
    elem_table = elem_table.sort_values("n_incompat", ascending=False).reset_index(drop=True)

    near_eq_df = df[df["tag"] == "near_equivalent"]
    near_eq_multiset_mismatch = int((near_eq_df["n_unique_shell1_multisets"] > 1).sum())

    incompat_ncs = incompat_df["n_center_sites"].value_counts().sort_index()

    # ---- Summary ----
    out = []
    out.append("=" * 72)
    out.append("Step 2.5 Phase D v2 — brute-force multi-site tagging summary")
    out.append("=" * 72)
    out.append(f"Total samples:           {len(df):,}")
    out.append(f"Wall-clock (compute):    {compute_time:.1f} s ({compute_time/60:.2f} min)")
    out.append(f"Wall-clock (total):      {time.time() - t_start:.1f} s")
    out.append(f"Sanity check:            {n_ok}/{n_total} samples matched Phase A "
               f"(MAE tol = 0.01 Å, count + species exact)")
    out.append("")
    out.append("Tag distribution (overall):")
    for t in ["single_site", "equivalent", "near_equivalent", "incompat",
              "primitive_error", "neighbor_error", "phase_a_mismatch"]:
        if tag_counts.get(t, 0):
            pct = 100 * tag_counts[t] / len(df)
            out.append(f"  {t:<22s} {tag_counts[t]:>7,}  ({pct:5.2f}%)")
    out.append("")
    out.append("Tag × split (counts):")
    out.append(tag_split_table.to_string())
    out.append("")
    out.append("Tag × split (% within split):")
    out.append(tag_split_pct.to_string())
    out.append("")
    out.append(f"Sub-statistic: 'near_equivalent' with multiset mismatch")
    out.append(f"  count: {near_eq_multiset_mismatch:,} of {len(near_eq_df):,} near_equivalent rows "
               f"({100*near_eq_multiset_mismatch/max(1,len(near_eq_df)):.2f}%)")
    out.append("")
    out.append("Incompat by center_element (top 15 by absolute count):")
    out.append(elem_table.head(15).to_string(index=False))
    out.append("")
    out.append("Incompat by n_center_sites bucket:")
    for k, v in incompat_ncs.items():
        pct = 100 * v / max(1, tag_counts.get("incompat", 1))
        out.append(f"  n_sites={int(k):>3d}: {v:>7,}  ({pct:5.2f}% of incompat)")
    out.append("")
    out.append("Notes:")
    out.append(f"  - Neighbor finder: pure-numpy brute-force (pymatgen Cython broken in env)")
    out.append(f"  - Sanity-checked vs Phase A's saved distances on 5 samples before run")
    out.append(f"  - shell1_outer source: shell_boundaries.pkl shell_ends[0]")
    out.append(f"  - r_query = shell1_outer + {EPS_QUERY} Å")
    out.append(f"  - filter tolerance: distance <= shell1_outer + {EPS_INCLUDE} Å")
    out.append(f"  - 'equivalent':     all multisets equal AND max site-pair MAE < {EQUIV_MAE_TOL} Å")
    out.append(f"  - 'near_equivalent': all counts equal AND max site-pair MAE < {NEAR_EQUIV_MAE} Å")
    out.append(f"  - max_shell1_MAE = {SENTINEL_MAE} → counts mismatch (incompat)")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n  saved {SUMMARY_PATH}")
    print("\n" + "\n".join(out[-30:]))
    print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s")


class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for t in self.s: t.write(x)
    def flush(self):
        for t in self.s: t.flush()


if __name__ == "__main__":
    log_buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, log_buf)
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"\nLog: {LOG_PATH}")
