"""
Step 2.5 Phase A.1 — Compute neighbor distances for all 128,382 samples.
========================================================================

Strategy
--------
* Group samples by mp_id (41,431 groups, avg 3.1 samples/group).
* Each group = one parallel task. Load POSCAR + compute primitive ONCE
  per group, then iterate rows (different center_element) sharing that
  primitive. Eliminates ~67.7% of redundant primitive computations.
* Pool(8) with imap_unordered. Groups sorted by size desc for better
  load balancing (slow groups dispatched first).
* r_cutoff = 10 Å, symprec = 0.1  (LOCKED per handoff §5.1)

Outputs (in experiment4/step2_5/)
---------------------------------
* step2_5_neighbor_distances.pkl  — dict[sample_name] → {...}
* step2_5_failures.csv            — samples with status != 'ok'
* step2_5a_compute.log            — execution log

Estimated runtime
-----------------
~3-5 min on Pool(8) given precheck median = 0.011 s/sample.

Run
---
cd C:\\Users\\T-Cat\\Desktop\\DiffCSP-main\\experiment4\\step2_5
C:/Users/T-Cat/AppData/Local/Microsoft/WindowsApps/python3.9.exe .\\step2_5a_compute_neighbors.py
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
FAILURES_CSV_PATH = os.path.join(STEP25_DIR, "step2_5_failures.csv")
LOG_PATH          = os.path.join(STEP25_DIR, "step2_5a_compute.log")

R_CUTOFF = 10.0   # 🔒 LOCKED
SYMPREC  = 0.1    # 🔒 LOCKED (match Exp2)
N_WORKERS = 8     # from precheck: 20 cores, leave 12 for OS + other work

os.makedirs(STEP25_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Worker: process one mp_id's samples
# -----------------------------------------------------------------------------
def process_mp_id_group(args):
    """
    Process all samples sharing one mp_id.

    args: (mp_id, [ {sample_name, center_element, poscar_path}, ... ])
    Returns: list of per-sample result dicts.
    """
    # Import inside worker so each process initializes its own pymatgen/spglib
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    mp_id, rows = args
    results = []

    # ---- Load POSCAR + primitive ONCE for this mp_id group ----
    poscar_path = rows[0]["poscar_path"]
    try:
        s_super = Structure.from_file(poscar_path)
        prim = SpacegroupAnalyzer(
            s_super, symprec=SYMPREC
        ).get_primitive_standard_structure()
    except Exception as e:
        err = f"primitive_error:{type(e).__name__}"
        for r in rows:
            results.append(_fail_record(r, mp_id, err, 0))
        return results

    # Pre-index primitive by element symbol — avoid re-scanning for each row
    elem_to_indices = defaultdict(list)
    for i, site in enumerate(prim):
        elem_to_indices[site.specie.symbol].append(i)

    # ---- Iterate rows, each with potentially different center_element ----
    for r in rows:
        center_el = r["center_element"]
        center_sites = elem_to_indices.get(center_el, [])
        n_center_sites = len(center_sites)

        if n_center_sites == 0:
            results.append(_fail_record(r, mp_id, "no_center_atom", 0))
            continue

        try:
            neighbors = prim.get_neighbors(prim[center_sites[0]], r=R_CUTOFF)
        except Exception as e:
            results.append(_fail_record(r, mp_id,
                                        f"neighbor_error:{type(e).__name__}",
                                        n_center_sites))
            continue

        if len(neighbors) == 0:
            results.append(_fail_record(r, mp_id, "no_neighbors", n_center_sites))
            continue

        pairs = sorted(
            [(nbr.nn_distance, nbr.specie.Z) for nbr in neighbors],
            key=lambda x: x[0],
        )
        distances = np.array([p[0] for p in pairs], dtype=np.float32)
        species_Z = np.array([p[1] for p in pairs], dtype=np.int8)

        results.append({
            "sample_name": r["sample_name"],
            "mp_id": mp_id,
            "center_element": center_el,
            "status": "ok",
            "distances": distances,
            "species_Z": species_Z,
            "n_center_sites": n_center_sites,
            "n_neighbors": len(distances),
        })

    return results


def _fail_record(r, mp_id, status, n_center_sites):
    return {
        "sample_name": r["sample_name"],
        "mp_id": mp_id,
        "center_element": r["center_element"],
        "status": status,
        "distances": None,
        "species_Z": None,
        "n_center_sites": n_center_sites,
        "n_neighbors": 0,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"Step 2.5 Phase A.1 — neighbor distance computation")
    print(f"  detected CPU: {cpu_count()}, using N_WORKERS = {N_WORKERS}")
    print(f"  reading: {INVENTORY_CSV}")

    inv = pd.read_csv(INVENTORY_CSV)
    assert inv.shape[0] == 128382, f"inventory rows: {inv.shape[0]}"
    print(f"  inventory shape: {inv.shape}")

    # ---- Build mp_id groups ----
    groups = []
    for mp_id, gdf in inv.groupby("mp_id"):
        rows = gdf[["sample_name", "center_element", "poscar_path"]].to_dict("records")
        groups.append((mp_id, rows))
    n_groups = len(groups)
    print(f"  n_groups (by mp_id): {n_groups:,}")
    print(f"  avg samples/group:   {inv.shape[0] / n_groups:.2f}")
    grp_sizes = [len(g[1]) for g in groups]
    print(f"  group-size distribution: "
          f"min={min(grp_sizes)}, median={int(np.median(grp_sizes))}, "
          f"max={max(grp_sizes)}, p95={int(np.percentile(grp_sizes, 95))}")

    # Sort large groups first → better load balancing
    groups.sort(key=lambda g: -len(g[1]))

    # ---- Parallel execute ----
    print(f"\nStarting Pool({N_WORKERS}) ...")
    t_compute = time.time()
    all_results = []

    with Pool(processes=N_WORKERS) as pool:
        for grp_results in tqdm(
            pool.imap_unordered(process_mp_id_group, groups, chunksize=1),
            total=n_groups,
            desc="mp_id groups",
        ):
            all_results.extend(grp_results)

    compute_time = time.time() - t_compute
    print(f"\nCompute wall-clock: {compute_time:.1f}s ({compute_time / 60:.2f} min)")

    # ---- Validation ----
    assert len(all_results) == 128382, f"Got {len(all_results)}, expected 128,382"

    status_counts = Counter(r["status"] for r in all_results)
    print(f"\nStatus distribution:")
    for s, c in status_counts.most_common():
        print(f"  {s:<30s} {c:>7,}")

    # ---- n_center_sites distribution ----
    ok_results = [r for r in all_results if r["status"] == "ok"]
    ncs_counter = Counter(r["n_center_sites"] for r in ok_results)
    print(f"\nn_center_sites distribution (ok only, n={len(ok_results):,}):")
    multi_site_cnt = 0
    for k in sorted(ncs_counter.keys()):
        v = ncs_counter[k]
        pct = 100 * v / len(ok_results)
        print(f"  {k:>3d} sites: {v:>7,} ({pct:5.2f}%)")
        if k >= 2:
            multi_site_cnt += v
    multi_pct = 100 * multi_site_cnt / len(ok_results)
    flag = "   ⚠ FLAG (>20%)" if multi_pct > 20 else ""
    print(f"  n_center_sites >=2: {multi_site_cnt:,} ({multi_pct:.2f}%){flag}")

    # ---- Sanity: neighbor-count distribution ----
    n_neighbors_list = [r["n_neighbors"] for r in ok_results]
    print(f"\nn_neighbors(10Å) stats (ok only):")
    print(f"  min={min(n_neighbors_list)}, max={max(n_neighbors_list)}, "
          f"median={int(np.median(n_neighbors_list))}, "
          f"p5={int(np.percentile(n_neighbors_list, 5))}, "
          f"p95={int(np.percentile(n_neighbors_list, 95))}")
    n_with_lt20 = sum(1 for n in n_neighbors_list if n < 20)
    print(f"  samples with <20 neighbors: {n_with_lt20}")

    # ---- Save pickle: dict[sample_name] -> payload ----
    print(f"\nSerialising pickle ...")
    pkl = {}
    for r in all_results:
        pkl[r["sample_name"]] = {
            "distances": r["distances"],
            "species_Z": r["species_Z"],
            "n_center_sites": r["n_center_sites"],
            "n_neighbors": r["n_neighbors"],
            "status": r["status"],
            "center_element": r["center_element"],
        }
    with open(NEIGHBOR_PKL_PATH, "wb") as f:
        pickle.dump(pkl, f, protocol=4)
    pkl_mb = os.path.getsize(NEIGHBOR_PKL_PATH) / 1024 / 1024
    print(f"  {NEIGHBOR_PKL_PATH}  ({pkl_mb:.1f} MB)")

    # ---- Save failures CSV ----
    failures = [r for r in all_results if r["status"] != "ok"]
    fail_cols = ["sample_name", "mp_id", "center_element", "status", "n_center_sites"]
    if failures:
        pd.DataFrame([{k: r[k] for k in fail_cols} for r in failures]).to_csv(
            FAILURES_CSV_PATH, index=False
        )
        print(f"  {FAILURES_CSV_PATH}  ({len(failures)} rows)")
    else:
        pd.DataFrame(columns=fail_cols).to_csv(FAILURES_CSV_PATH, index=False)
        print(f"  {FAILURES_CSV_PATH}  (empty — zero failures)")

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
