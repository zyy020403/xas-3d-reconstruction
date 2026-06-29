"""
Step 2.5 Phase D — Full-dataset multi-site equivalence tagging
===============================================================

For each of 128,382 samples, produce a tag in
  {single_site, equivalent, near_equivalent, incompat}

Per-sample logic (per MA spec):
  1. Load primitive (cached by mp_id, group-level)
  2. If n_center_sites == 1 → tag = "single_site"   [fast path, no neighbor work]
  3. Else for each center site i:
       shell1_outer_for_compare = shell_boundaries[sample]["shell_ends"][0]
       neighbors_i = prim.get_neighbors(prim[i], r=shell1_outer + 0.05)
       shell1_atoms_i = [n for n in neighbors_i if n.nn_distance <= shell1_outer + 1e-4]
       multiset_i = Counter([Z for n in shell1_atoms_i])
  4. Decision:
       all_multisets_equal AND max_pairwise_MAE < 0.01  → "equivalent"
       all_counts_equal    AND max_pairwise_MAE < 0.1   → "near_equivalent"
       else                                             → "incompat"

Note (per MA literal rule): "near_equivalent" requires only count equality
+ MAE bound — multisets may differ. Summary tracks this sub-case separately.

Inputs
------
* data_inventory.csv          → split, center_element, poscar_path
* step2_5_neighbor_distances.pkl → n_center_sites
* shell_boundaries.pkl        → shell_ends[0] per sample (only this field needed)

Outputs
-------
* site_equivalence_tag.csv     (128,382 rows: sample_name, n_center_sites, tag,
                                 max_shell1_MAE, n_unique_shell1_multisets)
* step2_5d_summary.txt
* step2_5d_tag.log

Run
---
python step2_5d_full_multisite_tag.py
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

NEIGHBOR_PKL_PATH    = os.path.join(STEP25_DIR, "step2_5_neighbor_distances.pkl")
SHELL_BOUND_PATH     = os.path.join(STEP25_DIR, "shell_boundaries.pkl")
TAG_CSV_PATH         = os.path.join(STEP25_DIR, "site_equivalence_tag.csv")
SUMMARY_PATH         = os.path.join(STEP25_DIR, "step2_5d_summary.txt")
LOG_PATH             = os.path.join(STEP25_DIR, "step2_5d_tag.log")

SYMPREC      = 0.1
EPS_QUERY    = 0.05      # r = shell1_outer + EPS_QUERY for safety margin
EPS_INCLUDE  = 1e-4      # tolerance when filtering by shell1_outer
N_WORKERS    = 8

EQUIV_MAE_TOL    = 0.01
NEAR_EQUIV_MAE   = 0.10
SENTINEL_MAE     = 999.0


# -----------------------------------------------------------------------------
# Worker: process all rows of one mp_id group
# -----------------------------------------------------------------------------
def process_mp_id_group(args):
    """
    args: (mp_id, [ {sample_name, center_element, poscar_path,
                     n_center_sites, shell1_outer}, ... ])
    Returns: list of result dicts.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    mp_id, rows = args
    results = []

    # Multi-site rows in this group decide whether we even need primitive
    needs_primitive = any(r["n_center_sites"] >= 2 for r in rows)
    prim = None
    elem_to_indices = None
    primitive_error = None

    if needs_primitive:
        try:
            s_super = Structure.from_file(rows[0]["poscar_path"])
            prim = SpacegroupAnalyzer(
                s_super, symprec=SYMPREC
            ).get_primitive_standard_structure()
            elem_to_indices = defaultdict(list)
            for i, site in enumerate(prim):
                elem_to_indices[site.specie.symbol].append(i)
        except Exception as e:
            primitive_error = f"primitive_error:{type(e).__name__}"

    for r in rows:
        n_sites = r["n_center_sites"]

        # Fast path: single-site samples skip all geometry
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

        # Cross-check: number of indices in primitive must match Phase A's count
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

        # Compute shell-1 atom set for each site (within shell1_outer from Phase B)
        shell1_outer = float(r["shell1_outer"])
        r_query = shell1_outer + EPS_QUERY
        per_site_distances = []
        per_site_multisets = []
        ok_geometry = True

        for ci in center_indices:
            try:
                nbrs = prim.get_neighbors(prim[ci], r=r_query)
            except Exception:
                ok_geometry = False
                break
            valid = [(n.nn_distance, int(n.specie.Z)) for n in nbrs
                     if n.nn_distance <= shell1_outer + EPS_INCLUDE]
            valid.sort(key=lambda x: x[0])
            ds = np.array([v[0] for v in valid], dtype=np.float32)
            Zs = [v[1] for v in valid]
            per_site_distances.append(ds)
            per_site_multisets.append(frozenset(Counter(Zs).items()))

        if not ok_geometry:
            results.append({
                "sample_name": r["sample_name"],
                "n_center_sites": n_sites,
                "tag": "neighbor_error",
                "max_shell1_MAE": SENTINEL_MAE,
                "n_unique_shell1_multisets": -1,
            })
            continue

        # ---- Build comparison metrics ----
        n_unique_multisets = len(set(per_site_multisets))
        all_multisets_equal = (n_unique_multisets == 1)

        sizes = {len(d) for d in per_site_distances}
        all_counts_equal = (len(sizes) == 1)

        # Pairwise max MAE (only meaningful when counts equal)
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
            max_mae = float("inf")          # forces fall-through to "incompat"
            max_mae_recorded = SENTINEL_MAE

        # ---- Tag decision (MA spec) ----
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
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"Step 2.5 Phase D — full multi-site equivalence tagging")
    print(f"  detected CPU: {cpu_count()}, using N_WORKERS = {N_WORKERS}")

    # ---- Load inventory ----
    print(f"\n  loading inventory ...")
    inv = pd.read_csv(INVENTORY_CSV)
    print(f"    shape: {inv.shape}")

    # ---- Load just the bits we need from the two pickles ----
    print(f"  loading n_center_sites from neighbors pickle ...")
    with open(NEIGHBOR_PKL_PATH, "rb") as f:
        neighbors = pickle.load(f)
    sample_to_n_sites = {s: int(rec["n_center_sites"])
                         for s, rec in neighbors.items()}
    print(f"    entries: {len(sample_to_n_sites):,}")
    del neighbors  # free memory

    print(f"  loading shell_ends[0] from shell_boundaries pickle ...")
    with open(SHELL_BOUND_PATH, "rb") as f:
        shell_bound = pickle.load(f)
    sample_to_shell1_outer = {s: float(rec["shell_ends"][0])
                              for s, rec in shell_bound.items()}
    print(f"    entries: {len(sample_to_shell1_outer):,}")
    del shell_bound

    # ---- Build groups ----
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
        print(f"    ⚠ samples missing from one of the pickles: {missing}")
    n_groups = len(groups)
    print(f"    n_groups: {n_groups:,}")

    # Sort by group size desc → better load balance
    groups.sort(key=lambda g: -len(g[1]))

    # Single-site fast-path stats (no primitive needed)
    n_single = sum(1 for _, rs in groups for r in rs if r["n_center_sites"] == 1)
    n_multi  = sum(1 for _, rs in groups for r in rs if r["n_center_sites"] >= 2)
    print(f"    single-site rows: {n_single:,}  (fast path)")
    print(f"    multi-site rows:  {n_multi:,}   (full primitive)")

    # ---- Parallel execute ----
    print(f"\n  starting Pool({N_WORKERS}) ...")
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
    print(f"\n  compute wall-clock: {compute_time:.1f}s ({compute_time / 60:.2f} min)")

    # ---- Validation ----
    if len(all_results) != 128382 - missing:
        print(f"  ⚠ result count {len(all_results)} != expected {128382 - missing}")

    tag_counts = Counter(r["tag"] for r in all_results)
    print(f"\n  tag distribution:")
    for t in ["equivalent", "near_equivalent", "incompat",
              "single_site", "primitive_error", "neighbor_error",
              "phase_a_mismatch"]:
        if tag_counts.get(t, 0):
            pct = 100 * tag_counts[t] / len(all_results)
            print(f"    {t:<22s} {tag_counts[t]:>7,}  ({pct:5.2f}%)")

    # ---- Build DataFrame & merge with inventory for split + element ----
    df = pd.DataFrame(all_results)
    df = df.merge(inv[["sample_name", "split", "center_element", "mp_id"]],
                  on="sample_name", how="left")

    # Reorder columns
    cols = ["sample_name", "mp_id", "center_element", "split",
            "n_center_sites", "tag",
            "max_shell1_MAE", "n_unique_shell1_multisets"]
    df = df[cols]

    df.to_csv(TAG_CSV_PATH, index=False)
    print(f"\n  saved {TAG_CSV_PATH}  ({len(df):,} rows)")

    # ---- Aggregations for summary ----
    # 1. tag x split
    tag_split_table = pd.crosstab(df["tag"], df["split"], margins=True,
                                  margins_name="ALL")
    # column order
    col_order = [c for c in ["train", "val", "test", "holdout", "ALL"]
                 if c in tag_split_table.columns]
    tag_split_table = tag_split_table[col_order]
    # row order
    row_order = [r for r in ["single_site", "equivalent", "near_equivalent",
                              "incompat", "primitive_error", "neighbor_error",
                              "phase_a_mismatch", "ALL"]
                 if r in tag_split_table.index]
    tag_split_table = tag_split_table.loc[row_order]

    # 1b. tag x split as percentage *within column*
    tag_split_pct = (tag_split_table.div(tag_split_table.loc["ALL"], axis=1)
                                    * 100.0).round(2)

    # 2. incompat by element (top-15 by absolute count)
    incompat_df = df[df["tag"] == "incompat"]
    incompat_by_elem = (incompat_df.groupby("center_element")
                                   .size()
                                   .reset_index(name="n_incompat"))
    total_by_elem = (df.groupby("center_element")
                       .size()
                       .reset_index(name="n_total"))
    elem_table = incompat_by_elem.merge(total_by_elem, on="center_element")
    elem_table["incompat_pct"] = (100.0 * elem_table["n_incompat"]
                                  / elem_table["n_total"]).round(2)
    elem_table = elem_table.sort_values("n_incompat", ascending=False).reset_index(drop=True)

    # 3. near_equivalent with multiset mismatch (literal MA rule allows this)
    near_eq_df = df[df["tag"] == "near_equivalent"]
    near_eq_multiset_mismatch = int((near_eq_df["n_unique_shell1_multisets"] > 1).sum())

    # 4. n_center_sites distribution within incompat (which buckets are problematic)
    incompat_ncs = (incompat_df["n_center_sites"]
                    .value_counts()
                    .sort_index())

    # ---- Write summary ----
    out = []
    out.append("=" * 72)
    out.append("Step 2.5 Phase D — Full-dataset multi-site tagging summary")
    out.append("=" * 72)
    out.append(f"Total samples:           {len(df):,}")
    out.append(f"Wall-clock (compute):    {compute_time:.1f} s ({compute_time/60:.2f} min)")
    out.append(f"Wall-clock (total):      {time.time() - t0:.1f} s")
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
    out.append(f"Sub-statistic: 'near_equivalent' tag with multiset mismatch")
    out.append(f"  (counts equal + MAE<0.1 but element composition differs)")
    out.append(f"  count: {near_eq_multiset_mismatch:,} of {len(near_eq_df):,}"
               f" near_equivalent rows "
               f"({100 * near_eq_multiset_mismatch / max(1, len(near_eq_df)):.2f}%)")
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
    out.append(f"  - shell1_outer source: shell_boundaries.pkl shell_ends[0]")
    out.append(f"  - r_query = shell1_outer + {EPS_QUERY} Å")
    out.append(f"  - 'equivalent' threshold:    multisets equal AND max MAE < {EQUIV_MAE_TOL} Å")
    out.append(f"  - 'near_equivalent' threshold: counts equal AND max MAE < {NEAR_EQUIV_MAE} Å")
    out.append(f"  - max_shell1_MAE = {SENTINEL_MAE} → counts mismatch (size-incompat)")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n  saved {SUMMARY_PATH}")
    print("\n" + "\n".join(out[-30:]))    # show tail
    print(f"\nTotal wall-clock: {time.time() - t0:.1f}s")


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
