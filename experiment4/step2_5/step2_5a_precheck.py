"""
Step 2.5 Phase A — Pre-flight check
====================================

Purpose:
  1) Verify Step 1 outputs are readable and schema matches expectations.
  2) Run single-sample pipeline on 5 diverse center elements (O/Fe/Cu/La/U).
  3) Estimate per-sample time → recommend single-thread vs multiprocessing.

Output:
  - console log
  - step2_5a_precheck.log in experiment4/step2_5/
  - (no data artifacts — that's phase A main's job)

Run:
  cd C:\\Users\\T-Cat\\Desktop\\DiffCSP-main\\experiment4\\step2_5
  C:/Users/T-Cat/AppData/Local/Microsoft/WindowsApps/python3.9.exe .\\step2_5a_precheck.py

Expected runtime: 5-30 seconds.
"""
import io
import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EXP4_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR     = os.path.join(EXP4_ROOT, "step1")
STEP25_DIR    = os.path.join(EXP4_ROOT, "step2_5")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")

os.makedirs(STEP25_DIR, exist_ok=True)

# Diverse center elements covering s/p, 3d, 4d-5d, lanthanide, actinide.
TARGET_ELEMENTS = ["O", "Fe", "Cu", "La", "U"]


# -----------------------------------------------------------------------------
# Tee stdout so we also save a log file
# -----------------------------------------------------------------------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


# -----------------------------------------------------------------------------
# Single-sample pipeline (same logic as §5.1 of handoff)
# -----------------------------------------------------------------------------
def process_one_sample(row):
    """Returns (distances, species_Z, n_center_sites, status, elapsed_s)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    t0 = time.time()
    try:
        s_super = Structure.from_file(row["poscar_path"])
        prim = SpacegroupAnalyzer(s_super, symprec=0.1).get_primitive_standard_structure()
    except Exception as e:
        return None, None, 0, f"primitive_error:{type(e).__name__}", time.time() - t0

    center_sites = [i for i, site in enumerate(prim)
                    if site.specie.symbol == row["center_element"]]
    n_center_sites = len(center_sites)

    if n_center_sites == 0:
        return None, None, 0, "no_center_atom", time.time() - t0

    try:
        neighbors = prim.get_neighbors(prim[center_sites[0]], r=10.0)
    except Exception as e:
        return None, None, n_center_sites, f"neighbor_error:{type(e).__name__}", time.time() - t0

    if len(neighbors) == 0:
        return None, None, n_center_sites, "no_neighbors", time.time() - t0

    pairs = sorted([(nbr.nn_distance, nbr.specie.Z) for nbr in neighbors],
                   key=lambda x: x[0])
    distances = np.array([p[0] for p in pairs], dtype=np.float32)
    species_Z = np.array([p[1] for p in pairs], dtype=np.int8)
    return distances, species_Z, n_center_sites, "ok", time.time() - t0


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
def check_step1_outputs():
    print("=" * 72)
    print("CHECK 1: Step 1 output integrity")
    print("=" * 72)

    assert os.path.isfile(INVENTORY_CSV), f"MISSING: {INVENTORY_CSV}"
    inv = pd.read_csv(INVENTORY_CSV)
    print(f"  data_inventory.csv shape: {inv.shape}")

    if inv.shape[0] != 128382:
        print(f"  ⚠ expected 128,382 rows, got {inv.shape[0]} — investigate before continuing")

    required = {"sample_name", "mp_id", "center_element", "poscar_path", "split"}
    missing = required - set(inv.columns)
    if missing:
        print(f"  ✗ MISSING COLUMNS: {missing}")
        print(f"    actual columns: {list(inv.columns)}")
        raise SystemExit(1)
    print(f"  ✓ all required columns present: {sorted(required)}")

    splits = dict(inv["split"].value_counts())
    print(f"  split distribution: {splits}")
    assert set(splits.keys()) == {"train", "val", "test", "holdout"}, \
        f"Unexpected split values: {set(splits.keys())}"

    # sanity: sample POSCAR readable
    sample_row = inv.iloc[0]
    sample_poscar = sample_row["poscar_path"]
    if not os.path.isfile(sample_poscar):
        print(f"  ✗ sample POSCAR NOT FOUND: {sample_poscar}")
        raise SystemExit(1)
    print(f"  ✓ sample POSCAR exists: {sample_poscar}")

    # center_element coverage
    n_elements = inv["center_element"].nunique()
    print(f"  unique center_elements: {n_elements}")

    # mp_id stats
    n_mpid = inv["mp_id"].nunique()
    print(f"  unique mp_ids: {n_mpid}")
    expected_cache_hit_rate = 1.0 - n_mpid / len(inv)
    print(f"  expected primitive-cache hit rate: {expected_cache_hit_rate:.1%}")

    return inv


def pick_diverse_samples(inv):
    print()
    print("=" * 72)
    print("CHECK 2: Pick 5 diverse samples from train split")
    print("=" * 72)

    train_inv = inv[inv["split"] == "train"]
    picked = []
    for elem in TARGET_ELEMENTS:
        m = train_inv[train_inv["center_element"] == elem]
        if len(m) == 0:
            print(f"  ⚠ {elem}: not in train split, skipping")
            continue
        row = m.iloc[0]
        picked.append(row)
        print(f"  ✓ {elem}: {row['sample_name']}  (mp_id={row['mp_id']})")

    assert len(picked) >= 3, f"Only picked {len(picked)} — need ≥3 for meaningful timing"
    return picked


def run_pipeline_check(samples):
    print()
    print("=" * 72)
    print("CHECK 3: Single-sample pipeline")
    print("=" * 72)

    times_ok = []
    for row in samples:
        print(f"\n  --- {row['sample_name']}  (center={row['center_element']}) ---")
        distances, species_Z, n_center_sites, status, elapsed = process_one_sample(row)

        print(f"    status:           {status}")
        print(f"    elapsed:          {elapsed:.3f} s")
        print(f"    n_center_sites:   {n_center_sites}")

        if status != "ok":
            print(f"    ✗ FAILED — skipping timing for this sample")
            continue

        times_ok.append(elapsed)
        print(f"    n_neighbors(10Å): {len(distances)}")
        print(f"    first distance:   {distances[0]:.4f} Å")
        if len(distances) >= 20:
            print(f"    d[19] (20th nbr): {distances[19]:.4f} Å")
        else:
            print(f"    ⚠ only {len(distances)} neighbors, <20")
        print(f"    first 25 d (Å):   {np.round(distances[:25], 3).tolist()}")

        # sanity checks (abort if any fail — these are invariants)
        assert np.all(np.diff(distances) >= 0), f"distances not monotonic!"
        assert distances[0] > 1.0, f"first distance suspiciously small: {distances[0]}"
        assert len(distances) >= 5, f"too few neighbors: {len(distances)}"

    return times_ok


def recommend_parallelism(times):
    print()
    print("=" * 72)
    print("CHECK 4: Time estimate & parallelism recommendation")
    print("=" * 72)

    mean_t   = float(np.mean(times))
    median_t = float(np.median(times))
    p95_t    = float(np.percentile(times, 95))
    max_t    = float(np.max(times))
    print(f"  per-sample time  mean={mean_t:.3f}s  median={median_t:.3f}s  p95={p95_t:.3f}s  max={max_t:.3f}s")

    # Use median to de-emphasize first-sample cold-cache outlier
    total = 128382
    serial_min = median_t * total / 60
    print(f"  serial estimate (median × 128,382): {serial_min:.1f} min")

    # primitive_cache expected hit rate ~68% → saves pymatgen spglib call
    # but NOT the Structure.from_file + get_neighbors. Conservative 35% speedup.
    cached_serial_min = serial_min * 0.65
    print(f"  with mp_id primitive cache (~35% faster): ~{cached_serial_min:.1f} min")

    n_cpu = multiprocessing.cpu_count()
    recommended = max(1, min(8, n_cpu - 1))
    # 15% overhead for IPC
    parallel_min = cached_serial_min / recommended * 1.15
    print(f"  with {recommended} processes + cache: ~{parallel_min:.1f} min")
    print(f"  detected CPU count: {n_cpu}")

    print()
    if median_t < 0.2:
        print(f"  → RECOMMENDATION: single-threaded OK (<7 min projected)")
        print(f"    mode = 'serial'")
    elif median_t < 1.0:
        print(f"  → RECOMMENDATION: multiprocessing.Pool with {recommended} workers")
        print(f"    mode = 'parallel', processes = {recommended}")
    else:
        print(f"  → RECOMMENDATION: REQUIRED multiprocessing.Pool with {recommended} workers")
        print(f"    (serial would take >35 min)")
        print(f"    mode = 'parallel_required', processes = {recommended}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    log_buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, log_buf)

    try:
        inv = check_step1_outputs()
        samples = pick_diverse_samples(inv)
        times = run_pipeline_check(samples)
        if len(times) >= 3:
            recommend_parallelism(times)
        else:
            print("\n  ✗ Not enough successful samples to estimate time — investigate failures")
    finally:
        sys.stdout = sys.__stdout__
        log_path = os.path.join(STEP25_DIR, "step2_5a_precheck.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print()
        print("=" * 72)
        print("PRECHECK DONE")
        print("=" * 72)
        print(f"  log saved to: {log_path}")
        print(f"  → paste the console output back to Claude to proceed.")


if __name__ == "__main__":
    main()
