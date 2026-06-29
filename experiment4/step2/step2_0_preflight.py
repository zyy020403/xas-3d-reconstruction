"""
step2_0_preflight.py
====================
Pre-flight sanity checks before Step 2 main preprocessing.

Runs the 3 checks from STEP2_SUBAGENT_HANDOFF §2:
  1. Step 1 outputs exist and load correctly
  2. 3 random xmu/chi CSVs have expected format (columns, row count, monotonic x/k)
  3. 5 random E0 values are in sane range [10, 130000] eV

If any check fails, the script prints a clear [PREFLIGHT FAIL] message and exits
non-zero. Do NOT run step2_1_preprocess_spectra.py until this script passes.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# -------- paths --------
EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR = os.path.join(EXP4_ROOT, "step1")

INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_PKL      = os.path.join(STEP1_DIR, "feff_features_imputed.pkl")
SCALER_PKL    = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")

EXPECTED_N = 128_382
EXPECTED_SPLITS = {"train": 102_660, "val": 12_912, "test": 7_696, "holdout": 5_114}

# Element K-edge references (eV) for the Check-3 Q3 sanity range
K_EDGE_REF = {
    "H":    13.6, "C":    284.0, "O":    543.0,
    "Fe": 7112.0, "Cu":  8979.0, "U":  115606.0,
}


def fail(msg: str) -> None:
    print(f"[PREFLIGHT FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


# ---------- CHECK 1 ----------
def check_1_step1_outputs():
    print("\n=== Check 1: Step 1 outputs ===")

    for f in [INVENTORY_CSV, FEFF_PKL, SCALER_PKL]:
        if not os.path.isfile(f):
            fail(f"missing file: {f}")
        ok(f"exists: {os.path.basename(f)}")

    split_files = [
        "train_ids.txt", "val_ids.txt", "test_ids.txt", "holdout_ids.txt",
        "train_samples.csv", "val_samples.csv", "test_samples.csv", "holdout_samples.csv",
    ]
    for fname in split_files:
        p = os.path.join(STEP1_DIR, fname)
        if not os.path.isfile(p):
            fail(f"missing split artefact: {p}")
    ok(f"{len(split_files)} split artefacts present")

    inv = pd.read_csv(INVENTORY_CSV)
    if inv.shape[0] != EXPECTED_N:
        fail(f"inventory rows = {inv.shape[0]}, expected {EXPECTED_N}")
    ok(f"inventory shape = {inv.shape}")

    required_cols = {"sample_name", "mp_id", "center_element",
                     "chi_path", "xmu_path", "split"}
    missing = required_cols - set(inv.columns)
    if missing:
        fail(f"inventory missing columns: {missing}")
    ok("inventory has all required columns")

    # unique sample_name
    if not inv["sample_name"].is_unique:
        fail("inventory sample_name is not unique")
    ok("inventory sample_name is unique")

    # split counts
    split_counts = inv["split"].value_counts().to_dict()
    for s, n in EXPECTED_SPLITS.items():
        if split_counts.get(s) != n:
            fail(f"split {s}: got {split_counts.get(s)}, expected {n}")
    ok(f"split counts match: {dict(sorted(split_counts.items()))}")

    # feff
    feff = pd.read_pickle(FEFF_PKL)
    if feff.shape != (EXPECTED_N, 74):
        fail(f"feff shape = {feff.shape}, expected ({EXPECTED_N}, 74)")
    if not feff.index.is_unique:
        fail("feff.index is not unique")
    if "E0" not in feff.columns:
        fail("feff missing 'E0' column")
    # inventory sample_names must all be in feff.index
    missing_in_feff = set(inv["sample_name"]) - set(feff.index)
    if missing_in_feff:
        fail(f"{len(missing_in_feff)} inventory sample_names missing from feff "
             f"(first 3: {list(missing_in_feff)[:3]})")
    ok(f"feff shape = {feff.shape}, index unique, E0 column present, "
       f"covers all inventory sample_names")

    scaler = joblib.load(SCALER_PKL)
    ok(f"scaler loads ({type(scaler).__name__})")

    return inv, feff


# ---------- CHECK 2 ----------
def check_2_csv_format(inv: pd.DataFrame):
    print("\n=== Check 2: xmu/chi CSV format (3 random) ===")
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(inv), size=3, replace=False)

    for i in idxs:
        row = inv.iloc[i]
        name  = row["sample_name"]
        xmu_p = row["xmu_path"]
        chi_p = row["chi_path"]

        # ---- xmu ----
        if not os.path.isfile(xmu_p):
            fail(f"xmu not found: {xmu_p}")
        xmu = pd.read_csv(xmu_p)
        if list(xmu.columns) != ["x", "y"]:
            fail(f"{name} xmu columns = {list(xmu.columns)}, expected ['x','y']")
        if len(xmu) != 400:
            fail(f"{name} xmu rows = {len(xmu)}, expected 400")
        x_arr = xmu["x"].to_numpy()
        if not np.all(np.diff(x_arr) > 0):
            fail(f"{name} xmu 'x' not strictly monotonic increasing")
        print(f"  [xmu] {name}  shape={tuple(xmu.shape)}  "
              f"E ∈ [{x_arr[0]:.2f}, {x_arr[-1]:.2f}]  OK")

        # ---- chi ----
        if not os.path.isfile(chi_p):
            fail(f"chi not found: {chi_p}")
        chi = pd.read_csv(chi_p)
        if list(chi.columns) != ["k", "chi", "chi1", "chi2"]:
            fail(f"{name} chi columns = {list(chi.columns)}, "
                 f"expected ['k','chi','chi1','chi2']")
        if len(chi) != 400:
            fail(f"{name} chi rows = {len(chi)}, expected 400")
        k_arr = chi["k"].to_numpy()
        if not np.all(np.diff(k_arr) > 0):
            fail(f"{name} chi 'k' not strictly monotonic increasing")
        chi1_arr = chi["chi1"].to_numpy()
        if not np.isfinite(chi1_arr).all():
            fail(f"{name} chi1 has NaN/Inf")
        print(f"  [chi] {name}  shape={tuple(chi.shape)}  "
              f"k ∈ [{k_arr[0]:.3f}, {k_arr[-1]:.3f}]  "
              f"chi1 ∈ [{chi1_arr.min():.3g}, {chi1_arr.max():.3g}]  OK")

    ok("3 random CSV pairs pass format checks")


# ---------- CHECK 3 ----------
def check_3_E0_sanity(inv: pd.DataFrame, feff: pd.DataFrame):
    print("\n=== Check 3: E0 sanity (5 random) ===")
    rng = np.random.default_rng(7)
    idxs = rng.choice(len(inv), size=5, replace=False)

    bad_range = []
    for i in idxs:
        row = inv.iloc[i]
        name = row["sample_name"]
        elem = row["center_element"]
        E0   = float(feff.loc[name, "E0"])
        print(f"  {name}  elem={elem:>3s}  E0={E0:>10.2f} eV")
        if not (10.0 <= E0 <= 130_000.0):
            bad_range.append((name, elem, E0))
        if elem in K_EDGE_REF:
            ref = K_EDGE_REF[elem]
            rel = abs(E0 - ref) / max(ref, 1.0)
            if rel > 0.5:
                print(f"    [NOTE] {elem} K-edge ref ≈ {ref} eV, observed deviates {rel:.0%} "
                      f"(not necessarily a bug — FEFF E0 can drift from tabulated edge)")

    if bad_range:
        fail(f"E0 out of [10, 130000] eV for: {bad_range}")
    ok("all 5 E0 values in [10, 130000] eV")


# ---------- MAIN ----------
if __name__ == "__main__":
    inv, feff = check_1_step1_outputs()
    check_2_csv_format(inv)
    check_3_E0_sanity(inv, feff)
    print("\n[ALL PREFLIGHT CHECKS PASSED] safe to run step2_1_preprocess_spectra.py")
