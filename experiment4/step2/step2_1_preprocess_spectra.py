"""
step2_1_preprocess_spectra.py
=============================
Step 2 main preprocessing for Experiment 4.

Reads 128,382 (xmu.csv, chi.csv) pairs listed in data_inventory.csv, interpolates
each pair to fixed-length arrays, and packs per-split pickle files for Step 3.

LOCKED parameters (do NOT modify):
  - xmu window: [E0-50, E0+150] eV, 150 points, np.interp (linear + constant extrap)
  - chi window: [0, 12] Å⁻¹,      200 points, uses the 'chi1' column
  - dtype: float32 for xmu / chi1 / E0; list[str] for sample_names
  - no scaling / normalisation of xmu or chi1

Outputs (all in experiment4\\step2\\):
  - spectra_train.pkl / spectra_val.pkl / spectra_test.pkl / spectra_holdout.pkl
  - step2_spectra_stats.csv
  - step2_extrapolation_log.csv
  - step2_summary.txt

Failure policy: if per-sample processing raises, we log the first 3, keep a
counter, and abort once failures > 10. No pkl is written on abort. We never
silently skip (Step 1 already filtered via chi_valid/xmu_valid).
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import psutil
    _PROC = psutil.Process()
    def mem_gb() -> float:
        return _PROC.memory_info().rss / 1e9
except ImportError:
    def mem_gb() -> float:
        return -1.0


# ---------- paths & constants ----------
EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR = os.path.join(EXP4_ROOT, "step1")
STEP2_DIR = os.path.join(EXP4_ROOT, "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_PKL      = os.path.join(STEP1_DIR, "feff_features_imputed.pkl")

# LOCKED 🔒
XMU_N_POINTS = 150
XMU_WINDOW   = (-50.0, 150.0)     # eV, relative to E0
CHI_N_POINTS = 200
CHI_K_RANGE  = (0.0, 12.0)        # Å⁻¹, absolute
CHI_COLUMN   = "chi1"             # NOT 'chi', NOT 'chi2'

SPLITS = ["train", "val", "test", "holdout"]
EXPECTED_COUNTS = {"train": 102_660, "val": 12_912, "test": 7_696, "holdout": 5_114}

# Pre-computed target grids
XMU_OFFSETS  = np.linspace(XMU_WINDOW[0], XMU_WINDOW[1], XMU_N_POINTS)   # (150,)
CHI_K_TARGET = np.linspace(CHI_K_RANGE[0], CHI_K_RANGE[1], CHI_N_POINTS)  # (200,)

MAX_FAILURES = 10     # abort threshold
STATS_REL_TOL = 0.20  # train vs {val,test,holdout} std relative diff warning threshold


# ---------- per-sample processing ----------
def process_xmu(xmu_path: str, E0: float):
    """Return (xmu_150 float32, left_extrap_bool, right_extrap_bool)."""
    raw = pd.read_csv(xmu_path)
    E  = raw["x"].to_numpy()
    mu = raw["y"].to_numpy()

    # defensive sort (FEFF output is already monotonic, but cheap insurance)
    order = np.argsort(E)
    E, mu = E[order], mu[order]

    E_lo = E0 + XMU_WINDOW[0]
    E_hi = E0 + XMU_WINDOW[1]
    left_extrap  = bool(E[0]  > E_lo)
    right_extrap = bool(E[-1] < E_hi)

    E_target = E0 + XMU_OFFSETS
    # np.interp: outside [E[0], E[-1]] it uses fp[0] / fp[-1] as constants — what we want
    xmu_150 = np.interp(E_target, E, mu).astype(np.float32)
    return xmu_150, left_extrap, right_extrap


def process_chi(chi_path: str):
    """Return (chi_200 float32, right_extrap_bool)."""
    raw = pd.read_csv(chi_path)
    k    = raw["k"].to_numpy()
    chi1 = raw[CHI_COLUMN].to_numpy()

    # drop negative k (numeric artefacts), then defensive sort
    mask = k >= 0.0
    k, chi1 = k[mask], chi1[mask]
    order = np.argsort(k)
    k, chi1 = k[order], chi1[order]

    right_extrap = bool(k[-1] < CHI_K_RANGE[1])

    chi_200 = np.interp(CHI_K_TARGET, k, chi1).astype(np.float32)
    return chi_200, right_extrap


# ---------- main ----------
def main() -> None:
    t_start = time.time()
    print(f"[start] mem={mem_gb():.2f} GB")

    inv = pd.read_csv(INVENTORY_CSV)
    assert inv.shape[0] == 128_382, f"inventory rows = {inv.shape[0]}"
    print(f"[loaded inventory] {inv.shape}, mem={mem_gb():.2f} GB")

    feff = pd.read_pickle(FEFF_PKL)
    assert feff.shape == (128_382, 74), f"feff shape = {feff.shape}"
    assert "E0" in feff.columns
    E0_map = feff["E0"].to_dict()   # dict[sample_name -> E0], faster lookup than .loc in a tight loop
    del feff
    print(f"[loaded feff E0] mem={mem_gb():.2f} GB")

    # per-split buckets (lists grow during loop; stacked once at the end)
    buckets = {s: {"sample_names": [], "xmu": [], "chi1": [], "E0": []} for s in SPLITS}
    extrap  = {s: {"xmu_left": 0, "xmu_right": 0, "chi_right": 0} for s in SPLITS}
    failures: list[tuple[str, str]] = []

    # tqdm over itertuples — fastest iteration over a DataFrame
    for i, row in enumerate(tqdm(inv.itertuples(index=False),
                                 total=len(inv),
                                 desc="preprocess",
                                 unit="sample")):
        name  = row.sample_name
        split = row.split

        try:
            E0 = E0_map[name]
        except KeyError:
            failures.append((name, "E0 missing from feff map"))
            if len(failures) > MAX_FAILURES:
                raise RuntimeError(f"{len(failures)} failures; aborting")
            continue

        try:
            xmu_150, lx, rx = process_xmu(row.xmu_path, float(E0))
            chi_200, rc     = process_chi(row.chi_path)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            if len(failures) <= 3:
                print(f"\n[FAIL {len(failures)}] {name}: {type(e).__name__}: {e}")
            if len(failures) > MAX_FAILURES:
                raise RuntimeError(f"{len(failures)} failures; aborting — see log above")
            continue

        b = buckets[split]
        b["sample_names"].append(name)
        b["xmu"].append(xmu_150)
        b["chi1"].append(chi_200)
        b["E0"].append(np.float32(E0))

        ex = extrap[split]
        ex["xmu_left"]  += int(lx)
        ex["xmu_right"] += int(rx)
        ex["chi_right"] += int(rc)

        if (i + 1) % 10_000 == 0:
            print(f"  [{i+1:>7d}/{len(inv)}]  mem={mem_gb():.2f} GB  "
                  f"failures={len(failures)}")

    if failures:
        raise RuntimeError(
            f"processing failed for {len(failures)} samples; NOT writing any pkl. "
            f"First 5:\n  " + "\n  ".join(f"{n}: {err}" for n, err in failures[:5])
        )

    wall = time.time() - t_start
    print(f"\n[all processed] mem={mem_gb():.2f} GB, elapsed={wall:.1f}s "
          f"({wall/60:.1f} min)")

    # ---- stack per-split arrays once ----
    arrs = {}
    for s in SPLITS:
        b = buckets[s]
        n_got = len(b["sample_names"])
        if n_got != EXPECTED_COUNTS[s]:
            raise RuntimeError(
                f"split {s}: got {n_got}, expected {EXPECTED_COUNTS[s]}"
            )
        xmu_arr = np.stack(b["xmu"],  axis=0).astype(np.float32, copy=False)
        chi_arr = np.stack(b["chi1"], axis=0).astype(np.float32, copy=False)
        e0_arr  = np.asarray(b["E0"],   dtype=np.float32)

        assert xmu_arr.shape == (n_got, XMU_N_POINTS), f"{s} xmu shape {xmu_arr.shape}"
        assert chi_arr.shape == (n_got, CHI_N_POINTS), f"{s} chi shape {chi_arr.shape}"
        assert np.isfinite(xmu_arr).all(), f"{s} xmu has NaN/Inf"
        assert np.isfinite(chi_arr).all(), f"{s} chi1 has NaN/Inf"
        assert np.isfinite(e0_arr).all(),  f"{s} E0 has NaN/Inf"

        arrs[s] = {"sample_names": b["sample_names"],
                   "xmu": xmu_arr, "chi1": chi_arr, "E0": e0_arr}
        # free the growing lists
        buckets[s]["xmu"]  = None
        buckets[s]["chi1"] = None
        buckets[s]["E0"]   = None

    # ---- cross-split uniqueness ----
    all_names = []
    for s in SPLITS:
        all_names.extend(arrs[s]["sample_names"])
    if len(all_names) != 128_382:
        raise RuntimeError(f"sum of split names = {len(all_names)}, expected 128382")
    if len(set(all_names)) != 128_382:
        raise RuntimeError(f"duplicates across splits: {len(all_names) - len(set(all_names))}")
    print(f"[cross-split check] 128,382 unique sample_names, zero overlap ✓")

    # ---- dump 4 pkl ----
    print()
    for s in SPLITS:
        a = arrs[s]
        name_to_idx = {n: i for i, n in enumerate(a["sample_names"])}
        assert len(name_to_idx) == len(a["sample_names"]), f"{s} duplicate sample_names"

        obj = {
            "sample_names": a["sample_names"],   # list[str]
            "xmu":          a["xmu"],            # (N, 150) float32
            "chi1":         a["chi1"],           # (N, 200) float32
            "name_to_idx":  name_to_idx,         # dict[str, int]
            "E0":           a["E0"],             # (N,) float32
            "meta": {
                "xmu_window_eV":  list(XMU_WINDOW),
                "xmu_n_points":   XMU_N_POINTS,
                "chi_k_range":    list(CHI_K_RANGE),
                "chi_n_points":   CHI_N_POINTS,
                "chi_column":     CHI_COLUMN,
                "dtype":          "float32",
                "interp":         "np.interp (linear, constant extrapolation)",
            },
        }

        out = os.path.join(STEP2_DIR, f"spectra_{s}.pkl")
        with open(out, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        size_mb = os.path.getsize(out) / 1e6
        print(f"  -> {os.path.basename(out):24s}  N={len(a['sample_names']):>6d}  "
              f"{size_mb:7.1f} MB")

    # ---- stats CSV ----
    stats_rows = []
    for s in SPLITS:
        x = arrs[s]["xmu"]
        c = arrs[s]["chi1"]
        stats_rows.append({
            "split":     s,
            "n_samples": x.shape[0],
            "xmu_mean":  float(x.mean()),
            "xmu_std":   float(x.std()),
            "xmu_min":   float(x.min()),
            "xmu_max":   float(x.max()),
            "chi1_mean": float(c.mean()),
            "chi1_std":  float(c.std()),
            "chi1_min":  float(c.min()),
            "chi1_max":  float(c.max()),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(STEP2_DIR, "step2_spectra_stats.csv"), index=False)
    print("\n[stats]")
    print(stats_df.to_string(index=False))

    # ---- extrap CSV ----
    extrap_rows = []
    totals = {"xmu_left": 0, "xmu_right": 0, "chi_right": 0}
    for s in SPLITS:
        extrap_rows.append({
            "split":             s,
            "xmu_left_extrap":   extrap[s]["xmu_left"],
            "xmu_right_extrap":  extrap[s]["xmu_right"],
            "chi_right_extrap":  extrap[s]["chi_right"],
        })
        for k in totals:
            totals[k] += extrap[s][k]
    extrap_rows.append({
        "split": "TOTAL",
        "xmu_left_extrap":  totals["xmu_left"],
        "xmu_right_extrap": totals["xmu_right"],
        "chi_right_extrap": totals["chi_right"],
    })
    extrap_df = pd.DataFrame(extrap_rows)
    extrap_df.to_csv(os.path.join(STEP2_DIR, "step2_extrapolation_log.csv"), index=False)
    print("\n[extrapolation]")
    print(extrap_df.to_string(index=False))

    # ---- warnings ----
    warnings: list[str] = []
    if totals["xmu_right"] > 5000:
        warnings.append(f"xmu_right_extrap total = {totals['xmu_right']} > 5000")
    if totals["chi_right"] > 5000:
        warnings.append(f"chi_right_extrap total = {totals['chi_right']} > 5000")

    train_xmu_std = stats_df.loc[stats_df["split"] == "train", "xmu_std"].iloc[0]
    train_chi_std = stats_df.loc[stats_df["split"] == "train", "chi1_std"].iloc[0]
    for s in ["val", "test", "holdout"]:
        sx = stats_df.loc[stats_df["split"] == s, "xmu_std"].iloc[0]
        sc = stats_df.loc[stats_df["split"] == s, "chi1_std"].iloc[0]
        if train_xmu_std > 0:
            rx = abs(sx - train_xmu_std) / train_xmu_std
            if rx > STATS_REL_TOL:
                warnings.append(f"{s} xmu_std {sx:.4g} deviates {rx:.1%} from train "
                                f"{train_xmu_std:.4g} (> {STATS_REL_TOL:.0%} tol)")
        if train_chi_std > 0:
            rc = abs(sc - train_chi_std) / train_chi_std
            if rc > STATS_REL_TOL:
                warnings.append(f"{s} chi1_std {sc:.4g} deviates {rc:.1%} from train "
                                f"{train_chi_std:.4g} (> {STATS_REL_TOL:.0%} tol)")

    # ---- summary.txt ----
    lines = []
    lines.append("=" * 60)
    lines.append("Step 2 summary")
    lines.append("=" * 60)
    lines.append(f"wall-clock         : {wall:.1f} s ({wall/60:.1f} min)")
    lines.append(f"samples processed  : {sum(len(arrs[s]['sample_names']) for s in SPLITS)} / 128382")
    lines.append(f"failures           : {len(failures)}")
    lines.append("")
    lines.append("-- per-split sizes --")
    for s in SPLITS:
        out = os.path.join(STEP2_DIR, f"spectra_{s}.pkl")
        mb  = os.path.getsize(out) / 1e6
        lines.append(f"  {s:7s}: N={len(arrs[s]['sample_names']):>6d}  "
                     f"(expected {EXPECTED_COUNTS[s]})  pkl={mb:.1f} MB")
    lines.append("")
    lines.append("-- stats --")
    lines.append(stats_df.to_string(index=False))
    lines.append("")
    lines.append("-- extrapolation --")
    lines.append(extrap_df.to_string(index=False))
    lines.append("")
    lines.append("-- warnings --")
    if warnings:
        lines.extend(f"  [WARN] {w}" for w in warnings)
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("-- locked params --")
    lines.append(f"  xmu window : [E0{XMU_WINDOW[0]:+g}, E0{XMU_WINDOW[1]:+g}] eV, N={XMU_N_POINTS}")
    lines.append(f"  chi window : k ∈ {list(CHI_K_RANGE)} Å⁻¹, N={CHI_N_POINTS}, column='{CHI_COLUMN}'")
    lines.append(f"  dtype=float32, interp=np.interp (linear, constant extrapolation)")
    lines.append("=" * 60)
    summary = "\n".join(lines)

    with open(os.path.join(STEP2_DIR, "step2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\n[done] total mem={mem_gb():.2f} GB, elapsed={wall:.1f}s")


if __name__ == "__main__":
    main()
