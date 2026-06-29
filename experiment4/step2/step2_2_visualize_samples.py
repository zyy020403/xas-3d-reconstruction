"""
step2_2_visualize_samples.py
============================
Visual QC: plot native (CSV) vs resampled (pkl) spectra for 5 samples across
diverse center elements — O (light), Fe (first-row TM), Cu (first-row TM late),
La (lanthanide), U (actinide).

Output: step2_qc_samples.png — 2 rows × 5 cols
  row 0: xmu native curve + resampled dots
  row 1: chi1 native curve + resampled dots

The human operator (you) must eyeball this PNG and confirm:
  - resampled dots hug the native curve inside the window
  - outside the native range, resampled is a flat plateau (constant extrap) — OK
  - no wild oscillation, no NaN holes, no vertical spikes
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- paths ----------
EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR = os.path.join(EXP4_ROOT, "step1")
STEP2_DIR = os.path.join(EXP4_ROOT, "step2")

INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_PKL      = os.path.join(STEP1_DIR, "feff_features_imputed.pkl")

QC_ELEMENTS  = ["O", "Fe", "Cu", "La", "U"]
XMU_WINDOW   = (-50.0, 150.0)
XMU_N_POINTS = 150
CHI_K_RANGE  = (0.0, 12.0)
CHI_N_POINTS = 200


def pick_one(inv: pd.DataFrame, element: str, rng: np.random.Generator):
    """Pick one sample with given center_element, prefer train split."""
    sub = inv[(inv["center_element"] == element) & (inv["split"] == "train")]
    if sub.empty:
        sub = inv[inv["center_element"] == element]
    if sub.empty:
        return None
    return sub.iloc[int(rng.integers(0, len(sub)))]


def main() -> None:
    inv  = pd.read_csv(INVENTORY_CSV)
    feff = pd.read_pickle(FEFF_PKL)
    rng  = np.random.default_rng(20260423)

    # load all 4 pkls; we'll find each sample wherever it is
    loaded = {}
    for s in ["train", "val", "test", "holdout"]:
        p = os.path.join(STEP2_DIR, f"spectra_{s}.pkl")
        with open(p, "rb") as f:
            loaded[s] = pickle.load(f)
        print(f"[loaded] spectra_{s}.pkl  N={len(loaded[s]['sample_names'])}")

    def find_resampled(name: str):
        for s, obj in loaded.items():
            if name in obj["name_to_idx"]:
                idx = obj["name_to_idx"][name]
                return obj["xmu"][idx], obj["chi1"][idx], s
        raise KeyError(f"{name} not found in any split pkl")

    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    picks_log = []

    for col, elem in enumerate(QC_ELEMENTS):
        row = pick_one(inv, elem, rng)
        if row is None:
            axes[0, col].set_title(f"(no sample for {elem})")
            axes[1, col].set_title("")
            continue

        name = row["sample_name"]
        E0   = float(feff.loc[name, "E0"])

        xmu_raw = pd.read_csv(row["xmu_path"])
        chi_raw = pd.read_csv(row["chi_path"])

        xmu_re, chi_re, which_split = find_resampled(name)
        E_target = E0 + np.linspace(XMU_WINDOW[0], XMU_WINDOW[1], XMU_N_POINTS)
        k_target = np.linspace(CHI_K_RANGE[0], CHI_K_RANGE[1], CHI_N_POINTS)

        # ---- xmu subplot (top row) ----
        ax = axes[0, col]
        ax.plot(xmu_raw["x"], xmu_raw["y"], "-",
                lw=0.8, color="tab:blue", alpha=0.7, label="native")
        ax.plot(E_target, xmu_re, ".",
                ms=3.5, color="tab:red", label="resampled (150)")
        ax.axvline(E0, color="k", ls=":", lw=0.7, label=f"E0={E0:.1f}")
        ax.axvline(E0 + XMU_WINDOW[0], color="gray", ls="--", lw=0.5)
        ax.axvline(E0 + XMU_WINDOW[1], color="gray", ls="--", lw=0.5)
        ax.set_xlim(E0 + XMU_WINDOW[0] - 30, E0 + XMU_WINDOW[1] + 30)
        ax.set_title(f"{elem} — {which_split}\n{name[:40]}", fontsize=8)
        ax.set_xlabel("E (eV)", fontsize=8)
        ax.set_ylabel("μ(E)",   fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

        # ---- chi subplot (bottom row) ----
        ax = axes[1, col]
        ax.plot(chi_raw["k"], chi_raw["chi1"], "-",
                lw=0.8, color="tab:blue", alpha=0.7, label="native chi1")
        ax.plot(k_target, chi_re, ".",
                ms=3.5, color="tab:red", label="resampled (200)")
        ax.axvline(CHI_K_RANGE[0], color="gray", ls="--", lw=0.5)
        ax.axvline(CHI_K_RANGE[1], color="gray", ls="--", lw=0.5)
        ax.set_xlim(-1.0, 14.0)
        ax.set_xlabel("k (Å⁻¹)", fontsize=8)
        ax.set_ylabel("k¹·χ(k)", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

        picks_log.append((elem, name, E0, which_split))

    fig.suptitle("Step 2 QC — native (thin line) vs resampled (red dots)  |  "
                 "dashed = window edges, dotted = E0",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(STEP2_DIR, "step2_qc_samples.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"\n[saved] {out}")
    print("\npicks:")
    print(f"  {'elem':>4s} | {'split':>7s} | {'E0 (eV)':>10s} | sample_name")
    print(f"  {'-'*4:>4s} | {'-'*7:>7s} | {'-'*10:>10s} | {'-'*60}")
    for elem, name, E0, s in picks_log:
        print(f"  {elem:>4s} | {s:>7s} | {E0:>10.2f} | {name}")


if __name__ == "__main__":
    main()
