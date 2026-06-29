# compare_spectra.py
# 本地 Windows 跑 — 对比 3 套 EXAFS 谱图
# A. 原始全晶胞      MP_all_EXAFS_only_chi_csv/{sn}_chi.csv  +  MP_all_EXAFS_only_csv/{sn}.csv
# B. 截取真实 FEFF   feff_inputs_all_xyz/{sn}_true/{chi.dat, xmu.dat}
# C. 截取预测 FEFF   feff_inputs_all_xyz/{sn}_pred/{chi.dat, xmu.dat}
#
# 用法: python compare_spectra.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
DIR_CHI_ORIG = os.path.join(DATA_ROOT, "MP_all_EXAFS_only_chi_csv", "MP_all_EXAFS_only_chi_csv")
DIR_XMU_ORIG = os.path.join(DATA_ROOT, "MP_all_EXAFS_only_csv",      "MP_all_EXAFS_only_csv")
DIR_FEFF     = os.path.join(DATA_ROOT, "feff_inputs_all_xyz")
OUT_DIR      = os.path.join(DATA_ROOT, "..", "spectra_comparison")
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv_2col(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    return df[cols[0]].values, df[cols[1]].values


def load_dat(path, x_col, y_col, min_cols):
    """Load FEFF .dat file (skip # lines), return (x, y) from given column indices."""
    if not os.path.exists(path):
        return None
    data = []
    with open(path) as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            parts = ls.split()
            try:
                row = [float(p) for p in parts]
                if len(row) >= min_cols:
                    data.append(row)
            except ValueError:
                continue
    if not data:
        return None
    arr = np.array(data)
    return arr[:, x_col], arr[:, y_col]


def align_resample(x1, y1, x2, y2, n=200):
    x_min = max(np.min(x1), np.min(x2))
    x_max = min(np.max(x1), np.max(x2))
    if x_max <= x_min:
        return None
    x = np.linspace(x_min, x_max, n)
    f1 = interp1d(x1, y1, kind="linear", fill_value="extrapolate")
    f2 = interp1d(x2, y2, kind="linear", fill_value="extrapolate")
    return x, f1(x), f2(x)


def metrics(y_ref, y_test):
    eps = 1e-12
    diff = y_test - y_ref
    R = float(np.sum(diff**2) / (np.sum(y_ref**2) + eps))
    P = float(np.corrcoef(y_ref, y_test)[0, 1]) if (np.std(y_ref) > eps and np.std(y_test) > eps) else 0.0
    RMSE = float(np.sqrt(np.mean(diff**2)))
    MAE = float(np.mean(np.abs(diff)))
    return {"R_factor": R, "Pearson": P, "RMSE": RMSE, "MAE": MAE}


def main():
    print("=" * 70)
    print("Spectra comparison — 3 sources (orig / true / pred)")
    print(f"  data: {DATA_ROOT}")
    print(f"  out:  {OUT_DIR}")
    print("=" * 70)

    # 1. find samples
    folders = [f for f in os.listdir(DIR_FEFF) if os.path.isdir(os.path.join(DIR_FEFF, f))]
    pred_set = {f.replace("_pred", "") for f in folders if f.endswith("_pred")}
    true_set = {f.replace("_true", "") for f in folders if f.endswith("_true")}
    samples = sorted(pred_set | true_set)
    print(f"\nFound {len(samples)} sample names in feff_inputs_all_xyz")

    # 2. load + compute metrics for every available comparison
    rows = []
    cache = {}
    for sn in samples:
        print(f"\n--- {sn} ---")
        chi_orig = load_csv_2col(os.path.join(DIR_CHI_ORIG, f"{sn}_chi.csv"))
        xmu_orig = load_csv_2col(os.path.join(DIR_XMU_ORIG, f"{sn}.csv"))
        # chi.dat: cols (E_rel, mu0, mu, chi) -> x=col0, y=col3
        chi_true = load_dat(os.path.join(DIR_FEFF, f"{sn}_true", "chi.dat"), 0, 3, 4) if sn in true_set else None
        chi_pred = load_dat(os.path.join(DIR_FEFF, f"{sn}_pred", "chi.dat"), 0, 3, 4) if sn in pred_set else None
        # xmu.dat: cols (E_abs, e_rel, k, mu0, chi, mu_smooth) -> x=col0, y=col5
        xmu_true = load_dat(os.path.join(DIR_FEFF, f"{sn}_true", "xmu.dat"), 0, 5, 6) if sn in true_set else None
        xmu_pred = load_dat(os.path.join(DIR_FEFF, f"{sn}_pred", "xmu.dat"), 0, 5, 6) if sn in pred_set else None

        cache[sn] = {
            "chi_orig": chi_orig, "xmu_orig": xmu_orig,
            "chi_true": chi_true, "xmu_true": xmu_true,
            "chi_pred": chi_pred, "xmu_pred": xmu_pred,
        }

        # 6 comparisons
        comps = [
            ("chi", "orig", "pred", chi_orig, chi_pred),
            ("chi", "orig", "true", chi_orig, chi_true),
            ("chi", "true", "pred", chi_true, chi_pred),
            ("xmu", "orig", "pred", xmu_orig, xmu_pred),
            ("xmu", "orig", "true", xmu_orig, xmu_true),
            ("xmu", "true", "pred", xmu_true, xmu_pred),
        ]
        for kind, ref, cmp_, ref_d, test_d in comps:
            row = {"sample_name": sn, "kind": kind, "reference": ref, "compared_to": cmp_}
            if ref_d is None or test_d is None:
                row.update({"R_factor": np.nan, "Pearson": np.nan,
                            "RMSE": np.nan, "MAE": np.nan, "n_pts": 0})
            else:
                aligned = align_resample(ref_d[0], ref_d[1], test_d[0], test_d[1])
                if aligned is None:
                    row.update({"R_factor": np.nan, "Pearson": np.nan,
                                "RMSE": np.nan, "MAE": np.nan, "n_pts": 0})
                else:
                    _, yr, yt = aligned
                    row.update(metrics(yr, yt))
                    row["n_pts"] = 200
                    print(f"  {kind} {ref} vs {cmp_}: R={row['R_factor']:.4f}  "
                          f"Pearson={row['Pearson']:.4f}")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    print(f"\nMetrics CSV: {os.path.join(OUT_DIR, 'metrics_summary.csv')}")

    # Aggregate
    agg = (df.dropna(subset=["R_factor"])
           .groupby(["kind", "reference", "compared_to"])
           [["R_factor", "Pearson", "RMSE", "MAE"]]
           .agg(["mean", "median", "count"]))
    print("\n=== AGGREGATE ===")
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, "metrics_aggregate.csv"))

    # 3. Plots — chi overlay
    print("\nPlotting chi overlay ...")
    plot_samples = [sn for sn in samples
                    if cache[sn]["chi_orig"] is not None and cache[sn]["chi_pred"] is not None]
    if plot_samples:
        n = len(plot_samples)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]
            if c["chi_orig"] is not None:
                ax.plot(c["chi_orig"][0], c["chi_orig"][1], label="A orig", color="black", lw=1.5)
            if c["chi_true"] is not None:
                ax.plot(c["chi_true"][0], c["chi_true"][1], label="B true 20", color="steelblue", lw=1.2, ls="--")
            if c["chi_pred"] is not None:
                ax.plot(c["chi_pred"][0], c["chi_pred"][1], label="C pred 20", color="orange", lw=1.2, ls=":")

            r_op = df[(df.sample_name == sn) & (df.kind == "chi") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            r_str = f"R(A,C)={r_op[0]:.3f}" if len(r_op) and not np.isnan(r_op[0]) else "R=N/A"
            ax.set_title(f"{sn}\n{r_str}", fontsize=8)
            ax.set_xlabel("k or E (a.u.)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("χ(k)" if i % ncols == 0 else "")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("EXAFS χ overlay: orig vs truncated-true vs truncated-pred", fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_chi.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_chi.png")

    # xmu overlay
    print("Plotting xmu overlay ...")
    plot_samples_x = [sn for sn in samples
                      if cache[sn]["xmu_orig"] is not None and cache[sn]["xmu_pred"] is not None]
    if plot_samples_x:
        n = len(plot_samples_x)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples_x):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]
            if c["xmu_orig"] is not None:
                ax.plot(c["xmu_orig"][0], c["xmu_orig"][1], label="A orig", color="black", lw=1.5)
            if c["xmu_true"] is not None:
                ax.plot(c["xmu_true"][0], c["xmu_true"][1], label="B true 20", color="steelblue", lw=1.2, ls="--")
            if c["xmu_pred"] is not None:
                ax.plot(c["xmu_pred"][0], c["xmu_pred"][1], label="C pred 20", color="orange", lw=1.2, ls=":")
            r_op = df[(df.sample_name == sn) & (df.kind == "xmu") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            r_str = f"R(A,C)={r_op[0]:.3f}" if len(r_op) and not np.isnan(r_op[0]) else "R=N/A"
            ax.set_title(f"{sn}\n{r_str}", fontsize=8)
            ax.set_xlabel("E (eV)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("μ(E)" if i % ncols == 0 else "")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("XANES μ(E) overlay: orig vs truncated-true vs truncated-pred", fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_xmu.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_xmu.png")

    # R-factor bar
    print("Plotting R-factor bar ...")
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(samples)*0.6), 8))
    for ax, kind, ylab in zip(axes, ["chi", "xmu"], ["χ(k)", "μ(E)"]):
        sub = df[df.kind == kind].dropna(subset=["R_factor"])
        if not len(sub):
            ax.set_title(f"{ylab}: no data")
            continue
        pivot = sub.pivot_table(index="sample_name", columns=["reference", "compared_to"], values="R_factor")
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"R-factor for {ylab} (lower = better match)")
        ax.set_ylabel("R-factor")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: fig_metrics_bar.png")

    print("\n" + "=" * 70)
    print(f"DONE. {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
