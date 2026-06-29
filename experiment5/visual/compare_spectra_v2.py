# compare_spectra_v2.py
# 修正版:正确读 FEFF .dat 列 + 漂亮可视化
# ============================================================================
# 修正 v1 的 bug:
#   chi.dat 列:  k, chi, mag, phase    (col 0 = k, col 1 = chi)  v1 错读 col 3
#   xmu.dat 列:  omega, e, k, mu, mu0, chi  (col 0 = E, col 3 = mu)  v1 错读 col 5
#
# 用法: python compare_spectra_v2.py
# ============================================================================

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
OUT_DIR      = os.path.join(DATA_ROOT, "..", "spectra_comparison_v2")
os.makedirs(OUT_DIR, exist_ok=True)


def sanitize(x, y):
    """Drop NaN/Inf and absurdly large values (FEFF sometimes overflows)."""
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 1e10)
    if mask.sum() < 5:
        return None
    return x[mask], y[mask]


def load_csv_2col(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        return sanitize(df[cols[0]].values, df[cols[1]].values)
    except Exception as e:
        print(f"   load_csv fail: {e}")
        return None


def load_dat(path, x_col, y_col):
    """Load FEFF .dat, skip # lines."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            parts = ls.split()
            try:
                row = [float(p) for p in parts]
                if len(row) > max(x_col, y_col):
                    rows.append(row)
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.array(rows)
    return sanitize(arr[:, x_col], arr[:, y_col])


def load_chi_dat(folder):
    # chi.dat columns: k, chi, mag, phase  →  (k, chi)
    return load_dat(os.path.join(folder, "chi.dat"), x_col=0, y_col=1)


def load_xmu_dat(folder):
    # xmu.dat columns: omega, e, k, mu, mu0, chi  →  (E, mu)
    return load_dat(os.path.join(folder, "xmu.dat"), x_col=0, y_col=3)


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
    print("Spectra comparison v2 — corrected .dat column reading")
    print("=" * 70)

    folders = [f for f in os.listdir(DIR_FEFF) if os.path.isdir(os.path.join(DIR_FEFF, f))]
    pred_set = {f.replace("_pred", "") for f in folders if f.endswith("_pred")}
    true_set = {f.replace("_true", "") for f in folders if f.endswith("_true")}
    samples = sorted(pred_set | true_set)
    print(f"Found {len(samples)} sample names")

    rows = []
    cache = {}
    for sn in samples:
        chi_orig = load_csv_2col(os.path.join(DIR_CHI_ORIG, f"{sn}_chi.csv"))
        xmu_orig = load_csv_2col(os.path.join(DIR_XMU_ORIG, f"{sn}.csv"))
        chi_true = load_chi_dat(os.path.join(DIR_FEFF, f"{sn}_true")) if sn in true_set else None
        chi_pred = load_chi_dat(os.path.join(DIR_FEFF, f"{sn}_pred")) if sn in pred_set else None
        xmu_true = load_xmu_dat(os.path.join(DIR_FEFF, f"{sn}_true")) if sn in true_set else None
        xmu_pred = load_xmu_dat(os.path.join(DIR_FEFF, f"{sn}_pred")) if sn in pred_set else None

        cache[sn] = {"chi_orig": chi_orig, "xmu_orig": xmu_orig,
                     "chi_true": chi_true, "xmu_true": xmu_true,
                     "chi_pred": chi_pred, "xmu_pred": xmu_pred}

        # status print
        flags = []
        for k in ["chi_orig","chi_true","chi_pred","xmu_orig","xmu_true","xmu_pred"]:
            flags.append(f"{k}={'Y' if cache[sn][k] is not None else '-'}")
        print(f"  {sn}: " + "  ".join(flags))

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
                row.update({"R_factor": np.nan, "Pearson": np.nan, "RMSE": np.nan, "MAE": np.nan})
            else:
                aligned = align_resample(ref_d[0], ref_d[1], test_d[0], test_d[1])
                if aligned is None:
                    row.update({"R_factor": np.nan, "Pearson": np.nan, "RMSE": np.nan, "MAE": np.nan})
                else:
                    _, yr, yt = aligned
                    row.update(metrics(yr, yt))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

    # ─── Aggregate ────────────────────────────────────────────────────────
    agg = (df.dropna(subset=["R_factor"])
           .groupby(["kind", "reference", "compared_to"])
           [["R_factor", "Pearson", "RMSE", "MAE"]]
           .agg(["mean", "median", "count"]))
    print("\n=== AGGREGATE METRICS ===")
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, "metrics_aggregate.csv"))

    # ============================================================================
    # PLOT 1: chi(k) overlay grid
    # ============================================================================
    print("\nPlotting chi overlay ...")
    plot_samples_chi = [sn for sn in samples
                        if cache[sn]["chi_orig"] is not None or cache[sn]["chi_pred"] is not None]
    if plot_samples_chi:
        n = len(plot_samples_chi)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples_chi):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]
            if c["chi_orig"] is not None:
                ax.plot(c["chi_orig"][0], c["chi_orig"][1], label="orig (full)", color="black", lw=1.5, zorder=3)
            if c["chi_true"] is not None:
                ax.plot(c["chi_true"][0], c["chi_true"][1], label="true 20-atom", color="steelblue", lw=1.2, ls="--")
            if c["chi_pred"] is not None:
                ax.plot(c["chi_pred"][0], c["chi_pred"][1], label="pred 20-atom", color="orange", lw=1.2, ls=":")

            r_op = df[(df.sample_name == sn) & (df.kind == "chi") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            r_str = f"R(orig,pred)={r_op[0]:.3f}" if len(r_op) and not np.isnan(r_op[0]) else ""
            short_sn = sn.replace("__mp-", "_").replace("-EXAFS-", "_")[:30]
            ax.set_title(f"{short_sn}\n{r_str}", fontsize=8)
            ax.set_xlabel("k (Å⁻¹)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("χ(k)" if i % ncols == 0 else "")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 16)  # standard EXAFS k-range
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("EXAFS χ(k) overlay: original full crystal vs truncated true (20 atoms) vs truncated predicted (20 atoms)",
                     fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_chi.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_chi.png")

    # ============================================================================
    # PLOT 2: xmu(E) overlay grid — normalized to peak for visual comparison
    # ============================================================================
    print("Plotting xmu overlay ...")
    plot_samples_xmu = [sn for sn in samples
                        if cache[sn]["xmu_orig"] is not None or cache[sn]["xmu_pred"] is not None]
    if plot_samples_xmu:
        n = len(plot_samples_xmu)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples_xmu):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]

            # Each spectrum is normalized to its own max (for visual shape comparison)
            def norm(yarr):
                m = np.max(np.abs(yarr))
                return yarr / m if m > 1e-12 else yarr

            if c["xmu_orig"] is not None:
                E, mu = c["xmu_orig"]
                ax.plot(E, norm(mu), label="orig (full)", color="black", lw=1.5, zorder=3)
            if c["xmu_true"] is not None:
                E, mu = c["xmu_true"]
                ax.plot(E, norm(mu), label="true 20-atom", color="steelblue", lw=1.2, ls="--")
            if c["xmu_pred"] is not None:
                E, mu = c["xmu_pred"]
                ax.plot(E, norm(mu), label="pred 20-atom", color="orange", lw=1.2, ls=":")

            r_op = df[(df.sample_name == sn) & (df.kind == "xmu") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            r_str = f"R(orig,pred)={r_op[0]:.3f}" if len(r_op) and not np.isnan(r_op[0]) else ""
            short_sn = sn.replace("__mp-", "_").replace("-EXAFS-", "_")[:30]
            ax.set_title(f"{short_sn}\n{r_str}", fontsize=8)
            ax.set_xlabel("E (eV)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("μ(E) normalized" if i % ncols == 0 else "")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle("XANES μ(E) overlay (each spectrum self-normalized for shape comparison)",
                     fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_xmu.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_xmu.png")

    # ============================================================================
    # PLOT 3: Pearson summary heatmap (per sample × per comparison)
    # ============================================================================
    print("Plotting Pearson heatmap ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(samples)*0.3)))
    for ax, kind, ylab in zip(axes, ["chi", "xmu"], ["χ(k)", "μ(E)"]):
        sub = df[df.kind == kind]
        pivot = sub.pivot_table(index="sample_name", columns=["reference", "compared_to"],
                                values="Pearson")
        # Reorder columns so they read: orig-vs-pred, orig-vs-true, true-vs-pred
        col_order = [("orig", "pred"), ("orig", "true"), ("true", "pred")]
        existing = [c for c in col_order if c in pivot.columns]
        pivot = pivot[existing]
        # Friendly column labels
        pivot.columns = ["A vs C\n(orig vs pred)", "A vs B\n(orig vs true)",
                         "B vs C\n(true vs pred)"][:len(existing)]
        # Heatmap
        if pivot.size > 0:
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           vmin=-1, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([sn.replace("__mp-", "_")[:25] for sn in pivot.index],
                               fontsize=7)
            ax.set_title(f"Pearson correlation — {ylab}", fontsize=11)
            # Annotate cells
            for ii in range(pivot.shape[0]):
                for jj in range(pivot.shape[1]):
                    v = pivot.values[ii, jj]
                    if not np.isnan(v):
                        ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="black" if abs(v) < 0.6 else "white")
            plt.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_pearson_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: fig_pearson_heatmap.png")

    # ============================================================================
    # PLOT 4: Aggregate bar chart — mean Pearson for each comparison group
    # ============================================================================
    print("Plotting aggregate bar ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind, ylab in zip(axes, ["chi", "xmu"], ["χ(k)", "μ(E)"]):
        sub = df[df.kind == kind].dropna(subset=["Pearson"])
        gb = sub.groupby(["reference", "compared_to"])["Pearson"].agg(["mean", "std", "count"])
        if len(gb):
            labels = [f"{a} vs {b}" for (a, b) in gb.index]
            means = gb["mean"].values
            stds = gb["std"].values
            colors = ["#1f77b4", "#2ca02c", "#ff7f0e"][:len(labels)]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
            for i, (m, s, n) in enumerate(zip(means, stds, gb["count"].values)):
                ax.text(i, m + (s if not np.isnan(s) else 0) + 0.02,
                        f"{m:.3f}\n(n={n})", ha="center", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel("Mean Pearson r")
            ax.set_title(f"{ylab} — mean Pearson by comparison")
            ax.set_ylim(-0.2, 1.1)
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Spectral similarity (Pearson r): higher = more similar", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_aggregate_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: fig_aggregate_bar.png")

    print("\n" + "=" * 70)
    print(f"DONE. Outputs in: {OUT_DIR}")
    print("=" * 70)
    print("\n关键文件:")
    print("  fig_overlay_chi.png      — χ(k) 三套谱图叠加")
    print("  fig_overlay_xmu.png      — μ(E) 三套谱图叠加(自归一化便于看形状)")
    print("  fig_pearson_heatmap.png  — 每样本 Pearson 相关性热图")
    print("  fig_aggregate_bar.png    — 平均 Pearson 柱状图(导师最看这个)")
    print("  metrics_summary.csv      — 详细数据")
    print("  metrics_aggregate.csv    — 聚合统计")


if __name__ == "__main__":
    main()
