# compare_spectra_v3.py
# 修正 v2 的展示问题: 加 outlier 剔除 + skipped 列表
# ============================================================================
# v3 新加:
#   - 谱图绝对值 > 100(任何谱图,包括 chi/mu)直接判 outlier(FEFF 数值溢出)
#   - 谱图最大幅度 / 中值幅度 > 100(尖峰震荡 = numerical instability)
#   - 输出 skipped_samples.csv 记录哪些样本被剔除 + 原因
#   - 主图只画 clean 样本,outlier 样本单独输出可选小图便于检视
#
# 用法: python compare_spectra_v3.py
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
OUT_DIR      = os.path.join(DATA_ROOT, "..", "spectra_comparison_v3_clean")
os.makedirs(OUT_DIR, exist_ok=True)

# Outlier thresholds — agresive enough to catch FEFF overflow but won't kill normal noisy spectra
MAX_ABS_VALUE  = 100.0   # any |y| > 100 → outlier (normal chi: ~1, normal mu: ~1-2)
MAX_PEAK_RATIO = 50.0    # max(|y|) / median(|y|+ε) > 50 → numerical instability


def is_spectrum_sane(y):
    """Return (is_sane, reason)."""
    if y is None:
        return False, "missing"
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        return False, f"NaN/Inf present"
    max_abs = np.max(np.abs(y))
    if max_abs > MAX_ABS_VALUE:
        return False, f"max|y|={max_abs:.2e} > {MAX_ABS_VALUE} (FEFF overflow)"
    med_abs = np.median(np.abs(y)) + 1e-9
    ratio = max_abs / med_abs
    if ratio > MAX_PEAK_RATIO:
        return False, f"peak/median={ratio:.1f} > {MAX_PEAK_RATIO} (numerical spike)"
    return True, "ok"


def sanitize(x, y):
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
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
    except Exception:
        return None


def load_dat(path, x_col, y_col):
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
    return load_dat(os.path.join(folder, "chi.dat"), x_col=0, y_col=1)


def load_xmu_dat(folder):
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
    print("Spectra comparison v3 — with outlier filtering")
    print(f"  thresholds: max|y|<{MAX_ABS_VALUE}, peak/median<{MAX_PEAK_RATIO}")
    print("=" * 70)

    folders = [f for f in os.listdir(DIR_FEFF) if os.path.isdir(os.path.join(DIR_FEFF, f))]
    pred_set = {f.replace("_pred", "") for f in folders if f.endswith("_pred")}
    true_set = {f.replace("_true", "") for f in folders if f.endswith("_true")}
    samples = sorted(pred_set | true_set)
    print(f"\nFound {len(samples)} sample names")

    # === Load + sanity check ===
    cache = {}
    skip_log = []
    for sn in samples:
        chi_orig = load_csv_2col(os.path.join(DIR_CHI_ORIG, f"{sn}_chi.csv"))
        xmu_orig = load_csv_2col(os.path.join(DIR_XMU_ORIG, f"{sn}.csv"))
        chi_true = load_chi_dat(os.path.join(DIR_FEFF, f"{sn}_true")) if sn in true_set else None
        chi_pred = load_chi_dat(os.path.join(DIR_FEFF, f"{sn}_pred")) if sn in pred_set else None
        xmu_true = load_xmu_dat(os.path.join(DIR_FEFF, f"{sn}_true")) if sn in true_set else None
        xmu_pred = load_xmu_dat(os.path.join(DIR_FEFF, f"{sn}_pred")) if sn in pred_set else None

        # Apply sanity checks per spectrum
        spectra_status = {}
        for name, spec in [("chi_orig", chi_orig), ("chi_true", chi_true), ("chi_pred", chi_pred),
                           ("xmu_orig", xmu_orig), ("xmu_true", xmu_true), ("xmu_pred", xmu_pred)]:
            if spec is None:
                spectra_status[name] = (None, "missing")
                continue
            sane, reason = is_spectrum_sane(spec[1])
            if sane:
                spectra_status[name] = (spec, "ok")
            else:
                spectra_status[name] = (None, f"OUTLIER: {reason}")
                skip_log.append({"sample_name": sn, "spectrum": name, "reason": reason})

        cache[sn] = {k: v[0] for k, v in spectra_status.items()}

        # Quick status print
        flags = []
        for k, (spec, status) in spectra_status.items():
            mark = "✓" if status == "ok" else ("-" if status == "missing" else "✗")
            flags.append(f"{k.replace('_','·'):8s}{mark}")
        print(f"  {sn[:40]:40s}  " + " ".join(flags))

    skip_df = pd.DataFrame(skip_log)
    skip_df.to_csv(os.path.join(OUT_DIR, "skipped_spectra.csv"), index=False)
    print(f"\nSkipped (outlier or missing): {len(skip_log)} spectrum entries → skipped_spectra.csv")

    # === Compute metrics ===
    rows = []
    for sn, c in cache.items():
        comps = [
            ("chi", "orig", "pred", c["chi_orig"], c["chi_pred"]),
            ("chi", "orig", "true", c["chi_orig"], c["chi_true"]),
            ("chi", "true", "pred", c["chi_true"], c["chi_pred"]),
            ("xmu", "orig", "pred", c["xmu_orig"], c["xmu_pred"]),
            ("xmu", "orig", "true", c["xmu_orig"], c["xmu_true"]),
            ("xmu", "true", "pred", c["xmu_true"], c["xmu_pred"]),
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

    agg = (df.dropna(subset=["R_factor"])
           .groupby(["kind", "reference", "compared_to"])
           [["R_factor", "Pearson", "RMSE", "MAE"]]
           .agg(["mean", "median", "count"]))
    print("\n=== AGGREGATE METRICS (after outlier filtering) ===")
    print(agg.to_string())
    agg.to_csv(os.path.join(OUT_DIR, "metrics_aggregate.csv"))

    # === PLOT 1: chi(k) overlay (CLEAN samples only) ===
    print("\nPlotting chi overlay (clean) ...")
    plot_samples = [sn for sn in samples
                    if cache[sn]["chi_orig"] is not None and cache[sn]["chi_pred"] is not None]
    if plot_samples:
        n = len(plot_samples)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]
            if c["chi_orig"] is not None:
                ax.plot(c["chi_orig"][0], c["chi_orig"][1], label="orig", color="black", lw=1.5, zorder=3)
            if c["chi_true"] is not None:
                ax.plot(c["chi_true"][0], c["chi_true"][1], label="true 20", color="steelblue", lw=1.2, ls="--")
            if c["chi_pred"] is not None:
                ax.plot(c["chi_pred"][0], c["chi_pred"][1], label="pred 20", color="orange", lw=1.2, ls=":")

            r_op = df[(df.sample_name == sn) & (df.kind == "chi") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            p_op = df[(df.sample_name == sn) & (df.kind == "chi") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["Pearson"].values
            metric_str = ""
            if len(r_op) and not np.isnan(r_op[0]):
                metric_str = f"R={r_op[0]:.2f}, ρ={p_op[0]:.2f}"
            short_sn = sn.replace("__mp-", "_").replace("-EXAFS-", "_")[:30]
            ax.set_title(f"{short_sn}\n{metric_str}", fontsize=8)
            ax.set_xlabel("k (Å⁻¹)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("χ(k)" if i % ncols == 0 else "")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 16)
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"EXAFS χ(k) overlay — clean samples only (outliers removed: {len(skip_log)} entries)",
                     fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_chi.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_chi.png ({n} samples)")

    # === PLOT 2: xmu(E) normalized overlay ===
    print("Plotting xmu overlay (clean, normalized) ...")
    plot_samples_x = [sn for sn in samples
                      if cache[sn]["xmu_orig"] is not None and cache[sn]["xmu_pred"] is not None]
    if plot_samples_x:
        n = len(plot_samples_x)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.2*nrows), squeeze=False)
        for i, sn in enumerate(plot_samples_x):
            ax = axes[i // ncols][i % ncols]
            c = cache[sn]
            def norm(y):
                m = np.max(np.abs(y))
                return y / m if m > 1e-12 else y
            if c["xmu_orig"] is not None:
                E, mu = c["xmu_orig"]
                ax.plot(E, norm(mu), label="orig", color="black", lw=1.5, zorder=3)
            if c["xmu_true"] is not None:
                E, mu = c["xmu_true"]
                ax.plot(E, norm(mu), label="true 20", color="steelblue", lw=1.2, ls="--")
            if c["xmu_pred"] is not None:
                E, mu = c["xmu_pred"]
                ax.plot(E, norm(mu), label="pred 20", color="orange", lw=1.2, ls=":")

            r_op = df[(df.sample_name == sn) & (df.kind == "xmu") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["R_factor"].values
            p_op = df[(df.sample_name == sn) & (df.kind == "xmu") &
                      (df.reference == "orig") & (df.compared_to == "pred")]["Pearson"].values
            metric_str = ""
            if len(r_op) and not np.isnan(r_op[0]):
                metric_str = f"R={r_op[0]:.3f}, ρ={p_op[0]:.2f}"
            short_sn = sn.replace("__mp-", "_").replace("-EXAFS-", "_")[:30]
            ax.set_title(f"{short_sn}\n{metric_str}", fontsize=8)
            ax.set_xlabel("E (eV)" if i // ncols == nrows-1 else "")
            ax.set_ylabel("μ(E) norm." if i % ncols == 0 else "")
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
        for j in range(i+1, nrows*ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"XANES μ(E) overlay — clean samples only, self-normalized",
                     fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_overlay_xmu.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: fig_overlay_xmu.png ({n} samples)")

    # === PLOT 3: Pearson aggregate bar chart ===
    print("Plotting aggregate Pearson bar ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind, ylab in zip(axes, ["chi", "xmu"], ["EXAFS χ(k)", "XANES μ(E)"]):
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
                ax.text(i, m + (s if not np.isnan(s) else 0) + 0.03,
                        f"{m:.3f}\n(n={n})", ha="center", fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel("Mean Pearson r", fontsize=11)
            ax.set_title(f"{ylab}", fontsize=12)
            ax.set_ylim(-0.2, 1.15)
            ax.axhline(0, color="black", lw=0.5)
            ax.axhline(0.7, color="red", lw=0.7, ls="--", alpha=0.5, label="r=0.7 strong")
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Spectral similarity (Pearson r): higher = more similar  [outliers removed]",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_aggregate_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: fig_aggregate_bar.png")

    print("\n" + "=" * 70)
    print(f"DONE. Outputs in: {OUT_DIR}")
    print("=" * 70)
    print("\n关键文件:")
    print(f"  fig_overlay_chi.png    — χ(k) 三套谱图叠加(只 clean 样本)")
    print(f"  fig_overlay_xmu.png    — μ(E) 三套谱图叠加(只 clean 样本)")
    print(f"  fig_aggregate_bar.png  — 平均 Pearson 柱状图(组会用这张)")
    print(f"  skipped_spectra.csv    — 被剔除的样本清单 + 原因")
    print(f"  metrics_summary.csv    — 详细数据")
    print(f"  metrics_aggregate.csv  — 聚合统计")


if __name__ == "__main__":
    main()
