# step6_visualize.py
# DiffCSP-Exp4 Step 6 — 结果可视化(Step6Agent, 基于 Exp2 step6_visualize.py 改造)
# ============================================================================
# 输入:
#   /home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_{val,test,holdout}.csv
#   /home/tcat/diffcsp_exp4/code/step5/predictions_val.pt   (fig3 / fig5 用)
#   /home/tcat/diffcsp_exp4/data/data_inventory_v2.csv     (fig3 中心元素)
# 输出:
#   /home/tcat/diffcsp_exp4/code/step6/figures/fig1..fig5*.png
#
# 6 张图:
#   1   RMSD 分布(3-panel: val/test/holdout)
#   2   TypeAcc 分布(3-panel)
#   2b  TypeAcc by eval_cutoff Tier (boxplot, 3 splits)            ★ Exp4 新增
#   3   3D 结构对比(6 panel: 2 best / 2 mid / 2 worst, val only)
#   4   RMSD vs TypeAcc 散点(3 split 叠加)
#   5   TypeAcc by neighbor distance rank (val only, 20 bars)      ★ Exp4 新增
#
# 不变量(继承 Exp2 / Step4d):
#   L = 6.0
#   坐标系 frac ∈ [-0.5, 0.5],无 % 1.
#   匹配:最小镜像 + 匈牙利(与 step5_2_compute_metrics.py 一致)
#
# 用法:
#   /home/tcat/conda_envs/mlff/bin/python step6_visualize.py
#   /home/tcat/conda_envs/mlff/bin/python step6_visualize.py --only 1 2b
# ============================================================================

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.optimize import linear_sum_assignment
from scipy import stats

from pymatgen.core import Element
from pymatgen.vis.structure_vtk import EL_COLORS

warnings.filterwarnings("ignore")


# ─── Paths ───────────────────────────────────────────────────────────────────
EXP_ROOT     = "/home/tcat/diffcsp_exp4"
STEP5_DIR    = os.path.join(EXP_ROOT, "code", "step5")
STEP6_DIR    = os.path.join(EXP_ROOT, "code", "step6")
FIG_DIR      = os.path.join(STEP6_DIR, "figures")
DATA_DIR     = os.path.join(EXP_ROOT, "data")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATHS = {
    "val":     os.path.join(STEP5_DIR, "per_sample_metrics_val.csv"),
    "test":    os.path.join(STEP5_DIR, "per_sample_metrics_test.csv"),
    "holdout": os.path.join(STEP5_DIR, "per_sample_metrics_holdout.csv"),
}
PT_VAL = os.path.join(STEP5_DIR, "predictions_val.pt")
INVENTORY_CSV = os.path.join(DATA_DIR, "data_inventory_v2.csv")


# ─── Constants ───────────────────────────────────────────────────────────────
L = 6.0
RANDOM_RMSD_BASELINE = (L / 2) * (3 / 5) ** 0.5    # ≈ 2.324 Å
RANDOM_TACC_BASELINE = 1.0 / 88                     # ≈ 0.01136 (88-element regime)
EXP2_HOLDOUT_TYPEACC = 0.241                        # Exp2 Fe-only holdout baseline

SPLIT_COLOR  = {"val": "#1f77b4", "test": "#ff7f0e", "holdout": "#2ca02c"}
SPLIT_OFFSET = {"val": -0.25,     "test": 0.0,       "holdout":  0.25}

TIER_ORDER = ["A", "B", "C", "D"]
TIER_LABEL = {"A": "A: ≤3 Å",  "B": "B: 3–4 Å",
              "C": "C: 4–5 Å", "D": "D: >5 Å"}


def get_tier(eval_cutoff: float) -> str:
    if eval_cutoff <= 3.0: return "A"
    if eval_cutoff <= 4.0: return "B"
    if eval_cutoff <= 5.0: return "C"
    return "D"


# ─── Style ───────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize":  9,
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})


# ─── Element color (pymatgen Jmol palette) ───────────────────────────────────
JMOL = EL_COLORS["Jmol"]   # dict: symbol -> CommentedSeq([R, G, B]), 0-255

def z_to_symbol(z) -> str:
    try:
        return Element.from_Z(int(z)).symbol
    except Exception:
        return "?"

def element_color(z) -> tuple:
    """Z (int) -> (R, G, B) floats in [0, 1]. Falls back to gray if unknown."""
    sym = z_to_symbol(z)
    rgb = JMOL.get(sym, [128, 128, 128])
    return tuple(float(c) / 255.0 for c in rgb)


# ─── Tensor / array helpers ──────────────────────────────────────────────────
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ─── Hungarian min-image matching (mirror of step5_2_compute_metrics) ────────
def hungarian_match(pred_frac: np.ndarray, true_frac: np.ndarray, L: float = L):
    n = pred_frac.shape[0]
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)                # min-image
        cost[i] = np.linalg.norm(delta * L, axis=1)
    row, col = linear_sum_assignment(cost)
    return row, col


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, L=L):
    row, col = hungarian_match(pred_frac, true_frac, L)
    matched_sq = []
    for ri, ci in zip(row, col):
        d = pred_frac[ri] - true_frac[ci]
        d -= np.round(d)
        matched_sq.append(np.sum((d * L) ** 2))
    rmsd     = float(np.sqrt(np.mean(matched_sq)))
    type_acc = float((pred_types[row] == true_types[col]).mean())
    return rmsd, type_acc, row, col


# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_csvs() -> dict:
    dfs = {}
    for split, path in CSV_PATHS.items():
        df = pd.read_csv(path)
        df["tier"] = df["eval_cutoff"].apply(get_tier)
        dfs[split] = df
    return dfs

def load_inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV, usecols=["sample_name", "center_element"])

def load_predictions_val() -> dict:
    print(f"  loading {PT_VAL} ...")
    t0 = time.time()
    preds = torch.load(PT_VAL, map_location="cpu", weights_only=False)
    print(f"  loaded N={len(preds['mp_id'])} in {time.time()-t0:.1f}s")
    return preds


# ============================================================================
# Figure 1: RMSD distribution (3-panel)
# ============================================================================
def plot_fig1(dfs, save_path):
    print("\n[fig1] RMSD distribution (3-panel)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, split in zip(axes, ["val", "test", "holdout"]):
        df = dfs[split]
        rmsds = df["rmsd"].values
        m, s = float(rmsds.mean()), float(rmsds.std())
        ax.hist(rmsds, bins=40, range=(0, 4),
                color=SPLIT_COLOR[split], edgecolor="white", alpha=0.88)
        ax.axvline(m, color="#FF7F0E", linestyle="--", linewidth=2,
                   label=f"mean = {m:.4f} Å")
        ax.axvline(RANDOM_RMSD_BASELINE, color="#D62728", linestyle="--",
                   linewidth=2,
                   label=f"random = {RANDOM_RMSD_BASELINE:.2f} Å")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Number of samples")
        ax.set_xlim(0, 4)
        ax.set_title(f"{split} (N={len(rmsds)})\nmean = {m:.4f} ± {s:.4f}")
        ax.legend(loc="upper right")
        print(f"  {split:7s} N={len(rmsds):5d}  "
              f"mean={m:.4f}  std={s:.4f}  "
              f"min={rmsds.min():.4f}  max={rmsds.max():.4f}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 2: TypeAcc distribution (3-panel)
# ============================================================================
def plot_fig2(dfs, save_path):
    print("\n[fig2] TypeAcc distribution (3-panel)")
    edges = np.linspace(-0.5 / 20, 20.5 / 20, 22)   # 21 bins centered on k/20
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, split in zip(axes, ["val", "test", "holdout"]):
        df = dfs[split]
        ta = df["type_acc"].values
        m, s = float(ta.mean()), float(ta.std())
        ax.hist(ta, bins=edges, color=SPLIT_COLOR[split],
                edgecolor="white", alpha=0.88)
        ax.axvline(m, color="#FF7F0E", linestyle="--", linewidth=2,
                   label=f"mean = {m:.4f}")
        ax.axvline(RANDOM_TACC_BASELINE, color="#D62728", linestyle="--",
                   linewidth=2,
                   label=f"random (1/88) = {RANDOM_TACC_BASELINE:.4f}")
        ax.set_xlabel("Type Accuracy (correct / 20)")
        ax.set_ylabel("Number of samples")
        ax.set_xlim(0, 1)
        ax.set_title(f"{split} (N={len(ta)})\nmean = {m:.4f} ± {s:.4f}")
        ax.legend(loc="upper right")
        print(f"  {split:7s} N={len(ta):5d}  "
              f"mean={m:.4f}  std={s:.4f}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 2b: TypeAcc by eval_cutoff Tier (boxplot, 3 splits)
# ============================================================================
def plot_fig2b(dfs, save_path):
    print("\n[fig2b] TypeAcc by eval_cutoff Tier (3 splits, boxplot)")
    fig, ax = plt.subplots(figsize=(11, 6.5))

    legend_handles = []
    for split in ["val", "test", "holdout"]:
        df = dfs[split]
        for ti, tier in enumerate(TIER_ORDER):
            vals = df.loc[df["tier"] == tier, "type_acc"].values
            x_pos = ti + SPLIT_OFFSET[split]
            n = len(vals)
            if n == 0:
                ax.text(x_pos, 0.04, "N/A", ha="center", va="bottom",
                        fontsize=9, color=SPLIT_COLOR[split], fontweight="bold")
                print(f"  tier={tier} split={split:7s} N=    0  (empty box → N/A)")
                continue
            bp = ax.boxplot([vals], positions=[x_pos], widths=0.22,
                            patch_artist=True, showfliers=False,
                            manage_ticks=False)
            bp["boxes"][0].set_facecolor(SPLIT_COLOR[split])
            bp["boxes"][0].set_alpha(0.65)
            bp["boxes"][0].set_edgecolor("black")
            for elem in ["whiskers", "caps", "medians"]:
                for line in bp[elem]:
                    line.set_color("black")
            print(f"  tier={tier} split={split:7s} N={n:5d}  "
                  f"mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
                  f"q25={np.percentile(vals,25):.4f}  "
                  f"q75={np.percentile(vals,75):.4f}")
        legend_handles.append(Patch(facecolor=SPLIT_COLOR[split],
                                    alpha=0.65, edgecolor="black",
                                    label=split))

    # Exp2 Fe-only holdout reference (full-width horizontal)
    ax.axhline(EXP2_HOLDOUT_TYPEACC, color="#D62728", linestyle="--",
               linewidth=1.5, alpha=0.85)
    legend_handles.append(Line2D(
        [0], [0], color="#D62728", linestyle="--", linewidth=1.5,
        label=f"Exp2 Fe-only baseline = {EXP2_HOLDOUT_TYPEACC:.3f}"))

    # Random baseline (1/88)
    ax.axhline(RANDOM_TACC_BASELINE, color="#888888", linestyle=":",
               linewidth=1.0, alpha=0.7)
    legend_handles.append(Line2D(
        [0], [0], color="#888888", linestyle=":", linewidth=1.0,
        label=f"random (1/88) = {RANDOM_TACC_BASELINE:.4f}"))

    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([TIER_LABEL[t] for t in TIER_ORDER])
    ax.set_xlabel("eval_cutoff Tier")
    ax.set_ylabel("Type Accuracy")
    ax.set_xlim(-0.6, len(TIER_ORDER) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Type Accuracy by eval_cutoff Tier (val / test / holdout)\n"
        "Tier B (3–4 Å, 1st/2nd shell) ≈ Exp2 parity. "
        "Monotone decrease reflects XANES near-shell information limit.",
        fontsize=11)
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 3: 3D structure comparison (6 panel, val only)
# ============================================================================
def select_six_samples(df_val):
    """2 best / 2 mid (~1.485) / 2 worst (RMSD ≤ 3.5)."""
    by = df_val.sort_values("rmsd").reset_index(drop=True)
    best_2 = by.head(2)["sample_name"].tolist()

    near = df_val[(df_val["rmsd"] >= 1.40) & (df_val["rmsd"] <= 1.50)].copy()
    if len(near) >= 2:
        near["d"] = (near["rmsd"] - 1.485).abs()
        mid_2 = near.sort_values("d").head(2)["sample_name"].tolist()
    else:
        fb = df_val.copy()
        fb["d"] = (fb["rmsd"] - 1.485).abs()
        mid_2 = fb.sort_values("d").head(2)["sample_name"].tolist()

    pool = df_val[df_val["rmsd"] <= 3.5]
    worst_2 = pool.sort_values("rmsd", ascending=False)\
                  .head(2)["sample_name"].tolist()

    return best_2 + mid_2 + worst_2


def _draw_panel_3d(ax, rec, group_label):
    pf, pt = rec["pred_frac"], rec["pred_types"]
    tf, tt = rec["true_frac"], rec["true_types"]
    row, col = rec["row"], rec["col"]
    center_elem = rec["center_element"]

    # already in [-0.5, 0.5] but fold defensively
    pf_f = pf - np.round(pf)
    tf_f = tf - np.round(tf)
    pc = pf_f * L
    tc = tf_f * L

    # True atoms (large filled spheres, black edge)
    for i in range(tc.shape[0]):
        c = element_color(int(tt[i]))
        ax.scatter(tc[i, 0], tc[i, 1], tc[i, 2],
                   s=90, c=[c], edgecolors="black",
                   linewidths=0.8, depthshade=True)

    # Predicted atoms (small hollow circles, colored edge)
    for i in range(pc.shape[0]):
        c = element_color(int(pt[i]))
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2],
                   s=55, facecolors="none", edgecolors=[c],
                   linewidths=1.6, depthshade=True)

    # Matched pairs (dashed)
    for ri, ci in zip(row, col):
        ax.plot([pc[ri, 0], tc[ci, 0]],
                [pc[ri, 1], tc[ci, 1]],
                [pc[ri, 2], tc[ci, 2]],
                color="k", linestyle="--", linewidth=0.5, alpha=0.45)

    # Center atom (origin) — red star
    ax.scatter([0], [0], [0], marker="*", s=320, c="red",
               edgecolors="black", linewidths=1.0,
               depthshade=False, zorder=20)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel("x (Å)", fontsize=8)
    ax.set_ylabel("y (Å)", fontsize=8)
    ax.set_zlabel("z (Å)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"[{group_label}]  {center_elem} center\n"
        f"RMSD={rec['rmsd']:.3f} Å,  TypeAcc={rec['type_acc']:.3f}",
        fontsize=10,
    )


def plot_fig3(preds_val, df_val, df_inv, save_path):
    print("\n[fig3] 3D structure comparison (6 panel, val)")
    chosen_sn = select_six_samples(df_val)
    labels = ["Best #1", "Best #2", "Mid #1", "Mid #2", "Worst #1", "Worst #2"]

    sn_to_idx = {sn: i for i, sn in enumerate(preds_val["sample_name"])}
    inv_map = df_inv.set_index("sample_name")["center_element"].to_dict()

    records = []
    for sn in chosen_sn:
        if sn not in sn_to_idx:
            print(f"  WARN: sample_name {sn!r} not in predictions_val.pt — skipping")
            continue
        i = sn_to_idx[sn]
        pf = _to_np(preds_val["pred_frac_coords"][i])
        pt = _to_np(preds_val["pred_atom_types"][i])
        tf = _to_np(preds_val["true_frac_coords"][i])
        tt = _to_np(preds_val["true_atom_types"][i])
        rmsd, tacc, row, col = evaluate_sample(pf, pt, tf, tt, L=L)
        records.append({
            "sample_name":    sn,
            "center_element": inv_map.get(sn, "?"),
            "rmsd":           rmsd,
            "type_acc":       tacc,
            "pred_frac":      pf, "pred_types": pt,
            "true_frac":      tf, "true_types": tt,
            "row":            row, "col":       col,
        })

    fig = plt.figure(figsize=(17, 10))
    for k, (rec, lab) in enumerate(zip(records, labels), start=1):
        ax = fig.add_subplot(2, 3, k, projection="3d")
        _draw_panel_3d(ax, rec, lab)
        print(f"  {lab:10s} sn={rec['sample_name']:42s} "
              f"center={rec['center_element']:3s} "
              f"RMSD={rec['rmsd']:.3f}  TypeAcc={rec['type_acc']:.3f}")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=11, label="True atoms"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markeredgewidth=1.6, markersize=10,
               label="Predicted atoms"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=17,
               label="Center atom (origin)"),
        Line2D([0], [0], linestyle="--", color="k", alpha=0.5,
               label="Matched pair"),
    ]
    fig.legend(handles=legend_handles,
               loc="center right", bbox_to_anchor=(0.995, 0.5),
               frameon=True, fontsize=10)

    fig.suptitle(
        "3D Structure Comparison: True vs. Predicted (Val Set)\n"
        "Atom colors follow Jmol convention",
        fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 4: RMSD vs TypeAcc (3-split overlay)
# ============================================================================
def plot_fig4(dfs, save_path):
    print("\n[fig4] RMSD vs TypeAcc (3-split overlay)")
    fig, ax = plt.subplots(figsize=(9, 6))

    annot_lines = []
    for split in ["val", "test", "holdout"]:
        df = dfs[split]
        x = df["rmsd"].values
        y = df["type_acc"].values
        c = SPLIT_COLOR[split]
        ax.scatter(x, y, s=8, alpha=0.30, color=c, edgecolors="none",
                   label=f"{split} (N={len(x)})")
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.95)
        annot_lines.append(f"{split:8s}: r={r_val:+.3f}, p={p_val:.2e}")
        print(f"  {split:7s} r={r_val:+.4f}  p={p_val:.2e}  "
              f"slope={slope:+.4f}  intercept={intercept:+.4f}")

    ax.text(0.97, 0.97, "\n".join(annot_lines),
            transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Type Accuracy")
    ax.set_title("RMSD vs Type Accuracy (3-split overlay)")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 5: TypeAcc by neighbor distance rank (val only)
# ============================================================================
def plot_fig5(preds_val, save_path, df_val_for_check=None):
    print("\n[fig5] TypeAcc by neighbor distance rank (val)")
    N = len(preds_val["mp_id"])
    correct_at_rank = np.zeros(20, dtype=np.int64)
    total_at_rank   = np.zeros(20, dtype=np.int64)
    n_skipped = 0

    t0 = time.time()
    for i in range(N):
        pf = _to_np(preds_val["pred_frac_coords"][i])
        pt = _to_np(preds_val["pred_atom_types"][i])
        tf = _to_np(preds_val["true_frac_coords"][i])
        tt = _to_np(preds_val["true_atom_types"][i])
        if pf.shape[0] != 20 or tf.shape[0] != 20:
            n_skipped += 1
            continue

        # rank true neighbors by distance to origin (cartesian = frac * L)
        true_dists = np.linalg.norm(tf * L, axis=1)
        rank_order = np.argsort(true_dists)        # rank 0 = closest
        rank_of_true = np.empty(20, dtype=int)
        for r, ti in enumerate(rank_order):
            rank_of_true[ti] = r

        # Hungarian (same as metrics)
        row, col = hungarian_match(pf, tf, L)
        for ri, ci in zip(row, col):
            r = rank_of_true[ci]
            total_at_rank[r] += 1
            if pt[ri] == tt[ci]:
                correct_at_rank[r] += 1

    dt = time.time() - t0
    acc_at_rank = correct_at_rank / np.maximum(total_at_rank, 1)
    weighted_avg = correct_at_rank.sum() / max(total_at_rank.sum(), 1)

    print(f"  Hungarian over N={N} samples in {dt:.1f}s "
          f"(skipped={n_skipped})")
    print(f"  weighted-avg TypeAcc = {weighted_avg:.4f}")

    if df_val_for_check is not None:
        ref = float(df_val_for_check["type_acc"].mean())
        drift = abs(weighted_avg - ref)
        print(f"  vs val_csv mean = {ref:.4f}  →  |Δ| = {drift:.4f}  "
              f"({'OK' if drift < 0.001 else 'WARNING — drift > 0.001'})")
        if drift >= 0.001:
            print("  RED LINE: fig5 Hungarian disagrees with Step5 metrics — "
                  "investigate before trusting fig5")

    print("  per-rank TypeAcc:")
    for r in range(20):
        print(f"    rank {r+1:2d}: {acc_at_rank[r]:.4f}  (N={total_at_rank[r]})")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ranks = np.arange(1, 21)
    ax.bar(ranks, acc_at_rank, color="#1f77b4",
           edgecolor="white", alpha=0.88)
    ax.axhline(RANDOM_TACC_BASELINE, color="#D62728", linestyle="--",
               linewidth=1.5,
               label=f"random (1/88) = {RANDOM_TACC_BASELINE:.4f}")
    ax.axhline(weighted_avg, color="#FF7F0E", linestyle="--", linewidth=1.5,
               label=f"overall mean = {weighted_avg:.4f}")
    ax.set_xticks(ranks)
    ax.set_xlabel("Neighbor rank (1 = closest to center)")
    ax.set_ylabel("Type Accuracy")
    ax.set_ylim(0, max(0.7, acc_at_rank.max() * 1.2))
    ax.set_title(
        f"Type Accuracy by Neighbor Distance Rank "
        f"(Val, N={N - n_skipped})\n"
        f"XANES near-shell sensitivity: rank-1 ≫ random; "
        f"far-rank approaches information floor.",
        fontsize=11)

    # value labels at rank 1, 5, 10, 20
    for r in [1, 5, 10, 20]:
        v = acc_at_rank[r - 1]
        ax.text(r, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    default=["1", "2", "2b", "3", "4", "5"],
                    choices=["1", "2", "2b", "3", "4", "5"],
                    help="which figure(s) to render (default = all 6)")
    args = ap.parse_args()
    sel = set(args.only)

    print("=" * 78)
    print(f"DiffCSP-Exp4 Step 6 Visualization")
    print(f"  figures requested : {sorted(sel)}")
    print(f"  output dir        : {FIG_DIR}")
    print("=" * 78)

    print("\n── loading inputs ──")
    dfs = load_csvs()
    print(f"  CSVs loaded: val={len(dfs['val'])}  "
          f"test={len(dfs['test'])}  holdout={len(dfs['holdout'])}")

    df_inv    = load_inventory()    if ("3" in sel) else None
    preds_val = load_predictions_val() if ("3" in sel or "5" in sel) else None
    if df_inv is not None:
        print(f"  inventory: {len(df_inv)} rows")

    f1  = os.path.join(FIG_DIR, "fig1_rmsd_distribution.png")
    f2  = os.path.join(FIG_DIR, "fig2_typeacc_distribution.png")
    f2b = os.path.join(FIG_DIR, "fig2b_typeacc_by_tier.png")
    f3  = os.path.join(FIG_DIR, "fig3_structure_comparison.png")
    f4  = os.path.join(FIG_DIR, "fig4_rmsd_vs_typeacc.png")
    f5  = os.path.join(FIG_DIR, "fig5_typeacc_by_rank.png")

    t_global = time.time()
    if "1"  in sel: plot_fig1 (dfs, f1)
    if "2"  in sel: plot_fig2 (dfs, f2)
    if "2b" in sel: plot_fig2b(dfs, f2b)
    if "3"  in sel: plot_fig3 (preds_val, dfs["val"], df_inv, f3)
    if "4"  in sel: plot_fig4 (dfs, f4)
    if "5"  in sel: plot_fig5 (preds_val, f5, df_val_for_check=dfs["val"])

    print("\n" + "=" * 78)
    print(f"Done in {time.time()-t_global:.1f}s. Figures in: {FIG_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
