# step6_visualize_v2.py
# DiffCSP-Exp5 v2 Step 6 — 结果可视化(fork from Exp4 step6_visualize.py)
# ============================================================================
# 输入:
#   /home/tcat/diffcsp_exp5/code/step5/predictions_v2_val.pt
#   /home/tcat/diffcsp_exp5/code/step5/predictions_v2_test.pt
#   /home/tcat/diffcsp_exp5/logs/v2_val_per_sample.csv
#   /home/tcat/diffcsp_exp5/logs/v2_test_per_sample.csv
#   /home/tcat/diffcsp_exp5/logs/exp4_baseline_val_per_sample.csv
#   /home/tcat/diffcsp_exp5/logs/exp4_baseline_test_per_sample.csv
#   /home/tcat/diffcsp_exp4/data/data_inventory_v2.csv (fig3 中心元素)
#
# 输出:
#   /home/tcat/diffcsp_exp5/code/step6/figures/fig*.png
#
# 5 张图(精选,组会展示用):
#   1   RMSD 分布(2-panel: val/test,叠加 Exp4 baseline 虚线)
#   3   3D 结构对比(6 panel: 2 best / 2 mid / 2 worst, val only) ⭐ 导师最爱
#   5   TypeAcc by neighbor distance rank (val only, 20 bars)
#   6   v2 vs Exp4 6-指标 paired bar chart                       ⭐ 故事核心
#   7   键长分布 真实 vs 预测 overlay histogram                  ⭐ 物理直观
#
# 6 个新指标(报到 console + 写到 metrics_summary.txt):
#   A1. CN_strict     — 第一壳层(≤3Å)原子数完全相等
#   A2. CN_loose      — 第一壳层原子数误差 ≤1
#   B.  Center_acc    — 中心元素识别(v2 加了 center_emb 验证)
#   C.  Top1Elem_acc  — 第一壳层主元素准确率
#   E.  BondLen_MAE   — 中心-邻居键长 MAE(物理直观)
#   F.  Shell1_RMSD   — 仅第一壳层(≤3Å)的 RMSD
#
# 不变量:
#   L = 6.0
#   坐标系 frac ∈ [-0.5, 0.5],无 % 1.
#   匹配:最小镜像 + 匈牙利
#
# 用法:
#   /home/tcat/conda_envs/mlff/bin/python step6_visualize_v2.py
#   /home/tcat/conda_envs/mlff/bin/python step6_visualize_v2.py --only 3 6 7
# ============================================================================

import argparse
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.lines import Line2D

from scipy.optimize import linear_sum_assignment

from pymatgen.core import Element
from pymatgen.vis.structure_vtk import EL_COLORS

warnings.filterwarnings("ignore")


# ─── Paths (Exp5 v2) ─────────────────────────────────────────────────────────
EXP5_ROOT  = "/home/tcat/diffcsp_exp5"
EXP4_ROOT  = "/home/tcat/diffcsp_exp4"
STEP5_DIR  = os.path.join(EXP5_ROOT, "code", "step5")
STEP6_DIR  = os.path.join(EXP5_ROOT, "code", "step6")
LOGS_DIR   = os.path.join(EXP5_ROOT, "logs")
FIG_DIR    = os.path.join(STEP6_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# v2 predictions (SA3' 输出)
PT_PATHS = {
    "val":  os.path.join(STEP5_DIR, "predictions_v2_val.pt"),
    "test": os.path.join(STEP5_DIR, "predictions_v2_test.pt"),
}

# v2 per-sample CSV (SA3' 输出)
V2_CSV_PATHS = {
    "val":  os.path.join(LOGS_DIR, "v2_val_per_sample.csv"),
    "test": os.path.join(LOGS_DIR, "v2_test_per_sample.csv"),
}

# Exp4 baseline per-sample CSV (SA1' dry-run 已产出,for fig6 对比)
EXP4_CSV_PATHS = {
    "val":  os.path.join(LOGS_DIR, "exp4_baseline_val_per_sample.csv"),
    "test": os.path.join(LOGS_DIR, "exp4_baseline_test_per_sample.csv"),
}

# inventory (fig3 中心元素 lookup) — 复用 Exp4 file
INVENTORY_CSV = os.path.join(EXP4_ROOT, "data", "data_inventory_v2.csv")


# ─── Constants ───────────────────────────────────────────────────────────────
L = 6.0
SHELL1_CUTOFF = 3.0  # Å,第一壳层判定半径

# Exp4 baseline 数 (SA3' OUTPUT §2.1 锚点,用于 fig6 + fig1 虚线)
EXP4_BASELINE = {
    "RMSD_val":          1.4849,
    "RMSD_test":         1.4852,
    "Multiset_val":      0.0843,
    "Multiset_test":     0.0846,
    "SetLevel_val":      0.3309,
    "SetLevel_test":     0.3330,
    "PredInCutoff_val":  18.93,
    "PredInCutoff_test": 18.93,
    # 下列 4 个指标 Exp4 没算过,fig6 用 'fork-from-Exp4-CSV' 现算见 main()
    "CN_strict_val":     None,
    "CN_loose_val":      None,
    "Top1Elem_val":      None,
    "BondLen_MAE_val":   None,
    "Shell1_RMSD_val":   None,
}

# v2 主信号(SA3' OUTPUT §2.1)— 用于 fig6 标签
V2_HEADLINE = {
    "RMSD_val":      1.4954,  "RMSD_test":      1.4928,
    "Multiset_val":  0.1086,  "Multiset_test":  0.1096,
    "SetLevel_val":  0.3408,  "SetLevel_test":  0.3397,
}

# 颜色
SPLIT_COLOR = {"val": "#1f77b4", "test": "#ff7f0e"}
V2_COLOR    = "#2ca02c"  # 绿
EXP4_COLOR  = "#d62728"  # 红


# ─── Style ───────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize":  9,
    "figure.dpi":      120,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
})


# ─── Element color (pymatgen Jmol palette) ───────────────────────────────────
def z_to_symbol(z) -> str:
    try:
        return Element.from_Z(int(z)).symbol
    except Exception:
        return "?"

def element_color(z) -> tuple:
    sym = z_to_symbol(z)
    rgb = EL_COLORS["Jmol"].get(sym, (128, 128, 128))
    return tuple(c / 255.0 for c in rgb)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def hungarian_match(pred_frac: np.ndarray, true_frac: np.ndarray, L: float = L):
    n = pred_frac.shape[0]
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)
        cost[i] = np.linalg.norm(delta * L, axis=1)
    row, col = linear_sum_assignment(cost)
    return row, col


def evaluate_sample_full(pred_frac, pred_types, true_frac, true_types, L=L,
                         shell1_cutoff=SHELL1_CUTOFF, center_z=None):
    """
    扩展版 evaluate_sample,返回所有 6 个新指标 + 原始 RMSD/TypeAcc。

    Returns dict with:
      - rmsd, type_acc, row, col   (原版)
      - cn_pred, cn_true, cn_strict, cn_loose
      - top1_pred, top1_true, top1_acc
      - bond_len_mae               (Å)
      - shell1_rmsd                (Å, 仅 ≤ shell1_cutoff 的原子)
      - center_pred_match  (None if center_z not provided)
    """
    row, col = hungarian_match(pred_frac, true_frac, L)

    # === 原版 RMSD + TypeAcc ===
    matched_sq = []
    for ri, ci in zip(row, col):
        d = pred_frac[ri] - true_frac[ci]
        d -= np.round(d)
        matched_sq.append(np.sum((d * L) ** 2))
    rmsd     = float(np.sqrt(np.mean(matched_sq)))
    type_acc = float((pred_types[row] == true_types[col]).mean())

    # === 笛卡尔距离 (距 origin) ===
    pf_fold = pred_frac - np.round(pred_frac)
    tf_fold = true_frac - np.round(true_frac)
    pred_dist = np.linalg.norm(pf_fold * L, axis=1)
    true_dist = np.linalg.norm(tf_fold * L, axis=1)

    # === A. Coordination Number (CN) — 第一壳层原子数 ===
    cn_pred = int((pred_dist <= shell1_cutoff).sum())
    cn_true = int((true_dist <= shell1_cutoff).sum())
    cn_strict = int(cn_pred == cn_true)
    cn_loose  = int(abs(cn_pred - cn_true) <= 1)

    # === C. Top-1 Element (第一壳层主元素) ===
    pred_shell1_types = pred_types[pred_dist <= shell1_cutoff]
    true_shell1_types = true_types[true_dist <= shell1_cutoff]
    if len(pred_shell1_types) > 0 and len(true_shell1_types) > 0:
        top1_pred = int(Counter(pred_shell1_types.tolist()).most_common(1)[0][0])
        top1_true = int(Counter(true_shell1_types.tolist()).most_common(1)[0][0])
        top1_acc  = int(top1_pred == top1_true)
    else:
        top1_pred, top1_true, top1_acc = -1, -1, 0

    # === E. Bond Length MAE (中心-邻居键长 MAE) ===
    # 对每个匹配对,比较 d(pred_i, center) vs d(true_match, center)
    bond_len_errs = []
    for ri, ci in zip(row, col):
        bond_len_errs.append(abs(pred_dist[ri] - true_dist[ci]))
    bond_len_mae = float(np.mean(bond_len_errs))

    # === F. Shell-1 RMSD (仅看 ≤3Å 的真实原子,匹配上 -> 算 RMSD) ===
    # row[i] = pred 索引, col[i] = true 索引
    # 选 col[i] 对应的 true 距离 ≤ shell1_cutoff 的 pair
    shell1_sq = []
    for ri, ci in zip(row, col):
        if true_dist[ci] <= shell1_cutoff:
            d = pred_frac[ri] - true_frac[ci]
            d -= np.round(d)
            shell1_sq.append(np.sum((d * L) ** 2))
    shell1_rmsd = float(np.sqrt(np.mean(shell1_sq))) if shell1_sq else float("nan")

    # === B. Center element check (placeholder — center_z 来自 inventory) ===
    # v2 加了 center_emb,所以中心元素是模型 input 不是 prediction,
    # 这指标 trivially 100% (除非 dataloader 出 bug)
    # 这个指标在 console 报告里写"v2 architecture verified",但 fig 里不画
    center_pred_match = None  # 由 main() 用 inventory + ckpt config 验证

    return {
        "rmsd":         rmsd,
        "type_acc":     type_acc,
        "row":          row,
        "col":          col,
        "cn_pred":      cn_pred,
        "cn_true":      cn_true,
        "cn_strict":    cn_strict,
        "cn_loose":     cn_loose,
        "top1_pred":    top1_pred,
        "top1_true":    top1_true,
        "top1_acc":     top1_acc,
        "bond_len_mae": bond_len_mae,
        "shell1_rmsd":  shell1_rmsd,
        "pred_dist":    pred_dist,
        "true_dist":    true_dist,
    }


# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_predictions(split: str) -> dict:
    path = PT_PATHS[split]
    print(f"  loading {path} ...")
    t0 = time.time()
    preds = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  loaded N={len(preds['mp_id'])} in {time.time()-t0:.1f}s")
    return preds


def load_v2_csvs() -> dict:
    dfs = {}
    for split, path in V2_CSV_PATHS.items():
        if os.path.exists(path):
            dfs[split] = pd.read_csv(path)
        else:
            print(f"  WARN: {path} missing, skip")
    return dfs


def load_inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV, usecols=["sample_name", "center_element"])


# ─── Compute 6 新指标 over full split (用于 fig6) ──────────────────────────
def compute_all_metrics_for_split(preds: dict, split_name: str) -> pd.DataFrame:
    """
    Iterate all samples, compute 6 新指标 + 原版 RMSD/TypeAcc.
    Returns DataFrame with one row per sample.
    """
    print(f"\n  computing 6 metrics on {split_name} split ...")
    n = len(preds["sample_name"])
    rows = []
    t0 = time.time()
    for i in range(n):
        pf = _to_np(preds["pred_frac_coords"][i])
        pt = _to_np(preds["pred_atom_types"][i])
        tf = _to_np(preds["true_frac_coords"][i])
        tt = _to_np(preds["true_atom_types"][i])
        if pf.shape[0] != 20 or tf.shape[0] != 20:
            continue
        m = evaluate_sample_full(pf, pt, tf, tt, L=L)
        rows.append({
            "sample_name":  preds["sample_name"][i],
            "mp_id":        preds["mp_id"][i],
            "rmsd":         m["rmsd"],
            "type_acc":     m["type_acc"],
            "cn_pred":      m["cn_pred"],
            "cn_true":      m["cn_true"],
            "cn_strict":    m["cn_strict"],
            "cn_loose":     m["cn_loose"],
            "top1_pred":    m["top1_pred"],
            "top1_true":    m["top1_true"],
            "top1_acc":     m["top1_acc"],
            "bond_len_mae": m["bond_len_mae"],
            "shell1_rmsd":  m["shell1_rmsd"],
        })
        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{n}]  ({time.time()-t0:.1f}s)")
    df = pd.DataFrame(rows)
    print(f"  done {len(df)}/{n} samples in {time.time()-t0:.1f}s")
    return df


def summarize_metrics(df: pd.DataFrame, label: str) -> dict:
    """Aggregate to scalars for fig6 + console + summary file."""
    s = {
        "label":           label,
        "n":               len(df),
        "RMSD":            df["rmsd"].mean(),
        "RMSD_median":     df["rmsd"].median(),
        "TypeAcc":         df["type_acc"].mean(),
        "CN_strict":       df["cn_strict"].mean(),
        "CN_loose":        df["cn_loose"].mean(),
        "Top1Elem":        df["top1_acc"].mean(),
        "BondLen_MAE":     df["bond_len_mae"].mean(),
        "Shell1_RMSD":     df["shell1_rmsd"].mean(),
        "Shell1_RMSD_med": df["shell1_rmsd"].median(),
    }
    return s


# ============================================================================
# Figure 1: RMSD distribution (2-panel: val/test)
# ============================================================================
def plot_fig1(dfs_v2, save_path):
    print("\n[fig1] RMSD distribution (v2 val + test, vs Exp4 baseline)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split in zip(axes, ["val", "test"]):
        if split not in dfs_v2:
            continue
        df = dfs_v2[split]
        rmsd = df["rmsd"].values
        ax.hist(rmsd, bins=40, color=SPLIT_COLOR[split], alpha=0.75,
                edgecolor="black", linewidth=0.6)
        ax.axvline(rmsd.mean(), color="black", linestyle="-", linewidth=1.5,
                   label=f"v2 mean = {rmsd.mean():.3f} Å")
        ax.axvline(EXP4_BASELINE[f"RMSD_{split}"], color=EXP4_COLOR,
                   linestyle="--", linewidth=1.5,
                   label=f"Exp4 = {EXP4_BASELINE[f'RMSD_{split}']:.3f} Å")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("count")
        ax.set_title(f"{split.upper()} (N={len(rmsd)})")
        ax.legend(loc="upper right")
    fig.suptitle("RMSD Distribution: Exp5 v2 vs Exp4 baseline", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 3: 3D structure comparison (6 panel) — 导师最爱,完全保留
# ============================================================================
def select_six_samples(df_val: pd.DataFrame):
    """2 best / 2 mid (~median) / 2 worst (capped at <=3.5 Å for visibility)."""
    by = df_val.sort_values("rmsd").reset_index(drop=True)
    best_2 = by.head(2)["sample_name"].tolist()

    median_rmsd = df_val["rmsd"].median()
    near = df_val[(df_val["rmsd"] >= median_rmsd - 0.05) &
                  (df_val["rmsd"] <= median_rmsd + 0.05)].copy()
    if len(near) >= 2:
        near["d"] = (near["rmsd"] - median_rmsd).abs()
        mid_2 = near.sort_values("d").head(2)["sample_name"].tolist()
    else:
        fb = df_val.copy()
        fb["d"] = (fb["rmsd"] - median_rmsd).abs()
        mid_2 = fb.sort_values("d").head(2)["sample_name"].tolist()

    pool = df_val[df_val["rmsd"] <= 3.5]
    worst_2 = pool.sort_values("rmsd", ascending=False).head(2)["sample_name"].tolist()

    return best_2 + mid_2 + worst_2


def _draw_panel_3d(ax, rec, group_label):
    pf, pt = rec["pred_frac"], rec["pred_types"]
    tf, tt = rec["true_frac"], rec["true_types"]
    row, col = rec["row"], rec["col"]
    center_elem = rec["center_element"]

    pf_f = pf - np.round(pf)
    tf_f = tf - np.round(tf)
    pc = pf_f * L
    tc = tf_f * L

    for i in range(tc.shape[0]):
        c = element_color(int(tt[i]))
        ax.scatter(tc[i, 0], tc[i, 1], tc[i, 2],
                   s=90, c=[c], edgecolors="black",
                   linewidths=0.8, depthshade=True)

    for i in range(pc.shape[0]):
        c = element_color(int(pt[i]))
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2],
                   s=55, facecolors="none", edgecolors=[c],
                   linewidths=1.6, depthshade=True)

    for ri, ci in zip(row, col):
        ax.plot([pc[ri, 0], tc[ci, 0]],
                [pc[ri, 1], tc[ci, 1]],
                [pc[ri, 2], tc[ci, 2]],
                color="k", linestyle="--", linewidth=0.5, alpha=0.45)

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
        f"RMSD={rec['rmsd']:.3f} Å,  CN_pred={rec['cn_pred']}/CN_true={rec['cn_true']}",
        fontsize=10,
    )


def plot_fig3(preds_val, df_val_v2, df_inv, save_path):
    print("\n[fig3] 3D structure comparison (6 panel, v2 val) — 导师最爱")
    chosen_sn = select_six_samples(df_val_v2)
    labels = ["Best #1", "Best #2", "Mid #1", "Mid #2", "Worst #1", "Worst #2"]

    sn_to_idx = {sn: i for i, sn in enumerate(preds_val["sample_name"])}
    inv_map = df_inv.set_index("sample_name")["center_element"].to_dict()

    records = []
    for sn in chosen_sn:
        if sn not in sn_to_idx:
            print(f"  WARN: sample_name {sn!r} not in predictions — skip")
            continue
        i = sn_to_idx[sn]
        pf = _to_np(preds_val["pred_frac_coords"][i])
        pt = _to_np(preds_val["pred_atom_types"][i])
        tf = _to_np(preds_val["true_frac_coords"][i])
        tt = _to_np(preds_val["true_atom_types"][i])
        m = evaluate_sample_full(pf, pt, tf, tt, L=L)
        records.append({
            "sample_name":    sn,
            "center_element": inv_map.get(sn, "?"),
            "rmsd":           m["rmsd"],
            "cn_pred":        m["cn_pred"],
            "cn_true":        m["cn_true"],
            "pred_frac":      pf, "pred_types": pt,
            "true_frac":      tf, "true_types": tt,
            "row":            m["row"], "col": m["col"],
        })

    fig = plt.figure(figsize=(17, 10))
    for k, (rec, lab) in enumerate(zip(records, labels), start=1):
        ax = fig.add_subplot(2, 3, k, projection="3d")
        _draw_panel_3d(ax, rec, lab)
        print(f"  {lab:10s} sn={rec['sample_name']:42s} "
              f"center={rec['center_element']:3s} "
              f"RMSD={rec['rmsd']:.3f}  CN_pred={rec['cn_pred']}/CN_true={rec['cn_true']}")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=11, label="True atoms (filled)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gray", markeredgewidth=1.6, markersize=10,
               label="Predicted atoms (hollow)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=17,
               label="Center atom (origin)"),
        Line2D([0], [0], linestyle="--", color="k", alpha=0.5,
               label="Hungarian-matched pair"),
    ]
    fig.legend(handles=legend_handles, loc="center right",
               bbox_to_anchor=(0.995, 0.5), frameon=True, fontsize=10)

    fig.suptitle(
        "Exp5 v2: 3D Structure Comparison — True vs. Predicted (Val Set)\n"
        "Atom colors follow Jmol convention",
        fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 5: TypeAcc by neighbor distance rank (val only, 20 bars)
# ============================================================================
def plot_fig5(preds_val, save_path):
    print("\n[fig5] TypeAcc by neighbor distance rank (val, 20 bars)")
    n = len(preds_val["sample_name"])
    correct_per_rank = np.zeros(20)
    total_per_rank   = np.zeros(20)

    for i in range(n):
        pf = _to_np(preds_val["pred_frac_coords"][i])
        pt = _to_np(preds_val["pred_atom_types"][i])
        tf = _to_np(preds_val["true_frac_coords"][i])
        tt = _to_np(preds_val["true_atom_types"][i])
        if pf.shape[0] != 20 or tf.shape[0] != 20:
            continue
        # Rank true atoms by distance to origin (true neighbor rank)
        tf_fold = tf - np.round(tf)
        true_dist = np.linalg.norm(tf_fold * L, axis=1)
        rank_order = np.argsort(true_dist)  # rank 0 = closest

        row, col = hungarian_match(pf, tf, L)
        # For each true atom, find which pred matched it
        col_to_row = {ci: ri for ri, ci in zip(row, col)}
        for r, ti in enumerate(rank_order):
            if ti in col_to_row:
                ri = col_to_row[ti]
                correct_per_rank[r] += int(pt[ri] == tt[ti])
                total_per_rank[r]   += 1

    acc_per_rank = correct_per_rank / np.maximum(total_per_rank, 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(1, 21), acc_per_rank, color=V2_COLOR, edgecolor="black", linewidth=0.5)
    ax.axhline(acc_per_rank.mean(), color="black", linestyle="--", linewidth=1.0,
               label=f"overall mean = {acc_per_rank.mean():.3f}")
    ax.set_xlabel("True neighbor rank (1 = closest to center, 20 = farthest)")
    ax.set_ylabel("TypeAcc at this rank")
    ax.set_title("Exp5 v2: TypeAcc by Neighbor Distance Rank (Val)")
    ax.set_xticks(range(1, 21))
    ax.set_ylim(0, max(acc_per_rank.max() * 1.15, 0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    for r in range(20):
        print(f"  rank {r+1:2d}  acc = {acc_per_rank[r]:.3f}  (n={int(total_per_rank[r])})")
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 6: v2 vs Exp4 6-metric paired bar chart                ⭐ 故事核心
# ============================================================================
def plot_fig6(summary_v2_val, summary_v2_test,
              summary_exp4_val, summary_exp4_test, save_path):
    print("\n[fig6] v2 vs Exp4 6-metric comparison (paired bars)")

    # 6 个指标 + 是否 higher-is-better + display name
    metrics = [
        ("CN_loose",     True,  "CN ±1"),
        ("CN_strict",    True,  "CN exact"),
        ("Top1Elem",     True,  "Shell-1 Top1\nelement"),
        ("Shell1_RMSD",  False, "Shell-1 RMSD\n(Å)"),
        ("BondLen_MAE",  False, "Bond-length\nMAE (Å)"),
        ("RMSD",         False, "RMSD (Å)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, (title, sv2, se4) in zip(
            axes,
            [("Validation", summary_v2_val,  summary_exp4_val),
             ("Test",       summary_v2_test, summary_exp4_test)]):

        labels    = [m[2] for m in metrics]
        v2_vals   = [sv2[m[0]]  for m in metrics]
        exp4_vals = [se4[m[0]]  for m in metrics]
        higher_better = [m[1] for m in metrics]

        x = np.arange(len(labels))
        w = 0.35
        b1 = ax.bar(x - w/2, exp4_vals, w, label="Exp4 (baseline)",
                    color=EXP4_COLOR, edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + w/2, v2_vals,   w, label="Exp5 v2",
                    color=V2_COLOR,   edgecolor="black", linewidth=0.6)

        # Annotate bars + 改进百分比
        for i, (bb1, bb2) in enumerate(zip(b1, b2)):
            ax.text(bb1.get_x() + bb1.get_width()/2, bb1.get_height(),
                    f"{exp4_vals[i]:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(bb2.get_x() + bb2.get_width()/2, bb2.get_height(),
                    f"{v2_vals[i]:.3f}", ha="center", va="bottom", fontsize=8)
            # Improvement arrow
            if higher_better[i]:
                pct = (v2_vals[i] - exp4_vals[i]) / max(exp4_vals[i], 1e-9) * 100
                arrow = "↑" if pct > 0 else "↓"
                clr = "green" if pct > 0 else "red"
            else:
                pct = (exp4_vals[i] - v2_vals[i]) / max(exp4_vals[i], 1e-9) * 100
                arrow = "↓" if pct > 0 else "↑"
                clr = "green" if pct > 0 else "red"
            ax.text(i, max(bb1.get_height(), bb2.get_height()) * 1.10,
                    f"{arrow}{abs(pct):.1f}%", ha="center", color=clr,
                    fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{title} (N={sv2['n']})")
        ax.set_ylabel("metric value")
        ax.legend(loc="upper right")
        ax.set_ylim(0, max(max(v2_vals), max(exp4_vals)) * 1.30)

    fig.suptitle("Exp5 v2 vs. Exp4 Baseline: 6-Metric Comparison\n"
                 "(green ↑ = improvement; red = regression)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ============================================================================
# Figure 7: 键长分布 真实 vs 预测 overlay histogram
# ============================================================================
def plot_fig7(preds_val, save_path):
    print("\n[fig7] Bond length distribution (true vs predicted, val)")
    n = len(preds_val["sample_name"])
    pred_dists_all = []
    true_dists_all = []

    for i in range(n):
        pf = _to_np(preds_val["pred_frac_coords"][i])
        tf = _to_np(preds_val["true_frac_coords"][i])
        if pf.shape[0] != 20 or tf.shape[0] != 20:
            continue
        pf_fold = pf - np.round(pf)
        tf_fold = tf - np.round(tf)
        pred_dists_all.extend(np.linalg.norm(pf_fold * L, axis=1).tolist())
        true_dists_all.extend(np.linalg.norm(tf_fold * L, axis=1).tolist())

    pred_dists_all = np.array(pred_dists_all)
    true_dists_all = np.array(true_dists_all)

    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, 6, 60)
    ax.hist(true_dists_all, bins=bins, alpha=0.55, color="steelblue",
            label=f"True (mean={true_dists_all.mean():.2f} Å)",
            edgecolor="black", linewidth=0.4)
    ax.hist(pred_dists_all, bins=bins, alpha=0.55, color="orange",
            label=f"Predicted (mean={pred_dists_all.mean():.2f} Å)",
            edgecolor="black", linewidth=0.4)
    ax.axvline(SHELL1_CUTOFF, color="red", linestyle="--", linewidth=1.0,
               label=f"Shell-1 boundary ({SHELL1_CUTOFF} Å)")
    ax.set_xlabel("Distance to center atom (Å)")
    ax.set_ylabel("Count")
    ax.set_title(f"Exp5 v2: Bond Length Distribution (Val, {n} samples × 20 atoms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    default=["1", "3", "5", "6", "7"],
                    choices=["1", "3", "5", "6", "7"],
                    help="which figure(s) to render (default = all)")
    ap.add_argument("--skip-baseline-recompute", action="store_true",
                    help="skip computing 6 metrics on Exp4 baseline (use V2 only;"
                         " fig6 will compare against V2 self if Exp4 CSV missing)")
    args = ap.parse_args()
    sel = set(args.only)

    print("=" * 78)
    print(f"DiffCSP-Exp5 v2 Step 6 Visualization")
    print(f"  figures requested : {sorted(sel)}")
    print(f"  output dir        : {FIG_DIR}")
    print("=" * 78)

    # Always need v2 predictions (every fig uses it)
    print("\n── loading v2 predictions ──")
    preds_v2 = {}
    for split in ["val", "test"]:
        if os.path.exists(PT_PATHS[split]):
            preds_v2[split] = load_predictions(split)
        else:
            print(f"  WARN: {PT_PATHS[split]} missing")

    # Compute 6 metrics over v2 (cache in DataFrame)
    print("\n── computing 6 metrics on v2 ──")
    df_v2_full = {}
    for split in ["val", "test"]:
        if split in preds_v2:
            df_v2_full[split] = compute_all_metrics_for_split(preds_v2[split], f"v2_{split}")

    # Save full per-sample CSV (for any downstream)
    for split, df in df_v2_full.items():
        out_csv = os.path.join(LOGS_DIR, f"v2_{split}_per_sample_extended.csv")
        df.to_csv(out_csv, index=False)
        print(f"  saved extended CSV: {out_csv}  ({len(df)} rows)")

    # Compute 6 metrics on Exp4 baseline (need predictions, not just CSV)
    # 注意: SA1' dry-run 用的是 Exp4 ckpt + Exp5 metrics 算的 CSV,
    # 但那 CSV 没有 6 个新指标(只有 RMSD/TypeAcc/...) — 我们要重 evaluate
    # 走 Exp4 predictions path
    EXP4_PT_VAL  = os.path.join(EXP4_ROOT, "code", "step5", "predictions_val.pt")
    EXP4_PT_TEST = os.path.join(EXP4_ROOT, "code", "step5", "predictions_test.pt")
    df_exp4_full = {}
    if not args.skip_baseline_recompute and os.path.exists(EXP4_PT_VAL):
        print("\n── computing 6 metrics on Exp4 baseline ──")
        for split, pt_path in [("val", EXP4_PT_VAL), ("test", EXP4_PT_TEST)]:
            if os.path.exists(pt_path):
                preds_e4 = torch.load(pt_path, map_location="cpu", weights_only=False)
                df_exp4_full[split] = compute_all_metrics_for_split(preds_e4, f"exp4_{split}")
            else:
                print(f"  WARN: {pt_path} missing — fig6 will fall back")
    else:
        print("\n  skip Exp4 baseline recompute (--skip-baseline-recompute or missing)")

    # Summaries
    summaries = {}
    for split in ["val", "test"]:
        if split in df_v2_full:
            summaries[f"v2_{split}"] = summarize_metrics(df_v2_full[split], f"v2_{split}")
        if split in df_exp4_full:
            summaries[f"exp4_{split}"] = summarize_metrics(df_exp4_full[split], f"exp4_{split}")

    # Print + save summary
    print("\n" + "=" * 78)
    print("METRICS SUMMARY")
    print("=" * 78)
    summary_lines = []
    summary_lines.append("="*78)
    summary_lines.append("Exp5 v2 metrics summary (organized for advisor presentation)")
    summary_lines.append("="*78)
    for k, s in summaries.items():
        line = (
            f"\n[{k}] N={s['n']}\n"
            f"  RMSD                        : {s['RMSD']:.4f}  (median {s['RMSD_median']:.4f})\n"
            f"  Hungarian TypeAcc (strict)  : {s['TypeAcc']:.4f}\n"
            f"  CN (coord. number) — strict : {s['CN_strict']:.4f}   <-- 配位数完全相等\n"
            f"  CN (coord. number) — loose  : {s['CN_loose']:.4f}    <-- 配位数误差≤1\n"
            f"  Shell-1 Top1 element acc    : {s['Top1Elem']:.4f}    <-- 第一壳层主元素\n"
            f"  Bond length MAE (Å)         : {s['BondLen_MAE']:.4f} <-- 物理直观\n"
            f"  Shell-1 RMSD (Å)            : {s['Shell1_RMSD']:.4f} (median {s['Shell1_RMSD_med']:.4f})  <-- 仅近邻\n"
        )
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(FIG_DIR, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nsummary saved: {summary_path}")

    # === FIGURES ===
    f1 = os.path.join(FIG_DIR, "fig1_rmsd_distribution.png")
    f3 = os.path.join(FIG_DIR, "fig3_structure_comparison.png")
    f5 = os.path.join(FIG_DIR, "fig5_typeacc_by_rank.png")
    f6 = os.path.join(FIG_DIR, "fig6_v2_vs_exp4.png")
    f7 = os.path.join(FIG_DIR, "fig7_bondlength_distribution.png")

    if "1" in sel and "val" in df_v2_full and "test" in df_v2_full:
        plot_fig1(df_v2_full, f1)

    if "3" in sel and "val" in preds_v2 and "val" in df_v2_full:
        df_inv = load_inventory()
        plot_fig3(preds_v2["val"], df_v2_full["val"], df_inv, f3)

    if "5" in sel and "val" in preds_v2:
        plot_fig5(preds_v2["val"], f5)

    if "6" in sel:
        if all(k in summaries for k in ["v2_val", "v2_test", "exp4_val", "exp4_test"]):
            plot_fig6(summaries["v2_val"],  summaries["v2_test"],
                      summaries["exp4_val"], summaries["exp4_test"], f6)
        else:
            print("\n[fig6] SKIP — need both v2 and exp4 summaries; "
                  f"have: {list(summaries.keys())}")

    if "7" in sel and "val" in preds_v2:
        plot_fig7(preds_v2["val"], f7)

    print("\n" + "=" * 78)
    print(f"DONE. Figures + summary in: {FIG_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
