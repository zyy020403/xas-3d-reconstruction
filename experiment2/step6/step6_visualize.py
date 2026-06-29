# step6_visualize.py
# Step6 — 结果可视化（Experiment 2）
# ============================================================
# 输入：experiment2/step4d/predictions_val.pt
# 输出：experiment2/step6/figures/fig{1..4}_*.png
#
# 图1：RMSD 分布直方图
# 图2：Type Accuracy 分布直方图
# 图3：3D 结构对比（6 个样本：best×2 / mid×2 / worst×2）
# 图4：RMSD vs Type Accuracy 散点 + 线性回归
#
# 坐标约定：frac ∈ [-0.5, 0.5]，Cartesian = frac × L（L=6.0）
# 匹配：最小镜像匈牙利匹配（与 step4d_4_compute_metrics.py 一致）
# ============================================================

import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (register 3d projection)
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP4D_DIR   = os.path.join(EXP2_ROOT, "step4d")
STEP6_DIR    = os.path.join(EXP2_ROOT, "step6")
FIG_DIR      = os.path.join(STEP6_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

L = 6.0
RANDOM_RMSD_BASELINE = (L / 2) * (3 / 5) ** 0.5   # ≈ 2.32 Å

# ─── Global style ────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'axes.titlesize':  14,
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi':      150,
})


# ─── Hungarian matching（最小镜像）────────────────────────────
def hungarian_match(pred_frac: np.ndarray, true_frac: np.ndarray, L: float = 6.0):
    n = pred_frac.shape[0]
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        delta = pred_frac[i] - true_frac                 # (n, 3)
        delta -= np.round(delta)                          # 最小镜像
        cost[i] = np.linalg.norm(delta * L, axis=1)
    row, col = linear_sum_assignment(cost)
    return row, col


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, L=6.0):
    row, col = hungarian_match(pred_frac, true_frac, L)
    matched_sq = []
    for ri, ci in zip(row, col):
        d = pred_frac[ri] - true_frac[ci]
        d -= np.round(d)
        matched_sq.append(np.sum((d * L) ** 2))
    rmsd     = float(np.sqrt(np.mean(matched_sq)))
    type_acc = float((pred_types[row] == true_types[col]).mean())
    return rmsd, type_acc, row, col


# ─── 元素配色（按原子序数，不暴露化学式）────────────────────
_TAB10 = list(plt.get_cmap('tab10').colors)
# 将 O / Fe 对应的位置从 tab10 中剔除，避免重复
_OTHER_PALETTE = [c for i, c in enumerate(_TAB10) if i not in (1, 3)]


def element_color(Z: int):
    if Z == 8:    # O
        return '#D62728'          # 红
    if Z == 26:   # Fe
        return '#FF7F0E'          # 橙
    return _OTHER_PALETTE[int(Z) % len(_OTHER_PALETTE)]


# ─── Data loading ─────────────────────────────────────────────
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_predictions(path: str) -> dict:
    print(f"Loading: {path}")
    preds = torch.load(path, map_location='cpu', weights_only=False)
    return preds


def compute_all_records(preds: dict):
    """Returns a list of per-sample records with metrics + arrays."""
    n = len(preds['mp_id'])
    records, skipped = [], 0
    for i in range(n):
        pf = _to_np(preds['pred_frac_coords'][i])
        pt = _to_np(preds['pred_atom_types'][i])
        tf = _to_np(preds['true_frac_coords'][i])
        tt = _to_np(preds['true_atom_types'][i])

        if pf.shape[0] != 20 or tf.shape[0] != 20:
            skipped += 1
            continue

        rmsd, tacc, row, col = evaluate_sample(pf, pt, tf, tt, L=L)
        records.append({
            'idx':        i,
            'rmsd':       rmsd,
            'type_acc':   tacc,
            'pred_frac':  pf,
            'pred_types': pt,
            'true_frac':  tf,
            'true_types': tt,
            'row':        row,
            'col':        col,
        })
    print(f"Valid samples: {len(records)}/{n}  (skipped={skipped})")
    return records


# ─── Figure 1: RMSD distribution ──────────────────────────────
def plot_fig1_rmsd(records, save_path):
    rmsds = np.array([r['rmsd'] for r in records])
    mean_rmsd = float(rmsds.mean())

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(rmsds, bins=40, range=(0, 4),
            color='#1f77b4', edgecolor='white', alpha=0.88)
    ax.axvline(mean_rmsd, color='#FF7F0E', linestyle='--', linewidth=2,
               label=f'Mean RMSD = {mean_rmsd:.2f} Å')
    ax.axvline(RANDOM_RMSD_BASELINE, color='#D62728', linestyle='--', linewidth=2,
               label=f'Random baseline = {RANDOM_RMSD_BASELINE:.2f} Å')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Number of samples')
    ax.set_title(f'RMSD Distribution (Val Set, N={len(rmsds)})')
    ax.set_xlim(0, 4)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {save_path}  (mean RMSD = {mean_rmsd:.4f} Å)")
    return mean_rmsd


# ─── Figure 2: Type Accuracy distribution ─────────────────────
def plot_fig2_typeacc(records, save_path):
    taccs = np.array([r['type_acc'] for r in records])
    mean_tacc = float(taccs.mean())

    # 21 bins covering k/20 for k=0..20
    edges = np.linspace(-0.5 / 20, 20.5 / 20, 22)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(taccs, bins=edges, color='#1f77b4', edgecolor='white', alpha=0.88)
    ax.axvline(mean_tacc, color='#FF7F0E', linestyle='--', linewidth=2,
               label=f'Mean Type Acc = {mean_tacc:.3f}')
    ax.axvline(0.01, color='#D62728', linestyle='--', linewidth=2,
               label='Random baseline = 0.01')
    ax.set_xlabel('Type Accuracy (correct / 20)')
    ax.set_ylabel('Number of samples')
    ax.set_title(f'Type Accuracy Distribution (Val Set, N={len(taccs)})')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {save_path}  (mean TypeAcc = {mean_tacc:.4f})")
    return mean_tacc


# ─── Figure 3: 3D structure comparison ────────────────────────
def select_six_samples(records):
    """2 best, 2 near mean (1.4~1.5), 2 worst (<=3.5)."""
    by_rmsd = sorted(records, key=lambda r: r['rmsd'])
    best = by_rmsd[:2]

    near = [r for r in records if 1.40 <= r['rmsd'] <= 1.50]
    if len(near) >= 2:
        near = sorted(near, key=lambda r: abs(r['rmsd'] - 1.47))
    else:
        # 兜底：最接近 1.47 的两个
        near = sorted(records, key=lambda r: abs(r['rmsd'] - 1.47))
    mid = near[:2]

    bounded = [r for r in records if r['rmsd'] <= 3.5]
    worst = sorted(bounded, key=lambda r: r['rmsd'], reverse=True)[:2]

    return best + mid + worst


def _draw_panel(ax, rec, group_label):
    pf, pt = rec['pred_frac'], rec['pred_types']
    tf, tt = rec['true_frac'], rec['true_types']
    row, col = rec['row'], rec['col']

    # 折叠到 [-0.5, 0.5]（对该坐标系是 no-op，保险起见）
    pf_f = pf - np.round(pf)
    tf_f = tf - np.round(tf)
    pc = pf_f * L
    tc = tf_f * L

    # True atoms（大实心球 + 黑边）
    for i in range(tc.shape[0]):
        c = element_color(int(tt[i]))
        ax.scatter(tc[i, 0], tc[i, 1], tc[i, 2],
                   s=90, c=[c], edgecolors='black',
                   linewidths=0.8, depthshade=True)

    # Predicted atoms（小空心圆 + 同色边）
    for i in range(pc.shape[0]):
        c = element_color(int(pt[i]))
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2],
                   s=55, facecolors='none', edgecolors=[c],
                   linewidths=1.6, depthshade=True)

    # 匹配对（虚线）
    for ri, ci in zip(row, col):
        ax.plot([pc[ri, 0], tc[ci, 0]],
                [pc[ri, 1], tc[ci, 1]],
                [pc[ri, 2], tc[ci, 2]],
                color='k', linestyle='--', linewidth=0.5, alpha=0.45)

    # Fe 原点（红星）
    ax.scatter([0], [0], [0], marker='*', s=320, c='red',
               edgecolors='black', linewidths=1.0, depthshade=False, zorder=20)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-3.5, 3.5)
    ax.set_xlabel('x (Å)', fontsize=9)
    ax.set_ylabel('y (Å)', fontsize=9)
    ax.set_zlabel('z (Å)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"[{group_label}]  RMSD={rec['rmsd']:.2f} Å,  TypeAcc={rec['type_acc']:.2f}",
        fontsize=11,
    )


def plot_fig3_structures(records, save_path):
    chosen = select_six_samples(records)
    labels = ['Best #1', 'Best #2', 'Mid #1', 'Mid #2', 'Worst #1', 'Worst #2']

    fig = plt.figure(figsize=(17, 10))
    for k, (rec, lab) in enumerate(zip(chosen, labels), start=1):
        ax = fig.add_subplot(2, 3, k, projection='3d')
        _draw_panel(ax, rec, lab)

    # 全局图例（右侧）
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=11, label='True atoms'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markeredgewidth=1.6, markersize=10,
               label='Predicted atoms'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=17, label='Fe center (origin)'),
        Line2D([0], [0], linestyle='--', color='k', alpha=0.5,
               label='Matched pair'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D62728',
               markeredgecolor='black', markersize=10, label='O (Z=8)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E',
               markeredgecolor='black', markersize=10, label='Fe (Z=26)'),
    ]
    fig.legend(handles=legend_handles,
               loc='center right', bbox_to_anchor=(0.995, 0.5),
               frameon=True, fontsize=10)

    fig.suptitle('3D Structure Comparison: True vs. Predicted (Val Set)',
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    info = [(lab, r['rmsd'], r['type_acc']) for lab, r in zip(labels, chosen)]
    print(f"  ✅ {save_path}")
    for lab, rm, ta in info:
        print(f"     {lab:10s}  RMSD={rm:.3f} Å  TypeAcc={ta:.3f}")
    return info


# ─── Figure 4: RMSD vs Type Accuracy ─────────────────────────
def plot_fig4_corr(records, save_path):
    rmsds = np.array([r['rmsd']     for r in records])
    taccs = np.array([r['type_acc'] for r in records])

    r_val, p_val = stats.pearsonr(rmsds, taccs)
    slope, intercept, *_ = stats.linregress(rmsds, taccs)
    xs = np.linspace(rmsds.min(), rmsds.max(), 200)
    ys = slope * xs + intercept

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(rmsds, taccs, s=8, alpha=0.3, color='#1f77b4',
               edgecolors='none', label=f'Samples (N={len(rmsds)})')
    ax.plot(xs, ys, color='#D62728', linewidth=2, label='Linear fit')
    ax.set_xlabel('RMSD (Å)')
    ax.set_ylabel('Type Accuracy')
    ax.set_title('RMSD vs Type Accuracy Correlation (Val Set)')

    annot = f'Pearson r = {r_val:.3f}\np = {p_val:.2e}'
    ax.text(0.97, 0.97, annot,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {save_path}  (Pearson r = {r_val:.4f}, p = {p_val:.2e})")
    return r_val, p_val


# ─── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step6 Visualization")
    print("=" * 60)

    val_path = os.path.join(STEP4D_DIR, "predictions_val.pt")
    preds = load_predictions(val_path)
    records = compute_all_records(preds)

    if len(records) == 0:
        print("❌ No valid samples loaded — aborting.")
        return

    f1 = os.path.join(FIG_DIR, "fig1_rmsd_distribution.png")
    f2 = os.path.join(FIG_DIR, "fig2_typeacc_distribution.png")
    f3 = os.path.join(FIG_DIR, "fig3_structure_comparison.png")
    f4 = os.path.join(FIG_DIR, "fig4_rmsd_vs_typeacc.png")

    print("\n── Generating figures ──")
    plot_fig1_rmsd(records,       f1)
    plot_fig2_typeacc(records,    f2)
    plot_fig3_structures(records, f3)
    plot_fig4_corr(records,       f4)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
