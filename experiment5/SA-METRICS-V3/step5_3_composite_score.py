#!/usr/bin/env python
"""
step5_3_composite_score.py — Exp5 v2 SA-METRICS-V3

7 项复合评分 + min_d 1.5 Å 物理 gate,基于 SA3' predictions_v2_*.pt
(from SA2 baseline ckpt epoch=484-val_loss=0.7065).

Anchor: EXP5_SA_METRICS_V3_LAUNCH_NOTE.md §3
Author: SA-METRICS-V3 (Exp5 v2 extension sub-agent), 2026-05-01

不 import step5_2_compute_metrics.py 任何函数(防 SA1' 5.5 Å fallback bug 传染)。
任何不确定查询失败 → raise,不静默 fallback。
"""
import argparse
import csv
import hashlib
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------
# 写死常量(MA2 拍板,跨实验不变)
# ----------------------------------------------------------------------------
SHELL_GAP_THRESHOLD = 0.1563  # Å, Exp4 Step 2.5 p10 全局阈值(MA2 拍板)
MIN_PAIRWISE_DIST   = 1.5     # Å, FEFF/EXAFS 化学键物理下限(NOT a tunable hyper)

CNO_SET   = frozenset({6, 7, 8})  # Z(C)=6, Z(N)=7, Z(O)=8 — EXAFS amplitude 几乎不可分
CNO_TOKEN = -1                    # 合并 token,避开真实 Z 值

# 容错(launch note §3.3 / proposal §B.2 拍板)
TOL_SHELL1_COORD = 1.5  # ±1.5 atoms
TOL_SHELL2_COORD = 3.0  # ±3 atoms
TOL_DISTANCE     = 0.2  # ±0.2 Å (both shells)

# 评分函数衰减斜率分母(launch note §B.2)
COORD_DECAY_DENOM    = 3.0
DISTANCE_DECAY_DENOM = 0.5

# 权重(总和 = 1.0)
W_SHELL1 = 0.20  # × 3 sub-scores = 0.60
W_SHELL2 = 0.10  # × 3 sub-scores = 0.30
# total weights = 0.60 + 0.30 = 0.90, NOT 1.0 — 这是 proposal §B.2 原始拍板
# 设计意图:shell-1 信号比 shell-2 重要,总权重 0.90 是历史选择
# 可选归一化(/0.90 让 max=1.0)— launch note §3.3 / proposal 都未要求,这里保留 0.90 max

ASSERT_EVAL_CUTOFF_TOL = 1e-4  # eval_cutoff pred vs sb 一致性容差

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
SHELL_BOUND_PATH = '/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl'
PRED_DIR         = '/home/tcat/diffcsp_exp5/code/step5'
LOG_DIR          = '/home/tcat/diffcsp_exp5/logs'

# ----------------------------------------------------------------------------
# NOTE on import hygiene (file_guide §8):
# This file imports stdlib + numpy + torch only — no diffcsp_exp5 / diffcsp_exp4
# modules — so the SA1' module.__file__ assertion pattern (used in train.py to
# detect Exp4 backbone shadow) does not apply here. step5_2 is deliberately NOT
# imported (launch note §0.1 红线: prevent R_max 5.5 Å fallback bug propagation).
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------------
def compute_min_pairwise(pred_xyz: torch.Tensor) -> float:
    """pred_xyz: (20, 3) cart Å. 全 20 原子 pre-cutoff(物理 gate 不削弱)."""
    assert pred_xyz.dim() == 2 and pred_xyz.shape[1] == 3, \
        f'pred_xyz must be (N, 3), got {pred_xyz.shape}'
    pw = torch.cdist(pred_xyz, pred_xyz)
    pw.fill_diagonal_(float('inf'))
    return pw.min().item()


# ----------------------------------------------------------------------------
# Step 2.5 gap algorithm 用在 pred 端(镜像真值端切壳逻辑)
# ----------------------------------------------------------------------------
def assign_pred_shells(pred_frac_coords: torch.Tensor,
                       pred_atom_types: torch.Tensor,
                       L: float,
                       eval_cutoff: float,
                       gap_threshold: float = SHELL_GAP_THRESHOLD) -> dict:
    """gap > threshold 处切壳,镜像 Exp4 Step 2.5 真值端算法。

    Returns dict with: n_pred_shells, shell{1,2}_distances, shell{1,2}_species_Z, shell{1,2}_n.
    NOTE: shell-2 includes any further shells (shell-3+) — see report header annotation.
    """
    pred_xyz = pred_frac_coords * L                                   # (20, 3)
    radial   = pred_xyz.norm(dim=1)                                   # (20,)

    in_mask  = radial <= eval_cutoff
    radial_in = radial[in_mask]
    Z_in      = pred_atom_types[in_mask]

    # 排序
    order    = radial_in.argsort()
    sorted_d = radial_in[order]
    sorted_Z = Z_in[order]

    # Edge case: 0 或 1 in-cutoff atom
    if len(sorted_d) < 2:
        return {
            'n_pred_shells': 1 if len(sorted_d) == 1 else 0,
            'shell1_distances': sorted_d, 'shell1_species_Z': sorted_Z,
            'shell1_n': int(len(sorted_d)),
            'shell2_distances': None, 'shell2_species_Z': None, 'shell2_n': 0,
        }

    # gap > threshold 处为壳层边界
    gaps       = sorted_d[1:] - sorted_d[:-1]
    boundaries = (gaps > gap_threshold).nonzero(as_tuple=True)[0] + 1  # int indices

    n_pred_shells = int(boundaries.numel()) + 1
    if n_pred_shells == 1:
        shell1_end   = len(sorted_d)
        shell2_start = None
        shell2_end   = None
    else:
        shell1_end   = int(boundaries[0].item())
        shell2_start = shell1_end
        shell2_end   = len(sorted_d)  # shell-2 吸收所有后续 shells(launch note §3.2 注释)

    pred_shell1_distances = sorted_d[:shell1_end]
    pred_shell1_species_Z = sorted_Z[:shell1_end]
    if shell2_start is not None:
        pred_shell2_distances = sorted_d[shell2_start:shell2_end]
        pred_shell2_species_Z = sorted_Z[shell2_start:shell2_end]
    else:
        pred_shell2_distances = None
        pred_shell2_species_Z = None

    return {
        'n_pred_shells':    n_pred_shells,
        'shell1_distances': pred_shell1_distances,
        'shell1_species_Z': pred_shell1_species_Z,
        'shell1_n':         int(len(pred_shell1_distances)),
        'shell2_distances': pred_shell2_distances,
        'shell2_species_Z': pred_shell2_species_Z,
        'shell2_n':         int(len(pred_shell2_distances)) if pred_shell2_distances is not None else 0,
    }


# ----------------------------------------------------------------------------
# Score functions
# ----------------------------------------------------------------------------
def score_coord_n(pred_n: int, true_n: int, tol: float) -> float:
    delta = abs(pred_n - true_n)
    if delta <= tol:
        return 1.0
    return max(0.0, 1.0 - (delta - tol) / COORD_DECAY_DENOM)


def score_distance(pred_d_mean, true_d_mean, tol: float = TOL_DISTANCE) -> float:
    """Both args: float or None. None on either side → 0.0 (壳层缺失公平惩罚)."""
    if pred_d_mean is None or true_d_mean is None:
        return 0.0
    delta = abs(float(pred_d_mean) - float(true_d_mean))
    if delta <= tol:
        return 1.0
    return max(0.0, 1.0 - (delta - tol) / DISTANCE_DECAY_DENOM)


def cno_token(z) -> int:
    """C/N/O → CNO_TOKEN, others → int(z)."""
    z_int = int(z)
    return CNO_TOKEN if z_int in CNO_SET else z_int


def score_element(pred_Z, true_Z) -> float:
    """Multiset 交集 / 总数,C/N/O 合并 token."""
    if pred_Z is None or true_Z is None:
        return 0.0
    pred_tokens = [cno_token(z) for z in pred_Z.tolist()]
    true_tokens = [cno_token(z) for z in true_Z.tolist()]
    pred_c = Counter(pred_tokens)
    true_c = Counter(true_tokens)
    inter  = sum((pred_c & true_c).values())
    total  = max(sum(pred_c.values()), sum(true_c.values()))
    return inter / total if total > 0 else 0.0


# ----------------------------------------------------------------------------
# Per-sample 主逻辑
# ----------------------------------------------------------------------------
def compute_one_sample(sample_name: str, pred_fc: torch.Tensor,
                       pred_at: torch.Tensor, pred_eval_cutoff: float,
                       L: float, sb_i: dict) -> dict:
    """Returns 12-field csv row dict + 6 raw sub-scores for aggregate stats."""
    # --- Gate (pre-cutoff, all 20 atoms) ---
    pred_xyz  = pred_fc * L
    min_d     = compute_min_pairwise(pred_xyz)
    gate_pass = bool(min_d >= MIN_PAIRWISE_DIST)

    # --- eval_cutoff 一致性 assert (launch note §3.5 红线) ---
    sb_eval_cutoff = float(sb_i['eval_cutoff'])
    if abs(pred_eval_cutoff - sb_eval_cutoff) >= ASSERT_EVAL_CUTOFF_TOL:
        raise RuntimeError(
            f'eval_cutoff mismatch for {sample_name!r}: '
            f'pred={pred_eval_cutoff} vs sb={sb_eval_cutoff}'
        )

    # --- Pred shell 分配(用 eval_cutoff filter 后,launch note §3.2)---
    pred_shells = assign_pred_shells(pred_fc, pred_at, L, pred_eval_cutoff)

    # --- 真值端 lookup(直接用 Exp4 Step 2.5 已切好的 ground truth)---
    sb_distances    = np.asarray(sb_i['distances'])
    sb_species_Z    = np.asarray(sb_i['species_Z'])
    sb_shell_of     = np.asarray(sb_i['shell_of_atom'])
    sb_shell_n_atom = np.asarray(sb_i['shell_n_atoms'])

    # 真值 shell-1
    true_s1_mask = (sb_shell_of == 0)
    true_s1_d    = torch.from_numpy(sb_distances[true_s1_mask].astype(np.float32))
    true_s1_Z    = torch.from_numpy(sb_species_Z[true_s1_mask].astype(np.int64))
    true_s1_n    = int(sb_shell_n_atom[0]) if len(sb_shell_n_atom) >= 1 else 0

    # 真值 shell-2(可能不存在)
    if len(sb_shell_n_atom) >= 2 and int(sb_shell_n_atom[1]) > 0:
        true_s2_mask = (sb_shell_of == 1)
        true_s2_d    = torch.from_numpy(sb_distances[true_s2_mask].astype(np.float32))
        true_s2_Z    = torch.from_numpy(sb_species_Z[true_s2_mask].astype(np.int64))
        true_s2_n    = int(sb_shell_n_atom[1])
    else:
        true_s2_d, true_s2_Z, true_s2_n = None, None, 0

    # --- 6 sub-scores ---
    if gate_pass:
        s1n = score_coord_n(pred_shells['shell1_n'], true_s1_n, tol=TOL_SHELL1_COORD)
        s1d = score_distance(
            pred_shells['shell1_distances'].mean().item() if pred_shells['shell1_n'] > 0 else None,
            true_s1_d.mean().item() if true_s1_n > 0 else None,
        )
        s1e = score_element(pred_shells['shell1_species_Z'], true_s1_Z)
        s2n = score_coord_n(pred_shells['shell2_n'], true_s2_n, tol=TOL_SHELL2_COORD)
        s2d = score_distance(
            pred_shells['shell2_distances'].mean().item() if pred_shells['shell2_n'] > 0 else None,
            true_s2_d.mean().item() if true_s2_d is not None else None,
        )
        s2e = score_element(pred_shells['shell2_species_Z'], true_s2_Z)
        total = (W_SHELL1 * s1n + W_SHELL1 * s1d + W_SHELL1 * s1e
                 + W_SHELL2 * s2n + W_SHELL2 * s2d + W_SHELL2 * s2e)
    else:
        s1n = s1d = s1e = s2n = s2d = s2e = 0.0
        total = 0.0

    return {
        'sample_name':         sample_name,
        'gate_pass':           int(gate_pass),
        'min_d':               round(min_d, 6),
        'n_pred_shells':       pred_shells['n_pred_shells'],
        'score_shell1_coord':  round(s1n, 6),
        'score_shell1_dist':   round(s1d, 6),
        'score_shell1_elem':   round(s1e, 6),
        'score_shell2_coord':  round(s2n, 6),
        'score_shell2_dist':   round(s2d, 6),
        'score_shell2_elem':   round(s2e, 6),
        'total_score':         round(total, 6),
    }


# ----------------------------------------------------------------------------
# Aggregate + 主报告
# ----------------------------------------------------------------------------
def write_main_report(rows: list, split: str, out_path: Path,
                      pred_pt_path: str, ckpt_label: str) -> None:
    """主报告 txt(launch note §3.4 模板原样)."""
    N         = len(rows)
    gate_pass = [r for r in rows if r['gate_pass'] == 1]
    n_gp      = len(gate_pass)
    n_gf      = N - n_gp

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # All-samples means (gate-fail counted as 0)
    means_all = {
        'shell1_coord': _mean([r['score_shell1_coord'] for r in rows]),
        'shell1_dist':  _mean([r['score_shell1_dist']  for r in rows]),
        'shell1_elem':  _mean([r['score_shell1_elem']  for r in rows]),
        'shell2_coord': _mean([r['score_shell2_coord'] for r in rows]),
        'shell2_dist':  _mean([r['score_shell2_dist']  for r in rows]),
        'shell2_elem':  _mean([r['score_shell2_elem']  for r in rows]),
        'total':        _mean([r['total_score']       for r in rows]),
    }

    # Gate-pass subset means (key diagnostic)
    means_gp = {
        'shell1_coord': _mean([r['score_shell1_coord'] for r in gate_pass]),
        'shell1_dist':  _mean([r['score_shell1_dist']  for r in gate_pass]),
        'shell1_elem':  _mean([r['score_shell1_elem']  for r in gate_pass]),
        'shell2_coord': _mean([r['score_shell2_coord'] for r in gate_pass]),
        'shell2_dist':  _mean([r['score_shell2_dist']  for r in gate_pass]),
        'shell2_elem':  _mean([r['score_shell2_elem']  for r in gate_pass]),
        'total':        _mean([r['total_score']       for r in gate_pass]),
    } if n_gp > 0 else {k: 0.0 for k in means_all}

    # min_d distribution
    min_ds = np.array([r['min_d'] for r in rows], dtype=np.float64)
    md_stats = {
        'mean':   float(min_ds.mean()),
        'median': float(np.median(min_ds)),
        'p10':    float(np.percentile(min_ds, 10)),
        'p1':     float(np.percentile(min_ds, 1)),
        'min':    float(min_ds.min()),
        'max':    float(min_ds.max()),
    }
    md_buckets = {
        '<1.5 (gate fail)': int((min_ds < 1.5).sum()),
        '<1.0':             int((min_ds < 1.0).sum()),
        '<0.5':             int((min_ds < 0.5).sum()),
        '<0.1 (overlap)':   int((min_ds < 0.1).sum()),
    }

    # n_pred_shells distribution
    nps = np.array([r['n_pred_shells'] for r in rows], dtype=np.int32)
    nps_buckets = {
        '0':   int((nps == 0).sum()),
        '1':   int((nps == 1).sum()),
        '2':   int((nps == 2).sum()),
        '3':   int((nps == 3).sum()),
        '>=4': int((nps >= 4).sum()),
    }

    lines = []
    lines.append(f'=== EXP5 V2 SA-METRICS-V3 COMPOSITE SCORE - {split} ===')
    lines.append(f'Source: {pred_pt_path}')
    lines.append(f'Ckpt:   {ckpt_label}')
    lines.append('Shell algorithm: Exp4 Step 2.5 gap-based, threshold=0.1563 \u00c5 '
                 '(applied to both true and pred)')
    lines.append('Note: shell-2 = first remaining gap-bounded group after shell-1; '
                 'may include shell-3+ atoms.')
    lines.append('Note: MIN_PAIRWISE_DIST = 1.5 \u00c5 is a physical lower bound, '
                 'NOT a tunable hyper.')
    lines.append('')
    lines.append(f'Total samples:           {N}')
    lines.append(f'min_d gate pass:         {n_gp} / {N} ({100*n_gp/N:.1f}%)')
    lines.append(f'min_d gate fail:         {n_gf} / {N} ({100*n_gf/N:.1f}%)')
    lines.append('')
    lines.append('--- Composite score (ALL samples, gate-fail counts as 0) ---')
    lines.append(f'Total weighted mean:     {means_all["total"]:.4f}')
    lines.append(f'  shell-1 coord_n:       {means_all["shell1_coord"]:.4f}  (weight {W_SHELL1:.2f})')
    lines.append(f'  shell-1 distance:      {means_all["shell1_dist"]:.4f}   (weight {W_SHELL1:.2f})')
    lines.append(f'  shell-1 elem (CNO eq): {means_all["shell1_elem"]:.4f}   (weight {W_SHELL1:.2f})')
    lines.append(f'  shell-2 coord_n:       {means_all["shell2_coord"]:.4f}  (weight {W_SHELL2:.2f})')
    lines.append(f'  shell-2 distance:      {means_all["shell2_dist"]:.4f}   (weight {W_SHELL2:.2f})')
    lines.append(f'  shell-2 elem (CNO eq): {means_all["shell2_elem"]:.4f}   (weight {W_SHELL2:.2f})')
    lines.append('')
    lines.append('--- Composite score (GATE-PASS subset only, key diagnostic) ---')
    lines.append(f'N_gate_pass:             {n_gp}')
    lines.append(f'Total weighted mean:     {means_gp["total"]:.4f}')
    lines.append(f'  shell-1 coord_n:       {means_gp["shell1_coord"]:.4f}')
    lines.append(f'  shell-1 distance:      {means_gp["shell1_dist"]:.4f}')
    lines.append(f'  shell-1 elem (CNO eq): {means_gp["shell1_elem"]:.4f}')
    lines.append(f'  shell-2 coord_n:       {means_gp["shell2_coord"]:.4f}')
    lines.append(f'  shell-2 distance:      {means_gp["shell2_dist"]:.4f}')
    lines.append(f'  shell-2 elem (CNO eq): {means_gp["shell2_elem"]:.4f}')
    lines.append('')
    lines.append('--- min_d distribution ---')
    lines.append(f'mean={md_stats["mean"]:.4f}, median={md_stats["median"]:.4f}, '
                 f'p10={md_stats["p10"]:.4f}, p1={md_stats["p1"]:.4f}, '
                 f'min={md_stats["min"]:.4f}, max={md_stats["max"]:.4f} (\u00c5)')
    for k, v in md_buckets.items():
        lines.append(f'samples with min_d {k:20s}: {v}')
    lines.append('')
    lines.append('--- n_pred_shells distribution ---')
    for k, v in nps_buckets.items():
        lines.append(f'  {k:5s} shells: {v}')
    lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')


def write_per_sample_csv(rows: list, out_path: Path) -> None:
    fieldnames = [
        'sample_name', 'gate_pass', 'min_d', 'n_pred_shells',
        'score_shell1_coord', 'score_shell1_dist', 'score_shell1_elem',
        'score_shell2_coord', 'score_shell2_dist', 'score_shell2_elem',
        'total_score',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_violations_csv(rows: list, out_path: Path) -> None:
    """gate_pass == 0 子集,Exp5' λ schedule 直接用."""
    fieldnames = ['sample_name', 'min_d', 'n_pred_shells']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r['gate_pass'] == 0:
                w.writerow({k: r[k] for k in fieldnames})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Exp5 v2 SA-METRICS-V3 composite scoring')
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    ap.add_argument('--debug-n-samples', type=int, default=None,
                    help='Dry-run cap; output filenames suffixed _debug<N> when set.')
    ap.add_argument('--shell-bound-path', default=SHELL_BOUND_PATH)
    ap.add_argument('--pred-dir', default=PRED_DIR)
    ap.add_argument('--log-dir', default=LOG_DIR)
    args = ap.parse_args()

    t0 = time.time()
    pred_pt_path = f'{args.pred_dir}/predictions_v2_{args.split}.pt'
    print(f'[step5_3] split={args.split}', flush=True)
    print(f'[step5_3] loading shell_boundaries: {args.shell_bound_path}', flush=True)
    with open(args.shell_bound_path, 'rb') as f:
        sb = pickle.load(f)
    print(f'[step5_3]   loaded {len(sb)} samples in shell_boundaries.pkl', flush=True)

    print(f'[step5_3] loading predictions: {pred_pt_path}', flush=True)
    p = torch.load(pred_pt_path, map_location='cpu', weights_only=False)
    sample_names     = p['sample_name']
    pred_fc_list     = p['pred_frac_coords']
    pred_at_list     = p['pred_atom_types']
    eval_cutoff_list = p['eval_cutoff']
    L                = float(p['L'])
    ckpt_label       = p.get('checkpoint', 'unknown')

    N_total = len(sample_names)
    if args.debug_n_samples is not None:
        N = min(args.debug_n_samples, N_total)
        print(f'[step5_3] DRY-RUN: processing first {N} of {N_total} samples', flush=True)
        suffix = f'_debug{args.debug_n_samples}'
    else:
        N = N_total
        suffix = ''
        print(f'[step5_3] FULL RUN: processing all {N} samples', flush=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_loop = time.time()
    for i in range(N):
        sn = sample_names[i]
        if sn not in sb:
            raise RuntimeError(f'sample_name not in shell_boundaries: {sn!r} (i={i})')
        row = compute_one_sample(
            sample_name=sn,
            pred_fc=pred_fc_list[i],
            pred_at=pred_at_list[i],
            pred_eval_cutoff=float(eval_cutoff_list[i]),
            L=L,
            sb_i=sb[sn],
        )
        rows.append(row)
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            elapsed = time.time() - t_loop
            rate    = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f'[step5_3]   {i+1}/{N}  rate={rate:.0f} samples/s', flush=True)

    # Outputs
    main_path = log_dir / f'composite_score_{args.split}{suffix}.txt'
    perr_path = log_dir / f'composite_score_per_sample_{args.split}{suffix}.csv'
    viol_path = log_dir / f'min_d_violations_{args.split}{suffix}.csv'

    write_main_report(rows, args.split, main_path, pred_pt_path, str(ckpt_label))
    write_per_sample_csv(rows, perr_path)
    write_violations_csv(rows, viol_path)

    wall = time.time() - t0
    print(f'[step5_3] DONE split={args.split} N={N} wall={wall:.1f}s', flush=True)
    print(f'[step5_3]   main:       {main_path}', flush=True)
    print(f'[step5_3]   per-sample: {perr_path}', flush=True)
    print(f'[step5_3]   violations: {viol_path}', flush=True)
    for path in (main_path, perr_path, viol_path):
        size = path.stat().st_size
        md5  = hashlib.md5(path.read_bytes()).hexdigest()
        print(f'[step5_3]   {path.name}: {size} bytes, md5={md5}', flush=True)


if __name__ == '__main__':
    main()
