# step6_composite_eval.py
# DiffCSP-Exp4 Step 6 — Composite Re-Scoring of Exp4 Predictions
# ============================================================================
# Re-scores predictions_{val,test,holdout}.pt with EXP5_PROPOSAL_v2_AMENDED §B-
# style criteria, applied to Exp4 outputs.
#
# Front gate:
#   Min pairwise CARTESIAN distance over the 20 predicted atoms ≥ 1.5 Å.
#   (Raw cartesian, no PBC / min-image — L=6 is a local-cluster bounding box,
#    not a periodic supercell.)
#
# 6 weighted sub-scores (gate-passed samples only, weights sum to 1.0):
#   s1_CN   w=0.20  shell-1 coordination number     tol = ±1.5 atoms
#   s1_d    w=0.20  shell-1 mean distance           tol = ±0.2 Å
#   s1_elem w=0.20  shell-1 element multiset        multiset Jaccard, CNO-eq
#   s2_CN   w=0.10  shell-2 coordination number     tol = ±3.0 atoms
#   s2_d    w=0.10  shell-2 mean distance           tol = ±0.2 Å
#   s2_elem w=0.10  shell-2 element multiset        multiset Jaccard, CNO-eq
#
# Scoring formulas:
#   tolerance score: score = max(0, 1 - |delta| / tolerance)        (linear decay)
#   multiset Jaccard: |P ∩ T| / |P ∪ T| over multiset counts
#   CNO equivalence: Z ∈ {6, 7, 8} all map to single token "CNO";
#     other Z map to pymatgen Element.from_Z(z).symbol
#
# Shell boundaries:
#   /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
#   keyed by sample_name; entry exposes shell_starts / shell_ends (cartesian Å)
#   either as dict items or as attributes — both supported.
#
# Coordinates:
#   L = 6.0, frac ∈ [-0.5, 0.5], cartesian = frac * L
#
# Output:
#   /home/tcat/diffcsp_exp4/code/step6/composite_per_sample_{split}.csv
#   /home/tcat/diffcsp_exp4/code/step6/composite_summary.csv
#   stdout: probe + per-split breakdown + 3-split summary table
#
# Run with EXPLICIT mlff env:
#   /home/tcat/conda_envs/mlff/bin/python step6_composite_eval.py
#   /home/tcat/conda_envs/mlff/bin/python step6_composite_eval.py --probe-only
#   /home/tcat/conda_envs/mlff/bin/python step6_composite_eval.py --splits val
# ============================================================================

import argparse
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from pymatgen.core import Element


# ─── Paths ───────────────────────────────────────────────────────────────────
EXP_ROOT  = "/home/tcat/diffcsp_exp4"
STEP5_DIR = os.path.join(EXP_ROOT, "code", "step5")
STEP6_DIR = os.path.join(EXP_ROOT, "code", "step6")
DATA_DIR  = os.path.join(EXP_ROOT, "data")
OUT_DIR   = STEP6_DIR
os.makedirs(OUT_DIR, exist_ok=True)

PT_PATHS = {
    "val":     os.path.join(STEP5_DIR, "predictions_val.pt"),
    "test":    os.path.join(STEP5_DIR, "predictions_test.pt"),
    "holdout": os.path.join(STEP5_DIR, "predictions_holdout.pt"),
}
SHELL_PKL = os.path.join(DATA_DIR, "shell_boundaries.pkl")


# ─── Scoring constants ───────────────────────────────────────────────────────
L = 6.0
GATE_MIN_DIST = 1.5   # Å — minimum pairwise cartesian distance among 20 atoms

WEIGHTS = {
    "s1_CN":   0.20,
    "s1_d":    0.20,
    "s1_elem": 0.20,
    "s2_CN":   0.10,
    "s2_d":    0.10,
    "s2_elem": 0.10,
}
TOL = {
    "s1_CN": 1.5,   # atoms
    "s1_d":  0.2,   # Å
    "s2_CN": 3.0,   # atoms
    "s2_d":  0.2,   # Å
}
SUB_KEYS = ["s1_CN", "s1_d", "s1_elem", "s2_CN", "s2_d", "s2_elem"]

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def z_to_token(z) -> str:
    """Element symbol with CNO equivalence: Z=6/7/8 → 'CNO'."""
    z = int(z)
    if z in (6, 7, 8):
        return "CNO"
    try:
        return Element.from_Z(z).symbol
    except Exception:
        return f"Z{z}"


def score_tolerance(delta: float, tol: float) -> float:
    """Linear-decay tolerance score: |delta|=0 → 1.0, |delta|≥tol → 0.0."""
    if tol <= 0:
        return 1.0 if delta == 0 else 0.0
    return max(0.0, 1.0 - abs(float(delta)) / float(tol))


def multiset_jaccard(p_tokens, t_tokens) -> float:
    """Multiset Jaccard |P ∩ T| / |P ∪ T|.
       Both empty → 1.0 (vacuous agreement).
       Exactly one empty → 0.0."""
    if len(p_tokens) == 0 and len(t_tokens) == 0:
        return 1.0
    p_cnt = Counter(p_tokens)
    t_cnt = Counter(t_tokens)
    inter = sum((p_cnt & t_cnt).values())
    union = sum((p_cnt | t_cnt).values())
    return float(inter) / float(union) if union > 0 else 0.0


def compute_gate(pred_frac_np: np.ndarray, L: float = L):
    """Raw cartesian pairwise distances over 20 atoms. No PBC.
       Returns (min_dist_float, pass_bool)."""
    pc = pred_frac_np * L                              # (n, 3) cartesian
    diff = pc[:, None, :] - pc[None, :, :]             # (n, n, 3)
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    min_d = float(d.min())
    return min_d, bool(min_d >= GATE_MIN_DIST)


def get_shell_bounds(entry):
    """Defensive extractor — dict-style or attr-style."""
    if isinstance(entry, dict):
        starts = entry["shell_starts"]
        ends   = entry["shell_ends"]
    elif hasattr(entry, "shell_starts") and hasattr(entry, "shell_ends"):
        starts = entry.shell_starts
        ends   = entry.shell_ends
    else:
        raise TypeError(f"shell_boundaries entry has unexpected type: "
                        f"{type(entry).__name__} (keys/attrs not recognized)")
    return [float(x) for x in starts], [float(x) for x in ends]


def classify_shell(dist_to_center, starts, ends, shell_idx) -> np.ndarray:
    """Boolean mask of atoms in [starts[shell_idx], ends[shell_idx]] (inclusive)."""
    if shell_idx >= len(starts) or shell_idx >= len(ends):
        return np.zeros(dist_to_center.shape[0], dtype=bool)
    lo = starts[shell_idx]
    hi = ends[shell_idx]
    return (dist_to_center >= lo) & (dist_to_center <= hi)


def score_sample(pred_frac, pred_types, true_frac, true_types,
                 shell_starts, shell_ends, L: float = L) -> dict:
    """Compute all sub-scores + composite for one sample. Assumes gate passed.
       Returns dict with sub-scores, composite, and debug columns."""
    pc = pred_frac * L
    tc = true_frac * L
    p_d = np.linalg.norm(pc, axis=1)
    t_d = np.linalg.norm(tc, axis=1)

    out: dict = {}
    debug: dict = {}

    for shell_idx, prefix in [(0, "s1"), (1, "s2")]:
        if shell_idx >= len(shell_starts) or shell_idx >= len(shell_ends):
            # missing shell entry — mark sub-scores NaN (will null composite)
            for tail in ["CN", "d", "elem"]:
                out[f"{prefix}_{tail}"] = np.nan
            for col in [f"pred_{prefix}_CN", f"true_{prefix}_CN",
                        f"pred_{prefix}_d",  f"true_{prefix}_d",
                        f"pred_{prefix}_elem", f"true_{prefix}_elem"]:
                debug[col] = np.nan
            continue

        p_mask = classify_shell(p_d, shell_starts, shell_ends, shell_idx)
        t_mask = classify_shell(t_d, shell_starts, shell_ends, shell_idx)

        p_CN = int(p_mask.sum())
        t_CN = int(t_mask.sum())
        debug[f"pred_{prefix}_CN"] = p_CN
        debug[f"true_{prefix}_CN"] = t_CN

        # CN sub-score (linear decay)
        out[f"{prefix}_CN"] = score_tolerance(p_CN - t_CN, TOL[f"{prefix}_CN"])

        # mean-distance sub-score
        if p_CN > 0 and t_CN > 0:
            p_davg = float(p_d[p_mask].mean())
            t_davg = float(t_d[t_mask].mean())
            out[f"{prefix}_d"] = score_tolerance(p_davg - t_davg, TOL[f"{prefix}_d"])
        elif p_CN == 0 and t_CN == 0:
            p_davg = float("nan"); t_davg = float("nan")
            out[f"{prefix}_d"] = 1.0       # both empty → vacuous agreement
        else:
            p_davg = float(p_d[p_mask].mean()) if p_CN > 0 else float("nan")
            t_davg = float(t_d[t_mask].mean()) if t_CN > 0 else float("nan")
            out[f"{prefix}_d"] = 0.0       # one-sided empty → disagree
        debug[f"pred_{prefix}_d"] = p_davg
        debug[f"true_{prefix}_d"] = t_davg

        # element multiset sub-score with CNO equivalence
        p_tokens = [z_to_token(z) for z in pred_types[p_mask]]
        t_tokens = [z_to_token(z) for z in true_types[t_mask]]
        out[f"{prefix}_elem"] = multiset_jaccard(p_tokens, t_tokens)
        debug[f"pred_{prefix}_elem"] = "|".join(sorted(p_tokens)) if p_tokens else ""
        debug[f"true_{prefix}_elem"] = "|".join(sorted(t_tokens)) if t_tokens else ""

    # composite (NaN-propagating sum)
    sub = [out[k] for k in SUB_KEYS]
    if any(np.isnan(v) for v in sub):
        out["composite"] = np.nan
    else:
        out["composite"] = float(sum(WEIGHTS[k] * out[k] for k in SUB_KEYS))

    out.update(debug)
    return out


# ─── Probe ───────────────────────────────────────────────────────────────────
def probe_shell_pkl() -> dict:
    print("=" * 78)
    print("Probe: shell_boundaries.pkl schema")
    print("=" * 78)
    if not os.path.exists(SHELL_PKL):
        print(f"  FAIL: {SHELL_PKL} not found")
        sys.exit(1)
    sz_mb = os.path.getsize(SHELL_PKL) / (1024 * 1024)
    print(f"  path  : {SHELL_PKL}")
    print(f"  size  : {sz_mb:.2f} MB")
    with open(SHELL_PKL, "rb") as f:
        sb = pickle.load(f)
    print(f"  type  : {type(sb).__name__}")
    if not isinstance(sb, dict):
        print("  FAIL: top-level is not a dict; aborting.")
        sys.exit(1)
    print(f"  keys  : {len(sb)}")
    sample_keys = list(sb.keys())[:3]
    print(f"  first 3 keys: {sample_keys}")
    for k in sample_keys:
        entry = sb[k]
        print(f"    [{k}]")
        print(f"      entry type: {type(entry).__name__}")
        try:
            starts, ends = get_shell_bounds(entry)
            print(f"      shell_starts: {starts}")
            print(f"      shell_ends  : {ends}")
            print(f"      n_shells    : {len(starts)}")
        except Exception as e:
            print(f"      bound extraction FAIL: {e}")
            sys.exit(1)
    print()
    return sb


def cross_check_coverage(sb: dict):
    print("Coverage: sample_names in shell_boundaries.pkl")
    sb_keys = set(sb.keys())
    for split, path in PT_PATHS.items():
        if not os.path.exists(path):
            print(f"  {split:7s}  pt FILE MISSING — {path}")
            continue
        preds = torch.load(path, map_location="cpu", weights_only=False)
        sn = set(preds["sample_name"])
        inter = sn & sb_keys
        missing = sn - sb_keys
        print(f"  {split:7s}  {len(inter):5d}/{len(sn):5d} in pkl  "
              f"(missing: {len(missing)})")
        if missing and len(missing) <= 5:
            print(f"           missing examples: {sorted(missing)[:5]}")
        del preds


# ─── Eval loop ───────────────────────────────────────────────────────────────
COL_ORDER = [
    "sample_name", "mp_id",
    "gate_min_dist", "gate_pass", "has_shell_bounds",
    "shell_1_start", "shell_1_end", "shell_2_start", "shell_2_end",
    "s1_CN", "s1_d", "s1_elem", "s2_CN", "s2_d", "s2_elem", "composite",
    "pred_s1_CN", "true_s1_CN", "pred_s1_d", "true_s1_d",
    "pred_s1_elem", "true_s1_elem",
    "pred_s2_CN", "true_s2_CN", "pred_s2_d", "true_s2_d",
    "pred_s2_elem", "true_s2_elem",
]


def evaluate_split(split: str, sb: dict) -> pd.DataFrame:
    print(f"\n[{split}] processing ...")
    t0 = time.time()
    preds = torch.load(PT_PATHS[split], map_location="cpu", weights_only=False)
    N = len(preds["sample_name"])

    rows = []
    n_no_bounds = 0
    n_passed_gate = 0

    for i in range(N):
        sn = preds["sample_name"][i]
        pf = _to_np(preds["pred_frac_coords"][i])
        pt = _to_np(preds["pred_atom_types"][i])
        tf = _to_np(preds["true_frac_coords"][i])
        tt = _to_np(preds["true_atom_types"][i])

        min_d, gate_pass = compute_gate(pf, L)
        if gate_pass:
            n_passed_gate += 1

        row = {
            "sample_name":    sn,
            "mp_id":          preds["mp_id"][i] if "mp_id" in preds else "",
            "gate_min_dist":  min_d,
            "gate_pass":      gate_pass,
            "has_shell_bounds": sn in sb,
        }

        if sn not in sb:
            n_no_bounds += 1
            for k in SUB_KEYS + ["composite"]:
                row[k] = np.nan
            rows.append(row)
            continue

        starts, ends = get_shell_bounds(sb[sn])
        row["shell_1_start"] = starts[0] if len(starts) > 0 else np.nan
        row["shell_1_end"]   = ends[0]   if len(ends)   > 0 else np.nan
        row["shell_2_start"] = starts[1] if len(starts) > 1 else np.nan
        row["shell_2_end"]   = ends[1]   if len(ends)   > 1 else np.nan

        if not gate_pass:
            for k in SUB_KEYS + ["composite"]:
                row[k] = np.nan
        else:
            scores = score_sample(pf, pt, tf, tt, starts, ends, L=L)
            row.update(scores)
        rows.append(row)

    dt = time.time() - t0
    df = pd.DataFrame(rows).reindex(columns=COL_ORDER)

    # split-level summary
    pass_rate = n_passed_gate / N if N > 0 else 0.0
    df_scored = df[df["gate_pass"] & df["has_shell_bounds"] &
                   df["composite"].notna()]
    n_scored = len(df_scored)

    print(f"  N total            : {N}")
    print(f"  gate pass          : {n_passed_gate}/{N} = {pass_rate*100:.2f}%")
    print(f"  missing shell pkl  : {n_no_bounds}")
    print(f"  fully scored       : {n_scored}")
    if n_scored > 0:
        c_pass = df_scored["composite"].mean()
        c_all0 = df["composite"].fillna(0).sum() / N
        print(f"  composite mean (scored only)         : {c_pass:.4f}")
        print(f"  composite mean (all, gate-fail = 0)  : {c_all0:.4f}")
        print(f"  sub-scores (over scored samples):")
        for k in SUB_KEYS:
            v = df_scored[k].mean()
            print(f"    {k:8s} = {v:.4f}   (weight={WEIGHTS[k]:.2f})")
    print(f"  elapsed            : {dt:.1f}s")

    out_csv = os.path.join(OUT_DIR, f"composite_per_sample_{split}.csv")
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  CSV saved          : {out_csv}")
    return df


def print_summary_table(dfs: dict):
    print("\n" + "=" * 78)
    print("Summary: composite re-scoring across 3 splits")
    print("=" * 78)
    header = ["split", "N", "gate%", "comp(pass)", "comp(all)"] + SUB_KEYS
    rows = []
    for split, df in dfs.items():
        N = len(df)
        n_pass = int(df["gate_pass"].sum())
        df_s = df[df["gate_pass"] & df["has_shell_bounds"] &
                  df["composite"].notna()]
        row = [split, str(N), f"{n_pass/N*100:.2f}"]
        if len(df_s) > 0:
            row.append(f"{df_s['composite'].mean():.4f}")
            row.append(f"{df['composite'].fillna(0).sum()/N:.4f}")
            for k in SUB_KEYS:
                row.append(f"{df_s[k].mean():.4f}")
        else:
            row.extend(["n/a"] * (2 + len(SUB_KEYS)))
        rows.append(row)

    widths = [max(len(h), max(len(r[i]) for r in rows)) + 2
              for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * (w - 2) for w in widths]))
    for r in rows:
        print(fmt.format(*r))

    summary_df = pd.DataFrame(rows, columns=header)
    out_sum = os.path.join(OUT_DIR, "composite_summary.csv")
    summary_df.to_csv(out_sum, index=False)
    print(f"\nSummary CSV saved : {out_sum}")


def print_formula_header():
    print("\nScoring formula (printed for transparency):")
    print("  Gate         : min pairwise cartesian distance over 20 atoms ≥ 1.5 Å")
    print("                 (raw cartesian, NO PBC / min-image)")
    print("  CN  / d sub  : score = max(0, 1 - |delta| / tolerance)   (linear decay)")
    print("  elem sub     : multiset Jaccard |P ∩ T| / |P ∪ T|")
    print("  CNO equiv    : Z ∈ {6, 7, 8} → token 'CNO'; other Z → element symbol")
    print(f"  Weights      : {WEIGHTS}")
    print(f"  Tolerances   : {TOL}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["val", "test", "holdout"],
                    choices=["val", "test", "holdout"])
    ap.add_argument("--probe-only", action="store_true",
                    help="Probe shell_boundaries.pkl schema + coverage and exit.")
    args = ap.parse_args()

    print("=" * 78)
    print("DiffCSP-Exp4 Step 6 — Composite Re-Scoring")
    print("=" * 78)
    print_formula_header()
    print()

    sb = probe_shell_pkl()
    cross_check_coverage(sb)

    if args.probe_only:
        print("\n--probe-only set; exiting before evaluation.")
        return

    t_global = time.time()
    dfs = {}
    for split in args.splits:
        dfs[split] = evaluate_split(split, sb)

    print_summary_table(dfs)

    print("\n" + "=" * 78)
    print(f"Done in {time.time()-t_global:.1f}s. Outputs in: {OUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
