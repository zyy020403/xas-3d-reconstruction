"""
Exp6 Step 1.1 — Recompute Exp4 Set-Level TypeAcc baseline.

Why:
  Proposal §10.1 v3 requires this scalar to backfill the §10.1 acceptance
  threshold. Exp4 final report's 0.197 is position-by-position TypeAcc, which
  is a fake metric (Exp3 lesson, see EXP4_FINAL_REPORT_ERRATA_2.md §2). We
  recompute the *true* Set-Level baseline so Exp6 has a meaningful comparison.

Source:
  /home/tcat/diffcsp_exp4/code/step5/predictions_val.pt
  Schema (Step 0.5 dump): dict with 14 keys, including:
    - pred_atom_types: list[7621] of (20,) int64 Z values
    - true_atom_types: list[7621] of (20,) int64 Z values
  Both are in Z-space (atomic numbers), NOT dense vocab indices.

Output:
  Scalar mean Set-Level TypeAcc reported to stdout, copy-paste into
  EXP6_PHASE1_OUTPUT.md as proposal §10.1 backfill.

KEY DECISION (handoff §7 grey-zone, recorded for MA1 review):
  handoff §3.2 said:
      "Exp4 has no no_object concept. valid_pred = pred_types_argmax (no filter)"
  ε schema dump revealed this is INACCURATE — Exp4 ground truth uses Z=0 as
  padding for samples with < 20 real neighbors. Including Z=0 in both
  pred_counter and gt_counter would create false positive intersection hits
  (padding-padding "matches"), inflating the baseline.

  SA1 decision: filter Z=0 on BOTH sides before computing Set-Level. This is
  the operationally correct interpretation. Proposal §7.1 indicator 2 formula
  structure (multiset intersection / max(|pred|,|gt|)) is unchanged — only
  the no_object filter target changes from NO_OBJECT_IDX to Z=0 padding for
  this Exp4-context recomputation.

  Documented in EXP6_PHASE1_OUTPUT.md "implementation choices" + flagged
  for MA1 acknowledgment.
"""
import torch
from collections import Counter

PRED_PATH = '/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt'
PADDING_Z = 0


def set_level_typeacc_z_space(pred_z, gt_z, padding=PADDING_Z):
    """
    Set-Level TypeAcc operating directly in Z-space (no vocab mapping needed).

    Implements proposal §7.1 indicator 2 with Z=padding filtered on both sides.
    Inlined here (not via eval_metrics.py) because eval_metrics expects dense
    vocab indices + NO_OBJECT_IDX, while Exp4 predictions are raw Z + Z=0
    padding. Different filter target, same formula structure.
    """
    pred_filtered = pred_z[pred_z != padding].tolist()
    gt_filtered = gt_z[gt_z != padding].tolist()

    pc = Counter(pred_filtered)
    gc = Counter(gt_filtered)

    intersection = sum((pc & gc).values())
    denominator = max(len(pred_filtered), len(gt_filtered))
    if denominator == 0:
        return 0.0
    return intersection / denominator


def main():
    print("=" * 70)
    print("Exp6 Step 1.1: recompute Exp4 Set-Level TypeAcc baseline (val)")
    print("=" * 70)

    print(f"Loading {PRED_PATH}...")
    d = torch.load(PRED_PATH, weights_only=False)

    pred_list = d['pred_atom_types']
    true_list = d['true_atom_types']
    n = len(pred_list)
    assert n == len(true_list), f"length mismatch: pred={n}, true={len(true_list)}"
    print(f"n_samples = {n}")
    print()

    # Sanity peek at first 2 samples
    for i in range(2):
        print(f"sample {i}:")
        print(f"  pred Z (raw):  {pred_list[i].tolist()}")
        print(f"  true Z (raw):  {true_list[i].tolist()}")
    print()

    # Padding stats (justifies Z=0 filter decision)
    n_pred_pad = sum((p == PADDING_Z).sum().item() for p in pred_list)
    n_true_pad = sum((g == PADDING_Z).sum().item() for g in true_list)
    n_total = n * 20
    print(f"Z=0 padding stats:")
    print(f"  pred: {n_pred_pad}/{n_total} ({100*n_pred_pad/n_total:.2f}%)")
    print(f"  true: {n_true_pad}/{n_total} ({100*n_true_pad/n_total:.2f}%)")
    print()

    # Compute per-sample Set-Level
    print("Computing per-sample Set-Level TypeAcc (Z-space, padding filtered)...")
    accs = []
    for i in range(n):
        acc = set_level_typeacc_z_space(pred_list[i], true_list[i])
        accs.append(acc)

    accs_t = torch.tensor(accs, dtype=torch.float32)
    mean = accs_t.mean().item()
    std = accs_t.std().item()
    median = accs_t.median().item()
    p25 = torch.quantile(accs_t, 0.25).item()
    p75 = torch.quantile(accs_t, 0.75).item()

    print()
    print("=" * 70)
    print("RESULT (proposal §10.1 backfill):")
    print(f"  exp4_setlevel_typeacc_val_mean   = {mean:.4f}")
    print(f"  exp4_setlevel_typeacc_val_median = {median:.4f}")
    print(f"  exp4_setlevel_typeacc_val_std    = {std:.4f}")
    print(f"  exp4_setlevel_typeacc_val_p25    = {p25:.4f}")
    print(f"  exp4_setlevel_typeacc_val_p75    = {p75:.4f}")
    print(f"  n_samples                         = {n}")
    print("=" * 70)
    print()

    # Sanity: compare against a no-padding-filter version to quantify the
    # effect of the SA1 filter decision.
    print("Sanity (no padding filter, would be inflated baseline):")
    accs_nofilter = []
    for i in range(n):
        pred_list_full = pred_list[i].tolist()
        true_list_full = true_list[i].tolist()
        pc = Counter(pred_list_full)
        gc = Counter(true_list_full)
        inter = sum((pc & gc).values())
        denom = max(len(pred_list_full), len(true_list_full))
        accs_nofilter.append(inter / denom if denom > 0 else 0.0)
    nf_mean = torch.tensor(accs_nofilter).mean().item()
    print(f"  no_filter_mean = {nf_mean:.4f}  (would be inflated by Z=0 ∩ Z=0 hits)")
    print(f"  filtered_mean  = {mean:.4f}  (the correct baseline)")
    print(f"  Δ (inflation prevented) = {nf_mean - mean:+.4f}")
    print()
    print("Step 1.1 DONE. Copy mean to EXP6_PHASE1_OUTPUT.md.")


if __name__ == "__main__":
    main()
