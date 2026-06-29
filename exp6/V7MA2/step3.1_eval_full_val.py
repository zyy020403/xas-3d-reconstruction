cat > /home/tcat/experiment6_v7/step3/step3.1_eval_full_val.py << 'SCRIPT'
"""
Exp6 v7 — Step 3.1: Full val evaluation with best ckpt
Reports 7 metrics as required by MA1 handoff §4 Task 1:
  1. hungarian_rmsd
  2. set_level_type_acc
  3. multiset_f1_macro
  4. pred_in_cutoff / true_in_cutoff
  5. close_pair_type_acc
  6. val_cps_mean  (full val, NOT 200-sample subset)
  7. pairwise_violation_rate

Also reports CPS bypass-PV sub-scores:
  C1, D1, T1, C2, D2, T2, outside_shells_ratio

Run:
  cd /home/tcat/experiment6_v7
  /home/tcat/conda_envs/mlff/bin/python step3/step3.1_eval_full_val.py
"""
import sys, json, time
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

BASE = Path("/home/tcat/experiment6_v7")
sys.path.insert(0, str(BASE / "shared"))

import composite_score as cs
import eval_metrics as em
from detr_xas import build_detr_xas
from exp6_data_adapter import adapt, reset_first_batch_flag

# ── Constants ──────────────────────────────────────────────────────────────
with open(BASE / "shared" / "exp6_element_vocab.json") as f:
    _V = json.load(f)
N_CENTER_TYPES   = int(_V["center"]["N_TYPES"])
N_NEIGHBOR_TYPES = int(_V["neighbor"]["N_TYPES"])
NO_OBJECT_IDX    = int(_V["neighbor"]["no_object_idx"])
IDX_TO_Z = {int(k): int(v) for k, v in _V["neighbor"]["idx_to_Z"].items()}

cs.init_constants()
cs.set_no_object_idx(NO_OBJECT_IDX)
em.set_no_object_idx(NO_OBJECT_IDX)

L = 20.0
LENGTHS = torch.tensor([L, L, L])

BEST_CKPT = BASE / "checkpoints" / "epochepoch=168-rmsdval_rmsd=1.7258.ckpt"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

print("=" * 70)
print("Exp6 v7 FULL VAL EVALUATION — step3.1")
print(f"  ckpt  : {BEST_CKPT}")
print(f"  device: {DEVICE}")
print("=" * 70)

# ── Load model ────────────────────────────────────────────────────────────
print("\n[1/4] Loading model ...")
model = build_detr_xas(
    n_neighbor_types=N_NEIGHBOR_TYPES,
    n_center_types=N_CENTER_TYPES,
    no_object_idx=NO_OBJECT_IDX,
    d_model=256, nhead=8,
    num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1, n_queries=20,
)
state = torch.load(BEST_CKPT, map_location="cpu")
# Lightning checkpoint: weights under "state_dict"
sd = state.get("state_dict", state)
# strip "model." prefix if present
sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"  WARNING: missing keys: {missing[:5]}")
if unexpected:
    print(f"  WARNING: unexpected keys: {unexpected[:5]}")
model.to(DEVICE)
model.eval()
print("  model loaded")

# ── Load val dataloader ────────────────────────────────────────────────────
print("\n[2/4] Loading val dataloader ...")
sys.path.insert(0, str(BASE / "shared"))
from xas_local_datamodule_v2 import XasLocalDataModuleV2

dm = XasLocalDataModuleV2(
    data_dir=str(BASE / "data"),
    batch_size=BATCH_SIZE,
    num_workers=4,
)
dm.setup("fit")
val_loader = dm.val_dataloader()
n_batches  = len(val_loader)
print(f"  val batches: {n_batches}")

# ── Inference loop ────────────────────────────────────────────────────────
print("\n[3/4] Running inference ...")

# Accumulators for metrics 1-5
all_rmsd              = []
all_set_level_acc     = []
all_multiset_pred     = []   # list of per-sample valid pred type tensors
all_multiset_gt       = []   # list of per-sample gt type tensors
all_pred_in_cutoff    = []
all_true_in_cutoff    = []
all_close_pair_acc    = []

# Accumulators for metric 6+7 (CPS)
all_cps               = []
all_cps_bypass        = []
all_pv_pass           = []
all_outside_ratio     = []

# CPS sub-scores
all_C1 = []; all_D1 = []; all_T1 = []
all_C2 = []; all_D2 = []; all_T2 = []

reset_first_batch_flag()
t0 = time.time()

with torch.no_grad():
    for batch_idx, raw_batch in enumerate(tqdm(val_loader, desc="eval")):
        model_batch, targets = adapt(raw_batch, model)
        if model_batch is None:
            continue

        # Move to device
        for k in model_batch:
            if isinstance(model_batch[k], torch.Tensor):
                model_batch[k] = model_batch[k].to(DEVICE)

        outputs = model(model_batch)
        pred_logits = outputs["pred_logits"].cpu()  # (B, 20, C+1)
        pred_pos    = outputs["pred_pos"].cpu()     # (B, 20, 3)
        argmax      = pred_logits.argmax(-1)        # (B, 20)

        B = pred_logits.shape[0]

        for bi in range(B):
            tgt = targets[bi]
            gt_pos    = tgt["boxes"]               # (N, 3) frac
            gt_types  = tgt["labels"]              # (N,) int
            lengths_i = tgt.get("lengths", LENGTHS)
            eval_cutoff = float(tgt.get("eval_cutoff", 10.0))
            sample_name = tgt["sample_name"]

            plogits_i = pred_logits[bi]   # (20, C+1)
            ppos_i    = pred_pos[bi]      # (20, 3)
            argmax_i  = argmax[bi]        # (20,)

            # ── metric 1: Hungarian RMSD ──────────────────────────────────
            rmsd_i = em.hungarian_rmsd(ppos_i, argmax_i, gt_pos, gt_types, lengths_i)
            all_rmsd.append(rmsd_i)

            # ── metric 2: Set-Level TypeAcc ───────────────────────────────
            sla_i = em.set_level_type_acc(argmax_i, gt_types)
            all_set_level_acc.append(sla_i)

            # ── metric 3: Multiset F1 (accumulate per-sample lists) ───────
            valid_pred = argmax_i[argmax_i != NO_OBJECT_IDX]
            all_multiset_pred.append(valid_pred)
            all_multiset_gt.append(gt_types)

            # ── metric 4: pred_in_cutoff / true_in_cutoff ─────────────────
            n_pred_in, n_true_in = em.in_cutoff_counts(
                ppos_i, argmax_i, gt_pos, eval_cutoff, lengths_i
            )
            all_pred_in_cutoff.append(n_pred_in)
            all_true_in_cutoff.append(n_true_in)

            # ── metric 5: close_pair_type_acc ─────────────────────────────
            cpa_i = em.close_pair_type_acc(
                ppos_i, argmax_i, gt_pos, gt_types, lengths_i
            )
            all_close_pair_acc.append(cpa_i)

            # ── metric 6+7: CPS (with PV gate) ───────────────────────────
            try:
                cps_full, bd = cs.composite_physical_score(
                    ppos_i, argmax_i, sample_name, lengths_i, IDX_TO_Z,
                )
                cps_bypass, bd2 = cs.composite_physical_score(
                    ppos_i, argmax_i, sample_name, lengths_i, IDX_TO_Z,
                    bypass_pv=True,
                )
                pv_pass = bool(bd["PV"])

                # outside_shells_ratio: fraction of valid pred atoms
                # that fall outside both GT shell 1 and shell 2
                n_valid = int((argmax_i != NO_OBJECT_IDX).sum().item())
                in_shell = bd2.get("n_pred_in_any_shell", None)
                if in_shell is not None and n_valid > 0:
                    outside_ratio = 1.0 - in_shell / n_valid
                else:
                    outside_ratio = float("nan")

                all_cps.append(cps_full)
                all_cps_bypass.append(cps_bypass)
                all_pv_pass.append(float(pv_pass))
                all_outside_ratio.append(outside_ratio)
                all_C1.append(bd2.get("C1", 0.0))
                all_D1.append(bd2.get("D1", 0.0))
                all_T1.append(bd2.get("T1", 0.0))
                all_C2.append(bd2.get("C2", 0.0))
                all_D2.append(bd2.get("D2", 0.0))
                all_T2.append(bd2.get("T2", 0.0))
            except Exception as e:
                # CPS failed for this sample — still record nan, don't skip
                all_cps.append(float("nan"))
                all_cps_bypass.append(float("nan"))
                all_pv_pass.append(0.0)
                all_outside_ratio.append(float("nan"))
                for lst in [all_C1, all_D1, all_T1, all_C2, all_D2, all_T2]:
                    lst.append(float("nan"))

elapsed = time.time() - t0
n_samples = len(all_rmsd)
print(f"  done: {n_samples} samples in {elapsed:.1f}s")

# ── Compute final metrics ─────────────────────────────────────────────────
print("\n[4/4] Computing final metrics ...")

def nanmean(lst):
    arr = np.array([x for x in lst if not (isinstance(x, float) and np.isnan(x))])
    return float(arr.mean()) if len(arr) > 0 else float("nan")

# metric 1
hungarian_rmsd_mean = nanmean(all_rmsd)

# metric 2
set_level_acc_mean  = nanmean(all_set_level_acc)

# metric 3: dataset-level macro F1
multiset_f1 = em.multiset_f1_macro(all_multiset_pred, all_multiset_gt, N_NEIGHBOR_TYPES)

# metric 4
pred_in_mean = nanmean(all_pred_in_cutoff)
true_in_mean = nanmean(all_true_in_cutoff)

# metric 5
close_pair_acc_mean = nanmean(all_close_pair_acc)
n_valid_close = sum(1 for x in all_close_pair_acc if x > 0 or x == 0)

# metric 6
cps_mean_with_pv    = nanmean(all_cps)
cps_mean_bypass_pv  = nanmean(all_cps_bypass)

# metric 7
pv_pass_rate = float(np.mean(all_pv_pass))

# sub-scores
outside_ratio_mean  = nanmean(all_outside_ratio)
C1_mean = nanmean(all_C1); D1_mean = nanmean(all_D1); T1_mean = nanmean(all_T1)
C2_mean = nanmean(all_C2); D2_mean = nanmean(all_D2); T2_mean = nanmean(all_T2)

# ── Print results ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FULL VAL EVALUATION RESULTS  (Exp6 v7, best ckpt epoch=168)")
print("=" * 70)
print(f"\n--- 7 Required Metrics ---")
print(f"  1. hungarian_rmsd           = {hungarian_rmsd_mean:.4f} Å")
print(f"  2. set_level_type_acc       = {set_level_acc_mean:.4f}")
print(f"  3. multiset_f1_macro        = {multiset_f1:.4f}")
print(f"  4. pred_in_cutoff (mean)    = {pred_in_mean:.2f}")
print(f"     true_in_cutoff (mean)    = {true_in_mean:.2f}")
print(f"  5. close_pair_type_acc      = {close_pair_acc_mean:.4f}")
print(f"  6. val_cps_mean (with PV)   = {cps_mean_with_pv:.4f}")
print(f"     val_cps_bypass_pv        = {cps_mean_bypass_pv:.4f}")
print(f"  7. pairwise_violation_rate  = {1.0 - pv_pass_rate:.4f}  "
      f"(pv_pass_rate = {pv_pass_rate:.4f})")

print(f"\n--- CPS Sub-scores (bypass PV) ---")
print(f"  C1 = {C1_mean:.4f}  (weight 0.25, Exp4 baseline: 0.6023)")
print(f"  D1 = {D1_mean:.4f}  (weight 0.20, Exp4 baseline: 0.7406)")
print(f"  T1 = {T1_mean:.4f}  (weight 0.17, Exp4 baseline: 0.0898)")
print(f"  C2 = {C2_mean:.4f}  (weight 0.15, Exp4 baseline: 0.7444)")
print(f"  D2 = {D2_mean:.4f}  (weight 0.13, Exp4 baseline: 0.8154)")
print(f"  T2 = {T2_mean:.4f}  (weight 0.10, Exp4 baseline: 0.0916)")

print(f"\n--- Physical Sanity ---")
print(f"  outside_shells_ratio        = {outside_ratio_mean:.4f}  "
      f"(Exp4 baseline: 0.8069, pass threshold: < 0.857)")

print(f"\n--- Baselines (from baseline_cps.json) ---")
print(f"  Exp4 bypass-PV CPS          = 0.5408")
print(f"  Exp4 with-PV CPS            = 0.0092")
print(f"  Exp4 pv_pass_rate           = 0.0217")
print(f"  val_cps_pass threshold      = max(0.5408, Exp5'_TBD) + 0.05")
print(f"  val_rmsd_pass threshold     = < 1.4866 Å  (Exp4 holdout)")
print(f"  pv_pass threshold           = ≥ 95%")

print(f"\n--- Sample Count ---")
print(f"  n_samples evaluated         = {n_samples}")

print("\n" + "=" * 70)
print("VERDICT CHECK")
print("=" * 70)
rmsd_pass  = hungarian_rmsd_mean < 1.4866
pv_pass_ok = pv_pass_rate >= 0.95
cps_pass   = cps_mean_bypass_pv >= 0.6381  # max(0.5408, Exp5'_TBD=0.5881) + 0.05

print(f"  val_rmsd < 1.4866 Å ?       {'PASS' if rmsd_pass else 'FAIL'}  "
      f"({hungarian_rmsd_mean:.4f})")
print(f"  pv_pass_rate ≥ 95% ?         {'PASS' if pv_pass_ok else 'FAIL'}  "
      f"({pv_pass_rate:.4f})")
print(f"  val_cps_bypass ≥ 0.6381 ?   {'PASS' if cps_pass else 'FAIL'}  "
      f"({cps_mean_bypass_pv:.4f})")

if rmsd_pass and pv_pass_ok and cps_pass:
    verdict = "PASS (§10.1)"
elif cps_mean_bypass_pv >= 0.5881:
    verdict = "PARTIAL SUCCESS (§10.2)"
else:
    verdict = "FAIL (§10.3)"
print(f"\n  PRELIMINARY VERDICT: {verdict}")
print("  (final verdict requires user physical sanity §10.1)")

# ── Save results ──────────────────────────────────────────────────────────
results = {
    "ckpt": str(BEST_CKPT),
    "n_samples": n_samples,
    "hungarian_rmsd": hungarian_rmsd_mean,
    "set_level_type_acc": set_level_acc_mean,
    "multiset_f1_macro": multiset_f1,
    "pred_in_cutoff_mean": pred_in_mean,
    "true_in_cutoff_mean": true_in_mean,
    "close_pair_type_acc": close_pair_acc_mean,
    "val_cps_with_pv": cps_mean_with_pv,
    "val_cps_bypass_pv": cps_mean_bypass_pv,
    "pv_pass_rate": pv_pass_rate,
    "pairwise_violation_rate": 1.0 - pv_pass_rate,
    "outside_shells_ratio": outside_ratio_mean,
    "C1_mean": C1_mean, "D1_mean": D1_mean, "T1_mean": T1_mean,
    "C2_mean": C2_mean, "D2_mean": D2_mean, "T2_mean": T2_mean,
}

out_path = BASE / "step3" / "eval_full_val_results.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  results saved → {out_path}")
SCRIPT