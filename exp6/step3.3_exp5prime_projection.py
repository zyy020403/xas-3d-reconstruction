"""
Exp5' predictions_val.pt 后处理投影评估
对比：投影前 vs 投影后 的 PV pass rate 和 CPS
"""
import sys, json, time
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

BASE = Path("/home/tcat/experiment6_v7")
sys.path.insert(0, str(BASE / "shared"))

import composite_score as cs
import eval_metrics as em

with open(BASE / "shared" / "exp6_element_vocab.json") as f:
    _V = json.load(f)
N_NEIGHBOR_TYPES = int(_V["neighbor"]["N_TYPES"])
NO_OBJECT_IDX    = int(_V["neighbor"]["no_object_idx"])
IDX_TO_Z = {int(k): int(v) for k, v in _V["neighbor"]["idx_to_Z"].items()}

# Exp5' 用 Z 直接作为 atom type，不是 dense idx，需要建反向映射
# pred_atom_types 存的是 Z 值还是 dense idx？先检查
cs.init_constants()
cs.set_no_object_idx(NO_OBJECT_IDX)
em.set_no_object_idx(NO_OBJECT_IDX)

MIN_PDIST = cs.MIN_PDIST
L         = 20.0
LENGTHS   = torch.tensor([L, L, L])
MAX_ITER  = 200
STEP_SIZE = 0.05

PRED_PATH = Path("/home/tcat/diffcsp_exp5_prime/predictions/predictions_val.pt")

print("=" * 70)
print("Exp5' POST-HOC PROJECTION EVAL — step3.3")
print(f"  MIN_PDIST={MIN_PDIST:.4f} Å  MAX_ITER={MAX_ITER}  STEP={STEP_SIZE} Å")
print("=" * 70)

# ── Load predictions ──────────────────────────────────────────────────────
d = torch.load(PRED_PATH, map_location="cpu")
sample_names    = d["sample_name"]
pred_coords_all = d["pred_frac_coords"]   # list of (20, 3)
pred_types_all  = d["pred_atom_types"]    # list of (20,)
n = len(sample_names)
print(f"  loaded {n} samples")

# Check what pred_atom_types contains — Z or dense idx?
# Exp5' diffusion outputs Z directly (not dense neighbor idx)
# CPS composite_physical_score expects dense neighbor idx
# Build Z->dense_idx map
Z_to_idx = {int(v): int(k) for k, v in _V["neighbor"]["idx_to_Z"].items()}

# Check sample
sample_t = pred_types_all[0]
print(f"  pred_atom_types[0] sample values: {sample_t[:5].tolist()}")
print(f"  (if > 89 or looks like Z, need Z->idx conversion)")

# Determine if conversion needed
max_type_val = max(t.max().item() for t in pred_types_all[:20])
needs_z_conversion = max_type_val > N_NEIGHBOR_TYPES
print(f"  max type value in first 20 samples: {max_type_val}")
print(f"  needs Z->dense_idx conversion: {needs_z_conversion}")

# ── Projection function (same as step3.2) ─────────────────────────────────
def project_repulsion(pos, lengths, min_pdist=MIN_PDIST,
                      max_iter=MAX_ITER, step=STEP_SIZE):
    pos = pos.clone().float()
    N = pos.shape[0]
    if N <= 1:
        return pos, 0
    step_frac = step / L
    for it in range(max_iter):
        diff_frac = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N,N,3)
        diff_frac = diff_frac - diff_frac.round()
        diff_cart = diff_frac * lengths
        dist = diff_cart.norm(dim=-1)                      # (N,N)
        eye  = torch.eye(N, dtype=torch.bool)
        dist[eye] = 1e9
        if dist.min().item() >= min_pdist:
            return pos, it
        rows, cols = (dist < min_pdist).nonzero(as_tuple=True)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r >= c:
                continue
            d_rc = dist[r, c].item()
            if d_rc < 1e-6:
                direction = torch.randn(3)
            else:
                direction = diff_frac[r, c]
            direction = direction / (direction.norm() + 1e-9)
            push = step_frac * direction
            pos[r] = (pos[r] + push).clamp(-0.5, 0.5)
            pos[c] = (pos[c] - push).clamp(-0.5, 0.5)
    return pos, max_iter

# ── Main loop ─────────────────────────────────────────────────────────────
print("\nRunning projection ...")
results_raw  = []
results_proj = []
iter_counts  = []
t0 = time.time()

for i in tqdm(range(n)):
    sname   = sample_names[i]
    ppos    = pred_coords_all[i]    # (20, 3)
    ptypes  = pred_types_all[i]     # (20,)

    # Convert Z->dense_idx if needed
    if needs_z_conversion:
        ptypes_idx = torch.tensor(
            [Z_to_idx.get(int(z), NO_OBJECT_IDX) for z in ptypes.tolist()],
            dtype=torch.long)
    else:
        ptypes_idx = ptypes.long()

    # RAW
    try:
        cps_r, bd_r = cs.composite_physical_score(
            ppos, ptypes_idx, sname, LENGTHS, IDX_TO_Z)
        results_raw.append((cps_r, bd_r["PV"]))
    except Exception as e:
        results_raw.append((0.0, False))

    # PROJECT (all 20 atoms — Exp5' has no no_object concept)
    ppos_proj, n_it = project_repulsion(ppos, LENGTHS)
    iter_counts.append(n_it)

    # PROJECTED
    try:
        cps_p, bd_p = cs.composite_physical_score(
            ppos_proj, ptypes_idx, sname, LENGTHS, IDX_TO_Z)
        results_proj.append((cps_p, bd_p["PV"]))
    except Exception:
        results_proj.append((0.0, False))

elapsed = time.time() - t0
print(f"  done in {elapsed:.1f}s")

# ── Results ───────────────────────────────────────────────────────────────
cps_raw_m  = np.mean([x[0] for x in results_raw])
pv_raw     = np.mean([float(x[1]) for x in results_raw])
cps_proj_m = np.mean([x[0] for x in results_proj])
pv_proj    = np.mean([float(x[1]) for x in results_proj])
iter_mean  = np.mean(iter_counts)
iter_maxed = np.mean([float(x == MAX_ITER) for x in iter_counts])

print("\n" + "=" * 70)
print(f"Exp5' PROJECTION RESULTS  (n={n})")
print("=" * 70)
print(f"{'':35s}  {'RAW':>8}  {'PROJECTED':>10}  {'DELTA':>8}")
print(f"{'pv_pass_rate':35s}  {pv_raw:8.4f}  {pv_proj:10.4f}  {pv_proj-pv_raw:+8.4f}")
print(f"{'val_cps_mean':35s}  {cps_raw_m:8.4f}  {cps_proj_m:10.4f}  {cps_proj_m-cps_raw_m:+8.4f}")
print(f"\n  avg iters: {iter_mean:.1f}/{MAX_ITER}  "
      f"maxed_frac: {iter_maxed:.3f} "
      f"({'WARNING' if iter_maxed > 0.3 else 'OK'})")

print("\n  Reference (baseline_cps.json):")
print(f"  Exp4 pv_pass_rate  = 0.0217  bypass-PV CPS = 0.5408")
print(f"  Exp5' pv_pass_rate = 0.4510  bypass-PV CPS = 0.5881  (from MA1 handoff)")

out = {
    "n_samples": n, "max_iter": MAX_ITER, "step_size": STEP_SIZE,
    "min_pdist": MIN_PDIST,
    "raw":  {"pv_pass": pv_raw,  "cps": cps_raw_m},
    "proj": {"pv_pass": pv_proj, "cps": cps_proj_m},
    "iter_mean": iter_mean, "iter_maxed_frac": iter_maxed,
}
out_path = BASE / "step3" / "eval_exp5prime_projection_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  saved → {out_path}")