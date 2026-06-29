"""
Exp6 v7 — Step 3.2: Full val evaluation WITH post-hoc repulsion projection
对每个样本的预测坐标做迭代推开，直到所有 pred-pred 对距离 >= MIN_PDIST，
然后重新计算 CPS / PV pass rate，与 step3.1 对比。

目的：判断 PV 是否是唯一瓶颈——如果投影后 CPS 大幅提升，说明模型学到了
结构信息，只是缺乏物理可行性保证；如果投影后 CPS 仍低，说明结构本身没学到。
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
from detr_xas import build_detr_xas
from exp6_data_adapter import adapt, reset_first_batch_flag

with open(BASE / "shared" / "exp6_element_vocab.json") as f:
    _V = json.load(f)
N_CENTER_TYPES   = int(_V["center"]["N_TYPES"])
N_NEIGHBOR_TYPES = int(_V["neighbor"]["N_TYPES"])
NO_OBJECT_IDX    = int(_V["neighbor"]["no_object_idx"])
IDX_TO_Z = {int(k): int(v) for k, v in _V["neighbor"]["idx_to_Z"].items()}

cs.init_constants()
cs.set_no_object_idx(NO_OBJECT_IDX)
em.set_no_object_idx(NO_OBJECT_IDX)

L         = 20.0
LENGTHS   = torch.tensor([L, L, L])
MIN_PDIST = cs.MIN_PDIST          # 1.5076 Å
MAX_ITER  = 100                   # 最大推开迭代次数
STEP_SIZE = 0.05                  # 每次推开步长（Å，在 frac 单位 = 0.05/20）

BEST_CKPT  = BASE / "checkpoints" / "epochepoch=168-rmsdval_rmsd=1.7258.ckpt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

print("=" * 70)
print("Exp6 v7 EVAL WITH POST-HOC PROJECTION — step3.2")
print(f"  MIN_PDIST = {MIN_PDIST:.4f} Å  MAX_ITER = {MAX_ITER}  STEP = {STEP_SIZE} Å")
print("=" * 70)

# ── Post-hoc repulsion projection ─────────────────────────────────────────
def project_repulsion(pred_pos, lengths, min_pdist=MIN_PDIST,
                       max_iter=MAX_ITER, step=STEP_SIZE):
    """
    pred_pos: (N, 3) frac coords, in-place iterative repulsion.
    Returns projected pos (N, 3) and n_iters_used.
    """
    pos = pred_pos.clone().float()  # (N, 3)
    N = pos.shape[0]
    if N <= 1:
        return pos, 0

    step_frac = step / L  # convert Å step to frac units

    for it in range(max_iter):
        # cartesian positions
        cart = pos * lengths  # (N, 3)
        # pairwise diff (min-image)
        diff = cart.unsqueeze(0) - cart.unsqueeze(1)  # (N, N, 3)
        diff_frac = pos.unsqueeze(0) - pos.unsqueeze(1)
        diff_frac = diff_frac - diff_frac.round()     # min-image in frac
        diff_cart = diff_frac * lengths               # (N, N, 3)
        dist = diff_cart.norm(dim=-1)                 # (N, N)

        # mask: only lower triangle, exclude self
        eye = torch.eye(N, dtype=torch.bool)
        dist_masked = dist.clone()
        dist_masked[eye] = 1e9
        dist_masked = torch.tril(dist_masked) + torch.triu(
            torch.full_like(dist_masked, 1e9), 1)

        violations = (dist_masked < min_pdist) & ~eye
        if not violations.any():
            return pos, it

        # for each violating pair, push apart along connection vector
        rows, cols = violations.nonzero(as_tuple=True)
        for r, c in zip(rows.tolist(), cols.tolist()):
            d = dist[r, c].item()
            if d < 1e-6:
                # degenerate: push in random direction
                direction = torch.randn(3)
                direction = direction / direction.norm()
            else:
                direction = diff_frac[r, c] / (d / L + 1e-9)
                direction = direction / (direction.norm() + 1e-9)
            push = step_frac * direction
            pos[r] = pos[r] + push
            pos[c] = pos[c] - push
            # clamp to [-0.5, 0.5]
            pos[r] = pos[r].clamp(-0.5, 0.5)
            pos[c] = pos[c].clamp(-0.5, 0.5)

    return pos, max_iter

# ── Load model ────────────────────────────────────────────────────────────
print("\n[1/4] Loading model ...")
model = build_detr_xas(
    n_neighbor_types=N_NEIGHBOR_TYPES, n_center_types=N_CENTER_TYPES,
    no_object_idx=NO_OBJECT_IDX, d_model=256, nhead=8,
    num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1, n_queries=20,
)
sd = torch.load(BEST_CKPT, map_location="cpu")
sd = sd.get("state_dict", sd)
sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(DEVICE); model.eval()
print("  model loaded")

# ── Val dataloader ────────────────────────────────────────────────────────
print("\n[2/4] Loading val dataloader ...")
from xas_local_datamodule_v2 import XasLocalDataModuleV2
dm = XasLocalDataModuleV2(
    data_dir=str(BASE / "data"), batch_size=BATCH_SIZE, num_workers=4)
dm.setup("fit")
val_loader = dm.val_dataloader()
print(f"  val batches: {len(val_loader)}")

# ── Inference + projection ────────────────────────────────────────────────
print("\n[3/4] Inference + projection ...")

results_raw  = []   # (cps, pv_pass) before projection
results_proj = []   # (cps, pv_pass) after projection
iter_counts  = []
rmsd_raw = []; rmsd_proj = []

reset_first_batch_flag()
t0 = time.time()

with torch.no_grad():
    for raw_batch in tqdm(val_loader, desc="eval+proj"):
        model_batch, targets = adapt(raw_batch, model)
        if model_batch is None:
            continue
        for k in model_batch:
            if isinstance(model_batch[k], torch.Tensor):
                model_batch[k] = model_batch[k].to(DEVICE)

        outputs  = model(model_batch)
        pred_pos = outputs["pred_pos"].cpu()     # (B, 20, 3)
        argmax   = outputs["pred_logits"].cpu().argmax(-1)  # (B, 20)
        B = pred_pos.shape[0]

        for bi in range(B):
            tgt         = targets[bi]
            gt_pos      = tgt["pos"]
            gt_types    = tgt["labels"]
            lengths_i   = tgt.get("lengths", LENGTHS)
            sample_name = tgt["sample_name"]

            ppos_i  = pred_pos[bi]   # (20, 3)
            arg_i   = argmax[bi]     # (20,)

            # valid mask (non-no_object)
            valid   = arg_i != NO_OBJECT_IDX
            ppos_v  = ppos_i[valid]  # (n_valid, 3)
            types_v = arg_i[valid]

            # ── RAW metrics ───────────────────────────────────────────────
            rmsd_r, _ = em.hungarian_rmsd(ppos_i, arg_i, gt_pos, gt_types, lengths_i)
            rmsd_raw.append(rmsd_r)
            try:
                cps_r, bd_r = cs.composite_physical_score(
                    ppos_i, arg_i, sample_name, lengths_i, IDX_TO_Z)
                results_raw.append((cps_r, bd_r["PV"]))
            except Exception:
                results_raw.append((0.0, False))

            # ── PROJECT valid atoms only ──────────────────────────────────
            if len(ppos_v) > 1:
                ppos_proj, n_it = project_repulsion(ppos_v, lengths_i)
                iter_counts.append(n_it)
            else:
                ppos_proj = ppos_v
                iter_counts.append(0)

            # Rebuild full pred_pos with projected valid atoms
            ppos_full_proj = ppos_i.clone()
            ppos_full_proj[valid] = ppos_proj

            # ── PROJECTED metrics ─────────────────────────────────────────
            rmsd_p, _ = em.hungarian_rmsd(
                ppos_full_proj, arg_i, gt_pos, gt_types, lengths_i)
            rmsd_proj.append(rmsd_p)
            try:
                cps_p, bd_p = cs.composite_physical_score(
                    ppos_full_proj, arg_i, sample_name, lengths_i, IDX_TO_Z)
                results_proj.append((cps_p, bd_p["PV"]))
            except Exception:
                results_proj.append((0.0, False))

elapsed = time.time() - t0
n = len(results_raw)
print(f"  done: {n} samples in {elapsed:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n[4/4] Results ...")

cps_raw_mean  = np.mean([x[0] for x in results_raw])
pv_raw        = np.mean([float(x[1]) for x in results_raw])
cps_proj_mean = np.mean([x[0] for x in results_proj])
pv_proj       = np.mean([float(x[1]) for x in results_proj])
rmsd_raw_m    = np.mean(rmsd_raw)
rmsd_proj_m   = np.mean(rmsd_proj)
iter_mean     = np.mean(iter_counts)
iter_maxed    = np.mean([float(x == MAX_ITER) for x in iter_counts])

print("\n" + "=" * 70)
print("PROJECTION COMPARISON  (Exp6 v7, epoch=168, full val n={})".format(n))
print("=" * 70)
print(f"{'':30s}  {'RAW':>10}  {'PROJECTED':>10}  {'DELTA':>10}")
print(f"{'hungarian_rmsd (Å)':30s}  {rmsd_raw_m:10.4f}  {rmsd_proj_m:10.4f}  "
      f"{rmsd_proj_m - rmsd_raw_m:+10.4f}")
print(f"{'pv_pass_rate':30s}  {pv_raw:10.4f}  {pv_proj:10.4f}  "
      f"{pv_proj - pv_raw:+10.4f}")
print(f"{'val_cps_mean':30s}  {cps_raw_mean:10.4f}  {cps_proj_mean:10.4f}  "
      f"{cps_proj_mean - cps_raw_mean:+10.4f}")
print(f"\n  projection stats:")
print(f"  avg iters used          = {iter_mean:.1f} / {MAX_ITER}")
print(f"  frac hitting max_iter   = {iter_maxed:.3f}  "
      f"({'WARNING: many samples not converged' if iter_maxed > 0.3 else 'OK'})")

print("\n  Interpretation:")
if cps_proj_mean > 0.3:
    print("  → CPS大幅提升: 模型学到了结构,PV是主要瓶颈。Exp7方向:强化物理约束。")
elif cps_proj_mean > 0.1:
    print("  → CPS中等提升: 模型部分学到结构,但坐标精度本身也不足。")
else:
    print("  → CPS仍极低: 投影后也没用。模型根本没学到shell结构,PV不是唯一问题。")

# Save
out = {
    "n_samples": n, "max_iter": MAX_ITER, "step_size": STEP_SIZE,
    "min_pdist": MIN_PDIST,
    "raw":  {"rmsd": rmsd_raw_m,  "pv_pass": pv_raw,  "cps": cps_raw_mean},
    "proj": {"rmsd": rmsd_proj_m, "pv_pass": pv_proj, "cps": cps_proj_mean},
    "iter_mean": iter_mean, "iter_maxed_frac": iter_maxed,
}
out_path = BASE / "step3" / "eval_projection_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  saved → {out_path}")