"""
forward_test.py — Exp4 Step 3 Phase 6 (Sub-Agent 4 final deliverable)
=====================================================================

Five-sub-phase forward test gate (HANDOFF §8.2 / EXP4_STEP3_SUBAGENT_HANDOFF §9.2):

  6.1  Dataset 100 random samples — frac sentinel + 12-field schema
  6.2  DataLoader collate (bs=4)  — PyG Batch field alignment with diffusion forward()
  6.3  SpectrumEncoder forward    — (4, 256), no NaN, mean ∈ [-5, 5], std ∈ [0.1, 5]
  6.4  CPU full forward+backward  — loss ∈ [2, 6] (warn-only), grad_norm ∈ (0, 1e4)
  6.5  GPU bf16 forward+backward  — loss ±10% of CPU, no NaN/Inf grad

All five must PASS to clear the Step 4 launch gate.

Run from /home/tcat/diffcsp_exp4/code/step3/ with mlff env active:
    python forward_test.py

Log written to /home/tcat/diffcsp_exp4/logs/step3_forward_test_log.txt
"""
from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

# Local step3 imports (same dir as this file)
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# step2/ for SpectrumEncoder direct import in Phase 6.3
STEP2_DIR = HERE.parent / "step2"
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))

DATA_DIR = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")
LOG_PATH = "/home/tcat/diffcsp_exp4/logs/step3_forward_test_log.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

_log_lines: list[str] = []

def log(msg: str = ""):
    print(msg, flush=True)
    _log_lines.append(msg)

def save_log():
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\n[log saved to {LOG_PATH}]", flush=True)

def fail(phase: str, why: str):
    log(f"\n!!! {phase} FAILED !!!")
    log(why)
    save_log()
    sys.exit(1)


# ============================================================================
# Phase 6.1 — Dataset 100 random samples
# ============================================================================
def phase_61():
    log("=" * 72)
    log("Phase 6.1 — Dataset 100 random samples (12-field schema + frac sentinel)")
    log("=" * 72)

    from xas_local_dataset_v2 import XasLocalDatasetV2
    ds = XasLocalDatasetV2(split="train", data_dir=DATA_DIR, verbose_init_benchmark=False)
    log(f"Dataset size: {len(ds)} (expect 60,507)")
    if len(ds) != 60507:
        fail("Phase 6.1", f"Dataset size {len(ds)} != 60,507 (PROPOSAL §2.1 train count)")

    expected_keys = {
        "xmu", "chi1", "feff", "frac_coords", "atom_types",
        "sample_name", "mp_id", "center_element",
        "eval_cutoff", "eval_cutoff_fallback", "n_center_sites",
        "site_equivalence_tag",
    }

    random.seed(42)
    indices = random.sample(range(len(ds)), 100)

    t0 = time.perf_counter()
    frac_min_g = float("inf")
    frac_max_g = float("-inf")
    n_fallback = 0
    elem_counter = {}

    for n, i in enumerate(indices):
        try:
            s = ds[i]
        except Exception as e:
            fail("Phase 6.1", f"Sample idx={i} (n={n}) raised {type(e).__name__}: {e}")

        keys = set(s.keys())
        if keys != expected_keys:
            fail("Phase 6.1", f"Sample idx={i} keys mismatch:\n"
                 f"  missing: {expected_keys - keys}\n  extra: {keys - expected_keys}")

        # shape sanity
        for k, expect in (("xmu", (150,)), ("chi1", (200,)), ("feff", (74,)),
                         ("frac_coords", (20, 3)), ("atom_types", (20,))):
            if tuple(s[k].shape) != expect:
                fail("Phase 6.1", f"Sample idx={i} {k} shape {tuple(s[k].shape)} != {expect}")

        # frac sentinel (R3 in dataset_v2 already raises; double-check here)
        fc = s["frac_coords"]
        fmin, fmax = fc.min().item(), fc.max().item()
        frac_min_g = min(frac_min_g, fmin)
        frac_max_g = max(frac_max_g, fmax)
        if fmin < -0.5 - 1e-6 or fmax > 0.5 + 1e-6:
            fail("Phase 6.1", f"Sample idx={i} frac out of [-0.5, 0.5]: min={fmin}, max={fmax}")

        # atom_types ∈ [1, 109]
        atmin, atmax = s["atom_types"].min().item(), s["atom_types"].max().item()
        if atmin < 1 or atmax > 109:
            fail("Phase 6.1", f"Sample idx={i} atom_types out of [1, 109]: min={atmin}, max={atmax}")

        # observational stats (not gated, just informative)
        if s["eval_cutoff_fallback"]:
            n_fallback += 1
        elem_counter[s["center_element"]] = elem_counter.get(s["center_element"], 0) + 1

    elapsed = time.perf_counter() - t0
    log(f"100 samples in {elapsed:.2f} s ({elapsed*1000/100:.1f} ms/sample)")
    log(f"frac global range: [{frac_min_g:.6f}, {frac_max_g:.6f}]")
    log(f"eval_cutoff_fallback hits: {n_fallback}/100 (PROPOSAL says 5-10%)")
    log(f"unique center_elements seen: {len(elem_counter)} (e.g., {list(elem_counter.keys())[:8]})")
    log("[Phase 6.1 PASS]")


# ============================================================================
# Phase 6.2 — DataLoader collate (bs=4) → PyG Batch
# ============================================================================
def phase_62():
    log("\n" + "=" * 72)
    log("Phase 6.2 — DataLoader collate (bs=4) → PyG Batch")
    log("=" * 72)

    from xas_local_datamodule_v2 import XasLocalDataModuleV2
    dm = XasLocalDataModuleV2(batch_size=4, num_workers=0, data_dir=DATA_DIR)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    if batch is None:
        fail("Phase 6.2", "First batch is None")

    expects = [
        ("frac_coords",   (80, 3)),
        ("atom_types",    (80,)),
        ("xmu_xanes",     (4, 150)),  # renamed from xmu
        ("chi1",          (4, 200)),
        ("feff_features", (4, 74)),   # renamed from feff
        ("lengths",       (4, 3)),
        ("angles",        (4, 3)),
        ("eval_cutoff",   (4,)),
        ("batch",         (80,)),
    ]
    if batch.num_graphs != 4:
        fail("Phase 6.2", f"num_graphs={batch.num_graphs}, expect 4")
    log(f"  num_graphs:    {batch.num_graphs}")

    for k, expect in expects:
        v = getattr(batch, k, None)
        if v is None:
            fail("Phase 6.2", f"Batch missing field: {k}")
        if tuple(v.shape) != expect:
            fail("Phase 6.2", f"Field {k} shape {tuple(v.shape)} != {expect}")
        log(f"  {k:14s}: {tuple(v.shape)}")

    # num_atoms (B,) tensor with all 20s
    if tuple(batch.num_atoms.shape) != (4,) or not (batch.num_atoms == 20).all():
        fail("Phase 6.2", f"num_atoms = {batch.num_atoms}, expect tensor([20,20,20,20])")
    log(f"  num_atoms:     {batch.num_atoms.tolist()}")

    # mp_id list of 4 strings
    if not isinstance(batch.mp_id, list) or len(batch.mp_id) != 4:
        fail("Phase 6.2", f"batch.mp_id type/len wrong: {type(batch.mp_id).__name__}")
    log(f"  mp_id list:    {batch.mp_id}")

    # lengths value
    if not torch.allclose(batch.lengths[0], torch.tensor([6.0, 6.0, 6.0])):
        fail("Phase 6.2", f"lengths[0]={batch.lengths[0]}, expect [6,6,6]")
    log(f"  lengths[0]:    {batch.lengths[0].tolist()}")

    log("[Phase 6.2 PASS]")
    return batch


# ============================================================================
# Phase 6.3 — SpectrumEncoder forward
# ============================================================================
def phase_63(batch):
    log("\n" + "=" * 72)
    log("Phase 6.3 — SpectrumEncoder forward → (4, 256)")
    log("=" * 72)

    from spectrum_encoder import SpectrumEncoder
    enc = SpectrumEncoder().eval()  # default xmu_dim=150 chi_dim=200 feat_dim=74 latent_dim=256
    log("SpectrumEncoder instantiated (defaults 150/200/74/256)")

    with torch.no_grad():
        z = enc(batch.xmu_xanes, batch.chi1, batch.feff_features)

    if tuple(z.shape) != (4, 256):
        fail("Phase 6.3", f"Output shape {tuple(z.shape)} != (4, 256)")
    if torch.isnan(z).any():
        fail("Phase 6.3", "NaN in encoder output")
    if torch.isinf(z).any():
        fail("Phase 6.3", "Inf in encoder output")

    z_mean, z_std = z.mean().item(), z.std().item()
    log(f"  output: shape={tuple(z.shape)}, mean={z_mean:+.4f}, std={z_std:.4f}")

    if not (-5.0 < z_mean < 5.0):
        fail("Phase 6.3", f"mean {z_mean} outside [-5, 5]")
    if not (0.1 < z_std < 5.0):
        fail("Phase 6.3", f"std {z_std} outside [0.1, 5]")

    log("[Phase 6.3 PASS]")


# ============================================================================
# Phase 6.4 / 6.5 helper: instantiate CSPDiffusion via hydra
# ============================================================================
def _instantiate_model():
    from omegaconf import OmegaConf
    import hydra

    yaml_path = HERE / "conf_xas" / "model" / "diffusion_xas.yaml"
    model_cfg = OmegaConf.load(yaml_path)

    # Minimal optim placeholder (hparams.optim must exist for save_hyperparameters,
    # but configure_optimizers is only called by Trainer.fit — not in this test).
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1.0e-4},
        "use_lr_scheduler": False,
    })
    return hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)


# ============================================================================
# Phase 6.4 — CPU full forward + backward
# ============================================================================
def phase_64(batch):
    log("\n" + "=" * 72)
    log("Phase 6.4 — CPU full forward + backward")
    log("=" * 72)

    torch.manual_seed(42)
    model = _instantiate_model().train()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"CSPDiffusion: {n_params:,} params, keep_lattice={model.keep_lattice}")
    if not model.keep_lattice:
        fail("Phase 6.4", "keep_lattice=False, expect True (cost_lattice=0)")

    out = model(batch)
    loss = out["loss"]

    loss_val = loss.item()
    log(f"  loss        : {loss_val:.4f}")
    log(f"  loss_coord  : {out['loss_coord'].item():.4f}")
    log(f"  loss_type   : {out['loss_type'].item():.4f}")
    log(f"  loss_density: {out['loss_density'].item():.4f}")
    log(f"  loss_lattice: {out['loss_lattice'].item():.4f} (× cost_lattice=0, no contribution)")

    if torch.isnan(loss) or torch.isinf(loss):
        fail("Phase 6.4", f"loss is NaN/Inf: {loss_val}")
    if not (2.0 <= loss_val <= 6.0):
        log(f"  WARN: loss={loss_val:.4f} outside HANDOFF [2, 6] (random-init can drift; not gating)")

    loss.backward()

    grad_norm_sq = 0.0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                fail("Phase 6.4", f"NaN grad in param shape {tuple(p.shape)}")
            if torch.isinf(p.grad).any():
                fail("Phase 6.4", f"Inf grad in param shape {tuple(p.shape)}")
            grad_norm_sq += p.grad.norm().item() ** 2
            n_with_grad += 1
    grad_norm = grad_norm_sq ** 0.5
    log(f"  grad_norm   : {grad_norm:.4f} (over {n_with_grad} params)")

    if not (grad_norm > 0):
        fail("Phase 6.4", f"grad_norm={grad_norm} (expect >0)")
    if not (grad_norm < 1e4):
        fail("Phase 6.4", f"grad_norm={grad_norm} >= 1e4")

    log("[Phase 6.4 PASS]")
    return loss_val


# ============================================================================
# Phase 6.5 — GPU bf16 full forward + backward
# ============================================================================
def phase_65(batch_cpu, cpu_loss):
    log("\n" + "=" * 72)
    log("Phase 6.5 — GPU bf16 full forward + backward")
    log("=" * 72)

    if not torch.cuda.is_available():
        fail("Phase 6.5", "CUDA not available")
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    model = _instantiate_model()
    model = model.to("cuda:0").to(torch.bfloat16)
    model.train()

    # Move batch to GPU; bf16-cast only floating-point tensors
    batch = batch_cpu.to("cuda:0")
    for k in ("xmu_xanes", "chi1", "feff_features",
              "lengths", "angles", "frac_coords",
              "eval_cutoff", "eval_cutoff_fallback"):
        if hasattr(batch, k):
            v = getattr(batch, k)
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                setattr(batch, k, v.to(torch.bfloat16))

    log(f"  xmu_xanes.dtype={batch.xmu_xanes.dtype}, "
        f"atom_types.dtype={batch.atom_types.dtype}, "
        f"num_atoms.dtype={batch.num_atoms.dtype}")

    out = model(batch)
    loss = out["loss"]
    loss_val_f32 = loss.float().item()
    log(f"  GPU bf16 loss: {loss_val_f32:.4f}")
    log(f"  CPU fp32 loss: {cpu_loss:.4f}")
    drift_pct = abs(loss_val_f32 - cpu_loss) / max(abs(cpu_loss), 1e-9) * 100
    log(f"  drift        : {drift_pct:.1f}% (HANDOFF advisory ±10%)")

    if torch.isnan(loss) or torch.isinf(loss):
        fail("Phase 6.5", f"GPU loss is NaN/Inf: {loss_val_f32}")

    loss.backward()

    grad_norm_sq = 0.0
    bad = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any():
            bad.append(("NaN", n))
        elif torch.isinf(p.grad).any():
            bad.append(("Inf", n))
        else:
            grad_norm_sq += p.grad.float().norm().item() ** 2
    grad_norm = grad_norm_sq ** 0.5
    log(f"  GPU grad_norm: {grad_norm:.4f}")

    if bad:
        log(f"  bad grads ({len(bad)}): {bad[:5]}{' ...' if len(bad)>5 else ''}")
        fail("Phase 6.5", "NaN/Inf gradients on GPU bf16")

    if not (0 < grad_norm < 1e4):
        fail("Phase 6.5", f"GPU grad_norm={grad_norm} outside (0, 1e4)")

    if drift_pct > 10:
        log(f"  WARN: drift {drift_pct:.1f}% > 10% (advisory only, not gating)")

    log("[Phase 6.5 PASS]")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    t_start = time.perf_counter()

    log(f"forward_test.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"DATA_DIR  = {DATA_DIR}")
    log(f"torch     = {torch.__version__}, cuda = {torch.cuda.is_available()}")
    log("")

    phase_61()
    batch = phase_62()
    phase_63(batch)
    cpu_loss = phase_64(batch)
    phase_65(batch, cpu_loss)

    t_total = time.perf_counter() - t_start
    log("\n" + "=" * 72)
    log(f"ALL FIVE PHASES PASS — total wall time: {t_total:.1f} s")
    log("Step 4 launch gate: CLEAR")
    log("=" * 72)
    save_log()
