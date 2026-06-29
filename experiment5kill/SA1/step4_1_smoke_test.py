"""
step4_1_smoke_test.py — Exp5 SA1 smoke test
=================================================================

NOTE: This file did NOT exist in /home/tcat/diffcsp_exp4/code/step4/ as of
2026-04-28. EXP4_FILE_GUIDE.md §2.1 listed it but `cat` returned
"No such file or directory". SA1 wrote this minimal smoke from scratch
to satisfy EXP5_STEP1_HANDOFF §5 acceptance gate item 2.

Purpose: verify the Exp5 SA1 architecture-modified training step does NOT
crash over a few real batches, and that loss components are at sane scales.
This is NOT a training run — runs 5 batches CPU, no checkpoints, no logs to
disk beyond stdout.

Acceptance (handoff §5 item 2):
  ✓ 5–10 batches forward + backward, no crash
  ✓ All four losses finite (no NaN/Inf):
      - loss_coord            (diffusion coord MSE; Exp4 saw ~ 1.0)
      - loss_diffusion_type   (diffusion inner type MSE; Exp4 saw ~ small;
                               handoff §5 item 2 says "~4-5 close to ln(89)" —
                               that text appears to confuse MSE with CE; SA1
                               flags as OPEN QUESTION in OUTPUT.md)
      - loss_type_ce_head     (Exp5 head CE; init ~ ln(100) = 4.6)
      - loss_total            (sum + density + lattice*0)
  ✓ Three type_loss_modes (diffusion_only / head_only / both) all run

Run from /home/tcat/diffcsp_exp5/code/step3/ (NOT step4/) with mlff env:
    /home/tcat/conda_envs/mlff/bin/python /home/tcat/diffcsp_exp5/code/step4/step4_1_smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

# Resolve paths — script lives in step4/, deps live in step3/ and step2/
HERE = Path(__file__).parent.resolve()
STEP3_DIR = HERE.parent / "step3"
STEP2_DIR = HERE.parent / "step2"
for d in (STEP3_DIR, STEP2_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

DATA_DIR = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp5/data")
N_BATCHES = int(os.environ.get("SMOKE_N_BATCHES", "5"))
BATCH_SIZE = int(os.environ.get("SMOKE_BATCH_SIZE", "4"))


def _instantiate_model():
    """Mirror forward_test.py:_instantiate_model — load yaml + minimal optim placeholder."""
    from omegaconf import OmegaConf
    import hydra

    yaml_path = STEP3_DIR / "conf_xas" / "model" / "diffusion_xas.yaml"
    model_cfg = OmegaConf.load(yaml_path)

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1.0e-4},
        "use_lr_scheduler": False,
    })
    return hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)


def _check_finite(loss_val: float, name: str):
    if loss_val != loss_val:   # NaN
        raise RuntimeError(f"{name} is NaN")
    if loss_val == float("inf") or loss_val == float("-inf"):
        raise RuntimeError(f"{name} is Inf: {loss_val}")


def smoke_loop(mode: str | None = None):
    """One full smoke pass — N_BATCHES iters, optionally overriding type_loss_mode."""
    print("=" * 72)
    print(f"smoke_loop(mode={mode!r}, n_batches={N_BATCHES}, batch_size={BATCH_SIZE})")
    print("=" * 72)

    from xas_local_datamodule_v2 import XasLocalDataModuleV2

    torch.manual_seed(42)
    dm = XasLocalDataModuleV2(batch_size=BATCH_SIZE, num_workers=0, data_dir=DATA_DIR)
    dm.setup("fit")
    loader = dm.train_dataloader()

    model = _instantiate_model().train()
    if mode is not None:
        model.type_loss_mode = mode

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params, type_loss_mode={model.type_loss_mode!r}")
    print(f"  diff_type_w={model.diffusion_type_weight}, "
          f"head_type_w={model.head_type_weight}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    t0 = time.perf_counter()
    losses_seen = []

    iter_loader = iter(loader)
    for i in range(N_BATCHES):
        try:
            batch = next(iter_loader)
        except StopIteration:
            print(f"  WARN: loader exhausted after {i} batches")
            break
        if batch is None:
            print(f"  WARN: batch {i} is None (collate filtered all samples), skip")
            continue

        opt.zero_grad()
        out = model(batch)
        loss = out["loss"]

        loss_val = loss.item()
        _check_finite(loss_val, f"batch {i} total loss")
        for k in ("loss_coord", "loss_diffusion_type", "loss_type_ce_head",
                  "loss_type_total", "loss_density"):
            _check_finite(out[k].item(), f"batch {i} {k}")

        loss.backward()

        # grad sanity
        gnorm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    raise RuntimeError(f"batch {i}: NaN/Inf grad in param {tuple(p.shape)}")
                gnorm_sq += p.grad.norm().item() ** 2
        gnorm = gnorm_sq ** 0.5

        opt.step()

        losses_seen.append({
            "total":           loss_val,
            "coord":           out["loss_coord"].item(),
            "diff_type":       out["loss_diffusion_type"].item(),
            "head_ce":         out["loss_type_ce_head"].item(),
            "type_total":      out["loss_type_total"].item(),
            "density":         out["loss_density"].item(),
            "gnorm":           gnorm,
        })
        print(f"  batch {i}: total={loss_val:7.4f}  coord={out['loss_coord'].item():6.4f}  "
              f"diff_t={out['loss_diffusion_type'].item():6.4f}  "
              f"head_ce={out['loss_type_ce_head'].item():6.4f}  "
              f"type_tot={out['loss_type_total'].item():6.4f}  "
              f"gnorm={gnorm:7.2f}")

    elapsed = time.perf_counter() - t0
    n_done = len(losses_seen)
    if n_done == 0:
        raise RuntimeError("smoke_loop: zero successful batches!")
    print(f"  {n_done}/{N_BATCHES} batches succeeded in {elapsed:.1f} s "
          f"({elapsed/n_done*1000:.0f} ms/batch)")

    # Summary
    avg = lambda k: sum(d[k] for d in losses_seen) / n_done
    print(f"  avg total={avg('total'):.4f}  coord={avg('coord'):.4f}  "
          f"diff_type={avg('diff_type'):.4f}  head_ce={avg('head_ce'):.4f}  "
          f"type_total={avg('type_total'):.4f}")

    # head_ce should be near ln(100) = 4.605 at random init
    head_ce_avg = avg("head_ce")
    if not (1.0 < head_ce_avg < 8.0):
        print(f"  WARN: head_ce avg {head_ce_avg:.4f} outside [1, 8] — random init expected ~4.6")

    print(f"[smoke mode={mode!r} PASS]")
    return losses_seen


def main():
    print(f"step4_1_smoke_test.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DATA_DIR  = {DATA_DIR}")
    print(f"torch     = {torch.__version__}, cuda = {torch.cuda.is_available()}")
    print()

    # 1. Default mode (yaml says 'both')
    smoke_loop(mode=None)
    print()

    # 2. Each of the three modes explicitly
    for mode in ("diffusion_only", "head_only", "both"):
        smoke_loop(mode=mode)
        print()

    print("=" * 72)
    print("ALL SMOKE PASSES — Exp5 SA1 architecture clears training loop sanity")
    print("=" * 72)


if __name__ == "__main__":
    main()
