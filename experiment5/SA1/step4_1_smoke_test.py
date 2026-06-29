"""
step4_1_smoke_test.py — Exp5 v2 SA1' smoke test (1 mode, 2 epoch × 10 batch)
=================================================================

Purpose: verify Exp5 v2 architecture (撤销 v1 head + MV-attention fusion + cost_density 0.2)
does NOT crash over a few real batches, loss components are at sane scales, and
val_loss decreases over 2 epochs (basic sanity).

This is NOT a full training run — limited to N_BATCHES=10 train + 5 val per
epoch × 2 epochs. ckpt is dropped to /home/tcat/diffcsp_exp5/checkpoints/_smoke/
(SA1' must `rm -rf` after smoke passes per handoff §6.9 红线).

Acceptance (handoff §6.9):
  ✓ 2 epochs × 10 batches forward + backward, no crash
  ✓ All four losses finite (no NaN/Inf):
      - loss        (total = lattice*0 + coord + type + density*0.2)
      - coord_loss  (diffusion coord MSE; Exp4 saw ~ 1.0)
      - type_loss   (diffusion-internal type MSE)
      - density_loss
  ✓ val_loss(epoch 1) < val_loss(epoch 0)  (rough sanity, not gating)
  ✓ no v1 head fields (loss_type_ce_head / loss_diffusion_type / loss_type_total)
  ✓ ckpt dropped at /home/tcat/diffcsp_exp5/checkpoints/_smoke/

Run from /home/tcat/diffcsp_exp5/code/step4/ with mlff env active:

    cd /home/tcat/diffcsp_exp5/code/step4
    PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
    EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
      /home/tcat/conda_envs/mlff/bin/python step4_1_smoke_test.py 2>&1 | \
      tee /home/tcat/diffcsp_exp5/logs/step1_smoke_v2.log

After SMOKE PASS — clean up to avoid disk pollution:
    rm -rf /home/tcat/diffcsp_exp5/checkpoints/_smoke/
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

# ── PYTHONPATH self-check (handoff §1.2 + §6.6 D, carry-over from v1 SA1 §5.6) ──
HERE = Path(__file__).parent.resolve()
EXP5_STEP3 = "/home/tcat/diffcsp_exp5/code/step3"
EXP5_STEP2 = "/home/tcat/diffcsp_exp5/code/step2"
EXP4_BACKBONE = "/home/tcat/diffcsp_exp4/code"

for p in (EXP5_STEP2, EXP5_STEP3):
    if p not in sys.path:
        sys.path.insert(0, p)
if EXP4_BACKBONE not in sys.path:
    sys.path.append(EXP4_BACKBONE)

# Verify we're loading Exp5 versions, not Exp4 same-named files
import diffusion_w_type_xas    # noqa: E402
import spectrum_encoder        # noqa: E402
assert "/diffcsp_exp5/" in diffusion_w_type_xas.__file__, \
    f"WRONG diffusion_w_type_xas: {diffusion_w_type_xas.__file__}"
assert "/diffcsp_exp5/" in spectrum_encoder.__file__, \
    f"WRONG spectrum_encoder: {spectrum_encoder.__file__}"
print(f"[PYTHONPATH check] diffusion_w_type_xas: {diffusion_w_type_xas.__file__}")
print(f"[PYTHONPATH check] spectrum_encoder:     {spectrum_encoder.__file__}")

DATA_DIR  = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp5/data")
CKPT_DIR  = os.environ.get("SMOKE_CKPT_DIR",
                            "/home/tcat/diffcsp_exp5/checkpoints/_smoke")
N_TRAIN_BATCHES = int(os.environ.get("SMOKE_N_TRAIN", "10"))
N_VAL_BATCHES   = int(os.environ.get("SMOKE_N_VAL",   "5"))
N_EPOCHS        = int(os.environ.get("SMOKE_N_EPOCHS", "2"))
BATCH_SIZE      = int(os.environ.get("SMOKE_BATCH_SIZE", "4"))


def _instantiate_model():
    """Mirror forward_test.py:_instantiate_model — load yaml + minimal optim placeholder."""
    from omegaconf import OmegaConf
    import hydra

    yaml_path = Path(EXP5_STEP3) / "conf_xas" / "model" / "diffusion_xas.yaml"
    model_cfg = OmegaConf.load(yaml_path)

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1.0e-4},
        "use_lr_scheduler": False,
    })
    return hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)


def _check_finite(loss_val: float, name: str):
    if loss_val != loss_val:
        raise RuntimeError(f"{name} is NaN")
    if loss_val == float("inf") or loss_val == float("-inf"):
        raise RuntimeError(f"{name} is Inf: {loss_val}")


def main():
    print(f"\nstep4_1_smoke_test.py — {time.strftime('%Y-%m-%d %H:%M:%S')}  (Exp5 v2 SA1')")
    print(f"DATA_DIR  = {DATA_DIR}")
    print(f"CKPT_DIR  = {CKPT_DIR}")
    print(f"torch     = {torch.__version__}, cuda = {torch.cuda.is_available()}")
    print(f"config    : {N_EPOCHS} epochs × {N_TRAIN_BATCHES} train + {N_VAL_BATCHES} val "
          f"@ batch_size={BATCH_SIZE}\n")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── DataModule + Model ─────────────────────────────────────────────────
    from xas_local_datamodule_v2 import XasLocalDataModuleV2

    torch.manual_seed(42)
    dm = XasLocalDataModuleV2(batch_size=BATCH_SIZE, num_workers=0, data_dir=DATA_DIR)
    dm.setup("fit")

    model = _instantiate_model().train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CSPDiffusion, {n_params:,} params, "
          f"keep_lattice={model.keep_lattice}, cost_density={model.cost_density}")

    # ── Verify v1 head 痕迹 已彻底删干净 ──────────────────────────────────
    forbidden = ['type_head', 'type_loss_mode', 'diffusion_type_weight',
                 'head_type_weight', 'head_predict_types']
    for attr in forbidden:
        if hasattr(model, attr):
            raise RuntimeError(f"FAIL: model still has v1 head attribute: {attr}")
    print(f"  ✓ no v1 head attributes ({', '.join(forbidden)} all absent)")

    # ── Verify MV-attention components present ──────────────────────────
    enc = model.spectrum_encoder
    for attr in ['mv_attn', 'mv_query', 'mv_layernorm', 'mv_proj', 'center_emb']:
        if not hasattr(enc, attr):
            raise RuntimeError(f"FAIL: SpectrumEncoder missing {attr}")
    if hasattr(enc, 'fusion'):
        raise RuntimeError("FAIL: SpectrumEncoder still has v1 fusion block")
    print(f"  ✓ MV-attention components: mv_attn (heads={enc.mv_attn.num_heads}), "
          f"mv_query, mv_layernorm, mv_proj, center_emb")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── Training loop: 2 epochs × 10 train batches + 5 val batches ──────
    epoch_val_losses = []

    for epoch in range(N_EPOCHS):
        print(f"\n========== EPOCH {epoch} ==========")

        # train pass
        model.train()
        train_loader = dm.train_dataloader()
        train_iter = iter(train_loader)
        t0 = time.perf_counter()
        train_total = 0.0
        n_train_done = 0

        for i in range(N_TRAIN_BATCHES):
            try:
                batch = next(train_iter)
            except StopIteration:
                print(f"  WARN: train loader exhausted after {i} batches")
                break
            if batch is None:
                print(f"  WARN: train batch {i} is None (collate filtered all), skip")
                continue

            opt.zero_grad()
            out = model(batch)
            loss = out["loss"]

            loss_val = loss.item()
            _check_finite(loss_val, f"epoch {epoch} train batch {i} total loss")
            for k in ("loss_coord", "loss_type", "loss_density", "loss_lattice"):
                _check_finite(out[k].item(), f"epoch {epoch} train batch {i} {k}")

            # Verify no v1 head fields in output dict
            for forbidden_k in ("loss_type_ce_head", "loss_diffusion_type",
                                 "loss_type_total"):
                if forbidden_k in out:
                    raise RuntimeError(
                        f"FAIL: forward output still contains v1 head field: {forbidden_k}")

            loss.backward()

            gnorm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        raise RuntimeError(
                            f"epoch {epoch} train batch {i}: NaN/Inf grad in "
                            f"param {tuple(p.shape)}")
                    gnorm_sq += p.grad.norm().item() ** 2
            gnorm = gnorm_sq ** 0.5

            opt.step()

            train_total += loss_val
            n_train_done += 1
            print(f"  train b{i:02d}: loss={loss_val:7.4f}  "
                  f"coord={out['loss_coord'].item():6.4f}  "
                  f"type={out['loss_type'].item():6.4f}  "
                  f"density={out['loss_density'].item():6.4f}  "
                  f"gnorm={gnorm:7.2f}")

        train_avg = train_total / max(n_train_done, 1)
        elapsed = time.perf_counter() - t0
        print(f"  TRAIN epoch {epoch}: avg_loss={train_avg:.4f}  "
              f"({n_train_done} batches in {elapsed:.1f}s)")

        # val pass
        model.eval()
        val_loader = dm.val_dataloader()
        val_iter = iter(val_loader)
        val_total = 0.0
        n_val_done = 0
        with torch.no_grad():
            for i in range(N_VAL_BATCHES):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                if batch is None:
                    continue
                out = model(batch)
                loss_val = out["loss"].item()
                _check_finite(loss_val, f"epoch {epoch} val batch {i} loss")
                val_total += loss_val
                n_val_done += 1

        val_avg = val_total / max(n_val_done, 1)
        epoch_val_losses.append(val_avg)
        print(f"  VAL   epoch {epoch}: avg_loss={val_avg:.4f}  ({n_val_done} batches)")

    # ── Sanity: val_loss should decrease across 2 epochs (rough check) ──
    print("\n========== SUMMARY ==========")
    print(f"  val_loss epoch 0 → {N_EPOCHS-1}: " +
          " → ".join(f"{v:.4f}" for v in epoch_val_losses))

    if len(epoch_val_losses) >= 2:
        delta = epoch_val_losses[-1] - epoch_val_losses[0]
        if delta < 0:
            print(f"  ✓ val_loss decreased by {-delta:.4f} (sanity check passes)")
        else:
            print(f"  ⚠️  val_loss INCREASED by {delta:.4f} "
                  f"(only 10 batches/epoch — not gating; SA2' should monitor full run)")

    # ── Save final ckpt ────────────────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, "smoke_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch_val_losses": epoch_val_losses,
        "n_params": n_params,
    }, ckpt_path)
    print(f"  ckpt saved: {ckpt_path}")

    print("\n" + "=" * 72)
    print("SMOKE PASS — Exp5 v2 architecture clears training loop sanity")
    print("Reminder: rm -rf /home/tcat/diffcsp_exp5/checkpoints/_smoke/  (handoff §6.9 红线)")
    print("=" * 72)


if __name__ == "__main__":
    main()
