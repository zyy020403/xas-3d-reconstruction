#!/usr/bin/env python
"""
step4_2_train.py — Exp5 SA2 main training entry
=================================================================
Purpose
-------
Warm-start Exp5 baseline_v2 from Exp4 best ckpt (best-epoch366-val0.7300.ckpt)
and train to convergence with phased training + early stop. Produces
best.ckpt at /home/tcat/diffcsp_exp5/checkpoints/.

Architecture additions trained (random-init from SA1):
    - spectrum_encoder.center_emb     nn.Embedding(95, 16)              (1 key)
    - type_head.fc.{0,2}.{weight,bias} TypeClassifierHead Sequential   (4 keys)
    - decoder.atom_latent_emb.{weight,bias} (Exp4 in=512 → Exp5 in=528) (2 keys)
                                       7 missing keys total

Phased training (handoff §2.1, §2.4 option A):
    Phase 1 (epoch 0-5):   freeze backbone except 7 new keys; lr = 1e-3 (head warmup)
    Phase 2 (epoch 6-end): unfreeze all; differential lr — head=1e-4, backbone=1e-5

Acceptance gates (handoff §6, all 8 wired into log):
    1. PYTHONPATH self-check both /diffcsp_exp5/                  — startup print
    2. OQ-1 sanity scan: max(atom_types) < 100                    — pre-fit print
    3. ckpt warm-start: missing/unexpected match handoff §5.2     — pre-fit print
    4. val_loss converged or max_epochs hit                        — fit-end print
    5. val_coord_loss(end) ≤ 0.7500                                — fit-end print
    6. no NaN/Inf in 3 loss components throughout training         — log inspection
    7. best.ckpt + last.ckpt + train log saved                     — fit-end print
    8. (handoff §6 item 8) write OUTPUT.md — out of script scope

Usage (do NOT cd elsewhere — PYTHONPATH is set by env, not cwd):
    cd /home/tcat/diffcsp_exp5/code/step3
    PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
    EXP4_DATA_DIR=/tmp/diffcsp_cache \
    nohup /home/tcat/conda_envs/mlff/bin/python /home/tcat/diffcsp_exp5/code/step4/step4_2_train.py \
      > /home/tcat/diffcsp_exp5/logs/step2_train.log 2>&1 &
    echo "Train PID: $!"

Optional resume (after crash):
    RESUME_FROM_CKPT=/path/to/last.ckpt python step4_2_train.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import types
from pathlib import Path

# ── §5.6 PYTHONPATH self-check (CARRY-OVER from SA1 OUTPUT §5.6) ────────────
# Insert Exp5 dirs FIRST so Python sees Exp5 versions of duplicated module names.
# Exp4 code path appended (not inserted) so that diffcsp.* sub-package resolves.
EXP5_STEP3 = "/home/tcat/diffcsp_exp5/code/step3"
EXP5_STEP2 = "/home/tcat/diffcsp_exp5/code/step2"
EXP4_CODE  = "/home/tcat/diffcsp_exp4/code"
for p in (EXP5_STEP3, EXP5_STEP2):
    if p not in sys.path:
        sys.path.insert(0, p)
if EXP4_CODE not in sys.path:
    sys.path.append(EXP4_CODE)

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import hydra

# Imports from Exp5 step2/step3 — will fail loudly if PYTHONPATH wrong
import diffusion_w_type_xas
import spectrum_encoder

# ── Acceptance gate 1: PYTHONPATH self-check ────────────────────────────────
assert "/diffcsp_exp5/" in diffusion_w_type_xas.__file__, (
    f"WRONG IMPORT PATH for diffusion_w_type_xas: {diffusion_w_type_xas.__file__}\n"
    f"Expected /home/tcat/diffcsp_exp5/code/step3/. Check PYTHONPATH order."
)
assert "/diffcsp_exp5/" in spectrum_encoder.__file__, (
    f"WRONG IMPORT PATH for spectrum_encoder: {spectrum_encoder.__file__}\n"
    f"Expected /home/tcat/diffcsp_exp5/code/step2/."
)
print(f"[PYTHONPATH self-check] diffusion_w_type_xas: {diffusion_w_type_xas.__file__}")
print(f"[PYTHONPATH self-check] spectrum_encoder:     {spectrum_encoder.__file__}")

from diffusion_w_type_xas import CSPDiffusion           # noqa: E402
from xas_local_datamodule_v2 import XasLocalDataModuleV2  # noqa: E402

from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger


# ── Constants ───────────────────────────────────────────────────────────────
EXP4_CKPT     = "/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt"
EXP4_CKPT_MD5 = "dc9d2c9b371c78125f285a5a6478d404"          # SA0 §1 verified
EXP5_CKPT_DIR = "/home/tcat/diffcsp_exp5/checkpoints"
EXP5_LOG_DIR  = "/home/tcat/diffcsp_exp5/logs"
DATA_DIR      = os.environ.get("EXP4_DATA_DIR", "/tmp/diffcsp_cache")
RESUME_FROM   = os.environ.get("RESUME_FROM_CKPT", None)

YAML_PATH = Path(EXP5_STEP3) / "conf_xas" / "model" / "diffusion_xas.yaml"

# Hyperparams (handoff §2.3)
SEED                = 42
BATCH_SIZE          = 16
NUM_WORKERS         = 4
MAX_EPOCHS          = 400
PHASE_SWITCH_EPOCH  = 6      # epoch < 6 → Phase 1; epoch >= 6 → Phase 2
EARLY_STOP_PATIENCE = 30
GRAD_CLIP_VAL       = 1.0    # handoff §3.3 grad_norm clip default

# LR (handoff §2.1)
PHASE1_HEAD_LR     = 1e-3    # head warmup all 7 new params
PHASE2_HEAD_LR     = 1e-4    # joint fine-tune, head/new params
PHASE2_BACKBONE_LR = 1e-5    # joint fine-tune, backbone

# Param prefixes that count as "head" — Phase 1 unfrozen, Phase 2 higher lr
HEAD_PREFIXES = (
    "type_head.",
    "spectrum_encoder.center_emb.",
    "decoder.atom_latent_emb.",
)

# Expected shape-mismatch keys (Exp4 ckpt has these at Exp4 shape; Exp5 needs new shape).
# IMPORTANT: PT strict=False does NOT skip shape mismatches — it raises RuntimeError.
# We pop these from ckpt state_dict before load_state_dict; they then appear in
# `missing` and the model's random init is retained. (See SA2 OUTPUT.md note.)
SHAPE_MISMATCH_KEYS = [
    "decoder.atom_latent_emb.weight",
    "decoder.atom_latent_emb.bias",
]

# Expected missing keys (handoff §5.2 + SA1 OUTPUT §5.2). 7 keys exactly.
EXPECTED_MISSING_KEYS = {
    "spectrum_encoder.center_emb.weight",        # nn.Embedding, no bias
    "type_head.fc.0.weight", "type_head.fc.0.bias",   # Sequential[0] = Linear
    "type_head.fc.2.weight", "type_head.fc.2.bias",   # Sequential[2] = Linear
    "decoder.atom_latent_emb.weight",            # popped above, re-appears as missing
    "decoder.atom_latent_emb.bias",
}


def is_head_param(name: str) -> bool:
    return any(name.startswith(p) for p in HEAD_PREFIXES)


# ── Phased training callback (handoff §2.4 option A) ────────────────────────
class PhasedTrainingCallback(pl.Callback):
    """
    Switches requires_grad + per-group lr at phase boundary.

    Phase 1 (epoch [0, switch)):
        head_params:     requires_grad=True,  lr=PHASE1_HEAD_LR
        backbone_params: requires_grad=False, lr=0.0  (AdamW skips None-grad)

    Phase 2 (epoch [switch, ∞)):
        head_params:     requires_grad=True,  lr=PHASE2_HEAD_LR
        backbone_params: requires_grad=True,  lr=PHASE2_BACKBONE_LR
    """
    def __init__(self, switch_epoch: int = PHASE_SWITCH_EPOCH):
        super().__init__()
        self.switch_epoch = switch_epoch
        self._current_phase = 0   # 0=not entered, 1=phase1, 2=phase2

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        target = 1 if epoch < self.switch_epoch else 2
        if target == self._current_phase:
            return

        opt = trainer.optimizers[0]
        # configure_optimizers sets param_groups[0]=head, [1]=backbone
        if target == 1:
            for name, p in pl_module.named_parameters():
                p.requires_grad = is_head_param(name)
            opt.param_groups[0]['lr'] = PHASE1_HEAD_LR
            opt.param_groups[1]['lr'] = 0.0
            self._announce(epoch, "PHASE 1 (head warmup, backbone frozen)",
                           opt, pl_module)
        else:  # target == 2
            for p in pl_module.parameters():
                p.requires_grad = True
            opt.param_groups[0]['lr'] = PHASE2_HEAD_LR
            opt.param_groups[1]['lr'] = PHASE2_BACKBONE_LR
            self._announce(epoch, "PHASE 2 (joint fine-tune, all unfrozen)",
                           opt, pl_module)
        self._current_phase = target

    @staticmethod
    def _announce(epoch, label, opt, pl_module):
        n_train = sum(p.requires_grad for p in pl_module.parameters())
        n_total = sum(1 for _ in pl_module.parameters())
        n_train_numel = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        n_total_numel = sum(p.numel() for p in pl_module.parameters())
        bar = "=" * 72
        print(f"\n{bar}")
        print(f"[{label}] entered at epoch {epoch}")
        print(f"  param_group 0 (head):     lr = {opt.param_groups[0]['lr']:.1e}")
        print(f"  param_group 1 (backbone): lr = {opt.param_groups[1]['lr']:.1e}")
        print(f"  trainable: {n_train}/{n_total} tensors, "
              f"{n_train_numel:,}/{n_total_numel:,} params")
        print(f"{bar}\n")


# ── Model instantiation (mirror forward_test.py:_instantiate_model) ─────────
def instantiate_model() -> CSPDiffusion:
    model_cfg = OmegaConf.load(YAML_PATH)
    # Minimal optim placeholder so save_hyperparameters() has hparams.optim;
    # configure_optimizers is monkey-patched below.
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
        "use_lr_scheduler": False,
    })
    return hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)


def custom_configure_optimizers(self):
    """Override BaseModule.configure_optimizers — 2 param-groups for differential lr.

    Initial state = Phase 1 (head_lr=PHASE1_HEAD_LR, backbone_lr=0).
    PhasedTrainingCallback.on_train_epoch_start updates lrs at phase boundaries.
    """
    head_params, backbone_params = [], []
    for name, p in self.named_parameters():
        (head_params if is_head_param(name) else backbone_params).append(p)
    print(f"[OPTIM] head_params:     {len(head_params):>3} tensors, "
          f"{sum(p.numel() for p in head_params):>10,} params")
    print(f"[OPTIM] backbone_params: {len(backbone_params):>3} tensors, "
          f"{sum(p.numel() for p in backbone_params):>10,} params")
    return torch.optim.AdamW([
        {"params": head_params,     "lr": PHASE1_HEAD_LR, "name": "head"},
        {"params": backbone_params, "lr": 0.0,            "name": "backbone"},
    ], weight_decay=0.0)   # handoff §2.3: weight_decay=0 (Exp4 sustained)


# ── Acceptance gate 3: ckpt warm-start with shape-mismatch pop ──────────────
def warm_start_from_exp4(model: CSPDiffusion, ckpt_path: str):
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"[CKPT WARM-START] {ckpt_path}")
    print(f"{bar}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Pop shape-mismatch keys (PT strict=False raises on size mismatch)
    popped = {}
    for k in SHAPE_MISMATCH_KEYS:
        if k in state_dict:
            popped[k] = tuple(state_dict.pop(k).shape)
    print(f"[CKPT] popped {len(popped)} shape-mismatch key(s):")
    for k, sh in popped.items():
        sh_new = tuple(model.state_dict()[k].shape)
        print(f"  - {k}: Exp4{sh} → Exp5{sh_new} (will appear in `missing`)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"\n[CKPT] missing keys ({len(missing)}):")
    for k in sorted(missing):
        sh = tuple(model.state_dict()[k].shape) if k in model.state_dict() else "?"
        print(f"  - {k}  (Exp5 shape: {sh})")
    print(f"\n[CKPT] unexpected keys ({len(unexpected)}):")
    for k in sorted(unexpected):
        print(f"  - {k}")

    # ── Strict verification (handoff §6 gate 3) ──
    missing_set = set(missing)
    unexpected_missing = missing_set - EXPECTED_MISSING_KEYS
    expected_but_absent = EXPECTED_MISSING_KEYS - missing_set

    if unexpected:
        raise RuntimeError(
            f"\nCKPT WARM-START FAIL: {len(unexpected)} unexpected key(s) in Exp4 ckpt:\n"
            f"  {sorted(unexpected)}\n"
            f"Expected empty list (handoff §5.2). STOP — investigate Exp4 ckpt structure."
        )
    if expected_but_absent:
        raise RuntimeError(
            f"\nCKPT WARM-START FAIL: expected missing keys NOT in `missing`:\n"
            f"  {sorted(expected_but_absent)}\n"
            f"Either Exp4 ckpt was already a partially-Exp5 ckpt, or SA1 architecture "
            f"diverged from handoff §5.2. STOP."
        )
    if unexpected_missing:
        raise RuntimeError(
            f"\nCKPT WARM-START FAIL: extra missing keys (not in expected 7):\n"
            f"  {sorted(unexpected_missing)}\n"
            f"SA1 architecture has unannounced new params. STOP and reconcile."
        )

    print(f"\n[CKPT] WARM-START verification PASSED:")
    print(f"  - 7/7 expected missing keys present (1 center_emb + 4 type_head + 2 atom_latent_emb)")
    print(f"  - 0 unexpected keys (Exp4 ckpt fully consumed)")
    print(f"  - decoder.atom_latent_emb.* shape-mismatch handled via pop+strict=False")
    print(f"{bar}\n")


# ── Acceptance gate 2: OQ-1 sanity scan ──────────────────────────────────────
def oq1_sanity_scan(datamodule: XasLocalDataModuleV2):
    """handoff §3.2: scan train atom_types, assert max < MAX_ATOMIC_NUM=100."""
    bar = "=" * 72
    print(f"\n{bar}")
    print("[OQ-1 sanity] scanning train atom_types...")
    print(f"{bar}")

    max_z = 0
    n_scanned = 0
    for batch in datamodule.train_dataloader():
        if batch is None:
            continue
        bmax = int(batch.atom_types.max())
        if bmax > max_z:
            max_z = bmax
        n_scanned += 1
        if max_z >= 100:
            break

    print(f"[OQ-1 sanity] scanned {n_scanned} batches, max(atom_types) = {max_z}")
    if max_z >= 100:
        raise RuntimeError(
            f"OQ-1 FAIL: max(atom_types)={max_z} >= MAX_ATOMIC_NUM=100. "
            f"yaml `n_elements=100` is insufficient. STOP — report MA. "
            f"Fix: expand n_elements (e.g., 110), reinit head, retrain from scratch."
        )
    print(f"[OQ-1 sanity] PASS (max(atom_types)={max_z} < 100)\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    bar = "=" * 72
    print(f"step4_2_train.py — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DATA_DIR        = {DATA_DIR}")
    print(f"RESUME_FROM_CKPT= {RESUME_FROM}")
    print(f"torch           = {torch.__version__}, cuda = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             = {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
    else:
        raise RuntimeError("CUDA not available — Exp5 training requires GPU")

    # md5 verify Exp4 ckpt
    md5 = hashlib.md5()
    with open(EXP4_CKPT, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    actual_md5 = md5.hexdigest()
    print(f"[CKPT md5] {actual_md5}  expected={EXP4_CKPT_MD5}  "
          f"{'OK' if actual_md5 == EXP4_CKPT_MD5 else 'MISMATCH'}")
    if actual_md5 != EXP4_CKPT_MD5:
        raise RuntimeError(f"Exp4 ckpt md5 mismatch (got {actual_md5}, expected {EXP4_CKPT_MD5})")

    pl.seed_everything(SEED, workers=True)

    # ── DataModule ──
    print(f"\n[DATAMODULE] init from {DATA_DIR}, batch_size={BATCH_SIZE}, "
          f"num_workers={NUM_WORKERS}")
    dm = XasLocalDataModuleV2(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              data_dir=DATA_DIR)
    dm.setup("fit")

    # Gate 2: OQ-1 sanity scan (must pass before model touches data)
    oq1_sanity_scan(dm)

    # ── Model ──
    print("[MODEL] instantiating from yaml...")
    model = instantiate_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {n_params:,} params")
    print(f"[MODEL] type_loss_mode      = {model.type_loss_mode!r}")
    print(f"[MODEL] diffusion_type_w    = {model.diffusion_type_weight}")
    print(f"[MODEL] head_type_w         = {model.head_type_weight}")
    print(f"[MODEL] cost_lattice        = {model.hparams.cost_lattice}")
    print(f"[MODEL] cost_coord          = {model.hparams.cost_coord}")
    print(f"[MODEL] cost_density        = {model.cost_density}")
    print(f"[MODEL] keep_lattice        = {model.keep_lattice}")

    # Override configure_optimizers (monkey-patch instance, NOT class)
    model.configure_optimizers = types.MethodType(custom_configure_optimizers, model)

    # Gate 3: warm-start from Exp4 (raises if anything off)
    warm_start_from_exp4(model, EXP4_CKPT)

    # ── Trainer ──
    os.makedirs(EXP5_CKPT_DIR, exist_ok=True)
    os.makedirs(EXP5_LOG_DIR, exist_ok=True)

    csv_logger = CSVLogger(save_dir=EXP5_LOG_DIR, name="step2_train")

    # NOTE on monitor metric:
    #   SA1 logs `val_loss` = grand total (loss_lattice*0 + loss_coord + loss_type_total
    #                                       + loss_density). Handoff calls this
    #   `val_loss_total` but the actual logged key is `val_loss`. Documenting in OUTPUT.md.
    callbacks = [
        PhasedTrainingCallback(switch_epoch=PHASE_SWITCH_EPOCH),
        ModelCheckpoint(
            dirpath=EXP5_CKPT_DIR,
            filename="best-epoch{epoch:03d}-val_loss{val_loss:.4f}",
            monitor="val_loss",            # grand total — see note above
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOP_PATIENCE,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=[0],
        precision=32,                      # MA4 D1: fp32 (handoff §7 red line)
        gradient_clip_val=GRAD_CLIP_VAL,
        log_every_n_steps=50,
        val_check_interval=1.0,
        callbacks=callbacks,
        logger=csv_logger,
        enable_checkpointing=True,
        deterministic=False,
    )

    print(f"\n{bar}")
    print(f"[TRAINER] starting fit")
    print(f"  max_epochs        = {MAX_EPOCHS}")
    print(f"  batch_size        = {BATCH_SIZE}")
    print(f"  precision         = fp32")
    print(f"  early_stop monitor= val_loss (= grand total; see compute_stats)")
    print(f"  early_stop patience= {EARLY_STOP_PATIENCE}")
    print(f"  phased switch     = epoch {PHASE_SWITCH_EPOCH}")
    print(f"  grad_clip         = {GRAD_CLIP_VAL}")
    print(f"  resume_from       = {RESUME_FROM}")
    print(f"{bar}\n")

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=RESUME_FROM)
    elapsed = time.perf_counter() - t0

    # ── Post-fit summary (gates 4, 5, 7) ──
    print(f"\n{bar}")
    print(f"[TRAINER] fit finished in {elapsed/3600:.2f} hours "
          f"({trainer.current_epoch + 1} epochs)")
    cb = trainer.checkpoint_callback
    print(f"[GATE 7] best ckpt:   {cb.best_model_path}")
    print(f"[GATE 7] best score:  {cb.best_model_score}")
    print(f"[GATE 7] last ckpt:   {cb.last_model_path}")
    print(f"[GATE 7] csv log dir: {csv_logger.log_dir}")

    # Inspect final val_coord_loss for gate 5
    final_metrics = trainer.callback_metrics
    final_val_coord = final_metrics.get("val_coord_loss")
    if final_val_coord is not None:
        v = float(final_val_coord)
        passed = v <= 0.7500
        print(f"[GATE 5] final val_coord_loss = {v:.4f}  "
              f"(threshold ≤ 0.7500: {'PASS' if passed else 'FAIL'})")
        if not passed:
            print(f"[GATE 5] WARN: val_coord_loss exceeded threshold — phased training "
                  f"may have hurt coords. SA2 should investigate before declaring done.")
    else:
        print(f"[GATE 5] WARN: val_coord_loss not in callback_metrics; check csv log")

    print(f"{bar}")
    print("[DONE] SA2 training complete. Next: write EXP5_STEP2_OUTPUT.md (gate 8).")


if __name__ == "__main__":
    main()
