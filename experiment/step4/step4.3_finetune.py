# =============================================================================
# 脚本编号: step4.3
# 脚本名称: step4.3_finetune.py
# 输入:
#   - experiment/step4/finetune_output/epochepoch=264-valval_loss=0.9207.ckpt
# 输出:
#   - experiment/step4/finetune2_output/  (新一轮 fine-tune checkpoint)
#   - experiment/step4/finetune2_output/val_metrics.csv
# =============================================================================

import os
import sys
import shutil

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP3_DIR  = os.path.join(EXPERIMENT_DIR, "step3")
STEP4_DIR  = os.path.join(EXPERIMENT_DIR, "step4")
STEP1_DIR  = os.path.join(EXPERIMENT_DIR, "step1")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STEP3_DIR)

OUTPUT_DIR = os.path.join(STEP4_DIR, "finetune2_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_RESUME = os.path.join(
    STEP4_DIR, "finetune_output",
    "epochepoch=264-valval_loss=0.9207.ckpt"
)

DIFFUSION_PY = os.path.join(PROJECT_ROOT, "diffcsp", "pl_modules", "diffusion.py")
BACKUP_STEP3 = os.path.join(PROJECT_ROOT, "diffcsp", "pl_modules", "diffusion_backup_step3.py")
BACKUP_STEP4 = os.path.join(PROJECT_ROOT, "diffcsp", "pl_modules", "diffusion_backup_step4.py")

# ─── 备份检查 ─────────────────────────────────────────────────────────────────
print("[Backup] Checking backup files ...")
if not os.path.exists(BACKUP_STEP3):
    print(f"  WARNING: {BACKUP_STEP3} not found.")
    print("  Creating backup of current diffusion.py as diffusion_backup_step3.py ...")
    shutil.copy2(DIFFUSION_PY, BACKUP_STEP3)
    print("  Backup created.")
else:
    print(f"  OK: diffusion_backup_step3.py exists.")

shutil.copy2(DIFFUSION_PY, BACKUP_STEP4)
print(f"  Created: diffusion_backup_step4.py (pre-modification snapshot)")

# ─── 修改 configure_optimizers ───────────────────────────────────────────────
print("\n[Patch] Patching configure_optimizers in diffusion.py ...")

with open(DIFFUSION_PY, "r", encoding="utf-8") as f:
    source = f.read()

if "CosineAnnealingLR" in source:
    print("  Already patched (CosineAnnealingLR found). Skipping patch.")
else:
    OLD_RETURN = "        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}"

    NEW_RETURN = """        # ── Step4.3 patch: CosineAnnealingLR ──────────────────────────
        import torch.optim.lr_scheduler as lr_sched
        scheduler = lr_sched.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }"""

    if OLD_RETURN in source:
        source = source.replace(OLD_RETURN, NEW_RETURN)
        print("  Found standard return pattern, patched successfully.")
    else:
        import re
        pattern = r"(    def configure_optimizers\(self\).*?)(        return optimizer)"
        match = re.search(pattern, source, re.DOTALL)
        if match:
            old_block = match.group(0)
            new_block = old_block.replace(
                "        return optimizer",
                """        # ── Step4.3 patch: CosineAnnealingLR ──────────────────────────
        import torch.optim.lr_scheduler as lr_sched
        scheduler = lr_sched.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }"""
            )
            source = source.replace(old_block, new_block)
            print("  Found 'return optimizer' pattern, patched successfully.")
        else:
            print("  WARNING: Could not auto-patch configure_optimizers.")
            print("  Continuing without patch (scheduler won't be active).")

    with open(DIFFUSION_PY, "w", encoding="utf-8") as f:
        f.write(source)
    print("  diffusion.py written.")

# ─── 训练 ─────────────────────────────────────────────────────────────────────
print("\n[Train] Starting fine-tune from epoch=264 ...")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)

torch.set_float32_matmul_precision("high")

from xas_datamodule import XASDataModule
from diffcsp.pl_modules.diffusion import CSPDiffusion

print(f"[Model] Loading checkpoint: {CKPT_RESUME}")
model = CSPDiffusion.load_from_checkpoint(
    CKPT_RESUME,
    map_location="cpu",
    strict=False,
)
print("[Model] Loaded.")

datamodule = XASDataModule(
    batch_size  = {"train": 16, "val": 16, "test": 16},
    num_workers = {"train": 0,  "val": 0,  "test": 0},
    step1_dir   = STEP1_DIR,
)

checkpoint_cb = ModelCheckpoint(
    dirpath   = OUTPUT_DIR,
    filename  = "{epoch:03d}-{val_loss:.4f}",
    monitor   = "val_loss",
    save_top_k= 3,
    mode      = "min",
    save_last = True,
)

# ── 修复：check_on_train_epoch_end=False，在 validation 结束后才检查 val_loss ──
early_stop_cb = EarlyStopping(
    monitor                  = "val_loss",
    patience                 = 30,
    mode                     = "min",
    verbose                  = True,
    check_on_train_epoch_end = False,
)

lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

from pytorch_lightning.loggers import CSVLogger
csv_logger = CSVLogger(
    save_dir = OUTPUT_DIR,
    name     = "logs",
)

trainer = pl.Trainer(
    max_epochs         = 400,
    gradient_clip_val  = 1.0,
    precision          = "bf16",
    devices            = 1,
    accelerator        = "gpu",
    callbacks          = [checkpoint_cb, early_stop_cb, lr_monitor_cb],
    logger             = csv_logger,
    log_every_n_steps  = 10,
    enable_progress_bar= True,
)

print("[Train] Trainer configured. Starting fit ...")
print(f"  Resume from: {CKPT_RESUME}")
print(f"  Output dir : {OUTPUT_DIR}")
print(f"  Max epochs : 400 (cosine T_max=100, eta_min=1e-6)")
print(f"  EarlyStopping patience=30")

trainer.fit(
    model,
    datamodule = datamodule,
    ckpt_path  = CKPT_RESUME,
)

print("\n[Done] Fine-tune complete.")
print(f"  Best checkpoint: {checkpoint_cb.best_model_path}")
print(f"  Best val_loss  : {checkpoint_cb.best_model_score:.4f}")