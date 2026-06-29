# step4c_2_train.py
# Step4c formal training script
# ============================================================
# Differences from step4_2_train.py:
#   - STEP4_DIR points to experiment2/step4c
#   - CKPT_DIR  points to step4c/checkpoints
#   - early_stop patience = 30 (per handoff spec)
#   - Requires diffusion_w_type_xas.py v3 and xas_local_dataset.py v5
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4c_DIR   = os.path.join(EXP2_ROOT, "step4c")
CKPT_DIR     = os.path.join(STEP4c_DIR, "checkpoints")
LOG_DIR      = os.path.join(STEP4c_DIR, "logs")
CONF_DIR     = os.path.join(STEP3_DIR, "conf_xas")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

MAX_EPOCHS     = 500
BATCH_SIZE     = 16
LR             = 1e-4
GRADIENT_CLIP  = 1.0
EARLY_STOP_PAT = 30
PRECISION      = 'bf16'
NUM_WORKERS    = 0

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import torch
    import hydra
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, LearningRateMonitor, EarlyStopping)
    from pytorch_lightning.loggers import CSVLogger

    from xas_local_datamodule import XASDataModule

    logger.info("=" * 60)
    logger.info("Step4c formal training")
    logger.info("Coordinate system: [-0.5, 0.5]")
    logger.info("diffusion v3 + dataset v5")
    logger.info("=" * 60)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="train4c", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    logger.info(f"  cost_lattice={model_cfg.cost_lattice}  "
                f"latent_dim={model_cfg.latent_dim}  "
                f"time_dim={model_cfg.time_dim}")

    assert float(model_cfg.cost_lattice) < 1e-5, \
        f"cost_lattice={model_cfg.cost_lattice} != 0, stopping."

    optim_cfg = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": LR,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
        },
        "use_lr_scheduler": True,
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": MAX_EPOCHS,
            "eta_min": 1e-6,
        },
    })

    logger.info("Instantiating CSPDiffusion...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  params={n_params:,}  keep_lattice={model.keep_lattice}")

    logger.info("Initializing XASDataModule...")
    datamodule = XASDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, L=12.0)
    datamodule.setup("fit")
    logger.info(f"  train={len(datamodule.train_dataset)}  "
                f"val={len(datamodule.val_dataset)}")

    last_ckpt = os.path.join(CKPT_DIR, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    if ckpt_path:
        logger.info(f"  Resuming from: {ckpt_path}")
    else:
        logger.info("  No checkpoint found, training from scratch")

    ckpt_cb = ModelCheckpoint(
        dirpath    = CKPT_DIR,
        filename   = "epoch={epoch:03d}-val_loss={val_loss:.4f}",
        monitor    = "val_loss",
        save_top_k = 3,
        mode       = "min",
        save_last  = True,
        verbose    = True,
        auto_insert_metric_name = False,
    )
    lr_cb    = LearningRateMonitor(logging_interval="epoch")
    early_cb = EarlyStopping(
        monitor  = "val_loss",
        patience = EARLY_STOP_PAT,
        mode     = "min",
        verbose  = True,
    )

    csv_logger = CSVLogger(save_dir=LOG_DIR, name="xas_diffusion")

    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        default_root_dir          = STEP4c_DIR,
        logger                    = csv_logger,
        callbacks                 = [ckpt_cb, lr_cb, early_cb],
        precision                 = PRECISION,
        devices                   = 1,
        accelerator               = "gpu",
        gradient_clip_val         = GRADIENT_CLIP,
        max_epochs                = MAX_EPOCHS,
        check_val_every_n_epoch   = 5,
        log_every_n_steps         = 10,
        enable_progress_bar       = True,
    )

    logger.info("=" * 60)
    logger.info("Training config")
    logger.info(f"  CKPT dir      : {CKPT_DIR}")
    logger.info(f"  batch_size    : {BATCH_SIZE}")
    logger.info(f"  max_epochs    : {MAX_EPOCHS}")
    logger.info(f"  early_stop    : patience={EARLY_STOP_PAT}")
    logger.info(f"  precision     : {PRECISION}")
    logger.info("=" * 60)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("Training complete.")
    logger.info(f"Best checkpoint : {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        logger.info(f"Best val_loss   : {ckpt_cb.best_model_score:.6f}")

    best_path_file = os.path.join(STEP4c_DIR, "best_checkpoint_path.txt")
    with open(best_path_file, "w") as f:
        f.write(ckpt_cb.best_model_path)
    logger.info(f"Best path written to: {best_path_file}")
