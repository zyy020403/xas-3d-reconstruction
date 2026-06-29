# step3.3_train.py

import os
import sys
import logging
import warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP3_DIR      = os.path.join(EXPERIMENT_DIR, "step3")
STEP1_DIR      = os.path.join(EXPERIMENT_DIR, "step1")
OUTPUT_DIR     = os.path.join(STEP3_DIR, "training_output")
FINETUNE_DIR   = os.path.join(EXPERIMENT_DIR, "step4", "finetune_output")

# ★ 必须在保护块外，import diffcsp 之前就设好
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

for p in [PROJECT_ROOT,
          os.path.join(EXPERIMENT_DIR, "step2"),
          STEP3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    import shutil
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import hydra
    from omegaconf import OmegaConf

    from xas_datamodule import XASDataModule

    os.makedirs(OUTPUT_DIR,   exist_ok=True)
    os.makedirs(FINETUNE_DIR, exist_ok=True)

    # 备份 diffusion.py
    backup_path = os.path.join(STEP3_DIR, "diffusion_backup.py")
    orig_path   = os.path.join(PROJECT_ROOT, "diffcsp", "pl_modules", "diffusion.py")
    if not os.path.exists(backup_path):
        shutil.copy(orig_path, backup_path)
        logger.info(f"diffusion.py 已备份至 {backup_path}")

    # ── 1. 加载模型配置 ────────────────────────────────────────────────────
    CONF_MODEL_DIR = os.path.join(PROJECT_ROOT, "conf", "model")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_MODEL_DIR,
                               job_name="xas_train",
                               version_base=None):
        _raw = compose(config_name="diffusion")

    _raw_dict = OmegaConf.to_container(_raw, resolve=False, throw_on_missing=False)
    full_cfg  = OmegaConf.create({"model": _raw_dict})
    model_cfg = full_cfg.model

    logger.info(f"  model._target_ = {model_cfg._target_}")
    logger.info(f"  latent_dim     = {model_cfg.latent_dim}")
    logger.info(f"  time_dim       = {model_cfg.time_dim}")

    assert model_cfg.latent_dim == 256, (
        f"diffusion.yaml 中 latent_dim={model_cfg.latent_dim}，应为 256。")

    # ── 2. Optim 配置（lr 从 1e-4 降至 1e-5） ─────────────────────────────
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-5},
        "use_lr_scheduler": False,
        "lr_scheduler": None,
    })

    # ── 3. 实例化模型 ──────────────────────────────────────────────────────
    logger.info("实例化 CSPDiffusion...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = None
    model.scaler         = None
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. DataModule（★ 修复：补上 step1_dir） ───────────────────────────
    logger.info("实例化 XASDataModule...")
    datamodule = XASDataModule(
        batch_size  = {"train": 16, "val": 8, "test": 8},
        num_workers = {"train": 0,  "val": 0, "test": 0},
        step1_dir   = STEP1_DIR,     # ★ 修复：原脚本漏掉了这个参数
    )
    datamodule.setup("fit")

    # ── 5. ★ 修复：先确定 ckpt_path，再打印日志（原脚本顺序反了）──────────
   
    RESUME_CKPT = os.path.join(FINETUNE_DIR, "last.ckpt")
    if os.path.exists(RESUME_CKPT):
        _ckpt_path = RESUME_CKPT
        logger.info(f"从 checkpoint 续训: {RESUME_CKPT}")
    else:
        _ckpt_path = None
        logger.info("未找到已有 checkpoint，从头开始训练")

    # ── 6. Callbacks ──────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=FINETUNE_DIR,
        filename="epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss", mode="min",
        save_top_k=3, save_last=True, verbose=True,
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    # ── 7. Logger & Trainer ───────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=OUTPUT_DIR, name="logs")

    trainer = pl.Trainer(
        default_root_dir=OUTPUT_DIR,
        logger=csv_logger,
        callbacks=[checkpoint_cb, lr_monitor_cb, early_stop_cb],
        precision="bf16",
        devices=1, accelerator="gpu",
        gradient_clip_val=1.0,
        max_epochs=400,
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
    )

    # ── 8. 启动训练 ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  checkpoint 输出目录 : {FINETUNE_DIR}")
    logger.info(f"  续训 checkpoint     : {_ckpt_path or '无（从头开始）'}")
    logger.info(f"  batch_size(train)   : {datamodule.batch_size['train']}")
    logger.info(f"  max_epochs          : {trainer.max_epochs}")
    logger.info("=" * 60)

    torch.set_float32_matmul_precision('medium')
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=_ckpt_path)

    logger.info("训练完成。")
    logger.info(f"最优 checkpoint : {checkpoint_cb.best_model_path}")
    if checkpoint_cb.best_model_score is not None:
        logger.info(f"最优 val_loss   : {checkpoint_cb.best_model_score:.6f}")