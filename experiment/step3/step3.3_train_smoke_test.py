# step3.3_train_smoke_test.py

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP3_DIR      = os.path.join(EXPERIMENT_DIR, "step3")
SMOKE_OUTPUT   = os.path.join(STEP3_DIR, "smoke_output")

# ★ 修复：必须在任何 diffcsp import 之前设置环境变量
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

for p in [PROJECT_ROOT,
          os.path.join(EXPERIMENT_DIR, "step2"),
          STEP3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    os.makedirs(SMOKE_OUTPUT, exist_ok=True)

    import shutil, torch, pytorch_lightning as pl
    import pandas as pd
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import hydra
    from omegaconf import OmegaConf

    from xas_dataset import XASCrystalDataset, xas_collate_fn

    # ── 1. 加载配置 ────────────────────────────────────────────────────────
    CONF_MODEL_DIR = os.path.join(PROJECT_ROOT, "conf", "model")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_MODEL_DIR,
                               job_name="smoke_test",
                               version_base=None):
        _raw = compose(config_name="diffusion")

    full_cfg  = OmegaConf.create({"model": OmegaConf.to_container(_raw, resolve=False)})
    model_cfg = full_cfg.model
    logger.info(f"latent_dim={model_cfg.latent_dim}  time_dim={model_cfg.time_dim}")

    # ── 2. 构建 optim 配置（lr=1e-5，无 scheduler） ────────────────────────
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-5},
        "use_lr_scheduler": False,
        "lr_scheduler": None,
    })

    # ── 3. 实例化模型 ──────────────────────────────────────────────────────
    logger.info("实例化模型...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = None
    model.scaler         = None
    logger.info(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. 小数据集：只取前 50 train / 前 20 val ──────────────────────────
    STEP1_DIR    = os.path.join(EXPERIMENT_DIR, "step1")
    inventory_df = pd.read_csv(os.path.join(STEP1_DIR, "data_inventory.csv"))
    holdout_ids  = set()
    holdout_path = os.path.join(STEP1_DIR, "holdout_1000_ids.txt")
    if os.path.exists(holdout_path):
        with open(holdout_path) as f:
            holdout_ids = {int(l.strip()) for l in f if l.strip()}

    def _read_ids(fname, limit):
        path = os.path.join(STEP1_DIR, fname)
        with open(path) as f:
            ids = [int(l.strip()) for l in f if l.strip()]
        return [i for i in ids if i not in holdout_ids][:limit]

    TRAIN_N = 50
    VAL_N   = 20

    logger.info(f"smoke test 规模: train={TRAIN_N}  val={VAL_N}")
    train_ids = _read_ids("train_ids.txt", TRAIN_N)
    val_ids   = _read_ids("val_ids.txt",   VAL_N)

    logger.info("加载 train dataset...")
    train_ds = XASCrystalDataset(train_ids, inventory_df)
    logger.info("加载 val dataset...")
    val_ds   = XASCrystalDataset(val_ids,   inventory_df)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=xas_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              collate_fn=xas_collate_fn, num_workers=0)

    # ── 5. 快速验证第一个 batch 形状 ──────────────────────────────────────
    logger.info("--- batch 形状检查 ---")
    sample_batch = next(iter(train_loader))
    logger.info(f"  num_graphs    : {sample_batch.num_graphs}")
    logger.info(f"  spectra shape : {sample_batch.spectra.shape}")
    logger.info(f"  n_sites       : {sample_batch.n_sites.tolist()}")
    logger.info(f"  frac_coords   : {sample_batch.frac_coords.shape}")
    logger.info(f"  atom_types    : {sample_batch.atom_types.shape}")
    logger.info("形状检查完成")

    # ── 6. Trainer（3 epoch） ─────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=SMOKE_OUTPUT,
        filename="smoke-epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=SMOKE_OUTPUT,
        logger=CSVLogger(save_dir=SMOKE_OUTPUT, name="logs"),
        callbacks=[ckpt_cb, lr_cb],
        precision="bf16",
        devices=1, accelerator="gpu",
        gradient_clip_val=1.0,
        max_epochs=3,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    # ── 7. 开始训练 ───────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("开始 smoke test（3 epoch）")
    logger.info("=" * 50)
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # ── 8. 结果汇报 ───────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("smoke test 完成")
    best_score = ckpt_cb.best_model_score
    if best_score is not None:
        logger.info(f"最优 val_loss : {best_score:.6f}")
        logger.info("✅ 全链路验证通过，可以运行正式训练脚本")
    else:
        logger.info("⚠️  未记录到 val_loss，请检查 validation_step 输出")