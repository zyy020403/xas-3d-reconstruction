# step4_5_finetune.py
# Step 4.5 — 追加训练（密度正则化）
# ============================================================
# 从最优 checkpoint 续训，使用新增密度正则损失
# loss = cost_coord * loss_coord + cost_type * loss_type
#      + cost_density * loss_density   ← ★ 新增
#
# 修改文件：
#   diffusion_w_type_xas.py → v2（已更新，加 loss_density）
#   diffusion_xas.yaml      → 加 cost_density: 0.5
#
# 输出目录：experiment2/step4/checkpoints_v2/
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message="xmu.dat 能量窗口超出数据范围")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4_DIR    = os.path.join(EXP2_ROOT, "step4")
CKPT_DIR_V1  = os.path.join(STEP4_DIR, "checkpoints")
CKPT_DIR_V2  = os.path.join(STEP4_DIR, "checkpoints_v2")
LOG_DIR      = os.path.join(STEP4_DIR, "logs_v2")
CONF_DIR     = os.path.join(STEP3_DIR, "conf_xas")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

os.makedirs(CKPT_DIR_V2, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

# 追加训练超参（相对保守，防止遗忘）
MAX_EPOCHS     = 300
BATCH_SIZE     = 16
LR             = 3e-5     # 比初训低，避免破坏已有特征
GRADIENT_CLIP  = 1.0
EARLY_STOP_PAT = 40
PRECISION = "bf16"   # 原来是 "bf16-mixed"
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
    logger.info("Step 4.5  追加训练（原子密度正则化）")
    logger.info("=" * 60)

    # ── 1. 确定起点 checkpoint ─────────────────────────────────────────────
    best_path_file = os.path.join(STEP4_DIR, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file) as f:
            resume_ckpt = f.read().strip()
    else:
        import glob
        ckpts = glob.glob(os.path.join(CKPT_DIR_V1, "epoch=*.ckpt"))
        if not ckpts:
            logger.error("❌ 找不到 v1 checkpoint！请先完成 step4_2_train.py")
            sys.exit(1)
        def _loss(p):
            import re
            m = re.search(r'val_loss=([\d.]+)', os.path.basename(p))
            return float(m.group(1)) if m else 9999.0
        resume_ckpt = min(ckpts, key=_loss)

    logger.info(f"起点 checkpoint：{resume_ckpt}")

    # ── 2. 加载 YAML（含新增 cost_density）────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="finetune", version_base=None):
        _raw = compose(config_name="diffusion_xas")

    _raw_dict = OmegaConf.to_container(_raw, resolve=False, throw_on_missing=False)

    # 注入 cost_density（若 YAML 已有则覆盖，若无则新增）
    COST_DENSITY = 0.5
    _raw_dict['cost_density'] = COST_DENSITY
    logger.info(f"cost_density = {COST_DENSITY}")

    model_cfg = OmegaConf.create({"model": _raw_dict}).model

    assert float(model_cfg.cost_lattice) < 1e-5, "cost_lattice 必须为 0！"

    # ── 3. 实例化模型（v2，带 density loss）────────────────────────────────
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

    logger.info("实例化 CSPDiffusion v2（含密度正则）...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    # 从 v1 checkpoint 加载权重（strict=False，因为模型结构未变，只有 loss 新增）
    ckpt_data = torch.load(resume_ckpt, map_location="cpu")
    state = ckpt_data.get("state_dict", ckpt_data)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"  unexpected keys: {unexpected[:5]}")
    logger.info(f"  权重加载成功  keep_lattice={model.keep_lattice}  "
                f"cost_density={model.cost_density}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  参数量：{n_params:,}")

    # ── 4. 快速验证 forward（cost_density 是否生效）────────────────────────
    logger.info("验证 forward pass 含 loss_density...")
    dm_check = XASDataModule(batch_size=4, num_workers=0, L=12.0)
    dm_check.setup("fit")
    from torch_geometric.data import Batch
    mini = None
    for b in dm_check.val_dataloader():
        if b is not None and b.num_graphs >= 4:
            mini = Batch.from_data_list(b.to_data_list()[:4]); break
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_tmp = model.to(device)
    mini = mini.to(device)
    with torch.no_grad():
        out = model_tmp(mini)
    logger.info(f"  loss={out['loss']:.4f}  "
                f"coord={out['loss_coord']:.4f}  "
                f"type={out['loss_type']:.4f}  "
                f"density={out['loss_density']:.4f}")
    assert 'loss_density' in out, "❌ loss_density 未出现在输出中！"
    logger.info("  ✅ density loss 已激活")

    # ── 5. DataModule ──────────────────────────────────────────────────────
    datamodule = XASDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, L=12.0)
    datamodule.setup("fit")
    logger.info(f"  train={len(datamodule.train_dataset)}  "
                f"val={len(datamodule.val_dataset)}")

    # ── 6. 续训检测（step4_5 自身是否有断点）────────────────────────────────
    last_v2 = os.path.join(CKPT_DIR_V2, "last.ckpt")
    ckpt_path = last_v2 if os.path.exists(last_v2) else None
    if ckpt_path:
        logger.info(f"  从 v2 断点续训：{ckpt_path}")
    else:
        logger.info("  从 v1 最优权重开始密度正则化微调")

    # ── 7. Callbacks ──────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath    = CKPT_DIR_V2,
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
        monitor="val_loss", patience=EARLY_STOP_PAT, mode="min", verbose=True)

    csv_logger = CSVLogger(save_dir=LOG_DIR, name="finetune")

    # ── 8. Trainer ─────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        default_root_dir      = STEP4_DIR,
        logger                = csv_logger,
        callbacks             = [ckpt_cb, lr_cb, early_cb],
        precision             = PRECISION,
        devices               = 1,
        accelerator           = "gpu",
        gradient_clip_val     = GRADIENT_CLIP,
        max_epochs            = MAX_EPOCHS,
        check_val_every_n_epoch = 5,
        log_every_n_steps     = 10,
        enable_progress_bar   = True,
    )

    logger.info("=" * 60)
    logger.info("开始追加训练")
    logger.info(f"  起点权重    : {resume_ckpt}")
    logger.info(f"  CKPT 目录   : {CKPT_DIR_V2}")
    logger.info(f"  LR          : {LR}")
    logger.info(f"  cost_density: {COST_DENSITY}")
    logger.info(f"  max_epochs  : {MAX_EPOCHS}")
    logger.info(f"  early_stop  : patience={EARLY_STOP_PAT}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("收敛判断参考：")
    logger.info("  val_loss 比 v1 最优（0.6178）略升后趋于稳定 → 正常（density loss 拉高了总 loss）")
    logger.info("  density_loss 持续下降 → 原子集中效果在提升")
    logger.info("  若 density_loss 不下降 → 调低 cost_density 至 0.2 重试")
    logger.info("=" * 60)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("追加训练完成。")
    logger.info(f"最优 checkpoint : {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        logger.info(f"最优 val_loss   : {ckpt_cb.best_model_score:.6f}")

    best_v2_file = os.path.join(STEP4_DIR, "best_checkpoint_v2_path.txt")
    with open(best_v2_file, "w") as f:
        f.write(ckpt_cb.best_model_path)
    logger.info(f"最优路径已写入 → {best_v2_file}")