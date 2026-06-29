"""
step4_2_train.py — Exp5 v2 SA1' 训练入口 (from-scratch)
=================================================================

Run from /home/tcat/diffcsp_exp5/code/step4/ with mlff env active.

启动命令 (SA2' 用):

    cd /home/tcat/diffcsp_exp5/code/step4
    PYTHONPATH=/home/tcat/diffcsp_exp5/code/step3:/home/tcat/diffcsp_exp5/code/step2:/home/tcat/diffcsp_exp4/code \
    EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
    nohup /home/tcat/conda_envs/mlff/bin/python step4_2_train.py \
        > /home/tcat/diffcsp_exp5/logs/step4_train_v2_stdout.log \
        2> /home/tcat/diffcsp_exp5/logs/step4_train_v2_stderr.log &

SA2' 启动后:
    1. 头 30 min 守屏看 val_loss 初始合理 (random-init ≈ 2-4)
    2. 关 ssh, 等 ~ 32h 训练 (500 epoch + early_stop patience=30)
    3. best ckpt 落到 /home/tcat/diffcsp_exp5/checkpoints/

关键配置 (handoff §1.1):
    MAX_EPOCHS=500, batch_size=16, LR=1e-4
    early_stop patience=30, save_top_k=1
    precision=32 (fp32, MA4 D1)
    num_workers=0 (pymatgen SGA worker safety)
    monitor='val_loss'

红线 (handoff §2):
    - 不 fine-tune from Exp4 ckpt (decoder shape 528 vs 512 mismatch + v2 from-scratch)
    - 不动 holdout
    - 训练监控只 log val_loss 等 Exp4 baseline 4-loss
      (Set-Level / Multiset / Collapse 在 step5_2 sample 后算 — handoff §6.6 E
       "推荐: 训练监控用简化版" 但 forward 不出 per-sample type 预测,
       简化版 inline 实现成本/价值不划算; SA1' 决定不加,SA3 期 step5_2 算)
"""

import os
import sys
import logging
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# ── PYTHONPATH self-check (handoff §1.2 + §6.6 D, carry-over from v1 SA1 §5.6) ──
EXP5_ROOT     = "/home/tcat/diffcsp_exp5"
EXP5_STEP3    = f"{EXP5_ROOT}/code/step3"
EXP5_STEP2    = f"{EXP5_ROOT}/code/step2"
EXP4_BACKBONE = "/home/tcat/diffcsp_exp4/code"

for p in (EXP5_STEP2, EXP5_STEP3):
    if p not in sys.path:
        sys.path.insert(0, p)
if EXP4_BACKBONE not in sys.path:
    sys.path.append(EXP4_BACKBONE)

# Verify we're loading Exp5 versions (anti-shadowing check)
import diffusion_w_type_xas    # noqa: E402
import spectrum_encoder        # noqa: E402
assert "/diffcsp_exp5/" in diffusion_w_type_xas.__file__, \
    f"WRONG diffusion_w_type_xas: {diffusion_w_type_xas.__file__} (expect /diffcsp_exp5/...)"
assert "/diffcsp_exp5/" in spectrum_encoder.__file__, \
    f"WRONG spectrum_encoder: {spectrum_encoder.__file__} (expect /diffcsp_exp5/...)"
print(f"[PYTHONPATH check] diffusion_w_type_xas: {diffusion_w_type_xas.__file__}")
print(f"[PYTHONPATH check] spectrum_encoder:     {spectrum_encoder.__file__}")

# ── Project paths ──────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("EXP4_DATA_DIR", f"{EXP5_ROOT}/data")
CKPT_DIR    = f"{EXP5_ROOT}/checkpoints"
LOG_DIR     = f"{EXP5_ROOT}/logs"
CONF_DIR    = f"{EXP5_STEP3}/conf_xas"

os.environ["PROJECT_ROOT"] = EXP4_BACKBONE   # diffcsp.common.utils.PROJECT_ROOT
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Hyperparameters (handoff §1.1) ────────────────────────────────────────
MAX_EPOCHS     = 500
BATCH_SIZE     = 16
LR             = 1e-4
GRADIENT_CLIP  = 1.0
EARLY_STOP_PAT = 30           # Exp4 standard
SAVE_TOP_K     = 1            # Exp4 standard
PRECISION      = 32           # fp32 (MA4 D1, NOT bf16-mixed)
NUM_WORKERS    = 0            # pymatgen SGA worker safety
L              = 6.0          # Exp2 step4d coord box edge

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

    from xas_local_datamodule_v2 import XasLocalDataModuleV2

    logger.info("=" * 60)
    logger.info("Step 4.2  Exp5 v2 SA1'  正式训练 (from-scratch)")
    logger.info("=" * 60)
    logger.info(f"  DATA_DIR    : {DATA_DIR}")
    logger.info(f"  CKPT_DIR    : {CKPT_DIR}")
    logger.info(f"  LOG_DIR     : {LOG_DIR}")
    logger.info(f"  L           : {L}")
    logger.info(f"  precision   : {PRECISION} (fp32, MA4 D1)")
    logger.info(f"  batch_size  : {BATCH_SIZE}")
    logger.info(f"  max_epochs  : {MAX_EPOCHS}")
    logger.info(f"  early_stop  : patience={EARLY_STOP_PAT}")
    logger.info(f"  save_top_k  : {SAVE_TOP_K}")
    logger.info(f"  monitor     : val_loss")

    # ── 1. 加载模型配置 ────────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="train", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    logger.info(f"  cost_lattice  = {model_cfg.cost_lattice}")
    logger.info(f"  cost_coord    = {model_cfg.cost_coord}")
    logger.info(f"  cost_type     = {model_cfg.cost_type}")
    logger.info(f"  cost_density  = {model_cfg.cost_density}  (Exp5 v2: 0.2)")
    logger.info(f"  latent_dim    = {model_cfg.latent_dim}    (= 256 + 16 = 272)")
    logger.info(f"  decoder.latent_dim = {model_cfg.decoder.latent_dim}  (= 256 + 272 = 528)")
    logger.info(f"  mv_attention  = num_heads={model_cfg.mv_attention.num_heads}, "
                f"residual_alpha={model_cfg.mv_attention.residual_alpha}")
    logger.info(f"  n_center_elements = {model_cfg.n_center_elements}")

    # 防御性检查
    assert float(model_cfg.cost_lattice) < 1e-5, \
        f"cost_lattice={model_cfg.cost_lattice} != 0!  停止训练。"
    assert abs(float(model_cfg.cost_density) - 0.2) < 1e-6, \
        f"cost_density={model_cfg.cost_density} != 0.2!  Exp5 v2 主线 2 配置错误。"
    assert int(model_cfg.mv_attention.num_heads) == 4, \
        f"mv_attention.num_heads={model_cfg.mv_attention.num_heads} != 4!"
    assert abs(float(model_cfg.mv_attention.residual_alpha) - 0.5) < 1e-9, \
        f"mv_attention.residual_alpha={model_cfg.mv_attention.residual_alpha} != 0.5!"
    assert int(model_cfg.decoder.latent_dim) == 528, \
        f"decoder.latent_dim={model_cfg.decoder.latent_dim} != 528 (time 256 + spectrum 272)"

    # ── 2. 实例化模型 ──────────────────────────────────────────────────────
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

    logger.info("实例化 CSPDiffusion (Exp5 v2: MV-attention + center conditioning)...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  参数量 = {n_params:,}  keep_lattice = {model.keep_lattice}  "
                f"cost_density = {model.cost_density}")

    # 防御性检查 (撤销 v1 head)
    forbidden = ['type_head', 'type_loss_mode', 'diffusion_type_weight',
                 'head_type_weight', 'head_predict_types']
    for attr in forbidden:
        if hasattr(model, attr):
            raise RuntimeError(f"v2 model still has v1 head attribute: {attr}")
    logger.info(f"  ✓ no v1 head attributes ({', '.join(forbidden)} all absent)")

    # 防御性检查 (MV-attention)
    for attr in ['mv_attn', 'mv_query', 'mv_layernorm', 'mv_proj', 'center_emb']:
        if not hasattr(model.spectrum_encoder, attr):
            raise RuntimeError(f"SpectrumEncoder missing MV-attention attribute: {attr}")
    if hasattr(model.spectrum_encoder, 'fusion'):
        raise RuntimeError("SpectrumEncoder still has v1 fusion block")
    logger.info(f"  ✓ MV-attention components present, v1 fusion removed")

    # ── 3. DataModule ──────────────────────────────────────────────────────
    logger.info("初始化 XasLocalDataModuleV2...")
    datamodule = XasLocalDataModuleV2(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_dir=DATA_DIR,
    )
    datamodule.setup("fit")
    train_size = len(datamodule.train_dataset)
    val_size   = len(datamodule.val_dataset)
    logger.info(f"  train = {train_size}  val = {val_size}")

    # ── 4. 续训检测 (v2 自身的 last.ckpt, 不是 Exp4 warm-start) ──────────
    last_ckpt = os.path.join(CKPT_DIR, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    if ckpt_path:
        logger.info(f"  续训 checkpoint (v2 自身): {ckpt_path}")
        logger.warning("  ⚠️  CKPT_DIR 中检测到 last.ckpt — 确认是 Exp5 v2 自身的 ckpt,")
        logger.warning("      不是 Exp4 遗留 (decoder shape 528 vs 512 不兼容)。")
        logger.warning(f"      若不确定: rm -f {last_ckpt} 后重启脚本。")
    else:
        logger.info("  未找到 checkpoint, 从头开始训练 (from-scratch, NOT warm-start)")

    # ── 5. Callbacks ──────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath    = CKPT_DIR,
        filename   = "epoch={epoch:03d}-val_loss={val_loss:.4f}",
        monitor    = "val_loss",
        save_top_k = SAVE_TOP_K,
        mode       = "min",
        save_last  = True,
        verbose    = True,
        auto_insert_metric_name = False,
    )
    lr_cb    = LearningRateMonitor(logging_interval="epoch")
    early_cb = EarlyStopping(
        monitor   = "val_loss",
        patience  = EARLY_STOP_PAT,
        mode      = "min",
        verbose   = True,
    )

    # ── 6. Logger ──────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="xas_diffusion_v2")

    # ── 7. Trainer ─────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        default_root_dir        = EXP5_ROOT,
        logger                  = csv_logger,
        callbacks               = [ckpt_cb, lr_cb, early_cb],
        precision               = PRECISION,           # 32 = fp32 (MA4 D1)
        devices                 = 1,
        accelerator             = "gpu",
        gradient_clip_val       = GRADIENT_CLIP,
        max_epochs              = MAX_EPOCHS,
        check_val_every_n_epoch = 5,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    logger.info("")
    logger.info("收敛判断参考:")
    logger.info("  val_loss 持续下降 → 继续")
    logger.info("  val_loss ≈ 1.0 震荡 30 epoch → early stop 触发 (正常)")
    logger.info("  val_loss > 4.0 且 30 epoch 无改善 → 停止, 汇报 MA5")
    logger.info("  RMSD < 1.5 Å + Multiset Macro-F1 > 0.05 (Step5) 才是真实有效的判断")
    logger.info("=" * 60)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("训练完成。")
    logger.info(f"最优 checkpoint : {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        logger.info(f"最优 val_loss   : {ckpt_cb.best_model_score:.6f}")

    # 保存最优 ckpt 路径供 step5 使用
    best_path_file = os.path.join(EXP5_ROOT, "best_checkpoint_path.txt")
    with open(best_path_file, "w") as f:
        f.write(ckpt_cb.best_model_path)
    logger.info(f"最优路径已写入 → {best_path_file}")
