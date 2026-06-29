# step4d_1_quick_test.py
# Step4d — 30-epoch 快速验证
# ============================================================
# Step4c agent 提示（坑 5）：
#   L=6 是否有效，30 epoch 内即可判断。
#   epoch0 随机基线 RMSD ≈ 2.32 Å（L=6 的 [-3,3]³ 期望值）
#   epoch30 RMSD 降到 1.8 Å 以下 → 方向正确，可以开 500epoch 正式训练
#   epoch30 RMSD ≈ 2.3 Å（≈随机基线） → 立即汇报，不要跑 500epoch
#
# 本脚本输出：
#   experiment2/step4d/quick_test/checkpoints/
#   experiment2/step4d/quick_test/predictions_val_ep30.pt
#   experiment2/step4d/quick_test/quick_test_rmsd.txt
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT     = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR     = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR     = os.path.join(EXP2_ROOT, "step3")
STEP4d_DIR    = os.path.join(EXP2_ROOT, "step4d")
QT_DIR        = os.path.join(STEP4d_DIR, "quick_test")
QT_CKPT_DIR   = os.path.join(QT_DIR, "checkpoints")
CONF_DIR      = os.path.join(STEP3_DIR, "conf_xas")
L             = 6.0
QUICK_EPOCHS  = 30

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

os.makedirs(QT_CKPT_DIR, exist_ok=True)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import torch, numpy as np
    import hydra, pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    from tqdm import tqdm

    from xas_local_datamodule import XASDataModule
    from scipy.optimize import linear_sum_assignment

    logger.info("=" * 60)
    logger.info(f"Step4d 快速测试：{QUICK_EPOCHS} epochs，L={L}Å")
    logger.info("=" * 60)

    # ── 模型 ──────────────────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="quick_test", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4,
                      "betas": [0.9, 0.999], "weight_decay": 0.0},
        "use_lr_scheduler": True,
        "lr_scheduler": {"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                         "T_max": QUICK_EPOCHS, "eta_min": 1e-6},
    })

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    # ── DataModule ────────────────────────────────────────────────────────
    datamodule = XASDataModule(batch_size=16, num_workers=0, L=L)
    datamodule.setup("fit")
    logger.info(f"  train={len(datamodule.train_dataset)}  "
                f"val={len(datamodule.val_dataset)}")

    # ── 训练 30 epoch ─────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath    = QT_CKPT_DIR,
        filename   = "ep{epoch:03d}-val{val_loss:.4f}",
        monitor    = "val_loss",
        save_top_k = 1,
        mode       = "min",
        save_last  = True,
        auto_insert_metric_name = False,
    )
    csv_logger = CSVLogger(save_dir=QT_DIR, name="quick_test")

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        default_root_dir      = QT_DIR,
        logger                = csv_logger,
        callbacks             = [ckpt_cb],
        precision             = "bf16",
        devices               = 1,
        accelerator           = "gpu",
        gradient_clip_val     = 1.0,
        max_epochs            = QUICK_EPOCHS,
        check_val_every_n_epoch = 5,
        log_every_n_steps     = 10,
        enable_progress_bar   = True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    logger.info(f"  快速训练完成。best val_loss={ckpt_cb.best_model_score}")

    # ── 采样（val，前 100 个样本）────────────────────────────────────────
    best_ckpt = ckpt_cb.best_model_path or os.path.join(QT_CKPT_DIR, "last.ckpt")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    val_loader = datamodule.val_dataloader()
    all_pred_frac, all_true_frac = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Quick sampling (val)"):
            if batch is None: continue
            batch = batch.to(device)
            traj_final, _ = model.sample(batch)

            num_atoms     = batch.num_atoms
            pred_frac_all = traj_final['frac_coords'].cpu()
            true_frac_all = batch.frac_coords.cpu()

            splits_p = torch.split(pred_frac_all,  num_atoms.tolist())
            splits_t = torch.split(true_frac_all,  num_atoms.tolist())
            for pf, tf in zip(splits_p, splits_t):
                all_pred_frac.append(pf.numpy())
                all_true_frac.append(tf.numpy())

            if len(all_pred_frac) >= 100:
                break

    # ── 计算 RMSD（最小镜像）─────────────────────────────────────────────
    def min_image_rmsd(pred_frac, true_frac, L):
        n = pred_frac.shape[0]
        cost = np.zeros((n, n))
        for i in range(n):
            d = pred_frac[i] - true_frac
            d -= np.round(d)
            cost[i] = np.linalg.norm(d * L, axis=1)
        ri, ci = linear_sum_assignment(cost)
        sq = []
        for r, c in zip(ri, ci):
            d = pred_frac[r] - true_frac[c]
            d -= np.round(d)
            sq.append(np.sum((d * L) ** 2))
        return float(np.sqrt(np.mean(sq)))

    rmsds = [min_image_rmsd(pf, tf, L)
             for pf, tf in zip(all_pred_frac, all_true_frac)]
    mean_rmsd = float(np.mean(rmsds))
    rand_baseline = (L / 2) * (3 / 5) ** 0.5   # ≈ 2.32 Å

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"快速测试结果（val，{len(rmsds)} 样本，epoch={QUICK_EPOCHS}）")
    logger.info(f"  RMSD        : {mean_rmsd:.4f} Å")
    logger.info(f"  随机基线    : {rand_baseline:.2f} Å")
    logger.info(f"  比值        : {mean_rmsd / rand_baseline:.2%}")

    if mean_rmsd < 1.8:
        verdict = "✅✅ RMSD < 1.8 Å，方向正确，可以开 500epoch 正式训练"
    elif mean_rmsd < rand_baseline * 0.9:
        verdict = "✅ 有效优于随机基线，可继续，但收敛较慢"
    else:
        verdict = "❌ RMSD 接近随机基线，L=6 未见效，立即汇报 Main Agent 2"

    logger.info(f"  判断        : {verdict}")
    logger.info("=" * 60)

    report = "\n".join([
        "Step4d Quick Test Report",
        f"Epochs={QUICK_EPOCHS}, L={L}Å",
        f"N_samples={len(rmsds)}",
        f"RMSD={mean_rmsd:.4f} Å",
        f"random_baseline={rand_baseline:.2f} Å",
        f"ratio={mean_rmsd / rand_baseline:.2%}",
        verdict,
    ])
    report_path = os.path.join(QT_DIR, "quick_test_rmsd.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"报告已保存 → {report_path}")
