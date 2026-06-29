# step4c_test_full.py
# Step4c 一体化测试脚本（取代 test_train + test_sample + test_metrics）
# ============================================================
# 核心逻辑：
#   1. 4 项前置检查（坐标系管道验证）
#   2. 在 epoch=0（未训练模型）采样 → 算 RMSD_0（应 ≈ 随机基线 4.65Å）
#   3. 训练 30 epoch
#   4. 在 epoch=30 采样 → 算 RMSD_30
#   5. 判断标准：RMSD_30 < RMSD_0 * 0.80（比随机基线改善 ≥ 20%）
#
# 为什么这样设计：
#   - 之前多次调试均出现"loss 下降但 RMSD ≈ 随机"，说明 loss 下降与最终
#     预测质量是解耦的，只看 loss 无法判断坐标管道是否正确。
#   - 在相同数据子集上对比 epoch 0 vs epoch 30 的 RMSD，能直接判断模型
#     是否从谱信息中学到了局部结构信号。
#   - 如果 30 epoch 后 RMSD 仍 ≈ 随机基线，500 epoch 也不会收敛，应立即
#     停下汇报 Main Agent 2。
#
# 测试配置：
#   N_EVAL  = 80 条固定 val 子集（够统计，但采样快）
#   N_TRAIN = 400 条 train 子集
#   MAX_EPOCHS = 30，early_stop patience=10
# ============================================================

import os, sys, logging, warnings, math, time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*UserWarning.*")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4c_DIR   = os.path.join(EXP2_ROOT, "step4c")
CKPT_DIR     = os.path.join(STEP4c_DIR, "test_checkpoints")
LOG_DIR      = os.path.join(STEP4c_DIR, "test_logs")
CONF_DIR     = os.path.join(STEP3_DIR, "conf_xas")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── 配置 ──────────────────────────────────────────────────────────────────────
MAX_EPOCHS     = 30
BATCH_SIZE     = 16
LR             = 1e-4
GRADIENT_CLIP  = 1.0
EARLY_STOP_PAT = 10
PRECISION      = 'bf16'
NUM_WORKERS    = 0
N_TRAIN        = 400
N_EVAL         = 80    # epoch0 和 epoch30 用同一批 80 个 val 样本
L              = 12.0
RANDOM_BASELINE = L / 2 * (3 / 5) ** 0.5   # ≈ 4.65 Å
IMPROVE_THRESHOLD = 0.80   # RMSD_30 < RMSD_0 * 0.80 才算通过


# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

def compute_rmsd_batch(model, eval_loader, device, desc=""):
    """对 eval_loader 中的所有样本采样，返回平均 RMSD（最小镜像匈牙利匹配）"""
    import torch
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    model.eval()
    rmsds = []

    for batch in eval_loader:
        if batch is None:
            continue
        batch = batch.to(device)

        with torch.no_grad():
            traj_final, _ = model.sample(batch)

        pred_frac_all  = traj_final['frac_coords'].cpu().numpy()   # (N_total, 3)
        true_frac_all  = batch.frac_coords.cpu().numpy()
        num_atoms_list = batch.num_atoms.tolist()

        # 按图拆分
        idx = 0
        for na in num_atoms_list:
            pf = pred_frac_all[idx:idx + na]
            tf = true_frac_all[idx:idx + na]
            idx += na

            if pf.shape[0] != 20:
                continue

            # 最小镜像匈牙利匹配
            cost = np.zeros((20, 20))
            for i in range(20):
                d = pf[i] - tf
                d -= np.round(d)
                cost[i] = np.linalg.norm(d * L, axis=1)
            ri, ci = linear_sum_assignment(cost)
            sq = []
            for r, c in zip(ri, ci):
                d = pf[r] - tf[c]; d -= np.round(d)
                sq.append(np.sum((d * L) ** 2))
            rmsds.append(float(np.sqrt(np.mean(sq))))

    mean_rmsd = float(np.mean(rmsds)) if rmsds else float('nan')
    frac_below_random = float(np.mean(np.array(rmsds) < RANDOM_BASELINE)) if rmsds else 0.0
    return mean_rmsd, frac_below_random, len(rmsds)


def run_checks(model, datamodule, device, logger):
    """4 项强制检查，任一失败抛 AssertionError"""
    import torch
    from torch_geometric.data import Batch

    logger.info("")
    logger.info("=" * 60)
    logger.info("开训前强制检查（共 4 项）")
    logger.info("=" * 60)

    # 取 5 个样本
    samples = []
    for i in range(len(datamodule.train_dataset)):
        s = datamodule.train_dataset[i]
        if s is not None: samples.append(s)
        if len(samples) >= 5: break

    # 检查 1：Dataset 坐标范围
    logger.info("\n[检查 1] Dataset frac_coords 范围（应在 [-0.5, 0.5]）")
    all_min = min(s.frac_coords.min().item() for s in samples)
    all_max = max(s.frac_coords.max().item() for s in samples)
    c1 = (all_min >= -0.5 - 1e-4) and (all_max <= 0.5 + 1e-4)
    logger.info(f"  min={all_min:.4f}  max={all_max:.4f}  →  {'✅ 通过' if c1 else '❌ 失败'}")
    assert c1, f"检查1失败：frac_coords 超出 [-0.5,0.5]，min={all_min} max={all_max}"

    # 检查 2：forward() 加噪后有负值（说明 % 1. 已删除）
    logger.info("\n[检查 2] forward() 加噪后坐标含负值（若无负值说明 % 1. 未删除）")
    batch = Batch.from_data_list(samples[:4]).to(device)
    model.eval()
    with torch.no_grad():
        out = model.forward(batch, _return_noisy_frac=True)
    noisy = out['_input_frac_coords'].cpu()
    n_min, n_max = noisy.min().item(), noisy.max().item()
    has_neg = (noisy < -0.05).any().item()
    c2 = has_neg and (n_min >= -1.5) and (n_max <= 1.5)
    logger.info(f"  min={n_min:.4f}  max={n_max:.4f}  含负值={has_neg}  →  {'✅ 通过' if c2 else '❌ 失败'}")
    assert c2, f"检查2失败：加噪后无负值（min={n_min:.4f}），forward() 中 %1. 可能未删除"

    # 检查 3：loss 数值正常
    logger.info("\n[检查 3] loss 数值合理（不应 NaN/Inf 或 >200）")
    loss       = out['loss'].item()
    loss_coord = out['loss_coord'].item()
    loss_type  = out['loss_type'].item()
    c3 = (not math.isnan(loss)) and (not math.isinf(loss)) and (0 < loss < 200)
    logger.info(f"  loss={loss:.4f}  loss_coord={loss_coord:.4f}  loss_type={loss_type:.4f}  →  {'✅ 通过' if c3 else '❌ 失败'}")
    assert c3, f"检查3失败：loss={loss}"

    # 检查 4：sample() 输出坐标集中于 [-0.5, 0.5]
    logger.info("\n[检查 4] sample() 输出坐标应集中于 [-0.5, 0.5]（>0.8 占比 <5%）")
    with torch.no_grad():
        tf, _ = model.sample(batch)
    pf = tf['frac_coords'].cpu()
    s_min, s_max = pf.min().item(), pf.max().item()
    gt08 = (pf > 0.8).float().mean().item()
    c4 = (s_min >= -0.6) and (s_max <= 0.6) and (gt08 < 0.05)
    logger.info(f"  min={s_min:.4f}  max={s_max:.4f}  >0.8占比={gt08*100:.1f}%  →  {'✅ 通过' if c4 else '❌ 失败（sample()中可能有残余 %1.）'}")
    assert c4, f"检查4失败：sample()输出 >0.8 占比={gt08*100:.1f}%，坐标未集中于[-0.5,0.5]"

    logger.info("")
    logger.info("✅✅ 全部 4 项检查通过")
    logger.info("=" * 60)
    model.train()


# ═════════════════════════════════════════════════════════════════════════════
# 主程序
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    t_start = time.time()

    import torch
    from torch.utils.data import Subset
    import hydra
    import pytorch_lightning as pl
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, LearningRateMonitor, EarlyStopping)
    from pytorch_lightning.loggers import CSVLogger
    from torch_geometric.loader import DataLoader

    from xas_local_datamodule import XASDataModule

    logger.info("=" * 60)
    logger.info("Step4c 一体化测试：epoch0 RMSD vs epoch30 RMSD")
    logger.info(f"  随机基线 RMSD ≈ {RANDOM_BASELINE:.2f} Å")
    logger.info(f"  通过标准：RMSD_30 < RMSD_0 × {IMPROVE_THRESHOLD}")
    logger.info("=" * 60)

    # ── 1. 加载配置 & 模型 ─────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="test_full", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    assert float(model_cfg.cost_lattice) < 1e-5, "cost_lattice != 0，停止。"

    optim_cfg = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": LR, "betas": [0.9, 0.999], "weight_decay": 0.0,
        },
        "use_lr_scheduler": True,
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": MAX_EPOCHS, "eta_min": 1e-6,
        },
    })

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    # ── 2. DataModule ──────────────────────────────────────────────────────
    datamodule = XASDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, L=L)
    datamodule.setup("fit")

    datamodule.train_dataset = Subset(
        datamodule.train_dataset, range(min(N_TRAIN, len(datamodule.train_dataset))))
    # val 子集固定，epoch0 和 epoch30 用同一批
    val_indices = list(range(min(N_EVAL, len(datamodule.val_dataset))))
    datamodule.val_dataset = Subset(datamodule.val_dataset, val_indices)

    logger.info(f"数据集：train={len(datamodule.train_dataset)}  "
                f"eval（固定）={len(datamodule.val_dataset)}")

    # 构建固定 eval loader（不 shuffle）
    eval_loader = DataLoader(
        datamodule.val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=datamodule._collate_fn if hasattr(datamodule, '_collate_fn') else None,
    )

    # ── 3. 4 项强制检查 ────────────────────────────────────────────────────
    run_checks(model, datamodule, device, logger)

    # ── 4. Epoch 0（未训练）RMSD ───────────────────────────────────────────
    logger.info("")
    logger.info("─── Epoch 0（未训练模型）采样 → 计算 RMSD_0 ───────────────")
    rmsd_0, frac_below_0, n_0 = compute_rmsd_batch(model, eval_loader, device)
    logger.info(f"  RMSD_0  = {rmsd_0:.4f} Å  （随机基线 ≈ {RANDOM_BASELINE:.2f} Å）")
    logger.info(f"  样本数  = {n_0}，低于随机基线占比 = {frac_below_0*100:.1f}%")

    if rmsd_0 < RANDOM_BASELINE * 0.5:
        logger.warning(
            f"  ⚠️  RMSD_0={rmsd_0:.2f}Å 显著低于随机基线，"
            "未训练模型不应有此结果，请检查 eval_loader 或数据是否泄露。")
    elif rmsd_0 > RANDOM_BASELINE * 1.3:
        logger.warning(
            f"  ⚠️  RMSD_0={rmsd_0:.2f}Å 高于随机基线 30%，"
            "sample()坐标系或匹配逻辑可能有问题，继续训练但请关注。")
    else:
        logger.info(f"  ✅ RMSD_0 在随机基线附近（正常）")

    # ── 5. 正式训练 30 epoch ───────────────────────────────────────────────
    logger.info("")
    logger.info("─── 开始训练 30 epoch ─────────────────────────────────────")
    model.train()

    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="test-ep{epoch:02d}-vl{val_loss:.4f}",
        monitor="val_loss", save_top_k=1, mode="min",
        save_last=True, verbose=False,
        auto_insert_metric_name=False,
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PAT,
                             mode="min", verbose=True)
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="test_full")

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        default_root_dir        = STEP4c_DIR,
        logger                  = csv_logger,
        callbacks               = [ckpt_cb, LearningRateMonitor("epoch"), early_cb],
        precision               = PRECISION,
        devices                 = 1,
        accelerator             = "gpu",
        gradient_clip_val       = GRADIENT_CLIP,
        max_epochs              = MAX_EPOCHS,
        check_val_every_n_epoch = 2,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    best_val_loss = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score else float('nan')
    logger.info(f"训练完成。best val_loss = {best_val_loss:.4f}")

    # ── 6. 加载最优 ckpt，采样 → RMSD_30 ────────────────────────────────
    logger.info("")
    logger.info("─── Epoch 30（最优 ckpt）采样 → 计算 RMSD_30 ─────────────")
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        state = ckpt_data.get("state_dict", ckpt_data)
        model.load_state_dict(state, strict=False)
        logger.info(f"加载最优 ckpt：{best_ckpt}")
    else:
        logger.warning("未找到最优 ckpt，使用训练结束时的模型权重")

    model = model.to(device)
    rmsd_30, frac_below_30, n_30 = compute_rmsd_batch(model, eval_loader, device)
    logger.info(f"  RMSD_30 = {rmsd_30:.4f} Å")
    logger.info(f"  样本数  = {n_30}，低于随机基线占比 = {frac_below_30*100:.1f}%")

    # ── 7. 最终判断 ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    improve_ratio = rmsd_30 / rmsd_0 if rmsd_0 > 0 else 999.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    logger.info(f"  随机基线  RMSD ≈ {RANDOM_BASELINE:.2f} Å")
    logger.info(f"  RMSD_0  （epoch 0）  = {rmsd_0:.4f} Å")
    logger.info(f"  RMSD_30 （epoch 30） = {rmsd_30:.4f} Å")
    logger.info(f"  改善比例            = {(1-improve_ratio)*100:.1f}%  "
                f"（{rmsd_0:.4f} → {rmsd_30:.4f}）")
    logger.info(f"  best val_loss       = {best_val_loss:.4f}")
    logger.info(f"  总耗时              = {elapsed/60:.1f} 分钟")
    logger.info("")

    # 三个维度判断
    ok_rmsd_improve = improve_ratio < IMPROVE_THRESHOLD   # 改善 ≥ 20%
    ok_below_random = rmsd_30 < RANDOM_BASELINE           # 低于随机基线
    ok_val_loss_dir = best_val_loss < 999.0               # val_loss 有记录

    logger.info(f"  判断1：RMSD_30 < RMSD_0 × {IMPROVE_THRESHOLD}  "
                f"（{rmsd_30:.4f} < {rmsd_0 * IMPROVE_THRESHOLD:.4f}）"
                f"  →  {'✅' if ok_rmsd_improve else '❌'}")
    logger.info(f"  判断2：RMSD_30 < 随机基线  "
                f"（{rmsd_30:.4f} < {RANDOM_BASELINE:.2f}）"
                f"  →  {'✅' if ok_below_random else '❌'}")
    logger.info(f"  判断3：val_loss 有效下降   "
                f"（best={best_val_loss:.4f}）"
                f"  →  {'✅' if ok_val_loss_dir else '❌'}")

    logger.info("")
    if ok_rmsd_improve and ok_below_random:
        logger.info("✅✅ 测试通过 → 建议启动正式 500-epoch 训练")
        verdict = "PASS"
    elif ok_below_random and not ok_rmsd_improve:
        logger.info("⚠️  RMSD 低于随机但改善 <20%，可能需要更多 epoch 才能体现"
                    "，建议谨慎启动正式训练并关注前 100 epoch 的 RMSD 趋势")
        verdict = "MARGINAL"
    else:
        logger.info("❌ 测试失败 → 30 epoch 后 RMSD 未改善，不建议启动正式训练")
        logger.info("   可能原因：")
        logger.info("   a) 谱特征与局部结构相关性弱（需检查 feff_features 质量）")
        logger.info("   b) 模型容量/lr 设置问题（建议汇报 Main Agent 2）")
        logger.info("   c) 采样步数不足（当前使用默认 step_lr=1e-5）")
        verdict = "FAIL"

    # ── 8. 写入报告 ────────────────────────────────────────────────────────
    report = [
        "Step4c Test Full Report",
        "=" * 60,
        f"Random baseline RMSD : {RANDOM_BASELINE:.4f} Å",
        f"RMSD_0  (epoch  0)   : {rmsd_0:.4f} Å  (n={n_0})",
        f"RMSD_30 (epoch 30)   : {rmsd_30:.4f} Å  (n={n_30})",
        f"Improvement ratio    : {(1-improve_ratio)*100:.1f}%",
        f"Best val_loss        : {best_val_loss:.4f}",
        f"Elapsed              : {elapsed/60:.1f} min",
        f"Verdict              : {verdict}",
        "",
        "Check 1 (dataset range)    : PASS",
        "Check 2 (forward no %1.)   : PASS",
        "Check 3 (loss finite)      : PASS",
        "Check 4 (sample range)     : PASS",
    ]
    report_path = os.path.join(STEP4c_DIR, "test_full_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # 保存 ckpt 路径供正式训练用（如果通过）
    with open(os.path.join(STEP4c_DIR, "test_best_checkpoint_path.txt"), "w") as f:
        f.write(best_ckpt or "")

    logger.info("")
    logger.info(f"报告 → {report_path}")
    logger.info("=" * 60)
