# step4_1_health_check.py
# Step 4.1 — 训练前健康检查
# ============================================================
# 检查项：
#   1. forward pass 无 NaN
#   2. keep_lattice=True → pred_lengths ≈ 12.0  （最核心！）
#   3. frac_coords 在 [0, 1]
#   4. atom_types 在 [1, 100]
# ★ 通过后汇报 Main Agent 确认，才允许运行 step4_2_train.py
# ============================================================

import os, sys, logging, warnings
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4_DIR    = os.path.join(EXP2_ROOT, "step4")
CONF_DIR     = os.path.join(STEP3_DIR, "conf_xas")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)
os.makedirs(STEP4_DIR, exist_ok=True)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch_geometric.data import Batch
    from xas_local_datamodule import XASDataModule

    logger.info("=" * 60)
    logger.info("Step 4.1  训练前健康检查")
    logger.info("=" * 60)

    # 1. 加载 YAML
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="hc", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({"model":
        OmegaConf.to_container(_raw, resolve=False)}).model

    logger.info(f"cost_lattice={model_cfg.cost_lattice}  "
                f"latent_dim={model_cfg.latent_dim}  "
                f"time_dim={model_cfg.time_dim}")
    if float(model_cfg.cost_lattice) >= 1e-5:
        logger.error("❌ cost_lattice != 0！停止。"); sys.exit(1)

    # 2. 实例化模型
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None})
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量={n_params:,}  keep_lattice={model.keep_lattice}")
    if not model.keep_lattice:
        logger.error("❌ keep_lattice=False！停止。"); sys.exit(1)

    # 3. 取 5 个 val 样本
    dm = XASDataModule(batch_size=8, num_workers=0, L=12.0)
    dm.setup("fit")
    mini_batch = None
    for b in dm.val_dataloader():
        if b is not None and b.num_graphs >= 5:
            mini_batch = Batch.from_data_list(b.to_data_list()[:5]); break
    if mini_batch is None:
        logger.error("❌ 无法获取 mini_batch！"); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    mini_batch = mini_batch.to(device)

    # 检查 1：forward
    logger.info("\n【检查 1/4】forward pass")
    with torch.no_grad():
        out = model(mini_batch)
    loss, lc, lt, ll = (out['loss'].item(), out['loss_coord'].item(),
                        out['loss_type'].item(), out['loss_lattice'].item())
    logger.info(f"  loss={loss:.4f} coord={lc:.4f} type={lt:.4f} lattice={ll:.4f}")
    if any(v != v for v in [loss, lc, lt, ll]):
        logger.error("❌ 检查 1 FAIL：NaN"); sys.exit(1)
    logger.info("  ✅ 检查 1 PASS")

    # 检查 2-4：sample（50步）
    logger.info("\n【检查 2-4/4】sample() 50步快速采样")
    _orig = model.beta_scheduler.timesteps
    model.beta_scheduler.timesteps = 50
    try:
        traj_final, traj_stack = model.sample(mini_batch)
    finally:
        model.beta_scheduler.timesteps = _orig

    # 检查 2：晶格（最核心）
    logger.info("\n【检查 2/4】pred_lengths（期望 ≈ 12.0）")
    pl_all = traj_final['lattices'].cpu()
    pred_lengths = torch.stack([
        torch.tensor([pl_all[i,0,0].abs(), pl_all[i,1,1].abs(), pl_all[i,2,2].abs()])
        for i in range(pl_all.shape[0])])
    c2_pass = True
    for i, pl in enumerate(pred_lengths):
        ok = bool((pl >= 11.0).all() and (pl <= 13.0).all())
        logger.info(f"  {'✅' if ok else '❌'} 样本{i}: "
                    f"[{pl[0]:.4f}, {pl[1]:.4f}, {pl[2]:.4f}]")
        if not ok: c2_pass = False
    if not c2_pass:
        logger.error("❌ 检查 2 FAIL：晶格未固定！"); sys.exit(1)
    logger.info("  ✅ 检查 2 PASS")

    # 检查 3：frac_coords
    logger.info("\n【检查 3/4】frac_coords")
    pf = traj_final['frac_coords'].cpu()
    fmin, fmax = pf.min().item(), pf.max().item()
    logger.info(f"  范围：[{fmin:.4f}, {fmax:.4f}]  期望 [0,1]")
    if 0.0 <= fmin and fmax <= 1.0: logger.info("  ✅ 检查 3 PASS")
    else: logger.warning("  ⚠️  检查 3 WARNING（不阻断）")

    # 检查 4：atom_types
    logger.info("\n【检查 4/4】atom_types")
    pt = traj_stack['atom_types'].cpu()
    tmin, tmax = pt.min().item(), pt.max().item()
    logger.info(f"  范围：[{tmin}, {tmax}]  期望 [1,100]")
    if not (1 <= tmin and tmax <= 100):
        logger.error("❌ 检查 4 FAIL"); sys.exit(1)
    logger.info("  ✅ 检查 4 PASS")

    # 汇总报告
    report = "\n".join([
        "", "=" * 60,
        "Step 4.1 健康检查完成  ──  汇报给 Main Agent",
        "=" * 60,
        f"参数量        : {n_params:,}",
        f"keep_lattice  : {model.keep_lattice}",
        f"forward loss  : {loss:.4f}  (coord={lc:.4f}, type={lt:.4f}, lattice={ll:.4f})",
        "", "pred_lengths（期望 ≈ 12.0）:",
        *[f"  样本{i}: [{pl[0]:.4f},{pl[1]:.4f},{pl[2]:.4f}]"
          for i, pl in enumerate(pred_lengths)],
        "", f"pred_frac 范围  : [{fmin:.4f}, {fmax:.4f}]",
        f"atom_types 范围 : [{tmin}, {tmax}]",
        "", "★ 等待 Main Agent 确认后，方可运行 step4_2_train.py",
        "=" * 60,
    ])
    logger.info(report)
    with open(os.path.join(STEP4_DIR, "health_check_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)