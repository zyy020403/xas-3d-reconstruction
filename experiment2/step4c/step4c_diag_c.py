# step4c_diag_c.py
# Direction C diagnostic: track coordinate std across all reverse diffusion steps
# ============================================================
# Loads best checkpoint, runs sample() on 10 val samples,
# prints per-step std of frac_coords throughout reverse diffusion.
#
# Expected outcomes:
#   - std stays flat at ~0.29 throughout -> condition not guiding denoising
#   - std gradually shrinks from ~0.29 toward ~0.10 -> model learning but not enough
#
# Uniform[-0.5,0.5] std = 1/sqrt(12) ≈ 0.289
# True data std (coords clustered near origin, L=12) ≈ 4A/12 ≈ 0.10-0.15
# ============================================================

import os, sys, logging, warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4c_DIR   = os.path.join(EXP2_ROOT, "step4c")
CKPT_DIR     = os.path.join(STEP4c_DIR, "checkpoints")
CONF_DIR     = os.path.join(STEP3_DIR, "conf_xas")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

N_SAMPLES = 10   # enough to get stable std estimate, fast to run

if __name__ == "__main__":
    import math
    import torch
    import numpy as np
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset

    from xas_local_datamodule import XASDataModule

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Direction C diagnostic: coord std across reverse diffusion")
    logger.info("=" * 60)

    # ── Load model ────────────────────────────────────────────
    best_path_file = os.path.join(STEP4c_DIR, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file) as f:
            ckpt_path = f.read().strip()
    else:
        import glob, re
        ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt"))
        assert ckpts, f"No checkpoint found in {CKPT_DIR}"
        def _v(p):
            m = re.search(r'val_loss=([\d.]+)', os.path.basename(p))
            return float(m.group(1)) if m else 9999.
        ckpt_path = min(ckpts, key=_v)

    logger.info(f"Checkpoint: {ckpt_path}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="diag_c", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None})

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    # ── Load a small fixed val batch ─────────────────────────
    dm = XASDataModule(batch_size=N_SAMPLES, num_workers=0, L=12.0)
    dm.setup("fit")
    val_sub = Subset(dm.val_dataset, range(N_SAMPLES))
    loader  = DataLoader(val_sub, batch_size=N_SAMPLES, shuffle=False,
                         num_workers=0, collate_fn=lambda b: __import__(
                             'torch_geometric.data', fromlist=['Batch']
                         ).Batch.from_data_list([x for x in b if x is not None]))
    batch = next(iter(loader)).to(device)
    logger.info(f"Running reverse diffusion on {batch.num_graphs} samples...")

    # ── Patched sample() that logs std at every step ─────────
    # We re-implement the loop here instead of modifying the model file
    import torch.nn.functional as F
    from diffcsp.common.data_utils import lattice_params_to_matrix_torch
    from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

    MAX_ATOMIC_NUM = 100
    step_lr = 1e-5

    with torch.no_grad():
        batch_size = batch.num_graphs

        spectrum_cond = model.spectrum_encoder(
            batch.xmu_xanes, batch.chi1, batch.feff_features)

        x_T = torch.rand([batch.num_nodes, 3]).to(device) - 0.5
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(device)
        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        x_t = x_T.clone()
        t_t = t_T.clone()
        l_t = l_T.clone()

        timesteps = model.beta_scheduler.timesteps
        log_every = max(1, timesteps // 50)  # log ~50 points regardless of timesteps

        std_log   = []   # (step, std_x, std_y, std_z, std_all)
        step_ids  = []

        for t in range(timesteps, 0, -1):
            times    = torch.full((batch_size,), t, device=device)
            time_emb = model.time_embedding(times)
            condition = torch.cat([time_emb, spectrum_cond], dim=-1)

            sigma_x    = model.sigma_scheduler.sigmas[t]
            sigma_norm = model.sigma_scheduler.sigmas_norm[t]
            alphas     = model.beta_scheduler.alphas[t]
            alphas_cp  = model.beta_scheduler.alphas_cumprod[t]
            sigmas_b   = model.beta_scheduler.sigmas[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cp)

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)

            step_size = step_lr * (sigma_x / model.sigma_scheduler.sigma_begin) ** 2
            std_x_s   = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_t = model.decoder(
                condition, t_t, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x_s * rand_x

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)

            adj_sigma_x = model.sigma_scheduler.sigmas[t - 1]
            step_size2  = sigma_x ** 2 - adj_sigma_x ** 2
            std_x_s2    = torch.sqrt(
                (adj_sigma_x ** 2 * step_size2) / (sigma_x ** 2))

            pred_l, pred_x, pred_t = model.decoder(
                condition, t_t, x_t_minus_05, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size2 * pred_x + std_x_s2 * rand_x
            t_t_minus_1 = c0 * (t_t - c1 * pred_t) + sigmas_b * rand_t
            l_t_minus_1 = l_t  # keep_lattice=True

            # min-image fold to [-0.5, 0.5]
            x_t = x_t_minus_1 - torch.round(x_t_minus_1)
            t_t = t_t_minus_1
            l_t = l_t_minus_1

            # log std
            if (timesteps - t) % log_every == 0 or t == 1:
                std_xyz = x_t.std(dim=0).cpu()   # (3,)
                std_all = x_t.std().item()
                std_log.append((t, std_xyz[0].item(), std_xyz[1].item(),
                                std_xyz[2].item(), std_all))
                step_ids.append(t)

    # ── Print results ─────────────────────────────────────────
    uniform_std = 1.0 / math.sqrt(12)   # ≈ 0.289
    true_std_approx = 0.12              # 4A / 12A / sqrt(3) approx

    logger.info("")
    logger.info("=" * 60)
    logger.info("Coord std across reverse diffusion (t=T → t=1)")
    logger.info(f"  Uniform[-0.5,0.5] baseline std ≈ {uniform_std:.3f}")
    logger.info(f"  True data std (approx)          ≈ {true_std_approx:.3f}")
    logger.info(f"  {'Step':>6}  {'std_x':>7}  {'std_y':>7}  {'std_z':>7}  {'std_all':>8}")
    logger.info(f"  {'------':>6}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'--------':>8}")
    for (t, sx, sy, sz, sa) in std_log:
        logger.info(f"  {t:>6}  {sx:>7.4f}  {sy:>7.4f}  {sz:>7.4f}  {sa:>8.4f}")

    # ── Diagnosis ─────────────────────────────────────────────
    first_std = std_log[0][4]
    last_std  = std_log[-1][4]
    reduction = (first_std - last_std) / first_std * 100

    logger.info("")
    logger.info("─── Diagnosis ───────────────────────────────────────────")
    logger.info(f"  std at t=T  : {first_std:.4f}")
    logger.info(f"  std at t=1  : {last_std:.4f}")
    logger.info(f"  Reduction   : {reduction:.1f}%")
    logger.info("")

    if reduction < 5:
        logger.info("  RESULT: std essentially FLAT throughout")
        logger.info("  -> Condition is NOT guiding denoising at all")
        logger.info("  -> Likely cause: spectrum_encoder output ignored by decoder,")
        logger.info("     or condition injection architecture issue")
        logger.info("  -> Recommended: Direction B (change prior) or debug condition path")
    elif reduction < 25:
        logger.info("  RESULT: std shrinks slightly but not enough")
        logger.info(f"  -> Model is learning but prior mismatch too large")
        logger.info("  -> Recommended: Direction A (reduce L from 12 to 6)")
    else:
        logger.info("  RESULT: std shrinks significantly")
        logger.info("  -> Denoising is working; RMSD issue may be evaluation related")
        logger.info("  -> Investigate metrics script or try more epochs")

    # Save to file
    out_path = os.path.join(STEP4c_DIR, "diag_c_std_log.txt")
    with open(out_path, "w") as f:
        f.write(f"uniform_baseline_std={uniform_std:.4f}  true_approx_std={true_std_approx:.4f}\n")
        f.write(f"std_at_T={first_std:.4f}  std_at_1={last_std:.4f}  reduction={reduction:.1f}%\n\n")
        f.write(f"{'step':>6}  {'std_x':>7}  {'std_y':>7}  {'std_z':>7}  {'std_all':>8}\n")
        for (t, sx, sy, sz, sa) in std_log:
            f.write(f"{t:>6}  {sx:>7.4f}  {sy:>7.4f}  {sz:>7.4f}  {sa:>8.4f}\n")
    logger.info(f"\n  Full log saved -> {out_path}")
    logger.info("=" * 60)
