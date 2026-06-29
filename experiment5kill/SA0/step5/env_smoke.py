#!/usr/bin/env python
"""
env_smoke.py
========================================================================
Exp5 SA0 — env smoke test BEFORE the 1.5h K=10 run.

Validates:
  1. CUDA actually works in mlff env (driver/build compat risk)
  2. hydra + omegaconf + datamodule importable
  3. ckpt md5 matches pre-flight
  4. model instantiates + state_dict loads cleanly (missing/unexpected counts)
  5. one real val batch flows through model.sample()
  6. same seed → bit-exact (or near-exact) reproducible (RNG sanity)
  7. different seed → meaningfully different output (TTA premise sanity)

Wall: ~3-5 min. ALWAYS run this before multisample.py.

Usage:
  conda activate mlff
  export PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2
  CUDA_VISIBLE_DEVICES=1 python env_smoke.py 2>&1 | \
    tee /home/tcat/diffcsp_exp5/sa0/logs/env_smoke.log
"""
import sys, os, time, logging, hashlib, traceback
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

EXP4_ROOT = "/home/tcat/diffcsp_exp4"
DATA_DIR  = f"{EXP4_ROOT}/data"
CODE_DIR  = f"{EXP4_ROOT}/code"
CKPT_PATH = f"{EXP4_ROOT}/checkpoints/best-epoch366-val0.7300.ckpt"
CONF_DIR  = f"{CODE_DIR}/step3/conf_xas"
EXPECTED_CKPT_MD5 = "dc9d2c9b371c78125f285a5a6478d404"

for _p in [CODE_DIR, f"{CODE_DIR}/step3", f"{CODE_DIR}/step2"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def md5_full(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("=" * 60)
    logger.info("Exp5 SA0  env_smoke")
    logger.info("=" * 60)

    # ── [1] imports + CUDA ───────────────────────────────────────────
    logger.info("\n[1/7] imports + CUDA ...")
    import torch
    logger.info(f"    torch {torch.__version__}")
    logger.info(f"    torch cuda build: {torch.version.cuda}")
    logger.info(f"    cuda available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available; refusing to proceed")
        logger.error("   (mlff env's torch.cuda.is_available()==False — driver/build mismatch?)")
        sys.exit(1)
    logger.info(f"    cuda devices    : {torch.cuda.device_count()}")
    logger.info(f"    active device   : {torch.cuda.get_device_name(0)}")
    # Allocate a tensor to make sure CUDA isn't lying
    try:
        _t = torch.randn(1024, 1024, device="cuda")
        _t = _t @ _t.T
        torch.cuda.synchronize()
        del _t
        logger.info(f"    cuda matmul probe: OK")
    except Exception as e:
        logger.error(f"❌ CUDA matmul failed despite is_available()==True: {e}")
        sys.exit(1)

    import numpy as np
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    logger.info(f"    hydra OK")

    try:
        from xas_local_datamodule_v2 import XasLocalDataModuleV2
        logger.info(f"    XasLocalDataModuleV2 OK")
    except Exception as e:
        logger.error(f"❌ datamodule import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── [2] ckpt md5 ─────────────────────────────────────────────────
    logger.info("\n[2/7] ckpt md5 ...")
    actual_md5 = md5_full(CKPT_PATH)
    logger.info(f"    md5: {actual_md5}")
    if actual_md5 != EXPECTED_CKPT_MD5:
        logger.error(f"❌ md5 mismatch! expected={EXPECTED_CKPT_MD5}")
        logger.error("   ckpt has changed since pre-flight — investigate before continuing")
        sys.exit(1)
    logger.info(f"    matches pre-flight ✓")

    # ── [3] build model + load ckpt ──────────────────────────────────
    logger.info("\n[3/7] build model + load ckpt ...")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="env_smoke", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({"model": OmegaConf.to_container(_raw, resolve=False)}).model
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None,
    })
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"    state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        logger.warning(f"⚠️  unusual count")
        logger.warning(f"      missing[:5]    = {missing[:5]}")
        logger.warning(f"      unexpected[:5] = {unexpected[:5]}")
    logger.info(f"    epoch={ckpt.get('epoch')}, global_step={ckpt.get('global_step')}")
    if ckpt.get("epoch") != 366:
        logger.warning(f"⚠️  epoch != 366 (Exp4 best); got {ckpt.get('epoch')}")
    device = torch.device("cuda")
    model = model.to(device).eval()
    logger.info(f"    on device: {device}")

    # ── [4] datamodule + first val batch ─────────────────────────────
    logger.info("\n[4/7] datamodule + first val batch ...")
    dm = XasLocalDataModuleV2(batch_size=8, num_workers=0, data_dir=DATA_DIR)
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    logger.info(f"    val_loader.dataset size: {len(val_loader.dataset)}")
    logger.info(f"    val_loader.n_batches   : {len(val_loader)}")

    batch = None
    for b in val_loader:
        if b is not None:
            batch = b
            break
    if batch is None:
        logger.error("❌ no non-None batch in val_loader")
        sys.exit(1)
    batch = batch.to(device)
    logger.info(f"    first batch: num_graphs={batch.num_graphs}  "
                f"frac_coords={tuple(batch.frac_coords.shape)}  "
                f"atom_types={tuple(batch.atom_types.shape)}  "
                f"eval_cutoff={tuple(batch.eval_cutoff.shape)}")
    sn0 = getattr(batch, 'sample_name', None)
    logger.info(f"    sample_name[0]: {sn0[0] if sn0 else 'N/A'!r}")

    # ── [5] model.sample, seed=42 (a) ────────────────────────────────
    logger.info("\n[5/7] model.sample @ seed=42 (a) ...")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    t0 = time.time()
    with torch.no_grad():
        traj_a, _ = model.sample(batch)
    wall_a = time.time() - t0
    fc_a = traj_a['frac_coords']
    at_a = traj_a['atom_types']
    logger.info(f"    wall = {wall_a:.2f}s  ({wall_a/batch.num_graphs*1000:.0f} ms/sample)")
    logger.info(f"    traj_final.keys     : {list(traj_a.keys())}")
    logger.info(f"    frac_coords         : shape={tuple(fc_a.shape)} dtype={fc_a.dtype} "
                f"min={fc_a.min().item():.4f} max={fc_a.max().item():.4f}")
    logger.info(f"    atom_types (logits) : shape={tuple(at_a.shape)} dtype={at_a.dtype}")
    pred_z_a = (at_a.argmax(dim=-1) + 1).cpu()
    logger.info(f"    pred Z (argmax+1)   : unique={sorted(set(pred_z_a.tolist()))[:15]}...")

    # Wall-time projection for the K=10 run
    expected_per_sample = wall_a / batch.num_graphs
    proj_K10 = expected_per_sample * 500 * 10
    logger.info(f"    PROJECTION: 500 × K=10 ≈ {proj_K10:.0f}s = {proj_K10/60:.1f} min "
                f"= {proj_K10/3600:.2f} h")

    # ── [6] same seed determinism ────────────────────────────────────
    logger.info("\n[6/7] determinism: seed=42 (b) — should match (a) ...")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        traj_b, _ = model.sample(batch)
    delta_fc = (traj_b['frac_coords'] - fc_a).abs().max().item()
    delta_at = (traj_b['atom_types'].argmax(dim=-1) != at_a.argmax(dim=-1)).sum().item()
    logger.info(f"    max |Δfrac|       = {delta_fc:.2e}")
    logger.info(f"    type-flip count   = {delta_at} / {at_a.shape[0]}")
    if delta_fc > 1e-4 or delta_at > 0:
        logger.warning(f"⚠️  not bit-exact reproducible (likely cuDNN nondeterminism)")
        logger.warning(f"    acceptable for SA0 — RMSD impact ≪ 1e-2 expected")
    else:
        logger.info(f"    bit-exact reproducible ✓")

    # ── [7] different seed → different sample ───────────────────────
    logger.info("\n[7/7] variance: seed=43 — should differ from seed=42 ...")
    torch.manual_seed(43); torch.cuda.manual_seed_all(43)
    with torch.no_grad():
        traj_c, _ = model.sample(batch)
    delta_v = (traj_c['frac_coords'] - fc_a).abs().max().item()
    delta_v_mean = (traj_c['frac_coords'] - fc_a).abs().mean().item()
    type_flips = (traj_c['atom_types'].argmax(dim=-1) != at_a.argmax(dim=-1)).sum().item()
    logger.info(f"    max  |Δfrac| (seed 43 vs 42) = {delta_v:.4f}")
    logger.info(f"    mean |Δfrac|                 = {delta_v_mean:.4f}")
    logger.info(f"    type-flip count              = {type_flips} / {at_a.shape[0]}")
    if delta_v < 1e-3:
        logger.error("❌ different seeds give nearly-identical output — TTA premise broken")
        logger.error("   K-sample averaging will not work; investigate model.sample's RNG path")
        sys.exit(2)
    logger.info(f"    seeds DO produce different samples ✓ (TTA premise holds)")

    logger.info("\n" + "=" * 60)
    logger.info("env_smoke PASS — proceed with make_subset.py + multisample.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
