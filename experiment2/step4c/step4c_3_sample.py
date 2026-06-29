# step4c_3_sample.py
# Step4c formal sampling script
# ============================================================
# Differences from step4_3_sample.py:
#   - STEP4_DIR points to step4c
#   - pred_frac_centered line removed: sample() already returns [-0.5, 0.5]
#     (diffusion v3 applies min-image fold internally)
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4c_DIR   = os.path.join(EXP2_ROOT, "step4c")
CKPT_DIR     = os.path.join(STEP4c_DIR, "checkpoints")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

BATCH_SIZE_SAMPLE = 8

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from xas_local_datamodule import XASDataModule

    logger.info("=" * 60)
    logger.info("Step4c sampling (val + test)")
    logger.info("=" * 60)

    # Locate best checkpoint
    best_path_file = os.path.join(STEP4c_DIR, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file) as f:
            ckpt_path = f.read().strip()
    else:
        import glob, re
        ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt"))
        if not ckpts:
            logger.error(f"No checkpoint found in {CKPT_DIR}. Run training first.")
            sys.exit(1)
        def _val(p):
            m = re.search(r'val_loss=([\d.]+)', os.path.basename(p))
            return float(m.group(1)) if m else 9999.0
        ckpt_path = min(ckpts, key=_val)

    logger.info(f"Checkpoint: {ckpt_path}")

    CONF_DIR = os.path.join(STEP3_DIR, "conf_xas")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="sample4c", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None})

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("Weights loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    def run_sampling(loader, split_name):
        logger.info(f"\nSampling {split_name}...")

        all_mp_ids      = []
        all_pred_frac   = []
        all_pred_types  = []
        all_true_frac   = []
        all_true_types  = []
        all_eval_cutoff = []

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Sampling {split_name}")):
            if batch is None:
                continue
            batch = batch.to(device)

            with torch.no_grad():
                traj_final, _ = model.sample(batch)

            num_atoms      = batch.num_atoms
            # v3 sample() already returns [-0.5, 0.5] — no manual shift needed
            pred_frac_all  = traj_final['frac_coords'].cpu()
            pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1
            true_frac_all  = batch.frac_coords.cpu()
            true_types_all = batch.atom_types.cpu()
            eval_cutoffs   = batch.eval_cutoff.cpu()

            splits_p_frac  = torch.split(pred_frac_all,  num_atoms.tolist())
            splits_p_types = torch.split(pred_types_all, num_atoms.tolist())
            splits_t_frac  = torch.split(true_frac_all,  num_atoms.tolist())
            splits_t_types = torch.split(true_types_all, num_atoms.tolist())

            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                mp_id = getattr(data, 'mp_id', f"unk_{batch_idx}_{i}")
                all_mp_ids.append(mp_id)
                all_pred_frac.append(splits_p_frac[i])
                all_pred_types.append(splits_p_types[i])
                all_true_frac.append(splits_t_frac[i])
                all_true_types.append(splits_t_types[i])
                all_eval_cutoff.append(eval_cutoffs[i].item())

        predictions = {
            'mp_id':            all_mp_ids,
            'pred_frac_coords': all_pred_frac,
            'pred_atom_types':  all_pred_types,
            'true_frac_coords': all_true_frac,
            'true_atom_types':  all_true_types,
            'eval_cutoff':      all_eval_cutoff,
            'L':                12.0,
            'checkpoint':       ckpt_path,
        }

        out_path = os.path.join(STEP4c_DIR, f"predictions_{split_name}.pt")
        torch.save(predictions, out_path)
        logger.info(f"  Saved {len(all_mp_ids)} samples -> {out_path}")
        return out_path

    datamodule = XASDataModule(batch_size=BATCH_SIZE_SAMPLE, num_workers=0, L=12.0)

    datamodule.setup("fit")
    val_out = run_sampling(datamodule.val_dataloader(), "val")

    datamodule.setup("test")
    test_out = run_sampling(datamodule.test_dataloader(), "test")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sampling complete.")
    logger.info(f"  val  -> {val_out}")
    logger.info(f"  test -> {test_out}")
    logger.info("Next: run step4c_4_compute_metrics.py")
    logger.info("=" * 60)
