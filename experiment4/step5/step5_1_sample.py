#!/usr/bin/env python
"""
step5_1_sample.py
========================================================================
DiffCSP-Exp4 Step 5.1 — val + test reverse-diffusion sampling

Adapted from Exp2 step5_1_sample.py with these critical changes:
  - precision: bf16 → fp32  (MA4 D1 decision)
  - paths: Windows → /home/tcat/diffcsp_exp4/...
  - dataset: XASLocalStructureDataset (Exp2 L6, direct ids_file)
            → XasLocalDataModuleV2 (Exp4, samples_v2.csv via DM)
  - splits: hardcoded holdout → argparse, default [val, test]
  - holdout safety: raise immediately if "holdout" in --splits
                    (MA5 phase 5b explicit-approval gate)
  - schema: add sample_name as primary key; n_nominal/n_effective for
            Phase 4.6 silent-drop caveat reporting

Model loading: hydra instantiate + load_state_dict(strict=False)
  (Exp2 pattern, proven; bypasses PL load_from_checkpoint signature issues)

Usage:
  cd /home/tcat/diffcsp_exp4/code/step5
  PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
    CUDA_VISIBLE_DEVICES=0 \
    python step5_1_sample.py --splits val test 2>&1 | \
    tee /home/tcat/diffcsp_exp4/logs/step5_sample_val_test.log
"""

import argparse, os, sys, logging, warnings, time, traceback

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Linux / server paths (handoff locked) ───────────────────────────────
DIFFCSP_ROOT = "/home/tcat/diffcsp_exp4"
DATA_DIR     = f"{DIFFCSP_ROOT}/data"
CODE_DIR     = f"{DIFFCSP_ROOT}/code"
CKPT_PATH    = f"{DIFFCSP_ROOT}/checkpoints/best-epoch366-val0.7300.ckpt"
CONF_DIR     = f"{CODE_DIR}/step3/conf_xas"
OUT_DIR      = f"{CODE_DIR}/step5"

L           = 6.0
N_NEIGHBORS = 20
BATCH_SIZE_SAMPLE = 8   # Exp2 baseline; bs=16 may OOM in CSPNet attention

# Defensive sys.path injection (PYTHONPATH should cover, but safe)
for _p in [CODE_DIR, f"{CODE_DIR}/step3", f"{CODE_DIR}/step2"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="splits to sample (default: val test). holdout permitted post-MA5 phase 5b approval (2026-04-27).")
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE_SAMPLE)
    args = ap.parse_args()

    # ── HOLDOUT GATE (opened by MA5 phase 5b decision, 2026-04-27) ─────
    # Original gate raised RuntimeError on "holdout" in splits.
    # MA5 verdict: §6 conditions all pass on val+test, holdout authorized.
    # Audit anchor: backup at step5_1_sample.py.bak_phase5
    if "holdout" in args.splits:
        logger.warning("⚠️  HOLDOUT MODE — MA5 phase 5b authorized 2026-04-27. "
                       "This is a one-shot blind eval. Do NOT re-run except by MA5 directive.")

    logger.info("=" * 60)
    logger.info("Step 5.1  Exp4 reverse-diffusion sampling")
    logger.info(f"  ckpt   : {args.ckpt}")
    logger.info(f"  splits : {args.splits}")
    logger.info(f"  bs     : {args.batch_size}")
    logger.info(f"  L      : {L}, N_NEIGHBORS = {N_NEIGHBORS}")
    logger.info("=" * 60)

    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from tqdm import tqdm

    from xas_local_datamodule_v2 import XasLocalDataModuleV2

    if not os.path.exists(args.ckpt):
        logger.error(f"❌ ckpt missing: {args.ckpt}")
        sys.exit(1)

    # ── 1. Build model via hydra (Exp2 pattern, proven) ────────────────
    logger.info("\n[1/3] Instantiating model via hydra ...")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="step5_sample", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None,
    })
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None  # Exp2 trick to suppress warnings

    # Load weights (strict=False per Exp2; some buffers may not match)
    logger.info("[1/3] Loading checkpoint state_dict ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"       state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:    logger.info(f"         first 3 missing   : {missing[:3]}")
    if unexpected: logger.info(f"         first 3 unexpected: {unexpected[:3]}")
    logger.info(f"       ckpt epoch={ckpt.get('epoch')}, "
                f"global_step={ckpt.get('global_step')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(f"       device: {device}")

    # ── 2. DataModule (one instance, both splits) ──────────────────────
    logger.info("\n[2/3] Setting up DataModule ...")
    dm = XasLocalDataModuleV2(
        batch_size=args.batch_size,
        num_workers=0,           # sample is GPU-bound; 0 avoids fork overhead
        data_dir=DATA_DIR,
    )
    # PL convention: setup('fit') builds train+val; setup('test') builds test
    dm.setup("fit")
    try:
        dm.setup("test")
    except Exception as e:
        logger.warning(f"       dm.setup('test') raised: {e!r}; will retry during loader access")

    loader_map = {}
    try:
        loader_map["val"] = dm.val_dataloader()
    except Exception as e:
        logger.error(f"       val_dataloader FAIL: {e}")
    try:
        loader_map["test"] = dm.test_dataloader()
    except Exception as e:
        logger.error(f"       test_dataloader FAIL: {e}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 3. Sample each split ───────────────────────────────────────────
    logger.info("\n[3/3] Sampling ...")
    for split in args.splits:
        if split not in loader_map:
            logger.error(f"❌ unknown split '{split}', expected val|test. Skip.")
            continue

        loader = loader_map[split]
        n_nominal = len(loader.dataset)
        logger.info(f"\n[{split}] nominal={n_nominal}, n_batches={len(loader)}")

        all_sample_names = []
        all_mp_ids       = []
        all_pred_frac    = []
        all_pred_types   = []
        all_true_frac    = []
        all_true_types   = []
        all_eval_cutoff  = []

        n_none_batches = 0
        first_batch_logged = False
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"sample[{split}]")):
                if batch is None:
                    # Phase 4.6: collate returned None for fully-dropped batch
                    n_none_batches += 1
                    continue
                batch = batch.to(device)

                # First-batch sanity: print field shapes once
                if not first_batch_logged:
                    logger.info(f"  [first batch] num_graphs={batch.num_graphs}, "
                                f"frac_coords={tuple(batch.frac_coords.shape)}, "
                                f"atom_types={tuple(batch.atom_types.shape)}, "
                                f"eval_cutoff={tuple(batch.eval_cutoff.shape)}, "
                                f"num_atoms={tuple(batch.num_atoms.shape)}")
                    sn0 = getattr(batch, 'sample_name', None)
                    mp0 = getattr(batch, 'mp_id', None)
                    logger.info(f"  [first batch] sample_name[0]={sn0[0] if sn0 else 'N/A'!r}  "
                                f"mp_id[0]={mp0[0] if mp0 else 'N/A'!r}")
                    first_batch_logged = True

                # ── Reverse diffusion ──────────────────────────────────
                try:
                    traj_final, _ = model.sample(batch)
                except Exception as e:
                    logger.error(f"  batch {batch_idx} model.sample FAIL: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    raise   # red light: handoff §6 — stop, report

                num_atoms      = batch.num_atoms                      # (B,)
                pred_frac_all  = traj_final['frac_coords'].cpu()       # (B*20, 3) in [-0.5, 0.5]
                pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1   # (B*20,) Z (1-indexed)
                true_frac_all  = batch.frac_coords.cpu()
                true_types_all = batch.atom_types.cpu()
                eval_cutoffs   = batch.eval_cutoff.cpu()

                splits_p_frac  = torch.split(pred_frac_all,  num_atoms.tolist())
                splits_p_types = torch.split(pred_types_all, num_atoms.tolist())
                splits_t_frac  = torch.split(true_frac_all,  num_atoms.tolist())
                splits_t_types = torch.split(true_types_all, num_atoms.tolist())

                # Per-sample identifiers — robust dual-fallback
                # Path 1: batch.to_data_list() each Data has sample_name/mp_id attr
                # Path 2: batch.sample_name / batch.mp_id is a list (PyG string handling)
                batch_snames = getattr(batch, 'sample_name', None)
                batch_mpids  = getattr(batch, 'mp_id',       None)
                try:
                    data_list = batch.to_data_list()
                except Exception:
                    data_list = [None] * batch.num_graphs

                for i in range(batch.num_graphs):
                    data = data_list[i] if i < len(data_list) else None
                    sname = (
                        getattr(data, 'sample_name', None) if data is not None else None
                    ) or (
                        batch_snames[i] if (batch_snames is not None and i < len(batch_snames)) else None
                    ) or f"unk_{batch_idx}_{i}"
                    mpid  = (
                        getattr(data, 'mp_id', None) if data is not None else None
                    ) or (
                        batch_mpids[i]  if (batch_mpids  is not None and i < len(batch_mpids))  else None
                    ) or ""
                    all_sample_names.append(str(sname))
                    all_mp_ids.append(str(mpid))
                    all_pred_frac.append(splits_p_frac[i])
                    all_pred_types.append(splits_p_types[i])
                    all_true_frac.append(splits_t_frac[i])
                    all_true_types.append(splits_t_types[i])
                    ec_i = eval_cutoffs[i] if eval_cutoffs.dim() > 0 else eval_cutoffs
                    all_eval_cutoff.append(float(ec_i.item()))

        wall = time.time() - t0
        n_eff = len(all_sample_names)
        sd = n_nominal - n_eff
        ms_per = wall / max(n_eff, 1) * 1000
        logger.info(f"\n[{split}] done. effective={n_eff}/{n_nominal} "
                    f"(silent_drop={sd}, drop_pct={100*sd/max(n_nominal,1):.3f}%, "
                    f"None_batches={n_none_batches})")
        logger.info(f"[{split}] wall={wall/60:.1f} min  ({ms_per:.1f} ms/sample)")

        out_path = os.path.join(args.out_dir, f"predictions_{split}.pt")
        predictions = {
            "split":            split,
            "sample_name":      all_sample_names,
            "mp_id":            all_mp_ids,
            "pred_frac_coords": all_pred_frac,
            "pred_atom_types":  all_pred_types,
            "true_frac_coords": all_true_frac,
            "true_atom_types":  all_true_types,
            "eval_cutoff":      all_eval_cutoff,
            "L":                L,
            "checkpoint":       args.ckpt,
            "n_nominal":        n_nominal,
            "n_effective":      n_eff,
            "n_none_batches":   n_none_batches,
            "wall_seconds":     wall,
        }
        torch.save(predictions, out_path)
        logger.info(f"[{split}] saved → {out_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 5.1 sampling COMPLETE.")
    logger.info("Next:")
    logger.info("  python step5_2_compute_metrics.py --split val")
    logger.info("  python step5_2_compute_metrics.py --split test")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
