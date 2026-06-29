#!/usr/bin/env python
"""
multisample.py
========================================================================
Exp5 SA0 — K-sample test-time augmentation sweep.

Wraps Exp4's val dataset with torch.utils.data.Subset to restrict to
the 500 sample names in --subset_csv, then runs K independent
reverse-diffusion sweeps. Each sweep is seeded as (seed_base + k),
giving distinct (and reproducible) noise per sweep.

Output: samples_raw_K{K}.pt  containing per-sample × K replicates.
        Aggregate step reads this and computes K=1/5/10 metrics from
        the same raw data — no re-sweeping needed.

Wall: ~1.5h for K=10 on 500 samples (single RTX 4090).

Usage (must be in mlff env, with PYTHONPATH set, GPU pinned):
  conda activate mlff
  export PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2
  CUDA_VISIBLE_DEVICES=1 python multisample.py --K 10 \
      --subset_csv /home/tcat/diffcsp_exp5/sa0/results/sa0_subset_500.csv \
      --out_pt /home/tcat/diffcsp_exp5/sa0/results/samples_raw_K10.pt \
      2>&1 | tee /home/tcat/diffcsp_exp5/sa0/logs/multisample_K10.log
"""
import argparse, os, sys, logging, warnings, time, traceback, hashlib, csv

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

EXP4_ROOT = "/home/tcat/diffcsp_exp4"
DATA_DIR  = f"{EXP4_ROOT}/data"
CODE_DIR  = f"{EXP4_ROOT}/code"
CKPT_PATH = f"{EXP4_ROOT}/checkpoints/best-epoch366-val0.7300.ckpt"
CONF_DIR  = f"{CODE_DIR}/step3/conf_xas"
EXPECTED_CKPT_MD5 = "dc9d2c9b371c78125f285a5a6478d404"

L           = 6.0
N_NEIGHBORS = 20
BATCH_SIZE  = 8

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--K",          type=int, required=True)
    ap.add_argument("--subset_csv", type=str, required=True)
    ap.add_argument("--out_pt",     type=str, required=True)
    ap.add_argument("--seed_base",  type=int, default=1234567)
    ap.add_argument("--ckpt",       type=str, default=CKPT_PATH)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--skip_md5_check", action="store_true",
                    help="bypass ckpt md5 check (NOT recommended)")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("Exp5 SA0  multisample")
    logger.info(f"  K           : {args.K}")
    logger.info(f"  subset_csv  : {args.subset_csv}")
    logger.info(f"  out_pt      : {args.out_pt}")
    logger.info(f"  seed_base   : {args.seed_base}  (per-sweep seed = seed_base + k)")
    logger.info(f"  ckpt        : {args.ckpt}")
    logger.info(f"  batch_size  : {args.batch_size}")
    logger.info(f"  L           : {L}, N_NEIGHBORS = {N_NEIGHBORS}")
    logger.info("=" * 60)

    # ── [0] ckpt md5 paranoia ────────────────────────────────────────
    actual_md5 = md5_full(args.ckpt)
    logger.info(f"\n[0/5] ckpt md5 (full): {actual_md5}")
    if actual_md5 != EXPECTED_CKPT_MD5:
        if args.skip_md5_check:
            logger.warning(f"⚠️  md5 mismatch but --skip_md5_check given; proceeding")
        else:
            logger.error(f"❌ md5 mismatch! expected={EXPECTED_CKPT_MD5}")
            sys.exit(1)
    else:
        logger.info(f"      matches pre-flight ✓")

    # ── [1] read subset csv ──────────────────────────────────────────
    logger.info("\n[1/5] reading subset CSV ...")
    subset_names = []
    with open(args.subset_csv) as f:
        for r in csv.DictReader(f):
            subset_names.append(r["sample_name"])
    n_target = len(subset_names)
    if len(set(subset_names)) != n_target:
        logger.error(f"❌ subset_csv has duplicates")
        sys.exit(1)
    logger.info(f"      n_subset = {n_target}")

    # ── [2] heavy imports + model ────────────────────────────────────
    logger.info("\n[2/5] imports + build model ...")
    import torch, numpy as np
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch.utils.data import Subset
    from tqdm import tqdm
    from xas_local_datamodule_v2 import XasLocalDataModuleV2

    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        sys.exit(1)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="multisample", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({"model": OmegaConf.to_container(_raw, resolve=False)}).model
    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None,
    })
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"      state_dict missing={len(missing)} unexpected={len(unexpected)}  "
                f"epoch={ckpt.get('epoch')}")
    device = torch.device("cuda")
    model = model.to(device).eval()
    logger.info(f"      device: {device} ({torch.cuda.get_device_name(0)})")

    # ── [3] build subset dataloader ──────────────────────────────────
    logger.info("\n[3/5] building val Subset dataloader ...")
    dm = XasLocalDataModuleV2(batch_size=args.batch_size, num_workers=0, data_dir=DATA_DIR)
    dm.setup("fit")
    full_loader  = dm.val_dataloader()
    full_dataset = full_loader.dataset
    n_full = len(full_dataset)
    logger.info(f"      full val dataset size: {n_full}")

    # Build sample_name → dataset_idx map via val_loader iteration.
    # Why not via dataset[i]: this dataset returns PyG Data objects that do NOT
    # expose sample_name as a Python attribute (it lives in PyG's internal
    # _data_attrs and is only surfaced post-collation as batch.sample_name).
    # That mirrors step5_1's fallback pattern (lines 232-242 of step5_1_sample.py).
    #
    # With shuffle=False (default for val), the loader feeds dataset[0..len-1]
    # in order, batching by batch_size. For each FULL batch (num_graphs ==
    # batch_size), the j-th item came from dataset[iter_idx + j] — clean map.
    # PARTIAL batches (collate dropped items, num_graphs < batch_size) have
    # ambiguous survivor positions and are skipped (worst case ≈21 names lost
    # out of 7621 — well under our 500-subset margin).
    logger.info(f"      building name → idx map via val_loader iteration ...")
    name_to_idx = {}
    iter_idx = 0
    n_full_batches = 0
    n_partial_batches = 0
    n_none_batches = 0

    for batch in tqdm(full_loader, desc="indexing val"):
        if batch is None:
            n_none_batches += 1
            iter_idx += args.batch_size
            continue
        bs_actual = batch.num_graphs
        snames = getattr(batch, 'sample_name', None)
        if snames is None or len(snames) != bs_actual:
            n_partial_batches += 1
            iter_idx += args.batch_size
            continue
        if bs_actual == args.batch_size:
            for j in range(bs_actual):
                name_to_idx[str(snames[j])] = iter_idx + j
            n_full_batches += 1
        else:
            n_partial_batches += 1
        iter_idx += args.batch_size

    logger.info(f"      mapped {len(name_to_idx)} names from {n_full_batches} full batches")
    logger.info(f"      partial batches (unmappable): {n_partial_batches}  "
                f"(lose ≤{n_partial_batches * args.batch_size} possible positions)")
    logger.info(f"      None batches                : {n_none_batches}")

    missing_in_ds = [n for n in subset_names if n not in name_to_idx]
    if missing_in_ds:
        logger.warning(f"⚠️  {len(missing_in_ds)}/{len(subset_names)} subset names "
                       f"not mappable (likely fell in partial batches); dropping")
        for n in missing_in_ds[:5]:
            logger.warning(f"     {n}")
        subset_names = [n for n in subset_names if n in name_to_idx]

    if len(subset_names) == 0:
        logger.error(f"❌ no subset names mappable; aborting")
        sys.exit(1)
    n_target = len(subset_names)
    logger.info(f"      effective n_subset = {n_target}")
    subset_indices = [name_to_idx[n] for n in subset_names]

    subset_ds = Subset(full_dataset, subset_indices)
    # Reuse same DataLoader class + collate as the original val_loader
    DLCls = type(full_loader)
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=0)
    if hasattr(full_loader, "collate_fn") and full_loader.collate_fn is not None:
        loader_kwargs["collate_fn"] = full_loader.collate_fn
    try:
        subset_loader = DLCls(subset_ds, **loader_kwargs)
    except Exception as e:
        logger.warning(f"      failed to build with {DLCls.__name__}({loader_kwargs}): {e}")
        logger.warning(f"      falling back to torch_geometric.loader.DataLoader")
        from torch_geometric.loader import DataLoader as PyGDataLoader
        subset_loader = PyGDataLoader(subset_ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=0)
    logger.info(f"      subset_loader: class={type(subset_loader).__name__}  "
                f"n_batches={len(subset_loader)}")

    # ── [4] K-sweep loop ────────────────────────────────────────────
    logger.info(f"\n[4/5] running K={args.K} sweeps ...")
    pred_per_name = {n: {"frac":  [None] * args.K,
                         "types": [None] * args.K} for n in subset_names}
    static_info = {}  # name → {true_frac, true_types, eval_cutoff, mp_id}

    seeds = []
    wall_per_K = []

    overall_t0 = time.time()
    for k in range(args.K):
        seed_k = args.seed_base + k
        seeds.append(seed_k)
        torch.manual_seed(seed_k)
        torch.cuda.manual_seed_all(seed_k)
        np.random.seed(seed_k)

        logger.info(f"\n  ── K-sweep {k+1}/{args.K}  (seed={seed_k}) ──")
        t_k = time.time()
        n_none = 0
        n_recorded_this_k = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(subset_loader, desc=f"K={k}")):
                if batch is None:
                    n_none += 1
                    continue
                batch = batch.to(device)
                try:
                    traj_final, _ = model.sample(batch)
                except Exception as e:
                    logger.error(f"    batch {batch_idx} model.sample FAIL: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    raise

                num_atoms      = batch.num_atoms
                pred_frac_all  = traj_final['frac_coords'].cpu()
                pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1
                true_frac_all  = batch.frac_coords.cpu()
                true_types_all = batch.atom_types.cpu()
                eval_cutoffs   = batch.eval_cutoff.cpu()

                splits_p_frac  = torch.split(pred_frac_all,  num_atoms.tolist())
                splits_p_types = torch.split(pred_types_all, num_atoms.tolist())
                splits_t_frac  = torch.split(true_frac_all,  num_atoms.tolist())
                splits_t_types = torch.split(true_types_all, num_atoms.tolist())

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
                    )
                    if sname is None:
                        continue
                    sname = str(sname)
                    if sname not in pred_per_name:
                        # shouldn't happen — Subset filtered
                        continue

                    pred_per_name[sname]["frac"][k]  = splits_p_frac[i].clone()
                    pred_per_name[sname]["types"][k] = splits_p_types[i].clone()
                    n_recorded_this_k += 1

                    if k == 0:
                        mpid = (
                            getattr(data, 'mp_id', None) if data is not None else None
                        ) or (
                            batch_mpids[i] if (batch_mpids is not None and i < len(batch_mpids)) else ""
                        )
                        ec_i = eval_cutoffs[i] if eval_cutoffs.dim() > 0 else eval_cutoffs
                        static_info[sname] = {
                            "true_frac":   splits_t_frac[i].clone(),
                            "true_types":  splits_t_types[i].clone(),
                            "eval_cutoff": float(ec_i.item()),
                            "mp_id":       str(mpid),
                        }

        wall_k = time.time() - t_k
        wall_per_K.append(wall_k)
        logger.info(f"    sweep done: {n_recorded_this_k}/{n_target} samples recorded; "
                    f"None_batches={n_none}; wall={wall_k:.0f}s ({wall_k/60:.1f} min)")

    overall_wall = time.time() - overall_t0

    # ── [5] stack + save ─────────────────────────────────────────────
    logger.info(f"\n[5/5] stacking + saving ...")
    final_names         = []
    final_mp_ids        = []
    final_tiers         = []
    final_eval_cutoffs  = []
    final_true_frac     = []
    final_true_types    = []
    final_pred_frac_K   = []
    final_pred_types_K  = []
    n_dropped = 0
    for n in subset_names:
        if n not in static_info:
            n_dropped += 1
            continue
        if any(pred_per_name[n]["frac"][k]  is None for k in range(args.K)) or \
           any(pred_per_name[n]["types"][k] is None for k in range(args.K)):
            n_dropped += 1
            continue
        si = static_info[n]
        ec = si["eval_cutoff"]
        if   ec < 3.0: tier = "A"
        elif ec < 4.0: tier = "B"
        elif ec < 5.0: tier = "C"
        else:          tier = "D"
        final_names.append(n)
        final_mp_ids.append(si["mp_id"])
        final_tiers.append(tier)
        final_eval_cutoffs.append(ec)
        final_true_frac.append(si["true_frac"])
        final_true_types.append(si["true_types"])
        final_pred_frac_K.append(torch.stack(pred_per_name[n]["frac"],  dim=0))   # (K, 20, 3)
        final_pred_types_K.append(torch.stack(pred_per_name[n]["types"], dim=0))  # (K, 20)

    logger.info(f"      kept={len(final_names)} / {n_target}  (dropped={n_dropped})")
    if n_dropped > 0:
        logger.warning(f"⚠️  {n_dropped} samples dropped (probably collate-None during some sweep)")

    out = {
        "K":                   args.K,
        "n_target":            n_target,
        "n_kept":              len(final_names),
        "n_dropped":           n_dropped,
        "seed_base":           args.seed_base,
        "seeds":               seeds,
        "wall_per_K":          wall_per_K,
        "overall_wall":        overall_wall,
        "checkpoint":          args.ckpt,
        "ckpt_md5_full":       actual_md5,
        "subset_csv_path":     os.path.abspath(args.subset_csv),
        "L":                   L,
        "batch_size":          args.batch_size,
        "device":              str(device),
        "torch_version":       __import__("torch").__version__,
        "torch_cuda_build":    __import__("torch").version.cuda,
        "sample_names":        final_names,
        "mp_ids":              final_mp_ids,
        "tiers":               final_tiers,
        "eval_cutoffs":        final_eval_cutoffs,
        "true_frac_coords":    final_true_frac,
        "true_atom_types":     final_true_types,
        "pred_frac_coords_K":  final_pred_frac_K,
        "pred_atom_types_K":   final_pred_types_K,
    }
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(out, args.out_pt)
    sz = os.path.getsize(args.out_pt) / 1e6
    logger.info(f"\n      saved → {args.out_pt}  ({sz:.1f} MB)")

    logger.info(f"\n      total wall: {overall_wall:.0f}s = {overall_wall/60:.1f} min "
                f"= {overall_wall/3600:.2f} h")
    logger.info(f"      per-K wall: " + ", ".join(f"{w:.0f}s" for w in wall_per_K))
    logger.info("\n" + "=" * 60)
    logger.info("multisample COMPLETE.  Next: multisample_aggregate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
