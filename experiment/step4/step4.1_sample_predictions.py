# =============================================================================
# 脚本编号: step4.1
# 脚本名称: step4.1_sample_predictions.py
# =============================================================================

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP4_DIR = os.path.join(EXPERIMENT_DIR, "step4")
os.makedirs(STEP4_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

# ★ 修复：目录改为 finetune2_output，文件名与实际一致
CKPT_PATH = os.path.join(
    EXPERIMENT_DIR, "step4", "finetune2_output",
    "epoch=322-val_loss=0.9134.ckpt"
)
HOLDOUT_IDS_FILE = os.path.join(EXPERIMENT_DIR, "step1", "holdout_1000_ids.txt")
VAL_IDS_FILE     = os.path.join(EXPERIMENT_DIR, "step1", "val_ids.txt")
TEST_IDS_FILE    = os.path.join(EXPERIMENT_DIR, "step1", "test_ids.txt")

STEP_LR = 1e-5

with open(HOLDOUT_IDS_FILE, "r") as f:
    HOLDOUT_IDS = set(line.strip() for line in f if line.strip())
print(f"[Guard] Loaded {len(HOLDOUT_IDS)} holdout IDs — these will NEVER be sampled.")

STEP3_DIR = os.path.join(EXPERIMENT_DIR, "step3")
sys.path.insert(0, STEP3_DIR)

from xas_datamodule import XASDataModule
from diffcsp.pl_modules.diffusion import CSPDiffusion

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from eval_utils import lattices_to_params_shape


def load_model(ckpt_path: str) -> CSPDiffusion:
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    model = CSPDiffusion.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("[Model] Moved to CUDA.")
    else:
        print("[Model] WARNING: CUDA not available, running on CPU (slow).")
    return model


@torch.no_grad()
def run_sampling(loader, model, split_name: str, holdout_ids: set) -> dict:
    results = {}
    skipped_holdout = 0
    skipped_error   = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Sampling [{split_name}]")):
        if hasattr(batch, "mp_id"):
            batch_mp_ids = batch.mp_id
        elif hasattr(batch, "mp_ids"):
            batch_mp_ids = batch.mp_ids
        else:
            batch_mp_ids = [f"unk_{batch_idx}_{i}" for i in range(batch.num_graphs)]

        for mid in batch_mp_ids:
            if mid in holdout_ids:
                print(f"[GUARD] Skipping holdout mp_id: {mid}")
                skipped_holdout += 1

        valid_indices = [
            i for i, mid in enumerate(batch_mp_ids)
            if mid not in holdout_ids
        ]
        if len(valid_indices) == 0:
            continue

        if torch.cuda.is_available():
            batch = batch.cuda()

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                outputs, _traj = model.sample(batch, step_lr=STEP_LR)
        except Exception as e:
            print(f"[ERROR] Sampling failed for batch {batch_idx}: {e}")
            skipped_error += 1
            continue

        pred_frac_coords = outputs["frac_coords"].detach().cpu()
        pred_num_atoms   = outputs["num_atoms"].detach().cpu()
        pred_atom_types  = outputs["atom_types"].detach().cpu()
        pred_lattices    = outputs["lattices"].detach().cpu()

        pred_lengths, pred_angles = lattices_to_params_shape(pred_lattices)

        gt_frac_coords = batch.frac_coords.detach().cpu()
        gt_num_atoms   = batch.num_atoms.detach().cpu()
        gt_atom_types  = batch.atom_types.detach().cpu()

        if hasattr(batch, "lattice"):
            gt_lattices = batch.lattice.detach().cpu()
        elif hasattr(batch, "lengths") and hasattr(batch, "angles"):
            gt_lengths_raw = batch.lengths.detach().cpu()
            gt_angles_raw  = batch.angles.detach().cpu()
            gt_lattices    = None
        else:
            raise AttributeError(
                "batch 中找不到 lattice / lengths+angles，"
                "请检查 XASDataModule 的输出字段"
            )

        if gt_lattices is not None:
            gt_lengths_raw, gt_angles_raw = lattices_to_params_shape(gt_lattices)

        batch_size = pred_num_atoms.shape[0]
        pred_start = 0
        gt_start   = 0

        for i in range(batch_size):
            mp_id = batch_mp_ids[i]
            if mp_id in holdout_ids:
                pred_start += pred_num_atoms[i].item()
                gt_start   += gt_num_atoms[i].item()
                continue

            n_pred = pred_num_atoms[i].item()
            n_gt   = gt_num_atoms[i].item()

            entry = {
                "pred_frac_coords": pred_frac_coords[pred_start : pred_start + n_pred],
                "pred_lengths":     pred_lengths[i],
                "pred_angles":      pred_angles[i],
                "pred_atom_types":  pred_atom_types[pred_start : pred_start + n_pred],
                "gt_frac_coords":   gt_frac_coords[gt_start : gt_start + n_gt],
                "gt_lengths":       gt_lengths_raw[i],
                "gt_angles":        gt_angles_raw[i],
                "gt_atom_types":    gt_atom_types[gt_start : gt_start + n_gt],
                "n_atoms":          n_gt,
            }
            results[mp_id] = entry

            pred_start += n_pred
            gt_start   += n_gt

    print(f"[{split_name}] Done. "
          f"Collected {len(results)} compounds. "
          f"Holdout skipped: {skipped_holdout}, Error skipped: {skipped_error}.")
    return results


def main():
    model = load_model(CKPT_PATH)

    print("[Data] Initializing XASDataModule ...")
    datamodule = XASDataModule(
        batch_size  = {"train": 16, "val": 8, "test": 8},
        num_workers = {"train": 0,  "val": 0, "test": 0},
        step1_dir   = os.path.join(EXPERIMENT_DIR, "step1"),
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    val_loader  = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("\n===== Val Set Sampling =====")
    val_predictions = run_sampling(val_loader, model, "val", HOLDOUT_IDS)
    val_save_path = os.path.join(STEP4_DIR, "predictions_val.pt")
    torch.save(val_predictions, val_save_path)
    print(f"[Saved] {val_save_path}  ({len(val_predictions)} compounds)")

    print("\n===== Test Set Sampling =====")
    test_predictions = run_sampling(test_loader, model, "test", HOLDOUT_IDS)
    test_save_path = os.path.join(STEP4_DIR, "predictions_test.pt")
    torch.save(test_predictions, test_save_path)
    print(f"[Saved] {test_save_path}  ({len(test_predictions)} compounds)")

    print("\n===== Sanity Check =====")
    for split, preds in [("val", val_predictions), ("test", test_predictions)]:
        sample_id = next(iter(preds))
        sample    = preds[sample_id]
        print(f"[{split}] Total compounds: {len(preds)}")
        print(f"  Example mp_id : {sample_id}")
        print(f"  n_atoms       : {sample['n_atoms']}")
        print(f"  pred_frac_coords shape : {sample['pred_frac_coords'].shape}")
        print(f"  pred_lengths           : {sample['pred_lengths'].tolist()}")
        print(f"  pred_angles            : {sample['pred_angles'].tolist()}")
        print(f"  gt_frac_coords   shape : {sample['gt_frac_coords'].shape}")
        print(f"  gt_lengths             : {sample['gt_lengths'].tolist()}")
        print(f"  gt_angles              : {sample['gt_angles'].tolist()}")
        print()

    print("Step 4.1 complete.")


if __name__ == "__main__":
    main()