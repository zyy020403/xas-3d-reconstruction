# experiment/step4/diag.py

import os, sys, torch

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP3_DIR = os.path.join(EXPERIMENT_DIR, "step3")
STEP1_DIR = os.path.join(EXPERIMENT_DIR, "step1")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, STEP3_DIR)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':

    from diffcsp.pl_modules.diffusion import CSPDiffusion
    from xas_datamodule import XASDataModule

    CKPT = os.path.join(EXPERIMENT_DIR, "step3", "training_output",
                        "epochepoch=189-valval_loss=0.9522.ckpt")

    print("Loading model ...")
    model = CSPDiffusion.load_from_checkpoint(CKPT, map_location="cpu", strict=False)
    model.eval()
    model.keep_lattice = True
    model.cuda()
    print("Model loaded (keep_lattice=True forced).\n")

    print("Loading DataModule ...")
    dm = XASDataModule(
        batch_size={"train": 4, "val": 4, "test": 4},
        num_workers=0,
        step1_dir=STEP1_DIR
    )
    dm.setup("fit")
    loader = dm.val_dataloader()
    print("DataModule ready.\n")

    batch = next(iter(loader))
    batch = batch.cuda()

    gt_lengths = batch.lengths
    print("GT lengths (first batch):")
    for i in range(gt_lengths.shape[0]):
        print(f"  sample {i}: {[f'{v:.2f}' for v in gt_lengths[i].tolist()]}")
    print()

    print("Testing with GT lattice (step_lr=1e-5) ...")
    with torch.no_grad():
        outputs, _ = model.sample(batch, step_lr=1e-5)

    # 验证晶格确实用了 GT
    from diffcsp.common.data_utils import lattice_params_to_matrix_torch
    gt_lats  = lattice_params_to_matrix_torch(batch.lengths, batch.angles).cpu()
    pred_lats = outputs["lattices"].cpu()
    lat_diff = (pred_lats - gt_lats).abs().max().item()
    print(f"\n  Max lattice diff vs GT = {lat_diff:.6f}  (should be ~0)")

    # 坐标评估
    pred_coords = outputs["frac_coords"].cpu()   # [total_atoms, 3]
    gt_coords   = batch.frac_coords.cpu()

    # 原始误差（不考虑周期性）
    raw_err = (pred_coords - gt_coords).abs().mean().item()
    print(f"  Raw coord error        = {raw_err:.4f}")

    # ★ PBC 修正误差（考虑周期性边界，才是真实误差）
    diff    = (pred_coords - gt_coords + 0.5) % 1.0 - 0.5
    pbc_err = diff.abs().mean().item()
    print(f"  PBC-corrected error    = {pbc_err:.4f}  ← 这个才是真实误差")

    print()
    if pbc_err < 0.1:
        print("  ✓ 坐标预测很好，只需修晶格预测")
    elif pbc_err < 0.2:
        print("  △ 坐标凑合，晶格是主要问题")
    elif pbc_err < 0.25:
        print("  ✗ 坐标较差，模型可能需要更多训练")
    else:
        print("  ✗✗ 坐标接近随机猜（基线=0.25），模型训练有问题")