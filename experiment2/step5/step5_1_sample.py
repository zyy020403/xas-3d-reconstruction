# step5_1_sample.py
# Step 5.1 — Holdout 集采样（盲测）
# ============================================================
# 改自 step4_3_sample.py，修改两处：
#   1. ID 文件 → holdout_1000_ids.txt
#   2. 输出路径 → experiment2/step5/predictions_holdout.pt
#   3. checkpoint 路径 → step4d best checkpoint（epoch=249）
#   4. L=6.0（与 Step4d / xas_local_dataset_L6.py 一致）
#
# 严格禁止：任何形式的模型修改、重训、fine-tune
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4d_DIR   = os.path.join(EXP2_ROOT, "step4d")
STEP5_DIR    = os.path.join(EXP2_ROOT, "step5")
STEP1_DIR    = os.path.join(EXP2_ROOT, "step1")

# ── 关键路径常量（来自 STEP5_HANDOFF.md）─────────────────────────────────────
DATA_ROOT     = os.path.join(PROJECT_ROOT, "site_dataset_Fe_only_oxide_one_site")
FEFF_FEAT_CSV = os.path.join(PROJECT_ROOT, "tesst_feff_features_all_full_v4.csv")
HOLDOUT_IDS   = os.path.join(STEP1_DIR, "holdout_1000_ids.txt")
CKPT_PATH     = os.path.join(STEP4d_DIR, "checkpoints",
                              "epoch=249-val_loss=0.8554.ckpt")
L             = 6.0
N_NEIGHBORS   = 20

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

BATCH_SIZE_SAMPLE = 8

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step 5.1  Holdout 集采样（盲测，禁止修改模型）")
    logger.info("=" * 60)

    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from tqdm import tqdm

    # 导入 Step4d 使用的 Dataset（L=6, min-image 版本）
    # 文件位于 experiment2/step3/xas_local_dataset_L6.py
    # 若你的文件名是 xas_local_dataset.py，请将下行改为 from xas_local_dataset import ...
    try:
        from xas_local_dataset_L6 import XASLocalStructureDataset
        logger.info("Dataset 导入自 xas_local_dataset_L6")
    except ImportError:
        from xas_local_dataset import XASLocalStructureDataset
        logger.info("Dataset 导入自 xas_local_dataset")

    # ── 1. 确认 checkpoint ────────────────────────────────────────────────
    if not os.path.exists(CKPT_PATH):
        logger.error(f"❌ checkpoint 不存在：{CKPT_PATH}")
        logger.error("请确认文件路径，或将 CKPT_PATH 修改为正确路径。")
        sys.exit(1)
    logger.info(f"使用 checkpoint：{CKPT_PATH}")

    # ── 2. 加载模型 ────────────────────────────────────────────────────────
    CONF_DIR = os.path.join(STEP3_DIR, "conf_xas")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="sample_holdout", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None})

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("模型权重加载成功（strict=False）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()
    logger.info(f"运行设备：{device}")

    # ── 3. 构建 Holdout DataLoader ─────────────────────────────────────────
    logger.info(f"加载 Holdout IDs：{HOLDOUT_IDS}")
    if not os.path.exists(HOLDOUT_IDS):
        logger.error(f"❌ 找不到 {HOLDOUT_IDS}")
        sys.exit(1)

    INVENTORY_CSV   = os.path.join(STEP1_DIR, "data_inventory.csv")
    FEFF_SCALER_PKL = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")

    holdout_ds = XASLocalStructureDataset(
        data_root        = DATA_ROOT,
        inventory_csv    = INVENTORY_CSV,
        ids_file         = HOLDOUT_IDS,
        feff_feat_csv    = FEFF_FEAT_CSV,
        feff_scaler_path = FEFF_SCALER_PKL,
        L                = L,
    )
    logger.info(f"Holdout Dataset 样本数：{len(holdout_ds)}")

    def collate_fn_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)

    holdout_loader = PyGDataLoader(
        holdout_ds,
        batch_size  = BATCH_SIZE_SAMPLE,
        shuffle     = False,
        num_workers = 0,        # 严格遵守：num_workers=0
        collate_fn  = collate_fn_skip_none,
    )
    logger.info(f"DataLoader 构建完成，共 {len(holdout_loader)} 个 batch")

    # ── 4. 采样 ────────────────────────────────────────────────────────────
    all_mp_ids      = []
    all_pred_frac   = []
    all_pred_types  = []
    all_true_frac   = []
    all_true_types  = []
    all_eval_cutoff = []

    logger.info("\n开始采样 Holdout 集...")

    for batch_idx, batch in enumerate(tqdm(holdout_loader,
                                           desc="Sampling Holdout")):
        if batch is None:
            continue
        batch = batch.to(device)

        with torch.no_grad():
            traj_final, traj_stack = model.sample(batch)

        num_atoms     = batch.num_atoms
        pred_frac_all = traj_final['frac_coords'].cpu()

        # 采样坐标已是 [-0.5, 0.5]（扩散模型 sample() 内部处理），
        # 与 Dataset 的 min-image frac 坐标系一致，无需额外偏移
        pred_frac_centered = pred_frac_all

        pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1
        true_frac_all  = batch.frac_coords.cpu()
        true_types_all = batch.atom_types.cpu()
        eval_cutoffs   = batch.eval_cutoff.cpu()

        splits_p_frac  = torch.split(pred_frac_centered, num_atoms.tolist())
        splits_p_types = torch.split(pred_types_all,     num_atoms.tolist())
        splits_t_frac  = torch.split(true_frac_all,      num_atoms.tolist())
        splits_t_types = torch.split(true_types_all,     num_atoms.tolist())

        data_list = batch.to_data_list()
        for i, data in enumerate(data_list):
            mp_id = getattr(data, 'mp_id', f"unk_{batch_idx}_{i}")
            all_mp_ids.append(mp_id)
            all_pred_frac.append(splits_p_frac[i])
            all_pred_types.append(splits_p_types[i])
            all_true_frac.append(splits_t_frac[i])
            all_true_types.append(splits_t_types[i])
            all_eval_cutoff.append(eval_cutoffs[i].item())

    logger.info(f"采样完成，共处理 {len(all_mp_ids)} 个样本")

    # ── 5. 保存 ────────────────────────────────────────────────────────────
    os.makedirs(STEP5_DIR, exist_ok=True)
    out_path = os.path.join(STEP5_DIR, "predictions_holdout.pt")

    predictions = {
        'mp_id':            all_mp_ids,
        'pred_frac_coords': all_pred_frac,
        'pred_atom_types':  all_pred_types,
        'true_frac_coords': all_true_frac,
        'true_atom_types':  all_true_types,
        'eval_cutoff':      all_eval_cutoff,
        'L':                L,
        'checkpoint':       CKPT_PATH,
    }

    torch.save(predictions, out_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5.1 采样完成")
    logger.info(f"  保存 {len(all_mp_ids)} 个样本 → {out_path}")
    logger.info("下一步：运行 step5_2_compute_metrics.py")
    logger.info("=" * 60)
