# step4_3_sample.py
# Step 4.3 — val/test 集采样
# ============================================================
# 使用最优 checkpoint 对 val 集和 test 集进行采样。
# 每个样本生成 1 个预测结构。
# 输出：predictions_val.pt / predictions_test.pt
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
CKPT_DIR     = os.path.join(STEP4_DIR, "checkpoints")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

BATCH_SIZE_SAMPLE = 8   # 采样时 batch 小一点，避免 OOM

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
    logger.info("Step 4.3  val/test 集采样")
    logger.info("=" * 60)

    # ── 1. 确定最优 checkpoint ─────────────────────────────────────────────
    best_path_file = os.path.join(STEP4_DIR, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file) as f:
            ckpt_path = f.read().strip()
    else:
        # 手动扫描，取 val_loss 最低的
        import glob
        ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt"))
        if not ckpts:
            logger.error(f"❌ {CKPT_DIR} 中未找到 checkpoint！请先运行训练。")
            sys.exit(1)
        # 从文件名解析 val_loss
        def _parse_val_loss(p):
            import re
            m = re.search(r'val_loss=([\d.]+)', os.path.basename(p))
            return float(m.group(1)) if m else 9999.0
        ckpt_path = min(ckpts, key=_parse_val_loss)

    logger.info(f"使用 checkpoint：{ckpt_path}")

    # ── 2. 加载模型 ────────────────────────────────────────────────────────
    CONF_DIR = os.path.join(STEP3_DIR, "conf_xas")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                               job_name="sample", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False, "lr_scheduler": None})

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("模型权重加载成功")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    # ── 3. DataModule ──────────────────────────────────────────────────────
    datamodule = XASDataModule(
        batch_size=BATCH_SIZE_SAMPLE, num_workers=0, L=12.0)

    # ── 4. 采样函数 ────────────────────────────────────────────────────────
    def run_sampling(loader, split_name):
        logger.info(f"\n{'─'*40}")
        logger.info(f"采样 {split_name} 集...")

        all_mp_ids          = []
        all_pred_frac       = []
        all_pred_types      = []
        all_true_frac       = []
        all_true_types      = []
        all_eval_cutoff     = []

        n_batches = len(loader)
        for batch_idx, batch in enumerate(tqdm(loader,
                                               desc=f"Sampling {split_name}")):
            if batch is None:
                continue
            batch = batch.to(device)

            with torch.no_grad():
                traj_final, traj_stack = model.sample(batch)

            # pred_frac_coords: (N_atoms_total, 3)，需要按图拆分
            num_atoms     = batch.num_atoms              # (B,) int tensor
            pred_frac_all = traj_final['frac_coords'].cpu()   # (N_total, 3)
            # 采样结果是 % 1. 后的，映射回 [-0.5, 0.5] 以对齐训练目标
            pred_frac_centered = pred_frac_all - 0.5
            # traj_stack['atom_types'] shape=(T+1, N_total)，取最终步 traj[0]
            pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1
            true_frac_all  = batch.frac_coords.cpu()          # (N_total, 3)
            true_types_all = batch.atom_types.cpu()           # (N_total,)
            eval_cutoffs   = batch.eval_cutoff.cpu()          # (B,)

            # 按图拆分
            splits_p_frac  = torch.split(pred_frac_centered, num_atoms.tolist())
            splits_p_types = torch.split(pred_types_all, num_atoms.tolist())
            splits_t_frac  = torch.split(true_frac_all, num_atoms.tolist())
            splits_t_types = torch.split(true_types_all, num_atoms.tolist())

            # mp_id：从 batch 取（PyG Data 对象的 mp_id 属性）
            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                mp_id = getattr(data, 'mp_id', f"unk_{batch_idx}_{i}")
                all_mp_ids.append(mp_id)
                all_pred_frac.append(splits_p_frac[i])       # (20, 3)
                all_pred_types.append(splits_p_types[i])     # (20,)
                all_true_frac.append(splits_t_frac[i])       # (20, 3)
                all_true_types.append(splits_t_types[i])     # (20,)
                all_eval_cutoff.append(eval_cutoffs[i].item())

        predictions = {
            'mp_id':             all_mp_ids,
            'pred_frac_coords':  all_pred_frac,
            'pred_atom_types':   all_pred_types,
            'true_frac_coords':  all_true_frac,
            'true_atom_types':   all_true_types,
            'eval_cutoff':       all_eval_cutoff,
            'L':                 12.0,
            'checkpoint':        ckpt_path,
        }

        out_path = os.path.join(STEP4_DIR, f"predictions_{split_name}.pt")
        torch.save(predictions, out_path)
        logger.info(f"  保存 {len(all_mp_ids)} 个样本 → {out_path}")
        return out_path

    # ── 5. 对 val 和 test 分别采样 ────────────────────────────────────────
    datamodule.setup("fit")
    val_out = run_sampling(datamodule.val_dataloader(), "val")

    datamodule.setup("test")
    test_out = run_sampling(datamodule.test_dataloader(), "test")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4.3 采样完成")
    logger.info(f"  val  → {val_out}")
    logger.info(f"  test → {test_out}")
    logger.info("下一步：运行 step4_4_compute_metrics.py")
    logger.info("=" * 60)