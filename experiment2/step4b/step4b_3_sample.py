# step4b_3_sample.py
# Step4b.3 — val/test 集采样（v6 坐标系配套）
# ============================================================
# v3 → step4b 修改说明：
#
#   [BUG FIX] 删除 pred_frac_centered = pred_frac_all - 0.5
#     v3 中对预测坐标做了 -0.5，将 sample() 输出的 [0,1] 折叠回 [-0.5,0.5]，
#     与旧 v5 Dataset 的 [-0.5,0.5] 配套。
#     v6 Dataset 的 true_frac 已统一为 [0,1]，pred 也应保持 [0,1]，
#     不做任何偏移，直接保存。
#
#   [路径修改] STEP4_DIR → STEP4b_DIR
#     checkpoint 从 step4b/checkpoints/ 读取
#     预测文件保存到 step4b/predictions_val.pt / predictions_test.pt
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP4b_DIR   = os.path.join(EXP2_ROOT, "step4b")           # ← 改为 step4b
CKPT_DIR     = os.path.join(STEP4b_DIR, "checkpoints")     # ← step4b/checkpoints/

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
    logger.info("Step4b.3  val/test 集采样（v6 坐标系，无 -0.5 偏移）")
    logger.info("=" * 60)

    # ── 1. 确定最优 checkpoint ─────────────────────────────────────────────
    best_path_file = os.path.join(STEP4b_DIR, "best_checkpoint_path.txt")
    if os.path.exists(best_path_file):
        with open(best_path_file) as f:
            ckpt_path = f.read().strip()
    else:
        import glob, re
        ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt"))
        if not ckpts:
            logger.error(f"❌ {CKPT_DIR} 中未找到 checkpoint！请先完成 step4b 训练。")
            sys.exit(1)
        def _parse_val_loss(p):
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
                traj_final, traj_stack = model.sample(batch)

            num_atoms      = batch.num_atoms
            pred_frac_all  = traj_final['frac_coords'].cpu()   # (N_total, 3)，[0,1]

            # ★ step4b 修正：不做 -0.5 偏移，直接保留 [0,1] 坐标
            # v3 错误代码（已删除）：
            #   pred_frac_centered = pred_frac_all - 0.5
            pred_frac_out = pred_frac_all   # 保持 [0,1]

            pred_types_all = traj_final['atom_types'].cpu().argmax(dim=-1) + 1
            true_frac_all  = batch.frac_coords.cpu()   # v6 Dataset，[0,1]
            true_types_all = batch.atom_types.cpu()
            eval_cutoffs   = batch.eval_cutoff.cpu()

            splits_p_frac  = torch.split(pred_frac_out,    num_atoms.tolist())
            splits_p_types = torch.split(pred_types_all,   num_atoms.tolist())
            splits_t_frac  = torch.split(true_frac_all,    num_atoms.tolist())
            splits_t_types = torch.split(true_types_all,   num_atoms.tolist())

            data_list = batch.to_data_list()
            for i, data in enumerate(data_list):
                mp_id = getattr(data, 'mp_id', f"unk_{batch_idx}_{i}")
                all_mp_ids.append(mp_id)
                all_pred_frac.append(splits_p_frac[i])
                all_pred_types.append(splits_p_types[i])
                all_true_frac.append(splits_t_frac[i])
                all_true_types.append(splits_t_types[i])
                all_eval_cutoff.append(eval_cutoffs[i].item())

        os.makedirs(STEP4b_DIR, exist_ok=True)
        predictions = {
            'mp_id':            all_mp_ids,
            'pred_frac_coords': all_pred_frac,
            'pred_atom_types':  all_pred_types,
            'true_frac_coords': all_true_frac,
            'true_atom_types':  all_true_types,
            'eval_cutoff':      all_eval_cutoff,
            'L':                12.0,
            'checkpoint':       ckpt_path,
            'coord_system':     '[0,1]',   # 记录坐标系版本，方便 debug
        }

        out_path = os.path.join(STEP4b_DIR, f"predictions_{split_name}.pt")
        torch.save(predictions, out_path)
        logger.info(f"  保存 {len(all_mp_ids)} 个样本 → {out_path}")
        return out_path

    # ── 5. val + test ────────────────────────────────────────────────────
    datamodule.setup("fit")
    val_out = run_sampling(datamodule.val_dataloader(), "val")

    datamodule.setup("test")
    test_out = run_sampling(datamodule.test_dataloader(), "test")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step4b.3 采样完成")
    logger.info(f"  val  → {val_out}")
    logger.info(f"  test → {test_out}")
    logger.info("下一步：运行 step4b_4_compute_metrics.py")
    logger.info("=" * 60)
