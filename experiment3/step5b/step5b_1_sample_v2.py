# step5b_1_sample_v2.py
# Step 5b v2 — Experiment 3 val/test 采样（Step4f checkpoint）
# ============================================================
# 相比 step5b_1_sample.py 的改动：
#   1. CKPT_PATH  → experiment3/step4f/checkpoints/best_*.ckpt（epoch 57）
#   2. STEP5B_DIR → experiment3/step5b_v2/
#   3. 模型 import → diffusion_w_type_xas_exp3_step4f（Step4f 版本）
#   4. ★ TypeClassifier 输入修复：
#      Step4f 的 TypeClassifier 接受 latent(256) + feff_raw(73) = 329 维，
#      run_sampling() 中的 logits 计算同步改为拼接 feff_features。
# 其余逻辑（DataLoader、min-image、collate_fn）与 step5b_1_sample.py 完全相同。
# ============================================================

import os, sys, logging, warnings, json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT     = os.path.join(PROJECT_ROOT, "experiment2")
EXP3_ROOT     = os.path.join(PROJECT_ROOT, "experiment3")
STEP1_DIR     = os.path.join(EXP2_ROOT, "step1")
STEP2_DIR     = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR     = os.path.join(EXP2_ROOT, "step3")
STEP3C_DIR    = os.path.join(EXP3_ROOT, "step3c")
STEP4F_DIR    = os.path.join(EXP3_ROOT, "step4f")           # ★ 改为 step4f
STEP5B_DIR    = os.path.join(EXP3_ROOT, "step5b_v2")        # ★ 改为 step5b_v2

# ★ step4f checkpoint（epoch 57，val_type_acc=0.603）
CKPT_PATH     = os.path.join(STEP4F_DIR, "checkpoints",
                              "best_epoch=057-val_type_acc=0.6030-val_coord_loss=0.7524.ckpt")
VOCAB_PATH    = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
DATA_ROOT     = os.path.join(PROJECT_ROOT, "site_dataset_Fe_only_oxide_one_site")
FEFF_CSV      = os.path.join(PROJECT_ROOT, "tesst_feff_features_all_full_v4.csv")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
VAL_IDS       = os.path.join(STEP1_DIR, "val_ids.txt")
TEST_IDS      = os.path.join(STEP1_DIR, "test_ids.txt")

L             = 6.0
BATCH_SIZE    = 8

os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# sys.path 设置 — 必须在所有自定义 import 之前
for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR, STEP3C_DIR, STEP4F_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def get_best_ckpt():
    """优先使用指定 CKPT_PATH；若不存在则扫描 checkpoints 目录取 val_type_acc 最高。"""
    if os.path.exists(CKPT_PATH):
        return CKPT_PATH
    import glob, re
    # 优先匹配 best_epoch=* 前缀
    ckpts = glob.glob(os.path.join(STEP4F_DIR, "checkpoints", "best_*.ckpt"))
    if not ckpts:
        ckpts = glob.glob(os.path.join(STEP4F_DIR, "checkpoints", "epoch=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"在 {STEP4F_DIR}/checkpoints 中未找到任何 checkpoint！")

    def _parse_type_acc(p):
        m = re.search(r'val_type_acc=([\d.]+)', os.path.basename(p))
        return float(m.group(1)) if m else -1.0

    best = max(ckpts, key=_parse_type_acc)
    return best


def load_model(ckpt_path, logger):
    """
    加载 Step4f CSPDiffusion 模型。
    优先尝试 PL load_from_checkpoint；若失败则 fallback 到 hydra 手动实例化。
    """
    # ★ 从 step4f 目录导入（Step4f 修复版 TypeClassifier）
    from diffusion_w_type_xas_exp3_step4f import CSPDiffusion as CrystDiffPLModule

    # ── 方案 A：PL load_from_checkpoint ────────────────────────────────────
    try:
        logger.info("尝试 load_from_checkpoint ...")
        model = CrystDiffPLModule.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            vocab_path=VOCAB_PATH,
            strict=False,
        )
        logger.info("  load_from_checkpoint 成功")
        return model
    except Exception as e:
        logger.warning(f"  load_from_checkpoint 失败: {e}")

    # ── 方案 B：手动 hydra 实例化 ─────────────────────────────────────────
    logger.info("尝试 hydra 实例化 ...")
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import torch

    conf_candidates = [
        os.path.join(STEP4F_DIR,  "conf_xas"),
        os.path.join(STEP3C_DIR,  "conf_xas"),
        os.path.join(STEP3_DIR,   "conf_xas"),
        os.path.join(EXP3_ROOT,   "conf_xas"),
    ]
    conf_dir = next((d for d in conf_candidates if os.path.isdir(d)), None)
    if conf_dir is None:
        raise FileNotFoundError(f"找不到 conf 目录，已搜索: {conf_candidates}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(
            config_dir=os.path.join(conf_dir, "model"),
            job_name="sample5b_v2", version_base=None):
        _raw = compose(config_name="diffusion_xas")
    model_cfg = OmegaConf.create({
        "model": OmegaConf.to_container(_raw, resolve=False)}).model

    if hasattr(model_cfg, 'vocab_path'):
        model_cfg.vocab_path = VOCAB_PATH
    else:
        OmegaConf.update(model_cfg, "vocab_path", VOCAB_PATH, merge=True)

    optim_cfg = OmegaConf.create({
        "optimizer":         {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler":  False,
        "lr_scheduler":      None,
    })

    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    logger.info("  hydra 实例化 + 手动权重加载成功")
    return model


def build_loader(ids_file, split_name, logger):
    """
    构建 DataLoader，过滤 None 样本。
    使用 torch 原生 DataLoader + 自定义 collate_fn（PyG DataLoader 忽略 collate_fn）。
    """
    from xas_local_dataset_L6 import XASLocalStructureDataset
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch

    dataset = XASLocalStructureDataset(
        data_root     = DATA_ROOT,
        inventory_csv = INVENTORY_CSV,
        ids_file      = ids_file,
        feff_feat_csv = FEFF_CSV,
        L             = L,
    )
    logger.info(f"  {split_name} dataset: {len(dataset)} 样本（含可能 None）")

    def collate_fn(batch):
        valid = [b for b in batch if b is not None]
        if not valid:
            return None
        return Batch.from_data_list(valid)

    return DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )


def run_sampling(model, loader, split_name, device, logger):
    """
    对 loader 中所有 batch 做采样，返回 predictions dict。

    ★ Step4f 修复：TypeClassifier 输入为 latent(256) + feff_raw(73) = 329 维，
      logits 计算与 model.sample() 内部保持完全一致。
    """
    import torch
    from tqdm import tqdm

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    inv_vocab = {v: int(k) for k, v in vocab.items()}
    N_ELEM    = len(vocab)

    all_mp_ids      = []
    all_pred_frac   = []
    all_pred_types  = []
    all_pred_logits = []   # (20, N_elem) per sample
    all_true_frac   = []
    all_true_types  = []
    all_eval_cutoff = []

    logger.info(f"\n{'─'*40}")
    logger.info(f"采样 {split_name} 集 (L={L}, device={device}) ...")

    n_processed = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Sampling {split_name}")):
        if batch is None:
            continue
        batch = batch.to(device)

        with torch.no_grad():
            # ── 扩散采样 ────────────────────────────────────────────────────
            traj_final, traj_stack = model.sample(batch)

            # ── ★ Step4f 修复：拼接 feff_features，与 model.sample() 内部一致 ──
            _latent   = model.spectrum_encoder(
                batch.xmu_xanes,
                batch.chi1,
                batch.feff_features,
            )                                                   # (B, 256)
            _feff_raw = batch.feff_features.squeeze(1)          # (B, 73)
            _type_inp = torch.cat([_latent, _feff_raw], dim=-1) # (B, 329)
            _logits   = model.type_classifier(_type_inp)        # (B, 20, N_elem)
            _logits_np = _logits.cpu()

        # ── 拆解 batch ────────────────────────────────────────────────────
        num_atoms     = batch.num_atoms.cpu()
        B             = num_atoms.shape[0]

        pred_frac_all  = traj_final['frac_coords'].cpu()    # (N_total, 3)
        pred_types_all = traj_final['atom_types'].cpu()     # (N_total,)

        assert pred_types_all.min() >= 1 and pred_types_all.max() <= 94, (
            f"采样原子序数超界: [{pred_types_all.min()},{pred_types_all.max()}]"
        )

        true_frac_all  = batch.frac_coords.cpu()
        true_types_all = batch.atom_types.cpu()
        eval_cutoffs   = batch.eval_cutoff.cpu()

        splits_p_frac  = torch.split(pred_frac_all,  num_atoms.tolist())
        splits_p_types = torch.split(pred_types_all, num_atoms.tolist())
        splits_t_frac  = torch.split(true_frac_all,  num_atoms.tolist())
        splits_t_types = torch.split(true_types_all, num_atoms.tolist())
        splits_logits  = [_logits_np[i] for i in range(B)]

        data_list = batch.to_data_list()
        for i, data in enumerate(data_list):
            mp_id = getattr(data, 'mp_id', f"unk_{batch_idx}_{i}")
            all_mp_ids.append(mp_id)
            all_pred_frac.append(splits_p_frac[i])
            all_pred_types.append(splits_p_types[i])
            all_pred_logits.append(splits_logits[i])
            all_true_frac.append(splits_t_frac[i])
            all_true_types.append(splits_t_types[i])
            all_eval_cutoff.append(eval_cutoffs[i].item())
            n_processed += 1

    logger.info(f"  采样完成：{n_processed} 个样本")

    predictions = {
        'mp_id':             all_mp_ids,
        'pred_frac_coords':  all_pred_frac,
        'pred_atom_types':   all_pred_types,
        'pred_type_logits':  all_pred_logits,    # list of (20, N_elem)
        'true_frac_coords':  all_true_frac,
        'true_atom_types':   all_true_types,
        'eval_cutoff':       all_eval_cutoff,
        'L':                 L,
        'checkpoint':        CKPT_PATH,
        'vocab_path':        VOCAB_PATH,
        'n_elem':            N_ELEM,
    }
    return predictions


if __name__ == "__main__":
    import torch
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Step 5b v2  Experiment 3 — val/test 采样（Step4f）")
    logger.info("=" * 60)

    os.makedirs(STEP5B_DIR, exist_ok=True)

    # ── 1. 确定 checkpoint ────────────────────────────────────────────────────
    ckpt_path = get_best_ckpt()
    logger.info(f"使用 checkpoint：{ckpt_path}")

    # ── 2. 加载模型 ───────────────────────────────────────────────────────────
    model  = load_model(ckpt_path, logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()
    logger.info(f"模型加载完毕，运行设备：{device}")

    # ── 3. Val 集采样 ─────────────────────────────────────────────────────────
    logger.info("\n构建 val 数据集...")
    val_loader = build_loader(VAL_IDS, "val", logger)
    val_preds  = run_sampling(model, val_loader, "val", device, logger)

    val_out = os.path.join(STEP5B_DIR, "predictions_val.pt")
    torch.save(val_preds, val_out)
    logger.info(f"Val 预测保存 → {val_out}  ({len(val_preds['mp_id'])} 样本)")

    # ── 4. Test 集采样 ────────────────────────────────────────────────────────
    logger.info("\n构建 test 数据集...")
    test_loader = build_loader(TEST_IDS, "test", logger)
    test_preds  = run_sampling(model, test_loader, "test", device, logger)

    test_out = os.path.join(STEP5B_DIR, "predictions_test.pt")
    torch.save(test_preds, test_out)
    logger.info(f"Test 预测保存 → {test_out}  ({len(test_preds['mp_id'])} 样本)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5b v2 采样完成")
    logger.info(f"  val  → {val_out}")
    logger.info(f"  test → {test_out}")
    logger.info("下一步：运行 step5b_2_metrics_v2.py")
    logger.info("=" * 60)
