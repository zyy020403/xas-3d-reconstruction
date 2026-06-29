# step4d_0_precheck.py
# Step4d — 开训前强制检查（必须全部通过才能开训）
# ============================================================
# 检查 1：frac_coords 范围 ∈ [-0.5, 0.5]
# 检查 2：有效样本丢失率 < 5%
# 检查 3：forward() loss 无 NaN/Inf，数值合理
# 检查 4：晶格矩阵确认为 diag(6, 6, 6)
# ============================================================
# Step4c agent 额外提示（坑 1 + 坑 2）：
#   运行前先确认：
#   1. xas_local_datamodule.py 的 import 行指向 xas_local_dataset_L6
#   2. 清除 __pycache__（避免读旧 .pyc）
# ============================================================

import os, sys, warnings, logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP2_DIR    = os.path.join(EXP2_ROOT, "step2")
STEP3_DIR    = os.path.join(EXP2_ROOT, "step3")
STEP1_DIR    = os.path.join(EXP2_ROOT, "step1")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
for p in [PROJECT_ROOT, STEP2_DIR, STEP3_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

L = 6.0
N_CHECK_SAMPLES = 20   # 检查前 N 个有效样本

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step4d 开训前强制检查")
    logger.info("=" * 60)

    # ── 坑 1 提示：检查 datamodule 的 import 行 ─────────────────────────
    import subprocess, re as _re
    dm_path = os.path.join(STEP3_DIR, "xas_local_datamodule.py")
    if os.path.exists(dm_path):
        with open(dm_path, encoding="utf-8", errors="ignore") as _f:
            dm_content = _f.read()
        _import_lines = [l for l in dm_content.split('\n')
                         if l.strip().startswith('from xas_local_dataset')]
        if _import_lines:
            logger.info(f"[坑1检查] datamodule import 行: {_import_lines[0].strip()}")
            if 'L6' not in _import_lines[0] and 'xas_local_dataset_L6' not in _import_lines[0]:
                logger.warning("⚠️  datamodule 可能仍在 import 旧 dataset 文件！")
                logger.warning("   请确认 datamodule 的 import 行指向 xas_local_dataset_L6")
                logger.warning("   修改后务必清除 __pycache__ 再重新运行本脚本")
        else:
            logger.warning("⚠️  未在 datamodule 中找到 'from xas_local_dataset' import 行，请手动检查")
    else:
        logger.warning(f"⚠️  找不到 datamodule 文件：{dm_path}")

    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch_geometric.data import Batch

    # ── 加载 Dataset（直接用 L6 版本）────────────────────────────────────
    FEFF_CSV      = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
    DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
    INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
    TRAIN_IDS     = os.path.join(STEP1_DIR, "train_ids.txt")
    SCALER_PATH   = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")

    # 直接 import dataset（不走 datamodule，绕过坑 1 风险）
    _ds_path = os.path.join(STEP3_DIR, "xas_local_dataset_L6.py")
    if not os.path.exists(_ds_path):
        logger.error(f"❌ 找不到 {_ds_path}，请先将 xas_local_dataset_L6.py 复制到该目录")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location("xas_local_dataset_L6", _ds_path)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    XASLocalStructureDataset = _mod.XASLocalStructureDataset

    logger.info(f"直接加载 Dataset（L={L}）...")
    ds_train = XASLocalStructureDataset(
        data_root        = DATA_ROOT,
        inventory_csv    = INVENTORY_CSV,
        ids_file         = TRAIN_IDS,
        feff_feat_csv    = FEFF_CSV,
        feff_scaler_path = SCALER_PATH,
        L                = L,
    )
    logger.info(f"Dataset 声明长度：{len(ds_train)}")

    # ── 检查 2：有效样本数（统计 None 率）────────────────────────────────
    logger.info(f"\n[检查 2] 有效样本率（最多检查前 200 个）...")
    total_checked = min(200, len(ds_train))
    none_count    = 0
    valid_samples = []
    for i in range(total_checked):
        s = ds_train[i]
        if s is None:
            none_count += 1
        else:
            valid_samples.append(s)
    loss_rate = none_count / total_checked * 100
    logger.info(f"  前 {total_checked} 个样本：有效={total_checked - none_count}，"
                f"无效(None)={none_count}，丢失率={loss_rate:.1f}%")
    if loss_rate > 5.0:
        logger.error(f"❌ 检查 2 失败：丢失率={loss_rate:.1f}% > 5%，请汇报 Main Agent 2")
    else:
        logger.info(f"✅ 检查 2 通过：丢失率={loss_rate:.1f}% ≤ 5%")

    # ── 检查 1：frac_coords 范围 ──────────────────────────────────────────
    logger.info(f"\n[检查 1] frac_coords 范围（{N_CHECK_SAMPLES} 个有效样本）...")
    check_samples = [s for s in valid_samples if s is not None][:N_CHECK_SAMPLES]
    import numpy as np
    all_min = min(s.frac_coords.min().item() for s in check_samples)
    all_max = max(s.frac_coords.max().item() for s in check_samples)
    logger.info(f"  frac_coords: min={all_min:.4f}, max={all_max:.4f}")
    if all_min >= -0.5 and all_max <= 0.5:
        logger.info("✅ 检查 1 通过：frac_coords ∈ [-0.5, 0.5]")
    else:
        logger.error(f"❌ 检查 1 失败：存在超出 [-0.5, 0.5] 的坐标！")
        logger.error(f"   min={all_min:.4f}  max={all_max:.4f}")

    # ── 检查 4：晶格矩阵确认 ─────────────────────────────────────────────
    logger.info(f"\n[检查 4] 晶格矩阵确认...")
    s0 = check_samples[0]
    logger.info(f"  lengths: {s0.lengths}")
    logger.info(f"  angles:  {s0.angles}")
    lengths_vals = s0.lengths.view(-1).tolist()
    if all(abs(v - 6.0) < 0.01 for v in lengths_vals):
        logger.info("✅ 检查 4 通过：lengths = [6.0, 6.0, 6.0]（等价 diag(6,6,6)）")
    else:
        logger.error(f"❌ 检查 4 失败：lengths={lengths_vals}，预期 [6.0, 6.0, 6.0]")
        logger.error("   请检查 xas_local_dataset_L6.py 中 lengths tensor 是否用 self.L")

    # ── 检查 3：forward() loss ────────────────────────────────────────────
    logger.info(f"\n[检查 3] forward() loss...")
    try:
        CONF_DIR = os.path.join(STEP3_DIR, "conf_xas")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(CONF_DIR, "model"),
                                   job_name="precheck", version_base=None):
            _raw = compose(config_name="diffusion_xas")
        model_cfg = OmegaConf.create({
            "model": OmegaConf.to_container(_raw, resolve=False)}).model

        optim_cfg = OmegaConf.create({
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
            "use_lr_scheduler": False, "lr_scheduler": None})

        model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
        model.lattice_scaler = model.scaler = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()

        # 拼一个小 batch
        batch_samples = [s for s in check_samples if s is not None][:4]
        batch = Batch.from_data_list(batch_samples).to(device)

        with torch.no_grad():
            output = model(batch)

        loss       = output['loss'].item()
        loss_coord = output.get('loss_coord', output.get('loss_x', torch.tensor(float('nan')))).item()
        loss_type  = output.get('loss_type',  torch.tensor(float('nan'))).item()

        logger.info(f"  loss={loss:.4f}  loss_coord={loss_coord:.4f}  loss_type={loss_type:.4f}")

        if np.isnan(loss) or np.isinf(loss):
            logger.error("❌ 检查 3 失败：loss 含 NaN/Inf！")
        elif loss > 10:
            logger.warning(f"⚠️  检查 3：loss={loss:.4f} > 10，数值偏大，请确认（若无 NaN 可继续）")
        else:
            logger.info("✅ 检查 3 通过：loss 无 NaN/Inf，数值合理")

    except Exception as e:
        logger.error(f"❌ 检查 3 异常：{e}")
        import traceback; traceback.print_exc()

    logger.info("")
    logger.info("=" * 60)
    logger.info("开训前检查完成。全部 ✅ 后执行 step4d_2_train.py")
    logger.info("=" * 60)
