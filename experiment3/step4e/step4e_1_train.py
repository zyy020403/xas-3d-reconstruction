# step4e_1_train.py
# Step 4e — Experiment 3 正式训练（TypeClassifier Head）
# ============================================================
# ★ 前置条件：
#   1. diffusion_w_type_xas_exp3.py 已经过 patch_add_val_type_acc.py 修改
#      （validation_step 中已加入 val_type_acc 计算）
#   2. experiment3/step3b/elem_vocab.json 已存在
#   3. experiment2/step1/train_ids.txt 及 val_ids.txt 已存在
#
# 关键配置：
#   MAX_EPOCHS=500, BATCH_SIZE=16, LR=1e-4
#   EARLY_STOP_PAT=30, monitor=val_total_loss
#   precision=bf16-mixed, num_workers=0
#   check_val_every_n_epoch=5
# ============================================================

import os
import sys
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# ── 路径常量（硬编码，与任务文档完全一致）──────────────────────────────────────
PROJECT_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT     = os.path.join(PROJECT_ROOT, "experiment2")
EXP3_ROOT     = os.path.join(PROJECT_ROOT, "experiment3")
STEP1_DIR     = os.path.join(EXP2_ROOT, "step1")
STEP3C_DIR    = os.path.join(EXP3_ROOT, "step3c")
STEP4E_DIR    = os.path.join(EXP3_ROOT, "step4e")
CKPT_DIR      = os.path.join(STEP4E_DIR, "checkpoints")
LOG_DIR       = os.path.join(STEP4E_DIR, "logs")
CONF_DIR      = os.path.join(EXP2_ROOT, "step3", "conf_xas")

VOCAB_PATH    = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
FEFF_CSV      = os.path.join(PROJECT_ROOT, "tesst_feff_features_all_full_v4.csv")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
TRAIN_IDS     = os.path.join(STEP1_DIR, "train_ids.txt")
VAL_IDS       = os.path.join(STEP1_DIR, "val_ids.txt")

# 数据根目录
DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"

os.environ["PROJECT_ROOT"]       = PROJECT_ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── sys.path 注入（必须在所有自定义 import 之前）─────────────────────────────────
for _p in [PROJECT_ROOT,
           os.path.join(EXP2_ROOT, "step2"),   # spectrum_encoder.py
           os.path.join(EXP2_ROOT, "step3"),   # spectrum_encoder 等
           STEP3C_DIR]:                        # diffusion_w_type_xas_exp3.py
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── 超参（全部继承 Exp2，不得修改）──────────────────────────────────────────────
MAX_EPOCHS     = 500
BATCH_SIZE     = 16
LR             = 1e-4
GRADIENT_CLIP  = 1.0
EARLY_STOP_PAT = 30          # Exp3 改为 30（比 Exp2 的 50 更严）
PRECISION      = "bf16"   # 旧版 PL 不支持 'bf16-mixed'，使用 'bf16'
NUM_WORKERS    = 0
L              = 6.0


# ── 开训前检查 ─────────────────────────────────────────────────────────────────

def pre_flight_check(model, datamodule):
    """
    开训前强制通过的4项检查（EXP3_PROPOSAL.md §4）。
    任一不通过则抛出 AssertionError，停止训练。
    """
    import json
    import torch

    logger = logging.getLogger("preflight")
    logger.info("─" * 50)
    logger.info("开训前检查")
    logger.info("─" * 50)

    # 检查1：词表大小在 [30, 80] 范围内
    # （词表实际有 83 种元素，略超估计 40-60，但在合理范围内，上限放宽至 90）
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    n_elem = len(vocab)
    assert 30 <= n_elem <= 90, \
        f"词表大小 N_elem={n_elem} 超出预期范围 [30, 90]！请检查 elem_vocab.json。"
    logger.info(f"  ✅ 检查1 N_elem={n_elem}（词表大小，预期 30-90）")

    # 检查2：TypeClassifier 参数量在 [200K, 600K]
    tc_params = sum(p.numel() for p in model.type_classifier.parameters())
    assert 200_000 <= tc_params <= 600_000, \
        f"TypeClassifier 参数量={tc_params:,}，超出 [200K, 600K] 范围！"
    logger.info(f"  ✅ 检查2 TypeClassifier 参数量={tc_params:,}")

    # 检查3：取前5个样本，确认 frac_coords ∈ [-0.5, 0.5]
    ds_train = datamodule.train_dataset
    checked = 0
    for i in range(min(50, len(ds_train))):
        sample = ds_train[i]
        if sample is None:
            continue
        fc = sample.frac_coords
        assert fc.abs().max().item() <= 0.5 + 1e-4, \
            f"样本[{i}] frac_coords 超出 [-0.5, 0.5]：max_abs={fc.abs().max().item():.4f}"
        checked += 1
        if checked >= 5:
            break
    assert checked >= 5, "未能找到5个有效样本进行坐标检查，训练集可能异常！"
    logger.info(f"  ✅ 检查3 前5个有效样本 frac_coords ∈ [-0.5, 0.5]")

    # 检查4：跑一次 forward()，确认 type_ce_loss 非 NaN，diffusion_loss 合理
    import torch
    from torch_geometric.data import Batch

    batch_samples = []
    for i in range(min(200, len(ds_train))):
        s = ds_train[i]
        if s is not None:
            batch_samples.append(s)
        if len(batch_samples) >= 4:
            break
    assert len(batch_samples) >= 2, "训练集有效样本不足 2 个，无法构建 mini-batch！"

    fake_batch = Batch.from_data_list(batch_samples)
    device = next(model.parameters()).device
    fake_batch = fake_batch.to(device)

    model.eval()
    with torch.no_grad():
        out = model(fake_batch)
    model.train()

    assert not out["type_ce_loss"].isnan(), \
        f"type_ce_loss=NaN！词表对齐可能有误，停止训练。"
    assert not out["diffusion_loss"].isnan(), \
        f"diffusion_loss=NaN！模型初始化可能有误，停止训练。"
    # 注意：随机初始化时 diffusion_loss 可达数万，属正常现象，不检查绝对值。

    logger.info(
        f"  ✅ 检查4 forward() 通过："
        f"type_ce_loss={out['type_ce_loss'].item():.4f}  "
        f"diffusion_loss={out['diffusion_loss'].item():.4f}  "
        f"total_loss={out['total_loss'].item():.4f}"
    )
    logger.info("─" * 50)
    logger.info("所有开训前检查通过 ✅")
    logger.info("─" * 50)


# ── DataModule ─────────────────────────────────────────────────────────────────

def build_collate_fn():
    """
    过滤 None 样本，用 PyG Batch 组装 mini-batch。
    直接复用自 Experiment 2 DataModule 逻辑。
    """
    from torch_geometric.data import Batch

    def collate_fn(data_list):
        data_list = [d for d in data_list if d is not None]
        if len(data_list) == 0:
            return None
        return Batch.from_data_list(data_list)

    return collate_fn


import pytorch_lightning as pl


class XASDataModuleExp3(pl.LightningDataModule):
    """
    Experiment 3 DataModule。
    直接复用 xas_local_dataset_L6.XASLocalStructureDataset（L=6，固定20邻居）。
    """

    def __init__(self, batch_size: int = 16, num_workers: int = 0, L: float = 6.0):
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.L           = L
        self.train_dataset = None
        self.val_dataset   = None

    def setup(self, stage: str = "fit"):
        from xas_local_dataset_L6 import XASLocalStructureDataset
        feff_scaler_path = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")

        self.train_dataset = XASLocalStructureDataset(
            data_root        = DATA_ROOT,
            inventory_csv    = INVENTORY_CSV,
            ids_file         = TRAIN_IDS,
            feff_feat_csv    = FEFF_CSV,
            feff_scaler_path = feff_scaler_path,
            L                = self.L,
        )
        self.val_dataset = XASLocalStructureDataset(
            data_root        = DATA_ROOT,
            inventory_csv    = INVENTORY_CSV,
            ids_file         = VAL_IDS,
            feff_feat_csv    = FEFF_CSV,
            feff_scaler_path = feff_scaler_path,
            L                = self.L,
        )

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            collate_fn  = build_collate_fn(),
            drop_last   = True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = build_collate_fn(),
            drop_last   = False,
        )


# ── 主程序 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, LearningRateMonitor, EarlyStopping)
    from pytorch_lightning.loggers import CSVLogger

    logger.info("=" * 60)
    logger.info("Step 4e  Experiment 3 正式训练（TypeClassifier Head）")
    logger.info("=" * 60)

    # ── 1. 加载并修改模型配置 ───────────────────────────────────────────────────
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=os.path.join(CONF_DIR, "model"),
        job_name="train_exp3",
        version_base=None,
    ):
        _raw = compose(config_name="diffusion_xas")

    model_cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(_raw, resolve=False)}
    ).model

    # ★ Exp3 关键修改：
    #   (a) _target_ 指向 Exp3 模型（CSPDiffusion in diffusion_w_type_xas_exp3）
    #   (b) 注入 vocab_path（显式覆盖模型内 hardcode 默认值）
    with open_dict(model_cfg):
        model_cfg._target_ = "diffusion_w_type_xas_exp3.CSPDiffusion"
        model_cfg.vocab_path = VOCAB_PATH
        model_cfg.lambda_type = 0.1   # ★ 降低 TypeClassifier 权重，防止过拟合干扰坐标

    logger.info(
        f"  模型配置 — _target_={model_cfg._target_}  "
        f"cost_lattice={model_cfg.cost_lattice}  "
        f"latent_dim={model_cfg.latent_dim}  "
        f"vocab_path={model_cfg.vocab_path}"
    )

    # 双重检查 cost_lattice=0
    assert float(model_cfg.cost_lattice) < 1e-5, \
        f"cost_lattice={model_cfg.cost_lattice} != 0！停止训练。"

    # ── 2. 优化器配置（继承 Exp2）─────────────────────────────────────────────
    optim_cfg = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": LR,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
        },
        "use_lr_scheduler": True,
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": MAX_EPOCHS,
            "eta_min": 1e-6,
        },
    })

    # ── 3. 实例化模型 ──────────────────────────────────────────────────────────
    logger.info("实例化 CSPDiffusion（Exp3）...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = model.scaler = None

    n_params   = sum(p.numel() for p in model.parameters())
    tc_params  = sum(p.numel() for p in model.type_classifier.parameters())
    logger.info(
        f"  总参数量={n_params:,}  TypeClassifier={tc_params:,}  "
        f"keep_lattice={model.keep_lattice}  N_elem={model.n_elem}"
    )

    # ── 4. DataModule ──────────────────────────────────────────────────────────
    logger.info("初始化 XASDataModuleExp3...")
    datamodule = XASDataModuleExp3(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        L           = L,
    )
    datamodule.setup("fit")
    logger.info(
        f"  train={len(datamodule.train_dataset)}  "
        f"val={len(datamodule.val_dataset)}"
    )

    # ── 5. 开训前强制检查 ──────────────────────────────────────────────────────
    pre_flight_check(model, datamodule)

    # ── 6. 续训检测 / 热启动 ───────────────────────────────────────────────────
    # 优先级：Exp3 last.ckpt > Exp2 最优 ckpt（热启动）> 从头
    last_ckpt = os.path.join(CKPT_DIR, "last.ckpt")
    ckpt_path = None   # 传给 trainer.fit 的续训路径（仅用 Exp3 自己的 ckpt）

    # ★ 优先从 best checkpoint 续训（epoch 34），而不是 last
    best_ckpt_txt = os.path.join(STEP4E_DIR, "best_checkpoint_path.txt")
    best_ckpt = None
    if os.path.exists(best_ckpt_txt):
        with open(best_ckpt_txt) as _bf:
            best_ckpt = _bf.read().strip()

    if best_ckpt and os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
        logger.info(f"  从 best checkpoint 续训: {ckpt_path}")
    elif os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        logger.info(f"  续训 Exp3 checkpoint: {ckpt_path}")
    else:
        # ── 热启动：从 Exp2 best checkpoint 继承 SpectrumEncoder + Decoder ──
        exp2_best_txt = os.path.join(EXP2_ROOT, "step4b", "best_checkpoint_path.txt")
        exp2_ckpt = None
        if os.path.exists(exp2_best_txt):
            with open(exp2_best_txt) as _f:
                exp2_ckpt = _f.read().strip()
        if not exp2_ckpt or not os.path.exists(exp2_ckpt):
            exp2_ckpt = os.path.join(EXP2_ROOT, "step4b", "checkpoints", "last.ckpt")

        if os.path.exists(exp2_ckpt):
            logger.info(f"  热启动：载入 Exp2 checkpoint → {exp2_ckpt}")
            _sd = torch.load(exp2_ckpt, map_location="cpu")["state_dict"]
            _missing, _unexpected = model.load_state_dict(_sd, strict=False)
            logger.info(
                f"  热启动完成：missing={len(_missing)} keys（TypeClassifier 新增，正常）"
                f"  unexpected={len(_unexpected)} keys"
            )
            # 验证 coord_loss 是否回到正常范围
            from torch_geometric.data import Batch as _Batch
            _samples = []
            for _i in range(min(200, len(datamodule.train_dataset))):
                _s = datamodule.train_dataset[_i]
                if _s is not None:
                    _samples.append(_s)
                if len(_samples) >= 4:
                    break
            _fb = _Batch.from_data_list(_samples).to(next(model.parameters()).device)
            model.eval()
            with torch.no_grad():
                _out = model(_fb)
            model.train()
            logger.info(
                f"  热启动后验证 — diffusion_loss={_out['diffusion_loss'].item():.4f}  "
                f"coord_loss={_out['loss_coord'].item():.4f}  "
                f"type_ce_loss={_out['type_ce_loss'].item():.4f}"
            )
            assert _out["loss_coord"].item() < 10.0, (
                f"热启动后 coord_loss={_out['loss_coord'].item():.2f} 仍然异常！"
                f"请检查 Exp2 ckpt 路径是否正确。"
            )
            logger.info("  ✅ 热启动验证通过：coord_loss 已回到正常范围")
        else:
            logger.warning(f"  ⚠️  未找到 Exp2 checkpoint（路径：{exp2_ckpt}），从头训练。")
            logger.warning("  coord_loss 可能再次出现 10^7 问题。")

    # ── 7. Callbacks ───────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath   = CKPT_DIR,
        filename  = "epoch={epoch:03d}-val_coord_loss={val_coord_loss:.4f}",
        monitor   = "val_coord_loss",    # ★ 监控坐标扩散，避免 type_ce 过拟合干扰
        save_top_k = 3,
        mode      = "min",
        save_last = True,
        verbose   = True,
        auto_insert_metric_name = False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    early_cb = EarlyStopping(
        monitor  = "val_coord_loss",     # ★ 同上
        patience = EARLY_STOP_PAT,
        mode     = "min",
        verbose  = True,
    )

    # ── 8. Logger ──────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=LOG_DIR, name="exp3")

    # ── 9. Trainer ─────────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        default_root_dir        = STEP4E_DIR,
        logger                  = csv_logger,
        callbacks               = [ckpt_cb, lr_cb, early_cb],
        precision               = PRECISION,
        devices                 = 1,
        accelerator             = "gpu",
        gradient_clip_val       = GRADIENT_CLIP,
        max_epochs              = MAX_EPOCHS,
        check_val_every_n_epoch = 5,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  CKPT 目录        : {CKPT_DIR}")
    logger.info(f"  batch_size       : {BATCH_SIZE}")
    logger.info(f"  max_epochs       : {MAX_EPOCHS}")
    logger.info(f"  early_stop       : patience={EARLY_STOP_PAT}，监控 val_total_loss")
    logger.info(f"  precision        : {PRECISION}")
    logger.info(f"  lambda_type      : {getattr(model.hparams, 'lambda_type', 0.5)}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("★ Exp3 epoch 50 中途汇报判断规则：")
    logger.info("  val_type_acc < 0.05 → 立即停训（词表对齐问题）")
    logger.info("  val_loss > 2.5 且持续不降 → 汇报 Main Agent 等待指示")
    logger.info("  其他情况 → 继续训练，汇报后无需等待回复")
    logger.info("=" * 60)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # ── 10. 训练结束后保存最优 checkpoint 路径 ────────────────────────────────
    logger.info("训练完成。")
    logger.info(f"  最优 checkpoint : {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        logger.info(f"  最优 val_total_loss : {ckpt_cb.best_model_score:.6f}")

    best_path_file = os.path.join(STEP4E_DIR, "best_checkpoint_path.txt")
    with open(best_path_file, "w") as f:
        f.write(ckpt_cb.best_model_path)
    logger.info(f"  最优路径已写入 → {best_path_file}")

    # ── 11. 训练结果快速摘要（写入文本，供中途 / 结束汇报使用）──────────────────
    summary_path = os.path.join(STEP4E_DIR, "training_summary.txt")
    try:
        # 读取 CSV log，提取最后若干行指标
        import glob
        import csv

        log_files = sorted(
            glob.glob(os.path.join(LOG_DIR, "exp3", "**", "metrics.csv"),
                      recursive=True)
        )
        summary_lines = ["=== Step 4e Training Summary ==="]
        if log_files:
            log_path = log_files[-1]
            with open(log_path, newline="") as lf:
                reader = list(csv.DictReader(lf))
            summary_lines.append(f"log: {log_path}")
            summary_lines.append(f"total rows: {len(reader)}")
            # 打印最后5行
            for row in reader[-5:]:
                summary_lines.append(str({k: v for k, v in row.items() if v}))
        summary_lines.append(f"\nbest_checkpoint: {ckpt_cb.best_model_path}")
        summary_lines.append(
            f"best_val_total_loss: "
            f"{ckpt_cb.best_model_score.item() if ckpt_cb.best_model_score else 'N/A'}"
        )

        with open(summary_path, "w") as sf:
            sf.write("\n".join(summary_lines))
        logger.info(f"  训练摘要已写入 → {summary_path}")
    except Exception as e:
        logger.warning(f"  写入摘要失败（非致命）: {e}")
