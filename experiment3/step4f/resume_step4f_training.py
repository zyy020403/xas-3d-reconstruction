"""
resume_step4f_training.py
=========================
从 step4f/checkpoints/last.ckpt 续训。
与 run_step4f_training.py 的区别：
  - 跳过手动权重加载（step4e partial load），直接用 last.ckpt
  - trainer.fit() 传入 ckpt_path，Lightning 自动恢复 epoch/optimizer/scheduler 状态
  - max_epochs=180 维持不变（Lightning 会从 last.ckpt 记录的 epoch 继续计数）

使用方法（在 DiffCSP-main 根目录执行）：
  python experiment3/step4f/resume_step4f_training.py
"""

import os, sys, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

DIFFCSP_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(DIFFCSP_ROOT, "experiment2")
EXP3_ROOT    = os.path.join(DIFFCSP_ROOT, "experiment3")
STEP1_DIR    = os.path.join(EXP2_ROOT, "step1")
STEP4F_DIR   = os.path.join(EXP3_ROOT, "step4f")
LAST_CKPT    = os.path.join(STEP4F_DIR, "checkpoints", "last.ckpt")

sys.path.insert(0, DIFFCSP_ROOT)
sys.path.insert(0, os.path.join(EXP2_ROOT, "step2"))
sys.path.insert(0, os.path.join(EXP2_ROOT, "step3"))
sys.path.insert(0, STEP4F_DIR)
os.environ.setdefault('PROJECT_ROOT', DIFFCSP_ROOT)

from diffusion_w_type_xas_exp3_step4f import CSPDiffusion
from xas_local_dataset_L6 import XASLocalStructureDataset

assert os.path.exists(LAST_CKPT), f"找不到 last.ckpt：{LAST_CKPT}"
print(f"从 last.ckpt 续训：{LAST_CKPT}")

# ── 从 last.ckpt 读 hparams 实例化模型（不做手动权重加载，让 ckpt_path 接管）──
ckpt      = torch.load(LAST_CKPT, map_location='cpu')
hparams   = ckpt['hyper_parameters']
hparams['vocab_path'] = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
model     = CSPDiffusion(**hparams)
print(f"模型实例化完成，参数量：{sum(p.numel() for p in model.parameters()):,}")

# ── DataLoader ────────────────────────────────────────────────────────────────
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

train_loader = DataLoader(
    XASLocalStructureDataset(
        data_root        = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site"),
        inventory_csv    = os.path.join(STEP1_DIR, "data_inventory.csv"),
        ids_file         = os.path.join(STEP1_DIR, "train_ids.txt"),
        feff_feat_csv    = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv"),
        feff_scaler_path = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"),
    ), batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)

val_loader = DataLoader(
    XASLocalStructureDataset(
        data_root        = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site"),
        inventory_csv    = os.path.join(STEP1_DIR, "data_inventory.csv"),
        ids_file         = os.path.join(STEP1_DIR, "val_ids.txt"),
        feff_feat_csv    = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv"),
        feff_scaler_path = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"),
    ), batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = pl.Trainer(
    max_epochs        = 180,
    accelerator       = "gpu",
    devices           = 1,
    precision         = "bf16",
    gradient_clip_val = 1.0,
    callbacks         = [
        ModelCheckpoint(
            dirpath    = os.path.join(STEP4F_DIR, "checkpoints"),
            filename   = "best_{epoch:03d}-{val_type_acc:.4f}-{val_coord_loss:.4f}",
            monitor    = "val_type_acc",
            mode       = "max",
            save_top_k = 3,
            save_last  = True,
        ),
        EarlyStopping(
            monitor   = "val_type_acc",
            mode      = "max",
            patience  = 30,
            min_delta = 1e-4,
            verbose   = True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ],
    logger = CSVLogger(save_dir=STEP4F_DIR, name="logs"),
    log_every_n_steps = 10,
)

# ckpt_path 让 Lightning 完整恢复：epoch 计数、optimizer、scheduler 全部接上
trainer.fit(model, train_loader, val_loader, ckpt_path=LAST_CKPT)
print("训练完成")
