"""
run_step4f_training.py
======================
Experiment 3 — Step 4f 训练启动脚本

功能：
  1. 加载 Step4e 最优 checkpoint（epoch=419）
  2. 处理 TypeClassifier 第一层权重形状不兼容问题：
     旧：Linear(256, 256)  →  新：Linear(329, 256)
     策略：跳过形状不匹配的层，其余全部继承
  3. 继续训练至 max_epochs=600（再给 ~180 epochs）
  4. early_stop：监控 val_type_acc，patience=30
  5. 输出目录：experiment3/step4f/

文件存放规范（旧文件不动）：
  step3c/diffusion_w_type_xas_exp3.py        ← Step4e 原版，保留不动
  step4f/diffusion_w_type_xas_exp3_step4f.py ← Step4f 修复版（本脚本 import 此文件）
  step4f/run_step4f_training.py               ← 本脚本

使用方法（在 DiffCSP-main 根目录执行）：
  python experiment3/step4f/run_step4f_training.py
"""

import os
import sys
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
DIFFCSP_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT     = os.path.join(DIFFCSP_ROOT, "experiment2")
EXP3_ROOT     = os.path.join(DIFFCSP_ROOT, "experiment3")
STEP1_DIR     = os.path.join(EXP2_ROOT, "step1")
STEP3B_DIR    = os.path.join(EXP3_ROOT, "step3b")
STEP3C_DIR    = os.path.join(EXP3_ROOT, "step3c")   # 旧文件保留不动
STEP4F_SRC    = os.path.join(EXP3_ROOT, "step4f")   # 新脚本所在目录
STEP4E_CKPT   = os.path.join(EXP3_ROOT, "step4e", "checkpoints",
                              "epoch=419-val_coord_loss=0.7249.ckpt")
STEP4F_OUT    = os.path.join(EXP3_ROOT, "step4f")
DATA_ROOT     = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site")
FEFF_CSV      = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv")
VOCAB_PATH    = os.path.join(STEP3B_DIR, "elem_vocab.json")
FEAT_SCALER   = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
TRAIN_IDS     = os.path.join(STEP1_DIR, "train_ids.txt")
VAL_IDS       = os.path.join(STEP1_DIR, "val_ids.txt")

# ─────────────────────────────────────────────────────────────────────────────
# sys.path 设置
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, DIFFCSP_ROOT)
sys.path.insert(0, os.path.join(EXP2_ROOT, "step2"))
sys.path.insert(0, os.path.join(EXP2_ROOT, "step3"))   # xas_local_dataset_L6
sys.path.insert(0, STEP4F_SRC)   # ← 指向 step4f，旧 step3c 文件不动
os.environ.setdefault('PROJECT_ROOT', DIFFCSP_ROOT)

from diffusion_w_type_xas_exp3_step4f import CSPDiffusion   # Step4f 修复版
from xas_local_dataset_L6 import XASLocalStructureDataset

os.makedirs(STEP4F_OUT, exist_ok=True)
os.makedirs(os.path.join(STEP4F_OUT, "checkpoints"), exist_ok=True)
LOG_FILE = os.path.join(STEP4F_OUT, "train_log.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 工具：带 tee 的 print，同时写入 train_log.txt
# ─────────────────────────────────────────────────────────────────────────────
_log_f = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)

def log(msg):
    print(msg)
    _log_f.write(msg + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1：从旧 checkpoint 加载超参（hparams），然后用新架构重建模型
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("Step 4f — 修复 TypeClassifier 输入并继续训练")
log("=" * 60)
log(f"旧 checkpoint：{STEP4E_CKPT}")
log(f"输出目录：{STEP4F_OUT}")

assert os.path.exists(STEP4E_CKPT), f"❌ checkpoint 不存在：{STEP4E_CKPT}"
assert os.path.exists(VOCAB_PATH),  f"❌ elem_vocab.json 不存在：{VOCAB_PATH}"

# 读取旧 checkpoint（只提取 hparams 和 state_dict）
log("\n[1/5] 加载旧 checkpoint state_dict ...")
old_ckpt = torch.load(STEP4E_CKPT, map_location='cpu')
old_hparams = old_ckpt['hyper_parameters']
old_state   = old_ckpt['state_dict']

# 确认旧 TypeClassifier 第一层形状
tc_w_key = 'type_classifier.mlp.0.weight'
if tc_w_key in old_state:
    old_shape = old_state[tc_w_key].shape
    log(f"  旧 TypeClassifier 第一层 weight shape：{old_shape}")
    assert old_shape == (256, 256), f"预期 (256,256)，实际 {old_shape}"
else:
    log(f"  ⚠️  未找到 {tc_w_key}，跳过形状检查")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2：用旧 hparams 实例化新模型（TypeClassifier latent_dim=329）
# ─────────────────────────────────────────────────────────────────────────────
log("\n[2/5] 实例化 Step4f 模型（TypeClassifier latent_dim=329）...")

# 注入 vocab_path 到 hparams（旧模型可能用硬编码路径，这里统一）
old_hparams['vocab_path'] = VOCAB_PATH

# 用 Hydra OmegaConf-style dict 实例化（直接传 **hparams 给 BaseModule）
# CSPDiffusion 的 __init__ 接受 **kwargs 并通过 save_hyperparameters() 保存
model = CSPDiffusion(**old_hparams)
log(f"  新模型 TypeClassifier 参数量：{sum(p.numel() for p in model.type_classifier.parameters()):,}")

# 验证新 TypeClassifier 第一层形状
new_tc_w = dict(model.named_parameters())[tc_w_key]
log(f"  新 TypeClassifier 第一层 weight shape：{new_tc_w.shape}")
assert new_tc_w.shape == (256, 329), f"预期 (256,329)，实际 {new_tc_w.shape}"
log("  ✅ 新模型架构验证通过")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3：部分加载权重（跳过形状不匹配的 TypeClassifier 层）
# ─────────────────────────────────────────────────────────────────────────────
log("\n[3/5] 部分加载旧 checkpoint 权重（跳过形状不兼容层）...")

# 找出所有不兼容的 key
new_state    = model.state_dict()
skip_keys    = []
load_state   = {}

for key, old_val in old_state.items():
    if key not in new_state:
        skip_keys.append(f"{key}  [NOT IN NEW MODEL]")
        continue
    if old_val.shape != new_state[key].shape:
        skip_keys.append(f"{key}  old={old_val.shape} → new={new_state[key].shape}")
        continue
    load_state[key] = old_val

if skip_keys:
    log("  跳过以下 key（形状不匹配或不存在）：")
    for k in skip_keys:
        log(f"    ⚠️  {k}")

missing, unexpected = model.load_state_dict(load_state, strict=False)
log(f"  加载完成：{len(load_state)} keys 成功，"
    f"{len(missing)} missing，{len(unexpected)} unexpected")

# TypeClassifier 的跳过层将保持随机初始化（从头学习 329 维输入）
log("  TypeClassifier 形状不兼容层保持随机初始化（从头学习新的 329 维输入）")
log("  所有其他层（SpectrumEncoder、CSPNet、坐标扩散）权重完整继承")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4：数据加载
# ─────────────────────────────────────────────────────────────────────────────
log("\n[4/5] 准备 DataLoader ...")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

train_dataset = XASLocalStructureDataset(
    data_root        = DATA_ROOT,
    inventory_csv    = INVENTORY_CSV,
    ids_file         = TRAIN_IDS,
    feff_feat_csv    = FEFF_CSV,
    feff_scaler_path = FEAT_SCALER,
)
val_dataset = XASLocalStructureDataset(
    data_root        = DATA_ROOT,
    inventory_csv    = INVENTORY_CSV,
    ids_file         = VAL_IDS,
    feff_feat_csv    = FEFF_CSV,
    feff_scaler_path = FEAT_SCALER,
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=0, collate_fn=collate_fn)

log(f"  Train: {len(train_dataset)} 样本 | Val: {len(val_dataset)} 样本")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5：训练
# ─────────────────────────────────────────────────────────────────────────────
log("\n[5/5] 启动训练（max_epochs=600，从 epoch 420 继续）...")

checkpoint_cb = ModelCheckpoint(
    dirpath    = os.path.join(STEP4F_OUT, "checkpoints"),
    filename   = "best_{epoch:03d}-{val_type_acc:.4f}-{val_coord_loss:.4f}",
    monitor    = "val_type_acc",
    mode       = "max",
    save_top_k = 3,
    save_last  = True,
)

early_stop_cb = EarlyStopping(
    monitor   = "val_type_acc",
    mode      = "max",
    patience  = 30,
    min_delta = 1e-4,
    verbose   = True,
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

csv_logger = CSVLogger(
    save_dir = STEP4F_OUT,
    name     = "logs",
)

trainer = pl.Trainer(
    max_epochs           = 180,
    accelerator          = "gpu",
    devices              = 1,
    precision            = "bf16",
    gradient_clip_val    = 1.0,
    callbacks            = [checkpoint_cb, early_stop_cb, lr_monitor],
    logger               = csv_logger,
    log_every_n_steps    = 10,
    # 注意：不传 resume_from_checkpoint，因为 TypeClassifier 层形状已变
    # 权重已在上方手动部分加载，此处全新开始 Lightning 的 epoch 计数器
    # 实际上是从 epoch 0 开始重新计数，但模型参数（除 TC 首层外）已继承
)

# 早期监控：在 epoch 20 打印一次 val_type_acc 警告
class EarlyWarnCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        if epoch == 20:
            acc = metrics.get('val_type_acc', None)
            log(f"\n  [早期监控 epoch={epoch}] val_type_acc = {acc}")
            if acc is not None and acc < 0.25:
                log("  ⚠️  val_type_acc < 0.25，修复效果可能不足，请关注后续趋势")
            elif acc is not None and acc > 0.30:
                log("  ✅ val_type_acc > 0.30，修复有效！继续训练")

trainer.callbacks.append(EarlyWarnCallback())

log("\n  训练开始 >>>")
try:
    trainer.fit(model, train_loader, val_loader)
except Exception as e:
    log(f"\n  ❌ 训练异常终止：{e}")
    raise
finally:
    _log_f.close()

log("\n" + "=" * 60)
log("Step 4f 训练完成")
log(f"最优 checkpoint：{checkpoint_cb.best_model_path}")
log(f"最优 val_type_acc：{checkpoint_cb.best_model_score}")
log("=" * 60)
