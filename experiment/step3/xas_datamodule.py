# =============================================================================
# 脚本编号: step3.1 (datamodule)
# 脚本名称: xas_datamodule.py
# 输入:
#   - experiment/step1/data_inventory.csv
#   - experiment/step1/train_ids.txt
#   - experiment/step1/val_ids.txt
#   - experiment/step1/test_ids.txt
#   - experiment/step1/holdout_1000_ids.txt   ← 严禁泄露
# 输出:
#   - XASDataModule: 提供 train / val / test DataLoader
# 说明:
#   holdout mp_id 在 setup 中过滤，任何 split 均不包含。
#   DataLoader 使用 xas_collate_fn 处理变长位点。
# =============================================================================

import os
import logging
from typing import Optional

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from xas_dataset import XASCrystalDataset, xas_collate_fn

logger = logging.getLogger(__name__)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP1_DIR      = os.path.join(EXPERIMENT_DIR, "step1")


# ─────────────────────────────────────────────────────────────────────────────
#  辅助：读取 id 文件（每行一个 mp_id）
# ─────────────────────────────────────────────────────────────────────────────

def _read_ids(path: str):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  XASDataModule
# ─────────────────────────────────────────────────────────────────────────────

class XASDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    batch_size : dict  {"train": int, "val": int, "test": int}
    num_workers : dict {"train": int, "val": int, "test": int}
    step1_dir : str    Step 1 输出目录（含 id 文件和 data_inventory.csv）
    """

    def __init__(
        self,
        batch_size:  dict = None,
        num_workers: dict = None,
        step1_dir:   str  = STEP1_DIR,
    ):
        super().__init__()
        self.batch_size  = batch_size  or {"train": 16, "val": 8, "test": 8}
        self.num_workers = num_workers or {"train": 4,  "val": 2, "test": 2}
        self.step1_dir   = step1_dir

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

        # scaler 接口：与 run.py 对齐
        self.scaler         = None
        self.lattice_scaler = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        # ── 幂等保护：PL 会在 trainer.fit 内部再次调用 setup，直接跳过避免重复预加载 ──
        if stage in (None, "fit") and self.train_dataset is not None:
            logger.info("setup('fit') 已执行过，跳过重复预加载。")
            return
        if stage == "test" and self.test_dataset is not None:
            logger.info("setup('test') 已执行过，跳过重复预加载。")
            return

        # ── 读取 id 文件 ──────────────────────────────────────────────────────
        holdout_ids = set(_read_ids(os.path.join(self.step1_dir, "holdout_1000_ids.txt")))
        train_ids   = _read_ids(os.path.join(self.step1_dir, "train_ids.txt"))
        val_ids     = _read_ids(os.path.join(self.step1_dir, "val_ids.txt"))
        test_ids    = _read_ids(os.path.join(self.step1_dir, "test_ids.txt"))

        # ── 严禁 holdout 泄露 ─────────────────────────────────────────────────
        def _filter(ids):
            clean = [i for i in ids if i not in holdout_ids]
            leaked = len(ids) - len(clean)
            if leaked:
                logger.error(f"⚠️  从 split 中移除了 {leaked} 个 holdout mp_id！请检查 Step 1 划分逻辑。")
            return clean

        train_ids = _filter(train_ids)
        val_ids   = _filter(val_ids)
        test_ids  = _filter(test_ids)

        logger.info(f"Split 大小  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

        # ── 加载 inventory ────────────────────────────────────────────────────
        inventory_path = os.path.join(self.step1_dir, "data_inventory.csv")
        inventory_df   = pd.read_csv(inventory_path)

        # ── 实例化 Dataset ─────────────────────────────────────────────────────
        if stage in (None, "fit"):
            self.train_dataset = XASCrystalDataset(train_ids, inventory_df)
            self.val_dataset   = XASCrystalDataset(val_ids,   inventory_df)

        if stage in (None, "test"):
            self.test_dataset = XASCrystalDataset(test_ids, inventory_df)

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self):
        nw = self.num_workers["train"]
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=nw,
            collate_fn=xas_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(nw > 0),
        )

    def val_dataloader(self):
        nw = self.num_workers["val"]
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=nw,
            collate_fn=xas_collate_fn,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )

    def test_dataloader(self):
        nw = self.num_workers["test"]
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=nw,
            collate_fn=xas_collate_fn,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )

    def __repr__(self):
        return (
            f"XASDataModule("
            f"train={len(self.train_dataset) if self.train_dataset else 'N/A'}, "
            f"val={len(self.val_dataset) if self.val_dataset else 'N/A'}, "
            f"test={len(self.test_dataset) if self.test_dataset else 'N/A'})"
        )