# =============================================================================
# 脚本编号: step3_qt (datamodule)
# 脚本名称: step3_qt_datamodule.py
# 输入:
#   - experiment/quicktest/qt_inventory.csv
#   - experiment/quicktest/qt_train_ids.txt / qt_val_ids.txt / qt_test_ids.txt
# 输出:
#   - QTDataModule: 提供 train / val / test DataLoader
# 说明:
#   QuickTest 简化版 datamodule（对应正式服 xas_datamodule.py）。
#   差异：
#     - 无 holdout 过滤
#     - batch_size=8, num_workers=0（Windows）
#   ⚠️  step3_qt_dataset 的 import 放在 setup() 内（懒加载），避免顶层 import
#       链出错时截断本文件，导致外部 "from step3_qt_datamodule import QTDataModule"
#       报 "cannot import name" 的假错误。
# =============================================================================

import os
import logging
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")


def _read_ids(path: str):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


class QTDataModule(pl.LightningDataModule):
    """
    QuickTest DataModule。

    Parameters
    ----------
    batch_size  : int  默认 8
    num_workers : int  默认 0（Windows 多进程 DataLoader 有坑）
    qt_dir      : str  quicktest 输出目录
    """

    def __init__(
        self,
        batch_size:  int = 8,
        num_workers: int = 0,
        qt_dir:      str = QT_DIR,
    ):
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.qt_dir      = qt_dir

        self.train_dataset    = None
        self.val_dataset      = None
        self.test_dataset     = None
        self._qt_collate_fn   = None   # setup() 时填充

        # 与正式服 run.py 接口对齐
        self.scaler         = None
        self.lattice_scaler = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        # 幂等保护
        if stage in (None, "fit") and self.train_dataset is not None:
            logger.info("setup('fit') 已执行过，跳过。")
            return
        if stage == "test" and self.test_dataset is not None:
            logger.info("setup('test') 已执行过，跳过。")
            return

        # ── 懒加载 step3_qt_dataset ──────────────────────────────────────────
        # 放在 setup() 里而不是模块顶层，原因：
        #   step3_qt_dataset 依赖 step2_1_spectrum_encoder（由 train 脚本注入
        #   sys.modules）。若在模块顶层 import，注入可能尚未完成或 import 链
        #   中途异常会截断本文件，导致 QTDataModule 找不到。
        try:
            from step3_qt_dataset import QTCrystalDataset, qt_collate_fn
        except Exception as exc:
            logger.error(f"导入 step3_qt_dataset 失败: {exc}")
            raise

        self._qt_collate_fn = qt_collate_fn

        # ── 读取 id 文件 ──────────────────────────────────────────────────────
        train_ids = _read_ids(os.path.join(self.qt_dir, "qt_train_ids.txt"))
        val_ids   = _read_ids(os.path.join(self.qt_dir, "qt_val_ids.txt"))
        test_ids  = _read_ids(os.path.join(self.qt_dir, "qt_test_ids.txt"))

        logger.info(
            f"QT Split  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}"
        )

        # ── inventory ─────────────────────────────────────────────────────────
        inventory_df = pd.read_csv(os.path.join(self.qt_dir, "qt_inventory.csv"))

        # ── 实例化 Dataset ─────────────────────────────────────────────────────
        if stage in (None, "fit"):
            self.train_dataset = QTCrystalDataset(train_ids, inventory_df)
            self.val_dataset   = QTCrystalDataset(val_ids,   inventory_df)

        if stage in (None, "test"):
            self.test_dataset = QTCrystalDataset(test_ids, inventory_df)

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._qt_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._qt_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._qt_collate_fn,
            pin_memory=True,
        )