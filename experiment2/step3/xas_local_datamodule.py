"""
Step 3.2 — XASDataModule
=========================
PyTorch Lightning DataModule for XAS -> local Fe structure prediction.

核心职责：
  1. 读取 train/val/test ids 文件，构建三个 XASLocalStructureDataset
  2. collate_fn：过滤 None 样本，将 PyG Data 列表拼成 Batch
  3. num_workers=0（Windows 多进程不稳定）
"""

import os
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
import pytorch_lightning as pl

# 本目录下的 Dataset（同 step3 目录）
from xas_local_dataset_L6 import XASLocalStructureDataset
# ── 常量 ──────────────────────────────────────────────────────────────────────
DATA_ROOT    = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
STEP1_DIR    = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
FEFF_CSV     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"


# ── collate_fn ────────────────────────────────────────────────────────────────

def xas_collate_fn(batch: list) -> Optional[Batch]:
    """
    过滤 None 样本（邻居不足 20 或文件缺失），然后用 PyG Batch 拼接。

    PyG Batch.from_data_list 会自动处理：
      - frac_coords, atom_types  → 沿 node 维度拼接
      - lengths, angles          → 沿 batch 维度拼接
      - xmu_xanes, chi1, feff_features, eval_cutoff → 沿 batch 维度 stack
      - num_atoms, num_nodes     → 保留为列表 / 累加
      - batch.batch              → 自动生成 node->graph 映射
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)


# ── DataModule ────────────────────────────────────────────────────────────────

class XASDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    data_root : str
    step1_dir : str
        含 train_ids.txt / val_ids.txt / test_ids.txt / data_inventory.csv
    feff_feat_csv : str
    feff_scaler_path : str or None
    batch_size : int
    num_workers : int  (Windows 建议 0)
    L : float
    """

    def __init__(
        self,
        data_root: str         = DATA_ROOT,
        step1_dir: str         = STEP1_DIR,
        feff_feat_csv: str     = FEFF_CSV,
        feff_scaler_path: str  = None,
        batch_size: int        = 16,
        num_workers: int       = 0,
        L: float               = 12.0,
    ):
        super().__init__()
        self.data_root        = data_root
        self.step1_dir        = step1_dir
        self.feff_feat_csv    = feff_feat_csv
        self.feff_scaler_path = feff_scaler_path or os.path.join(
            step1_dir, 'feff_feature_scaler.pkl')
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.L           = L

        self._inventory_csv = os.path.join(step1_dir, 'data_inventory.csv')
        self._train_ids     = os.path.join(step1_dir, 'train_ids.txt')
        self._val_ids       = os.path.join(step1_dir, 'val_ids.txt')
        self._test_ids      = os.path.join(step1_dir, 'test_ids.txt')

        self.train_dataset: Optional[XASLocalStructureDataset] = None
        self.val_dataset:   Optional[XASLocalStructureDataset] = None
        self.test_dataset:  Optional[XASLocalStructureDataset] = None

    # -------------------------------------------------------------------------

    def _make_dataset(self, ids_file: str) -> XASLocalStructureDataset:
        return XASLocalStructureDataset(
            data_root        = self.data_root,
            inventory_csv    = self._inventory_csv,
            ids_file         = ids_file,
            feff_feat_csv    = self.feff_feat_csv,
            feff_scaler_path = self.feff_scaler_path,
            L                = self.L,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_dataset = self._make_dataset(self._train_ids)
            self.val_dataset   = self._make_dataset(self._val_ids)
            print(f"[DataModule] train={len(self.train_dataset)}  "
                  f"val={len(self.val_dataset)}")

        if stage in (None, 'test'):
            self.test_dataset = self._make_dataset(self._test_ids)
            print(f"[DataModule] test={len(self.test_dataset)}")

    # -------------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn,
            drop_last   = True,   # 避免最后一个 batch 只有 1 个样本导致 BN 报错
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn,
        )

    def __repr__(self) -> str:
        return (f"XASDataModule(batch_size={self.batch_size}, "
                f"L={self.L}A, num_workers={self.num_workers})")


# ── 快速测试 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dm = XASDataModule(batch_size=4)
    dm.setup('fit')

    loader = dm.train_dataloader()
    batch  = next(iter(loader))

    if batch is None:
        print("ERROR: 第一个 batch 为 None，请检查数据路径")
    else:
        print("=== Batch 字段检查 ===")
        print(f"num_graphs      : {batch.num_graphs}")
        print(f"num_atoms (sum) : {batch.num_atoms.sum().item()}")
        print(f"frac_coords     : {batch.frac_coords.shape}")   # (B*20, 3)
        print(f"atom_types      : {batch.atom_types.shape}")    # (B*20,)
        print(f"lengths         : {batch.lengths.shape}")       # (B, 3)
        print(f"xmu_xanes       : {batch.xmu_xanes.shape}")     # (B, 150)
        print(f"chi1            : {batch.chi1.shape}")          # (B, 200)
        print(f"feff_features   : {batch.feff_features.shape}") # (B, 73)
        print(f"eval_cutoff     : {batch.eval_cutoff.shape}")   # (B,)
        print(f"batch.batch     : {batch.batch.shape}")         # (B*20,)
        print(f"lengths[0]      : {batch.lengths[0]}")          # 应为 [12,12,12]
        print()
        print("OK: batch 构建成功")
