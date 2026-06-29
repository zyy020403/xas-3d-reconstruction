# =============================================================================
# 脚本编号: step3_qt (dataset)
# 脚本名称: step3_qt_dataset.py
# 输入:
#   - experiment/quicktest/qt_inventory.csv (通过 inventory_df 传入)
#   - chi.dat 谱文件
#   - POSCAR_supercell_fixed
# 输出:
#   - QTCrystalDataset: __getitem__ 返回单位点字典
#   - qt_collate_fn: 简化版 collate（n_sites 恒为 1，无需 padding）
# 说明:
#   QuickTest 简化版 dataset（对应正式服 xas_dataset.py）。
#   差异：
#     - 每个样本只有 1 个位点（qt_inventory 已保证每 mp_id 只有 1 行）
#     - n_sites 恒为 1，spectra/site_elements/is_ionic/quality_weights 维度固定
#     - collate_fn 直接 stack，不做 padding
#     - preprocess_chi 从 sys.modules["step2_1_spectrum_encoder"] 取（由 train 脚本注入）
# =============================================================================

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT      = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR    = os.path.join(PROJECT_ROOT, "experiment")
SITE_DATASET_DIR  = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"

# ── preprocess_chi：由 step3_qt_train.py 在 sys.modules 中注入 qt 版本 ────────
# 若直接运行本文件（单独测试），退回从正式服路径 import
if "step2_1_spectrum_encoder" not in sys.modules:
    _step2_dir = os.path.join(EXPERIMENT_DIR, "step2")
    if _step2_dir not in sys.path:
        sys.path.insert(0, _step2_dir)
from step2_1_spectrum_encoder import preprocess_chi  # noqa: E402

logger = logging.getLogger(__name__)

QUALITY_WEIGHT = {"A": 1.0, "B": 0.5, "C": 0.1}
QUALITY_WEIGHT_UNKNOWN = 0.3
SPECTRA_LEN = 512


# ─────────────────────────────────────────────────────────────────────────────
#  辅助：POSCAR 解析（与正式服相同）
# ─────────────────────────────────────────────────────────────────────────────

def _parse_poscar(poscar_path: str):
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(poscar_path)
        frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
        atom_types  = torch.tensor([s.specie.Z for s in struct], dtype=torch.long)
        lengths     = torch.tensor(list(struct.lattice.abc),    dtype=torch.float32)
        angles      = torch.tensor(list(struct.lattice.angles), dtype=torch.float32)
        return frac_coords, atom_types, lengths, angles, len(struct)
    except Exception as e:
        logger.warning(f"POSCAR 解析失败: {poscar_path}  ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  QTCrystalDataset
# ─────────────────────────────────────────────────────────────────────────────

class QTCrystalDataset(Dataset):
    """
    QuickTest 专用 Dataset。

    每个样本 = 一个化合物（mp_id）的单个 Fe 位点。
    n_sites 恒为 1，所有 spectra/elements/ionic/quality 张量第0维固定为1。

    __getitem__ 返回 dict：
        frac_coords     Tensor [N_atoms, 3]
        atom_types      Tensor [N_atoms]
        lengths         Tensor [3]
        angles          Tensor [3]
        num_atoms       int
        spectra         Tensor [1, 1, SPECTRA_LEN]   ← n_sites=1 固定
        site_elements   Tensor [1]                    ← 恒为 26 (Fe)
        is_ionic        Tensor [1]                    ← 恒为 0
        quality_weights Tensor [1]
        n_sites         int (= 1)
    """

    def __init__(self, mp_ids, inventory_df: pd.DataFrame):
        super().__init__()
        from pymatgen.core.periodic_table import Element as PmgElement

        mp_ids_set = set(int(x) for x in mp_ids)
        sub_df = inventory_df[inventory_df["mp_id"].isin(mp_ids_set)].copy()

        # qt_inventory 每个 mp_id 只有 1 行，直接遍历
        self._cache = []
        self.valid_mp_ids = []
        skipped = 0
        total = len(sub_df)
        logger.info(f"QTCrystalDataset 开始预加载（共 {total} 行）...")

        for idx, row in sub_df.iterrows():
            mp_id = int(row["mp_id"])

            # ── POSCAR ────────────────────────────────────────────────────────
            poscar_path = os.path.join(row["source_path"], "POSCAR_supercell_fixed")
            parsed = _parse_poscar(poscar_path)
            if parsed is None:
                skipped += 1
                continue
            frac_coords, atom_types, lengths, angles, num_atoms = parsed

            # ── 谱 ────────────────────────────────────────────────────────────
            chi_path = os.path.join(row["source_path"], "chi.dat")
            try:
                spec = preprocess_chi(chi_path)   # [1, 512]
            except Exception as e:
                logger.debug(f"chi.dat 失败: {chi_path} ({e})")
                spec = torch.zeros(1, SPECTRA_LEN, dtype=torch.float32)

            spectra = spec.unsqueeze(0)   # [1, 1, 512]

            # ── 元素（Fe=26）─────────────────────────────────────────────────
            try:
                z = PmgElement(row["element"]).Z
            except Exception:
                z = 26
            site_elements = torch.tensor([z], dtype=torch.long)

            # ── is_ionic（qt 只用 site_dataset，恒为 0）───────────────────────
            is_ionic = torch.tensor([0], dtype=torch.long)

            # ── quality_weight ────────────────────────────────────────────────
            tier = row.get("quality_tier", "")
            qw = QUALITY_WEIGHT.get(str(tier).upper(), QUALITY_WEIGHT_UNKNOWN)
            quality_weights = torch.tensor([qw], dtype=torch.float32)

            self._cache.append({
                "frac_coords":     frac_coords,
                "atom_types":      atom_types,
                "lengths":         lengths,
                "angles":          angles,
                "num_atoms":       num_atoms,
                "spectra":         spectra,
                "site_elements":   site_elements,
                "is_ionic":        is_ionic,
                "quality_weights": quality_weights,
                "n_sites":         1,
            })
            self.valid_mp_ids.append(mp_id)

        if skipped:
            logger.warning(f"跳过 {skipped} 个样本（POSCAR 解析失败）")
        logger.info(f"QTCrystalDataset 初始化完成，有效样本数: {len(self.valid_mp_ids)}")

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  qt_collate_fn（无需 padding，n_sites 恒为 1）
# ─────────────────────────────────────────────────────────────────────────────

def qt_collate_fn(batch):
    """
    QuickTest 简化版 collate。
    n_sites 恒为 1，不需要 padding，直接 stack 所有字段。

    输出 batch 额外属性：
        spectra         [B, 1, 1, 512]
        site_elements   [B, 1]
        is_ionic        [B, 1]
        quality_weights [B, 1]
        n_sites         [B]  （全为 1）
    """
    # ── 晶体字段 → PyG Batch ──────────────────────────────────────────────────
    data_list = [
        Data(
            frac_coords=item["frac_coords"],
            atom_types=item["atom_types"],
            lengths=item["lengths"].unsqueeze(0),
            angles=item["angles"].unsqueeze(0),
            num_atoms=item["num_atoms"],
            num_nodes=item["num_atoms"],
        )
        for item in batch
    ]
    pyg_batch = Batch.from_data_list(data_list)

    # ── 谱字段：直接 stack（n_sites=1，无需 padding）──────────────────────────
    pyg_batch.spectra         = torch.stack([item["spectra"]         for item in batch])  # [B, 1, 1, 512]
    pyg_batch.site_elements   = torch.stack([item["site_elements"]   for item in batch])  # [B, 1]
    pyg_batch.is_ionic        = torch.stack([item["is_ionic"]        for item in batch])  # [B, 1]
    pyg_batch.quality_weights = torch.stack([item["quality_weights"] for item in batch])  # [B, 1]
    pyg_batch.n_sites         = torch.ones(len(batch), dtype=torch.long)                  # [B]

    return pyg_batch