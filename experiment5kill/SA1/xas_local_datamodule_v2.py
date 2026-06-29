"""
xas_local_datamodule_v2.py — Exp4 Step 3 Phase 5b

PyTorch Lightning DataModule for XasLocalDatasetV2 + diffusion_w_type_xas.CSPDiffusion.

Sub-Agent 4 (Phase 5/5b/6) findings — bridged here, dataset_v2 untouched:
  Fact 1 (field naming):
    dataset_v2 returns dict with {xmu, chi1, feff, ...}
    diffusion_w_type_xas.py forward()/sample() reference batch.xmu_xanes /
    batch.chi1 / batch.feff_features
    → renamed in _dict_to_pyg_data: xmu→xmu_xanes, feff→feff_features.
  Fact 2 (dict vs PyG Batch):
    dataset_v2 returns dict; forward() expects PyG Batch with num_graphs /
    lengths / angles / num_atoms / num_nodes / batch.batch.
    → wrap each dict into PyG Data with virtual-lattice fields (lengths=[6,6,6],
      angles=[90,90,90], num_atoms=N_NEIGHBORS=20). Batch.from_data_list
      auto-derives num_graphs / batch.batch.
  Fact 3 (atoms count):
    Old xas_local_dataset_L6.py uses num_atoms=N_NEIGHBORS=20 (no center) per
    `Data(...)` construction (verified by Sub-Agent 4 in Phase 5/5b prep grep).
    Identical to dataset_v2 schema → bit-exact node-count alignment, no center
    insertion needed.

HANDOFF §7.1 constraints (all enforced):
  1. ✓ filename: xas_local_datamodule_v2.py
  2. ✓ classname: XasLocalDataModuleV2
  3. ✓ from xas_local_dataset_v2 import XasLocalDatasetV2
  4. ✓ data_dir from os.environ.get("EXP4_DATA_DIR", default)
  5. ✓ no v1 73-dim FEFF_CSV / 75cols / MP_all_EXAFS path constants
  6. ✓ PL 2.5.5 setup() signature: stage: Optional[str] = None
  7. ✓ four-method structure: setup / train_dataloader / val_dataloader / test_dataloader
  8. ✓ holdout NOT loaded (Step 5 instantiates separately)

Check Agent F5 M6 fix:
  dataset_v2 takes data_dir as constructor arg (not env var). DataModule reads
  env var here and passes through explicitly.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from xas_local_dataset_v2 import XasLocalDatasetV2

# ============================================================================
# Virtual lattice constants (HANDOFF §4.1 immutable; mirror dataset_v2 + L6)
# ============================================================================
L_VIRTUAL   = 6.0
N_NEIGHBORS = 20


# ============================================================================
# dict (dataset_v2 output) → PyG Data (forward() input contract)
# ============================================================================

def _dict_to_pyg_data(s: dict) -> Data:
    """
    Wrap one dataset_v2 dict sample into a PyG Data object aligned with
    diffusion_w_type_xas.py forward()/sample() attribute access pattern.

    Field mapping:
      Node-level (sized N_NEIGHBORS=20):
        s["frac_coords"]  (20, 3) float32  → data.frac_coords
        s["atom_types"]   (20,)   int64    → data.atom_types

      Graph-level (per-sample, stack along batch dim via PyG):
        s["xmu"]    (150,) float32  →  data.xmu_xanes     (1, 150)
        s["chi1"]   (200,) float32  →  data.chi1          (1, 200)
        s["feff"]   (74,)  float32  →  data.feff_features (1, 74)

      Virtual-lattice graph-level (mirror old dataset_L6):
        data.lengths    (1, 3)  [L_VIRTUAL, L_VIRTUAL, L_VIRTUAL]
        data.angles     (1, 3)  [90, 90, 90]
        data.num_atoms  int     20   (PyG collects list[int] → (B,) LongTensor)
        data.num_nodes  int     20   (PyG uses for batch.batch construction)

      Shell metadata (Step 5 audit; graph-level (1,)-tensors → (B,) at batch):
        eval_cutoff          float32
        eval_cutoff_fallback float32 (bool → float)
        n_center_sites       long

      Exp5 SA1 — center-element conditioning (graph-level (1,)-tensor → (B,) at batch):
        s["center_element_Z"]  int   →  data.center_element_Z  long
        SpectrumEncoder reads this through nn.Embedding(95, 16) lookup
        and concatenates with spectrum latent (256 + 16 = 272-dim).

      String metadata (Step 5 audit; PyG collects to list[str] in batch):
        sample_name, mp_id, center_element, site_equivalence_tag
    """
    data = Data(
        # ---- Node-level (PyG dim=0 stack, → (B*20, ...)) ----
        frac_coords = s["frac_coords"],          # (20, 3) float32
        atom_types  = s["atom_types"],           # (20,)   int64

        # ---- Graph-level (renamed for forward compatibility) ----
        xmu_xanes     = s["xmu"].unsqueeze(0),   # (1, 150) → (B, 150)
        chi1          = s["chi1"].unsqueeze(0),  # (1, 200) → (B, 200)
        feff_features = s["feff"].unsqueeze(0),  # (1, 74)  → (B, 74)

        # ---- Virtual lattice (mirror old dataset_L6 contract) ----
        lengths   = torch.tensor([[L_VIRTUAL, L_VIRTUAL, L_VIRTUAL]], dtype=torch.float32),
        angles    = torch.tensor([[90.0, 90.0, 90.0]], dtype=torch.float32),
        num_atoms = N_NEIGHBORS,   # int; PyG → (B,) LongTensor at batch
        num_nodes = N_NEIGHBORS,   # int; PyG uses for batch.batch construction

        # ---- Shell metadata (Step 5 audit, graph-level (1,)-tensors) ----
        eval_cutoff          = torch.tensor([s["eval_cutoff"]],          dtype=torch.float32),
        eval_cutoff_fallback = torch.tensor([float(s["eval_cutoff_fallback"])], dtype=torch.float32),
        n_center_sites       = torch.tensor([s["n_center_sites"]],       dtype=torch.long),

        # ---- Exp5 SA1: center-element conditioning ----
        # Graph-level (1,) long tensor → (B,) at batch; SpectrumEncoder uses
        # nn.Embedding(95, 16) lookup on this for center-element conditioning.
        center_element_Z     = torch.tensor([s["center_element_Z"]],     dtype=torch.long),
    )
    # ---- String metadata (PyG batches collect to list[str]) ----
    # forward() / sample() do NOT touch these. Step 5 audit reads them.
    data.sample_name          = s["sample_name"]
    data.mp_id                = s["mp_id"]
    data.center_element       = s["center_element"]
    data.site_equivalence_tag = s["site_equivalence_tag"]
    return data


def xas_collate_fn_v2(batch: list) -> Optional[Batch]:
    """
    Collate list of dataset_v2 dict samples into a PyG Batch.

    HANDOFF §6.5 R2: dataset_v2 raises RuntimeError on any malformed sample
    (frac sentinel / center missing / <20 neighbors). No silent None to filter.
    Defensive None-drop kept for safety; len==0 returns None (DataLoader handles).
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    data_list = [_dict_to_pyg_data(b) for b in batch]
    return Batch.from_data_list(data_list)


# ============================================================================
# DataModule
# ============================================================================

class XasLocalDataModuleV2(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule wrapping XasLocalDatasetV2 for diffusion training.

    Args:
        batch_size  : default 16 (HANDOFF §4.1 immutable batch_size for training)
        num_workers : default 0 (Linux/Windows compat; Step 4 may tune up)
        data_dir    : optional explicit override; otherwise EXP4_DATA_DIR env var,
                      otherwise default "/home/tcat/diffcsp_exp4/data".

    Notes:
      - holdout NEVER loaded (HANDOFF §7.1 item 8). Step 5 instantiates separately.
      - test_ds loaded at stage="test" or stage=None (Phase 6 forward test path).
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Explicit > env var > default (Check Agent F5 M6 resolution)
        self.data_dir = (
            data_dir
            if data_dir is not None
            else os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")
        )
        self.train_ds: Optional[XasLocalDatasetV2] = None
        self.val_ds:   Optional[XasLocalDatasetV2] = None
        self.test_ds:  Optional[XasLocalDatasetV2] = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = XasLocalDatasetV2(split="train", data_dir=self.data_dir)
            self.val_ds   = XasLocalDatasetV2(split="val",   data_dir=self.data_dir)
            print(f"[XasLocalDataModuleV2] train={len(self.train_ds)} val={len(self.val_ds)}")
        if stage in (None, "test"):
            self.test_ds = XasLocalDatasetV2(split="test", data_dir=self.data_dir)
            print(f"[XasLocalDataModuleV2] test={len(self.test_ds)}")
        # holdout NOT loaded (HANDOFF §7.1 item 8)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn_v2,
            drop_last   = True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn_v2,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = xas_collate_fn_v2,
        )

    def __repr__(self) -> str:
        return (
            f"XasLocalDataModuleV2(batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}, data_dir={self.data_dir})"
        )


# ============================================================================
# Smoke test (not the official Phase 6 forward_test.py)
# ============================================================================

if __name__ == "__main__":
    dm = XasLocalDataModuleV2(batch_size=4, num_workers=0)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    if batch is None:
        print("ERROR: first batch is None")
    else:
        print("=== Batch field shapes ===")
        print(f"  num_graphs:    {batch.num_graphs}")
        print(f"  num_atoms:     {batch.num_atoms} (expect tensor [20,20,20,20])")
        print(f"  frac_coords:   {tuple(batch.frac_coords.shape)}     (expect (80, 3))")
        print(f"  atom_types:    {tuple(batch.atom_types.shape)}      (expect (80,))")
        print(f"  lengths:       {tuple(batch.lengths.shape)}         (expect (4, 3))")
        print(f"  angles:        {tuple(batch.angles.shape)}          (expect (4, 3))")
        print(f"  xmu_xanes:     {tuple(batch.xmu_xanes.shape)}       (expect (4, 150))")
        print(f"  chi1:          {tuple(batch.chi1.shape)}             (expect (4, 200))")
        print(f"  feff_features: {tuple(batch.feff_features.shape)}    (expect (4, 74))")
        print(f"  eval_cutoff:   {tuple(batch.eval_cutoff.shape)}      (expect (4,))")
        print(f"  batch.batch:   {tuple(batch.batch.shape)}            (expect (80,))")
        print(f"  lengths[0]:    {batch.lengths[0]}             (expect [6, 6, 6])")
        print(f"  mp_id list:    {batch.mp_id}                  (expect list of 4 strings)")
        print()
        print("OK: PyG Batch construction succeeded.")
