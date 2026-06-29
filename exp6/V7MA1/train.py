"""
Exp6 v7 — step2/step2.1_train.py
Full training script. Proposal §6 + §7.1 + §7.2 + §附录B.

Key design decisions:
  - Single GPU (device=1, Exp5' uses GPU0)
  - batch_size=64, bf16 precision
  - 6-loss logging: ce / pos / rep / pmin / sdist / scount
  - CPS-based best ckpt selection (val_cps_mean)
  - shell collate fix: xas_collate_fn_v2 stores shell fields as
    _shell_starts_list etc. on the Batch; adapter reads per-index

Run from /home/tcat/experiment6_v7:
  CUDA_VISIBLE_DEVICES=1 /home/tcat/conda_envs/mlff/bin/python step2/step2.1_train.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric.data import Batch

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/home/tcat/experiment6_v7")
sys.path.insert(0, str(BASE / "shared"))

import composite_score as cs
import eval_metrics    as em
from detr_xas          import build_detr_xas
from matcher           import HungarianMatcher
from criterion         import SetCriterion
from exp6_data_adapter import adapt, reset_first_batch_flag

# ── Vocab / constants ───────────────────────────────────────────────────────
with open(BASE / "shared" / "exp6_element_vocab.json") as _f:
    _V = json.load(_f)
N_CENTER_TYPES   = int(_V["center"]["N_TYPES"])      # 88
N_NEIGHBOR_TYPES = int(_V["neighbor"]["N_TYPES"])    # 89
NO_OBJECT_IDX    = int(_V["neighbor"]["no_object_idx"])  # 89
IDX_TO_Z = {int(k): int(v) for k, v in _V["neighbor"]["idx_to_Z"].items()}

MIN_PDIST = cs.init_constants()
cs.set_no_object_idx(NO_OBJECT_IDX)
em.set_no_object_idx(NO_OBJECT_IDX)

L        = 20.0
LENGTHS  = torch.tensor([L, L, L])

# ── Hyperparameters ────────────────────────────────────────────────────────
MAX_EPOCHS       = 300
BATCH_SIZE       = 64
LR_TRANSFORMER   = 1e-4
LR_TOKENIZER     = 1e-5
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 0.1
EARLY_STOP_PAT   = 30
SAVE_TOP_K       = 3
CHECK_VAL_EVERY  = 1
NUM_WORKERS      = 4

LAMBDA_CLS    = 1.0
LAMBDA_POS    = 5.0
LAMBDA_REP    = 1.0
LAMBDA_PMIN   = 1.0
LAMBDA_SDIST  = 0.5
LAMBDA_SCOUNT = 0.2
EOS_COEF      = 0.1

CKPT_DIR = BASE / "checkpoints"
LOG_DIR  = BASE / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Shell-aware collate fix ─────────────────────────────────────────────────
# PyG Batch.from_data_list does not collate variable-length Python lists.
# We store shell fields as a flat Python list on the Batch object so
# adapter can retrieve per-sample shells by index.

sys.path.insert(0, str(BASE / "shared"))
from xas_local_datamodule_v2 import XasLocalDataModuleV2, xas_collate_fn_v2
from torch_geometric.data import Data


def _shell_collate_fn(raw_batch: list) -> Optional[Batch]:
    """
    Drop None, build PyG Batch, then attach shell fields as
    Batch._shell_starts_list / _shell_ends_list / _shell_n_atoms_list
    so adapter can do pyg_batch._shell_starts_list[i].
    """
    raw_batch = [b for b in raw_batch if b is not None]
    if not raw_batch:
        return None

    # Extract shell fields before _dict_to_pyg_data strips them
    # (shell_starts etc. ARE in the dict from dataset_v2)
    shell_starts_list  = [b.get("shell_starts",  []) for b in raw_batch]
    shell_ends_list    = [b.get("shell_ends",    []) for b in raw_batch]
    shell_n_atoms_list = [b.get("shell_n_atoms", []) for b in raw_batch]

    # Normal PyG collate
    from xas_local_datamodule_v2 import _dict_to_pyg_data
    data_list = [_dict_to_pyg_data(b) for b in raw_batch]
    pyg_batch = Batch.from_data_list(data_list)

    # Attach as plain Python lists (not tensors) — adapter reads by index
    pyg_batch._shell_starts_list  = shell_starts_list
    pyg_batch._shell_ends_list    = shell_ends_list
    pyg_batch._shell_n_atoms_list = shell_n_atoms_list

    return pyg_batch


# ── Updated adapter wrapper that reads shell fields from Batch ──────────────
# We monkey-patch the adapt function to pass shell fields correctly.
import exp6_data_adapter as _adapter_mod

_original_adapt = _adapter_mod.adapt

def _adapt_with_shells(pyg_batch, model):
    """
    Wrapper around adapt() that injects shell fields into targets
    from pyg_batch._shell_starts_list[i] etc.
    """
    model_batch, targets = _original_adapt(pyg_batch, model)
    if targets is None:
        return model_batch, targets

    for i, t in enumerate(targets):
        # Override shell fields with correctly collated per-sample data
        try:
            t["shell_starts"]  = pyg_batch._shell_starts_list[i]
            t["shell_ends"]    = pyg_batch._shell_ends_list[i]
            t["shell_n_atoms"] = pyg_batch._shell_n_atoms_list[i]
        except (AttributeError, IndexError):
            # Fallback: leave whatever adapter set (may be empty list)
            pass
    return model_batch, targets


# ── LightningModule ─────────────────────────────────────────────────────────
class DETRXasLightning(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = build_detr_xas(
            n_neighbor_types=N_NEIGHBOR_TYPES,
            n_center_types=N_CENTER_TYPES,
            no_object_idx=NO_OBJECT_IDX,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            n_queries=20,
        )
        weight_dict = {
            "loss_ce":     LAMBDA_CLS,
            "loss_pos":    LAMBDA_POS,
            "loss_rep":    LAMBDA_REP,
            "loss_pmin":   LAMBDA_PMIN,
            "loss_sdist":  LAMBDA_SDIST,
            "loss_scount": LAMBDA_SCOUNT,
        }
        for i in range(5):
            for k in list(weight_dict.keys()):
                weight_dict[f"{k}_{i}"] = weight_dict[k]

        matcher = HungarianMatcher(cost_class=LAMBDA_CLS, cost_pos=LAMBDA_POS)
        self.criterion = SetCriterion(
            num_classes=N_NEIGHBOR_TYPES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=EOS_COEF,
            losses=["labels", "cardinality", "pos", "repulsion",
                    "pairwise_min", "shell_dist", "shell_count"],
            lengths=LENGTHS.tolist(),
            no_object_idx=NO_OBJECT_IDX,
            min_pdist=MIN_PDIST,
        )
        self.weight_dict = weight_dict

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[DETRXasLightning] params={n_params:,}")

        # epoch-level accumulators
        self._val_cps_scores  = []
        self._val_pv_pass     = []
        self._val_rmsd        = []
        self._val_loss_items  = {k: [] for k in
                                 ["loss_ce","loss_pos","loss_rep",
                                  "loss_pmin","loss_sdist","loss_scount"]}

    # ── forward helpers ──────────────────────────────────────────────────

    def _compute_loss(self, pyg_batch):
        reset_first_batch_flag()
        model_batch, targets = _adapt_with_shells(pyg_batch, self.model)
        if model_batch is None:
            return None, None, None
        outputs = self.model(model_batch)
        loss_dict = self.criterion(outputs, targets)
        # Weighted sum
        total = sum(self.weight_dict[k] * loss_dict[k]
                    for k in loss_dict if k in self.weight_dict
                    and torch.isfinite(loss_dict[k]))
        return total, loss_dict, outputs

    # ── training step ───────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        total, loss_dict, _ = self._compute_loss(batch)
        if total is None:
            return None
        # Log 6 main-layer loss items
        for k in ["loss_ce","loss_pos","loss_rep","loss_pmin","loss_sdist","loss_scount"]:
            if k in loss_dict and torch.isfinite(loss_dict[k]):
                self.log(f"train_{k}", loss_dict[k].item(),
                         on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss", total.item(),
                 on_step=True, on_epoch=True, prog_bar=True)
        return total

    # ── validation step ─────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        total, loss_dict, outputs = self._compute_loss(batch)
        if total is None:
            return

        self.log("val_loss", total.item(),
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k in ["loss_ce","loss_pos","loss_rep","loss_pmin","loss_sdist","loss_scount"]:
            if k in loss_dict and torch.isfinite(loss_dict[k]):
                self._val_loss_items[k].append(loss_dict[k].item())

        # DETR health metrics
        argmax = outputs["pred_logits"].detach().argmax(-1)  # (B, 20)
        pred_pos = outputs["pred_pos"].detach()

        noobj_ratio = (argmax == NO_OBJECT_IDX).float().mean().item()
        qdiv = pred_pos.std(dim=1).mean(dim=-1).mean().item()

        # pairwise_violation_rate (hard indicator, B.1 resolution)
        pvr = cs.pairwise_violation_rate(pred_pos, argmax, LENGTHS.to(pred_pos.device))

        # CPS per sample
        reset_first_batch_flag()
        _, targets = _adapt_with_shells(batch, self.model)
        if targets is not None:
            for bi in range(pred_pos.shape[0]):
                name = targets[bi].get("sample_name", None)
                if name is None:
                    continue
                try:
                    score, bd = cs.composite_physical_score(
                        pred_pos[bi], argmax[bi], name,
                        LENGTHS.to(pred_pos.device), IDX_TO_Z)
                    self._val_cps_scores.append(score)
                    self._val_pv_pass.append(1 if bd["PV"] else 0)
                except Exception:
                    pass

            # RMSD (Hungarian)
            for bi in range(pred_pos.shape[0]):
                gt_pos   = targets[bi]["pos"]
                gt_types = targets[bi]["labels"]
                rmsd, _  = em.hungarian_rmsd(
                    pred_pos[bi], argmax[bi], gt_pos, gt_types,
                    LENGTHS.to(pred_pos.device))
                if rmsd != float("inf"):
                    self._val_rmsd.append(rmsd)

        self.log("val_noobj_ratio", noobj_ratio,
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_query_div",   qdiv,
                 on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_pv_rate",     1.0 - pvr,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # CPS mean
        if self._val_cps_scores:
            cps_mean = float(np.mean(self._val_cps_scores))
            pv_rate  = float(np.mean(self._val_pv_pass))
            self.log("val_cps_mean", cps_mean, prog_bar=True)
            self.log("val_pv_pass_rate", pv_rate, prog_bar=False)
        else:
            self.log("val_cps_mean", 0.0, prog_bar=True)

        # RMSD
        if self._val_rmsd:
            self.log("val_rmsd", float(np.mean(self._val_rmsd)), prog_bar=True)

        # 6-loss breakdown
        for k, vals in self._val_loss_items.items():
            if vals:
                self.log(f"val_{k}", float(np.mean(vals)), prog_bar=False)

        # Proposal §5.3 ratio check (log only, no assertion during training)
        items = {k: float(np.mean(v)) for k, v in self._val_loss_items.items() if v}
        if "loss_ce" in items and items["loss_ce"] > 0:
            for k in ["loss_pos","loss_rep","loss_pmin","loss_sdist","loss_scount"]:
                if k in items:
                    self.log(f"val_ratio_{k}_over_ce",
                             items[k] / items["loss_ce"], prog_bar=False)

        # Reset accumulators
        self._val_cps_scores  = []
        self._val_pv_pass     = []
        self._val_rmsd        = []
        for k in self._val_loss_items:
            self._val_loss_items[k] = []

    # ── optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Two param groups: tokenizer (lower LR) vs rest
        tokenizer_params = list(self.model.spectrum_tokenizer.parameters())
        tokenizer_ids    = {id(p) for p in tokenizer_params}
        other_params     = [p for p in self.model.parameters()
                            if id(p) not in tokenizer_ids]
        optimizer = torch.optim.AdamW([
            {"params": tokenizer_params, "lr": LR_TOKENIZER},
            {"params": other_params,     "lr": LR_TRANSFORMER},
        ], weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=200, gamma=0.1)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── DataModule with shell-aware collate ─────────────────────────────────────
class ShellAwareDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        from xas_local_datamodule_v2 import XasLocalDataModuleV2
        self._inner = XasLocalDataModuleV2(
            data_dir=str(BASE / "data"),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

    def setup(self, stage=None):
        self._inner.setup(stage)

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self._inner.train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=_shell_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self._inner.val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=_shell_collate_fn,
            pin_memory=True,
        )


# ── Callbacks ───────────────────────────────────────────────────────────────
def build_callbacks():
    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="epoch{epoch:03d}-cps{val_cps_mean:.4f}",
        monitor="val_cps_mean",
        mode="max",
        save_top_k=SAVE_TOP_K,
        save_last=True,
        verbose=True,
    )
    early_cb = EarlyStopping(
        monitor="val_cps_mean",
        mode="max",
        patience=EARLY_STOP_PAT,
        verbose=True,
    )
    return [ckpt_cb, early_cb]


# ── Sanity-run flag ──────────────────────────────────────────────────────────
# Set SANITY_RUN=1 for 5-epoch sanity (1000 samples subset)
SANITY_RUN = os.environ.get("SANITY_RUN", "0") == "1"


class SanitySubsetDataModule(ShellAwareDataModule):
    """Subset to 1000 train / 200 val for 5-epoch sanity check."""
    def train_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        ds = self._inner.train_ds
        idx = list(range(min(1000, len(ds))))
        return DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=_shell_collate_fn,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        ds = self._inner.val_ds
        idx = list(range(min(200, len(ds))))
        return DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, collate_fn=_shell_collate_fn,
                          pin_memory=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(42, workers=True)

    dm_cls  = SanitySubsetDataModule if SANITY_RUN else ShellAwareDataModule
    dm      = dm_cls()
    model   = DETRXasLightning()
    cbs     = build_callbacks()

    max_ep  = 5 if SANITY_RUN else MAX_EPOCHS
    print(f"\n{'='*60}")
    print(f"  SANITY_RUN={SANITY_RUN}  max_epochs={max_ep}  "
          f"batch={BATCH_SIZE}  GPU=CUDA_VISIBLE_DEVICES")
    print(f"  MIN_PDIST={MIN_PDIST:.4f} Å  L={L}  "
          f"N_NEIGHBOR={N_NEIGHBOR_TYPES}  NO_OBJ={NO_OBJECT_IDX}")
    print(f"{'='*60}\n")

    trainer = pl.Trainer(
        max_epochs=max_ep,
        accelerator="gpu",
        devices=1,          # uses whatever CUDA_VISIBLE_DEVICES exposes
        precision="bf16-mixed",
        gradient_clip_val=GRAD_CLIP,
        check_val_every_n_epoch=CHECK_VAL_EVERY,
        callbacks=cbs,
        default_root_dir=str(LOG_DIR),
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)

    best = cbs[0].best_model_path
    print(f"\n[DONE] best ckpt: {best}")
    print(f"       best val_cps_mean: {cbs[0].best_model_score}")


if __name__ == "__main__":
    main()
