"""
exp6_data_adapter.py — Exp4 PyG Batch → DETRXas forward input.

Bridges (handoff §2.2.2 + SA1 OUTPUT §6.4):

  Input: torch_geometric.data.Batch from xas_local_datamodule_v2.xas_collate_fn_v2
    Field names (Sub-Agent 4 renamed during Exp4 Phase 5b):
      batch.xmu_xanes        (B, 150)   float
      batch.chi1             (B, 200)   float
      batch.feff_features    (B, 74)    float
      batch.frac_coords      (B*20, 3)  float       node-level FLAT
      batch.atom_types       (B*20,)    int64       node-level FLAT, Z values
      batch.batch            (B*20,)    int64       sample idx per node
      batch.center_element   list[B] str            element symbol "Fe"/"O"/...
      batch.eval_cutoff      (B,)       float       (not used here; eval-only)

  Output: (model_batch dict, targets list[B])
    model_batch:
      xmu        (B, 150)   float    (renamed from xmu_xanes)
      chi1       (B, 200)   float
      feff       (B, 74)    float    (renamed from feff_features)
      center_idx (B,)       long     dense idx in model.center_Z_to_idx
    targets[i]:
      labels  (n_i,) long   dense idx in model.neighbor_Z_to_idx
      pos     (n_i, 3) float in [-0.5, 0.5]
      where n_i ≤ 20 after Z=0 filter (in practice == 20).

Constraints honored:
  * Vocab read from model attributes (model.center_Z_to_idx /
    model.neighbor_Z_to_idx), NOT re-loaded from json — anti-drift per SA1 §6.4.
  * Defensive Z=0 filter still applied (SA1 §6.1: belt-and-suspenders for
    cache/csv desync; in normal flow Exp4 datamodule None-drops invalid).
  * OOV center / neighbor element → KeyError raise with diagnostic. SA1
    smoke verified train vocab ⊇ all-train-neighbors; OOV implies val/test
    contains element absent from train. Handoff §4.1(c) requires raise, not
    silent index error.
  * Sanity print (handoff §2.2.2 mandatory): runs ONCE per process on first
    adapt() call, dumps shapes/dtypes/(atom_types==0).sum()/label-vocab range.
    Suppress flag is process-local — DataLoader workers each print once,
    typical num_workers=4 → 4 prints total at training start, acceptable.
"""
from __future__ import annotations

import torch
from pymatgen.core import Element

# Module-level latch so the sanity print runs ONCE per process, not per batch.
_FIRST_BATCH_PRINTED = False


def adapt(pyg_batch, model) -> tuple[dict | None, list[dict] | None]:
    """
    Convert one PyG Batch (xas_collate_fn_v2 output) into (model_batch, targets).

    Returns (None, None) if pyg_batch is None — happens when the entire
    DataLoader batch was dropped due to invalid samples (rare, ≤ 6/60507 ≈ 1e-4).
    Caller (training loop) must skip the step in that case.
    """
    if pyg_batch is None:
        return None, None

    B = int(pyg_batch.num_graphs)

    # ---------------- graph-level fields (rename to model contract) ----------
    xmu  = pyg_batch.xmu_xanes        # (B, 150)
    chi1 = pyg_batch.chi1             # (B, 200)
    feff = pyg_batch.feff_features    # (B, 74)
    device = xmu.device

    # ---------------- center_idx: symbol str → Z → dense idx ----------------
    c_map = model.center_Z_to_idx
    center_idx = torch.empty(B, dtype=torch.long, device=device)
    for i, sym in enumerate(pyg_batch.center_element):
        Z = Element(sym).Z
        if Z not in c_map:
            sn = (
                pyg_batch.sample_name[i]
                if hasattr(pyg_batch, "sample_name") else "?"
            )
            raise KeyError(
                f"OOV center element {sym!r} (Z={Z}) at batch idx {i} "
                f"sample={sn}; vocab has {len(c_map)} centers. "
                f"This means val/test contains a center element absent from "
                f"train — push MA1 (handoff §4.1(c) OOV policy)."
            )
        center_idx[i] = c_map[Z]

    # ---------------- node-level: split (B*20, ...) per sample --------------
    flat_atom_types  = pyg_batch.atom_types     # (B*20,) int64 Z values
    flat_frac_coords = pyg_batch.frac_coords    # (B*20, 3) float
    sample_idx       = pyg_batch.batch          # (B*20,) int64

    n_map = model.neighbor_Z_to_idx
    targets: list[dict] = []
    n_zeros_total = 0

    for i in range(B):
        node_mask = (sample_idx == i)
        s_atypes = flat_atom_types[node_mask]      # (20,)
        s_pos    = flat_frac_coords[node_mask]     # (20, 3)

        # Defensive Z=0 filter (SA1 §6.1)
        valid = s_atypes > 0
        n_zeros_total += int((~valid).sum().item())
        s_atypes = s_atypes[valid]
        s_pos    = s_pos[valid]

        # Z → dense neighbor idx
        n = s_atypes.numel()
        labels = torch.empty(n, dtype=torch.long, device=device)
        for j, Z in enumerate(s_atypes.tolist()):
            if Z not in n_map:
                raise KeyError(
                    f"OOV neighbor Z={Z} at batch idx {i} node {j}; "
                    f"vocab has {len(n_map)} neighbors. Train neighbor "
                    f"coverage broken — push MA1."
                )
            labels[j] = n_map[Z]

        targets.append({"labels": labels, "pos": s_pos.contiguous()})

    # ---------------- first-batch sanity (handoff §2.2.2) -------------------
    global _FIRST_BATCH_PRINTED
    if not _FIRST_BATCH_PRINTED:
        _print_first_batch_sanity(
            pyg_batch, model, center_idx, targets, n_zeros_total
        )
        _FIRST_BATCH_PRINTED = True

    model_batch = {
        "xmu":        xmu,
        "chi1":       chi1,
        "feff":       feff,
        "center_idx": center_idx,
    }
    return model_batch, targets


def _print_first_batch_sanity(pyg_batch, model, center_idx, targets, n_zeros):
    print("=" * 72)
    print("[exp6_data_adapter] FIRST BATCH SANITY (printed once per process)")
    print("=" * 72)
    B = int(pyg_batch.num_graphs)
    print(f"  batch_size B            = {B}")
    print(f"  xmu  shape/dtype        = {tuple(pyg_batch.xmu_xanes.shape)} "
          f"{pyg_batch.xmu_xanes.dtype}")
    print(f"  chi1 shape/dtype        = {tuple(pyg_batch.chi1.shape)} "
          f"{pyg_batch.chi1.dtype}")
    print(f"  feff shape/dtype        = {tuple(pyg_batch.feff_features.shape)} "
          f"{pyg_batch.feff_features.dtype}")
    print(f"  flat_atom_types         = {tuple(pyg_batch.atom_types.shape)} "
          f"{pyg_batch.atom_types.dtype}")
    print(f"  flat_frac_coords        = {tuple(pyg_batch.frac_coords.shape)} "
          f"{pyg_batch.frac_coords.dtype}")
    print(f"  center_idx              = {tuple(center_idx.shape)} {center_idx.dtype}")
    print(f"    range [{int(center_idx.min())}, {int(center_idx.max())}], "
          f"vocab N_CENTER={len(model.center_Z_to_idx)}")
    n_atoms_per = [t["labels"].numel() for t in targets]
    print(f"  n_atoms per sample      = {n_atoms_per}")
    print(f"  Z=0 padding entries     = {n_zeros}  (handoff §6.1 expects 0)")
    if n_zeros > 0:
        print("  ⚠️  Z=0 detected — valid_mask filtering may be off.")
    nonempty = [t for t in targets if t["labels"].numel() > 0]
    if nonempty:
        l_min = min(int(t["labels"].min()) for t in nonempty)
        l_max = max(int(t["labels"].max()) for t in nonempty)
        p_min = min(t["pos"].min().item() for t in nonempty)
        p_max = max(t["pos"].max().item() for t in nonempty)
        print(f"  labels range            = [{l_min}, {l_max}], "
              f"vocab N_NEIGHBOR={len(model.neighbor_Z_to_idx)}, "
              f"no_object_idx={model.no_object_idx}")
        print(f"  pos range               = [{p_min:.4f}, {p_max:.4f}]  "
              f"(expect ⊂ [-0.5, 0.5])")
    print("=" * 72)


def reset_first_batch_flag():
    """Test helper — re-arm the sanity print on next call."""
    global _FIRST_BATCH_PRINTED
    _FIRST_BATCH_PRINTED = False
