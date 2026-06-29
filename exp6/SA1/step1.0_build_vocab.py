"""
Exp6 Step 1.0 — Build neighbor & center element vocabs.

Source data (Exp4 cached):
  - center_vocab:   /home/tcat/diffcsp_exp4/data/train_samples_v2.csv ['center_element']
  - neighbor_vocab: /home/tcat/diffcsp_exp4/data/train_structure_cache.pt ['atom_types']
                    Tensor (60507, 20) int64, Z values 0-94, Z=0 = padding (excluded)

Output:
  - /home/tcat/experiment6/shared/exp6_element_vocab.json

Per proposal §4.1(c) v3 resolution:
  - dense vocab, no Z gaps
  - no_object index = N_NEIGHBOR_TYPES (extra slot beyond dense)
  - assert N_NEIGHBOR_TYPES >= N_CENTER_TYPES
  - if center has elements not in neighbor, push MA1

CRITICAL handling of Z=0 padding (SA1 finding from schema dump):
  Exp4 train_structure_cache.pt['atom_types'] uses Z=0 to pad samples with
  fewer than 20 neighbors. This is operationally distinct from a real element
  (H = Z=1). We exclude Z=0 from the dense neighbor vocab.

  Documented in EXP6_PHASE1_OUTPUT.md "implementation choices".
"""
import json
import os
import torch
import pandas as pd
from pymatgen.core import Element

EXP4_DATA_ROOT = '/home/tcat/diffcsp_exp4/data'
OUT_PATH = '/home/tcat/experiment6/shared/exp6_element_vocab.json'

# Padding sentinel in train_structure_cache.pt['atom_types']
# Z=0 is not a real element; H is Z=1. Confirmed by Step 0.6 ε dump.
PADDING_Z = 0


def build_center_vocab():
    """88 unique center elements from train_samples_v2.csv."""
    csv_path = f'{EXP4_DATA_ROOT}/train_samples_v2.csv'
    df = pd.read_csv(csv_path)
    print(f"[center] CSV shape: {df.shape}")

    elem_strs = sorted(df['center_element'].unique())
    print(f"[center] n_unique elements (string form): {len(elem_strs)}")

    Z_list = sorted(set(Element(e).Z for e in elem_strs))
    print(f"[center] Z range: {min(Z_list)} to {max(Z_list)}, n={len(Z_list)}")
    return Z_list


def build_neighbor_vocab():
    """Unique Z values in train_structure_cache.pt['atom_types'], excluding padding."""
    cache_path = f'{EXP4_DATA_ROOT}/train_structure_cache.pt'
    print(f"[neighbor] loading {cache_path}...")
    d = torch.load(cache_path, weights_only=False)

    at = d['atom_types']  # (60507, 20) int64
    print(f"[neighbor] atom_types shape: {tuple(at.shape)}, dtype: {at.dtype}")
    valid_mask = d['valid_mask']
    print(f"[neighbor] valid_mask: {valid_mask.sum().item()}/{valid_mask.numel()} samples valid")

    uniq_all = sorted(torch.unique(at).tolist())
    print(f"[neighbor] all unique values (incl padding): n={len(uniq_all)}")

    Z_list = sorted([z for z in uniq_all if z != PADDING_Z])
    print(f"[neighbor] after excluding Z={PADDING_Z} padding: n={len(Z_list)}")
    print(f"[neighbor] range {min(Z_list)} to {max(Z_list)}")

    n_padding = (at == PADDING_Z).sum().item()
    n_total = at.numel()
    print(f"[neighbor] padding cells: {n_padding}/{n_total} ({100*n_padding/n_total:.2f}%)")

    return Z_list


def main():
    print("=" * 70)
    print("Exp6 Step 1.0: build element vocabs")
    print("=" * 70)

    center_Z = build_center_vocab()
    print()
    neighbor_Z = build_neighbor_vocab()
    print()

    N_CENTER_TYPES = len(center_Z)
    N_NEIGHBOR_TYPES = len(neighbor_Z)

    # Set-diff diagnostics (proposal §4.1(c) assertion)
    center_set = set(center_Z)
    neighbor_set = set(neighbor_Z)
    center_minus_neighbor = sorted(center_set - neighbor_set)
    neighbor_minus_center = sorted(neighbor_set - center_set)

    print("=" * 70)
    print(f"N_CENTER_TYPES         = {N_CENTER_TYPES}")
    print(f"N_NEIGHBOR_TYPES       = {N_NEIGHBOR_TYPES}")
    print(f"|center \\ neighbor|    = {len(center_minus_neighbor)}: {center_minus_neighbor}")
    print(f"|neighbor \\ center|    = {len(neighbor_minus_center)}: {neighbor_minus_center}")
    print(f"|intersection|         = {len(center_set & neighbor_set)}")
    print()

    # Proposal §4.1(c) assertion
    if len(center_minus_neighbor) > 0:
        print("!" * 70)
        print("WARNING: center has elements absent from neighbor vocab.")
        print("Proposal §4.1(c): 'N_NEIGHBOR_TYPES >= N_CENTER_TYPES (邻居元素至少包含所有中心元素)'")
        print(f"Violating elements: Z = {center_minus_neighbor}")
        print("These centers can never produce themselves as neighbors → vocab consistency risk.")
        print("Handoff §7: PUSH BACK to MA1 before SA2 starts.")
        print("!" * 70)
    else:
        print("OK: center elements ⊆ neighbor elements (proposal §4.1(c) holds)")

    if N_NEIGHBOR_TYPES < N_CENTER_TYPES:
        raise AssertionError(
            f"FATAL: N_NEIGHBOR_TYPES ({N_NEIGHBOR_TYPES}) < N_CENTER_TYPES "
            f"({N_CENTER_TYPES}). center \\ neighbor = {center_minus_neighbor}. "
            f"Push MA1 immediately."
        )
    print()

    # Build dense Z↔idx mappings
    # JSON note: int keys are auto-converted to str by json.dump.
    # Loaders (detr_xas.py, dataset adapter) MUST cast back: int(key).
    center_Z_to_idx = {int(z): i for i, z in enumerate(center_Z)}
    neighbor_Z_to_idx = {int(z): i for i, z in enumerate(neighbor_Z)}

    vocab = {
        'center': {
            'N_TYPES': N_CENTER_TYPES,
            'Z_to_idx': center_Z_to_idx,
            'idx_to_Z': {i: int(z) for z, i in center_Z_to_idx.items()},
        },
        'neighbor': {
            'N_TYPES': N_NEIGHBOR_TYPES,
            'Z_to_idx': neighbor_Z_to_idx,
            'idx_to_Z': {i: int(z) for z, i in neighbor_Z_to_idx.items()},
            'no_object_idx': N_NEIGHBOR_TYPES,
            'padding_Z_excluded': PADDING_Z,
        },
        'meta': {
            'source': {
                'center': f'{EXP4_DATA_ROOT}/train_samples_v2.csv [center_element]',
                'neighbor': f'{EXP4_DATA_ROOT}/train_structure_cache.pt [atom_types]',
            },
            'sa1_decision_padding': (
                f'Z={PADDING_Z} excluded from neighbor vocab '
                '(Exp4 padding for samples with < 20 real neighbors)'
            ),
            'proposal_ref': 'EXP6_PROPOSAL_v3.md §4.1(c)',
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Wrote: {OUT_PATH}")
    print()

    print("First 5 center Z→idx mappings (sanity):")
    for k, v in list(center_Z_to_idx.items())[:5]:
        try:
            sym = Element.from_Z(k).symbol
        except Exception:
            sym = '?'
        print(f"  Z={k:3d} ({sym}) → idx {v}")
    print()
    print("First 5 neighbor Z→idx mappings (sanity):")
    for k, v in list(neighbor_Z_to_idx.items())[:5]:
        try:
            sym = Element.from_Z(k).symbol
        except Exception:
            sym = '?'
        print(f"  Z={k:3d} ({sym}) → idx {v}")
    print()
    print(f"no_object_idx = {N_NEIGHBOR_TYPES}")
    print()
    print("Step 1.0 DONE.")


if __name__ == "__main__":
    main()
