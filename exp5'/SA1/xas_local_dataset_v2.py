"""
XasLocalDatasetV2: Exp4 Step 3 Dataset for diffusion training.
                   Exp5 SA1 patch: adds `center_element_Z` field for center-element
                   conditioning embedding lookup (EXP5_STEP1_HANDOFF §2.2 row 1).
                   Exp5' STEP1 patch: injects 5 shell_boundaries fields per sample
                   for new physical loss training (true_shell{1,2}_d_mean,
                   has_shell2, true_shell{1,2}_n). See EXP5_PRIME_PROPOSAL.md §2.2
                   and EXP5_PRIME_STEP1_HANDOFF.md §2.

Returns local representation: center atom + N_NEIGHBORS=20 nearest neighbors,
embedded in a virtual lattice of edge L_VIRTUAL=6.0 Å (HANDOFF §4.4 immutable).

Per-sample dict schema (HANDOFF §6.4 + Exp5 §2.2 row 1 + Exp5' §2.2):
    xmu                      Tensor (150,)  float32   Step 2 preprocessed μ(E)
    chi1                     Tensor (200,)  float32   Step 2 preprocessed k·χ(k)
    feff                     Tensor (74,)   float32   scaler.transform applied
    frac_coords              Tensor (20, 3) float32   ∈ [-0.5, 0.5]
    atom_types               LongTensor (20,)         Z ∈ [1, 109], no padding
    sample_name              str
    mp_id                    str
    center_element           str
    center_element_Z         int                      Exp5 SA1: Z of center element,
                                                      ∈ [2, 94] empirically (88 elems);
                                                      collated → (B,) LongTensor for
                                                      nn.Embedding(95, 16) lookup
    eval_cutoff              float                    Step 5 audit (scalar)
    eval_cutoff_fallback     bool                     Step 5 audit
    n_center_sites           int                      Step 5 audit
    site_equivalence_tag     str                      Step 5 grouping
    # ---- Exp5' STEP1 inject (5 fields, per-sample 0-d tensors) ----
    true_shell1_d_mean       Tensor (scalar) float32  shell-1 mean radial distance
    true_shell2_d_mean       Tensor (scalar) float32  shell-2 mean radial distance (0 if absent)
    has_shell2               Tensor (scalar) bool     True if shell-2 exists
    true_shell1_n            Tensor (scalar) long     shell-1 atom count
    true_shell2_n            Tensor (scalar) long     shell-2 atom count (0 if absent)

MA3 decisions enforced:
  7.3 D — local catch_warnings on scaler unpickle (sklearn 1.6.1→1.7.2 ABI ok)
  7.5 A — df.loc lookup; benchmark 1000 samples in __init__
  7.6 A — POSCAR=41,496 truth; v2 mp_ids ⊂ on-disk verified in Phase 0.5
  Phase 0.1 — pymatgen SGA(symprec=0.1) + first-matching-site bit-exact with
              shell_boundaries.pkl; no brute-force fallback.

Defensive raise points (fail-fast, no silent skip):
  init A    — 4-source lookup sanity on first 5 samples
  init A'   — Exp5' STEP1: 100-sample shell_boundaries hit_rate ≥ 95/100 sanity
  __getitem__:
    R1 — center_element not in primitive cell (next(...) → StopIteration)
    R2 — fewer than N_NEIGHBORS=20 neighbors within CUTOFF_R Å (§6.5)
    R3 — frac_coords sentinel after min-image wrap (§2.C)
"""
from __future__ import annotations

import os
import time
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ============================================================================
# Immutable constants (HANDOFF §4.4 / §6.2; Phase 0.1 lock)
# ============================================================================
L_VIRTUAL    = 6.0   # Å, virtual lattice edge
N_NEIGHBORS  = 20
CUTOFF_R     = 10.0  # Å, pymatgen get_neighbors radius
SYMPREC      = 0.1   # SpacegroupAnalyzer
FEFF_DIM     = 74    # post-Step-2.5 (was 73 in early draft)
XMU_DIM      = 150
CHI1_DIM     = 200


# ============================================================================
# Exp5' STEP1 inject: shell_boundaries 5-field extraction
# (EXP5_PRIME_PROPOSAL.md §2.2 / EXP5_PRIME_STEP1_HANDOFF.md §2.2)
# ============================================================================
def _extract_shell_fields(shell_meta: dict) -> dict:
    """
    Extract 5 fields from shell_boundaries.pkl per-sample dict for training loss.

    Source schema (verified Exp5'-MA startup §8):
        shell_n_atoms  (5,) int32         — atom counts per shell, idx 0=shell-1
        shell_of_atom  (n,) int32         — shell index per neighbor (0=shell-1, 1=shell-2)
        distances      (n,) float32       — center→neighbor distance (full cutoff ~9.984 Å)

    Output dict (per-sample 0-d scalar tensors; collate → (1,) at PyG wrapper → (B,) at batch):
        true_shell1_d_mean : float32
        true_shell2_d_mean : float32  (0.0 if no shell-2)
        has_shell2         : bool
        true_shell1_n      : long
        true_shell2_n      : long      (0 if no shell-2)

    Note (Exp5'-MA watch-only, launch note §11 #2): truth uses ALL neighbors within
    cutoff (~200+ atoms), while training prediction uses N=20 truncated frac_coords.
    Known design inconsistency, accepted by proposal §2.2; SA does NOT modify formula.
    """
    shell_n       = shell_meta["shell_n_atoms"]      # (5,) int32
    shell_of_atom = shell_meta["shell_of_atom"]      # (n,) int32
    distances     = shell_meta["distances"]          # (n,) float32

    # shell-1: shell_of_atom == 0
    s1_mask = (shell_of_atom == 0)
    s1_d_arr = distances[s1_mask]
    true_s1_d_mean = float(s1_d_arr.mean()) if len(s1_d_arr) > 0 else 0.0
    true_s1_n      = int(shell_n[0]) if len(shell_n) > 0 else 0

    # shell-2: shell_of_atom == 1 (may not exist)
    if len(shell_n) > 1 and int(shell_n[1]) > 0:
        s2_mask = (shell_of_atom == 1)
        s2_d_arr = distances[s2_mask]
        true_s2_d_mean = float(s2_d_arr.mean())
        true_s2_n      = int(shell_n[1])
        has_s2         = True
    else:
        true_s2_d_mean = 0.0
        true_s2_n      = 0
        has_s2         = False

    return {
        "true_shell1_d_mean": torch.tensor(true_s1_d_mean, dtype=torch.float32),
        "true_shell2_d_mean": torch.tensor(true_s2_d_mean, dtype=torch.float32),
        "has_shell2":         torch.tensor(has_s2,         dtype=torch.bool),
        "true_shell1_n":      torch.tensor(true_s1_n,      dtype=torch.long),
        "true_shell2_n":      torch.tensor(true_s2_n,      dtype=torch.long),
    }


class XasLocalDatasetV2(Dataset):
    """
    Args:
        split:    one of {"train", "val", "test", "holdout"}
        data_dir: e.g. "/home/tcat/diffcsp_exp4/data"
        verbose_init_benchmark: print df.loc + POSCAR+SGA timing in __init__
                                (default True; turn off in production training
                                if Phase 4 init is on critical path of multi-GPU spawn)
    """

    def __init__(
        self,
        split: str,
        data_dir: str | os.PathLike,
        verbose_init_benchmark: bool = True,
        use_cache: bool | None = None,
    ):
        if split not in ("train", "val", "test", "holdout"):
            raise ValueError(f"split must be one of train/val/test/holdout, got {split!r}")

        self.split = split
        self.data_dir = Path(data_dir)
        self.poscar_dir = self.data_dir / "MP_all_POSCAR_flat"

        # ---- v2 split CSV ----
        self.samples = pd.read_csv(self.data_dir / f"{split}_samples_v2.csv")
        # 7.5A: keep DataFrame for now, lookup via .loc (sample_name unique within split)
        self.samples_indexed = self.samples.set_index("sample_name")

        # ---- spectra (xmu/chi1) ----
        with open(self.data_dir / f"spectra_{split}.pkl", "rb") as f:
            spec = pickle.load(f)
        # spec keys: sample_names / xmu / chi1 / name_to_idx / E0 / meta
        self.spec_name_to_idx: dict[str, int] = spec["name_to_idx"]
        self.spec_xmu  = np.asarray(spec["xmu"],  dtype=np.float32)   # (N, 150)
        self.spec_chi1 = np.asarray(spec["chi1"], dtype=np.float32)   # (N, 200)
        if self.spec_xmu.shape[1] != XMU_DIM:
            raise RuntimeError(
                f"spectra_{split}.pkl xmu dim={self.spec_xmu.shape[1]}, "
                f"expected {XMU_DIM} (HANDOFF §1.4 immutable)"
            )
        if self.spec_chi1.shape[1] != CHI1_DIM:
            raise RuntimeError(
                f"spectra_{split}.pkl chi1 dim={self.spec_chi1.shape[1]}, "
                f"expected {CHI1_DIM} (HANDOFF §1.4 immutable)"
            )

        # ---- feff features (raw) + scaler ----
        self.feff_raw = pd.read_pickle(self.data_dir / "feff_features_imputed.pkl")
        if self.feff_raw.shape[1] != FEFF_DIM:
            raise RuntimeError(
                f"feff_features_imputed.pkl dim={self.feff_raw.shape[1]}, "
                f"expected {FEFF_DIM} (post-Step-2.5)"
            )
        # 7.3 D: local catch_warnings on scaler unpickle.
        # sklearn 1.6.1 → 1.7.2: RobustScaler internal state is just center_/scale_
        # numpy arrays. Cross-version ABI unchanged. Phase 0.3 already validated
        # transform sanity (no NaN/Inf, shape aligned). Suppress unpickle warning
        # locally to keep training log clean; do NOT global-filter.
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            self.scaler = joblib.load(self.data_dir / "feff_feature_scaler.pkl")

        # ---- shell_boundaries (eval_cutoff scalars + Exp5' STEP1 5-field training inject) ----
        with open(self.data_dir / "shell_boundaries.pkl", "rb") as f:
            self.shells = pickle.load(f)

        # ---- pymatgen lazy import (fail-fast at init, not per-getitem) ----
        from pymatgen.core import Structure, Element
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        self._Structure = Structure
        self._SGA = SpacegroupAnalyzer

        # ---- Exp5 SA1: center_element symbol → Z lookup table -----------
        # Built once from the split's center_element column. Used in __getitem__
        # to populate `center_element_Z` field for both fast (cache) and slow
        # (live POSCAR) paths — avoids per-call pymatgen Element construction.
        # Empirically max(Z) = 94 (Pu) across full 88-element pool; sentinel 0
        # is reserved for padding. SpectrumEncoder needs nn.Embedding(95, 16).
        unique_elems = self.samples["center_element"].unique()
        self._symbol_to_Z: dict[str, int] = {
            sym: int(Element(sym).Z) for sym in unique_elems
        }
        if self._symbol_to_Z:
            zs = list(self._symbol_to_Z.values())
            print(
                f"[XasLocalDatasetV2] center_element_Z lookup built: "
                f"{len(unique_elems)} elements, Z ∈ [{min(zs)}, {max(zs)}]"
            )

        # ---- Init A defensive: first 5 samples, all 4 sources resolve ----
        feff_keys = set(self.feff_raw.index)
        shell_keys = set(self.shells.keys())
        spec_keys = set(self.spec_name_to_idx.keys())
        for sn in self.samples["sample_name"].head(5):
            row = self.samples_indexed.loc[sn]
            if sn not in spec_keys:
                raise RuntimeError(f"Init: sample_name {sn} missing in spectra_{split}.pkl name_to_idx")
            if sn not in feff_keys:
                raise RuntimeError(f"Init: sample_name {sn} missing in feff_features_imputed.pkl index")
            if sn not in shell_keys:
                raise RuntimeError(f"Init: sample_name {sn} missing in shell_boundaries.pkl")
            mp_id = row["mp_id"]
            poscar_path = self.poscar_dir / f"{mp_id}_POSCAR"
            if not poscar_path.exists():
                raise RuntimeError(f"Init: POSCAR not found for {mp_id} at {poscar_path}")

        # ---- Init A' (Exp5' STEP1 strict, launch note §0.4 #4) ----
        # 100-sample shell_boundaries hit_rate ≥ 95/100 sanity.
        # If schema mismatch, raise immediately with miss/key examples for triage.
        # SA does NOT modify sample_name generation logic to align with pkl
        # (launch note §10 red line); raise → log → MA review.
        n_check = min(100, len(self.samples))
        check_names = self.samples["sample_name"].head(n_check).tolist()
        hits = sum(1 for sn in check_names if sn in shell_keys)
        if hits < 95:
            misses = [sn for sn in check_names if sn not in shell_keys][:5]
            pkl_sample = list(shell_keys)[:5]
            raise RuntimeError(
                f"[Exp5' STEP1 inject] sample_name schema mismatch: {hits}/{n_check} hits.\n"
                f"  dataset misses (first 5): {misses}\n"
                f"  pkl keys      (first 5): {pkl_sample}\n"
                f"  expected schema: 'mp-XXXXX__mp-XXXXX-EXAFS-{{element}}-K'"
            )
        print(f"[XasLocalDatasetV2 Exp5'] shell_boundaries sanity OK: {hits}/{n_check} hits")

        # ---- Benchmarks (7.5A timing + Q3 POSCAR+SGA timing) ----
        if verbose_init_benchmark:
            self._run_benchmarks()

        # ---- Optional structure cache (Step 4 acceleration) ----
        # use_cache resolution: explicit arg > env var EXP4_USE_CACHE > default True
        if use_cache is None:
            use_cache = os.environ.get("EXP4_USE_CACHE", "1") not in ("0", "false", "False", "")
        self._cache_frac   = None
        self._cache_atype  = None
        self._cache_feff   = None
        self._cache_valid  = None
        self.cache_enabled = False
        if use_cache:
            cache_path = self.data_dir / f"{split}_structure_cache.pt"
            if cache_path.exists():
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                # sanity: split / sample order alignment
                if blob.get("split") != split:
                    raise RuntimeError(
                        f"Cache split mismatch: file has {blob.get('split')!r}, expected {split!r}"
                    )
                cache_order = blob["sample_order"]
                live_order  = self.samples["sample_name"].astype(str).tolist()
                if cache_order != live_order:
                    raise RuntimeError(
                        f"Cache sample_order mismatch for split={split}. "
                        f"Cache was built against a different {split}_samples_v2.csv. "
                        f"Re-run precompute_structure_cache.py."
                    )
                self._cache_frac  = blob["frac_coords"]   # (N, 20, 3) float32
                self._cache_atype = blob["atom_types"]    # (N, 20)    int64
                self._cache_feff  = blob["feff_scaled"]   # (N, 74)    float32
                self._cache_valid = blob["valid_mask"]    # (N,)       bool
                self.cache_enabled = True
                n_valid = int(self._cache_valid.sum())
                print(
                    f"[XasLocalDatasetV2] cache LOADED for {split}: "
                    f"valid={n_valid}/{len(live_order)} from {cache_path.name}"
                )
            else:
                print(
                    f"[XasLocalDatasetV2] cache NOT FOUND at {cache_path}; "
                    f"falling back to live POSCAR+SGA path."
                )
        else:
            print(f"[XasLocalDatasetV2] cache DISABLED via use_cache=False (split={split}).")

        print(f"[XasLocalDatasetV2] split={split} samples={len(self.samples)} ready.")

    def _run_benchmarks(self):
        rng = np.random.default_rng(42)
        # df.loc on feff (74-dim row, sample_name index)
        n_loc = min(1000, len(self.samples))
        names = rng.choice(self.samples["sample_name"].values, size=n_loc, replace=False)
        t0 = time.perf_counter()
        for sn in names:
            _ = self.feff_raw.loc[sn].values
        elapsed_loc = (time.perf_counter() - t0) / n_loc
        print(
            f"[XasLocalDatasetV2 benchmark] feff.loc avg: {elapsed_loc*1e6:.2f} µs/sample "
            f"(N={n_loc}); cutover threshold: 200 µs → consider dict cache (decision 7.5B)"
        )

        # POSCAR + SGA primitive
        n_pos = min(50, len(self.samples))
        sub = self.samples.sample(n=n_pos, random_state=0)
        t0 = time.perf_counter()
        for _, row in sub.iterrows():
            s = self._Structure.from_file(self.poscar_dir / f"{row['mp_id']}_POSCAR")
            _ = self._SGA(s, symprec=SYMPREC).get_primitive_standard_structure()
        elapsed_pos = (time.perf_counter() - t0) / n_pos
        print(
            f"[XasLocalDatasetV2 benchmark] POSCAR + SGA avg: {elapsed_pos*1e3:.2f} ms/sample "
            f"(N={n_pos}); >50 ms → consider lru_cache after Step 4 profile"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]
        sname: str = row["sample_name"]
        mp_id: str = row["mp_id"]
        center_elem: str = row["center_element"]

        # ============================================================
        # FAST PATH: structure cache (Step 4 acceleration)
        # ============================================================
        if self.cache_enabled:
            if not bool(self._cache_valid[idx].item()):
                # original __getitem__ would have returned None here too
                return None

            # spectra: same lookup as slow path (xmu/chi1 are already in RAM)
            spec_idx = self.spec_name_to_idx[sname]
            xmu  = torch.from_numpy(self.spec_xmu[spec_idx].copy())
            chi1 = torch.from_numpy(self.spec_chi1[spec_idx].copy())

            # cached tensors: clone() to make per-sample tensor independent
            # (avoid downstream in-place ops affecting the shared cache buffer)
            frac_coords_t = self._cache_frac[idx].clone()    # (20, 3) float32
            atom_types    = self._cache_atype[idx].clone()   # (20,)   int64
            feff          = self._cache_feff[idx].clone()    # (74,)   float32

            shell_meta = self.shells[sname]
            return {
                "xmu": xmu,
                "chi1": chi1,
                "feff": feff,
                "frac_coords": frac_coords_t,
                "atom_types": atom_types,
                "sample_name": sname,
                "mp_id": mp_id,
                "center_element": center_elem,
                "center_element_Z": self._symbol_to_Z[center_elem],   # Exp5 SA1
                "eval_cutoff": float(shell_meta["eval_cutoff"]),
                "eval_cutoff_fallback": bool(shell_meta.get("eval_cutoff_fallback", False)),
                "n_center_sites": int(shell_meta["n_center_sites"]),
                "site_equivalence_tag": row.get("site_equivalence_tag", "unknown"),
                # ---- Exp5' STEP1 inject: 5 shell training fields ----
                **_extract_shell_fields(shell_meta),
            }
        # ============================================================
        # SLOW PATH (original): live POSCAR + SGA + neighbors
        # ============================================================

        # ---- spectra (xmu, chi1) ----
        spec_idx = self.spec_name_to_idx[sname]
        xmu  = torch.from_numpy(self.spec_xmu[spec_idx].copy())   # (150,)
        chi1 = torch.from_numpy(self.spec_chi1[spec_idx].copy())  # (200,)

        # ---- feff (raw → scaler.transform) ----
        feff_raw = self.feff_raw.loc[sname].values.astype(np.float32).reshape(1, -1)  # (1, 74)
        feff_scaled = self.scaler.transform(feff_raw).astype(np.float32).reshape(-1)  # (74,)
        feff = torch.from_numpy(feff_scaled)

        # ---- structure: load + primitive ----
        struct = self._Structure.from_file(self.poscar_dir / f"{mp_id}_POSCAR")
        prim = self._SGA(struct, symprec=SYMPREC).get_primitive_standard_structure()

        # R1: center_idx
        try:
            center_idx = next(i for i, site in enumerate(prim) if site.specie.symbol == center_elem)
        except StopIteration:
            prim_species = [s.specie.symbol for s in prim]
            raise RuntimeError(
                f"Sample {sname} (mp_id={mp_id}): center_element={center_elem} "
                f"not found in primitive sites {prim_species}. "
                f"Possible SGA primitive transform changed atom species (partial occupancy?). "
                f"STOP and report MA3."
            )
        center_cart = np.asarray(prim[center_idx].coords, dtype=np.float64)  # (3,) Å

        # ---- neighbors ----
        nbrs = prim.get_neighbors(prim[center_idx], r=CUTOFF_R)
        if len(nbrs) < N_NEIGHBORS:
            # Phase 4.6 (MA5): Exp2 silent-drop behavior; collate filters None
            return None
        # R2 already checked. Now explicit argsort by distance (don't rely on pymatgen order)
        dists = np.array([n.nn_distance for n in nbrs], dtype=np.float64)
        coords_all = np.array([n.coords for n in nbrs], dtype=np.float64)         # (N, 3) Å
        Z_all = np.array([n.specie.Z for n in nbrs], dtype=np.int64)              # (N,)
        order = np.argsort(dists)[:N_NEIGHBORS]
        coords_top = coords_all[order]                                            # (20, 3)
        Z_top = Z_all[order]                                                      # (20,)

        # ---- frac_coords in virtual lattice ----
        relative_cart = coords_top - center_cart[None, :]                         # (20, 3) Å
        frac = relative_cart / L_VIRTUAL                                          # (20, 3)
        frac = frac - np.round(frac)                                              # min-image wrap → [-0.5, 0.5]

        # R3: frac sentinel (§2.C); Phase 4.6 (MA5): Exp2 silent-drop, collate filters
        violations = np.abs(frac) > 0.5 + 1e-6
        if violations.any():
            return None

        frac_coords_t = torch.from_numpy(frac.astype(np.float32))                 # (20, 3)
        atom_types    = torch.from_numpy(Z_top)                                   # (20,) int64

        # ---- shell metadata (scalars + Exp5' STEP1 5-field training inject) ----
        shell_meta = self.shells[sname]
        eval_cutoff           = float(shell_meta["eval_cutoff"])
        eval_cutoff_fallback  = bool(shell_meta.get("eval_cutoff_fallback", False))
        n_center_sites        = int(shell_meta["n_center_sites"])

        return {
            "xmu": xmu,
            "chi1": chi1,
            "feff": feff,
            "frac_coords": frac_coords_t,
            "atom_types": atom_types,
            "sample_name": sname,
            "mp_id": mp_id,
            "center_element": center_elem,
            "center_element_Z": self._symbol_to_Z[center_elem],   # Exp5 SA1
            "eval_cutoff": eval_cutoff,
            "eval_cutoff_fallback": eval_cutoff_fallback,
            "n_center_sites": n_center_sites,
            "site_equivalence_tag": row.get("site_equivalence_tag", "unknown"),
            # ---- Exp5' STEP1 inject: 5 shell training fields ----
            **_extract_shell_fields(shell_meta),
        }
