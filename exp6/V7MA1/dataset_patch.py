"""
Patch experiment6_v7/shared/xas_local_dataset_v2.py:
1. Add shell_boundaries.pkl loading in __init__
2. Add 5 shell fields to __getitem__ return dict
3. Add shell_boundaries sanity check (hit_rate >= 95/100)

Run from /home/tcat/experiment6_v7:
  /home/tcat/conda_envs/mlff/bin/python /home/tcat/dataset_patch.py
"""
from pathlib import Path

SRC = Path("/home/tcat/experiment6_v7/shared/xas_local_dataset_v2.py")
text = SRC.read_text()

# ── Patch 1: add pickle import if missing ─────────────────────────────────
if "import pickle" not in text:
    text = text.replace("import torch", "import pickle\nimport torch", 1)

# ── Patch 2: load shell_boundaries.pkl in __init__ ────────────────────────
# Anchor: just before the line that creates self._symbol_to_Z
# (confirmed present in Exp6 v7 shared copy — same base as Exp5')
# We insert right after feff/scaler loading block.
# Find the feff scaler load line as anchor.
OLD_SCALER = '        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)\n            self.scaler = joblib.load(self.data_dir / "feff_feature_scaler.pkl")'

# Fallback anchor: find the pymatgen lazy import block
OLD_PYMATGEN = "        # ---- pymatgen lazy import (fail-fast at init, not per-getitem) ----"
assert OLD_PYMATGEN in text, "PATCH 2 anchor (pymatgen lazy import) not found"

SHELL_INIT = '''        # ---- shell_boundaries.pkl — Exp6 v7 STEP1 inject ----
        _sb_path = self.data_dir / "shell_boundaries.pkl"
        with open(_sb_path, "rb") as _f:
            self.shells = pickle.load(_f)
        # Sanity: 100-sample hit_rate >= 95/100
        _n_check = min(100, len(self.samples))
        _check_names = self.samples["sample_name"].head(_n_check).tolist()
        _hits = sum(1 for _sn in _check_names if _sn in self.shells)
        if _hits < 95:
            _misses = [_sn for _sn in _check_names if _sn not in self.shells][:5]
            raise RuntimeError(
                f"[Exp6 v7 dataset] shell_boundaries schema mismatch: "
                f"{_hits}/{_n_check} hits. First 5 misses: {_misses}"
            )
        print(f"[XasLocalDatasetV2 Exp6 v7] shell_boundaries OK: {_hits}/{_n_check} hits")

'''
text = text.replace(OLD_PYMATGEN, SHELL_INIT + OLD_PYMATGEN, 1)

# ── Patch 3: add 5 shell fields to __getitem__ return dict ────────────────
# Anchor: the final return dict in __getitem__ (slow path, POSCAR-based)
OLD_RETURN = '''        return {
            "xmu": xmu,
            "chi1": chi1,
            "feff": feff,
            "frac_coords": frac_coords_t,
            "atom_types": atom_types,
            "sample_name": sname,
            "mp_id": mp_id,
            "center_element": center_elem,
            "eval_cutoff": eval_cutoff,
            "eval_cutoff_fallback": eval_cutoff_fallback,
            "n_center_sites": n_center_sites,
            "site_equivalence_tag": row.get("site_equivalence_tag", "unknown"),
        }'''

NEW_RETURN = '''        # ---- shell fields for v7 three-piece loss ----
        _sm = self.shells.get(sname, {})
        _s_starts  = _sm.get("shell_starts",  [])   # np (S,) float32
        _s_ends    = _sm.get("shell_ends",    [])   # np (S,) float32
        _s_n_atoms = _sm.get("shell_n_atoms", [])   # np (S,) int32
        # Only pass first 2 shells (proposal §5.2.2/§5.2.3)
        _n_shells = min(2, len(_s_starts))

        return {
            "xmu": xmu,
            "chi1": chi1,
            "feff": feff,
            "frac_coords": frac_coords_t,
            "atom_types": atom_types,
            "sample_name": sname,
            "mp_id": mp_id,
            "center_element": center_elem,
            "eval_cutoff": eval_cutoff,
            "eval_cutoff_fallback": eval_cutoff_fallback,
            "n_center_sites": n_center_sites,
            "site_equivalence_tag": row.get("site_equivalence_tag", "unknown"),
            # v7 shell fields (float lists, len <= 2)
            "shell_starts":  [float(_s_starts[i])  for i in range(_n_shells)],
            "shell_ends":    [float(_s_ends[i])    for i in range(_n_shells)],
            "shell_n_atoms": [int(_s_n_atoms[i])   for i in range(_n_shells)],
        }'''

assert OLD_RETURN in text, "PATCH 3 anchor (final return dict) not found"
text = text.replace(OLD_RETURN, NEW_RETURN, 1)

SRC.write_text(text)
print(f"Patched {SRC}")

# Verify
t2 = SRC.read_text()
checks = ["self.shells = pickle.load", "shell_boundaries OK",
          "shell_starts", "shell_ends", "shell_n_atoms",
          "# v7 shell fields"]
for c in checks:
    print(f"  {'✓' if c in t2 else 'MISSING'} {c}")
print("Done.")
