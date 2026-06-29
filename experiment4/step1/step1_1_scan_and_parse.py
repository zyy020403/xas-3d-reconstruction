# step1_1_scan_and_parse.py
# ------------------------------------------------------------
# Exp4 Step 1.1
#   - Load FEFF_CSV (133,718 rows)
#   - Parse sample_name via regex -> mp_id, center_element
#   - Construct chi/xmu/POSCAR absolute paths
#   - Validate each:
#       chi_valid : pd.read_csv ok + rows>=300 + std(chi1) > 1e-6
#       xmu_valid : pd.read_csv ok + rows>=300 + std(y)    > 1e-6
#       poscar_valid : Structure.from_file + SpacegroupAnalyzer + prim.n >=1
#   - POSCAR validation is cached per unique mp_id (POSCAR is shared
#     by all samples of the same mp_id)
#   - Save: step1_1_raw_inventory.pkl (all samples + validity flags + metadata)
# ------------------------------------------------------------

import os
import re
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)  # pymatgen verbosity

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# =========================================================
EXP4_DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
CHI_DIR    = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv")
XMU_DIR    = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv")
POSCAR_DIR = os.path.join(EXP4_DATA_ROOT, r"POSCAR_zip\MP_all_POSCAR_flat")
FEFF_CSV   = os.path.join(EXP4_DATA_ROOT, "feff_features_all_csv_75cols(in).csv")

EXP4_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR = os.path.join(EXP4_ROOT, "step1")
os.makedirs(STEP1_DIR, exist_ok=True)

NAME_RX = re.compile(r"^(mp-\d+)__\1-EXAFS-([A-Z][a-z]?)-K$")
META_COLS = ["sample_dir", "sample_name", "feature_version"]


# =========================================================
def validate_chi(path):
    if not os.path.isfile(path):
        return False, "file_missing"
    try:
        df = pd.read_csv(path, usecols=["chi1"], encoding="utf-8")
    except ValueError:
        return False, "no_chi1_col"
    except Exception as e:
        return False, f"{type(e).__name__}"
    if len(df) < 300:
        return False, f"rows<300"
    if df["chi1"].std() <= 1e-6:
        return False, "chi1_flat"
    return True, None


def validate_xmu(path):
    if not os.path.isfile(path):
        return False, "file_missing"
    try:
        df = pd.read_csv(path, usecols=["y"], encoding="utf-8")
    except ValueError:
        return False, "no_y_col"
    except Exception as e:
        return False, f"{type(e).__name__}"
    if len(df) < 300:
        return False, f"rows<300"
    if df["y"].std() <= 1e-6:
        return False, "y_flat"
    return True, None


def validate_poscar(path):
    """Returns (valid: bool, n_sites_primitive: int, reason: str|None)."""
    if not os.path.isfile(path):
        return False, -1, "file_missing"
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        s = Structure.from_file(path)
        prim = SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()
        n = len(prim)
        if n < 1:
            return False, n, "prim_empty"
        return True, n, None
    except Exception as e:
        return False, -1, f"{type(e).__name__}"


# =========================================================
def main():
    t0 = time.time()

    # ---------- Load FEFF_CSV ----------
    print(f"[1/5] Load FEFF_CSV\n  {FEFF_CSV}")
    feff = pd.read_csv(FEFF_CSV, encoding="utf-8")
    print(f"  shape: {feff.shape}")
    assert "sample_name" in feff.columns
    dups = feff["sample_name"].duplicated().sum()
    if dups:
        print(f"  WARN: {dups} duplicate sample_names, dropping (keep=first)")
        feff = feff.drop_duplicates(subset="sample_name", keep="first").reset_index(drop=True)

    # ---------- Parse sample_name ----------
    print(f"\n[2/5] Parse sample_name via regex")
    matches = feff["sample_name"].astype(str).apply(lambda x: NAME_RX.match(x))
    feff["_ok"] = matches.apply(lambda m: m is not None)
    feff["mp_id"] = matches.apply(lambda m: m.group(1) if m else None)
    feff["center_element"] = matches.apply(lambda m: m.group(2) if m else None)
    n_fail = int((~feff["_ok"]).sum())
    print(f"  parse_fail: {n_fail}")
    if n_fail:
        pf = feff.loc[~feff["_ok"], ["sample_name"]].copy()
        pf["mp_id"] = None
        pf["center_element"] = None
        pf["reason"] = "parse_fail"
        pf.to_csv(os.path.join(STEP1_DIR, "step1_1_parse_fail.csv"), index=False)
    feff = feff[feff["_ok"]].drop(columns=["_ok"]).reset_index(drop=True)
    print(f"  rows after parse: {len(feff)}")

    # ---------- Build paths ----------
    print(f"\n[3/5] Build chi/xmu/POSCAR paths")
    feff["chi_path"]    = feff["sample_name"].map(lambda n: os.path.join(CHI_DIR,    n + "_chi.csv"))
    feff["xmu_path"]    = feff["sample_name"].map(lambda n: os.path.join(XMU_DIR,    n + ".csv"))
    feff["poscar_path"] = feff["mp_id"     ].map(lambda m: os.path.join(POSCAR_DIR, m + "_POSCAR"))

    # ---------- POSCAR validate (per unique mp_id) ----------
    print(f"\n[4/5] Validate POSCAR (pymatgen, per unique mp_id)")
    unique_mps = sorted(feff["mp_id"].unique())
    print(f"  unique mp_ids: {len(unique_mps)}")
    poscar_cache = {}
    for mp in tqdm(unique_mps, desc="POSCAR"):
        poscar_cache[mp] = validate_poscar(os.path.join(POSCAR_DIR, mp + "_POSCAR"))
    feff["poscar_valid"]  = feff["mp_id"].map(lambda m: poscar_cache[m][0])
    feff["prim_n_atoms"]  = feff["mp_id"].map(lambda m: poscar_cache[m][1])
    feff["poscar_reason"] = feff["mp_id"].map(lambda m: poscar_cache[m][2])
    print(f"  POSCAR invalid samples: {(~feff['poscar_valid']).sum()}")
    print(f"  POSCAR file_missing samples: "
          f"{(feff['poscar_reason'] == 'file_missing').sum()}")

    # ---------- chi / xmu validate (per sample) ----------
    print(f"\n[5/5] Validate chi.csv and xmu.csv (per sample)")
    chi_res = [validate_chi(p) for p in tqdm(feff["chi_path"].tolist(), desc="chi")]
    feff["chi_valid"]  = [r[0] for r in chi_res]
    feff["chi_reason"] = [r[1] for r in chi_res]
    print(f"  chi invalid: {(~feff['chi_valid']).sum()}")

    xmu_res = [validate_xmu(p) for p in tqdm(feff["xmu_path"].tolist(), desc="xmu")]
    feff["xmu_valid"]  = [r[0] for r in xmu_res]
    feff["xmu_reason"] = [r[1] for r in xmu_res]
    print(f"  xmu invalid: {(~feff['xmu_valid']).sum()}")

    # ---------- Save ----------
    out = os.path.join(STEP1_DIR, "step1_1_raw_inventory.pkl")
    feff.to_pickle(out)
    print(f"\n[Save] {out}  shape={feff.shape}")

    print(f"\n----- Step 1.1 Summary -----")
    print(f"  FEFF raw rows:           {len(feff) + n_fail}")
    print(f"  parse_fail dropped:      {n_fail}")
    print(f"  rows carried to 1.2:     {len(feff)}")
    print(f"  unique mp_ids:           {len(unique_mps)}")
    print(f"  invalid POSCAR samples:  {(~feff['poscar_valid']).sum()}")
    print(f"  invalid chi samples:     {(~feff['chi_valid']).sum()}")
    print(f"  invalid xmu samples:     {(~feff['xmu_valid']).sum()}")
    print(f"  elapsed:                 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
