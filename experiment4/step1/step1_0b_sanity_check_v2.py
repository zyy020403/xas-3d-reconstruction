# step1_0b_sanity_check_v2.py
# ------------------------------------------------------------
# Exp4 Step 1 pre-flight v2. Run after v1 revealed:
#   - chi files have "_chi.csv" suffix
#   - POSCAR files are "{mp_id}_POSCAR" (no extension)
#   - missing_poscar_list contains full paths, not mp_ids
#   - v1 accidentally read log files as "first" file due to sort order
#
# v2 anchors every probe on FEFF_CSV's sample_name (canonical list)
# to guarantee we never read a log file again.
# ------------------------------------------------------------

import os, re
import pandas as pd

EXP4_DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
CHI_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv")
XMU_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv")
POSCAR_DIR     = os.path.join(EXP4_DATA_ROOT, r"POSCAR_zip\MP_all_POSCAR_flat")
FEFF_CSV       = os.path.join(EXP4_DATA_ROOT, "feff_features_all_csv_75cols(in).csv")
MISSING_POSCAR_CSV = os.path.join(POSCAR_DIR, "missing_poscar_list.csv")

NAME_RX = re.compile(r"^(mp-\d+)__\1-EXAFS-([A-Z][a-z]?)-K$")

def hdr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# --------------------------------------------------------------
# [1] Pick 1 real sample via FEFF_CSV and resolve all 3 file paths
# --------------------------------------------------------------
hdr("[1] Pick real sample from FEFF_CSV row 0; infer & verify paths")
s_name = pd.read_csv(FEFF_CSV, usecols=["sample_name"], nrows=1).iloc[0, 0]
m = NAME_RX.match(str(s_name))
mp_id, elem = m.group(1), m.group(2)
chi_path = os.path.join(CHI_DIR,    s_name + "_chi.csv")
xmu_path = os.path.join(XMU_DIR,    s_name + ".csv")
pos_path = os.path.join(POSCAR_DIR, mp_id  + "_POSCAR")
print(f"  sample_name   = {s_name}")
print(f"  mp_id         = {mp_id}")
print(f"  center_element= {elem}")
print(f"  chi_path      exists? {os.path.isfile(chi_path)}  -> {chi_path}")
print(f"  xmu_path      exists? {os.path.isfile(xmu_path)}  -> {xmu_path}")
print(f"  poscar_path   exists? {os.path.isfile(pos_path)}  -> {pos_path}")


# --------------------------------------------------------------
# [2] chi.csv: raw head + shape + columns + std(chi1)
# --------------------------------------------------------------
hdr("[2] Real chi.csv content")
try:
    with open(chi_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 5: break
            print("  " + line.rstrip())
    with open(chi_path, "r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    print(f"  total rows (incl header): {n}")
    df = pd.read_csv(chi_path, encoding="utf-8")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")
    for col in ("chi", "chi1", "chi2"):
        if col in df.columns:
            print(f"  std({col}) = {df[col].std():.6g}")
        else:
            print(f"  (no column '{col}')")
    print("  first 3 data rows:")
    print(df.head(3).to_string())
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")


# --------------------------------------------------------------
# [3] xmu.csv: raw head + shape + columns + std(y)
# --------------------------------------------------------------
hdr("[3] Real xmu.csv content")
try:
    with open(xmu_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 5: break
            print("  " + line.rstrip())
    with open(xmu_path, "r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    print(f"  total rows (incl header): {n}")
    df = pd.read_csv(xmu_path, encoding="utf-8")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")
    for col in ("x", "y"):
        if col in df.columns:
            print(f"  std({col}) = {df[col].std():.6g}")
        else:
            print(f"  (no column '{col}')")
    print("  first 3 data rows:")
    print(df.head(3).to_string())
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")


# --------------------------------------------------------------
# [4] POSCAR: raw head + pymatgen parse (2 fallbacks) + primitive
# --------------------------------------------------------------
hdr("[4] Real POSCAR content + pymatgen parse")
try:
    with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 12: break
            print("  " + line.rstrip())
except Exception as e:
    print(f"  [read ERROR] {type(e).__name__}: {e}")

# Try pymatgen both ways; crucial because filename has no extension
try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except Exception as e:
    print(f"  [pymatgen import ERROR] {e}")
else:
    # Attempt 1: Structure.from_file (may fail on unknown extension)
    s = None
    try:
        s = Structure.from_file(pos_path)
        print(f"  Structure.from_file: OK  n_sites={len(s)}")
    except Exception as e:
        print(f"  Structure.from_file FAILED: {type(e).__name__}: {e}")
        # Attempt 2: Structure.from_str with fmt='poscar'
        try:
            with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
                s = Structure.from_str(f.read(), fmt="poscar")
            print(f"  Structure.from_str(fmt='poscar'): OK  n_sites={len(s)}")
        except Exception as e2:
            print(f"  Structure.from_str FAILED: {type(e2).__name__}: {e2}")

    if s is not None:
        try:
            prim = SpacegroupAnalyzer(s, symprec=0.1).get_primitive_standard_structure()
            print(f"  primitive n_sites: {len(prim)}")
            print(f"  primitive formula: {prim.composition.reduced_formula}")
            prim_elems = {str(site.specie.symbol) for site in prim}
            print(f"  primitive element set: {sorted(prim_elems)}")
            print(f"  center_element '{elem}' present in primitive? {elem in prim_elems}")
        except Exception as e:
            print(f"  SpacegroupAnalyzer FAILED: {type(e).__name__}: {e}")


# --------------------------------------------------------------
# [5] Cross-check directory counts with real-file filters
# --------------------------------------------------------------
hdr("[5] Directory counts after filtering log/manifest files")
chi_all = os.listdir(CHI_DIR)
xmu_all = os.listdir(XMU_DIR)
pos_all = os.listdir(POSCAR_DIR)

chi_real = [f for f in chi_all if f.startswith("mp-") and f.endswith("_chi.csv")]
chi_log  = [f for f in chi_all if f not in chi_real]
xmu_real = [f for f in xmu_all if f.startswith("mp-") and f.endswith(".csv")]
xmu_log  = [f for f in xmu_all if f not in xmu_real]
pos_real = [f for f in pos_all if f.startswith("mp-") and f.endswith("_POSCAR")]
pos_log  = [f for f in pos_all if f not in pos_real]

print(f"  CHI_DIR:    total {len(chi_all)}, real {len(chi_real)}, excl: {chi_log}")
print(f"  XMU_DIR:    total {len(xmu_all)}, real {len(xmu_real)}, excl: {xmu_log}")
print(f"  POSCAR_DIR: total {len(pos_all)}, real {len(pos_real)}, excl: {pos_log}")


# --------------------------------------------------------------
# [6] FEFF sample_name vs directory contents cross-check
# --------------------------------------------------------------
hdr("[6] Cross-check FEFF ↔ CHI/XMU/POSCAR")
feff = pd.read_csv(FEFF_CSV, usecols=["sample_name"], encoding="utf-8")
print(f"  FEFF rows: {len(feff)}")

# parse rate
parsed = feff["sample_name"].astype(str).apply(
    lambda n: NAME_RX.match(n) is not None)
print(f"  regex-parseable sample_names: {parsed.sum()} / {len(feff)}")

mp_in_feff   = set(feff["sample_name"].astype(str).str.extract(r"^(mp-\d+)", expand=False).dropna())
mp_in_poscar = {f[:-len("_POSCAR")] for f in pos_real}
print(f"  distinct mp_ids in FEFF   : {len(mp_in_feff)}")
print(f"  distinct mp_ids in POSCAR : {len(mp_in_poscar)}")
print(f"  intersection              : {len(mp_in_feff & mp_in_poscar)}")
print(f"  in FEFF but no POSCAR file: {len(mp_in_feff - mp_in_poscar)}")
print(f"  POSCAR file but not in FEFF: {len(mp_in_poscar - mp_in_feff)}")

# chi/xmu filename sets (strip suffix)
chi_names = {f[:-len("_chi.csv")] for f in chi_real}
xmu_names = {f[:-len(".csv")] for f in xmu_real}
feff_names_set = set(feff["sample_name"].astype(str))
print(f"  sample_names in FEFF ∩ CHI : "
      f"{len(feff_names_set & chi_names)} / FEFF {len(feff_names_set)}")
print(f"  sample_names in FEFF ∩ XMU : "
      f"{len(feff_names_set & xmu_names)} / FEFF {len(feff_names_set)}")
print(f"  FEFF lacking CHI file: {len(feff_names_set - chi_names)}")
print(f"  FEFF lacking XMU file: {len(feff_names_set - xmu_names)}")


# --------------------------------------------------------------
# [7] missing_poscar_list parsing
# --------------------------------------------------------------
hdr("[7] missing_poscar_list.csv parsing (extract mp_id from path)")
miss = pd.read_csv(MISSING_POSCAR_CSV, encoding="utf-8")
print(f"  shape: {miss.shape}, column: {miss.columns[0]!r}")
raw = miss.iloc[:, 0].astype(str)
# Split on both '\' and '/' for safety
missing_ids = raw.str.rsplit("\\", n=1).str[-1].str.rsplit("/", n=1).str[-1]
print(f"  first 5 extracted mp_ids: {list(missing_ids.head())}")
mp_pat = re.compile(r"^mp-\d+$")
bad = [v for v in missing_ids if not mp_pat.match(str(v))]
print(f"  entries not matching 'mp-\\d+': {len(bad)}")
if bad: print(f"    examples: {bad[:5]}")
missing_set = set(missing_ids)
print(f"  unique missing mp_ids: {len(missing_set)}")
print(f"  missing ∩ mp_in_feff  : {len(missing_set & mp_in_feff)}")
print(f"  missing ∩ mp_in_poscar: {len(missing_set & mp_in_poscar)}  "
      f"(expected 0: missing-list mp_ids should NOT have POSCAR files)")

# How many FEFF rows would be dropped by missing_poscar filter
affected_rows = 0
feff_mp = feff["sample_name"].astype(str).str.extract(r"^(mp-\d+)", expand=False)
affected_rows = int(feff_mp.isin(missing_set).sum())
print(f"  FEFF sample rows affected by missing POSCAR: {affected_rows}")


# --------------------------------------------------------------
# [8] Bulk JOIN probe: first 5 FEFF samples, check all 3 paths
# --------------------------------------------------------------
hdr("[8] Bulk JOIN probe (first 5 FEFF samples)")
for name in feff["sample_name"].astype(str).head(5):
    m = NAME_RX.match(name)
    if not m:
        print(f"  {name}: regex FAIL"); continue
    mp, el = m.group(1), m.group(2)
    chi_ok = os.path.isfile(os.path.join(CHI_DIR, name + "_chi.csv"))
    xmu_ok = os.path.isfile(os.path.join(XMU_DIR, name + ".csv"))
    pos_ok = os.path.isfile(os.path.join(POSCAR_DIR, mp + "_POSCAR"))
    print(f"  {name:46s} chi={chi_ok} xmu={xmu_ok} poscar={pos_ok}")

print("\n" + "=" * 72)
print("v2 sanity check done. Paste full output back to Sub-Agent.")
print("=" * 72)
