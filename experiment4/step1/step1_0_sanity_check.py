# step1_0_sanity_check.py
# ------------------------------------------------------------
# Experiment 4 - Step 1 pre-flight sanity check.
# Run on USER'S Windows machine. Paste the ENTIRE console output
# back to the Sub-Agent.
#
# Purpose: verify directory layout, file naming, column format
# BEFORE writing the full step1_1..step1_6 pipeline.
# ------------------------------------------------------------

import os
import re
import sys
import pandas as pd

# =============================================================
# Paths - copied verbatim from STEP1_SUBAGENT_HANDOFF.md section 4
# =============================================================
EXP4_DATA_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data"
CHI_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_chi_csv\MP_all_EXAFS_only_chi_csv")
XMU_DIR        = os.path.join(EXP4_DATA_ROOT, r"MP_all_EXAFS_only_csv\MP_all_EXAFS_only_csv")
POSCAR_DIR     = os.path.join(EXP4_DATA_ROOT, r"POSCAR_zip\MP_all_POSCAR_flat")
FEFF_CSV       = os.path.join(EXP4_DATA_ROOT, "feff_features_all_csv_75cols(in).csv")
MISSING_POSCAR_CSV = os.path.join(POSCAR_DIR, "missing_poscar_list.csv")


def hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe(fn):
    """Run fn, catch and print exceptions, never abort the whole script."""
    try:
        fn()
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")


# -------------------------------------------------------------
# [1-3] listdir first 10 + total count for each directory
# -------------------------------------------------------------
chi_files = xmu_files = poscar_files = []

def _list(dir_path, label):
    global chi_files, xmu_files, poscar_files
    hdr(f"[{label}] {dir_path}")
    if not os.path.isdir(dir_path):
        print(f"  [ERROR] directory does not exist")
        return []
    files = sorted(os.listdir(dir_path))
    print(f"  total entries: {len(files)}")
    print(f"  first 10:")
    for f in files[:10]:
        print(f"    {f}")
    return files

safe(lambda: globals().update(chi_files=_list(CHI_DIR, "1 / CHI_DIR")))
safe(lambda: globals().update(xmu_files=_list(XMU_DIR, "2 / XMU_DIR")))
safe(lambda: globals().update(poscar_files=_list(POSCAR_DIR, "3 / POSCAR_DIR")))


# -------------------------------------------------------------
# [4] chi.csv: head 5 lines + total row count
# -------------------------------------------------------------
def _head_text(path, n_lines, label):
    hdr(label + f"   -> {os.path.basename(path)}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            print(f"    {line.rstrip()}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        total = sum(1 for _ in fh)
    print(f"  total rows (incl. header if any): {total}")

def _chk_chi():
    if not chi_files:
        print("  [SKIP] no chi files")
        return
    _head_text(os.path.join(CHI_DIR, chi_files[0]), 5, "[4] chi.csv first 5 lines")
safe(_chk_chi)

def _chk_xmu():
    if not xmu_files:
        print("  [SKIP] no xmu files")
        return
    _head_text(os.path.join(XMU_DIR, xmu_files[0]), 5, "[5] xmu.csv first 5 lines")
safe(_chk_xmu)

def _chk_poscar():
    if not poscar_files:
        print("  [SKIP] no POSCAR files")
        return
    _head_text(os.path.join(POSCAR_DIR, poscar_files[0]), 10, "[6] POSCAR first 10 lines")
safe(_chk_poscar)


# -------------------------------------------------------------
# [7] FEFF_CSV: shape + full column list + first 2 rows
# -------------------------------------------------------------
def _chk_feff():
    hdr(f"[7] FEFF_CSV -> {os.path.basename(FEFF_CSV)}")
    feff_head = pd.read_csv(FEFF_CSV, nrows=2, encoding="utf-8")
    print(f"  head shape: {feff_head.shape}")
    print(f"  column count: {len(feff_head.columns)}")
    print(f"  all columns with index:")
    for i, c in enumerate(feff_head.columns):
        print(f"    [{i:3d}] {c}")
    print(f"\n  first 2 rows (truncated per column for readability):")
    with pd.option_context("display.max_columns", None,
                           "display.width", 160,
                           "display.max_colwidth", 30):
        print(feff_head.to_string())
    # total row count by reading just column 0
    col0 = pd.read_csv(FEFF_CSV, usecols=[0], encoding="utf-8")
    print(f"\n  total data rows: {len(col0)}")
safe(_chk_feff)


# -------------------------------------------------------------
# [8] missing_poscar_list.csv: shape + columns + head
# -------------------------------------------------------------
def _chk_missing():
    hdr(f"[8] MISSING_POSCAR_CSV -> {os.path.basename(MISSING_POSCAR_CSV)}")
    # Try header=0 first, then header=None
    miss = pd.read_csv(MISSING_POSCAR_CSV, encoding="utf-8")
    print(f"  shape (with header=0): {miss.shape}")
    print(f"  columns: {list(miss.columns)}")
    print(f"  dtypes:\n{miss.dtypes.to_string()}")
    print(f"  first 5 rows:")
    print(miss.head().to_string())
    # Also preview with no header just in case
    miss_nh = pd.read_csv(MISSING_POSCAR_CSV, header=None, encoding="utf-8", nrows=5)
    print(f"\n  (for reference) first 5 rows if header=None:")
    print(miss_nh.to_string())
safe(_chk_missing)


# -------------------------------------------------------------
# [9] Regex sanity check on feff.sample_name (first 5)
# -------------------------------------------------------------
def _chk_regex():
    hdr("[9] sample_name regex sanity check (first 5 rows of FEFF_CSV)")
    pattern = r"^(mp-\d+)__\1-EXAFS-([A-Z][a-z]?)-K$"
    print(f"  pattern: {pattern}")
    feff_names = pd.read_csv(FEFF_CSV, usecols=["sample_name"],
                             encoding="utf-8", nrows=5)
    for name in feff_names["sample_name"]:
        m = re.match(pattern, str(name))
        if m:
            print(f"    OK    {name}  ->  mp_id={m.group(1)}  element={m.group(2)}")
        else:
            print(f"    FAIL  {name}  ->  regex did not match")
safe(_chk_regex)


# -------------------------------------------------------------
# [10] Cross-check a single sample_name exists as chi / xmu file
# -------------------------------------------------------------
def _chk_join():
    hdr("[10] JOIN sanity: does feff sample_name[0] exist as chi/xmu file?")
    s_name = pd.read_csv(FEFF_CSV, usecols=["sample_name"],
                         encoding="utf-8", nrows=1).iloc[0, 0]
    print(f"  sample_name[0] = {s_name}")
    # Candidates to probe
    for ext in (".csv", ""):
        chi_p = os.path.join(CHI_DIR, s_name + ext)
        xmu_p = os.path.join(XMU_DIR, s_name + ext)
        print(f"    CHI  '{s_name+ext}' exists? {os.path.isfile(chi_p)}")
        print(f"    XMU  '{s_name+ext}' exists? {os.path.isfile(xmu_p)}")
    # For POSCAR: by mp_id
    m = re.match(r"^(mp-\d+)__", str(s_name))
    if m:
        mp_id = m.group(1)
        print(f"  mp_id = {mp_id}, probing POSCAR_DIR for candidates:")
        for cand in (mp_id, mp_id + ".poscar", mp_id + ".vasp",
                     mp_id + ".POSCAR", mp_id.upper()):
            p = os.path.join(POSCAR_DIR, cand)
            print(f"    '{cand}' exists? {os.path.isfile(p)}")
safe(_chk_join)


print("\n" + "=" * 72)
print("Sanity check complete. Copy the ENTIRE output above and paste")
print("it back to the Sub-Agent. Do NOT edit or truncate.")
print("=" * 72)
