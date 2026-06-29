# step6_0_hard_check.py
# DiffCSP-Exp4 Step 6 — Phase 6.0 Hard Check (Step6Agent)
# ============================================================
# Purpose: verify EVERY assumption before writing step6_visualize.py.
# Run with EXPLICIT mlff env absolute path:
#     /home/tcat/conda_envs/mlff/bin/python step6_0_hard_check.py
#
# Exit code 0  = all green, proceed to Phase 6.1
# Exit code 1  = at least one fail, send full log back to Step6Agent
# ============================================================

import os
import sys
import importlib

FAILS = []
def ok(msg):  print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg):
    print(f"  [FAIL] {msg}")
    FAILS.append(msg)

print("=" * 72)
print("Step6Agent Phase 6.0 — Hard Check")
print("=" * 72)


# ─── §1 Env ──────────────────────────────────────────────────
print("\n[§1] Python env + key packages")
print(f"  sys.executable  : {sys.executable}")
print(f"  python version  : {sys.version.split()[0]}")
EXPECTED_PY = "/home/tcat/conda_envs/mlff/bin/python"
if sys.executable != EXPECTED_PY:
    fail(f"WRONG ENV. expected {EXPECTED_PY}, got {sys.executable}")
else:
    ok("mlff env confirmed (absolute path)")

for pkg in ["numpy", "pandas", "matplotlib", "scipy", "torch", "pymatgen"]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "unknown")
        ok(f"{pkg:12s} {v}")
    except Exception as e:
        fail(f"{pkg} import failed: {e}")


# ─── §2 Input files exist ────────────────────────────────────
print("\n[§2] Input files exist + sizes")
INPUTS = {
    "val_csv":     "/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv",
    "test_csv":    "/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_test.csv",
    "holdout_csv": "/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_holdout.csv",
    "val_pt":      "/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt",
    "inventory":   "/home/tcat/diffcsp_exp4/data/data_inventory_v2.csv",
}
for name, path in INPUTS.items():
    if not os.path.exists(path):
        fail(f"{name}: MISSING — {path}")
    else:
        sz_mb = os.path.getsize(path) / (1024 * 1024)
        ok(f"{name:12s} {sz_mb:8.2f} MB  {path}")


# ─── §3 per_sample_metrics CSV schema + row counts ───────────
print("\n[§3] per_sample_metrics CSV schemas + row counts")
import pandas as pd

EXPECTED_COLS = {"sample_name", "mp_id", "rmsd", "type_acc",
                 "n_pred_in", "n_true_in", "eval_cutoff"}
EXPECTED_N = {"val_csv": 7621, "test_csv": 4481, "holdout_csv": 3025}
EXPECTED_RMSD = {"val_csv": 1.4849, "test_csv": 1.4852, "holdout_csv": 1.4866}
EXPECTED_TACC = {"val_csv": 0.1877, "test_csv": 0.1904, "holdout_csv": 0.1973}

dfs = {}
for name in ["val_csv", "test_csv", "holdout_csv"]:
    path = INPUTS[name]
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    dfs[name] = df
    cols = set(df.columns)
    miss = EXPECTED_COLS - cols
    extra = cols - EXPECTED_COLS

    if miss:
        fail(f"{name}: missing cols {miss}")
        continue
    if len(df) != EXPECTED_N[name]:
        fail(f"{name}: rows={len(df)}, expected {EXPECTED_N[name]}")
    else:
        ok(f"{name}: {len(df)} rows, cols {sorted(cols)}")
    if extra:
        warn(f"{name}: extra cols (fine) {sorted(extra)}")

    # cross-check aggregates against Step5 report
    rm = df["rmsd"].mean()
    ta = df["type_acc"].mean()
    rm_exp = EXPECTED_RMSD[name]
    ta_exp = EXPECTED_TACC[name]
    print(f"     rmsd     mean={rm:.4f}  (Step5 report: {rm_exp:.4f}, "
          f"|Δ|={abs(rm-rm_exp):.4f})")
    print(f"     type_acc mean={ta:.4f}  (Step5 report: {ta_exp:.4f}, "
          f"|Δ|={abs(ta-ta_exp):.4f})")
    if abs(rm - rm_exp) > 0.001:
        fail(f"{name}: rmsd mean drift > 0.001")
    if abs(ta - ta_exp) > 0.001:
        fail(f"{name}: type_acc mean drift > 0.001")
    print(f"     eval_cutoff: min={df['eval_cutoff'].min():.3f}, "
          f"max={df['eval_cutoff'].max():.3f}, mean={df['eval_cutoff'].mean():.3f}")


# ─── §4 data_inventory_v2.csv schema ─────────────────────────
print("\n[§4] data_inventory_v2.csv schema")
inv = pd.read_csv(INPUTS["inventory"])
print(f"  rows: {len(inv)}")
print(f"  cols: {list(inv.columns)}")

needed = {"mp_id", "center_element", "sample_name"}
have = set(inv.columns)
miss = needed - have
if miss:
    fail(f"inventory missing required cols: {miss}")
else:
    ok("inventory has mp_id + center_element + sample_name")

if "center_element" in inv.columns:
    print(f"  center_element dtype : {inv['center_element'].dtype}")
    print(f"  center_element nunique: {inv['center_element'].nunique()}")
    print(f"  center_element head(5): {inv['center_element'].head(5).tolist()}")
    print(f"  is symbol-string?    : "
          f"{inv['center_element'].dtype == object and isinstance(inv['center_element'].iloc[0], str)}")


# ─── §5 join coverage: val CSV ↔ inventory ───────────────────
print("\n[§5] Join coverage (val CSV ↔ inventory)")
if "val_csv" in dfs and "sample_name" in inv.columns and "center_element" in inv.columns:
    df_val = dfs["val_csv"]

    # try sample_name first
    inv_sn = inv[["sample_name", "center_element"]].drop_duplicates("sample_name")
    j_sn = df_val.merge(inv_sn, on="sample_name", how="left")
    miss_sn = j_sn["center_element"].isna().sum()
    print(f"  via sample_name (preferred): missing = {miss_sn}/{len(df_val)}")

    # try mp_id (less specific in 88-element world)
    inv_mp = inv[["mp_id", "center_element"]].drop_duplicates("mp_id")
    j_mp = df_val.merge(inv_mp, on="mp_id", how="left")
    miss_mp = j_mp["center_element"].isna().sum()
    print(f"  via mp_id (fallback)       : missing = {miss_mp}/{len(df_val)}")

    if miss_sn == 0:
        ok("→ use 'sample_name' as join key for fig 3 center-element labeling")
    elif miss_mp == 0:
        warn("sample_name has gaps; mp_id is clean — fig 3 will use mp_id")
    else:
        fail(f"both join keys have missing entries (sn={miss_sn}, mp={miss_mp})")
else:
    warn("skipped — required columns not all present")


# ─── §6 pymatgen Jmol palette ────────────────────────────────
print("\n[§6] pymatgen Jmol palette + Element.from_Z")
JMOL_PATH = None
jmol = None

# try the path in handoff first
try:
    from pymatgen.vis.structure_vtk import EL_COLORS as _EL_COLORS_VTK
    if "Jmol" in _EL_COLORS_VTK:
        jmol = _EL_COLORS_VTK["Jmol"]
        JMOL_PATH = "pymatgen.vis.structure_vtk.EL_COLORS['Jmol']"
        ok(f"import path: {JMOL_PATH}")
except ImportError as e:
    warn(f"pymatgen.vis.structure_vtk import failed: {e}")
except KeyError:
    warn("EL_COLORS imported but no 'Jmol' key")

# fallback: try newer pymatgen path
if jmol is None:
    try:
        # pymatgen >=2023 keeps it under pymatgen.util.string or pymatgen.io.cif sometimes
        from pymatgen.io.babel import EL_COLORS as _EL_COLORS_BAB  # unlikely but try
        if "Jmol" in _EL_COLORS_BAB:
            jmol = _EL_COLORS_BAB["Jmol"]
            JMOL_PATH = "pymatgen.io.babel.EL_COLORS['Jmol']"
    except Exception:
        pass

if jmol is None:
    # last resort: try the raw json from pymatgen
    try:
        from pymatgen.core.periodic_table import Element  # noqa
        # Some pymatgen ship Jmol colors at ase.data or specific util — just probe
        import pymatgen
        pmg_root = os.path.dirname(pymatgen.__file__)
        cand = os.path.join(pmg_root, "vis", "ElementColorSchemes.yaml")
        if os.path.exists(cand):
            print(f"  found raw schema: {cand}")
            warn("pymatgen >= newer split EL_COLORS out of vis.structure_vtk; "
                 "see ElementColorSchemes.yaml — Step6Agent will hand-load")
    except Exception:
        pass

if jmol is None:
    fail("could not locate Jmol palette via any pymatgen path. Step6Agent "
         "needs this to confirm before writing fig 3. Send this log back.")
else:
    ok(f"Jmol palette located, size = {len(jmol)} entries")
    print(f"  sample keys: {list(jmol.keys())[:8]}")
    for sym in ["Fe", "O", "Cu", "H", "Na"]:
        v = jmol.get(sym)
        print(f"    {sym}: {v}  (type={type(v).__name__})")
    # verify shape
    fe = jmol.get("Fe")
    if fe is not None and len(fe) == 3 and max(fe) > 1:
        ok("RGB 0-255 format confirmed (will divide by 255 for matplotlib)")
    elif fe is not None and len(fe) == 3 and max(fe) <= 1.0:
        warn("RGB 0-1 format detected — Step6Agent must NOT divide by 255")

# Element.from_Z
try:
    from pymatgen.core import Element
    sym = Element.from_Z(26).symbol
    if sym == "Fe":
        ok("Element.from_Z(26).symbol = 'Fe' OK")
    else:
        fail(f"Element.from_Z(26).symbol = {sym!r}, expected 'Fe'")
    # spot-check a few that should appear in 88-element regime
    for z in [8, 26, 29, 1, 11]:
        s = Element.from_Z(z).symbol
        print(f"    Z={z:3d} → {s}")
except Exception as e:
    fail(f"pymatgen.core.Element.from_Z failed: {e}")


# ─── §7 predictions_val.pt schema (re-verify) ────────────────
print("\n[§7] predictions_val.pt schema (re-verify)")
import torch
import numpy as np

preds = torch.load(INPUTS["val_pt"], map_location="cpu", weights_only=False)
N = len(preds["mp_id"])
L = preds.get("L", None)
print(f"  N = {N} (expected 7621)")
print(f"  L = {L} (expected 6.0)")
print(f"  checkpoint = {preds.get('checkpoint', 'MISSING')}")
print(f"  n_nominal  = {preds.get('n_nominal', 'MISSING')}")
print(f"  n_effective= {preds.get('n_effective', 'MISSING')}")

if N != 7621:
    fail(f"N = {N}, expected 7621")
if L not in (6.0, 6):
    fail(f"L = {L}, expected 6.0")

# 3 random sample spot-check
import random
random.seed(0)
for i in random.sample(range(N), 3):
    pf = preds["pred_frac_coords"][i].cpu().numpy()
    pt = preds["pred_atom_types"][i].cpu().numpy()
    tf = preds["true_frac_coords"][i].cpu().numpy()
    tt = preds["true_atom_types"][i].cpu().numpy()
    sn = preds["sample_name"][i]
    print(f"  i={i:5d}  sn={sn}")
    print(f"    pred_frac shape={pf.shape}, range=[{pf.min():.3f}, {pf.max():.3f}]")
    print(f"    pred_types unique Z = {sorted(np.unique(pt).tolist())[:8]}")
    print(f"    true_frac shape={tf.shape}, range=[{tf.min():.3f}, {tf.max():.3f}]")
    print(f"    true_types unique Z = {sorted(np.unique(tt).tolist())[:8]}")
    if pf.shape != (20, 3) or tf.shape != (20, 3):
        fail(f"sample {i}: frac_coords shape != (20,3)")
    if pt.shape != (20,) or tt.shape != (20,):
        fail(f"sample {i}: atom_types shape != (20,)")
    if pf.min() < -0.51 or pf.max() > 0.51:
        fail(f"sample {i}: pred_frac out of [-0.5,0.5] (got [{pf.min()}, {pf.max()}])")


# ─── §8 sample_name alignment: pt ↔ val_csv ──────────────────
print("\n[§8] sample_name alignment between predictions_val.pt and val_csv")
if "val_csv" in dfs:
    pt_names = set(preds["sample_name"])
    csv_names = set(dfs["val_csv"]["sample_name"])
    inter = pt_names & csv_names
    only_pt  = pt_names - csv_names
    only_csv = csv_names - pt_names
    print(f"  pt sample_names  : {len(pt_names)}")
    print(f"  csv sample_names : {len(csv_names)}")
    print(f"  intersection     : {len(inter)}")
    print(f"  pt-only          : {len(only_pt)}")
    print(f"  csv-only         : {len(only_csv)}")
    if len(inter) == N == len(csv_names):
        ok("perfect alignment — sample_name is a stable join key for fig3/fig5")
    else:
        fail(f"alignment mismatch — fig3/fig5 need 1-to-1 sample_name map")


# ─── §9 output dir ───────────────────────────────────────────
print("\n[§9] Output dir")
out = "/home/tcat/diffcsp_exp4/code/step6/figures"
try:
    os.makedirs(out, exist_ok=True)
    ok(f"ensured: {out}")
except Exception as e:
    fail(f"mkdir {out}: {e}")


# ─── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 72)
if FAILS:
    print(f"Phase 6.0 RESULT: FAIL ({len(FAILS)} issue(s))")
    for f in FAILS:
        print(f"  - {f}")
    print("=" * 72)
    sys.exit(1)
else:
    print("Phase 6.0 RESULT: PASS — Step6Agent cleared to write step6_visualize.py")
    print("=" * 72)
    sys.exit(0)
