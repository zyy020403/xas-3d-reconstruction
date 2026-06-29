#!/usr/bin/env python
"""
exp5step0check.py
========================================================================
Exp5 SA0 pre-flight check — gathers everything Main Agent needs to know
before writing multisample.py.

READ-ONLY. Touches nothing in /home/tcat/diffcsp_exp4/.
Creates /home/tcat/diffcsp_exp5/sa0/{scripts,results,logs} if writable.
Exits 0 regardless of findings (this is a probe, not a gate).

Usage (anywhere; PYTHONPATH not required for this probe):
  mkdir -p /home/tcat/diffcsp_exp5/sa0/logs
  python exp5step0check.py 2>&1 | tee /home/tcat/diffcsp_exp5/sa0/logs/exp5step0check.log

Then paste the log back to Main Agent.
"""

import os, sys, json, hashlib, platform, traceback

EXP4_ROOT = "/home/tcat/diffcsp_exp4"
EXP5_ROOT = "/home/tcat/diffcsp_exp5"
SA0_ROOT  = f"{EXP5_ROOT}/sa0"

CKPT_PATH    = f"{EXP4_ROOT}/checkpoints/best-epoch366-val0.7300.ckpt"
STEP5_DIR    = f"{EXP4_ROOT}/code/step5"
PRED_VAL     = f"{STEP5_DIR}/predictions_val.pt"
PRED_TEST    = f"{STEP5_DIR}/predictions_test.pt"
PSM_VAL_CSV  = f"{STEP5_DIR}/per_sample_metrics_val.csv"
SAMPLE_PY    = f"{STEP5_DIR}/step5_1_sample.py"
METRICS_PY   = f"{STEP5_DIR}/step5_2_compute_metrics.py"
DM_PY        = f"{EXP4_ROOT}/code/xas_local_datamodule_v2.py"
CONF_DIR     = f"{EXP4_ROOT}/code/step3/conf_xas"

findings = []

def add(level, tag, msg):
    findings.append((level, tag, msg))
    print(f"[{level:4s}] {tag}: {msg}")

def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

def md5_head(path, n_bytes=10 * 1024 * 1024):
    """First 10 MB md5 — fast identity check for big ckpt, not full integrity."""
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()

def md5_full(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ── 0. Banner ────────────────────────────────────────────────────────────
section("Exp5 SA0 pre-flight check")
print(f"Python    : {sys.version.split()[0]}")
print(f"Platform  : {platform.platform()}")
print(f"CWD       : {os.getcwd()}")
print(f"User      : {os.environ.get('USER', '?')}")
print(f"Hostname  : {platform.node()}")

# ── 1. Path existence + sizes ────────────────────────────────────────────
section("1. Path existence + sizes")
required_files = [
    (CKPT_PATH,    "ckpt"),
    (SAMPLE_PY,    "step5_1_sample.py"),
    (METRICS_PY,   "step5_2_compute_metrics.py"),
    (DM_PY,        "xas_local_datamodule_v2.py"),
    (PRED_VAL,     "predictions_val.pt"),
    (PSM_VAL_CSV,  "per_sample_metrics_val.csv"),
]
optional_files = [
    (PRED_TEST,    "predictions_test.pt"),
]
required_dirs = [
    (EXP4_ROOT,  "exp4 root"),
    (STEP5_DIR,  "step5 dir"),
    (CONF_DIR,   "hydra conf dir"),
]

for path, label in required_files:
    if os.path.isfile(path):
        sz = os.path.getsize(path)
        add("PASS", f"file:{label}", f"{path}  size={sz/1e6:.2f} MB")
    else:
        add("FAIL", f"file:{label}", f"MISSING: {path}")

for path, label in optional_files:
    if os.path.isfile(path):
        sz = os.path.getsize(path)
        add("PASS", f"file:{label}", f"{path}  size={sz/1e6:.2f} MB")
    else:
        add("WARN", f"file:{label}", f"absent (optional): {path}")

for path, label in required_dirs:
    if os.path.isdir(path):
        add("PASS", f"dir:{label}", path)
    else:
        add("FAIL", f"dir:{label}", f"MISSING: {path}")

# ── 1b. Create + write-probe SA0 dir tree ────────────────────────────────
section("1b. SA0 dir tree (create + write probe)")
try:
    for sub in ["scripts", "results", "logs"]:
        d = os.path.join(SA0_ROOT, sub)
        os.makedirs(d, exist_ok=True)
    probe = os.path.join(SA0_ROOT, "logs", ".write_probe")
    with open(probe, "w") as f:
        f.write("ok")
    os.remove(probe)
    add("PASS", "sa0_dir", f"{SA0_ROOT}/{{scripts,results,logs}} present + writable")
except Exception as e:
    add("FAIL", "sa0_dir", f"cannot create/write {SA0_ROOT}: {type(e).__name__}: {e}")

# ── 2. ckpt fingerprint ──────────────────────────────────────────────────
section("2. Checkpoint identity")
if os.path.isfile(CKPT_PATH):
    sz = os.path.getsize(CKPT_PATH)
    add("INFO", "ckpt_size_bytes", str(sz))
    add("INFO", "ckpt_md5_head_10MB", md5_head(CKPT_PATH))
    # full md5 only if file is < 1 GB
    if sz < 1_000_000_000:
        add("INFO", "ckpt_md5_full", md5_full(CKPT_PATH))
    else:
        add("INFO", "ckpt_md5_full", "skipped (>1GB)")
else:
    add("FAIL", "ckpt", "not found, cannot fingerprint")

# ── 3. step5_1 / step5_2 / dm fingerprints ───────────────────────────────
section("3. Source file fingerprints (drift detection)")
for path, label in [(SAMPLE_PY, "step5_1"),
                    (METRICS_PY, "step5_2"),
                    (DM_PY, "datamodule_v2")]:
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                content = f.read()
            h = hashlib.md5(content).hexdigest()
            n_lines = content.count(b"\n") + 1
            add("INFO", f"md5:{label}", f"md5={h}  lines={n_lines}  bytes={len(content)}")
        except Exception as e:
            add("WARN", f"md5:{label}", f"read fail: {e}")

# ── 4. predictions_val.pt schema (THE critical one) ──────────────────────
section("4. predictions_val.pt schema (K=1 ground truth file)")
try:
    import torch
    if os.path.isfile(PRED_VAL):
        preds = torch.load(PRED_VAL, map_location="cpu", weights_only=False)
        if not isinstance(preds, dict):
            add("FAIL", "pred_val_type", f"expected dict, got {type(preds).__name__}")
        else:
            keys = sorted(preds.keys())
            add("INFO", "keys", str(keys))

            # scalar fields
            scalars = {}
            for k, v in preds.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    scalars[k] = v
            add("INFO", "scalar_fields", json.dumps(scalars, default=str))

            # sample_name list
            if "sample_name" in preds:
                sn = preds["sample_name"]
                add("INFO", "sample_name", f"type={type(sn).__name__}  len={len(sn)}  "
                                            f"first5={sn[:5]}  last1={sn[-1:]}")
                # uniqueness sanity
                add("INFO", "sample_name_unique", f"{len(set(sn))} / {len(sn)} unique")

            # mp_id
            if "mp_id" in preds:
                mp = preds["mp_id"]
                add("INFO", "mp_id", f"len={len(mp)}  first3={mp[:3]}")

            # eval_cutoff distribution + tier dist (CRUCIAL for stratified subset)
            if "eval_cutoff" in preds:
                import numpy as np
                ec = preds["eval_cutoff"]
                ec_arr = np.array(ec, dtype=np.float64)
                add("INFO", "eval_cutoff_stats",
                    f"len={len(ec)}  min={ec_arr.min():.4f}  max={ec_arr.max():.4f}  "
                    f"mean={ec_arr.mean():.4f}  median={float(np.median(ec_arr)):.4f}  "
                    f"std={ec_arr.std():.4f}")
                n_total = len(ec_arr)
                tier_a = int((ec_arr < 3.0).sum())
                tier_b = int(((ec_arr >= 3.0) & (ec_arr < 4.0)).sum())
                tier_c = int(((ec_arr >= 4.0) & (ec_arr < 5.0)).sum())
                tier_d = int((ec_arr >= 5.0).sum())
                add("INFO", "val_tier_dist",
                    f"A(<3)={tier_a}({100*tier_a/n_total:.1f}%) "
                    f"B(3-4)={tier_b}({100*tier_b/n_total:.1f}%) "
                    f"C(4-5)={tier_c}({100*tier_c/n_total:.1f}%) "
                    f"D(>=5)={tier_d}({100*tier_d/n_total:.1f}%) "
                    f"total={n_total}")

            # tensor field shapes (first sample)
            for k in ["pred_frac_coords", "pred_atom_types",
                      "true_frac_coords", "true_atom_types"]:
                if k in preds:
                    lst = preds[k]
                    add("INFO", f"{k}_listlen", f"{len(lst)}")
                    if len(lst) > 0:
                        t0 = lst[0]
                        if torch.is_tensor(t0):
                            try:
                                mn = t0.float().min().item()
                                mx = t0.float().max().item()
                            except Exception:
                                mn = mx = float("nan")
                            add("INFO", f"{k}[0]",
                                f"shape={tuple(t0.shape)}  dtype={t0.dtype}  "
                                f"min={mn:.4f}  max={mx:.4f}")
                            # for atom_types, also dump unique values
                            if "atom_types" in k:
                                try:
                                    uniq = sorted(set(t0.tolist()))
                                    add("INFO", f"{k}[0]_unique",
                                        f"n_unique={len(uniq)}  "
                                        f"head={uniq[:15]}{'...' if len(uniq)>15 else ''}")
                                except Exception:
                                    pass
                        else:
                            add("WARN", f"{k}[0]", f"NOT a tensor: {type(t0).__name__}")
    else:
        add("FAIL", "pred_val", "missing — cannot derive subset, cannot run K=1 sanity")
except Exception as e:
    add("FAIL", "pred_val_load", f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ── 5. per_sample_metrics_val.csv schema ─────────────────────────────────
section("5. per_sample_metrics_val.csv schema")
try:
    import csv
    if os.path.isfile(PSM_VAL_CSV):
        with open(PSM_VAL_CSV, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        add("INFO", "psm_header", str(header))
        add("INFO", "psm_n_rows", str(len(rows)))
        for i in range(min(3, len(rows))):
            add("INFO", f"psm_row[{i}]", str(rows[i]))
        # Try to compute mean rmsd/typeacc/pred_in for K=1 baseline reference
        try:
            import numpy as np
            col_idx = {c: i for i, c in enumerate(header)}
            rmsds   = np.array([float(r[col_idx["rmsd"]])      for r in rows])
            typeacc = np.array([float(r[col_idx["type_acc"]])  for r in rows])
            pin     = np.array([float(r[col_idx["n_pred_in"]]) for r in rows])
            add("INFO", "psm_K1_mean",
                f"RMSD={rmsds.mean():.4f}  TypeAcc={typeacc.mean():.4f}  "
                f"pred_in={pin.mean():.2f}  N={len(rows)}")
            # Bootstrap-style 95% CI on the mean for n=500 subset (analytic SE)
            n_sub = 500
            for name, arr in [("RMSD", rmsds), ("TypeAcc", typeacc), ("pred_in", pin)]:
                se_sub = arr.std(ddof=1) / (n_sub ** 0.5)
                add("INFO", f"K1_subset500_2sigma_band:{name}",
                    f"mean={arr.mean():.4f}  ±2σ_subset500={2*se_sub:.4f}  "
                    f"=> SA0 K=1 mean expected in [{arr.mean()-2*se_sub:.4f}, {arr.mean()+2*se_sub:.4f}]")
        except Exception as e:
            add("WARN", "psm_stats", f"could not compute K=1 stats: {e}")
    else:
        add("FAIL", "psm_missing", PSM_VAL_CSV)
except Exception as e:
    add("FAIL", "psm_load", f"{type(e).__name__}: {e}")

# ── 6. Library versions ──────────────────────────────────────────────────
section("6. Library versions")
def vget(modname):
    try:
        m = __import__(modname)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"NOT INSTALLED ({type(e).__name__}: {e})"
for m in ["torch", "numpy", "scipy", "pandas",
          "hydra", "omegaconf", "tqdm",
          "pytorch_lightning", "torch_geometric", "matplotlib"]:
    add("INFO", f"ver:{m}", vget(m))

# scipy linear_sum_assignment sanity (used by both Exp4 and SA0 hungarian agg)
try:
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    test_cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    r, c = linear_sum_assignment(test_cost)
    add("PASS", "scipy_lsa", f"row={r.tolist()}  col={c.tolist()}  "
                              f"total={test_cost[r, c].sum()}  (expected total=5.0)")
except Exception as e:
    add("FAIL", "scipy_lsa", f"{type(e).__name__}: {e}")

# ── 7. CUDA / GPU ────────────────────────────────────────────────────────
section("7. CUDA / GPU")
try:
    import torch
    cu = torch.cuda.is_available()
    add("INFO", "cuda_available", str(cu))
    if cu:
        n = torch.cuda.device_count()
        add("INFO", "cuda_device_count", str(n))
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            add("INFO", f"gpu[{i}]",
                f"{p.name}  total_mem={p.total_memory/1e9:.2f} GB  cc={p.major}.{p.minor}")
        add("INFO", "torch_cuda_ver", str(torch.version.cuda))
        add("INFO", "cudnn_ver", str(torch.backends.cudnn.version()))
except Exception as e:
    add("WARN", "cuda_probe", f"{type(e).__name__}: {e}")

# ── 8. Disk space ────────────────────────────────────────────────────────
section("8. Disk space")
for p_check in [EXP4_ROOT,
                EXP5_ROOT if os.path.isdir(EXP5_ROOT) else os.path.dirname(EXP5_ROOT),
                "/tmp"]:
    try:
        st = os.statvfs(p_check)
        free_gb = st.f_bavail * st.f_frsize / 1e9
        total_gb = st.f_blocks * st.f_frsize / 1e9
        add("INFO", f"disk:{p_check}",
            f"free={free_gb:.1f} GB / total={total_gb:.1f} GB "
            f"({100*free_gb/total_gb:.1f}% free)")
    except Exception as e:
        add("WARN", f"disk:{p_check}", str(e))

# ── 9. PYTHONPATH ────────────────────────────────────────────────────────
section("9. PYTHONPATH (current shell)")
pp = os.environ.get("PYTHONPATH", "(unset)")
add("INFO", "PYTHONPATH", pp)
add("INFO", "expected_PYTHONPATH",
    f"{EXP4_ROOT}/code:{EXP4_ROOT}/code/step3:{EXP4_ROOT}/code/step2  "
    f"(per step5_1_sample.py header)")

# ── 10. Import sanity (defensive sys.path injection like step5_1) ────────
section("10. Import sanity (mimics step5_1's defensive sys.path injection)")
for _p in [f"{EXP4_ROOT}/code", f"{EXP4_ROOT}/code/step3", f"{EXP4_ROOT}/code/step2"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import xas_local_datamodule_v2  # noqa: F401
    add("PASS", "import:xas_local_datamodule_v2", xas_local_datamodule_v2.__file__)
except Exception as e:
    add("WARN", "import:xas_local_datamodule_v2",
        f"{type(e).__name__}: {e}  (may need to chdir or set PYTHONPATH at SA0 runtime)")

# Hydra conf file check
diffusion_xas_yaml = os.path.join(CONF_DIR, "model", "diffusion_xas.yaml")
if os.path.isfile(diffusion_xas_yaml):
    add("PASS", "hydra_conf", diffusion_xas_yaml)
else:
    # alt: maybe .yaml extension differs
    model_dir = os.path.join(CONF_DIR, "model")
    if os.path.isdir(model_dir):
        listing = os.listdir(model_dir)
        add("WARN", "hydra_conf",
            f"diffusion_xas.yaml not at expected path; model/ contains: {listing}")
    else:
        add("FAIL", "hydra_conf", f"missing: {model_dir}")

# ── Summary ──────────────────────────────────────────────────────────────
section("SUMMARY")
n_pass = sum(1 for lv, _, _ in findings if lv == "PASS")
n_warn = sum(1 for lv, _, _ in findings if lv == "WARN")
n_fail = sum(1 for lv, _, _ in findings if lv == "FAIL")
print(f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")
if n_fail > 0:
    print("\nFAILS (must address before SA0 runtime):")
    for lv, tag, msg in findings:
        if lv == "FAIL":
            print(f"  ❌ {tag}: {msg}")
if n_warn > 0:
    print("\nWARNS (Main Agent will review):")
    for lv, tag, msg in findings:
        if lv == "WARN":
            print(f"  ⚠️  {tag}: {msg}")

print("\nDone. Paste this entire output back to Main Agent.")
print("(It's also saved to the tee'd log file.)")
