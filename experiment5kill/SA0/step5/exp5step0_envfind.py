#!/usr/bin/env python
"""
exp5step0_envfind.py
========================================================================
Helper: locate (a) the correct conda env Exp4 actually used, and
(b) where xas_local_datamodule_v2.py lives.

The first check ran in jhub_env which is missing hydra/lightning and
has no CUDA — so it can't possibly be Exp4's runtime env. This script
finds the right one.

READ-ONLY filesystem inspection. Doesn't activate any envs, doesn't
load any heavy modules. Should run in any env, any directory.

Usage (any env, any dir):
  python exp5step0_envfind.py 2>&1 | tee /home/tcat/diffcsp_exp5/sa0/logs/exp5step0_envfind.log
"""

import os, sys, subprocess, glob, traceback

EXP4_ROOT = "/home/tcat/diffcsp_exp4"

def section(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)

# ── 1. Where is xas_local_datamodule_v2.py? ──────────────────────────────
section("1. Locate xas_local_datamodule_v2.py")
search_roots = ["/home/tcat", "/opt", "/scratch", "/data", "/srv"]
hits = []
for root in search_roots:
    if not os.path.isdir(root):
        continue
    try:
        out = subprocess.run(
            ["find", root, "-name", "xas_local_datamodule*",
             "-type", "f", "-not", "-path", "*/.*"],
            capture_output=True, text=True, timeout=180
        )
        for line in out.stdout.strip().split("\n"):
            line = line.strip()
            if line:
                hits.append(line)
    except subprocess.TimeoutExpired:
        print(f"  (timeout searching {root}, skipping)")
    except Exception as e:
        print(f"  (error searching {root}: {e})")

if hits:
    for h in hits:
        try:
            sz = os.path.getsize(h)
            print(f"  FOUND: {h}  ({sz} B)")
        except Exception:
            print(f"  FOUND: {h}  (size unknown)")
else:
    print("  ❌ NOT FOUND under any of: " + ", ".join(search_roots))

# ── 1b. Listing of /home/tcat/diffcsp_exp4/code/ ─────────────────────────
section("1b. Listing of /home/tcat/diffcsp_exp4/code/ (top-level + step{2,3,5,6})")
exp4_code = f"{EXP4_ROOT}/code"
if os.path.isdir(exp4_code):
    print(f"  --- {exp4_code} top-level ---")
    for entry in sorted(os.listdir(exp4_code)):
        p = os.path.join(exp4_code, entry)
        kind = "DIR " if os.path.isdir(p) else "FILE"
        try:
            sz = os.path.getsize(p) if os.path.isfile(p) else "-"
        except Exception:
            sz = "?"
        print(f"    {kind}  {entry:40s}  {sz}")
    for sub in ["step2", "step3", "step4", "step5", "step6"]:
        sp = os.path.join(exp4_code, sub)
        if os.path.isdir(sp):
            py_files = sorted([e for e in os.listdir(sp) if e.endswith(".py")])
            print(f"\n  --- {sub}/ (.py files only, n={len(py_files)}) ---")
            for f in py_files:
                print(f"    {f}")

# ── 1c. Listing of /home/tcat/diffcsp_exp4/data/ (does data exist?) ──────
section("1c. /home/tcat/diffcsp_exp4/data/ contents (dataset availability)")
data_dir = f"{EXP4_ROOT}/data"
if os.path.isdir(data_dir):
    try:
        for entry in sorted(os.listdir(data_dir))[:50]:
            p = os.path.join(data_dir, entry)
            kind = "DIR " if os.path.isdir(p) else "FILE"
            try:
                sz = os.path.getsize(p) if os.path.isfile(p) else "-"
            except Exception:
                sz = "?"
            sz_str = f"{sz/1e6:.1f} MB" if isinstance(sz, int) and sz > 1e5 else str(sz)
            print(f"    {kind}  {entry:50s}  {sz_str}")
        n_total = len(os.listdir(data_dir))
        if n_total > 50:
            print(f"    ... ({n_total - 50} more entries)")
    except Exception as e:
        print(f"  list failed: {e}")
else:
    print(f"  ❌ {data_dir} NOT FOUND")

# ── 2. List conda envs ───────────────────────────────────────────────────
section("2. Available conda envs")
env_paths = []
try:
    out = subprocess.run(["conda", "env", "list"],
                         capture_output=True, text=True, timeout=30)
    print(out.stdout)
    for line in out.stdout.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Last column is path; first column may be name (or '*' if active)
        path = parts[-1]
        if not os.path.isdir(path):
            continue
        # Name guess: column 0 if not '*' or path
        if len(parts) >= 2 and not parts[0].startswith("*") and parts[0] != path:
            name = parts[0]
        elif len(parts) == 2 and parts[0] == "*":
            name = "(active, name?)"
        else:
            name = os.path.basename(path)
        env_paths.append((name, path))
except FileNotFoundError:
    print("  conda command not found; falling back to dir scan")
except Exception as e:
    print(f"  conda env list failed: {e}")

# Fallback: scan typical install locations
if not env_paths:
    for guess_root in ["/opt/miniconda3/envs", "/opt/anaconda3/envs",
                       "/opt/conda/envs",
                       os.path.expanduser("~/miniconda3/envs"),
                       os.path.expanduser("~/anaconda3/envs")]:
        if os.path.isdir(guess_root):
            for d in sorted(os.listdir(guess_root)):
                full = os.path.join(guess_root, d)
                if os.path.isdir(full):
                    env_paths.append((d, full))
print(f"\n  Total envs detected: {len(env_paths)}")

# ── 3. Per-env package + CUDA inspection ─────────────────────────────────
section("3. Per-env capability scan (filesystem only)")

def has_pkg(env_path, pkg):
    """Top-level package dir or .dist-info present?"""
    pats = [
        f"{env_path}/lib/python*/site-packages/{pkg}",
        f"{env_path}/lib/python*/site-packages/{pkg}-*.dist-info",
        f"{env_path}/lib/python*/site-packages/{pkg.replace('-','_')}",
        f"{env_path}/lib/python*/site-packages/{pkg.replace('-','_')}-*.dist-info",
    ]
    for p in pats:
        if glob.glob(p):
            return True
    return False

def torch_cuda_build(env_path):
    """Look for libtorch_cuda*.so — distinguishes CUDA build from CPU build."""
    return bool(glob.glob(f"{env_path}/lib/python*/site-packages/torch/lib/libtorch_cuda*"))

def torch_version_info(env_path):
    vfiles = glob.glob(f"{env_path}/lib/python*/site-packages/torch/version.py")
    if not vfiles:
        return "?"
    info = []
    try:
        with open(vfiles[0]) as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("__version__") or ln.startswith("cuda") or ln.startswith("debug") or ln.startswith("git_version"):
                    info.append(ln)
        return " | ".join(info)
    except Exception:
        return "(read error)"

print(f"  {'env':20s}  {'hyd':3s} {'omc':3s} {'lit':3s} {'tg':3s} {'tor':3s} {'CUDA':4s}  pick?")
print(f"  {'-'*20}  {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*4}  -----")
candidates = []
for name, path in env_paths:
    h = has_pkg(path, "hydra")
    o = has_pkg(path, "omegaconf")
    l = has_pkg(path, "pytorch_lightning")
    g = has_pkg(path, "torch_geometric")
    t = has_pkg(path, "torch")
    cu = torch_cuda_build(path) if t else False
    score = sum([h, o, l, g, t, cu])
    pick = ""
    if h and o and l and t and cu:
        pick = "★★★ TRY THIS"
        candidates.append((name, path, score))
    elif score >= 4:
        pick = "★★ partial"
        candidates.append((name, path, score))
    elif score >= 2:
        pick = "★ skip"
    print(f"  {name:20s}  {'Y' if h else '.':3s} {'Y' if o else '.':3s} "
          f"{'Y' if l else '.':3s} {'Y' if g else '.':3s} {'Y' if t else '.':3s} "
          f"{'Y' if cu else '.':4s}  {pick}")

# Detailed torch version for top candidates
section("3b. torch detail for promising envs")
candidates.sort(key=lambda x: -x[2])
for name, path, score in candidates[:5]:
    print(f"\n  --- {name} (score={score}) at {path} ---")
    print(f"    torch version.py:  {torch_version_info(path)}")
    # Also surface hydra/lightning version
    for pkg in ["hydra", "pytorch_lightning", "torch_geometric"]:
        di = glob.glob(f"{path}/lib/python*/site-packages/{pkg.replace('-','_')}-*.dist-info")
        if not di:
            di = glob.glob(f"{path}/lib/python*/site-packages/{pkg}-*.dist-info")
        if di:
            ver = os.path.basename(di[0]).replace(".dist-info", "")
            print(f"    {pkg}: {ver}")

# ── 4. Sniff Exp4 historical sample log to identify the env ─────────────
section("4. Exp4 historical step5 sample log (if exists)")
log_dirs = [f"{EXP4_ROOT}/logs", f"{EXP4_ROOT}/code/step5"]
for ld in log_dirs:
    if os.path.isdir(ld):
        print(f"  --- {ld}/ ---")
        for f in sorted(os.listdir(ld)):
            if f.endswith((".log", ".out", ".txt")):
                full = os.path.join(ld, f)
                try:
                    sz = os.path.getsize(full)
                    print(f"    {f}  ({sz/1024:.1f} KB)")
                except Exception:
                    print(f"    {f}  (size?)")

# Try reading first/last chunk of likely candidates
candidate_logs = [
    f"{EXP4_ROOT}/logs/step5_sample_val_test.log",
    f"{EXP4_ROOT}/logs/step5_sample_val.log",
    f"{EXP4_ROOT}/logs/step5_sample.log",
    f"{EXP4_ROOT}/logs/step5_metrics_val.log",
]
for c in candidate_logs:
    if os.path.isfile(c):
        print(f"\n  --- head -50 {c} ---")
        try:
            with open(c, "r", errors="replace") as f:
                lines = f.readlines()
            for line in lines[:50]:
                print(f"    {line.rstrip()}")
            print(f"    [... {max(0, len(lines)-50)} more lines ...]")
        except Exception as e:
            print(f"    read error: {e}")
        break
else:
    print("  (no Exp4 step5 sample log found at expected paths)")

# ── 5. nvidia-smi (host GPU visibility) ──────────────────────────────────
section("5. nvidia-smi (host GPU)")
try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if out.returncode == 0:
        for line in out.stdout.split("\n")[:25]:
            print(f"  {line}")
    else:
        print(f"  rc={out.returncode}\n  stderr: {out.stderr[:300]}")
except FileNotFoundError:
    print("  nvidia-smi not in PATH — could be HPC login node without GPU; need sbatch/qsub to GPU node?")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

# ── 6. /etc/profile.d hints (HPC module systems) ─────────────────────────
section("6. HPC env hints (module / lmod presence)")
for p in ["/etc/profile.d/modules.sh", "/etc/profile.d/lmod.sh",
          "/usr/share/modules/init/bash", "/usr/share/lmod/lmod/init/bash"]:
    if os.path.isfile(p):
        print(f"  found: {p}")
print(f"  $LMOD_CMD = {os.environ.get('LMOD_CMD', '(unset)')}")
print(f"  $MODULESHOME = {os.environ.get('MODULESHOME', '(unset)')}")

# ── 7. .bash_history / shell history hint (last cmds on this user) ──────
section("7. ~/.bash_history grep for 'conda activate' (recent pattern)")
hist = os.path.expanduser("~/.bash_history")
if os.path.isfile(hist):
    try:
        with open(hist, "r", errors="replace") as f:
            lines = f.readlines()
        # Last 200 lines, grep
        recent = lines[-500:]
        activate_lines = [l.rstrip() for l in recent
                          if "conda activate" in l or "source activate" in l
                          or "module load" in l or "source ~/" in l]
        seen = set()
        uniq = []
        for l in activate_lines:
            if l not in seen:
                seen.add(l)
                uniq.append(l)
        if uniq:
            print(f"  last unique env-related commands (most recent first):")
            for l in reversed(uniq[-20:]):
                print(f"    {l}")
        else:
            print("  no 'conda activate' / 'module load' lines in last 500 history entries")
    except Exception as e:
        print(f"  {e}")
else:
    print("  ~/.bash_history not readable")

print("\nDone. Paste this output back to Main Agent.")
