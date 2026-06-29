#!/usr/bin/env python
"""
step5_0_hard_check.py
========================================================================
DiffCSP-Exp4 Step 5 Phase 5.0 Hard Check

目的: 在 Step 5.1 sample 脚本写之前, 解析 Step5Agent 所需的全部未知量:
  - dataset_v2 / datamodule_v2 类名 + 构造签名
  - dataset_v2.__getitem__ 输出字段名 (mp_id vs sample_name)
  - ckpt 加载 sanity (epoch / state_dict / hyperparameters)
  - 14 个关键文件存在性 + disk / GPU / env

绝对禁令: 不读 holdout 内容 (只 stat).

用法:
  cd /home/tcat/diffcsp_exp4   # 任何目录都行, 路径都 hardcoded
  PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
    /home/tcat/conda_envs/mlff/bin/python step5_0_hard_check.py 2>&1 | \
    tee /home/tcat/diffcsp_exp4/logs/step5_0_hard_check.log

完成后把 .log 内容粘贴回 Step5Agent.
"""

import os, sys, shutil, subprocess, inspect, traceback, warnings
warnings.filterwarnings("ignore")

# ── 路径常量 (handoff 锁定) ─────────────────────────────────────────────
DIFFCSP_ROOT = "/home/tcat/diffcsp_exp4"
DATA_DIR     = f"{DIFFCSP_ROOT}/data"
CODE_DIR     = f"{DIFFCSP_ROOT}/code"
CKPT_PATH    = f"{DIFFCSP_ROOT}/checkpoints/best-epoch366-val0.7300.ckpt"

CRITICAL_FILES = [
    CKPT_PATH,
    f"{DATA_DIR}/data_inventory_v2.csv",
    f"{DATA_DIR}/val_samples_v2.csv",
    f"{DATA_DIR}/test_samples_v2.csv",
    f"{DATA_DIR}/holdout_samples_v2.csv",      # exist OK, NOT READ
    f"{DATA_DIR}/feff_features_imputed.pkl",
    f"{DATA_DIR}/feff_feature_scaler.pkl",
    f"{DATA_DIR}/spectra_val.pkl",
    f"{DATA_DIR}/spectra_test.pkl",
    f"{DATA_DIR}/spectra_holdout.pkl",          # exist OK, NOT READ
    f"{DATA_DIR}/shell_boundaries.pkl",
    f"{CODE_DIR}/step3/xas_local_dataset_v2.py",
    f"{CODE_DIR}/step3/xas_local_datamodule_v2.py",
    f"{CODE_DIR}/step3/diffusion_w_type_xas.py",
    f"{CODE_DIR}/step3/conf_xas/model/diffusion_xas.yaml",
    f"{CODE_DIR}/step2/spectrum_encoder.py",
]


def _section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _fmt_size(b):
    if b > 2**30: return f"{b/2**30:.1f} GB"
    if b > 2**20: return f"{b/2**20:.1f} MB"
    if b > 2**10: return f"{b/2**10:.1f} KB"
    return f"{b} B"


# ── 5.0.1 Disk / GPU / env basics ───────────────────────────────────────
_section("5.0.1  Disk / GPU / Python env")

total, used, free = shutil.disk_usage(os.path.expanduser("~"))
print(f"  disk(~)  total={total//2**30}GB  used={used//2**30}GB  free={free//2**30}GB")
print(f"  disk gate (>=30GB free): {'PASS ✓' if free >= 30*2**30 else 'FAIL ❌ (<30 GB)'}")

try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.free,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    print(f"  nvidia-smi:")
    for ln in smi.stdout.strip().splitlines():
        print(f"    {ln}")
except Exception as e:
    print(f"  nvidia-smi FAIL: {e}")

print(f"  python    {sys.executable}  ({sys.version.split()[0]})")
try:
    import torch, pytorch_lightning as pl, scipy
    print(f"  torch={torch.__version__}  pl={pl.__version__}  scipy={scipy.__version__}")
    print(f"  cuda.is_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
except ImportError as e:
    print(f"  IMPORT FAIL: {e}")
    sys.exit(1)


# ── 5.0.2 File existence ────────────────────────────────────────────────
_section("5.0.2  File existence (15 critical)")

missing = []
for f in CRITICAL_FILES:
    exists = os.path.exists(f)
    sz = _fmt_size(os.path.getsize(f)) if exists else "—"
    flag = "OK   " if exists else "MISS❌"
    print(f"  [{flag}]  {sz:>10s}  {f}")
    if not exists:
        missing.append(f)
if missing:
    print(f"\n  ❌ {len(missing)} files missing. Stopping.")
    sys.exit(1)


# ── 5.0.3 dataset_v2 / datamodule_v2 introspection ──────────────────────
_section("5.0.3  dataset_v2 / datamodule_v2 introspection")

# Inject sys.path defensively (PYTHONPATH should already cover, but safe)
for p in [CODE_DIR, f"{CODE_DIR}/step3", f"{CODE_DIR}/step2"]:
    if p not in sys.path:
        sys.path.insert(0, p)

DS_CLASS_NAME = None
DM_CLASS_NAME = None
ds_mod = dm_mod = None

try:
    import xas_local_dataset_v2 as ds_mod
    print(f"\n  xas_local_dataset_v2  →  {ds_mod.__file__}")
    classes = [n for n, o in inspect.getmembers(ds_mod, inspect.isclass)
               if o.__module__ == ds_mod.__name__]
    print(f"  classes: {classes}")
    from torch.utils.data import Dataset
    ds_subs = [c for c in classes if issubclass(getattr(ds_mod, c), Dataset)]
    print(f"  Dataset subclasses: {ds_subs}")
    if ds_subs:
        DS_CLASS_NAME = ds_subs[0]
        DSCls = getattr(ds_mod, DS_CLASS_NAME)
        sig = inspect.signature(DSCls.__init__)
        print(f"  → {DS_CLASS_NAME}.__init__{sig}")
except Exception as e:
    print(f"  IMPORT FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

try:
    import xas_local_datamodule_v2 as dm_mod
    print(f"\n  xas_local_datamodule_v2  →  {dm_mod.__file__}")
    classes = [n for n, o in inspect.getmembers(dm_mod, inspect.isclass)
               if o.__module__ == dm_mod.__name__]
    print(f"  classes: {classes}")
    import pytorch_lightning as pl
    dm_subs = [c for c in classes if issubclass(getattr(dm_mod, c), pl.LightningDataModule)]
    print(f"  LightningDataModule subclasses: {dm_subs}")
    if dm_subs:
        DM_CLASS_NAME = dm_subs[0]
        DMCls = getattr(dm_mod, DM_CLASS_NAME)
        sig = inspect.signature(DMCls.__init__)
        print(f"  → {DM_CLASS_NAME}.__init__{sig}")
    # collate fn detection
    fn_names = [n for n, _ in inspect.getmembers(dm_mod, inspect.isfunction)
                if 'collate' in n.lower()]
    print(f"  collate functions: {fn_names}")
    if 'xas_collate_fn_v2' in fn_names:
        print(f"  ✓ Phase 4.6 None-filter collate present")
except Exception as e:
    print(f"  IMPORT FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()


# ── 5.0.4 Look at training script for canonical DM instantiation ────────
_section("5.0.4  Train script DM / Trainer call (canonical args)")

train_script_candidates = [
    f"{CODE_DIR}/step4_exp4/step4_2_train.py",
    f"{CODE_DIR}/step4/step4_2_train.py",
    f"{CODE_DIR}/step4_exp4/step4_train.py",
]
train_script = next((p for p in train_script_candidates if os.path.exists(p)), None)
if train_script and DM_CLASS_NAME:
    print(f"  found: {train_script}")
    with open(train_script) as f:
        src = f.read()
    import re
    pat = re.compile(rf'\b{DM_CLASS_NAME}\s*\(([^)]*)\)', re.DOTALL)
    m = pat.search(src)
    if m:
        print(f"  → {DM_CLASS_NAME}({m.group(1).strip()[:600]})")
    else:
        print(f"  (no instantiation match for {DM_CLASS_NAME})")
else:
    print(f"  train script not found in candidates; will need manual peek")


# ── 5.0.5 Sample one val item, print fields ─────────────────────────────
_section("5.0.5  Sample one val item — field names + shapes")

sample_ok = False
if DM_CLASS_NAME and dm_mod is not None:
    try:
        DMCls = getattr(dm_mod, DM_CLASS_NAME)
        # try no-arg
        try:
            dm = DMCls()
            print(f"  DM() no-arg constructed ✓")
        except TypeError as te:
            # need args; try minimal common ones
            print(f"  DM() needs args ({te}); trying minimal kwargs...")
            try:
                dm = DMCls(
                    data_dir=DATA_DIR,
                    batch_size=16,
                    num_workers=0,
                )
                print(f"  DM(data_dir=, batch_size=16, num_workers=0) ✓")
            except Exception as te2:
                print(f"  Minimal kwargs FAIL: {te2}")
                dm = None

        if dm is not None:
            # try setup
            try:
                dm.setup("fit")
            except Exception as e:
                print(f"  dm.setup('fit') FAIL: {e}; trying setup(stage=None)")
                try:
                    dm.setup()
                except Exception as e2:
                    print(f"  dm.setup() FAIL: {e2}")

            # find val dataset
            ds = None
            for attr in ["val_ds", "val_dataset", "_val_ds", "val"]:
                if hasattr(dm, attr):
                    ds = getattr(dm, attr)
                    print(f"  dm.{attr} → len={len(ds) if ds else 'None'}")
                    break
            if ds is None:
                # try via val_dataloader
                try:
                    vl = dm.val_dataloader()
                    print(f"  dm.val_dataloader() → {type(vl).__name__}, len={len(vl)}")
                    ds = vl.dataset
                except Exception as e:
                    print(f"  no val dataset accessible: {e}")

            # fetch first non-None sample
            if ds is not None:
                sample = None
                for i in range(50):
                    try:
                        s = ds[i]
                    except Exception as e:
                        print(f"  ds[{i}] EXCEPTION: {e}")
                        continue
                    if s is not None:
                        sample = s
                        first_idx = i
                        break

                if sample is not None:
                    sample_ok = True
                    print(f"  first non-None sample at idx={first_idx}, type={type(sample).__name__}")
                    # PyG Data object
                    keys = None
                    if hasattr(sample, 'keys'):
                        try:
                            keys = sample.keys() if callable(sample.keys) else sample.keys
                        except Exception:
                            pass
                    if not keys:
                        # try .__dict__ or stores
                        keys = [k for k in dir(sample) if not k.startswith('_')]
                    print(f"  sample keys/attrs (filtered): {[k for k in keys if not callable(getattr(sample, k, None))][:30]}")

                    # check identifier fields
                    print(f"  ── identifier candidates ──")
                    for attr in ['mp_id', 'sample_name', 'mp_ids', 'sample_names',
                                 'name', 'id', 'sample_id', 'idx']:
                        if hasattr(sample, attr):
                            v = getattr(sample, attr)
                            print(f"    sample.{attr!s:20s} = {v!r}")

                    # tensor shapes
                    print(f"  ── tensor fields ──")
                    for attr in ['frac_coords', 'atom_types', 'lengths', 'angles',
                                 'eval_cutoff', 'num_atoms', 'num_nodes',
                                 'xmu_xanes', 'chi1', 'feff_features',
                                 'center_element']:
                        if hasattr(sample, attr):
                            v = getattr(sample, attr)
                            if hasattr(v, 'shape'):
                                print(f"    sample.{attr!s:20s} shape={tuple(v.shape)}  dtype={v.dtype}")
                            else:
                                print(f"    sample.{attr!s:20s} = {v!r}  ({type(v).__name__})")
                else:
                    print(f"  ❌ no non-None sample in first 50 of val (silent drop > expected)")
    except Exception as e:
        print(f"  setup/sample FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()


# ── 5.0.6 ckpt load sanity ──────────────────────────────────────────────
_section("5.0.6  ckpt load sanity")

import torch
try:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"  ckpt top-level keys: {list(ckpt.keys())}")
    print(f"  epoch              : {ckpt.get('epoch')}")
    print(f"  global_step        : {ckpt.get('global_step')}")
    sd = ckpt.get('state_dict', {})
    print(f"  state_dict size    : {len(sd)}")
    print(f"  state_dict[:5]     : {list(sd.keys())[:5]}")
    hp = ckpt.get('hyper_parameters', {})
    if hp:
        print(f"  hyper_parameters   : {len(hp)} keys")
        for kk in list(hp.keys())[:30]:
            v = hp[kk]
            vstr = str(v)[:80] if not hasattr(v, 'shape') else f"<tensor {tuple(v.shape)}>"
            print(f"    {str(kk):30s} = {vstr}")
except Exception as e:
    print(f"  ckpt load FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()


# ── 5.0.7 Holdout safety (exist but NOT read) ───────────────────────────
_section("5.0.7  Holdout safety check (file stat ONLY, no read)")

for f in [f"{DATA_DIR}/spectra_holdout.pkl",
          f"{DATA_DIR}/holdout_samples_v2.csv"]:
    if os.path.exists(f):
        print(f"  {os.path.basename(f):30s}  size={_fmt_size(os.path.getsize(f))}  (NOT OPENED)")
    else:
        print(f"  ❌ {os.path.basename(f)}: missing")


# ── 5.0.8 diffusion_w_type_xas grep for sample API ──────────────────────
_section("5.0.8  diffusion_w_type_xas.py — sample/forward API grep")

dwt_path = f"{CODE_DIR}/step3/diffusion_w_type_xas.py"
if os.path.exists(dwt_path):
    with open(dwt_path) as f:
        lines = f.readlines()
    print(f"  total lines: {len(lines)}")
    for keyword in ['def sample', 'def forward', 'class CSPDiffusion', 'def predict',
                    'num_steps', 'num_timesteps']:
        for i, ln in enumerate(lines):
            if keyword in ln:
                print(f"  L{i+1:4d}  {ln.rstrip()[:120]}")


print()
print("=" * 72)
print("  Hard check complete.")
print("  >>> Paste this output back to Step5Agent.")
print("=" * 72)
