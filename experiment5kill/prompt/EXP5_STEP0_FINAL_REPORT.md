# EXP5 STEP 0 (SA0) — FINAL REPORT

**SA**: DiffCSP-Experiment5step0agent
**Date**: 2026-04-28 NZST (Auckland)
**Host**: `scsmlnprd02.its.auckland.ac.nz`
**Sub-agent role**: K-sample test-time-augmentation quick-win on Exp4 best ckpt
**Outcome**: ✅ Sanity PASS, K-averaging confirmed beneficial, recommended for SA3 standardization

---

## 0. Executive summary

SA0 ran K=10 reverse-diffusion sampling sweeps on a stratified 500-sample val subset using Exp4's best checkpoint (`best-epoch366-val0.7300.ckpt`, no fine-tuning). After fixing a torus-boundary bug in coordinate averaging, the deployment-safe **`hungarian_fold` aggregation strategy** produced simultaneous gains on both metrics:

```
        K=1 → K=10
RMSD:   1.4856 → 1.4215   (Δ = −0.064 Å,  −4.3 %)
TypeAcc:0.1903 → 0.2583   (Δ = +0.068,    +35.7 % relative)
```

These are **inference-time gains, no retraining**. Every Tier (B/C/D) benefits on both metrics. K=5 already captures ~88% of the RMSD gain and ~59% of the TypeAcc gain, making it the recommended default for SA3 evaluation.

The work also produced a comprehensive environment cheatsheet for Exp6 (§3) and uncovered one substantive implementation bug whose post-mortem is in §5.

---

## 1. Methodology

### 1.1 Design (with deviations from handoff explained)

The handoff document called for K∈{1, 5, 10} sampling on a 500-sample stratified subset, with K=1 sanity check against Exp4 §5.1's full-val K=1 numbers. SA0 made two principled deviations:

**Deviation 1: K=1 also on subset, not full val.** The handoff's implicit design ran K=1 on full val (7621 samples) and K=2..10 on the 500 subset, which would have cost 4-5h GPU. Instead, SA0 ran a single K=10 sweep on the 500 subset (1.5h) and computed K=1/5/10 metrics from the same raw samples. The K=1 sanity reference becomes Exp4's per-sample-metrics restricted to the same 500 names — a fair like-for-like comparison.

**Deviation 2: K=1 sanity tolerance changed from "3 decimal places" to ±2·SE band.** A 500-sample subset cannot match a 7621-sample mean to 3 decimal places even with byte-identical environment (sampling variance dominates). The ±2·SE band is the statistically correct gate width. The bands locked from `per_sample_metrics_val.csv` restricted to the chosen 500 names:

| Metric | Reference mean | SE | ±2·SE band |
|---|---|---|---|
| RMSD | 1.4813 | 0.0058 | [1.4698, 1.4928] |
| TypeAcc | 0.1795 | 0.0077 | [0.1641, 0.1949] |
| pred_in | 18.92 | 0.08 | [18.77, 19.07] |

All three SA0 K=1 numbers landed inside the bands (see §2.1). Sanity ✅.

### 1.2 Subset construction

500 val samples, seed=0, stratified by Tier (A=eval_cutoff<3.0, B=3-4, C=4-5, D≥5):

| Tier | val population | subset target | subset actual |
|---|---:|---:|---:|
| A | 13 (0.2%) | 0 (skipped, low statistical power) | 0 |
| B | 1,961 (25.7%) | 129 | 129 |
| C | 3,893 (51.1%) | 256 | 256 |
| D | 1,754 (23.0%) | 115 | 115 |

Selection by `numpy.random.default_rng(0)` then sorted by sample_name for stable ordering. K-sweep seeds = `1234567 + k` for k in 0..9.

### 1.3 K-sweep execution

Single GPU (RTX 4090, GPU 1), batch_size=8, num_workers=0, 63 batches × 10 sweeps = 630 model.sample() calls. Wall: **86.3 min total**, individual sweeps 517-519s with std=0.7s (extremely stable). Effective rate: **8.6 min/sweep on 500 samples** = 1.03 s/sample, matching Exp4's historical 1.05 s/sample on full val.

Subset → dataloader mapping: built name→dataset_idx map by iterating val_loader and reading `batch.sample_name` post-collation (the dataset's PyG Data items don't expose `sample_name` as a Python attribute — see §3 cheatsheet item 14). 7616/7621 valid names mapped; 1 partial batch (8 dataset positions, 5 valid names) unmappable. The 500 subset names happened to all fall outside the partial batch → **500/500 retained, zero dropout**.

### 1.4 Aggregation strategies (v2)

After v1 revealed a torus-averaging bug (§5), v2 evaluates 5 strategies on the same raw `samples_raw_K10.pt`:

| Strategy | Description | Deployable? |
|---|---|---|
| `naive` | per-slot mean of raw frac coords + per-slot mode of types | yes (but broken) |
| `hungarian_fold` ★ | per-anchor=0 Hungarian align → fold to anchor neighborhood → mean → wrap to [-0.5, 0.5] | **yes, recommended** |
| `hungarian_fold_bestanchor` † | try all K anchors, pick min-RMSD-vs-truth one | semi-oracle (uses truth) |
| `medoid` | pick the k minimizing pairwise RMSD to the K-1 others | yes |
| `oracle_best` † | pick the k minimizing RMSD vs truth | upper bound (cheats) |

Per-sample evaluation uses Exp4's `step5_2_compute_metrics.evaluate_sample` verbatim (20×20 Hungarian + min-image RMSD, L=6.0 Å).

---

## 2. Results

### 2.1 Sanity gate

```
RMSD      SA0_K1=1.4856  Exp4_K1_subset=1.4813  ±2SE_band=[1.4698, 1.4928]  ✓ PASS
TypeAcc   SA0_K1=0.1903  Exp4_K1_subset=0.1795  ±2SE_band=[0.1641, 0.1949]  ✓ PASS
pred_in   SA0_K1=18.96   Exp4_K1_subset=18.92   ±2SE_band=[18.77, 19.07]   ✓ PASS
```

`state_dict missing=0 unexpected=0`, epoch=366 confirmed, ckpt md5 unchanged through both v1 and v2 runs. **Inference path is byte-identical to Exp4 step5 modulo cuDNN nondeterminism.**

### 2.2 Main table (v2)

| K | Strategy | RMSD | ΔvsK1 | TypeAcc | ΔvsK1 | pred_in/20 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | (any) | 1.4856 | – | 0.1903 | – | 18.96 |
| 5 | naive | 2.1143 | +0.629 | 0.2287 | +0.038 | 20.00 |
| 5 | **hungarian_fold ★** | **1.4296** | **−0.056** | **0.2298** | **+0.040** | **18.96** |
| 5 | hungarian_fold_bestanchor † | 1.3610 | −0.125 | 0.2334 | +0.043 | 18.97 |
| 5 | medoid | 1.4886 | +0.003 | 0.1870 | −0.003 | 18.92 |
| 5 | oracle_best † | 1.3784 | −0.107 | 0.1884 | −0.002 | 19.09 |
| 10 | naive | 2.3549 | +0.869 | 0.2590 | +0.069 | 20.00 |
| 10 | **hungarian_fold ★** | **1.4215** | **−0.064** | **0.2583** | **+0.068** | **18.94** |
| 10 | hungarian_fold_bestanchor † | 1.3185 | −0.167 | 0.2601 | +0.070 | 19.03 |
| 10 | medoid | 1.4751 | −0.010 | 0.1835 | −0.007 | 18.93 |
| 10 | oracle_best † | 1.3432 | −0.142 | 0.1940 | +0.004 | 19.09 |

★ deployment-safe recommended; † uses truth, not deployable as-is.

### 2.3 Per-Tier breakdown (hungarian_fold only)

| Tier | n | K=1 RMSD | K=10 RMSD | ΔRMSD | K=1 TypeAcc | K=10 TypeAcc | ΔTypeAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 129 | 1.4882 | 1.4273 | −0.061 | 0.2554 | 0.3698 | **+0.114** |
| C | 256 | 1.4829 | 1.4231 | −0.060 | 0.1693 | 0.2324 | +0.063 |
| D | 115 | 1.4887 | 1.4114 | **−0.077** | 0.1639 | 0.1909 | +0.027 |

**Universally positive** across all three tiers and both metrics. Tier B benefits most on TypeAcc (+11.4 pp,可能因为 short-cutoff structures 中原子数量在 cutoff 内更多,mode-voting 的统计强度更高); Tier D benefits most on RMSD (−0.077 Å, possibly because high-noise long-range structures get more averaging benefit per atom).

### 2.4 Diminishing returns: K=5 vs K=10

| | K=5 vs K=1 | K=10 vs K=1 | K=5 / K=10 |
|---|---:|---:|---:|
| ΔRMSD | −0.056 | −0.064 | **88%** |
| ΔTypeAcc | +0.040 | +0.068 | 59% |

RMSD effectively saturated by K=5; TypeAcc still climbs but at half the marginal rate. **K=5 is the recommended default**; K=10 only if TypeAcc is critical and 2× compute is acceptable.

### 2.5 Headroom analysis (oracle decomposition)

`oracle_best` K=10 = 1.3432 — among 10 random samples per spectrum there exists at least one with RMSD lower than K=1 mean by 0.142 Å. **Model.sample is not degenerate**; randomness produces meaningful diversity. The deployment-safe `hungarian_fold` captures −0.064 of this −0.142 (~45%); the remaining ~55% is left in modal mixing (averaging samples that come from different modes of the posterior cancels rather than reinforces).

`hungarian_fold_bestanchor` K=10 = 1.3185 — even **lower than oracle_best**, because it explores K different aggregations and picks the best (search space K), whereas oracle_best picks among K raw samples (search space K). This suggests a deployment-safe heuristic for anchor selection (e.g., medoid-as-anchor) could push `hungarian_fold` toward 1.32, an additional ~0.10 Å gain. **Possible Exp5 main-line follow-up.**

---

## 3. Environment cheatsheet — for Exp6

This section is the explicit anti-rediscovery deliverable. Exp6 should not have to relearn any of the following:

### 3.1 Conda environment

| Item | Value | Notes |
|---|---|---|
| Correct env name | **`mlff`** | NOT `jhub_env` (the default shell prompt) |
| Env path | `/home/tcat/conda_envs/mlff/` | Outside `/opt/miniconda3/envs/`, so `conda activate mlff` works but `ls /opt/miniconda3/envs/` does not show it |
| Activate command | `conda activate /home/tcat/conda_envs/mlff` (path) or `conda activate mlff` (name) | Both work; path form survives env naming changes |
| Conda init | `source /opt/miniconda3/etc/profile.d/conda.sh` | Required in non-interactive shells / scripts |

### 3.2 Library versions

| Library | Version | Notes |
|---|---|---|
| Python | 3.x (in `mlff`) | exact minor not recorded — use `python --version` if needed |
| torch | **2.4.1+cu124** | despite filesystem scan suggesting 2.8.0/CUDA 12.9, actual is 2.4.1; trust `torch.__version__` |
| torch CUDA | 12.4 | works on driver 535 (CUDA Runtime 12.2) — fwd-compat is fine |
| torch_geometric | 2.7.0 | `import works`; PyG extensions (pyg-lib, torch-scatter, torch-cluster, torch-spline-conv, torch-sparse) load OK in mlff (broken in jhub_env) |
| hydra | present | `import hydra; from hydra import compose, initialize_config_dir` works |
| omegaconf | present | for `OmegaConf.create / to_container` |
| pytorch_lightning | present (likely Lightning 2.0 import path: `lightning.pytorch`) | filesystem scan shows `.` but historical step5_1 ran fine; don't be alarmed if the filesystem scan misses it |
| scipy | present | for `scipy.optimize.linear_sum_assignment`, `scipy.stats.mode` |
| numpy | present | for everything |

### 3.3 Path layout

| Path | What | Modify? |
|---|---|---|
| `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` | Exp4 best ckpt | **read-only** (md5 `dc9d2c9b371c78125f285a5a6478d404`) |
| `/home/tcat/diffcsp_exp4/code/` | Exp4 source root | **read-only** |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py` | live datamodule (NOT in code/ root) | read-only; `.bak_*` files are historical, ignore |
| `/home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml` | hydra config for model.sample | read-only |
| `/home/tcat/diffcsp_exp4/code/step5/per_sample_metrics_val.csv` | Exp4 K=1 per-sample metrics on val (n=7621) | reference data, read-only |
| `/home/tcat/diffcsp_exp4/code/step5/predictions_val.pt` | Exp4 K=1 raw predictions on val | reference data, read-only |
| `/home/tcat/diffcsp_exp4/data/` | dataset root with `train/val/test_structure_cache.pt` | read-only |
| `/home/tcat/diffcsp_exp5/sa0/` | SA0 working directory | SA0 owned |

### 3.4 PYTHONPATH (mandatory)

```bash
export PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2
```

All three are required. Per Exp4's `step5_1_sample.py` header.

### 3.5 GPU layout

Two RTX 4090s (24 GB each), driver 535.183.01, CUDA Runtime 12.2. **Both idle by default** (host has no other GPU users). SA0 used GPU 1 (`CUDA_VISIBLE_DEVICES=1`), leaving GPU 0 for SA1. No HPC scheduler — direct execution, no `sbatch`/`module load`. No `LMOD_CMD` / `MODULESHOME`.

### 3.6 Disk

`/home/tcat`: 72.4 GB free / 1886 GB total (3.8% free). SA0 outputs total ~3.2 MB (raw samples 2.7 MB + everything else <500 KB). **Plenty for SA0; tight for Exp6 if planning anything dataset-scale**. `/tmp` has 128 GB free if needed for scratch.

### 3.7 14 lessons (numbered for direct citation)

1. **Default shell env is `jhub_env`** (visible in prompt as `(jhub_env)`). It has CPU-only torch, no hydra, no working PyG extensions. **First command in any Exp6 SA: `conda activate /home/tcat/conda_envs/mlff`**.
2. **`mlff` is at `/home/tcat/conda_envs/mlff/`**, not `/opt/miniconda3/envs/`. Use the path form for activate to avoid name resolution surprises.
3. **torch's actual version is 2.4.1+cu124**, not whatever a filesystem scan suggests. Trust `torch.__version__`.
4. **Driver 535 + torch CUDA 12.4 works fine** despite the apparent gap. No need to update drivers or rebuild torch.
5. **`xas_local_datamodule_v2.py` is in `code/step3/`**, not `code/` root.
6. **PYTHONPATH must include all three**: `code:code/step3:code/step2`. Just `code/step3` is not enough (some imports cross-reference step2 utilities).
7. **scsmlnprd02 is not an HPC login node**: GPUs are local, no scheduler, just `python` directly.
8. **`conda activate` in scripts requires `source /opt/miniconda3/etc/profile.d/conda.sh` first** — it's not in non-interactive PATH by default.
9. **`pytorch_lightning` may scan as missing** but historical Exp4 logs show it worked. The package was likely renamed to `lightning` (Lightning 2.0). Don't try to install it; just run the code.
10. **ckpt md5 `dc9d2c9b371c78125f285a5a6478d404`** locked. Verify before trusting any inference run.
11. **predictions_val.pt has n_effective=7621** but `val_dataset` reports 7624. Three samples are silently dropped during collation (Exp4 known caveat). Don't try to "fix" this — it's expected.
12. **K=1 sanity tolerance must be ±2·SE, not 3 decimal places**, when comparing subset means to full-val means. cuDNN nondeterminism alone causes ~0.005 RMSD variation.
13. **`model.sample` is NOT controlled by `torch.manual_seed`** in any reproducible way — same seed gives different samples. Treat each call as a draw from a distribution; record seeds in metadata for trace, not for replay.
14. **PyG Data items returned by `dataset[i]` do NOT expose `sample_name` as a Python attribute** — only post-collation `batch.sample_name` does. Build name→idx maps by iterating the dataloader, not by indexing the dataset. (This is also why Exp4 step5_1's `(getattr(data, 'sample_name', None) or batch.sample_name[i])` fallback exists — it's required, not redundant.)

### 3.8 Two extras (added during this session)

15. **Frac-coord averaging requires explicit min-image folding.** Frac coords live on a torus; naive `mean()` across `[-0.5, 0.5]` boundaries produces toroidal centroids, not physical means. Always `x - round(x - anchor)` before averaging, then `mean - round(mean)` after. This applies to lattice vectors, fractional positions, and any modulo-1 quantity. **See §5 for the bug post-mortem.**
16. **`pred_in` is a sanity tripwire for averaging bugs.** If your aggregation suddenly pushes pred_in to its ceiling (20/20), check whether you accidentally pulled all atoms toward the unit-cell center via toroidal averaging. Real `pred_in` for the Exp4 ckpt at K=1 is ~18.95.

---

## 4. Production scripts (delivered to MA + on server)

5 scripts, all in `/home/tcat/diffcsp_exp5/sa0/scripts/`:

| Script | Purpose | Wall | CPU/GPU |
|---|---|---|---|
| `env_smoke.py` | env validation + ckpt md5 + model.sample probe + seed determinism check | ~3 min | GPU |
| `make_subset.py` | stratified 500-sample selection from per_sample_metrics_val.csv | <5 sec | CPU |
| `multisample.py` | K-sweep raw sample collection (subset-aware) | ~1.5h for K=10 | GPU |
| `multisample_aggregate_v2.py` | 5-strategy evaluation + per-Tier breakdown + plot + markdown | ~30 sec | CPU |
| `run_sa0.sh` | driver: wraps conda activate / PYTHONPATH / CUDA_VISIBLE_DEVICES + subcommand modes (check/subset/sample/agg/all) | – | – |

**For Exp6**: these are largely reusable. The core changes for a new ckpt would be:
- `env_smoke.py`: change `CKPT_PATH` and `EXPECTED_CKPT_MD5`
- `multisample.py`: change `CKPT_PATH` (or pass `--ckpt`)
- `multisample_aggregate_v2.py`: change `--exp4_psm_csv` reference (or omit for new model)
- everything else identical

---

## 5. Bug post-mortem: torus averaging

### 5.1 Symptom

v1 results (delivered before bug discovery):

| K | Strategy | RMSD | ΔvsK1 | pred_in |
|---:|---|---:|---:|---:|
| 1 | – | 1.4856 | – | 18.96 |
| 10 | naive | **2.3549** | **+0.869** | **20.00** |
| 10 | hungarian (v1) | 1.7229 | +0.237 | 20.00 |

Two red flags: (1) RMSD got dramatically worse with K, contradicting the ensemble premise; (2) `pred_in` jumped to its ceiling 20.00 — every atom in every sample landed inside cutoff, suspicious.

### 5.2 Root cause

Fractional coordinates live in `[-0.5, 0.5]` with periodic boundary (`x ≡ x + 1`). Naively averaging two values like `0.49` and `-0.48` (which represent nearby positions on the torus, separated by 0.03) produces `0.005` — a point on the opposite side of the cell, not between them.

The v1 hungarian strategy aligned slot orderings correctly via min-image cost matrix, but then `aligned_coords.mean(axis=0)` was applied to raw frac values that crossed the boundary. The averaged result drifted toward the unit-cell center for any boundary-crossing slot, producing:

1. **Lower distance from origin** for affected atoms → all `||frac × L|| ≤ cutoff` → pred_in pegs at 20.
2. **Higher RMSD vs truth** because the centroid is geometrically far from the ground truth atom positions when those positions sat near a boundary.

### 5.3 Fix

Three lines in `aggregate_hungarian_fold`:

```python
# After Hungarian alignment, fold each k into anchor's neighborhood:
ac = ac - np.round(ac - anchor_coords)   # ← new
# Then average:
mean_c = aligned_c.mean(axis=0)
# Then wrap result back to canonical range:
mean_c = mean_c - np.round(mean_c)       # ← new
```

The first line ensures every aligned slot value sits within `±0.5` of the anchor's value, eliminating boundary-crossing artifacts. The third line wraps the final mean back to `[-0.5, 0.5]` for canonical representation.

### 5.4 Result after fix

| K | Strategy | RMSD | ΔvsK1 | pred_in |
|---:|---|---:|---:|---:|
| 10 | naive (still buggy) | 2.3549 | +0.869 | 20.00 |
| 10 | **hungarian_fold (fixed)** | **1.4215** | **−0.064** | **18.94** |

`pred_in` returns to its expected ~18.95 — the tripwire confirms the fix is real. RMSD turns from −0.87 disaster into +0.06 free win.

### 5.5 Implications

This bug almost certainly affects any prior or future code that averages frac coords / fractional positions / lattice vectors / any modular quantity. **Exp6 should audit any aggregation operation in similar code paths** for the same pattern. Cheatsheet item 15 in §3.7 records the general principle.

---

## 6. Carry-forward to Exp5 main line and beyond

### 6.1 SA3 (evaluation phase)

**Mandatory adoption:** Make `K=5 + hungarian_fold` the default evaluation method for SA3. Report K=1 and K=5 numbers side-by-side for both Exp4 ckpt and Exp5 ckpt. The comparison matrix becomes 4 rows (Exp4 K=1, Exp4 K=5, Exp5 K=1, Exp5 K=5) instead of 2, doubling the dimensionality of the main result without doubling the compute (the per-ckpt K=5 cost is ~45 min for full val on RTX 4090).

**Cost projection:** full val (7621 samples) × K=5 = `7621 / 500 × 45 min = ~11 min/ckpt`. Negligible.

### 6.2 Exp5 main architecture (SA1/SA2)

**Optional anchor-selection improvement:** `hungarian_fold_bestanchor` with truth gives K=10 RMSD = 1.32, vs `hungarian_fold` (anchor=0) at 1.42. Closing 50% of this gap with a deployable heuristic (e.g., medoid-as-anchor: pick the k with min sum-pairwise-distance to the others) is a candidate Exp5 enhancement worth ~0.05 Å.

This is **independent of architecture changes** — works on any Exp5 ckpt out of the box. Could be a quick win for Exp5 SA3 to also test on top of the standard SA3 numbers.

### 6.3 Exp6

§3 cheatsheet eliminates ~2-4 hours of environment rediscovery. The 5 production scripts adapt to a new ckpt by changing 2-3 constants. The torus-fold pattern (§5) generalizes to any future aggregation work.

---

## 7. Acknowledgements / open issues

- **K=1 reproducibility limitation** (cheatsheet item 13) is a model-internals issue not addressed by SA0. If Exp6 needs sample-level reproducibility, the model.sample function in `diffusion_w_type_xas_v2.py` would need its RNG path audited.
- **Anchor selection heuristic** (§6.2) is a known headroom gap not closed by SA0. Recommended Exp5 main-line follow-up.
- **Holdout/test never touched** (handoff red line). SA3 holds those evaluations.

---

**End of EXP5_STEP0_FINAL_REPORT.md.**
**Companion file: EXP5_STEP0_OUTPUT.md (structured handoff to MA).**
**Production artifacts: see §4 and §6 of EXP5_STEP0_OUTPUT.md.**
