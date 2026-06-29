# XAS-3D-Reconstruction

**Direct 3D Local Atomic Structure Reconstruction from X-ray Absorption Spectra via Diffusion-Based Generative Modeling**

Yanyu Zhu · Supervisor: Ziyun Wang · Department of Chemistry, University of Auckland

---

## Overview

Conventional XAS analysis recovers only scalar structural descriptors — coordination number, mean bond length, oxidation state. This work addresses the problem at its full dimensionality: given a single K-edge XAS spectrum, reconstruct the complete three-dimensional local coordination environment of the absorbing atom, including the Cartesian coordinates and elemental identities of all 20 nearest neighbours.

A conditional diffusion model based on the [DiffCSP](https://github.com/jiaor17/DiffCSP) framework is trained on **75,637 K-edge XAS spectra spanning 88 absorbing elements**, sourced from the Materials Project FEFF database. To our knowledge, this is the first machine-learning framework to solve the full 3D XAS inverse problem across a broad multi-element chemical space.

---

## Key Results

| Metric | Baseline (Exp4, unconstrained) | Constrained model (Exp5′) |
|---|---|---|
| Physical gate pass rate (min_d ≥ 1.5 Å) | 1.85–2.68% | **64%** (+25×) |
| Shell-1 mean bond distance (Pearson *r*) | 0.995–0.996 (gate-passed) | — |
| Shell-2 mean bond distance (Pearson *r*) | 0.990–0.997 (gate-passed) | — |
| RMSD vs random baseline | 1.487 Å vs 2.32 Å (−36%) | — |
| Collapse rate | ~97% | **0%** |

Enforcing the corrected virtual lattice (L = 6 Å → L = 20 Å) together with physics-inspired auxiliary losses raises the physical validity gate pass rate from ~2% to 64%, a 25× improvement. Among gate-passing predictions, shell mean bond distances are recovered with near-perfect linear accuracy (Pearson *r* = 0.990–0.999).

---

## Repository Structure

```
xas-3d-reconstruction/
├── data/                        # Dataset and preprocessing
│   ├── shell_boundaries.pkl     # Per-sample FEFF shell boundary annotations
│   └── data_inventory_v2.csv    # Sample inventory with center element metadata
│
├── code/
│   ├── step1/                   # Data cleaning and split construction
│   ├── step2/                   # Spectrum preprocessing (XANES window, chi normalisation)
│   ├── step3/                   # Dataset class and model definition
│   │   ├── xas_local_dataset_v2.py
│   │   └── diffusion_w_type_xas.py   # Main model (SpectrumEncoder + DiffCSP decoder)
│   ├── step4/                   # Training scripts
│   │   └── train.py
│   ├── step5/                   # Evaluation
│   │   ├── step5_2_compute_metrics.py   # RMSD, TypeAcc, pred_in_cutoff
│   │   └── step5_3_composite_score.py   # Physical gate + 7-component composite score
│   └── step6/                   # Visualisation
│       └── step6_visualize.py
│
├── checkpoints/                 # Model checkpoints (see below)
├── figures/                     # Reproduced figures from the thesis
├── requirements.txt
└── README.md
```

---

## Method

### Input

| Channel | Dimension | Description |
|---|---|---|
| XANES window | 150 pts | μ(E), E₀−50 to E₀+150 eV, z-normalised |
| EXAFS chi(k) | 200 pts | k¹χ(k), z-normalised |
| FEFF prior features | 73–74 dim | Physics-derived features from FEFF calculation |
| Center element embedding | 16 dim | Learnable embedding over 95 elements |

### Output

20-atom point cloud: fractional coordinates ∈ [−0.5, 0.5]³ and element types (88-class), representing the nearest neighbours of the absorbing atom in a virtual lattice of L = 20 Å.

### Physical Constraint Losses

Three auxiliary losses are added to the standard diffusion score-matching objective:

- **Pairwise minimum distance penalty** — `ReLU(1.5 − d)²` over all atom pairs; enforces d ≥ 1.5 Å (physical lower bound for EXAFS-relevant bond lengths). Self-starting from random initialisation.
- **Shell distance loss** — penalises deviation of predicted radial distributions from per-sample FEFF-derived shell boundaries.
- **Density regularisation** — discourages coordinate collapse toward the origin without imposing a structural prior.

### Virtual Lattice Fix (L = 6 Å → L = 20 Å)

With L = 6 Å, the half-box boundary (3 Å) is smaller than the neighbour search cutoff (~10 Å). Minimum-image folding then maps atoms at large separations to spuriously small distances, contaminating 64% of training samples. Setting L = 20 Å (so L/2 = 10 Å ≥ cutoff) eliminates this artefact entirely.

### Evaluation Protocol

**Physical gate:** a predicted structure passes if all pairwise interatomic distances ≥ 1.5 Å. Gate-failing samples receive composite score = 0.

**Composite score** (gate-passing samples only): weighted sum of six components —

| Component | Weight | Tolerance |
|---|---|---|
| Shell-1 coordination number | 0.20 | ±1.5 atoms |
| Shell-1 mean bond distance | 0.20 | ±0.2 Å |
| Shell-1 element composition (CNO-equiv.) | 0.20 | Multiset overlap |
| Shell-2 coordination number | 0.10 | ±3 atoms |
| Shell-2 mean bond distance | 0.10 | ±0.2 Å |
| Shell-2 element composition (CNO-equiv.) | 0.10 | Multiset overlap |

CNO equivalence (C/N/O treated as the same token) reflects the near-identical EXAFS scattering amplitudes of these elements at K-edge energies — a physical limitation, not a simplification.

---

## Checkpoints

| Model | Description | Gate pass rate |
|---|---|---|
| `Exp4_best_epoch366` | 88-element baseline, no physical constraints, L = 6 Å | 1.85–2.68% |
| `Exp5prime_epoch169` ⭐ | Full constraint suite, L = 20 Å — **recommended** | **64%** |

Download from [Releases](https://github.com/zyy020403/xas-3d-reconstruction/releases).

---

## Dataset

K-edge XAS spectra computed via FEFF from Materials Project crystal structures.

| Split | Samples |
|---|---|
| Train | 60,507 |
| Validation | 7,624 (7,621 effective after silent drop) |
| Test | 4,481 |
| Holdout | 3,025 |
| **Total** | **75,637** |

The holdout set was queried exactly once at the conclusion of model evaluation and was not used during training or hyperparameter selection.

Raw spectra and shell boundary annotations: see `data/` or download from the [Releases](https://github.com/zyy020403/xas-3d-reconstruction/releases) page.

---

## Installation

```bash
git clone https://github.com/zyy020403/xas-3d-reconstruction.git
cd xas-3d-reconstruction
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, PyTorch Lightning, PyTorch Geometric, pymatgen, numpy, scipy, pandas, matplotlib.

---

## Reproducing Results

**Training (Exp5′, recommended):**
```bash
python code/step4/train.py \
    --config configs/exp5prime.yaml \
    --data_root data/ \
    --checkpoint_dir checkpoints/
```

**Evaluation:**
```bash
python code/step5/step5_3_composite_score.py \
    --predictions checkpoints/Exp5prime_epoch169/predictions_holdout.pt \
    --shell_boundaries data/shell_boundaries.pkl \
    --split holdout
```

**Visualisation (reproduces thesis figures):**
```bash
python code/step6/step6_visualize.py --only 1 2 2b 3 4 5
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@thesis{zhu2026xas3d,
  title   = {Direct 3D Local Atomic Structure Reconstruction from
             X-ray Absorption Spectra via Diffusion-Based Generative Modeling},
  author  = {Zhu, Yanyu},
  year    = {2026},
  school  = {University of Auckland},
  type    = {Research Thesis},
  url     = {https://github.com/zyy020403/xas-3d-reconstruction}
}
```

This work builds on [DiffCSP](https://github.com/jiaor17/DiffCSP) (Jiao et al., NeurIPS 2023) and the [Materials Project FEFF database](https://materialsproject.org) (Mathew et al., Scientific Data 2018).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The XAS dataset is derived from Materials Project FEFF calculations and is subject to the [Materials Project terms of use](https://materialsproject.org/about/terms).
