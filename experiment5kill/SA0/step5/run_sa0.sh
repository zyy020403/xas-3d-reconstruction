#!/bin/bash
# ========================================================================
# run_sa0.sh — Exp5 SA0 driver
# ========================================================================
# Sets up mlff env + PYTHONPATH + GPU pinning, then runs subcommand stages.
#
# Usage:
#   bash run_sa0.sh check      # env_smoke (~3-5 min) — ALWAYS run first
#   bash run_sa0.sh subset     # make_subset (instant)
#   bash run_sa0.sh sample     # multisample K=10 (~1.5h GPU)
#   bash run_sa0.sh agg        # multisample_aggregate (~1 min)
#   bash run_sa0.sh all        # check → subset → sample → agg
#
# Override GPU:
#   CUDA_VISIBLE_DEVICES=0 bash run_sa0.sh sample
# ========================================================================
set -e

# ── paths ──
SA0_ROOT=/home/tcat/diffcsp_exp5/sa0
SCRIPTS=$SA0_ROOT/scripts
RESULTS=$SA0_ROOT/results
LOGS=$SA0_ROOT/logs
mkdir -p "$RESULTS" "$LOGS"

# ── env ──
# Pick up conda; mlff is at /home/tcat/conda_envs/mlff (per envfind §2)
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
else
    echo "❌ /opt/miniconda3/etc/profile.d/conda.sh not found; cannot activate conda"
    exit 1
fi

# Activate mlff (env path: /home/tcat/conda_envs/mlff — outside default envs/)
conda activate /home/tcat/conda_envs/mlff || conda activate mlff

# PYTHONPATH per Exp4 step5_1 header
export PYTHONPATH="/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2"

# GPU pin: default to GPU 1 (leave GPU 0 for SA1); allow override via env var
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

echo "==================================================================="
echo "Exp5 SA0 driver"
echo "  active env       : $(which python)"
echo "  PYTHONPATH       : $PYTHONPATH"
echo "  CUDA_VISIBLE_DEV : $CUDA_VISIBLE_DEVICES"
echo "  SA0_ROOT         : $SA0_ROOT"
echo "==================================================================="

cd "$SCRIPTS"

MODE=${1:-check}

run_check() {
    echo ""
    echo "=== [check] env_smoke (~3-5 min) ==="
    python env_smoke.py 2>&1 | tee "$LOGS/env_smoke.log"
}
run_subset() {
    echo ""
    echo "=== [subset] make_subset (instant) ==="
    python make_subset.py 2>&1 | tee "$LOGS/make_subset.log"
}
run_sample() {
    echo ""
    echo "=== [sample] multisample K=10 (~1.5h) ==="
    if [ ! -f "$RESULTS/sa0_subset_500.csv" ]; then
        echo "❌ $RESULTS/sa0_subset_500.csv not found — run 'subset' first"
        exit 1
    fi
    python multisample.py --K 10 \
        --subset_csv "$RESULTS/sa0_subset_500.csv" \
        --out_pt "$RESULTS/samples_raw_K10.pt" \
        2>&1 | tee "$LOGS/multisample_K10.log"
}
run_agg() {
    echo ""
    echo "=== [agg] multisample_aggregate (~1 min) ==="
    if [ ! -f "$RESULTS/samples_raw_K10.pt" ]; then
        echo "❌ $RESULTS/samples_raw_K10.pt not found — run 'sample' first"
        exit 1
    fi
    python multisample_aggregate.py \
        --samples_pt "$RESULTS/samples_raw_K10.pt" \
        --out_dir "$RESULTS" \
        --K_values 1 5 10 \
        --strategies naive hungarian \
        2>&1 | tee "$LOGS/multisample_aggregate.log"
}

case "$MODE" in
    check)  run_check ;;
    subset) run_subset ;;
    sample) run_sample ;;
    agg)    run_agg ;;
    all)    run_check; run_subset; run_sample; run_agg ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_sa0.sh {check|subset|sample|agg|all}"
        exit 1
        ;;
esac

echo ""
echo "==================================================================="
echo "[$MODE] DONE."
echo "==================================================================="
