#!/bin/bash
# run_pipeline.sh — Full MIA pipeline for Run_3_Full_MIMIR (single dataset)
# Usage:  bash run_pipeline.sh <DATASET> [START_FROM]
# Example: bash run_pipeline.sh arxiv
#          bash run_pipeline.sh github 4   (resume from stage 4)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET="${1:?ERROR: pass dataset name as first arg, e.g. bash run_pipeline.sh arxiv}"
START_FROM="${2:-1}"

# Source secrets if present (written by JarvisLabs deploy step)
[ -f ~/.env ] && source ~/.env

# Verify required env vars
: "${HF_TOKEN:?ERROR: HF_TOKEN not set. Export it or write to ~/.env}"

# Explicit wandb login so WANDB_API_KEY is always picked up
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login "$WANDB_API_KEY" --relogin 2>&1 | head -3 || true
fi

# ------------------------------------------------------------------
# SAMA repo setup (unchanged from Run_2 — dllm is still needed)
# ------------------------------------------------------------------
if [ -z "${SAMA_ROOT:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LOCAL_SAMA="$(realpath "$SCRIPT_DIR/../../../SAMA" 2>/dev/null || true)"

    if [ -d "$LOCAL_SAMA/attack" ]; then
        export SAMA_ROOT="$LOCAL_SAMA"
        echo "Using local SAMA at: $SAMA_ROOT"
    else
        if [ ! -d "/home/SAMA/attack" ]; then
            echo "Cloning SAMA repo (MIT licensed) to /home/SAMA..."
            rm -rf /home/SAMA
            git clone https://github.com/Stry233/SAMA.git /home/SAMA
        else
            echo "Using cached SAMA at: /home/SAMA"
        fi
        export SAMA_ROOT="/home/SAMA"
    fi
fi

# Install dllm without its conflicting deps (we only need the model architecture)
# dllm pins datasets==4.2.0 which breaks MIMIR loading; we use datasets<3.0 instead
python -c "import dllm" 2>/dev/null || { echo "Installing dllm (no-deps)..." && pip install --no-deps -e ./dllm -q; }

# Per-dataset output directories
mkdir -p "data/$DATASET" "results/$DATASET" "logs/$DATASET" "models/$DATASET"

echo ""
echo "=========================================="
echo " Run_3_Full_MIMIR MIA Pipeline"
echo " Dataset:   $DATASET"
echo " SAMA_ROOT: $SAMA_ROOT"
echo " Start from stage: $START_FROM"
echo "=========================================="
echo ""

if [ "$START_FROM" -le 1 ]; then
  echo "=== [1/8] prepare_data.py ==="
  python prepare_data.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/prepare.log"
fi

if [ "$START_FROM" -le 2 ]; then
  echo "=== [2/8] finetune.py ==="
  python finetune.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/finetune.log"
fi

if [ "$START_FROM" -le 3 ]; then
  echo "=== [3/8] verify_memorization.py ==="
  python verify_memorization.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/verify.log"
fi

if [ "$START_FROM" -le 4 ]; then
  echo "=== [4/8] run_sama.py ==="
  python run_sama.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/sama.log"
fi

if [ "$START_FROM" -le 5 ]; then
  echo "=== [5/8] run_attacks.py (Loss / Zlib / Ratio) ==="
  python run_attacks.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/attacks.log"
fi

if [ "$START_FROM" -le 6 ]; then
  echo "=== [6/8] run_signals.py ==="
  python run_signals.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/signals.log"
fi

if [ "$START_FROM" -le 7 ]; then
  echo "=== [7/8] train_classifier.py ==="
  python train_classifier.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/classifier.log"
fi

if [ "$START_FROM" -le 8 ]; then
  echo "=== [8/8] benchmark.py ==="
  python benchmark.py --dataset "$DATASET" \
    2>&1 | tee "logs/$DATASET/benchmark.log"
fi

echo ""
echo "Pipeline complete for dataset: $DATASET"
echo "Results in results/$DATASET/"
