#!/bin/bash
# run_shadow_launcher.sh — Launch 6 parallel JarvisLabs L4 instances for Run4.
#
# One L4 per domain. Each instance runs all 5 phases of run_shadow_mia.py:
#   Phase 1: load MIMIR ngram_13_0.2 surrogate data     (~3 min)
#   Phase 2+3: fine-tune + extract 3 shadow models       (~6.5 hrs)
#   Phase 4: classifiers, SHAP, distribution analysis    (~15 min)
#
# Total wall time per domain: ~6.5 hrs. All 6 run in parallel.
#
# Prerequisites:
#   jl CLI installed and authenticated  (pip install jlabs-cli && jl login)
#   HF_TOKEN and WANDB_API_KEY exported in your shell
#   Run3 results (X.pt, y.pt, classifier_results.pt, ablation_results.pt)
#     available locally at: ../Run_3_Full_MIMIR/results/{domain}/
#
# Usage:
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=wandb_...
#   bash run_shadow_launcher.sh                   # launch all 6 MIMIR domains
#   bash run_shadow_launcher.sh hackernews        # launch one domain (testing)
#   START_PHASE=4 bash run_shadow_launcher.sh     # resume from Phase 4
#
# Instance naming:  mia-run4-{domain}
# GPU:              L4 (24 GB VRAM)
# Disk:             60 GB
#
# After completion, download and aggregate:
#   bash run_shadow_launcher.sh --download
#   python code/aggregate_domains.py

set -euo pipefail

: "${HF_TOKEN:?ERROR: export HF_TOKEN first}"
: "${WANDB_API_KEY:?ERROR: export WANDB_API_KEY first}"

START_PHASE="${START_PHASE:-1}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN4_DIR="$(dirname "$CODE_DIR")"            # Run_4_Shadow_MIA/
RUN3_DIR="$(dirname "$RUN4_DIR")/Run_3_Full_MIMIR"

# ----------------------------------------------------------------
# Download-only mode: pull results after all runs finish
# ----------------------------------------------------------------
if [ "${1:-}" = "--download" ]; then
    echo "=== Downloading Run4 results from all instances ==="
    jl list
    MIMIR_DOMAINS=(arxiv github hackernews pubmed_central wikipedia pile_cc)
    for DOMAIN in "${MIMIR_DOMAINS[@]}"; do
        INSTANCE_NAME="mia-run4-${DOMAIN}"
        # Get integer ID for this instance
        INSTANCE_ID=$(jl list --json 2>/dev/null | \
            python3 -c "
import sys, json
machines = json.load(sys.stdin)
for m in machines:
    if m.get('name') == '${INSTANCE_NAME}':
        print(m.get('machine_id', m.get('id', '')))
        break
" 2>/dev/null || echo "")
        if [ -z "$INSTANCE_ID" ]; then
            echo "  [WARN] Cannot find instance $INSTANCE_NAME — skipping"
            continue
        fi
        echo "--- Downloading $INSTANCE_NAME (ID=$INSTANCE_ID) ---"
        mkdir -p "$RUN4_DIR/shadow_results/$DOMAIN"
        jl download "$INSTANCE_ID" "/home/run4/Run_4_Shadow_MIA/shadow_results/$DOMAIN" \
            "$RUN4_DIR/shadow_results/$DOMAIN" --recursive || \
            echo "  [WARN] Download failed for $INSTANCE_NAME"
    done
    echo ""
    echo "=== Download complete. Run aggregation: ==="
    echo "  python code/aggregate_domains.py"
    exit 0
fi

# ----------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------
if [ -n "${1:-}" ] && [ "${1}" != "--download" ]; then
    DATASETS=("$1")
else
    DATASETS=(arxiv github hackernews pubmed_central wikipedia pile_cc)
fi

echo "=== Run_4_Shadow_MIA — JarvisLabs Launcher ==="
echo "Datasets:    ${DATASETS[*]}"
echo "Start phase: $START_PHASE"
echo "Run4 dir:    $RUN4_DIR"
echo "Run3 dir:    $RUN3_DIR"
echo ""

# ----------------------------------------------------------------
# Check Run3 results exist locally before uploading
# ----------------------------------------------------------------
echo "Checking Run3 results are available locally..."
for DOMAIN in "${DATASETS[@]}"; do
    for f in X.pt y.pt classifier_results.pt ablation_results.pt; do
        p="$RUN3_DIR/results/$DOMAIN/$f"
        if [ ! -f "$p" ]; then
            echo "  [ERROR] Missing Run3 artifact: $p"
            echo "  Download Run3 results before launching Run4."
            exit 1
        fi
    done
done
echo "  All Run3 artifacts found."
echo ""

# ----------------------------------------------------------------
# Launch one instance per domain
# ----------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    INSTANCE_NAME="mia-run4-${DATASET}"
    echo "--- Launching: $INSTANCE_NAME ---"

    # ----------------------------------------------------------------
    # Create L4 instance (24 GB VRAM, sufficient for Qwen3-0.6B)
    # jl create returns JSON; parse integer ID
    # ----------------------------------------------------------------
    CREATE_JSON=$(jl create \
        --name      "$INSTANCE_NAME" \
        --gpu       "L4" \
        --num-gpus  1 \
        --storage   60 \
        --template  "pytorch" \
        --yes \
        --json 2>/dev/null || echo "")

    if [ -n "$CREATE_JSON" ]; then
        INSTANCE_ID=$(echo "$CREATE_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Handle both {machine_id: ...} and [{machine_id: ...}] shapes
if isinstance(data, list):
    data = data[0]
print(data.get('machine_id', data.get('id', '')))
" 2>/dev/null || echo "")
    fi

    # If create failed or instance already exists, look up by name
    if [ -z "${INSTANCE_ID:-}" ]; then
        echo "  [INFO] Create returned no ID — looking up existing instance..."
        INSTANCE_ID=$(jl list --json 2>/dev/null | python3 -c "
import sys, json
try:
    machines = json.load(sys.stdin)
    for m in machines:
        if m.get('name') == '$INSTANCE_NAME':
            print(m.get('machine_id', m.get('id', '')))
            break
except: pass
" 2>/dev/null || echo "")
    fi

    if [ -z "${INSTANCE_ID:-}" ]; then
        echo "  [ERROR] Cannot find or create $INSTANCE_NAME — skipping."
        continue
    fi
    echo "  Instance: $INSTANCE_NAME (ID=$INSTANCE_ID)"

    # Wait for instance to be ready (up to 3 min)
    echo "  Waiting for instance to be running..."
    for i in $(seq 1 36); do
        STATUS=$(jl get "$INSTANCE_ID" --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if isinstance(d, list): d = d[0]
    # JarvisLabs uses 'status' field
    print(d.get('status', d.get('state', 'unknown')))
except: print('unknown')
" 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "running" ]; then
            echo "  Instance is running."
            break
        fi
        echo "  Status: $STATUS — waiting 5s..."
        sleep 5
    done

    # ----------------------------------------------------------------
    # Write .env for secrets and upload it
    # Not under ~/run4/ — that directory may be wiped on re-upload
    # ----------------------------------------------------------------
    TMP_ENV=$(mktemp /tmp/run4_env_XXXX.sh)
    cat > "$TMP_ENV" <<EOF
export HF_TOKEN='${HF_TOKEN}'
export WANDB_API_KEY='${WANDB_API_KEY}'
EOF
    jl upload "$INSTANCE_ID" "$TMP_ENV" /home/.env || true
    rm -f "$TMP_ENV"

    # ----------------------------------------------------------------
    # Upload Run4 code directory
    # Note: jl upload <id> <local_dir> <remote_dest> places local_dir
    #       as a child of remote_dest, so Run_4_Shadow_MIA ends up at
    #       /home/run4/Run_4_Shadow_MIA/  — that's our REMOTE_RUN4
    # ----------------------------------------------------------------
    REMOTE_RUN4="/home/run4/Run_4_Shadow_MIA"
    echo "  Uploading Run4 code..."
    jl upload "$INSTANCE_ID" "$RUN4_DIR" /home/run4 || true

    # ----------------------------------------------------------------
    # Upload Run3 domain results into a sibling dir so setup script
    # can find them via ../run3_results (relative to code/)
    # ----------------------------------------------------------------
    echo "  Uploading Run3 results for $DATASET..."
    jl exec "$INSTANCE_ID" -- bash -c "mkdir -p ${REMOTE_RUN4}/run3_results/$DATASET" || true
    for f in X.pt y.pt classifier_results.pt ablation_results.pt signal_names.pt; do
        SRC="$RUN3_DIR/results/$DATASET/$f"
        if [ -f "$SRC" ]; then
            jl upload "$INSTANCE_ID" "$SRC" \
                "${REMOTE_RUN4}/run3_results/$DATASET/$f" || true
        fi
    done

    # ----------------------------------------------------------------
    # Create output dirs on instance
    # ----------------------------------------------------------------
    jl exec "$INSTANCE_ID" -- bash -c \
        "mkdir -p ${REMOTE_RUN4}/shadow_results/$DATASET" || true

    # ----------------------------------------------------------------
    # Run setup + pipeline in background via nohup
    # Logs go to shadow_results/{domain}/run4_full.log
    # ----------------------------------------------------------------
    LOG_PATH="${REMOTE_RUN4}/shadow_results/$DATASET/run4_full.log"
    CMD="nohup bash ${REMOTE_RUN4}/code/run_shadow_setup.sh '$DATASET' '$START_PHASE' > $LOG_PATH 2>&1 &"
    jl exec "$INSTANCE_ID" -- bash -c "source /home/.env; $CMD" || \
        echo "  [WARN] exec failed for $INSTANCE_NAME — check 'jl ssh $INSTANCE_ID'"

    echo "  Launched $INSTANCE_NAME in background."
    echo ""
done

echo "=== All instances launched ==="
echo ""
echo "Monitor:"
echo "  jl list"
echo "  jl exec <ID> -- tail -100 /home/run4/Run_4_Shadow_MIA/shadow_results/<DOMAIN>/run4_full.log"
echo "  wandb: project da5001-mia, runs named run4-<domain>-shadow"
echo ""
echo "When all domains are done:"
echo "  bash run_shadow_launcher.sh --download"
echo "  python code/aggregate_domains.py"
