#!/bin/bash
# ======================================================================
# Classical ablation with attention pooling on PROTEINS
#
# Classical implicit backbone with no quantum module anywhere.
# This script matches the standard classical ablation as closely as possible,
# while replacing the original sum readout with attention pooling.
#
# Sweep:
#   exp_name = ablation_classical_attention
#   dataset  = PROTEINS
#   seeds    = 42, 123, 456
#   8-fold CV  (3 × 8 = 24 runs)
#
# Shared hyper-params follow the classical ablation protocol, except that
# --pooling attention is used instead of the original sum pooling.
#
# Usage:
#   bash scripts/run_classical_PROTEINS.sh
#   GPUS="0 1 2 3" SESSION=classical_PROTEINS_attention \
#     CONDA_ENV=qenv2 bash scripts/run_classical_PROTEINS.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

SEEDS="${SEEDS:-42 123 456}"
N_FOLDS="${N_FOLDS:-8}"
DATASET="PROTEINS"
EXP_NAME="ablation_classical_attention"

RESULT_DIR="${PROJECT_DIR}/results/${DATASET}/${EXP_NAME}/${DATASET}"
TMP_TASK_FILE=$(mktemp /tmp/qignn_PROTEINS_classical_attention_XXXX.txt)

cleanup() {
    rm -f "$TMP_TASK_FILE"
}
trap cleanup EXIT

mkdir -p "$RESULT_DIR"

SHARED_FLAGS="--implicit_global --min_encoder --pooling attention \
--hidden 64 --n_qubits 4 --circuit_reps 1 \
--kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
--batch_size 32 --lr 0.0001 --dropout 0.4 \
--use_layer_norm --weight_decay 1e-4 --epochs 200 --patience 999 \
--no_topo"

CLASSICAL_FLAGS="--no_quantum"

for seed in $SEEDS; do
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        pattern="${RESULT_DIR}/result_fold${fold}_seed${seed}_*.json"
        if compgen -G "$pattern" > /dev/null; then
            echo "Already finished: seed=${seed} fold=${fold}"
        else
            printf '%s/%s|%s|%s|%s|%s %s --select_by_loss --n_folds %s\n' \
                "$EXP_NAME" "$DATASET" "$DATASET" "$seed" "$fold" \
                "$SHARED_FLAGS" "$CLASSICAL_FLAGS" "$N_FOLDS" >> "$TMP_TASK_FILE"
            echo "Queued missing:  seed=${seed} fold=${fold}"
        fi
    done
done

if [ ! -s "$TMP_TASK_FILE" ]; then
    echo "Nothing to run. All expected results are present."
    exit 0
fi

echo
echo "Task file: $TMP_TASK_FILE"
echo "Queued jobs:"
wc -l "$TMP_TASK_FILE"
echo

export GPUS="${GPUS:-0}"
export SESSION="${SESSION:-classical_PROTEINS_attention}"
export CONDA_ENV="${CONDA_ENV:-qignn2-gpu}"
export TU_ONLY=1
export SKIP_GIN=1
export SKIP_EXISTING=1
export PREBUILT_TASK_FILE="$TMP_TASK_FILE"

bash "${PROJECT_DIR}/scripts/run_final_main.sh"
