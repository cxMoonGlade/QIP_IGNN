#!/bin/bash
# ======================================================================
# Q-ind-node ablation on NCI1
#
# Φ_ind-node(Z) = h_θ(Z) + Q_ξ(A, H)
# Per-node static quantum injection (Z-independent) into implicit core.
#
# Sweep:
#   exp_name = ablation_q_ind_node
#   dataset  = NCI1
#   seeds    = 42, 123, 456
#   8-fold CV  (3 × 8 = 24 runs)
#
# Shared hyper-params taken from ablation_external_q reference config
# (result_fold0_seed42_0323_2137.json), with --no_topo removed so that
# topology features are available for per-node quantum conditioning.
#
# Usage:
#   bash scripts/run_q_ind_node_NCI1.sh
#   GPUS="0 1 2 3" SESSION=q_ind_node_NCI1 \
#     CONDA_ENV=qenv2 bash scripts/run_q_ind_node_NCI1.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

SEEDS="${SEEDS:-42 123 456}"
N_FOLDS="${N_FOLDS:-8}"
DATASET="NCI1"
EXP_NAME="ablation_q_ind_node"

RESULT_DIR="${PROJECT_DIR}/results/${DATASET}/${EXP_NAME}/${DATASET}"
TMP_TASK_FILE=$(mktemp /tmp/qignn_NCI1_q_ind_node_XXXX.txt)

cleanup() {
    rm -f "$TMP_TASK_FILE"
}
trap cleanup EXIT

mkdir -p "$RESULT_DIR"

SHARED_FLAGS="--implicit_global --min_encoder --pooling attention \
--hidden 64 --n_qubits 4 --circuit_reps 1 \
--kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
--batch_size 32 --lr 0.0001 --dropout 0.4 \
--use_layer_norm --weight_decay 1e-4 --epochs 200 --patience 999"

Q_IND_NODE_FLAGS="--q_ind_node --no_quantum"

for seed in $SEEDS; do
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        pattern="${RESULT_DIR}/result_fold${fold}_seed${seed}_*.json"
        if compgen -G "$pattern" > /dev/null; then
            echo "Already finished: seed=${seed} fold=${fold}"
        else
            printf '%s/%s|%s|%s|%s|%s %s --select_by_loss --n_folds %s\n' \
                "$EXP_NAME" "$DATASET" "$DATASET" "$seed" "$fold" \
                "$SHARED_FLAGS" "$Q_IND_NODE_FLAGS" "$N_FOLDS" >> "$TMP_TASK_FILE"
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
export SESSION="${SESSION:-q_ind_node_NCI1}"
export CONDA_ENV="${CONDA_ENV:-qignn2-gpu}"
export TU_ONLY=1
export SKIP_GIN=1
export SKIP_EXISTING=1
export PREBUILT_TASK_FILE="$TMP_TASK_FILE"

bash "${PROJECT_DIR}/scripts/run_final_main.sh"
