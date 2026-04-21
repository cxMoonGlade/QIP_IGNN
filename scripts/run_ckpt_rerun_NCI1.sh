#!/bin/bash
# ======================================================================
# Checkpoint rerun for analysis (NCI1)
#
# Produces trained .pt checkpoints for the four analysis variants so that
# downstream Fisher / parameter-space PCA / solver-trajectory PCA scripts
# can load best_model_state directly from disk. Each run writes both a
# result JSON and (via --save_checkpoint) a ckpt payload under:
#
#   results/NCI1/ckpts_for_analysis/<VARIANT>/ckpt_fold0_seed{42,123,456}_*.pt
#
# Sweep:
#   variants = IN, SD, BD, classical_attention
#   seeds    = 42, 123, 456
#   fold     = 0 only (single-fold standard scope, 12 runs total)
#
# Shared hyper-params match the mainline analysis protocol (attention pool,
# min_encoder, implicit_global, torchdeq 300 iter, 200 epochs, select by
# validation loss). --no_topo is applied only to classical_attention so all
# three quantum variants share an identical backbone / topology environment.
#
# Usage:
#   bash scripts/run_ckpt_rerun_NCI1.sh
#   GPUS="0 1 2" SESSION=ckpt_rerun_NCI1 CONDA_ENV=qignn2-gpu \
#     bash scripts/run_ckpt_rerun_NCI1.sh
#
# Dry run (prints the generated task file, does not launch tmux):
#   DRY_RUN=1 bash scripts/run_ckpt_rerun_NCI1.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

SEEDS="${SEEDS:-42 123 456}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-10}"
DATASET="NCI1"

# Variants to run. Space-separated names; each must be resolved below.
VARIANTS="${VARIANTS:-IN SD BD classical_attention}"

TMP_TASK_FILE=$(mktemp /tmp/qignn_ckpt_rerun_NCI1_XXXX.txt)
cleanup() {
    if [ "${DRY_RUN:-0}" != "1" ]; then
        rm -f "$TMP_TASK_FILE"
    fi
}
trap cleanup EXIT

SHARED_FLAGS="--implicit_global --min_encoder --pooling attention \
--hidden 64 --n_qubits 4 --circuit_reps 1 \
--qi_n_qubits 4 --qi_circuit_reps 1 \
--kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
--batch_size 32 --lr 0.0001 --dropout 0.4 \
--use_layer_norm --weight_decay 1e-4 --epochs 200 --patience 999 \
--select_by_loss --save_checkpoint"

variant_flags() {
    case "$1" in
        IN)
            # Independent injection: per-node static quantum signal, topology enabled.
            echo "--q_ind_node --no_quantum"
            ;;
        SD)
            # State-dependent: in-loop quantum residual on Z (qi_direct), topology enabled.
            echo "--quantum_inside --qi_direct --no_quantum"
            ;;
        BD)
            # Backbone-dependent: in-loop quantum residual on h_theta(Z) (no qi_direct), topology enabled.
            echo "--quantum_inside --no_quantum"
            ;;
        classical_attention)
            # Classical implicit backbone, no quantum module, topology disabled.
            echo "--no_quantum --no_topo"
            ;;
        *)
            echo "Unknown variant: $1" >&2
            return 1
            ;;
    esac
}

for variant in $VARIANTS; do
    VFLAGS=$(variant_flags "$variant")
    EXP_NAME="ckpts_for_analysis/${variant}"
    RESULT_DIR="${PROJECT_DIR}/results/${DATASET}/${EXP_NAME}"
    mkdir -p "$RESULT_DIR"
    for seed in $SEEDS; do
        pattern="${RESULT_DIR}/ckpt_fold${FOLD}_seed${seed}_*.pt"
        if compgen -G "$pattern" > /dev/null; then
            echo "[${variant}] Already finished: seed=${seed} fold=${FOLD}"
            continue
        fi
        printf '%s|%s|%s|%s|%s %s --n_folds %s\n' \
            "$EXP_NAME" "$DATASET" "$seed" "$FOLD" \
            "$SHARED_FLAGS" "$VFLAGS" "$N_FOLDS" >> "$TMP_TASK_FILE"
        echo "[${variant}] Queued: seed=${seed} fold=${FOLD}"
    done
done

if [ ! -s "$TMP_TASK_FILE" ]; then
    echo "Nothing to run. All expected checkpoints are present."
    exit 0
fi

echo
echo "Task file: $TMP_TASK_FILE"
echo "Queued jobs: $(wc -l < "$TMP_TASK_FILE")"
echo "----- tasks -----"
cat "$TMP_TASK_FILE"
echo "-----------------"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 set; not launching. Task file preserved at: $TMP_TASK_FILE"
    trap - EXIT
    exit 0
fi

export GPUS="${GPUS:-0}"
export SESSION="${SESSION:-ckpt_rerun_NCI1}"
export CONDA_ENV="${CONDA_ENV:-qignn2-gpu}"
export PREBUILT_TASK_FILE="$TMP_TASK_FILE"

bash "${PROJECT_DIR}/scripts/run_final_main.sh"
