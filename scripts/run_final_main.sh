#!/bin/bash
# ======================================================================
# Final paper sweep runner
#
# Runs tasks formatted as:
#   exp_name|dataset|seed|fold|flags
#
# Typical usage:
#   PREBUILT_TASK_FILE=/tmp/tasks.txt GPUS="0 1 2 3" \
#   CONDA_ENV=qignn2-gpu bash scripts/run_final_main.sh
# ======================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

TASK_FILE="${PREBUILT_TASK_FILE:-${PROJECT_DIR}/.paper_tasks.txt}"
GPUS_STR="${GPUS:-0}"
CONDA_ENV_NAME="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SESSION_NAME="${SESSION:-qignn_final_main}"
COPIED_TASK_FILE="${RUN_FINAL_MAIN_COPIED_TASK_FILE:-0}"

if [ ! -f "$TASK_FILE" ]; then
    echo "Task file not found: $TASK_FILE" >&2
    exit 1
fi

cleanup_inner_task_file() {
    if [ "${RUN_FINAL_MAIN_INNER:-0}" = "1" ] && [ "$COPIED_TASK_FILE" = "1" ] && [ -f "$TASK_FILE" ]; then
        rm -f "$TASK_FILE"
    fi
}
trap cleanup_inner_task_file EXIT

if [ "${RUN_FINAL_MAIN_INNER:-0}" != "1" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required but not found in PATH." >&2
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session already exists: $SESSION_NAME" >&2
        echo "Attach with: tmux attach -t $SESSION_NAME" >&2
        exit 1
    fi

    detached_task_file=$(mktemp /tmp/run_final_main_tasks_XXXX.txt)
    cp "$TASK_FILE" "$detached_task_file"

    quoted_project_dir=$(printf '%q' "$PROJECT_DIR")
    quoted_task_file=$(printf '%q' "$detached_task_file")
    quoted_gpus=$(printf '%q' "$GPUS_STR")
    quoted_conda_env=$(printf '%q' "$CONDA_ENV_NAME")
    quoted_python_bin=$(printf '%q' "$PYTHON_BIN")
    quoted_session=$(printf '%q' "$SESSION_NAME")
    quoted_script=$(printf '%q' "$PROJECT_DIR/scripts/run_final_main.sh")
    runner_cmd="cd $quoted_project_dir && export RUN_FINAL_MAIN_INNER=1 RUN_FINAL_MAIN_COPIED_TASK_FILE=1 PREBUILT_TASK_FILE=$quoted_task_file GPUS=$quoted_gpus CONDA_ENV=$quoted_conda_env PYTHON_BIN=$quoted_python_bin SESSION=$quoted_session && bash $quoted_script"

    tmux new-session -d -s "$SESSION_NAME" "bash"
    tmux set-option -t "$SESSION_NAME" remain-on-exit on >/dev/null
    tmux send-keys -t "$SESSION_NAME" "$runner_cmd" C-m

    echo "Started tmux session: $SESSION_NAME"
    echo "Task file: $detached_task_file"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

mapfile -t GPU_LIST < <(printf '%s\n' $GPUS_STR)
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    GPU_LIST=("0")
fi

use_conda_run=0
CONDA_BIN=""
if [ -n "$CONDA_ENV_NAME" ]; then
    CONDA_BIN=$(command -v conda 2>/dev/null || true)
    if [ -n "$CONDA_BIN" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
        use_conda_run=1
    fi
fi

run_task() {
    local gpu="$1"
    local task_line="$2"

    local exp_name dataset seed fold flags
    IFS='|' read -r exp_name dataset seed fold flags <<< "$task_line"

    if [ -z "${exp_name:-}" ] || [ -z "${dataset:-}" ] || [ -z "${seed:-}" ] || [ -z "${fold:-}" ] || [ -z "${flags:-}" ]; then
        echo "Malformed task line: $task_line" >&2
        return 1
    fi

    read -r -a extra_args <<< "$flags"

    echo "[GPU $gpu] Starting: exp=$exp_name dataset=$dataset seed=$seed fold=$fold"
    if [ "$use_conda_run" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" \
            "$PYTHON_BIN" "$PROJECT_DIR/train.py" \
            --dataset "$dataset" \
            --exp_name "$exp_name" \
            --seed "$seed" \
            --fold_idx "$fold" \
            "${extra_args[@]}"
    else
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$PROJECT_DIR/train.py" \
            --dataset "$dataset" \
            --exp_name "$exp_name" \
            --seed "$seed" \
            --fold_idx "$fold" \
            "${extra_args[@]}"
    fi
    echo "[GPU $gpu] Finished: exp=$exp_name dataset=$dataset seed=$seed fold=$fold"
}

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPUS=()
declare -a AVAILABLE_GPUS=("${GPU_LIST[@]}")

reclaim_finished_jobs() {
    local new_pids=()
    local new_gpus=()
    local pid gpu status

    for idx in "${!ACTIVE_PIDS[@]}"; do
        pid="${ACTIVE_PIDS[$idx]}"
        gpu="${ACTIVE_GPUS[$idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_gpus+=("$gpu")
        else
            if wait "$pid"; then
                AVAILABLE_GPUS+=("$gpu")
            else
                status=$?
                echo "[GPU $gpu] Task failed with exit code $status" >&2
                exit "$status"
            fi
        fi
    done

    ACTIVE_PIDS=("${new_pids[@]}")
    ACTIVE_GPUS=("${new_gpus[@]}")
}

launch_job() {
    local gpu="$1"
    local task_line="$2"

    (
        run_task "$gpu" "$task_line"
    ) &

    ACTIVE_PIDS+=("$!")
    ACTIVE_GPUS+=("$gpu")
}

while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    case "$line" in
        \#*) continue ;;
    esac

    while [ "${#AVAILABLE_GPUS[@]}" -eq 0 ]; do
        sleep 1
        reclaim_finished_jobs
    done

    gpu="${AVAILABLE_GPUS[0]}"
    if [ "${#AVAILABLE_GPUS[@]}" -gt 1 ]; then
        AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
    else
        AVAILABLE_GPUS=()
    fi

    launch_job "$gpu" "$line"
    reclaim_finished_jobs
done < "$TASK_FILE"

while [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
    sleep 1
    reclaim_finished_jobs
done

echo "All tasks completed."
