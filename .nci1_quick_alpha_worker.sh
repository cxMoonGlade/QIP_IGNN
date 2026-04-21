#!/bin/bash
GPU="$1"
PROJECT_DIR="$2"
TASK_FILE="$3"
INDEX_FILE="$4"
LOCK_FILE="$5"
TOTAL="$6"

eval "$(conda shell.bash hook)"
conda activate qenv2

cd "$PROJECT_DIR"
count=0

get_next_task() {
    (
        flock -x 200
        idx=$(cat "$INDEX_FILE")
        if [ "$idx" -ge "$TOTAL" ]; then
            exit 1
        fi
        sed -n "$((idx+1))p" "$TASK_FILE"
        echo $((idx + 1)) > "$INDEX_FILE"
    ) 200>"$LOCK_FILE"
}

while true; do
    task=$(get_next_task) || break
    [ -z "$task" ] && break

    IFS='|' read -r exp_name dataset seed fold flags <<< "$task"
    count=$((count + 1))

    echo "[GPU$GPU #$count] $exp_name  $dataset  seed=$seed fold=$fold"

    CUDA_VISIBLE_DEVICES=$GPU python train.py         --dataset "$dataset"         --exp_name "$exp_name"         --seed "$seed"         --fold_idx "$fold"         $flags
done

echo ""
echo "======================================================"
echo " [GPU$GPU] DONE — completed $count tasks"
echo "======================================================"
