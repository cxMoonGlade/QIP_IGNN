#!/bin/bash
GPU="$1"
PROJECT_DIR="$2"
TASK_FILE="$3"
INDEX_FILE="$4"
LOCK_FILE="$5"
TOTAL="$6"
IGNN_DIR="$7"
GIND_DIR="$8"
GIND_RUN_DIR="$9"
RESULTS_ROOT="${10}"

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

    IFS='|' read -r model dataset seed fold <<< "$task"
    count=$((count + 1))

    echo "[GPU$GPU #$count] $model $dataset seed=$seed fold=$fold"

    if [ "$model" = "IGNN" ]; then
        if [[ "$dataset" == ogbg-* ]]; then
            cd "$GIND_RUN_DIR"
            CUDA_VISIBLE_DEVICES=$GPU IGNN_DIR="$IGNN_DIR" python train_IGNN_ogb.py                 --dataset "$dataset"                 --seed "$seed"                 --fold_idx "$fold"                 --n_folds 10                 --device "$GPU"                 --epochs "100"                 --hidden 64                 --kappa 0.9                 --dropout 0.5                 --batch_size 64                 --lr 0.001                 --result_dir "$RESULTS_ROOT/IGNN_baseline"
        else
            cd "$IGNN_DIR"
            CUDA_VISIBLE_DEVICES=$GPU python train_IGNN_val.py                 --dataset "$dataset"                 --fold_idx "$fold"                 --seed "$seed"                 --device "$GPU"                 --epochs "200"                 --hidden 64                 --kappa 0.9                 --dropout 0.5                 --batch_size 64                 --lr 0.001                 --result_dir "$RESULTS_ROOT/IGNN_baseline"
        fi
    elif [ "$model" = "GIND" ]; then
        cd "$GIND_RUN_DIR"
        if [[ "$dataset" == ogbg-* ]]; then
            CUDA_VISIBLE_DEVICES=$GPU GIND_ROOT="$GIND_DIR" python train_GIND_ogb.py                 --dataset "$dataset"                 --seed "$seed"                 --fold_idx "$fold"                 --n_folds 10                 --device "$GPU"                 --epochs "100"                 --hidden 64                 --num_layers 3                 --alpha 0.5                 --dropout_exp 0.4                 --batch_size 64                 --lr 0.001                 --result_dir "$RESULTS_ROOT/GIND_baseline"
        else
            CUDA_VISIBLE_DEVICES=$GPU GIND_ROOT="$GIND_DIR" python train_GIND_val.py                 --dataset "$dataset"                 --fold_idx "$fold"                 --seed "$seed"                 --device "$GPU"                 --epochs "200"                 --hidden 64                 --num_layers 3                 --alpha 0.5                 --dropout_exp 0.4                 --batch_size 64                 --lr 0.001                 --result_dir "$RESULTS_ROOT/GIND_baseline"
        fi
    else
        echo "Unknown model: $model"
        exit 1
    fi
done

echo ""
echo "======================================================"
echo " [GPU$GPU] DONE — completed $count tasks"
echo "======================================================"
