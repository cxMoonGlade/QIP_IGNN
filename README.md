# Quantum Injection Pathways for Implicit Graph Neural Networks

Official code accompanying:

**Quantum Injection Pathways for Implicit Graph Neural Networks**  
Pengyuan Xu, Tristan Zaborniak, Felipe Rivera, Hausi A. Müller (University of Victoria)

This release contains the **training script**, **CLI and dataset construction**, and **`qignn` model code** used to report the main graph-classification experiments (independent, state-dependent, and backbone-dependent quantum injection in an implicit GNN, plus a classical implicit baseline). Datasets and run outputs are not distributed here.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: Python 3, PyTorch (≥2.0), PyTorch Geometric, NumPy, NetworkX, scikit-learn, tqdm, matplotlib, and [TorchDEQ](https://github.com/locuslab/torchdeq) for the implicit fixed-point solvers and backward pass used in training. The first run will download TU datasets into `data/`.

## Method vs. command-line flags

| Paper | `train.py` |
|--------|------------|
| Independent (IN) | `--q_ind_node` with `--no_quantum` (no encoder-side graph PQC; injection is the static per-node signal into the implicit map). |
| State-dependent (SD) | `--implicit_global --quantum_inside --qi_direct` (with `--no_quantum` on the encoder in the reported config). |
| Backbone-dependent (BD) | `--implicit_global --quantum_inside` and no `--qi_direct`. |
| Classical implicit baseline | Same stack without `--q_ind_node` and without `--quantum_inside`. |

Details of the operator appear in the paper; the implementation of \(\Phi_\Theta\) and the injection points is in `qignn/model.py` (class `TopoAwareQIGNN` and submodules). `model_factory.py` centralizes the argument parser, dataloading, and model construction; `train.py` runs the optimization loop and logging.

**Datasets in the paper:** NCI1, PROTEINS, and MUTAG. For a given pathway, NCI1 and PROTEINS use the **same** hyperparameters as in the MUTAG examples below except `--dataset` and the dataset segment in `--exp_name` (e.g. `ablation_q_ind_node/NCI1` instead of `ablation_q_ind_node/MUTAG`).

## Reference commands (MUTAG, fold 0, seed 42)


**IN**

```bash
python train.py --dataset MUTAG --data_dir data --exp_name "ablation_q_ind_node/MUTAG" \
  --hidden 64 --n_qubits 4 --circuit_reps 1 --min_encoder --no_quantum --use_layer_norm \
  --dropout 0.4 --pooling attention --max_cycle_length 20 \
  --implicit_global --kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
  --q_ind_node --lr 0.0001 --epochs 200 --batch_size 32 --weight_decay 0.0001 --scheduler cosine \
  --grad_clip 1.0 --patience 999 --select_by_loss --n_folds 8 --fold_idx 0 --seed 42
```

**SD**

```bash
python train.py --dataset MUTAG --data_dir data --exp_name "ablation_direct_q" \
  --hidden 64 --n_qubits 4 --circuit_reps 3 --min_encoder --no_quantum --use_layer_norm \
  --dropout 0.4 --pooling attention --max_cycle_length 20 --no_topo \
  --implicit_global --kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
  --quantum_inside --qi_n_qubits 4 --qi_circuit_reps 1 --qi_alpha 0.1 --qi_direct \
  --lr 0.0001 --epochs 200 --batch_size 32 --weight_decay 0.0001 --scheduler cosine \
  --grad_clip 1.0 --patience 999 --select_by_loss --n_folds 10 --fold_idx 0 --seed 42
```

**BD**

```bash
python train.py --dataset MUTAG --data_dir data --exp_name "BackBone_Dependent/MUTAG" \
  --hidden 64 --n_qubits 4 --circuit_reps 3 --min_encoder --no_quantum --use_layer_norm \
  --dropout 0.4 --pooling attention --max_cycle_length 20 --no_topo \
  --implicit_global --kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
  --quantum_inside --qi_n_qubits 4 --qi_circuit_reps 1 --qi_alpha 0.1 \
  --lr 0.0001 --epochs 200 --batch_size 32 --weight_decay 0.0001 --scheduler cosine \
  --grad_clip 1.0 --patience 999 --select_by_loss --n_folds 10 --fold_idx 0 --seed 42
```

**Classical**

```bash
python train.py --dataset MUTAG --data_dir data --exp_name "ablation_classical_attention/MUTAG" \
  --hidden 64 --n_qubits 4 --circuit_reps 1 --min_encoder --no_quantum --use_layer_norm \
  --dropout 0.4 --pooling attention --max_cycle_length 20 --no_topo \
  --implicit_global --kappa 0.8 --solver torchdeq --max_iter 300 --tol 1e-6 \
  --lr 0.0001 --epochs 200 --batch_size 32 --weight_decay 0.0001 --scheduler cosine \
  --grad_clip 1.0 --patience 999 --select_by_loss --n_folds 8 --fold_idx 0 --seed 42
```

**Shared training settings** (all four, unless noted above): `hidden=64`, `min_encoder`, `dropout=0.4`, `pooling=attention`, `use_layer_norm`, `implicit_global`, `kappa=0.8`, `solver=torchdeq`, `max_iter=300`, `lr=1e-4`, `epochs=200`, `batch_size=32`, `weight_decay=1e-4`, `scheduler=cosine`, `select_by_loss`, `grad_clip=1.0`, `patience=999`, GIN stack depth 5 and `jk_mode=sum` in `model_factory` defaults, no LQA, no FiLM, no `qi_topo` / `qi_classical`.

Use `--use_gin_splits` if you need the GIN-paper 10-fold protocol; the examples above use the same internal fold machinery as the stored paper runs with `use_gin_splits` off.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added here when the venue reference is final.

## Layout

```
train.py
model_factory.py
requirements.txt
qignn/
additional_figures_and_data/
```

## Additional figures and data

Some conference builds may omit long diagnostics from the main PDF. The folder [`additional_figures_and_data/`](additional_figures_and_data/) archives the **gradient-variance diagnostic figure** (`bp_main_3x2.pdf`) and **LaTeX snippets** matching the full paper draft (`gradient_landscape_diagnostic_from_paper.tex`, `broader_generalization_from_paper.tex`). See that folder’s [`README.md`](additional_figures_and_data/README.md) for details.
