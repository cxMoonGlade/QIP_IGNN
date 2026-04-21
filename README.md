# QIGNN: Quantum-Conditioned Implicit Graph Neural Network

A topology-aware graph neural network that combines **local quantum aggregation** (LQA), **topology-conditioned quantum circuits**, and **implicit equilibrium layers** for graph classification.

## Architecture

```
Input Graph
    |
    v
[LQA Encoder]                  Local quantum aggregation via entanglement
    |                           GIN-style: h = MLP((1+eps)*h + QuantumAgg(neighbors))
    v
[Topology Quantum Layer]        Cycle-basis features condition Ising parameters
    |                           Competitive gating (PCB-GNN inspired)
    v
[Implicit Equilibrium Core]     IGNN-style fixed-point: Z* = phi(Z*; X, A, Q)
    |                           CONE contraction guarantee, TorchDEQ solver
    v
[Graph Pooling] -> [Classifier] Sum pooling -> MLP head
```

### Key Components

- **LQA (Local Quantum Aggregator)**: Replaces classical GIN aggregation with a quantum circuit that entangles neighbor features across qubits. The entanglement *is* the aggregation.
- **Topology Quantum Layer**: Extracts cycle-basis features from graph topology and uses them to modulate quantum circuit parameters (encoding, Ising strengths, competitive gating).
- **Implicit Equilibrium**: DEQ-style fixed-point layer with IGNN contraction projection. Supports TorchDEQ (Anderson + IFT) or simple damped iteration.

## Project Structure

```
qignn/
├── __init__.py              # Package exports
├── ansatz.py                # Quantum gates, state ops, Deep XYZ circuit, qubit topologies
├── quantum_torch.py         # TorchQuantumLayer, TopoAwareQuantumLayer
├── lqa.py                   # Local Quantum Aggregator (entanglement-based GIN)
├── topology.py              # Cycle basis extraction, topological features
└── model.py                 # GINEncoder, BatchedImplicitCore, TopoAwareQIGNN
train.py                     # Training / evaluation / experiment logging
visualize_circuit.py         # Circuit diagram generation for paper figures
scripts/                     # Experiment run scripts (10-fold CV across 5 GPUs)
results/                     # Experiment result JSONs
```

## Setup

```bash
pip install -r requirements.txt
```

**Requirements**: PyTorch >= 2.0, PyG >= 2.4, NetworkX, scikit-learn, tqdm, matplotlib. Optional: [TorchDEQ](https://github.com/locuslab/torchdeq) for production-grade implicit differentiation.

## Quick Start

```bash
# Quick smoke test (5 epochs, 3 datasets)
bash scripts/run_quick_test.sh

# Single run: NCI1, fold 0, full training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset NCI1 --use_gin_splits --fold_idx 0 \
    --lqa --auto_lqa_neighbors \
    --lqa_qubits_per_neighbor 2 --lqa_conv_layers 1 \
    --n_encoder_layers 2 --hidden 64 \
    --epochs 350 --batch_size 32 --lr 0.001 \
    --implicit_global --max_iter 30 \
    --use_film --dropout 0.4 --use_layer_norm \
    --weight_decay 1e-5
```

## Full Experiments (10-Fold CV)

Each script runs 10 folds sequentially on one GPU:

```bash
bash scripts/run_datasets_gpu1.sh &   # GPU 1: MUTAG + PTC_MR
bash scripts/run_datasets_gpu2.sh &   # GPU 2: PROTEINS + IMDB-BINARY
bash scripts/run_datasets_gpu3.sh &   # GPU 3: IMDB-MULTI + COLLAB
bash scripts/run_datasets_gpu4.sh &   # GPU 4: NCI1
bash scripts/run_datasets_gpu5.sh &   # GPU 5: REDDIT-BINARY
```

Results are saved as JSON to `results/<dataset>/<exp_name>/`.

## Supported Datasets

MUTAG, PTC_MR, PROTEINS, NCI1, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, REDDIT-MULTI-5K, COLLAB

All use GIN-style 10-fold stratified cross-validation (`--use_gin_splits`).

## Key Arguments

### Encoder
| Argument | Description |
|----------|-------------|
| `--lqa` | Use Local Quantum Aggregator (entanglement-based) |
| `--auto_lqa_neighbors` | Auto-set max neighbors from dataset p95 degree |
| `--lqa_max_neighbors K` | Max neighbors to sample per node (default: 4) |
| `--lqa_qubits_per_neighbor Q` | Qubits per neighbor in LQA circuit (default: 4) |
| `--lqa_conv_layers L` | Conv+pool layers in LQA circuit (default: 2) |
| `--n_encoder_layers N` | GIN/LQA layers (default: 5) |

### Topology Quantum Layer
| Argument | Description |
|----------|-------------|
| `--no_quantum` | Disable topology quantum layer (GIN/LQA-only baseline) |
| `--n_qubits Q` | Qubits in topology quantum circuit (default: 4) |
| `--circuit_reps R` | Circuit repetitions (default: 3) |
| `--topo_encoding` | Modulate data encoding with cycle features |
| `--topo_ising` | Modulate Ising strengths with cycle features |
| `--use_gate` | Competitive gating (PCB-GNN style) |
| `--no_topo` | Disable all topology features |

### Implicit Core
| Argument | Description |
|----------|-------------|
| `--implicit_global` | Enable implicit equilibrium layer |
| `--kappa K` | Contraction factor (default: 0.999) |
| `--solver {simple,torchdeq}` | Fixed-point solver (default: torchdeq) |
| `--max_iter N` | Max solver iterations (default: 50) |
| `--use_film` | FiLM conditioning: Q(X) = gamma(X) * q + beta(X) |
| `--dynamic_film` | Dynamic FiLM: Q(X,Z) depends on iterate |

### Training
| Argument | Description |
|----------|-------------|
| `--epochs N` | Training epochs (default: 100) |
| `--lr F` | Learning rate (default: 0.001) |
| `--dropout F` | Dropout rate (default: 0.1) |
| `--weight_decay F` | Weight decay (default: 1e-4) |
| `--drop_edge F` | DropEdge augmentation rate (default: 0) |
| `--use_gin_splits` | Use GIN paper 10-fold splits |
| `--jac_reg F` | Jacobian regularization weight (default: 0) |

## Circuit Visualization

```bash
python visualize_circuit.py --max_neighbors 4 --n_qubits_per_neighbor 2 --conv_layers 1
```

Generates PDF/PNG circuit diagrams showing the LQA encoding, entanglement, and pooling layers.
