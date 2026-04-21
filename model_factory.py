#!/usr/bin/env python3
"""
``model_factory`` — command-line, datasets, and :class:`qignn.model.TopoAwareQIGNN` construction.

This module matches the pre-training setup used by ``train.py`` (same defaults and
flags).  Flags related to the paper’s *quantum injection pathways* (independent
``q_ind_node``, in-loop ``quantum_inside``, placement ``qi_direct`` for
state- vs. backbone-dependent residuals, ``--qi_alpha`` (residual scale *α* in the paper), etc.) are
defined in :func:`setup_args` and passed through :func:`build_model`.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from qignn.model import TopoAwareQIGNN
from qignn.topology import precompute_topology_features


# =============================================================================
# Dataset helpers (moved verbatim from train.py; identical behaviour)
# =============================================================================

OGB_DATASETS = ('ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-moltoxcast',
                'ogbg-molsider', 'ogbg-molclintox')


def _is_ogb_dataset(name: str) -> bool:
    return name.startswith('ogbg-')


def load_ogb_dataset(name: str, data_dir: str = 'data'):
    """Load OGB graph property prediction dataset (scaffold split, binary/multiclass)."""
    from ogb.graphproppred import PygGraphPropPredDataset
    dataset = PygGraphPropPredDataset(name=name, root=data_dir)
    split_idx = dataset.get_idx_split()
    num_tasks = dataset.num_tasks
    num_classes = 1 if num_tasks == 1 else dataset.meta_info.get('num_classes', 2)
    return dataset, split_idx, num_classes


def load_tu_dataset(name: str, data_dir: str = 'data'):
    """Load TUDataset with degree-as-tag encoding when node features are absent."""
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root=data_dir, name=name, use_node_attr=True)

    if dataset[0].x is None:
        num_classes = dataset.num_classes
        all_degree_tags = []
        tag_set = set()
        for data in dataset:
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
            deg.scatter_add_(0, data.edge_index[0],
                             torch.ones(data.edge_index.size(1), dtype=torch.long))
            deg_list = deg.tolist()
            all_degree_tags.append(deg_list)
            tag_set.update(deg_list)

        tag_list = sorted(tag_set)
        max_deg = max(tag_list)
        n_unique = len(tag_list)

        MAX_FEAT_DIM = 256
        if n_unique <= MAX_FEAT_DIM:
            tag2index = {tag: idx for idx, tag in enumerate(tag_list)}
            num_features = n_unique
            print(f"  [Degree-as-tag] {num_features} unique degrees (max: {max_deg})")
        else:
            num_features = MAX_FEAT_DIM
            tag2index = {tag: min(int(tag / (max_deg + 1) * MAX_FEAT_DIM), MAX_FEAT_DIM - 1)
                         for tag in tag_list}
            print(f"  [Degree-as-tag] {n_unique} unique -> binned to {MAX_FEAT_DIM} dims")

        dataset_list = list(dataset)
        for i, data in enumerate(dataset_list):
            indices = [tag2index[d] for d in all_degree_tags[i]]
            data.x = torch.zeros(len(indices), num_features)
            data.x[range(len(indices)), indices] = 1.0
        dataset = dataset_list
    else:
        num_features = dataset[0].x.size(1)
        num_classes = dataset.num_classes

    return dataset, num_features, num_classes


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    n = len(dataset)
    indices = np.random.RandomState(seed).permutation(n)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        [dataset[i] for i in indices[:train_end]],
        [dataset[i] for i in indices[train_end:val_end]],
        [dataset[i] for i in indices[val_end:]],
    )


def kfold_split(dataset, n_folds=10, fold_idx=0, seed=0):
    """Stratified 10-fold split with val set, matching original GIN splitting."""
    from sklearn.model_selection import StratifiedKFold
    assert 0 <= fold_idx < n_folds

    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]

    train_labels = [labels[i] for i in train_idx]
    val_skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=seed)
    sub_splits = list(val_skf.split(np.zeros(len(train_idx)), train_labels))
    sub_train_idx, sub_val_idx = sub_splits[0]

    final_train_idx = train_idx[sub_train_idx]
    final_val_idx = train_idx[sub_val_idx]

    return (
        [dataset[int(i)] for i in final_train_idx],
        [dataset[int(i)] for i in final_val_idx],
        [dataset[int(i)] for i in test_idx],
    )


def load_gin_splits(dataset, dataset_name: str, fold_idx: int = 0):
    """GIN paper StratifiedKFold splits (10-fold, no validation, last epoch)."""
    from sklearn.model_selection import StratifiedKFold
    assert 0 <= fold_idx < 10
    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]
    train_data = [dataset[int(i)] for i in train_idx]
    test_data = [dataset[int(i)] for i in test_idx]
    print(f"  GIN splits fold {fold_idx+1}/10: Train {len(train_data)}, Test {len(test_data)}")
    return train_data, test_data


def add_topo_features_to_dataset(dataset, topo_features_list):
    for data, topo in zip(dataset, topo_features_list):
        data.combined_node_features = topo['combined_node_features']
        data.graph_cycle_features = topo['graph_cycle_features']
    return dataset


# =============================================================================
# Quantum Circuit Gate Depth Statistics (for logging only)
# =============================================================================

def compute_gate_depth(max_neighbors, n_qubits_per_neighbor, conv_layers):
    """Compute gate depth and gate count statistics for the quantum circuit."""
    n = max_neighbors * n_qubits_per_neighbor
    encoding_depth = 3
    encoding_1q_gates = n * 3

    current_qubits = n
    total_zz_gates = 0
    total_conv_1q_gates = 0
    conv_depth = 0
    layer_details = []

    for layer in range(conv_layers):
        n_zz = current_qubits - 1
        n_rot = current_qubits * 3
        zz_depth = 2 if current_qubits > 2 else (1 if current_qubits == 2 else 0)
        rot_depth = 3
        layer_details.append({
            'layer': layer, 'qubits_before_pool': current_qubits,
            'zz_gates': n_zz, 'rotation_gates': n_rot,
            'zz_depth': zz_depth, 'rotation_depth': rot_depth,
            'layer_depth': zz_depth + rot_depth,
        })
        total_zz_gates += n_zz
        total_conv_1q_gates += n_rot
        conv_depth += zz_depth + rot_depth
        current_qubits = max((current_qubits + 1) // 2, 1)

    total_1q = encoding_1q_gates + total_conv_1q_gates
    total_2q = total_zz_gates
    return {
        'total_qubits': n,
        'max_neighbors': max_neighbors,
        'n_qubits_per_neighbor': n_qubits_per_neighbor,
        'conv_layers': conv_layers,
        'state_vector_dim': 2 ** n,
        'final_qubits_after_pooling': current_qubits,
        'encoding_depth': encoding_depth,
        'total_depth': encoding_depth + conv_depth,
        'total_1q_gates': total_1q,
        'total_2q_gates': total_2q,
        'total_gates': total_1q + total_2q,
    }


# =============================================================================
# Argument parser
# =============================================================================

def setup_args() -> argparse.ArgumentParser:
    """Full CLI for ``train.py`` (and any tool that reuses the same model constructor).

    Returns an *un-parsed* parser. The ``--quantum_inside`` / ``--qi_direct`` /
    ``--q_ind_node`` group selects the IN / SD / BD configurations described in the
    paper; see README and :class:`qignn.model.TopoAwareQIGNN`.
    """
    parser = argparse.ArgumentParser(description='Topology-Aware QIGNN')

    # Dataset
    parser.add_argument('--dataset', type=str, default='NCI1',
                        choices=['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1',
                                 'IMDB-BINARY', 'IMDB-MULTI',
                                 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB',
                                 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp',
                                 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molsider', 'ogbg-molclintox'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--exp_name', type=str, default=None)

    # Model
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--circuit_reps', type=int, default=3)
    parser.add_argument('--n_encoder_layers', type=int, default=5)
    parser.add_argument('--simple_encoder', action='store_true',
                        help='Use simple MLP encoder instead of GIN (ablation)')
    parser.add_argument('--min_encoder', action='store_true',
                        help='Use minimal Linear encoder (IGNN-style, no message passing)')
    parser.add_argument('--lqa', action='store_true',
                        help='Use Local Quantum Aggregator (entanglement-based aggregation)')
    parser.add_argument('--lqa_max_neighbors', type=int, default=4,
                        help='Max neighbors for LQA sampling (default: 4)')
    parser.add_argument('--auto_lqa_neighbors', action='store_true',
                        help='Auto-set lqa_max_neighbors from dataset p95 degree')
    parser.add_argument('--lqa_qubits_per_neighbor', type=int, default=4,
                        help='Qubits per neighbor in LQA circuit (default: 4)')
    parser.add_argument('--lqa_conv_layers', type=int, default=2,
                        help='Conv+pool layers in LQA circuit (default: 2)')
    parser.add_argument('--jk_mode', type=str, default='sum', choices=['last', 'sum', 'concat'])
    parser.add_argument('--n_decoder_layers', type=int, default=0,
                        help='GNN decoder layers after implicit core (default: 0 = disabled)')
    parser.add_argument('--no_decoder', action='store_true',
                        help='Explicitly disable decoder (same as --n_decoder_layers 0)')
    parser.add_argument('--no_quantum', action='store_true')
    parser.add_argument('--use_film', action='store_true')
    parser.add_argument('--dynamic_film', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--use_train_bn', action='store_true')
    parser.add_argument('--drop_edge', type=float, default=0.0)
    parser.add_argument('--pooling', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'concat', 'attention'])

    # Topology
    parser.add_argument('--max_cycle_length', type=int, default=20)
    parser.add_argument('--topo_encoding', action='store_true')
    parser.add_argument('--topo_ising', action='store_true')
    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--no_topo', action='store_true')
    parser.add_argument('--no_q_inject', action='store_true',
                        help='Ablation: zero out Q injection into implicit core (test graph-level quantum broadcast)')
    parser.add_argument('--q_inj_node_cond', action='store_true',
                        help='Ablation: node-conditioned Q injection (gate*h instead of broadcast same vector)')
    parser.add_argument('--topo_drop_enc', type=float, default=0.0)
    parser.add_argument('--topo_drop_ising', type=float, default=0.0)
    parser.add_argument('--topo_drop_gate', type=float, default=0.0)

    # Implicit core (Z* = Phi(Z*); see paper: shared backbone and fixed-point solve)
    parser.add_argument('--implicit_global', action='store_true')
    parser.add_argument('--implicit_self_loops', action='store_true')
    parser.add_argument('--no_normalize_adj', action='store_true')
    parser.add_argument('--kappa', type=float, default=0.999)
    parser.add_argument('--solver', type=str, default='torchdeq',
                        choices=['simple', 'anderson', 'torchdeq', 'unroll'],
                        help="'unroll' = Picard with full unrolling (BPTT through solver "
                             "steps; high memory). 'torchdeq' = Anderson + IFT (training default).")
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--quantum_inside', action='store_true',
                        help='In-loop residual g_xi: SD/BD pathways (alpha * g on Z or h(Z))')
    parser.add_argument('--qi_n_qubits', type=int, default=4,
                        help='Qubits for quantum-inside circuit (default: 4)')
    parser.add_argument('--qi_circuit_reps', type=int, default=1,
                        help='Circuit reps for quantum-inside (default: 1)')
    parser.add_argument('--qi_alpha', type=float, default=0.1,
                        help='Residual scale for quantum-inside (default: 0.1)')
    parser.add_argument('--qi_direct', action='store_true',
                        help='State-dependent: g_xi(Z). If off (with --quantum_inside): g_xi(h_theta(Z)) (BD).')
    parser.add_argument('--qi_classical', action='store_true',
                        help='Classical residual: replace PQC with parameter-matched MLP')
    parser.add_argument('--qi_topo', action='store_true',
                        help='Use topology-conditioned quantum circuit inside iterations')
    parser.add_argument('--perm_invariant', action='store_true',
                        help='Permutation-invariant quantum params (shared across qubits/edges)')
    parser.add_argument('--q_ind_node', action='store_true',
                        help='Independent injection Q_IN(H, tau): static per-node signal into tanh (IN pathway).')
    parser.add_argument('--ablation_lg', action='store_true',
                        help='Deprecated: q_ln is now always Identity (no-op, kept for script compat)')
    parser.add_argument('--ignn_injection', action='store_true',
                        help='IGNN-style diffused injection: B = Omega @ injection @ A')

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--jac_reg', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (0=disabled, default: 1.0)')
    parser.add_argument('--iters_per_epoch', type=int, default=0,
                        help='Fixed iters per epoch with random sampling (0=use DataLoader, >0=GIN-style)')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--select_by_loss', action='store_true')
    parser.add_argument('--select_by_rocauc', action='store_true',
                        help='Select model by best validation ROC-AUC (for OGB binary tasks)')
    parser.add_argument('--use_last_epoch', action='store_true')
    parser.add_argument('--select_by_gap_sum', action='store_true')
    parser.add_argument('--max_selection_gap', type=float, default=0.03)
    parser.add_argument('--selection_warmup', type=int, default=30)

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=0)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--use_gin_splits', action='store_true')

    # Gap monitoring
    parser.add_argument('--max_gap', type=float, default=0.15)
    parser.add_argument('--gap_patience', type=int, default=5)
    parser.add_argument('--gap_warmup', type=int, default=10)

    # L_g tracking (optional)
    parser.add_argument('--track_L_g', type=int, nargs='?', default=0, const=20,
                        metavar='N',
                        help='Track L_g for first N epochs (default: 20 when flag used, 0=disabled)')
    parser.add_argument('--track_Q_stats', type=int, nargs='?', default=0, const=20,
                        metavar='N',
                        help='Track Q/B/WZA: first N, middle N, last N epochs (3N total); if N>=T/3, all epochs')

    # Checkpoint
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save best_model_state to {result_dir}/ckpt_fold{fold}_seed{seed}_{ts}.pt')

    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=None)

    return parser


# =============================================================================
# Dataset construction
# =============================================================================

def build_dataset(args, device: torch.device, *, verbose: bool = True) -> Dict:
    """Load dataset, topology features, splits, and build DataLoaders.

    Mutates ``args`` when ``--auto_lqa_neighbors`` is set (matches original
    train.py semantics). Returns a dict with the following keys:

        train_loader, val_loader, test_loader
        num_features, num_classes
        topo_features, dataset_stats
        atom_encoder, ogb_evaluator
        is_ogb, is_ogb_binary, use_gin_splits
    """
    is_ogb = _is_ogb_dataset(args.dataset)
    atom_encoder: Optional[nn.Module] = None
    ogb_evaluator = None

    if is_ogb:
        if verbose:
            print("\nLoading OGB dataset...")
        dataset, split_idx, num_classes = load_ogb_dataset(args.dataset, args.data_dir)
        dataset_list = list(dataset)
        if args.n_folds > 0:
            train_data, val_data, test_data = kfold_split(
                dataset_list, n_folds=args.n_folds, fold_idx=args.fold_idx, seed=args.seed)
            if verbose:
                print(f"  Graphs: {len(dataset_list)}; stratified {args.n_folds}-fold CV "
                      f"(fold {args.fold_idx}): Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        else:
            train_data = [dataset[int(i)] for i in split_idx['train']]
            val_data = [dataset[int(i)] for i in split_idx['valid']]
            test_data = [dataset[int(i)] for i in split_idx['test']]
            if verbose:
                print(f"  Graphs: {len(dataset)}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        num_features = args.hidden  # AtomEncoder projects to hidden_dim
        from ogb.graphproppred.mol_encoder import AtomEncoder
        atom_encoder = nn.Sequential(AtomEncoder(emb_dim=args.hidden)).to(device)
        from ogb.graphproppred import Evaluator
        ogb_evaluator = Evaluator(name=args.dataset)
        if verbose:
            print(f"  Task: binary (ROC-AUC), AtomEncoder -> {num_features}d")
        is_ogb_binary = (num_classes == 1)
    else:
        dataset, num_features, num_classes = load_tu_dataset(args.dataset, args.data_dir)
        if verbose:
            print(f"  Graphs: {len(dataset)}, Features: {num_features}, Classes: {num_classes}")
        is_ogb_binary = False

    # Precompute topology features (OGB: skip; TU: optional)
    dataset_stats = None
    if is_ogb:
        if verbose:
            print("\nTopology features DISABLED (OGB molecular dataset)")
        topo_features = None
    elif not args.no_topo:
        if verbose:
            print("\nPrecomputing topology features...")
        topo_features, dataset_stats = precompute_topology_features(
            dataset, max_cycle_length=args.max_cycle_length,
            use_structural=True, verbose=verbose, return_stats=True)
        dataset = add_topo_features_to_dataset(list(dataset), topo_features)
    else:
        if verbose:
            print("\nTopology features DISABLED (baseline mode)")
        topo_features = None

    # Auto-set max_neighbors for LQA
    if args.lqa and dataset_stats is not None and args.auto_lqa_neighbors:
        MAX_TOTAL_QUBITS = 12
        MAX_QUBITS_PER_NEIGHBOR = 2
        p95 = dataset_stats['p95_degree']
        desired_neighbors = min(p95, 6)
        qpn = min(args.lqa_qubits_per_neighbor, MAX_QUBITS_PER_NEIGHBOR)
        if desired_neighbors * qpn <= MAX_TOTAL_QUBITS:
            args.lqa_max_neighbors = desired_neighbors
            args.lqa_qubits_per_neighbor = qpn
        else:
            qpn = 1
            args.lqa_max_neighbors = min(desired_neighbors, MAX_TOTAL_QUBITS)
            args.lqa_qubits_per_neighbor = qpn
        total_q = args.lqa_max_neighbors * args.lqa_qubits_per_neighbor
        if verbose:
            print(f"\n  [Auto LQA] max_neighbors={args.lqa_max_neighbors}, qpn={args.lqa_qubits_per_neighbor}, total={total_q} qubits")

    # Split selection (OGB already handled; TU: gin splits / kfold / random)
    use_gin_splits = False
    if not is_ogb:
        use_gin_splits = args.use_gin_splits
        if use_gin_splits:
            train_data, test_data = load_gin_splits(dataset, args.dataset, args.fold_idx)
            val_data = test_data
        elif args.n_folds > 0:
            train_data, val_data, test_data = kfold_split(
                dataset, n_folds=args.n_folds, fold_idx=args.fold_idx, seed=args.seed)
            if verbose:
                print(f"  {args.n_folds}-Fold CV fold {args.fold_idx}: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        else:
            train_data, val_data, test_data = split_dataset(dataset, seed=args.seed)
            if verbose:
                print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_features': num_features,
        'num_classes': num_classes,
        'topo_features': topo_features,
        'dataset_stats': dataset_stats,
        'atom_encoder': atom_encoder,
        'is_ogb': is_ogb,
        'is_ogb_binary': is_ogb_binary,
        'ogb_evaluator': ogb_evaluator,
        'use_gin_splits': use_gin_splits,
    }


# =============================================================================
# Model construction
# =============================================================================

def _resolve_topo_flags(args) -> Tuple[bool, bool, bool]:
    """Reproduce train.py's topo flag resolution (L721-L726)."""
    use_topo_encoding = args.topo_encoding and not args.no_topo
    use_topo_ising = args.topo_ising and not args.no_topo
    use_gate = args.use_gate and not args.no_topo
    if not args.no_topo and not (args.topo_encoding or args.topo_ising or args.use_gate):
        use_topo_encoding = use_topo_ising = use_gate = True
    return use_topo_encoding, use_topo_ising, use_gate


def build_model(
    args,
    num_features: int,
    num_classes: int,
    device: torch.device,
    *,
    verbose: bool = True,
) -> Tuple[TopoAwareQIGNN, Optional[Dict]]:
    """Construct TopoAwareQIGNN from args. Returns (model, gate_stats).

    ``gate_stats`` is the dict returned by ``compute_gate_depth`` when
    ``args.lqa`` is True, else None. Matches train.py L720-L842 behaviour.
    """
    use_topo_encoding, use_topo_ising, use_gate = _resolve_topo_flags(args)

    if verbose:
        print(f"\nTopology Conditioning:")
        print(f"  Encoding modulation: {use_topo_encoding}")
        print(f"  Ising modulation: {use_topo_ising}")
        print(f"  Competitive gate: {use_gate}")
        if args.topo_drop_enc > 0 or args.topo_drop_ising > 0 or args.topo_drop_gate > 0:
            print(f"\nStochastic Topo Conditioning:")
            print(f"  Drop encoding: {args.topo_drop_enc:.0%}")
            print(f"  Drop Ising:    {args.topo_drop_ising:.0%}")
            print(f"  Drop gate:     {args.topo_drop_gate:.0%}")
        print(f"\nPooling: {args.pooling}")
        if args.drop_edge > 0:
            print(f"DropEdge: {args.drop_edge:.0%} edges dropped during training")
        print(f"\nGlobal Layer: {'IMPLICIT (DEQ)' if args.implicit_global else 'EXPLICIT (feedforward)'}")
        if args.implicit_global:
            print(f"  kappa={args.kappa}, solver={args.solver}, max_iter={args.max_iter}")

    use_decoder = not args.no_decoder and args.n_decoder_layers > 0
    use_quantum = not args.no_quantum

    model = TopoAwareQIGNN(
        in_features=num_features,
        hidden_dim=args.hidden,
        num_classes=num_classes,
        n_qubits=args.n_qubits,
        circuit_reps=args.circuit_reps,
        n_encoder_layers=args.n_encoder_layers,
        jk_mode=args.jk_mode,
        dropout=args.dropout,
        max_cycle_length=args.max_cycle_length,
        use_topo_encoding=use_topo_encoding,
        use_topo_ising=use_topo_ising,
        use_competitive_gate=use_gate,
        implicit_global=args.implicit_global,
        implicit_self_loops=args.implicit_self_loops,
        normalize_implicit_adj=not args.no_normalize_adj,
        kappa=args.kappa,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        use_decoder=use_decoder,
        n_decoder_layers=args.n_decoder_layers,
        pooling=args.pooling,
        use_quantum=use_quantum,
        use_film=args.use_film,
        dynamic_film=args.dynamic_film,
        simple_encoder=args.simple_encoder,
        min_encoder=args.min_encoder,
        lqa=args.lqa,
        lqa_max_neighbors=args.lqa_max_neighbors,
        lqa_qubits_per_neighbor=args.lqa_qubits_per_neighbor,
        lqa_conv_layers=args.lqa_conv_layers,
        topo_drop_enc=args.topo_drop_enc,
        topo_drop_ising=args.topo_drop_ising,
        topo_drop_gate=args.topo_drop_gate,
        jac_reg=args.jac_reg,
        use_layer_norm=args.use_layer_norm,
        quantum_inside=args.quantum_inside,
        qi_n_qubits=args.qi_n_qubits,
        qi_circuit_reps=args.qi_circuit_reps,
        qi_alpha=args.qi_alpha,
        qi_topo=args.qi_topo,
        perm_invariant=args.perm_invariant,
        no_q_inject=args.no_q_inject,
        q_inj_node_cond=args.q_inj_node_cond,
        ignn_injection=args.ignn_injection,
        qi_direct=args.qi_direct,
        qi_classical=args.qi_classical,
        q_ind_node=args.q_ind_node,
    ).to(device)

    gate_stats = None
    if args.lqa:
        gate_stats = compute_gate_depth(
            args.lqa_max_neighbors, args.lqa_qubits_per_neighbor, args.lqa_conv_layers)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {n_params:,} parameters")
        param_breakdown: Dict[str, int] = {}
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                param_breakdown[name] = module_params
        if param_breakdown:
            print("  Parameter breakdown:")
            for name, count in sorted(param_breakdown.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count:,} ({100.0 * count / n_params:.1f}%)")
        if gate_stats is not None:
            print(f"  LQA Circuit: {gate_stats['total_qubits']} qubits, depth {gate_stats['total_depth']}, "
                  f"{gate_stats['total_gates']} gates")
        if args.lqa:
            print(f"  Encoder: LQA (max_neighbors={args.lqa_max_neighbors}, "
                  f"qpn={args.lqa_qubits_per_neighbor}, conv_layers={args.lqa_conv_layers})")
        elif args.min_encoder:
            print(f"  Encoder: MinEncoder (single Linear projection, no message passing)")
        elif args.simple_encoder:
            print(f"  Encoder: Simple MLP (no GNN)")
        else:
            print(f"  Encoder: GIN ({args.n_encoder_layers} layers, JK={args.jk_mode})")
        if args.use_layer_norm:
            print(f"  Normalization: LayerNorm")
        if args.no_q_inject and args.implicit_global:
            print(f"  Ablation: no_q_inject (Q=0, no external graph-level quantum broadcast)")
        if args.q_inj_node_cond and args.implicit_global and not args.no_q_inject:
            print(f"  Ablation: q_inj_node_cond (node-conditioned Q gate, not broadcast same vector)")
        if args.q_ind_node and args.implicit_global:
            print(f"  Injection: Q-ind-node (per-node static quantum signal, Z-independent)")
        if args.ignn_injection and args.implicit_global:
            print(f"  Injection: IGNN-style diffused (B = Omega @ injection @ A)")

    return model, gate_stats
