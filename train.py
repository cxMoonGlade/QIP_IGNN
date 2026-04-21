#!/usr/bin/env python3
"""
Training entry point for *Quantum Injection Pathways for Implicit Graph Neural Networks*.

Uses :func:`model_factory.setup_args`, :func:`model_factory.build_dataset`, and
:func:`model_factory.build_model` so the CLI and model match the paper’s reported
configurations.  See the repository README for example commands (IN/SD/BD and
classical implicit baseline on TU datasets).
"""

import os
import argparse


def _early_gpu_select():
    """Parse --gpu argument early and set CUDA_VISIBLE_DEVICES before torch import."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"[GPU] Selected GPU {args.gpu} via CUDA_VISIBLE_DEVICES")
    return args.gpu


_selected_gpu = _early_gpu_select()

# Fix for PyTorch 2.6+ / OGB compatibility (weights_only)
import torch as _torch
_orig_load = _torch.load
_torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

import json
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool

from qignn.topology import precompute_topology_features
from qignn.model import TopoAwareQIGNN

from model_factory import (
    setup_args,
    build_dataset,
    build_model,
    compute_gate_depth,
    _is_ogb_dataset,
    _resolve_topo_flags,
    load_ogb_dataset,
    load_tu_dataset,
    split_dataset,
    kfold_split,
    load_gin_splits,
    add_topo_features_to_dataset,
    OGB_DATASETS,
)


# =============================================================================
# DropEdge - Graph Data Augmentation
# =============================================================================

def drop_edge(edge_index: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Randomly drop edges (DropEdge, ICLR 2020)."""
    if p <= 0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, mask]


def apply_drop_edge_to_batch(batch: Batch, p: float) -> Batch:
    if p <= 0:
        return batch
    new_batch = batch.clone()
    new_batch.edge_index = drop_edge(batch.edge_index, p)
    return new_batch


# Dataset loaders, splitters, and OGB helpers are imported from model_factory.
# They remain re-exported for back-compat with any tooling that imports them
# via ``from train import ...``.


# =============================================================================
# Training and Evaluation
# =============================================================================

def _make_batch_iterator(loader, device, iters_per_epoch, epoch, show_progress):
    """Yield batches: either full DataLoader or GIN-style random sampling."""
    if iters_per_epoch > 0:
        all_data = list(loader.dataset)
        bs = loader.batch_size
        rng = tqdm(range(iters_per_epoch), desc=f"epoch {epoch+1}",
                   unit="batch", leave=True) if show_progress else range(iters_per_epoch)
        for _ in rng:
            idx = np.random.permutation(len(all_data))[:bs]
            yield Batch.from_data_list([all_data[i] for i in idx]).to(device)
    else:
        source = tqdm(loader, desc=f"epoch {epoch+1}",
                      unit="batch", leave=True) if show_progress else loader
        for batch in source:
            yield batch.to(device)


def train_epoch(
    model, loader, topo_features_list, optimizer, device, num_classes,
    label_smoothing=0.1, drop_edge_rate=0.0, jac_reg=0.0,
    epoch=0, show_progress=True, iters_per_epoch=0, grad_clip=0.0,
    track_L_g_epochs: int = 0,
    track_Q_stats_epochs: int = 0,
    total_epochs: int = 0,
    atom_encoder=None,
    is_ogb_binary=False,
):
    model.train()
    total_loss = 0
    total_jac_reg = 0
    correct = 0
    total = 0
    convergence_stats = {
        'local': {'n_iter': [], 'residual': [], 'converged': 0, 'count': 0},
        'global': {'n_iter': [], 'residual': [], 'converged': 0, 'count': 0, 'L_g': [],
                   'Q_mean': [], 'Q_std': [], 'Q_abs_mean': [], 'Q_max_abs': [], 'Q_norm': [],
                   'B_mean': [], 'B_std': [], 'B_abs_mean': [], 'B_max_abs': [], 'B_norm': [],
                   'B_pre_norm': [], 'B_post_norm': [],
                   'WZA_abs_mean': [], 'WZA_norm': [],
                   'Q_B_ratio_norm': [], 'Q_WZA_ratio_norm': [], 'Q_B_ratio_abs': [], 'Q_WZA_ratio_abs': []},
    }
    compute_L_g_this_epoch = track_L_g_epochs > 0 and (epoch < track_L_g_epochs)
    # Q stats: first N, middle N, last N epochs (3N total); if N >= total/3, all epochs
    N = track_Q_stats_epochs
    T = total_epochs if total_epochs > 0 else 9999
    if N <= 0:
        compute_Q_stats_this_epoch = False
    elif N * 3 >= T:
        compute_Q_stats_this_epoch = True
    else:
        is_first = epoch < N
        is_last = epoch >= T - N
        mid_start = (T - N) // 2
        is_mid = mid_start <= epoch < mid_start + N
        compute_Q_stats_this_epoch = is_first or is_mid or is_last

    for batch in _make_batch_iterator(loader, device, iters_per_epoch, epoch, show_progress):
        if drop_edge_rate > 0:
            batch = apply_drop_edge_to_batch(batch, drop_edge_rate)
        
        batch_topo = None
        if hasattr(batch, 'combined_node_features') and batch.combined_node_features is not None:
            node_feat = batch.combined_node_features
            graph_feat = batch.graph_cycle_features
            feat_dim = graph_feat.shape[0] // batch.num_graphs
            graph_feat = graph_feat.view(batch.num_graphs, feat_dim)
            batch_topo = {
                'combined_node_features': node_feat,
                'graph_cycle_features': graph_feat,
            }
        
        optimizer.zero_grad()
        compute_L_g = compute_L_g_this_epoch
        compute_Q_stats = compute_Q_stats_this_epoch
        if atom_encoder is not None:
            batch.x = atom_encoder(batch.x)
        out, diagnostics = model(batch, batch_topo, compute_L_g=compute_L_g,
                                 compute_Q_stats=compute_Q_stats)
        if is_ogb_binary:
            is_labeled = batch.y == batch.y
            ce_loss = F.binary_cross_entropy_with_logits(
                out[is_labeled].view(-1), batch.y[is_labeled].view(-1).float())
        else:
            ce_loss = F.cross_entropy(out, batch.y, label_smoothing=label_smoothing)

        jac_reg_loss = torch.tensor(0.0)
        if diagnostics and 'global' in diagnostics:
            jac_reg_loss = diagnostics['global'].get('jac_reg', torch.tensor(0.0))
        loss = ce_loss + jac_reg * jac_reg_loss
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        if jac_reg > 0:
            total_jac_reg += jac_reg_loss.item() * batch.num_graphs
        
        if diagnostics:
            for layer_name in ['local', 'global']:
                if layer_name in diagnostics and diagnostics[layer_name]:
                    info = diagnostics[layer_name]
                    convergence_stats[layer_name]['count'] += 1
                    if 'n_iter' in info:
                        convergence_stats[layer_name]['n_iter'].append(info['n_iter'])
                    if 'residual' in info:
                        convergence_stats[layer_name]['residual'].append(info['residual'])
                    if info.get('converged', False):
                        convergence_stats[layer_name]['converged'] += 1
                    if 'L_g' in info:
                        convergence_stats[layer_name]['L_g'].append(info['L_g'])
                    for k in ['Q_mean', 'Q_std', 'Q_abs_mean', 'Q_max_abs', 'Q_norm',
                              'B_mean', 'B_std', 'B_abs_mean', 'B_max_abs', 'B_norm',
                              'B_pre_norm', 'B_post_norm',
                              'WZA_abs_mean', 'WZA_norm',
                              'Q_B_ratio_norm', 'Q_WZA_ratio_norm', 'Q_B_ratio_abs', 'Q_WZA_ratio_abs']:
                        if k in info:
                            convergence_stats[layer_name][k].append(info[k])
        
        total_loss += ce_loss.item() * batch.num_graphs
        if is_ogb_binary:
            pred = (out.view(-1).sigmoid() > 0.5).long()
            y_flat = batch.y.view(-1)
            is_l = y_flat == y_flat
            correct += ((pred[is_l] == y_flat[is_l].long())).sum().item()
        else:
            pred = out.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    
    result = {'loss': total_loss / total, 'accuracy': correct / total * 100}
    if jac_reg > 0:
        result['jac_reg'] = total_jac_reg / total

    for layer_name in ['local', 'global']:
        stats = convergence_stats[layer_name]
        if stats['count'] > 0:
            result[f'{layer_name}_avg_iter'] = (
                sum(stats['n_iter']) / len(stats['n_iter']) if stats['n_iter'] else 0)
            result[f'{layer_name}_avg_residual'] = (
                sum(stats['residual']) / len(stats['residual']) if stats['residual'] else 0)
            result[f'{layer_name}_converged_pct'] = stats['converged'] / stats['count'] * 100
            if stats.get('L_g'):
                result[f'{layer_name}_L_g'] = sum(stats['L_g']) / len(stats['L_g'])
            for k in ['Q_mean', 'Q_std', 'Q_abs_mean', 'Q_max_abs', 'Q_norm',
                      'B_mean', 'B_std', 'B_abs_mean', 'B_max_abs', 'B_norm',
                      'B_pre_norm', 'B_post_norm',
                      'WZA_abs_mean', 'WZA_norm',
                      'Q_B_ratio_norm', 'Q_WZA_ratio_norm', 'Q_B_ratio_abs', 'Q_WZA_ratio_abs']:
                if stats.get(k):
                    result[f'{layer_name}_{k}'] = sum(stats[k]) / len(stats[k])
    
    return result


@torch.no_grad()
def evaluate(model, loader, device, debug_preds=False, use_train_bn=False,
             atom_encoder=None, is_ogb_binary=False, ogb_evaluator=None):
    if use_train_bn:
        model.train()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_y_pred = []
    all_y_true = []
    
    for batch in loader:
        batch = batch.to(device)
        if atom_encoder is not None:
            batch.x = atom_encoder(batch.x)
        batch_topo = None
        if hasattr(batch, 'combined_node_features') and batch.combined_node_features is not None:
            node_feat = batch.combined_node_features
            graph_feat = batch.graph_cycle_features
            feat_dim = graph_feat.shape[0] // batch.num_graphs
            graph_feat = graph_feat.view(batch.num_graphs, feat_dim)
            batch_topo = {
                'combined_node_features': node_feat,
                'graph_cycle_features': graph_feat,
            }
        
        out, _ = model(batch, batch_topo)
        if is_ogb_binary:
            is_labeled = batch.y == batch.y
            loss = F.binary_cross_entropy_with_logits(
                out[is_labeled].view(-1), batch.y[is_labeled].view(-1).float())
            pred = (out.view(-1).sigmoid() > 0.5).long()
            y_flat = batch.y.view(-1)
            is_l = y_flat == y_flat
            correct += (pred[is_l] == y_flat[is_l].long()).sum().item()
            all_y_pred.append(out.view(-1, 1).cpu())
            all_y_true.append(batch.y.view(-1, 1).cpu())
        else:
            loss = F.cross_entropy(out, batch.y)
            pred = out.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs
        all_preds.extend(pred.view(-1).cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())
    
    if debug_preds:
        from collections import Counter
        print(f"    [DEBUG] Pred distribution: {dict(Counter(all_preds))}")
        print(f"    [DEBUG] Label distribution: {dict(Counter(all_labels))}")

    result = {'loss': total_loss / total, 'accuracy': correct / total * 100}
    if is_ogb_binary and ogb_evaluator is not None and all_y_pred:
        y_true = torch.cat(all_y_true, dim=0).numpy()
        y_pred = torch.cat(all_y_pred, dim=0).numpy()
        result['rocauc'] = ogb_evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['rocauc']
    return result


# add_topo_features_to_dataset is imported from model_factory above.


# =============================================================================
# Main
# =============================================================================

def main():
    parser = setup_args()
    args = parser.parse_args()

    # Encoder mutual exclusivity
    encoder_flags = sum([args.lqa, args.min_encoder, args.simple_encoder])
    if encoder_flags > 1:
        raise ValueError(
            "Choose only one of --lqa, --min_encoder, --simple_encoder")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 70)
    print("Topology-Aware QIGNN")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")

    # Dataset + model via factored helpers (identical behaviour to previous inline blocks).
    dataset_info = build_dataset(args, device, verbose=True)
    train_loader = dataset_info['train_loader']
    val_loader = dataset_info['val_loader']
    test_loader = dataset_info['test_loader']
    num_features = dataset_info['num_features']
    num_classes = dataset_info['num_classes']
    topo_features = dataset_info['topo_features']
    atom_encoder = dataset_info['atom_encoder']
    ogb_evaluator = dataset_info['ogb_evaluator']
    is_ogb_binary = dataset_info['is_ogb_binary']
    use_gin_splits = dataset_info['use_gin_splits']

    model, gate_stats = build_model(args, num_features, num_classes, device, verbose=True)
    n_params = sum(p.numel() for p in model.parameters())

    # Recompute topo flags for result JSON reporting (same resolution as inside build_model).
    use_topo_encoding, use_topo_ising, use_gate = _resolve_topo_flags(args)
    
    # Optimizer with selective weight decay (include atom_encoder for OGB)
    decay_params, no_decay_params = [], []
    no_decay_names = []
    params_to_optimize = list(model.parameters())
    if atom_encoder is not None:
        params_to_optimize += list(atom_encoder.parameters())
        n_params += sum(p.numel() for p in atom_encoder.parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (
            name.endswith(".bias")
            or ".norm" in name or "norms." in name or ".bn" in name
            or name.startswith("quantum.")              # external quantum module
            or name.startswith("quantum_node.")         # ind-node quantum module
            or name.startswith("implicit_core.qc_inside.")
            or "quantum_aggs" in name                   # LQA
            or "conv_params" in name
            or name.endswith(".eps") or ".eps" in name
        )
        if is_no_decay:
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    
    if atom_encoder is not None:
        for param in atom_encoder.parameters():
            decay_params.append(param)
    if args.weight_decay > 0:
        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=args.lr)
    else:
        all_params = decay_params + no_decay_params
        optimizer = torch.optim.Adam(all_params, lr=args.lr)
    
    print(f"\nOptimizer: AdamW (selective weight decay)")
    print(f"  weight_decay={args.weight_decay}: {len(decay_params)} groups ({sum(p.numel() for p in decay_params):,} params)")
    print(f"  weight_decay=0: {len(no_decay_params)} groups ({sum(p.numel() for p in no_decay_params):,} params)")

    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        print(f"LR Scheduler: StepLR (step=50, gamma=0.5)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"LR Scheduler: CosineAnnealing (T_max={args.epochs})")
    
    if args.jac_reg > 0:
        print(f"Jacobian Regularization: weight={args.jac_reg}")
    if args.track_L_g > 0:
        print(f"L_g tracking: enabled (first {args.track_L_g} epochs, every batch, 1000 power iters)")
    if args.track_Q_stats > 0:
        N = args.track_Q_stats
        T = args.epochs
        if N * 3 >= T:
            print(f"Q/B stats tracking: all {T} epochs (N={N} >= T/3)")
        else:
            mid = (T - N) // 2
            print(f"Q/B stats tracking: first {N}, middle {N} (ep{mid+1}-{mid+N}), last {N} epochs")

    # Training loop
    if use_gin_splits:
        args.use_last_epoch = True

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 70)
    if not use_gin_splits:
        val_col = 'Val AUC' if is_ogb_binary else 'Val Acc'
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {val_col:>7} | {'Time':>6}")
        print("-" * 70)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    best_val_rocauc = 0.0
    best_model_state = None
    best_val_epoch = 0
    patience_counter = 0
    gap_patience_counter = 0
    history = []
    best_gap_sum = 0
    best_gap_sum_epoch = 0
    best_gap_sum_state = None
    best_gap_sum_val_acc = 0
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, topo_features, optimizer, device, num_classes,
            label_smoothing=args.label_smoothing, drop_edge_rate=args.drop_edge,
            jac_reg=args.jac_reg, epoch=epoch,
            iters_per_epoch=args.iters_per_epoch,
            grad_clip=args.grad_clip,
            track_L_g_epochs=args.track_L_g,
            track_Q_stats_epochs=args.track_Q_stats,
            total_epochs=args.epochs,
            atom_encoder=atom_encoder,
            is_ogb_binary=is_ogb_binary)

        debug_this_epoch = (epoch < 5) or (epoch % 50 == 0)
        val_metrics = evaluate(model, val_loader, device,
                               debug_preds=debug_this_epoch,
                               use_train_bn=args.use_train_bn,
                               atom_encoder=atom_encoder,
                               is_ogb_binary=is_ogb_binary,
                               ogb_evaluator=ogb_evaluator)
        
        scheduler.step()
        epoch_time = time.time() - t0
        train_val_gap = (train_metrics['accuracy'] - val_metrics['accuracy']) / 100.0
        
        if use_gin_splits:
            print(f"epoch {epoch+1:3d} | loss: {train_metrics['loss']:.4f} | "
                  f"train: {train_metrics['accuracy']:.2f}% | test: {val_metrics['accuracy']:.2f}% | "
                  f"time: {epoch_time:.1f}s")
        else:
            gap_warning = " !!" if train_val_gap > args.max_gap else ""
            val_disp = f"{val_metrics.get('rocauc', val_metrics['accuracy']/100):.4f}" if is_ogb_binary else f"{val_metrics['accuracy']:6.2f}%"
            print(f"{epoch+1:5d} | {train_metrics['loss']:10.4f} | {train_metrics['accuracy']:8.2f}% | "
                  f"{val_metrics['loss']:8.4f} | {val_disp:>7} | {epoch_time:5.1f}s{gap_warning}")
        
        # Convergence info
        conv_parts = []
        if 'local_avg_iter' in train_metrics:
            conv_parts.append(f"Local: {train_metrics['local_avg_iter']:.1f}iter")
        if 'global_avg_iter' in train_metrics:
            conv_parts.append(f"Global: {train_metrics['global_avg_iter']:.1f}iter")
        if 'global_L_g' in train_metrics:
            conv_parts.append(f"L_g: {train_metrics['global_L_g']:.4f}")
        if conv_parts:
            print(f"       [Conv] {' | '.join(conv_parts)}")

        # Q/B stats: one aggregated print per epoch (first N, middle N, last N; or all if N>=T/3)
        N = args.track_Q_stats
        T = args.epochs
        if N <= 0:
            _do_q_stats = False
        elif N * 3 >= T:
            _do_q_stats = True
        else:
            is_first = epoch < N
            is_last = epoch >= T - N
            mid_start = (T - N) // 2
            is_mid = mid_start <= epoch < mid_start + N
            _do_q_stats = is_first or is_mid or is_last
        print_Q_stats = _do_q_stats and 'global_Q_norm' in train_metrics
        if print_Q_stats:
            qm = train_metrics.get('global_Q_mean', 0)
            qs = train_metrics.get('global_Q_std', 0)
            qa = train_metrics.get('global_Q_abs_mean', 0)
            qx = train_metrics.get('global_Q_max_abs', 0)
            qn = train_metrics.get('global_Q_norm', 0)
            bm = train_metrics.get('global_B_mean', 0)
            bs = train_metrics.get('global_B_std', 0)
            ba = train_metrics.get('global_B_abs_mean', 0)
            bx = train_metrics.get('global_B_max_abs', 0)
            bn = train_metrics.get('global_B_norm', 0)
            wa = train_metrics.get('global_WZA_abs_mean', 0)
            wn = train_metrics.get('global_WZA_norm', 0)
            rn = train_metrics.get('global_Q_B_ratio_norm', 0)
            rw = train_metrics.get('global_Q_WZA_ratio_norm', 0)
            ra = train_metrics.get('global_Q_B_ratio_abs', 0)
            rwa = train_metrics.get('global_Q_WZA_ratio_abs', 0)
            print(f"       [Q/B ep{epoch+1}] "
                  f"Q(mean={qm:.4e}, std={qs:.4e}, abs_mean={qa:.4e}, max_abs={qx:.4e}, norm={qn:.4e}) | "
                  f"B(mean={bm:.4e}, std={bs:.4e}, abs_mean={ba:.4e}, max_abs={bx:.4e}, norm={bn:.4e})")
            b_pre = train_metrics.get('global_B_pre_norm')
            b_post = train_metrics.get('global_B_post_norm')
            if b_pre is not None and b_post is not None:
                print(f"       [DIFFUSED] B_pre_norm={b_pre:.4e} B_post_norm={b_post:.4e}")
            print(f"       [COMP] WZA(norm={wn:.4e}, abs_mean={wa:.4e}) | "
                  f"B(norm={bn:.4e}, abs_mean={ba:.4e}) | Q(norm={qn:.4e}, abs_mean={qa:.4e}) | "
                  f"Q/B(norm)={rn:.4e} Q/WZA(norm)={rw:.4e} Q/B(abs)={ra:.4e} Q/WZA(abs)={rwa:.4e}")

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'train_val_gap': train_val_gap,
            'time': epoch_time,
        }
        if is_ogb_binary and 'rocauc' in val_metrics:
            epoch_record['val_rocauc'] = val_metrics['rocauc']
        if 'local_avg_iter' in train_metrics:
            epoch_record['local_iter'] = train_metrics['local_avg_iter']
            epoch_record['local_residual'] = train_metrics['local_avg_residual']
            epoch_record['local_converged_pct'] = train_metrics['local_converged_pct']
        if 'global_avg_iter' in train_metrics:
            epoch_record['global_iter'] = train_metrics['global_avg_iter']
            epoch_record['global_residual'] = train_metrics['global_avg_residual']
            epoch_record['global_converged_pct'] = train_metrics['global_converged_pct']
        if 'global_L_g' in train_metrics:
            epoch_record['global_L_g'] = train_metrics['global_L_g']
        for k in ['Q_mean', 'Q_std', 'Q_abs_mean', 'Q_max_abs', 'Q_norm',
                  'B_mean', 'B_std', 'B_abs_mean', 'B_max_abs', 'B_norm',
                  'B_pre_norm', 'B_post_norm',
                  'WZA_abs_mean', 'WZA_norm',
                  'Q_B_ratio_norm', 'Q_WZA_ratio_norm', 'Q_B_ratio_abs', 'Q_WZA_ratio_abs']:
            key = f'global_{k}'
            if key in train_metrics:
                epoch_record[key] = train_metrics[key]
        history.append(epoch_record)
        
        def get_model_state():
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if atom_encoder is not None:
                state = {'model': state, 'atom_encoder': {k: v.cpu().clone() for k, v in atom_encoder.state_dict().items()}}
            return state
        
        # Model selection
        if args.select_by_gap_sum:
            current_sum = train_metrics['accuracy'] + val_metrics['accuracy']
            current_score = current_sum - abs(train_val_gap) * 100
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_state = get_model_state()
            if epoch + 1 >= args.selection_warmup and abs(train_val_gap) <= args.max_selection_gap:
                if current_score > best_gap_sum:
                    best_gap_sum = current_score
                    best_gap_sum_epoch = epoch + 1
                    best_gap_sum_state = get_model_state()
                    best_gap_sum_val_acc = val_metrics['accuracy']
        elif args.use_last_epoch:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_state = get_model_state()
        elif args.select_by_rocauc and is_ogb_binary:
            val_auc = val_metrics.get('rocauc', 0.0)
            if val_auc > best_val_rocauc:
                best_val_rocauc = val_auc
                best_val_acc = val_metrics['accuracy']
                best_val_loss = val_metrics['loss']
                best_model_state = get_model_state()
                best_val_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
        elif args.select_by_loss:
            if val_metrics['loss'] < best_val_loss:
                best_val_acc = val_metrics['accuracy']
                best_val_loss = val_metrics['loss']
                best_model_state = get_model_state()
                best_val_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_loss = val_metrics['loss']
                best_model_state = get_model_state()
                best_val_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Gap-based early stopping
        if not use_gin_splits and epoch + 1 > args.gap_warmup:
            if train_val_gap > args.max_gap:
                gap_patience_counter += 1
                if gap_patience_counter >= args.gap_patience:
                    print(f"\nGap-based early stopping at epoch {epoch + 1}")
                    break
            else:
                gap_patience_counter = 0
    
        if (not use_gin_splits and not args.use_last_epoch
                and not args.select_by_gap_sum and patience_counter >= args.patience):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Evaluate best model
    print("-" * 70)
    load_state = best_gap_sum_state if (args.select_by_gap_sum and best_gap_sum_state is not None) else best_model_state
    if atom_encoder is not None and isinstance(load_state, dict) and 'atom_encoder' in load_state:
        model.load_state_dict(load_state['model'])
        atom_encoder.load_state_dict(load_state['atom_encoder'])
    else:
        model.load_state_dict(load_state)
    if args.select_by_gap_sum and best_gap_sum_state is not None:
        best_val_acc = best_gap_sum_val_acc
        print(f"  Using gap_sum selected model from epoch {best_gap_sum_epoch}")
        print(f"  Best score (sum-gap): {best_gap_sum:.1f}")
    elif args.select_by_gap_sum:
        print(f"  No epoch satisfied |gap| <= {args.max_selection_gap:.0%}, using last epoch")
    
    test_metrics = evaluate(model, test_loader, device, debug_preds=True,
                            use_train_bn=args.use_train_bn,
                            atom_encoder=atom_encoder,
                            is_ogb_binary=is_ogb_binary,
                            ogb_evaluator=ogb_evaluator)
    
    if args.select_by_gap_sum and best_gap_sum_state is not None:
        final_train_acc = history[best_gap_sum_epoch - 1]['train_acc']
    elif args.use_last_epoch:
        final_train_acc = history[-1]['train_acc'] if history else 0
    elif best_val_epoch > 0 and best_val_epoch <= len(history):
        final_train_acc = history[best_val_epoch - 1]['train_acc']
    else:
        final_train_acc = history[-1]['train_acc'] if history else 0
    
    train_val_gap_final = final_train_acc - best_val_acc
    val_test_gap = best_val_acc - test_metrics['accuracy']
    train_test_gap = final_train_acc - test_metrics['accuracy']
    
    total_time = time.time() - training_start_time
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    if is_ogb_binary:
        test_auc = test_metrics.get('rocauc', test_metrics['accuracy'] / 100.0)
        val_auc = best_val_rocauc if best_val_rocauc > 0 else val_metrics.get('rocauc', 0.0)
        print(f"Best Val ROC-AUC:   {val_auc:.4f}")
        print(f"Test ROC-AUC:       {test_auc:.4f}")
    else:
        print(f"Best Val Accuracy:  {best_val_acc:.2f}%")
        print(f"Test Accuracy:      {test_metrics['accuracy']:.2f}%")
    print(f"Training Time:      {total_time:.1f}s ({total_time/60:.1f}min, {total_time/max(len(history),1):.1f}s/epoch)")
    if not is_ogb_binary:
        print(f"\nGeneralization Analysis:")
        print(f"  Train-Val Gap:    {train_val_gap_final:+.2f}%")
        print(f"  Val-Test Gap:     {val_test_gap:+.2f}%")
        print(f"  Train-Test Gap:   {train_test_gap:+.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.exp_name:
        exp_folder = args.exp_name
    else:
        implicit_tag = "ig" if args.implicit_global else "eg"
        quantum_tag = f"q{args.n_qubits}" if not args.no_quantum else "noq"
        exp_folder = f"{implicit_tag}_h{args.hidden}_{quantum_tag}"
    
    result_dir = f"results/{args.dataset}/{exp_folder}"
    os.makedirs(result_dir, exist_ok=True)
    
    selection_method = ('gap_sum' if args.select_by_gap_sum
                        else 'last_epoch' if args.use_last_epoch
                        else 'rocauc' if (args.select_by_rocauc and is_ogb_binary)
                        else 'loss' if args.select_by_loss
                        else 'val_acc')
    result = {
        'config': vars(args),
        'best_val_acc': best_val_acc,
        'best_val_rocauc': best_val_rocauc if is_ogb_binary else None,
        'test_acc': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
    }
    result.update({
        'n_params': n_params,
        'training_time': {
            'total_seconds': total_time,
            'avg_epoch_seconds': total_time / max(len(history), 1),
            'total_minutes': total_time / 60,
        },
        'history': history,
        'generalization': {
            'train_val_gap': train_val_gap_final,
            'val_test_gap': val_test_gap,
            'train_test_gap': train_test_gap,
        },
        'topology_conditioning': {
            'encoding': use_topo_encoding,
            'ising': use_topo_ising,
            'gate': use_gate,
        },
        'implicit_core': {
            'enabled': args.implicit_global,
            'kappa': args.kappa if args.implicit_global else None,
            'solver': args.solver if args.implicit_global else None,
        },
        'model_selection': {
            'method': selection_method,
            'gap_sum_epoch': best_gap_sum_epoch if args.select_by_gap_sum else None,
            'gap_sum_score': best_gap_sum if args.select_by_gap_sum else None,
            'selected_train_acc': final_train_acc,
        },
        'quantum_circuit': gate_stats,
    })
    if is_ogb_binary and 'rocauc' in test_metrics:
        result['test_auc'] = test_metrics['rocauc']
    
    fold_suffix = f"_fold{args.fold_idx}" if args.n_folds > 0 else ""
    result_path = f"{result_dir}/result{fold_suffix}_seed{args.seed}_{timestamp}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    if args.save_checkpoint:
        ckpt_path = f"{result_dir}/ckpt{fold_suffix}_seed{args.seed}_{timestamp}.pt"
        payload = {
            'model': {k: v.cpu() for k, v in model.state_dict().items()},
            'args': vars(args),
            'best_val_acc': best_val_acc,
            'test_acc': test_metrics['accuracy'],
            'best_val_epoch': best_val_epoch,
            'selection_method': selection_method,
            'timestamp': timestamp,
            'result_json': result_path,
        }
        if is_ogb_binary:
            payload['best_val_rocauc'] = best_val_rocauc
            if 'rocauc' in test_metrics:
                payload['test_rocauc'] = test_metrics['rocauc']
        if atom_encoder is not None:
            payload['atom_encoder'] = {k: v.cpu() for k, v in atom_encoder.state_dict().items()}
        torch.save(payload, ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
