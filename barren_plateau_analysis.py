#!/usr/bin/env python3
"""
Barren-plateau diagnostic for the three QIGNN injection variants (IN / SD / BD).

For each variant, we freeze all classical parameters, repeatedly re-initialise
only the TorchQuantumCircuit subset of PQC parameters from their original
Gaussian init distribution, run one forward+backward pass on a fixed NCI1 mini
batch, and record the variance of the resulting gradient across the resamples.

Quantum-circuit parameter set (see `qignn/ansatz.py` L312-329):
    encoding_scale, encoding_bias, eta,
    edge_weights_{zz,xx,yy}, node_biases_{z,x,y}, trainable_rots.
`eta` is a constant init; the rest are Gaussian-perturbed.

Scanned axes (three scans per run by default):
    --axis width   : sweep PQC qubit count  (fixed circuit_reps, max_iter)
    --axis depth   : sweep PQC circuit_reps (fixed n_qubits,    max_iter)
    --axis T       : sweep solver max_iter  (fixed n_qubits,    circuit_reps)

The solver is forced to '--solver unroll' so that BPTT flows through every
Picard step and every PQC evaluation inside the implicit core; this is what
makes the T-axis meaningful for SD/BD and flat for IN.

Outputs (under --output_dir, default results/barren_plateau/):
    bp_grid.csv          one row per (variant, axis, grid point, param group)
    bp_manifest.json     metadata: git hash, torch/cuda version, batch hash,
                         PQC parameter count, grid specs, timing

Usage
-----
    python barren_plateau_analysis.py \
    --dataset MUTAG --variants IN SD BD \
    --axes width depth \
    --n_resamples 256 --point_timeout_s 1200 \
    --bp_solver torchdeq \
    --output_dir results/barren_plateau/MUTAG/main_torchdeq
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from model_factory import setup_args, build_dataset, build_model


# =============================================================================
# Variant -> PQC parameter prefix
# =============================================================================

PQC_PREFIXES: Dict[str, str] = {
    "IN": "quantum_node.circuit.",
    "SD": "implicit_core.qc_inside.circuit.",
    "BD": "implicit_core.qc_inside.circuit.",
}

PARAM_GROUP_ORDER = [
    "edge_weights", "node_biases", "trainable_rots",
    "encoding_bias", "encoding_scale", "eta", "ALL",
]


def classify_param(pqc_name: str) -> str:
    """Map a TorchQuantumCircuit parameter name to a coarse group for CSV aggregation."""
    if pqc_name.startswith("edge_weights"):
        return "edge_weights"
    if pqc_name.startswith("node_biases"):
        return "node_biases"
    if pqc_name.startswith("trainable_rots"):
        return "trainable_rots"
    if pqc_name.startswith("encoding_scale"):
        return "encoding_scale"
    if pqc_name.startswith("encoding_bias"):
        return "encoding_bias"
    if pqc_name == "eta":
        return "eta"
    return "other"


def load_existing_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read all rows from an existing bp_grid.csv (as raw string dicts).

    Used by --resume to preserve prior results when restarting. Returns [] if
    the file does not exist.
    """
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def completed_point_keys(rows: List[Dict[str, str]], n_resamples_target: int,
                         required_groups: set) -> set:
    """Return the set of (variant, axis, n_qubits, circuit_reps, max_iter) grid
    points that are considered complete in the existing CSV:
        - all required_groups present
        - n_resamples_completed >= n_resamples_target (old CSVs without this
          column are treated as complete at their n_resamples value)
        - timed_out is False (timed-out points are retried)
    """
    groups_by_key: Dict[Tuple, set] = {}
    meta_by_key: Dict[Tuple, Dict] = {}
    for r in rows:
        try:
            key = (
                r.get("variant", ""),
                r.get("axis", ""),
                int(r.get("n_qubits") or 0),
                int(r.get("circuit_reps") or 0),
                int(r.get("max_iter") or 0),
            )
        except (TypeError, ValueError):
            continue
        groups_by_key.setdefault(key, set()).add(r.get("group", ""))
        # Take the max observed n_resamples_completed across groups of a key.
        if "n_resamples_completed" in r and r["n_resamples_completed"] not in ("", None):
            try:
                n_completed = int(r["n_resamples_completed"])
            except ValueError:
                n_completed = 0
        else:
            try:
                n_completed = int(r.get("n_resamples") or 0)
            except ValueError:
                n_completed = 0
        to = str(r.get("timed_out", "")).strip().lower() in ("1", "true", "t", "yes")
        prev = meta_by_key.get(key, {"n_completed": 0, "timed_out": False})
        meta_by_key[key] = {
            "n_completed": max(prev["n_completed"], n_completed),
            "timed_out": prev["timed_out"] or to,
        }
    done = set()
    for key, groups in groups_by_key.items():
        meta = meta_by_key[key]
        if meta["timed_out"]:
            continue
        if meta["n_completed"] < n_resamples_target:
            continue
        if not required_groups.issubset(groups):
            continue
        done.add(key)
    return done


# =============================================================================
# Model / parameter helpers
# =============================================================================

def list_pqc_params(model: torch.nn.Module, variant: str) -> List[Tuple[str, str, torch.nn.Parameter]]:
    """Return [(full_name, pqc_name, param)] for all TorchQuantumCircuit parameters
    belonging to the requested variant's PQC module.
    """
    prefix = PQC_PREFIXES[variant]
    out: List[Tuple[str, str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            out.append((name, name[len(prefix):], p))
    return out


def freeze_non_pqc(model: torch.nn.Module, variant: str) -> None:
    """Set requires_grad on only the PQC subset for the chosen variant."""
    prefix = PQC_PREFIXES[variant]
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefix))


def resample_single_(param: torch.nn.Parameter, pqc_name: str, gen: torch.Generator) -> None:
    """Re-initialise a single TorchQuantumCircuit parameter using its original init
    distribution. See `qignn/ansatz.py::_init_parameters`.
    """
    shape = param.shape
    dtype = param.dtype
    # torch.randn with a generator on cuda is fragile; sample on CPU and move.
    def randn_like():
        return torch.randn(shape, generator=gen, dtype=dtype).to(param.device)

    if pqc_name.startswith("encoding_scale"):
        # ones(q,3) * pi + randn(q,3) * 0.1
        new_val = torch.full(shape, float(np.pi), dtype=dtype, device=param.device) \
                  + randn_like() * 0.1
    elif pqc_name.startswith("encoding_bias"):
        new_val = randn_like() * 0.1
    elif pqc_name.startswith("node_biases"):
        new_val = randn_like() * 0.1
    elif pqc_name.startswith("trainable_rots"):
        new_val = randn_like() * 0.1
    elif pqc_name.startswith("edge_weights"):
        # ones(R,e) + randn * 0.1
        new_val = torch.ones(shape, dtype=dtype, device=param.device) + randn_like() * 0.1
    elif pqc_name == "eta":
        # Constant init (ones(R,3) * 0.5); leave untouched.
        return
    else:
        # Unknown pqc parameter; skip.
        return
    with torch.no_grad():
        param.data.copy_(new_val)


def resample_pqc_(pqc_params, gen: torch.Generator) -> None:
    for _, pqc_name, param in pqc_params:
        resample_single_(param, pqc_name, gen)


def _safe_variance(M: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Per-column variance computed in float64 on the finite-row subset.

    Returns
    -------
    var : np.ndarray of shape (n_params,)
        Variance over resamples; columns where no finite rows remain become inf.
    n_rows_finite : int
        Number of resamples whose gradient vector was entirely finite.
    n_rows_diverged : int
        Number of resamples that contained any non-finite entry (dropped).
    """
    M64 = M.astype(np.float64, copy=False)
    row_finite = np.isfinite(M64).all(axis=1)
    n_rows_finite = int(row_finite.sum())
    n_rows_diverged = int((~row_finite).sum())
    M_ok = M64[row_finite]
    if M_ok.shape[0] < 2:
        return (np.full(M64.shape[1], np.inf, dtype=np.float64),
                n_rows_finite, n_rows_diverged)
    with np.errstate(over="ignore", invalid="ignore"):
        var = M_ok.var(axis=0)
    # Any column that still overflowed (shouldn't in float64, but guard anyway).
    var = np.where(np.isfinite(var), var, np.inf)
    return var, n_rows_finite, n_rows_diverged


def _safe_log10_median_abs(M: np.ndarray) -> float:
    """log10 of the median of |M| over all finite entries. Robust to inf."""
    a = np.abs(M.astype(np.float64, copy=False))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("inf")
    med = float(np.median(a))
    if med <= 0:
        return float("-inf")
    return float(np.log10(med))


# =============================================================================
# Args construction
# =============================================================================

BASE_FLAGS = dict(
    dataset="NCI1",
    data_dir="data",
    exp_name=None,
    hidden=64,
    n_encoder_layers=5,
    min_encoder=True,
    simple_encoder=False,
    lqa=False,
    auto_lqa_neighbors=False,
    jk_mode="sum",
    n_decoder_layers=0,
    no_decoder=False,
    use_film=False,
    dynamic_film=False,
    dropout=0.0,   # BP analysis: dropout disabled to keep forward deterministic
    use_layer_norm=True,
    use_train_bn=False,
    drop_edge=0.0,
    pooling="attention",
    max_cycle_length=20,
    topo_encoding=False,
    topo_ising=False,
    use_gate=False,
    no_topo=False,
    no_q_inject=False,
    q_inj_node_cond=False,
    topo_drop_enc=0.0,
    topo_drop_ising=0.0,
    topo_drop_gate=0.0,
    implicit_global=True,
    implicit_self_loops=False,
    no_normalize_adj=False,
    kappa=0.8,
    solver="unroll",
    tol=1e-6,
    qi_alpha=0.1,
    qi_classical=False,
    qi_topo=False,
    perm_invariant=False,
    ablation_lg=False,
    ignn_injection=False,
    lr=0.001,
    epochs=1,
    batch_size=32,
    weight_decay=0.0,
    scheduler="cosine",
    jac_reg=0.0,
    label_smoothing=0.0,
    grad_clip=0.0,
    iters_per_epoch=0,
    patience=999,
    seed=42,
    select_by_loss=True,
    select_by_rocauc=False,
    use_last_epoch=False,
    select_by_gap_sum=False,
    max_selection_gap=0.03,
    selection_warmup=30,
    n_folds=10,
    fold_idx=0,
    use_gin_splits=False,
    max_gap=0.15,
    gap_patience=5,
    gap_warmup=10,
    track_L_g=0,
    track_Q_stats=0,
    save_checkpoint=False,
    device="cuda",
    gpu=None,
    # Fields below are variant-specific; filled by args_for_variant.
    n_qubits=4,
    circuit_reps=1,
    qi_n_qubits=4,
    qi_circuit_reps=1,
    max_iter=50,
    q_ind_node=False,
    quantum_inside=False,
    qi_direct=False,
    no_quantum=True,
    # LQA defaults (not used here, but required by setup_args)
    lqa_max_neighbors=4,
    lqa_qubits_per_neighbor=4,
    lqa_conv_layers=2,
)


def args_for_variant(variant: str, n_qubits: int, circuit_reps: int, max_iter: int,
                     solver: str = "unroll") -> argparse.Namespace:
    """Produce an args Namespace for one (variant, grid-point) model build.

    ``solver`` selects which BatchedImplicitCore backward path to use:
        - ``unroll``   : full BPTT through ``max_iter`` Picard steps (main BP
                         diagnostic; isolates the 'repeated solver application'
                         amplification discussed in the paper).
        - ``torchdeq`` : Implicit Function Theorem (one Jacobian-inverse solve);
                         matches the paper's main training loop. Expected to
                         render the T-axis nearly invariant.
        - ``simple``   : legacy simple solver (no_grad loop + IFT surrogate);
                         included for completeness but not recommended.
    """
    cfg = dict(BASE_FLAGS)
    cfg["max_iter"] = int(max_iter)
    cfg["solver"] = str(solver)

    if variant == "IN":
        cfg["q_ind_node"] = True
        cfg["quantum_inside"] = False
        cfg["qi_direct"] = False
        cfg["no_quantum"] = True
        cfg["n_qubits"] = int(n_qubits)
        cfg["circuit_reps"] = int(circuit_reps)
    elif variant == "SD":
        cfg["q_ind_node"] = False
        cfg["quantum_inside"] = True
        cfg["qi_direct"] = True
        cfg["no_quantum"] = True
        cfg["qi_n_qubits"] = int(n_qubits)
        cfg["qi_circuit_reps"] = int(circuit_reps)
    elif variant == "BD":
        cfg["q_ind_node"] = False
        cfg["quantum_inside"] = True
        cfg["qi_direct"] = False
        cfg["no_quantum"] = True
        cfg["qi_n_qubits"] = int(n_qubits)
        cfg["qi_circuit_reps"] = int(circuit_reps)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return argparse.Namespace(**cfg)


# =============================================================================
# Fixed batch preparation
# =============================================================================

def _batch_fingerprint(batch: Batch) -> str:
    """Stable hash over the immutable batch tensors so re-runs can verify identity."""
    h = hashlib.sha256()
    for name in ("x", "edge_index", "y", "batch"):
        t = getattr(batch, name, None)
        if t is None:
            continue
        arr = t.detach().cpu().contiguous().numpy().tobytes()
        h.update(name.encode())
        h.update(arr)
    return h.hexdigest()[:16]


def prepare_fixed_batch(loader, device: torch.device, batch_seed: int, batch_size: int
                        ) -> Tuple[Batch, Optional[Dict[str, torch.Tensor]], str]:
    """Pick a deterministic mini-batch from the loader's dataset."""
    g = torch.Generator().manual_seed(int(batch_seed))
    ds = list(loader.dataset)
    idx = torch.randperm(len(ds), generator=g)[: min(batch_size, len(ds))].tolist()
    batch = Batch.from_data_list([ds[i] for i in idx]).to(device)

    topo = None
    if hasattr(batch, "combined_node_features") and batch.combined_node_features is not None:
        graph_feat = batch.graph_cycle_features
        feat_dim = graph_feat.shape[0] // batch.num_graphs
        graph_feat = graph_feat.view(batch.num_graphs, feat_dim)
        topo = {
            "combined_node_features": batch.combined_node_features,
            "graph_cycle_features": graph_feat,
        }
    return batch, topo, _batch_fingerprint(batch)


# =============================================================================
# Core BP scan
# =============================================================================

def scan_point(
    variant: str,
    n_qubits: int,
    circuit_reps: int,
    max_iter: int,
    batch: Batch,
    topo: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    n_resamples: int,
    resample_seed_base: int,
    num_classes: int,
    dataset_info: dict,
    solver: str = "unroll",
    point_timeout_s: Optional[float] = None,
) -> List[Dict]:
    """Re-sample PQC `n_resamples` times at a single grid point and compute
    per-parameter gradient variance aggregates.

    If ``point_timeout_s`` is set, the resampling loop bails out as soon as the
    cumulative wall-time exceeds the budget (checked between resamples; a
    single resample in progress is not interrupted). Rows emitted in that case
    use the partial sample count and set ``timed_out=True`` so plots / downstream
    tooling can flag the point instead of silently reporting low-sample variance.

    Returns one list-row per parameter group (plus an ALL aggregate).
    """
    args = args_for_variant(variant, n_qubits, circuit_reps, max_iter, solver=solver)
    model, _ = build_model(args, dataset_info["num_features"], num_classes, device, verbose=False)
    # Note: we use model.train() rather than eval() because torchdeq's IFT
    # backward hook only attaches under training mode. Since BASE_FLAGS sets
    # dropout=0 and topo_drop_*=0, and the model uses LayerNorm (not BatchNorm),
    # train() and eval() produce numerically identical forwards for our setup,
    # so the BP forward is still deterministic w.r.t. the PQC resample.
    model.train()
    freeze_non_pqc(model, variant)
    pqc_params = list_pqc_params(model, variant)
    if not pqc_params:
        raise RuntimeError(
            f"No PQC parameters found under prefix '{PQC_PREFIXES[variant]}' for variant {variant}. "
            f"Check the variant flag mapping and the model constructor.")

    # Group metadata (ordered, stable across resamples)
    name_to_group = {full: classify_param(pname) for full, pname, _ in pqc_params}
    all_pqc_names = [full for full, _, _ in pqc_params]
    n_params_pqc = sum(p.numel() for _, _, p in pqc_params)

    # For each resample, flatten grads per param group into a single vector.
    # We store (n_resamples, n_params_in_group) float32 tensors on CPU to stay
    # lean on GPU memory.
    grad_matrices: Dict[str, List[np.ndarray]] = {}
    output_samples: List[np.ndarray] = []
    loss_samples: List[float] = []

    point_t0 = time.time()
    completed_resamples = 0
    timed_out = False

    for k in range(n_resamples):
        # Circuit-breaker: check at resample boundary only. A resample already
        # in progress will run to completion; the next one is skipped.
        if point_timeout_s is not None and (time.time() - point_t0) > point_timeout_s:
            timed_out = True
            print(f"[BP] [timeout] variant={variant} n={n_qubits} R={circuit_reps} "
                  f"T={max_iter}: exceeded {point_timeout_s:.0f}s budget after "
                  f"{completed_resamples}/{n_resamples} resamples; bailing.",
                  file=sys.stderr)
            break

        gen = torch.Generator().manual_seed(int(resample_seed_base + k))
        resample_pqc_(pqc_params, gen)

        for _, _, p in pqc_params:
            if p.grad is not None:
                p.grad.zero_()
        logits, _ = model(batch, topo)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()

        loss_samples.append(float(loss.detach().cpu().item()))
        output_samples.append(logits.detach().mean(dim=0).cpu().numpy().astype(np.float32))

        # Slice grads by param group
        per_group_chunks: Dict[str, List[np.ndarray]] = {}
        for full, _pname, p in pqc_params:
            grp = name_to_group[full]
            g_flat = p.grad.detach().flatten().cpu().numpy().astype(np.float32) \
                if p.grad is not None else np.zeros(p.numel(), dtype=np.float32)
            per_group_chunks.setdefault(grp, []).append(g_flat)
            per_group_chunks.setdefault("ALL", []).append(g_flat)

        for grp, chunks in per_group_chunks.items():
            stacked = np.concatenate(chunks)
            grad_matrices.setdefault(grp, []).append(stacked)
        completed_resamples = k + 1

    point_wall_s = time.time() - point_t0

    # Aggregate: per-parameter variance across resamples, then mean/median over params.
    # All variance computations happen in float64 to avoid float32 x*x overflow
    # when SD/BD diverge under the unroll solver.
    rows: List[Dict] = []
    if output_samples:
        out_mat = np.stack(output_samples, axis=0)  # [K, n_classes]
        var_output, _, _ = _safe_variance(out_mat)
    else:
        var_output = np.array([float("inf")])

    loss_arr = np.asarray(loss_samples, dtype=np.float64)
    finite_loss = loss_arr[np.isfinite(loss_arr)]
    var_loss = float(np.var(finite_loss)) if finite_loss.size >= 2 else float("inf")
    loss_mean = float(np.mean(finite_loss)) if finite_loss.size else float("inf")
    n_loss_diverged = int(np.sum(~np.isfinite(loss_arr)))

    for grp in PARAM_GROUP_ORDER:
        if grp not in grad_matrices:
            continue
        M = np.stack(grad_matrices[grp], axis=0)  # [K, n_params_in_group]
        var_grad, n_finite, n_div = _safe_variance(M)
        grad_log_abs_median = _safe_log10_median_abs(M)
        # Aggregates over finite variance entries only.
        finite_var = var_grad[np.isfinite(var_grad)]
        if finite_var.size:
            vgm = float(finite_var.mean())
            vgmed = float(np.median(finite_var))
            vgmax = float(finite_var.max())
        else:
            vgm = vgmed = vgmax = float("inf")
        rows.append({
            "variant": variant,
            "bp_solver": str(solver),
            "n_qubits": int(n_qubits),
            "circuit_reps": int(circuit_reps),
            "max_iter": int(max_iter),
            "n_params_pqc": int(n_params_pqc),
            "group": grp,
            "n_params_in_group": int(M.shape[1]),
            "n_resamples": int(n_resamples),
            "n_resamples_completed": int(completed_resamples),
            "timed_out": bool(timed_out),
            "point_wall_s": float(point_wall_s),
            "n_resamples_finite": int(n_finite),
            "n_resamples_diverged": int(n_div),
            "var_grad_mean": vgm,
            "var_grad_median": vgmed,
            "var_grad_max": vgmax,
            "grad_log10_abs_median": grad_log_abs_median,
            "var_output_mean": float(var_output.mean()) if np.isfinite(var_output).all() else float("inf"),
            "var_loss": var_loss,
            "loss_mean": loss_mean,
            "n_loss_diverged": n_loss_diverged,
        })

    del model, pqc_params, grad_matrices
    torch.cuda.empty_cache()
    return rows


# =============================================================================
# Grid definitions
# =============================================================================

WIDTH_GRID_DEFAULT = [2, 4, 6, 8, 10]
DEPTH_GRID_DEFAULT = [1, 2, 3, 4, 6, 8]
T_GRID_DEFAULT = [5, 10, 30, 50, 100, 300]


def iter_grid(axis: str, widths: List[int], depths: List[int], Ts: List[int],
              fix_n: int, fix_R: int, fix_T: int):
    if axis == "width":
        for n in widths:
            yield ("width", n, fix_R, fix_T)
    elif axis == "depth":
        for r in depths:
            yield ("depth", fix_n, r, fix_T)
    elif axis == "T":
        for t in Ts:
            yield ("T", fix_n, fix_R, t)
    else:
        raise ValueError(f"Unknown axis: {axis}")


# =============================================================================
# Main
# =============================================================================

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Barren-plateau diagnostic for QIGNN variants")
    parser.add_argument("--dataset", type=str, default="NCI1")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--variants", type=str, nargs="+", default=["IN", "SD", "BD"],
                        choices=["IN", "SD", "BD"])
    parser.add_argument("--axes", type=str, nargs="+", default=["width", "depth", "T"],
                        choices=["width", "depth", "T"])
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTH_GRID_DEFAULT)
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTH_GRID_DEFAULT)
    parser.add_argument("--Ts", type=int, nargs="+", default=T_GRID_DEFAULT)
    parser.add_argument("--fix_n", type=int, default=4, help="n_qubits when not on width axis")
    parser.add_argument("--fix_R", type=int, default=1, help="circuit_reps when not on depth axis")
    parser.add_argument("--fix_T", type=int, default=50, help="max_iter when not on T axis")
    parser.add_argument("--n_resamples", type=int, default=256)
    parser.add_argument("--resample_seed_base", type=int, default=10_000)
    parser.add_argument("--batch_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/barren_plateau")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to output filenames.")
    parser.add_argument("--bp_solver", type=str, default="unroll",
                        choices=["unroll", "torchdeq", "simple"],
                        help=(
                            "Solver used by the BP forward+backward. "
                            "'unroll' = full BPTT through max_iter Picard steps "
                            "(isolates the 'repeated solver application' amplification, "
                            "used for the paper appendix T-axis figure). "
                            "'torchdeq' = IFT, matches the paper's main training loop "
                            "(recommended for width/depth main figures). "
                            "'simple' = legacy no_grad + IFT surrogate."
                        ))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--point_timeout_s", type=float, default=None,
                        help="Per-grid-point wall-clock budget (seconds). When set, "
                             "each grid point bails out between resamples once this "
                             "budget is exceeded and records whatever data was "
                             "collected (flagged timed_out=True in the CSV). Useful "
                             "for circuit-breaking pathological torchdeq backward "
                             "solves on SD at large n_qubits. Default: no limit.")
    parser.add_argument("--force_overwrite", action="store_true",
                        help="Allow overwriting an existing bp_grid.csv / bp_manifest.json "
                             "even when its dataset / solver / axes differ from the new run. "
                             "Default: abort with a descriptive error (to prevent "
                             "accidental loss of previous multi-hour scans).")
    parser.add_argument("--resume", action="store_true",
                        help="If the output directory already contains a bp_grid.csv, "
                             "preserve its rows and skip any grid point that already has a "
                             "complete row set (all parameter groups, n_resamples_completed "
                             ">= --n_resamples, timed_out=False). Implies compatibility with "
                             "the existing manifest's dataset/solver/axes (checked separately "
                             "by the manifest guard unless --force_overwrite is given).")
    args = parser.parse_args()

    if args.bp_solver == "torchdeq" and "T" in args.axes:
        print("[BP] [warn] --bp_solver=torchdeq combined with --axes T: under IFT, "
              "the T-axis is approximately gradient-invariant (solver steps are "
              "hidden by the implicit function theorem). Consider switching to "
              "--bp_solver unroll for the T axis.", file=sys.stderr)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build dataset once with a vanilla args (no quantum) purely to get loaders/topo.
    boot_args = args_for_variant("IN", args.fix_n, args.fix_R, args.fix_T, solver=args.bp_solver)
    boot_args.dataset = args.dataset
    boot_args.data_dir = args.data_dir
    boot_args.batch_size = args.batch_size
    dataset_info = build_dataset(boot_args, device, verbose=False)
    num_classes = dataset_info["num_classes"]

    batch, topo, batch_hash = prepare_fixed_batch(
        dataset_info["train_loader"], device, args.batch_seed, args.batch_size)
    print(f"[BP] dataset={args.dataset} batch_size={batch.num_graphs} "
          f"num_classes={num_classes} batch_hash={batch_hash}")

    output_dir = Path(args.output_dir)
    if args.tag:
        output_dir = output_dir / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "bp_grid.csv"
    manifest_path = output_dir / "bp_manifest.json"

    # ---- Defensive: refuse to overwrite an unrelated previous scan. ----
    # Background: `open(csv_path, "w")` truncates on open. Running this script
    # against a populated output_dir whose manifest belongs to a different
    # (dataset, solver, axes) combination would silently destroy hours of
    # computed rows before the new scan has even produced its first grid point.
    if manifest_path.exists() and not args.force_overwrite:
        try:
            with open(manifest_path) as mf:
                prev = json.load(mf)
        except (json.JSONDecodeError, OSError):
            prev = None
        if isinstance(prev, dict):
            prev_dataset = prev.get("dataset")
            prev_solver = prev.get("bp_solver")
            prev_axes = sorted(prev.get("axes") or [])
            new_axes = sorted(args.axes)
            mismatch_reasons = []
            if prev_dataset and prev_dataset != args.dataset:
                mismatch_reasons.append(
                    f"dataset: existing='{prev_dataset}' vs new='{args.dataset}'")
            if prev_solver and prev_solver != args.bp_solver:
                mismatch_reasons.append(
                    f"bp_solver: existing='{prev_solver}' vs new='{args.bp_solver}'")
            if prev_axes and prev_axes != new_axes:
                mismatch_reasons.append(
                    f"axes: existing={prev_axes} vs new={new_axes}")
            if mismatch_reasons:
                raise SystemExit(
                    f"[BP] ABORT: output_dir '{output_dir}' already contains a "
                    f"different scan. Mismatches:\n  - "
                    + "\n  - ".join(mismatch_reasons)
                    + f"\nRefusing to truncate the existing bp_grid.csv.\n"
                      f"Options:\n"
                      f"  (1) Point --output_dir at a different path "
                      f"(e.g. results/barren_plateau/{args.dataset}/...).\n"
                      f"  (2) Delete the existing manifest+csv manually if you "
                      f"really want to overwrite.\n"
                      f"  (3) Pass --force_overwrite to bypass this check.")

    fieldnames = [
        "variant", "bp_solver", "axis", "n_qubits", "circuit_reps", "max_iter",
        "n_params_pqc", "group", "n_params_in_group", "n_resamples",
        "n_resamples_completed", "timed_out", "point_wall_s",
        "n_resamples_finite", "n_resamples_diverged",
        "var_grad_mean", "var_grad_median", "var_grad_max",
        "grad_log10_abs_median",
        "var_output_mean", "var_loss", "loss_mean", "n_loss_diverged",
        "batch_hash",
    ]

    # --resume: preserve existing rows and skip already-finished grid points.
    completed_keys: set = set()
    preserved_rows: List[Dict[str, str]] = []
    if args.resume:
        preserved_rows = load_existing_rows(csv_path)
        completed_keys = completed_point_keys(
            preserved_rows, args.n_resamples, set(PARAM_GROUP_ORDER))
        print(f"[BP] resume: preserved {len(preserved_rows)} rows from existing CSV; "
              f"skipping {len(completed_keys)} already-complete grid points.")

    t_start = time.time()
    rows_written = 0
    first_pqc_counts: Dict[str, int] = {}
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Replay preserved rows first so the new CSV is a complete superset.
        for row in preserved_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            rows_written += 1
        f.flush()
        for variant in args.variants:
            for axis in args.axes:
                for (_axis_name, n, r, t) in iter_grid(
                        axis, args.widths, args.depths, args.Ts,
                        args.fix_n, args.fix_R, args.fix_T):
                    key = (variant, axis, int(n), int(r), int(t))
                    if key in completed_keys:
                        print(f"[BP] SKIP {variant} axis={axis} n={n} R={r} T={t}: "
                              f"already complete in existing CSV ({args.n_resamples} resamples)")
                        continue
                    t0 = time.time()
                    try:
                        rows = scan_point(
                            variant=variant, n_qubits=n, circuit_reps=r, max_iter=t,
                            batch=batch, topo=topo, device=device,
                            n_resamples=args.n_resamples,
                            resample_seed_base=args.resample_seed_base,
                            num_classes=num_classes,
                            dataset_info=dataset_info,
                            solver=args.bp_solver,
                            point_timeout_s=args.point_timeout_s,
                        )
                    except Exception as e:
                        print(f"[BP] FAILED variant={variant} n={n} R={r} T={t}: "
                              f"{type(e).__name__}: {e}", file=sys.stderr)
                        raise
                    elapsed = time.time() - t0
                    for row in rows:
                        row["axis"] = axis
                        row["batch_hash"] = batch_hash
                        writer.writerow({k: row.get(k) for k in fieldnames})
                        rows_written += 1
                    f.flush()
                    if rows:
                        first_pqc_counts.setdefault(variant, rows[0]["n_params_pqc"])
                    total = time.time() - t_start
                    to_flag = " TIMED_OUT" if rows and rows[0].get("timed_out") else ""
                    n_done = rows[0].get("n_resamples_completed", args.n_resamples) if rows else 0
                    print(f"[BP] {variant} axis={axis} n={n} R={r} T={t}: "
                          f"{elapsed:6.1f}s{to_flag}  "
                          f"(completed={n_done}/{args.n_resamples} rows={len(rows)} "
                          f"total_rows={rows_written} elapsed={total/60:.1f}min)")

    manifest = {
        "dataset": args.dataset,
        "variants": args.variants,
        "axes": args.axes,
        "widths": args.widths,
        "depths": args.depths,
        "Ts": args.Ts,
        "fix_n": args.fix_n,
        "fix_R": args.fix_R,
        "fix_T": args.fix_T,
        "n_resamples": args.n_resamples,
        "resample_seed_base": args.resample_seed_base,
        "batch_seed": args.batch_seed,
        "batch_size_requested": args.batch_size,
        "batch_size_actual": int(batch.num_graphs),
        "batch_hash": batch_hash,
        "pqc_params_per_variant_at_fix_nR": first_pqc_counts,
        "bp_solver": args.bp_solver,
        "tol": 0.0,
        "model_mode": "train",  # required by torchdeq; deterministic b/c dropout=0 + LayerNorm
        "dropout": 0.0,
        "point_timeout_s": args.point_timeout_s,
        "git_hash": _git_hash(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device": str(device),
        "elapsed_seconds": time.time() - t_start,
        "rows_written": rows_written,
        "csv_path": str(csv_path.resolve()),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[BP] done. CSV -> {csv_path}  manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
