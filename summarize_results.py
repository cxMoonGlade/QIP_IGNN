#!/usr/bin/env python3
"""
Summarize QIGNN experiment results for paper artifacts.

This version treats the current results tree as the source of truth:
  - reruns are deduplicated by run key
  - raw aggregates are separated from matched-overlap views
  - stability traces are built from the same deduplicated selection
  - a machine-readable summary can be emitted for paper figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PAPER_DATASETS_TU = ["MUTAG", "NCI1", "PROTEINS", "REDDIT-BINARY"]
PAPER_DATASETS_OGB = ["ogbg-molhiv"]
DIRECT_POST_DATASETS = ["NCI1", "PROTEINS", "MUTAG"]
RESIDUAL_FAMILY_DATASETS = ["NCI1", "PROTEINS"]

RESIDUAL_FAMILY_EXPERIMENTS = {
    "quantum_post": ("Quantum post-backbone", "QIGNN_mainline"),
    "quantum_direct": ("Quantum direct", "ablation_direct_q"),
    "classical_post": ("Classical post-backbone", "resfam_post_c"),
    "classical_direct": ("Classical direct", "resfam_direct_c"),
}

TABLE1_EXPERIMENTS = {
    "GIN": "GIN_baseline",
    "QIGNN": "QIGNN_mainline",
}

TABLE2_EXPERIMENTS = {
    "Classical backbone only": "ablation_classical",
    "External graph-level quantum injection": "ablation_external_q",
    "Static FiLM injection": "ablation_film",
    "Direct-residual quantum": "ablation_direct_q",
    "Matched classical residual": "ablation_classical_resid",
    "Post-backbone residual": "QIGNN_mainline",
}

NCI1_CORE_EXPERIMENTS = {
    "classical": ("Classical backbone only", "ablation_classical"),
    "external": ("External graph-level quantum injection", "ablation_external_q"),
    "direct": ("Direct residual", "ablation_direct_q"),
    "post": ("Post-backbone residual", "QIGNN_mainline"),
}

FIGURE1_EXPERIMENTS = {
    "Classical backbone only": "ablation_classical",
    "External graph-level quantum injection": "ablation_external_q",
    "Direct residual": "ablation_direct_q",
    "Post-backbone residual": "QIGNN_mainline",
}

FIGURE1_DATASETS = ["MUTAG", "NCI1", "PROTEINS"]

RESULT_STAMP_RE = re.compile(r"_(\d{4})_(\d{4})\.json$")


def find_result_files(results_dir: str | Path, dataset: str, exp_name: str) -> List[Path]:
    """Find all result JSON files for a given dataset/experiment."""
    search_dir = Path(results_dir) / dataset / exp_name
    if not search_dir.exists():
        return []
    return sorted(search_dir.rglob("result*.json"))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def is_ogb(dataset: str) -> bool:
    return dataset.startswith("ogbg-")


def infer_run_key(dataset: str, record: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    cfg = record.get("config", {})
    seed = cfg.get("seed")
    if seed is None:
        return None
    if is_ogb(dataset):
        return (int(seed),)
    fold = cfg.get("fold_idx")
    if fold is None:
        return None
    return (int(fold), int(seed))


def is_valid_result(record: Dict[str, Any], dataset: str) -> bool:
    if is_ogb(dataset):
        return record.get("test_auc") is not None
    return record.get("test_acc") is not None


def file_rank(path: Path) -> Tuple[int, str]:
    match = RESULT_STAMP_RE.search(path.name)
    if match:
        return (int(match.group(1) + match.group(2)), path.name)
    try:
        stat = path.stat()
        return (int(stat.st_mtime_ns), path.name)
    except OSError:
        return (0, path.name)


def last_history_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    history = record.get("history", [])
    if history:
        return history[-1]
    return {}


def run_payload(path: Path, record: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    cfg = record.get("config", {})
    last = last_history_entry(record)
    return {
        "path": str(path),
        "filename": path.name,
        "key": list(infer_run_key(dataset, record) or ()),
        "seed": cfg.get("seed"),
        "fold_idx": cfg.get("fold_idx"),
        "n_folds": cfg.get("n_folds"),
        "test_acc": record.get("test_acc"),
        "test_loss": record.get("test_loss"),
        "best_val_acc": record.get("best_val_acc"),
        "best_val_rocauc": record.get("best_val_rocauc"),
        "test_auc": record.get("test_auc"),
        "avg_epoch_seconds": record.get("training_time", {}).get("avg_epoch_seconds"),
        "total_seconds": record.get("training_time", {}).get("total_seconds"),
        "last_global_iter": last.get("global_iter"),
        "last_global_residual": last.get("global_residual"),
        "history": record.get("history", []),
        "record": record,
    }


def dedupe_runs(results_dir: str | Path, dataset: str, exp_name: str) -> Dict[str, Any]:
    files = find_result_files(results_dir, dataset, exp_name)
    selected: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    duplicates = 0
    invalid = 0
    unkeyed = 0

    for path in files:
        record = load_json(path)
        if record is None or not is_valid_result(record, dataset):
            invalid += 1
            continue
        key = infer_run_key(dataset, record)
        if key is None:
            unkeyed += 1
            continue
        payload = run_payload(path, record, dataset)
        prev = selected.get(key)
        if prev is None:
            selected[key] = payload
        else:
            duplicates += 1
            if file_rank(path) > file_rank(Path(prev["path"])):
                selected[key] = payload

    deduped_runs = [selected[key] for key in sorted(selected)]
    unique_seeds = sorted({run["seed"] for run in deduped_runs if run["seed"] is not None})
    planned_runs = None
    if is_ogb(dataset):
        planned_runs = len(unique_seeds) if unique_seeds else None
    else:
        n_folds_values = {
            int(run["n_folds"])
            for run in deduped_runs
            if run.get("n_folds") is not None
        }
        if len(n_folds_values) == 1 and unique_seeds:
            planned_runs = next(iter(n_folds_values)) * len(unique_seeds)
    return {
        "dataset": dataset,
        "exp_name": exp_name,
        "files_found": len(files),
        "invalid_files": invalid,
        "unkeyed_files": unkeyed,
        "duplicates_removed": duplicates,
        "completed_runs": len(deduped_runs),
        "planned_runs": planned_runs,
        "runs": deduped_runs,
    }


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    return float(np.mean(clean)), float(np.std(clean))


def mean_only(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def aggregate_tu_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    acc_mean, acc_std = mean_std([run["test_acc"] for run in runs])
    loss_mean, loss_std = mean_std([run["test_loss"] for run in runs])
    iter_mean, iter_std = mean_std([run["last_global_iter"] for run in runs])
    time_mean, time_std = mean_std([run["avg_epoch_seconds"] for run in runs])
    return {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "test_loss_mean": loss_mean,
        "test_loss_std": loss_std,
        "avg_iter_mean": iter_mean,
        "avg_iter_std": iter_std,
        "avg_epoch_seconds_mean": time_mean,
        "avg_epoch_seconds_std": time_std,
    }


def aggregate_ogb_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    val_mean, val_std = mean_std([run["best_val_rocauc"] for run in runs])
    test_mean, test_std = mean_std([run["test_auc"] for run in runs])
    iter_mean, iter_std = mean_std([run["last_global_iter"] for run in runs])
    return {
        "val_auc_mean": val_mean,
        "val_auc_std": val_std,
        "test_auc_mean": test_mean,
        "test_auc_std": test_std,
        "avg_iter_mean": iter_mean,
        "avg_iter_std": iter_std,
    }


def extract_stability_traces(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    all_residuals: Dict[int, List[float]] = defaultdict(list)
    all_iters: Dict[int, List[float]] = defaultdict(list)

    for run in runs:
        for rec in run.get("history", []):
            epoch = rec.get("epoch")
            if epoch is None:
                continue
            if "global_residual" in rec:
                all_residuals[int(epoch)].append(float(rec["global_residual"]))
            if "global_iter" in rec:
                all_iters[int(epoch)].append(float(rec["global_iter"]))

    epochs = sorted(set(all_residuals) | set(all_iters))
    trace = []
    for epoch in epochs:
        entry: Dict[str, float] = {"epoch": float(epoch)}
        if epoch in all_residuals:
            residual_mean, residual_std = mean_std(all_residuals[epoch])
            entry["mean_residual"] = residual_mean
            entry["std_residual"] = residual_std
        if epoch in all_iters:
            iter_mean, iter_std = mean_std(all_iters[epoch])
            entry["mean_iter"] = iter_mean
            entry["std_iter"] = iter_std
        trace.append(entry)
    return trace


def build_table1_summary(results_dir: str | Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for paper_name, exp_name in TABLE1_EXPERIMENTS.items():
        ds_summary: Dict[str, Any] = {}
        for dataset in PAPER_DATASETS_TU:
            deduped = dedupe_runs(results_dir, dataset, exp_name)
            ds_summary[dataset] = {
                **aggregate_tu_runs(deduped["runs"]),
                "counts": {
                    "files_found": deduped["files_found"],
                    "completed_runs": deduped["completed_runs"],
                    "planned_runs": deduped["planned_runs"],
                    "duplicates_removed": deduped["duplicates_removed"],
                },
            }
        ogb_deduped = dedupe_runs(results_dir, "ogbg-molhiv", exp_name)
        ds_summary["ogbg-molhiv"] = {
            **aggregate_ogb_runs(ogb_deduped["runs"]),
            "counts": {
                "files_found": ogb_deduped["files_found"],
                "completed_runs": ogb_deduped["completed_runs"],
                "planned_runs": ogb_deduped["planned_runs"],
                "duplicates_removed": ogb_deduped["duplicates_removed"],
            },
        }
        summary[paper_name] = ds_summary
    return summary


def build_table2_summary(results_dir: str | Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for paper_name, exp_name in TABLE2_EXPERIMENTS.items():
        nci = dedupe_runs(results_dir, "NCI1", exp_name)
        ogb = dedupe_runs(results_dir, "ogbg-molhiv", exp_name)
        summary[paper_name] = {
            "nci1": {
                **aggregate_tu_runs(nci["runs"]),
                "counts": {
                    "files_found": nci["files_found"],
                    "completed_runs": nci["completed_runs"],
                    "planned_runs": nci["planned_runs"],
                    "duplicates_removed": nci["duplicates_removed"],
                },
            },
            "ogbg-molhiv": {
                **aggregate_ogb_runs(ogb["runs"]),
                "counts": {
                    "files_found": ogb["files_found"],
                    "completed_runs": ogb["completed_runs"],
                    "planned_runs": ogb["planned_runs"],
                    "duplicates_removed": ogb["duplicates_removed"],
                },
            },
        }
    return summary


def build_figure1_summary(results_dir: str | Path) -> Dict[str, Any]:
    traces: Dict[str, Any] = {}
    for dataset in FIGURE1_DATASETS:
        for paper_name, exp_name in FIGURE1_EXPERIMENTS.items():
            deduped = dedupe_runs(results_dir, dataset, exp_name)
            key = f"{paper_name} / {dataset}"
            traces[key] = {
                "counts": {
                    "files_found": deduped["files_found"],
                    "completed_runs": deduped["completed_runs"],
                    "planned_runs": deduped["planned_runs"],
                    "duplicates_removed": deduped["duplicates_removed"],
                },
                "trace": extract_stability_traces(deduped["runs"]),
            }
    return traces


def _run_map_by_key(runs: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out = {}
    for run in runs:
        key = tuple(run["key"])
        if key:
            out[key] = run
    return out


def _serialize_key(key: Tuple[Any, ...]) -> List[Any]:
    return [int(x) if isinstance(x, (int, np.integer)) else x for x in key]


def _pair_delta_stats(
    deltas: Sequence[float],
    positive_key: str,
    negative_key: str,
) -> Dict[str, Optional[float]]:
    positive = ties = negative = 0
    clean = []
    for delta in deltas:
        if delta is None:
            continue
        clean.append(float(delta))
        if delta > 0:
            positive += 1
        elif delta < 0:
            negative += 1
        else:
            ties += 1

    mean, std = mean_std(clean)
    return {
        "mean": mean,
        "std": std,
        "median": float(np.median(clean)) if clean else None,
        positive_key: positive,
        "ties": ties,
        negative_key: negative,
    }


def build_pair_dataset_summary(
    results_dir: str | Path,
    dataset: str,
    left_label: str,
    left_exp: str,
    right_label: str,
    right_exp: str,
    right_prefix: str = "post",
    left_prefix: str = "direct",
) -> Dict[str, Any]:
    left = dedupe_runs(results_dir, dataset, left_exp)
    right = dedupe_runs(results_dir, dataset, right_exp)
    left_map = _run_map_by_key(left["runs"])
    right_map = _run_map_by_key(right["runs"])
    matched_keys = sorted(set(left_map) & set(right_map))

    left_runs = [left_map[key] for key in matched_keys]
    right_runs = [right_map[key] for key in matched_keys]
    paired_rows = []
    delta_acc = []
    delta_iter = []
    delta_time = []

    for key in matched_keys:
        left_run = left_map[key]
        right_run = right_map[key]
        acc_delta = right_run["test_acc"] - left_run["test_acc"]
        iter_delta = None
        if right_run["last_global_iter"] is not None and left_run["last_global_iter"] is not None:
            iter_delta = right_run["last_global_iter"] - left_run["last_global_iter"]
            delta_iter.append(iter_delta)
        time_delta = None
        if right_run["avg_epoch_seconds"] is not None and left_run["avg_epoch_seconds"] is not None:
            time_delta = right_run["avg_epoch_seconds"] - left_run["avg_epoch_seconds"]
            delta_time.append(time_delta)

        delta_acc.append(acc_delta)
        paired_rows.append(
            {
                "key": _serialize_key(key),
                f"{right_prefix}_test_acc": right_run["test_acc"],
                f"{left_prefix}_test_acc": left_run["test_acc"],
                "delta_acc": acc_delta,
                f"{right_prefix}_avg_iter": right_run["last_global_iter"],
                f"{left_prefix}_avg_iter": left_run["last_global_iter"],
                "delta_iter": iter_delta,
                f"{right_prefix}_avg_epoch_seconds": right_run["avg_epoch_seconds"],
                f"{left_prefix}_avg_epoch_seconds": left_run["avg_epoch_seconds"],
                "delta_avg_epoch_seconds": time_delta,
                f"{right_prefix}_test_loss": right_run["test_loss"],
                f"{left_prefix}_test_loss": left_run["test_loss"],
            }
        )

    return {
        "dataset": dataset,
        "matched_runs": len(matched_keys),
        "matched_keys": [_serialize_key(key) for key in matched_keys],
        left_prefix: {
            "label": left_label,
            "exp_name": left_exp,
            "raw_counts": {
                "files_found": left["files_found"],
                "completed_runs": left["completed_runs"],
                "planned_runs": left["planned_runs"],
                "duplicates_removed": left["duplicates_removed"],
            },
            "matched_stats": aggregate_tu_runs(left_runs),
        },
        right_prefix: {
            "label": right_label,
            "exp_name": right_exp,
            "raw_counts": {
                "files_found": right["files_found"],
                "completed_runs": right["completed_runs"],
                "planned_runs": right["planned_runs"],
                "duplicates_removed": right["duplicates_removed"],
            },
            "matched_stats": aggregate_tu_runs(right_runs),
        },
        "delta_acc": _pair_delta_stats(
            delta_acc,
            f"wins_{right_prefix}",
            f"wins_{left_prefix}",
        ),
        "delta_iter": _pair_delta_stats(
            delta_iter,
            f"{right_prefix}_more_iter",
            f"{right_prefix}_fewer_iter",
        ),
        "delta_time": _pair_delta_stats(
            delta_time,
            f"{right_prefix}_slower",
            f"{right_prefix}_faster",
        ),
        "rows": paired_rows,
    }


def build_direct_post_dataset_summary(results_dir: str | Path, dataset: str) -> Dict[str, Any]:
    return build_pair_dataset_summary(
        results_dir,
        dataset,
        left_label="Direct residual",
        left_exp="ablation_direct_q",
        right_label="Post-backbone residual",
        right_exp="QIGNN_mainline",
        right_prefix="post",
        left_prefix="direct",
    )


def build_direct_post_summary(results_dir: str | Path) -> Dict[str, Any]:
    return {
        dataset: build_direct_post_dataset_summary(results_dir, dataset)
        for dataset in DIRECT_POST_DATASETS
    }


def build_residual_family_dataset_summary(results_dir: str | Path, dataset: str) -> Dict[str, Any]:
    run_maps: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
    raw_variants: Dict[str, Any] = {}

    for slug, (label, exp_name) in RESIDUAL_FAMILY_EXPERIMENTS.items():
        deduped = dedupe_runs(results_dir, dataset, exp_name)
        run_maps[slug] = _run_map_by_key(deduped["runs"])
        raw_variants[slug] = {
            "label": label,
            "exp_name": exp_name,
            "stats": aggregate_tu_runs(deduped["runs"]),
            "counts": {
                "files_found": deduped["files_found"],
                "completed_runs": deduped["completed_runs"],
                "planned_runs": deduped["planned_runs"],
                "duplicates_removed": deduped["duplicates_removed"],
            },
        }

    available_sets = [set(run_maps[slug]) for slug in RESIDUAL_FAMILY_EXPERIMENTS if run_maps[slug]]
    four_way_keys = sorted(set.intersection(*available_sets)) if available_sets and len(available_sets) == len(RESIDUAL_FAMILY_EXPERIMENTS) else []

    four_way_matched = {
        slug: {
            "label": raw_variants[slug]["label"],
            "stats": aggregate_tu_runs([run_maps[slug][key] for key in four_way_keys]),
            "matched_runs": len(four_way_keys),
        }
        for slug in RESIDUAL_FAMILY_EXPERIMENTS
    }

    quantum_pair = build_pair_dataset_summary(
        results_dir,
        dataset,
        left_label="Quantum direct",
        left_exp="ablation_direct_q",
        right_label="Quantum post-backbone",
        right_exp="QIGNN_mainline",
        right_prefix="post",
        left_prefix="direct",
    )
    classical_pair = build_pair_dataset_summary(
        results_dir,
        dataset,
        left_label="Classical direct",
        left_exp="resfam_direct_c",
        right_label="Classical post-backbone",
        right_exp="resfam_post_c",
        right_prefix="post",
        left_prefix="direct",
    )

    return {
        "dataset": dataset,
        "raw_variants": raw_variants,
        "four_way_overlap_keys": [_serialize_key(key) for key in four_way_keys],
        "four_way_matched": four_way_matched,
        "quantum_pair": quantum_pair,
        "classical_pair": classical_pair,
    }


def build_residual_family_summary(results_dir: str | Path) -> Dict[str, Any]:
    return {
        dataset: build_residual_family_dataset_summary(results_dir, dataset)
        for dataset in RESIDUAL_FAMILY_DATASETS
    }


def build_nci1_core_summary(results_dir: str | Path) -> Dict[str, Any]:
    raw_variants: Dict[str, Any] = {}
    run_maps: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}

    for slug, (label, exp_name) in NCI1_CORE_EXPERIMENTS.items():
        deduped = dedupe_runs(results_dir, "NCI1", exp_name)
        raw_variants[slug] = {
            "label": label,
            "exp_name": exp_name,
            "stats": aggregate_tu_runs(deduped["runs"]),
            "counts": {
                "files_found": deduped["files_found"],
                "completed_runs": deduped["completed_runs"],
                "planned_runs": deduped["planned_runs"],
                "duplicates_removed": deduped["duplicates_removed"],
            },
        }
        run_maps[slug] = _run_map_by_key(deduped["runs"])

    four_way_keys = sorted(set.intersection(*(set(run_maps[slug]) for slug in NCI1_CORE_EXPERIMENTS)))
    post_direct_keys = sorted(set(run_maps["post"]) & set(run_maps["direct"]))

    matched_four_way = {
        slug: {
            "label": raw_variants[slug]["label"],
            "stats": aggregate_tu_runs([run_maps[slug][key] for key in four_way_keys]),
            "matched_runs": len(four_way_keys),
        }
        for slug in NCI1_CORE_EXPERIMENTS
    }

    paired_rows = []
    delta_acc = []
    delta_iter = []
    post_wins = post_ties = direct_wins = 0
    post_fewer_iter = iter_ties = post_more_iter = 0

    for key in post_direct_keys:
        post_run = run_maps["post"][key]
        direct_run = run_maps["direct"][key]
        acc_delta = post_run["test_acc"] - direct_run["test_acc"]
        iter_delta = None
        if post_run["last_global_iter"] is not None and direct_run["last_global_iter"] is not None:
            iter_delta = post_run["last_global_iter"] - direct_run["last_global_iter"]
            delta_iter.append(iter_delta)
            if iter_delta < 0:
                post_fewer_iter += 1
            elif iter_delta > 0:
                post_more_iter += 1
            else:
                iter_ties += 1
        delta_acc.append(acc_delta)
        if acc_delta > 0:
            post_wins += 1
        elif acc_delta < 0:
            direct_wins += 1
        else:
            post_ties += 1
        paired_rows.append(
            {
                "key": _serialize_key(key),
                "post_test_acc": post_run["test_acc"],
                "direct_test_acc": direct_run["test_acc"],
                "delta_acc": acc_delta,
                "post_avg_iter": post_run["last_global_iter"],
                "direct_avg_iter": direct_run["last_global_iter"],
                "delta_iter": iter_delta,
                "post_test_loss": post_run["test_loss"],
                "direct_test_loss": direct_run["test_loss"],
            }
        )

    pair_summary = {
        "matched_runs": len(post_direct_keys),
        "delta_acc": {
            "mean": mean_only(delta_acc),
            "std": mean_std(delta_acc)[1],
            "median": float(np.median(delta_acc)) if delta_acc else None,
            "wins_post": post_wins,
            "ties": post_ties,
            "wins_direct": direct_wins,
        },
        "delta_iter": {
            "mean": mean_only(delta_iter),
            "std": mean_std(delta_iter)[1],
            "median": float(np.median(delta_iter)) if delta_iter else None,
            "post_fewer_iter": post_fewer_iter,
            "ties": iter_ties,
            "post_more_iter": post_more_iter,
        },
        "rows": paired_rows,
    }

    return {
        "raw_variants": raw_variants,
        "four_way_overlap_keys": [_serialize_key(key) for key in four_way_keys],
        "four_way_matched": matched_four_way,
        "post_vs_direct": pair_summary,
    }


def build_summary(results_dir: str | Path) -> Dict[str, Any]:
    return {
        "table1": build_table1_summary(results_dir),
        "table2": build_table2_summary(results_dir),
        "figure1": build_figure1_summary(results_dir),
        "nci1_core": build_nci1_core_summary(results_dir),
        "direct_post_pairs": build_direct_post_summary(results_dir),
        "residual_family": build_residual_family_summary(results_dir),
    }


def fmt_pct(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    if std is not None:
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.1f}"


def fmt_auc(mean: Optional[float], std: Optional[float] = None) -> str:
    if mean is None:
        return "---"
    if std is not None:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def fmt_iter(val: Optional[float]) -> str:
    if val is None:
        return "---"
    return f"{val:.1f}"


def print_table1(summary: Dict[str, Any]) -> None:
    print("=" * 90)
    print("TABLE 1: Benchmark Results")
    print("=" * 90)
    header = f"{'Model':<12}"
    for dataset in PAPER_DATASETS_TU:
        header += f" | {dataset:>18}"
    header += f" | {'molhiv Val':>18} | {'molhiv Test':>18}"
    print(header)
    print("-" * len(header))

    for paper_name in TABLE1_EXPERIMENTS:
        row = f"{paper_name:<12}"
        for dataset in PAPER_DATASETS_TU:
            ds_summary = summary[paper_name][dataset]
            row += f" | {fmt_pct(ds_summary['accuracy_mean'], ds_summary['accuracy_std']):>18}"
        ogb_summary = summary[paper_name]["ogbg-molhiv"]
        row += f" | {fmt_auc(ogb_summary['val_auc_mean'], ogb_summary['val_auc_std']):>18}"
        row += f" | {fmt_auc(ogb_summary['test_auc_mean'], ogb_summary['test_auc_std']):>18}"
        print(row)
    print()
    print("Note: IGNN and GIND baselines require separate codebases.")
    print()


def print_table2(summary: Dict[str, Any]) -> None:
    print("=" * 118)
    print("TABLE 2: Conditioning-Path Ablation")
    print("=" * 118)
    header = (
        f"{'Variant':<42} | {'NCI1':>18} | {'NCI1 Runs':>10} | "
        f"{'molhiv Val':>18} | {'molhiv Test':>18} | {'Avg Iter':>8}"
    )
    print(header)
    print("-" * len(header))

    for paper_name in TABLE2_EXPERIMENTS:
        variant_summary = summary[paper_name]
        nci = variant_summary["nci1"]
        ogb = variant_summary["ogbg-molhiv"]
        counts = nci["counts"]
        runs = f"{counts['completed_runs']}"
        if counts["planned_runs"] is not None:
            runs = f"{counts['completed_runs']}/{counts['planned_runs']}"
        row = f"{paper_name:<42}"
        row += f" | {fmt_pct(nci['accuracy_mean'], nci['accuracy_std']):>18}"
        row += f" | {runs:>10}"
        row += f" | {fmt_auc(ogb['val_auc_mean'], ogb['val_auc_std']):>18}"
        row += f" | {fmt_auc(ogb['test_auc_mean'], ogb['test_auc_std']):>18}"
        row += f" | {fmt_iter(nci['avg_iter_mean']):>8}"
        print(row)
    print()


def print_figure1(summary: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    print("=" * 90)
    print("FIGURE 1: Stability Diagnostics (per-epoch traces)")
    print("=" * 90)

    traces = summary
    serializable = {}
    for key, payload in traces.items():
        trace = payload["trace"]
        serializable[key] = payload
        if trace:
            first = trace[0]
            last = trace[-1]
            print(f"\n  {key}: {len(trace)} epochs")
            print(
                f"    Epoch {int(first['epoch'])}: residual={first.get('mean_residual', '?'):.2e}, "
                f"iter={first.get('mean_iter', '?')}"
            )
            print(
                f"    Epoch {int(last['epoch'])}: residual={last.get('mean_residual', '?'):.2e}, "
                f"iter={last.get('mean_iter', '?')}"
            )
        else:
            print(f"\n  {key}: NO DATA")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "figure1_traces.json")
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Traces saved to: {out_path}")
    print()


def print_nci1_core(summary: Dict[str, Any]) -> None:
    print("=" * 118)
    print("NCI1 CORE ABLATION SUMMARY")
    print("=" * 118)
    header = f"{'Variant':<22} | {'Raw Acc':>18} | {'Runs':>10} | {'4-way Acc':>18} | {'Avg Iter':>8}"
    print(header)
    print("-" * len(header))
    for slug in NCI1_CORE_EXPERIMENTS:
        raw = summary["raw_variants"][slug]
        matched = summary["four_way_matched"][slug]
        counts = raw["counts"]
        runs = f"{counts['completed_runs']}"
        if counts["planned_runs"] is not None:
            runs = f"{counts['completed_runs']}/{counts['planned_runs']}"
        row = f"{raw['label']:<22}"
        row += f" | {fmt_pct(raw['stats']['accuracy_mean'], raw['stats']['accuracy_std']):>18}"
        row += f" | {runs:>10}"
        row += f" | {fmt_pct(matched['stats']['accuracy_mean'], matched['stats']['accuracy_std']):>18}"
        row += f" | {fmt_iter(raw['stats']['avg_iter_mean']):>8}"
        print(row)

    pair = summary["post_vs_direct"]
    print()
    print(
        "Post vs Direct "
        f"(matched runs={pair['matched_runs']}): "
        f"delta_acc mean={pair['delta_acc']['mean']:.3f}, "
        f"median={pair['delta_acc']['median']:.3f}, "
        f"wins/ties/losses={pair['delta_acc']['wins_post']}/"
        f"{pair['delta_acc']['ties']}/{pair['delta_acc']['wins_direct']}; "
        f"delta_iter mean={pair['delta_iter']['mean']:.3f}, "
        f"median={pair['delta_iter']['median']:.3f}, "
        f"post_fewer/ties/more={pair['delta_iter']['post_fewer_iter']}/"
        f"{pair['delta_iter']['ties']}/{pair['delta_iter']['post_more_iter']}"
    )
    print()


def print_direct_post_pairs(summary: Dict[str, Any]) -> None:
    print("=" * 132)
    print("MATCHED DIRECT VS POST SUMMARY")
    print("=" * 132)
    header = (
        f"{'Dataset':<10} | {'Matched':>7} | {'Direct Acc':>18} | {'Post Acc':>18} | "
        f"{'Δ Acc':>17} | {'Δ Iter':>17} | {'Δ Time/Epoch':>17}"
    )
    print(header)
    print("-" * len(header))

    for dataset in DIRECT_POST_DATASETS:
        pair = summary[dataset]
        direct = pair["direct"]["matched_stats"]
        post = pair["post"]["matched_stats"]
        row = f"{dataset:<10}"
        row += f" | {pair['matched_runs']:>7}"
        row += f" | {fmt_pct(direct['accuracy_mean'], direct['accuracy_std']):>18}"
        row += f" | {fmt_pct(post['accuracy_mean'], post['accuracy_std']):>18}"
        row += (
            f" | {fmt_pct(pair['delta_acc']['mean'], pair['delta_acc']['std']):>17}"
            f" | {fmt_pct(pair['delta_iter']['mean'], pair['delta_iter']['std']):>17}"
            f" | {fmt_pct(pair['delta_time']['mean'], pair['delta_time']['std']):>17}"
        )
        print(row)

    print()


def print_residual_family(summary: Dict[str, Any]) -> None:
    print("=" * 138)
    print("RESIDUAL FAMILY SUMMARY")
    print("=" * 138)
    header = (
        f"{'Dataset':<10} | {'Pair':<18} | {'Matched':>7} | "
        f"{'Direct Acc':>18} | {'Post Acc':>18} | {'Δ Acc':>17} | {'Δ Iter':>17} | {'Δ Time/Epoch':>17}"
    )
    print(header)
    print("-" * len(header))

    for dataset in RESIDUAL_FAMILY_DATASETS:
        ds = summary[dataset]
        for pair_name, pair in [("quantum", ds["quantum_pair"]), ("classical", ds["classical_pair"])]:
            direct = pair["direct"]["matched_stats"]
            post = pair["post"]["matched_stats"]
            row = f"{dataset:<10} | {pair_name:<18} | {pair['matched_runs']:>7}"
            row += f" | {fmt_pct(direct['accuracy_mean'], direct['accuracy_std']):>18}"
            row += f" | {fmt_pct(post['accuracy_mean'], post['accuracy_std']):>18}"
            row += (
                f" | {fmt_pct(pair['delta_acc']['mean'], pair['delta_acc']['std']):>17}"
                f" | {fmt_pct(pair['delta_iter']['mean'], pair['delta_iter']['std']):>17}"
                f" | {fmt_pct(pair['delta_time']['mean'], pair['delta_time']['std']):>17}"
            )
            print(row)

    print()
    for dataset in RESIDUAL_FAMILY_DATASETS:
        ds = summary[dataset]
        matched = ds["four_way_matched"]
        print(
            f"{dataset} four-way matched runs={matched['quantum_post']['matched_runs']}: "
            f"Q-post {fmt_pct(matched['quantum_post']['stats']['accuracy_mean'], matched['quantum_post']['stats']['accuracy_std'])}, "
            f"Q-direct {fmt_pct(matched['quantum_direct']['stats']['accuracy_mean'], matched['quantum_direct']['stats']['accuracy_std'])}, "
            f"C-post {fmt_pct(matched['classical_post']['stats']['accuracy_mean'], matched['classical_post']['stats']['accuracy_std'])}, "
            f"C-direct {fmt_pct(matched['classical_direct']['stats']['accuracy_mean'], matched['classical_direct']['stats']['accuracy_std'])}"
        )
        for label, pair in [("quantum", ds["quantum_pair"]), ("classical", ds["classical_pair"])]:
            print(
                f"  {label}: "
                f"acc wins/ties/losses={pair['delta_acc']['wins_post']}/{pair['delta_acc']['ties']}/{pair['delta_acc']['wins_direct']}; "
                f"iter post-fewer/ties/more={pair['delta_iter']['post_fewer_iter']}/{pair['delta_iter']['ties']}/{pair['delta_iter']['post_more_iter']}; "
                f"time post-faster/ties/slower={pair['delta_time']['post_faster']}/{pair['delta_time']['ties']}/{pair['delta_time']['post_slower']}"
            )
    print()
    for dataset in DIRECT_POST_DATASETS:
        pair = summary[dataset]
        print(
            f"{dataset}: "
            f"acc wins/ties/losses={pair['delta_acc']['wins_post']}/"
            f"{pair['delta_acc']['ties']}/{pair['delta_acc']['wins_direct']}; "
            f"iter post-fewer/ties/more={pair['delta_iter']['post_fewer_iter']}/"
            f"{pair['delta_iter']['ties']}/{pair['delta_iter']['post_more_iter']}; "
            f"time post-faster/ties/slower={pair['delta_time']['post_faster']}/"
            f"{pair['delta_time']['ties']}/{pair['delta_time']['post_slower']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize QIGNN results for paper")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results directory")
    parser.add_argument("--table1", action="store_true", help="Print Table 1")
    parser.add_argument("--table2", action="store_true", help="Print Table 2")
    parser.add_argument("--figure1", action="store_true", help="Print Figure 1 traces")
    parser.add_argument("--nci1_core", action="store_true", help="Print matched NCI1 core ablation summary")
    parser.add_argument(
        "--direct_post_pairs",
        action="store_true",
        help="Print matched direct-vs-post summaries for TU datasets",
    )
    parser.add_argument(
        "--residual_family",
        action="store_true",
        help="Print matched four-variant summaries for quantum/classical residual families",
    )
    parser.add_argument("--all", action="store_true", help="Print all artifacts")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for saving trace JSONs and optional summary JSON",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path for a machine-readable summary JSON",
    )
    args = parser.parse_args()

    if not any([args.table1, args.table2, args.figure1, args.nci1_core, args.direct_post_pairs, args.residual_family, args.all]):
        args.all = True

    summary = build_summary(args.results_dir)

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    if args.all or args.table1:
        print_table1(summary["table1"])
    if args.all or args.table2:
        print_table2(summary["table2"])
    if args.all or args.figure1:
        print_figure1(summary["figure1"], output_dir=args.output_dir)
    if args.all or args.nci1_core:
        print_nci1_core(summary["nci1_core"])
    if args.all or args.direct_post_pairs:
        print_direct_post_pairs(summary["direct_post_pairs"])
    if args.all or args.residual_family:
        print_residual_family(summary["residual_family"])


if __name__ == "__main__":
    main()
