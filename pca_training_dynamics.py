#!/usr/bin/env python3
"""
PCA on training / solver dynamics from train.py result JSONs.

The default workflow is run-level PCA:
  1. Discover result*.json under results/<dataset>/<experiment>/.
  2. Load each file; infer paper-oriented variant label from path + config.
  3. Build one row per run from aggregated training / solver summaries.
  4. Median-impute missing numeric features on the fit subset only.
  5. Standardize features, fit PCA, and write scores / model / manifest.

Epoch mode is retained for exploratory analysis, but it mixes within-run training
chronology with between-run variation and should not be treated as the default
paper analysis.

Usage:
  cd QIGNN_release2
  python pca_training_dynamics.py --results_dir results --plot
  python pca_training_dynamics.py --results_dir results --mode run --datasets NCI1 --plot

  # Match qignn2.tex Table 2 (NCI1 + ogbg-molhiv, four reported conditioning variants only):
  python pca_training_dynamics.py --paper_table2 --paper_reported_ablations --mode run --plot

  # Exploratory epoch PCA on the full repo sweep:
  python pca_training_dynamics.py --paper_table2 --mode epoch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Optional plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


# ---------------------------------------------------------------------------
# Align with summarize_results.py experiment layout
# ---------------------------------------------------------------------------

# All experiment-folder names used by summarize_results.py Table 2 / sweeps.
TABLE2_EXPERIMENTS = [
    "ablation_classical",
    "ablation_external_q",
    "ablation_film",
    "ablation_direct_q",
    "ablation_classical_resid",
    "QIGNN_mainline",
]

# qignn2.tex Table (conditioning-path ablation): four rows only. Direct residual
# is explicitly excluded from the reported ablation; matched classical residual
# is a repo extra not named in that table.
PAPER_REPORTED_ABLATION_EXPERIMENTS = [
    "ablation_classical",
    "ablation_external_q",
    "ablation_film",
    "QIGNN_mainline",
]

PAPER_TABLE2_DATASETS = ["NCI1", "ogbg-molhiv"]

# History keys we may use as features (epoch mode); union discovered per corpus.
CORE_EPOCH_KEYS = [
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "train_val_gap",
    "time",
]

SOLVER_EPOCH_KEYS = [
    "local_iter",
    "local_residual",
    "local_converged_pct",
    "global_iter",
    "global_residual",
    "global_converged_pct",
    "global_L_g",
]

EXTRA_QB_KEYS = [
    "global_Q_mean",
    "global_Q_std",
    "global_Q_abs_mean",
    "global_Q_max_abs",
    "global_Q_norm",
    "global_B_mean",
    "global_B_std",
    "global_B_abs_mean",
    "global_B_max_abs",
    "global_B_norm",
    "global_B_pre_norm",
    "global_B_post_norm",
    "global_WZA_abs_mean",
    "global_WZA_norm",
    "global_Q_B_ratio_norm",
    "global_Q_WZA_ratio_norm",
    "global_Q_B_ratio_abs",
    "global_Q_WZA_ratio_abs",
]

OGB_KEYS = ["val_rocauc"]

DEFAULT_RUN_STAT_KEYS = [
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "train_val_gap",
    "time",
    "global_iter",
    "global_residual",
    "global_converged_pct",
    "global_L_g",
    "val_rocauc",
]

DEFAULT_RUN_EXTRA_KEYS = [
    "global_Q_mean",
    "global_Q_std",
    "global_Q_abs_mean",
    "global_Q_max_abs",
    "global_Q_norm",
    "global_B_mean",
    "global_B_std",
    "global_B_abs_mean",
    "global_B_max_abs",
    "global_B_norm",
    "global_B_pre_norm",
    "global_B_post_norm",
    "global_WZA_abs_mean",
    "global_WZA_norm",
    "global_Q_B_ratio_norm",
    "global_Q_WZA_ratio_norm",
    "global_Q_B_ratio_abs",
    "global_Q_WZA_ratio_abs",
]


def find_result_files(results_dir: str, dataset: str, exp_name: str) -> List[Path]:
    """Same discovery rule as summarize_results.find_result_files."""
    search_dir = Path(results_dir) / dataset / exp_name
    if not search_dir.exists():
        return []
    return sorted(search_dir.rglob("result*.json"))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _exp_slug_from_path(results_dir: Path, json_path: Path, dataset: str) -> str:
    """First path component under results/<dataset>/."""
    try:
        rel = json_path.resolve().relative_to((results_dir / dataset).resolve())
    except ValueError:
        parts = json_path.parts
        if dataset in parts:
            i = parts.index(dataset)
            if i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"
    parts = rel.parts
    return parts[0] if parts else "unknown"


def infer_variant_label(exp_slug: str, cfg: Dict[str, Any]) -> str:
    """
    Human-readable label for plots. Prefer experiment folder (paper sweep);
    use config flags only for disambiguation when slug is generic.
    """
    slug_map = {
        "QIGNN_mainline": "post_backbone (QIGNN_mainline)",
        "ablation_classical": "classical backbone only",
        "ablation_external_q": "external graph-level Q",
        "ablation_film": "static FiLM",
        "ablation_direct_q": "direct residual (repo; not in paper table)",
        "ablation_classical_resid": "matched classical residual",
        "GIN_baseline": "GIN baseline",
    }
    if exp_slug in slug_map:
        return slug_map[exp_slug]
    # Fallback from config if exp_name like "QIGNN_mainline/NCI1"
    en = cfg.get("exp_name") or ""
    if isinstance(en, str) and "/" in en:
        root = en.split("/")[0]
        if root in slug_map:
            return slug_map[root]
    return exp_slug or "unknown"


def collect_numeric_keys_from_history(
    files_data: List[Tuple[Path, Dict[str, Any]]],
    min_frac: float,
) -> List[str]:
    """Scan histories; return keys that are numeric in >= min_frac of epochs (epoch mode)."""
    key_hits: Dict[str, int] = {}
    key_total = 0
    for _, data in files_data:
        hist = data.get("history") or []
        for rec in hist:
            key_total += 1
            for k, v in rec.items():
                if k == "epoch":
                    continue
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    key_hits[k] = key_hits.get(k, 0) + 1
    if key_total == 0:
        return []
    thresh = min_frac * key_total
    ordered = CORE_EPOCH_KEYS + SOLVER_EPOCH_KEYS + OGB_KEYS + EXTRA_QB_KEYS
    seen = set()
    out = []
    for k in ordered:
        if key_hits.get(k, 0) >= thresh and k not in seen:
            out.append(k)
            seen.add(k)
    for k in sorted(key_hits.keys()):
        if k in seen:
            continue
        if key_hits[k] >= thresh:
            out.append(k)
            seen.add(k)
    return out


def epoch_rows_from_result(
    json_path: Path,
    data: Dict[str, Any],
    results_dir: Path,
    dataset: str,
    feature_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    cfg = data.get("config") or {}
    exp_slug = _exp_slug_from_path(results_dir, json_path, dataset)
    variant = infer_variant_label(exp_slug, cfg)
    run_id = json_path.stem
    fold_idx = cfg.get("fold_idx")
    seed = cfg.get("seed")
    rows = []
    for rec in data.get("history") or []:
        row: Dict[str, Any] = {
            "run_id": run_id,
            "json_path": str(json_path),
            "dataset": dataset,
            "exp_slug": exp_slug,
            "variant": variant,
            "fold_idx": fold_idx,
            "seed": seed,
            "epoch": rec.get("epoch"),
        }
        ok = True
        for k in feature_keys:
            if k not in rec:
                row[k] = float("nan")
            else:
                v = rec[k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = float(v)
                else:
                    row[k] = float("nan")
        # Require core metrics present for a valid training row
        for req in ("train_loss", "train_acc", "val_loss", "val_acc"):
            if req in feature_keys and math.isnan(row.get(req, float("nan"))):
                ok = False
                break
        if ok:
            rows.append(row)
    return rows


def run_summary_row(
    json_path: Path,
    data: Dict[str, Any],
    results_dir: Path,
    dataset: str,
    keys_for_stats: Sequence[str],
) -> Optional[Dict[str, Any]]:
    hist = data.get("history") or []
    if not hist:
        return None
    cfg = data.get("config") or {}
    exp_slug = _exp_slug_from_path(results_dir, json_path, dataset)
    variant = infer_variant_label(exp_slug, cfg)
    run_id = json_path.stem

    def series(k: str) -> List[float]:
        out = []
        for rec in hist:
            if k not in rec:
                continue
            v = rec[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not (isinstance(v, float) and math.isnan(v)):
                    out.append(float(v))
        return out

    row: Dict[str, Any] = {
        "run_id": run_id,
        "json_path": str(json_path),
        "dataset": dataset,
        "exp_slug": exp_slug,
        "variant": variant,
        "fold_idx": cfg.get("fold_idx"),
        "seed": cfg.get("seed"),
        "test_acc": data.get("test_acc"),
        "n_epochs": len(hist),
    }
    for k in keys_for_stats:
        s = series(k)
        if s:
            row[f"{k}_mean"] = float(np.mean(s))
            row[f"{k}_std"] = float(np.std(s))
            row[f"{k}_last"] = float(s[-1])
        else:
            row[f"{k}_mean"] = float("nan")
            row[f"{k}_std"] = float("nan")
            row[f"{k}_last"] = float("nan")

    return row


def build_matrix(
    rows: List[Dict[str, Any]],
    feature_cols: List[str],
    fit_mask: np.ndarray,
) -> Tuple[np.ndarray, List[str], SimpleImputer]:
    """
    Returns X_all (n_samples, n_features), kept column names, and the fitted imputer.
    All preprocessing decisions are fit on the fit subset only to avoid leakage.
    """
    n = len(rows)
    X = np.zeros((n, len(feature_cols)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, k in enumerate(feature_cols):
            v = row.get(k, float("nan"))
            X[i, j] = float(v) if v is not None else float("nan")

    X_fit = X[fit_mask]
    if X_fit.shape[0] == 0:
        raise ValueError("No fit rows available to build PCA matrix.")

    # Drop all-NaN columns based on the fit subset so held-out rows do not affect
    # feature selection or imputation statistics.
    nonempty_cols = [j for j in range(X.shape[1]) if not np.all(np.isnan(X_fit[:, j]))]
    if not nonempty_cols:
        raise ValueError("All features are NaN; cannot build matrix.")
    X_sub = X[:, nonempty_cols]
    X_fit_sub = X_fit[:, nonempty_cols]
    names_sub = [feature_cols[j] for j in nonempty_cols]

    imputer = SimpleImputer(strategy="median")
    X_fit_imp = imputer.fit_transform(X_fit_sub)
    X_imp = imputer.transform(X_sub)

    keep_idx = []
    kept_names = []
    for j, name in enumerate(names_sub):
        col = X_fit_imp[:, j]
        if np.nanstd(col) < 1e-12:
            continue
        keep_idx.append(j)
        kept_names.append(name)

    if not keep_idx:
        raise ValueError("No non-constant features after imputation.")

    X_final = X_imp[:, keep_idx]
    return X_final, kept_names, imputer


def save_scores_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    scores: np.ndarray,
    meta_keys: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pc = scores.shape[1]
    pc_names = [f"PC{i+1}" for i in range(n_pc)]
    fieldnames = list(meta_keys) + pc_names
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k) for k in meta_keys}
            for j, name in enumerate(pc_names):
                out[name] = scores[i, j]
            w.writerow(out)


def save_pca_json(
    path: Path,
    feature_names: List[str],
    pca: Any,
    scaler: StandardScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_components": pca.components_.tolist(),
        "n_components": int(pca.n_components_),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_pc_scatter(
    rows: List[Dict[str, Any]],
    scores: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available for --plot")
    vlist = sorted({r["variant"] for r in rows})
    dlist = sorted({r["dataset"] for r in rows})
    cmap = matplotlib.colormaps["tab10"]
    v_to_c = {v: cmap(i / max(len(vlist), 1)) for i, v in enumerate(vlist)}

    n_ds = len(dlist)
    fig_w = max(7, 4.6 * n_ds)
    fig, axes = plt.subplots(1, n_ds, figsize=(fig_w, 5.5), squeeze=False)
    axes_flat = axes[0]

    handles = []
    labels = []
    for ax, ds in zip(axes_flat, dlist):
        ds_rows = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        for v in vlist:
            idx = [i for i in ds_rows if rows[i]["variant"] == v]
            if not idx:
                continue
            xx = scores[idx, 0]
            yy = scores[idx, 1] if scores.shape[1] > 1 else np.zeros_like(xx)
            sc = ax.scatter(xx, yy, label=v[:40], alpha=0.45, s=18, c=[v_to_c[v]])
            if v not in labels:
                handles.append(sc)
                labels.append(v)
        ax.set_title(str(ds))
        ax.set_xlabel("PC1")
        if ax is axes_flat[0]:
            ax.set_ylabel("PC2" if scores.shape[1] > 1 else "(single PC)")
        ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)

    fig.suptitle(title)
    if handles:
        fig.legend(handles, [l[:40] for l in labels], bbox_to_anchor=(1.02, 0.98), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gather_files(
    results_dir: str,
    datasets: Sequence[str],
    experiments: Sequence[str],
) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs: List[Tuple[str, Path]] = []
    matched_scope: List[Tuple[str, str]] = []
    missing_scope: List[Tuple[str, str]] = []
    for ds in datasets:
        for exp in experiments:
            files = find_result_files(results_dir, ds, exp)
            if files:
                matched_scope.append((ds, exp))
            else:
                missing_scope.append((ds, exp))
            for p in files:
                pairs.append((ds, p))
    return pairs, matched_scope, missing_scope


def collect_run_feature_keys(rows: List[Dict[str, Any]]) -> List[str]:
    stat_suffixes = ("_mean", "_std", "_last")
    key_set: Set[str] = set()
    for row in rows:
        for key in row.keys():
            if key.endswith(stat_suffixes):
                key_set.add(key)
    ordered = []
    preferred = DEFAULT_RUN_STAT_KEYS + DEFAULT_RUN_EXTRA_KEYS
    for base in preferred:
        for suffix in stat_suffixes:
            key = f"{base}{suffix}"
            if key in key_set:
                ordered.append(key)
                key_set.remove(key)
    ordered.extend(sorted(key_set))
    return ordered


def scope_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    loaded_datasets = sorted({str(r.get("dataset")) for r in rows if r.get("dataset") is not None})
    loaded_experiments = sorted({str(r.get("exp_slug")) for r in rows if r.get("exp_slug") is not None})
    loaded_variants = sorted({str(r.get("variant")) for r in rows if r.get("variant") is not None})
    variant_counts: Dict[str, int] = {}
    for variant in loaded_variants:
        variant_counts[variant] = sum(1 for r in rows if r.get("variant") == variant)
    return {
        "loaded_datasets": loaded_datasets,
        "loaded_experiments": loaded_experiments,
        "loaded_variants": loaded_variants,
        "variant_row_counts": variant_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA on QIGNN result JSON training dynamics")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("epoch", "run"),
        default="run",
        help="run: one row per JSON with mean/std/last stats; epoch: one row per history step",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset folder names under results/. Default: paper Table2 pair if --paper_table2 else all unique from experiments",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Subfolders under each dataset. If provided (non-empty), overrides "
            "--paper_reported_ablations. Default: TABLE2_EXPERIMENTS (full repo list)."
        ),
    )
    parser.add_argument(
        "--paper_table2",
        action="store_true",
        help=(
            "Restrict datasets to %s (Table 2 geography in qignn2.tex). "
            "Experiment folders still follow --experiments / defaults unless "
            "--paper_reported_ablations is set."
        )
        % (PAPER_TABLE2_DATASETS,),
    )
    parser.add_argument(
        "--paper_reported_ablations",
        action="store_true",
        help=(
            "Use only the four conditioning variants in qignn2.tex Table 2: "
            "classical backbone, external graph-level Q, static FiLM, post-backbone "
            "(QIGNN_mainline). Ignored if --experiments is passed with at least one name. "
            "Excludes ablation_direct_q and ablation_classical_resid."
        ),
    )
    parser.add_argument(
        "--min_key_frac",
        type=float,
        default=0.05,
        help="Epoch mode: include optional history keys present in at least this fraction of epoch rows",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="Number of principal components",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results_dir/pca_out/<mode>)",
    )
    parser.add_argument("--plot", action="store_true", help="Save PC1 vs PC2 scatter PNG")
    parser.add_argument(
        "--exclude_fold",
        type=int,
        default=None,
        help="If set, fit scaler+PCA only on rows with fold_idx != this value; transform all",
    )
    parser.add_argument(
        "--allow_partial_scope",
        action="store_true",
        help=(
            "Allow PCA to run even if some requested dataset/experiment pairs have no "
            "matching result JSONs. By default, paper-scoped runs fail loudly."
        ),
    )
    parser.add_argument(
        "--include_qb_stats",
        action="store_true",
        help="Run mode: include optional Q/B diagnostic summaries when they are present.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"results_dir not found: {results_dir}")

    # Resolve experiments before default dataset discovery (scan uses these folder names).
    if args.experiments is not None and len(args.experiments) > 0:
        experiments = list(args.experiments)
    elif args.paper_reported_ablations:
        experiments = list(PAPER_REPORTED_ABLATION_EXPERIMENTS)
    else:
        experiments = list(TABLE2_EXPERIMENTS)

    datasets = args.datasets
    if args.paper_table2:
        datasets = PAPER_TABLE2_DATASETS
    if datasets is None:
        # Default: every dataset that has at least one of the experiment folders
        datasets = []
        for p in results_dir.iterdir():
            if not p.is_dir():
                continue
            ok = False
            for exp in experiments:
                if (p / exp).exists():
                    ok = True
                    break
            if ok:
                datasets.append(p.name)
        datasets = sorted(datasets)
        if not datasets:
            raise SystemExit("No datasets found; pass --datasets explicitly")

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "pca_out" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    file_pairs, matched_scope, missing_scope = gather_files(str(results_dir), datasets, experiments)
    if not file_pairs:
        raise SystemExit("No result JSON files matched datasets/experiments.")
    if missing_scope and not args.allow_partial_scope and (args.paper_table2 or args.paper_reported_ablations):
        preview = ", ".join(f"{ds}/{exp}" for ds, exp in missing_scope[:8])
        if len(missing_scope) > 8:
            preview += ", ..."
        raise SystemExit(
            "Requested paper scope is incomplete. Missing dataset/experiment pairs: "
            f"{preview}. Re-run with --allow_partial_scope only if you intentionally want "
            "an exploratory partial-data PCA."
        )

    loaded: List[Tuple[str, Path, Dict[str, Any]]] = []
    for ds, path in file_pairs:
        data = load_json(path)
        if data:
            loaded.append((ds, path, data))

    if not loaded:
        raise SystemExit("No JSON files could be loaded.")

    fit_mask = np.ones(len(loaded), dtype=bool)
    if args.exclude_fold is not None:
        for i, (_, _, data) in enumerate(loaded):
            fold_idx = (data.get("config") or {}).get("fold_idx")
            if fold_idx == args.exclude_fold:
                fit_mask[i] = False
    if fit_mask.sum() < 2:
        raise SystemExit("Not enough fit rows after --exclude_fold (need >= 2 files).")

    if args.mode == "epoch":
        print(
            "[warn] epoch mode is exploratory: principal components may be driven by "
            "training chronology rather than between-run differences."
        )
        files_data = [(p, d) for _, p, d in loaded]
        feature_keys = collect_numeric_keys_from_history(
            files_data, min_frac=args.min_key_frac
        )
        all_rows: List[Dict[str, Any]] = []
        row_source_idx: List[int] = []
        for src_idx, (ds, path, data) in enumerate(loaded):
            rows = epoch_rows_from_result(path, data, results_dir, ds, feature_keys)
            all_rows.extend(rows)
            row_source_idx.extend([src_idx] * len(rows))
        if not all_rows:
            raise SystemExit("No epoch rows extracted (empty history?).")
        meta_keys = [
            "run_id",
            "dataset",
            "exp_slug",
            "variant",
            "fold_idx",
            "seed",
            "epoch",
        ]
        row_fit_mask = np.array([fit_mask[i] for i in row_source_idx], dtype=bool)
        if row_fit_mask.sum() < 2:
            raise SystemExit("Not enough epoch rows for PCA after --exclude_fold.")
        X, feat_names, imputer = build_matrix(all_rows, list(feature_keys), row_fit_mask)
    else:
        stat_keys = list(DEFAULT_RUN_STAT_KEYS)
        if args.include_qb_stats:
            stat_keys.extend(DEFAULT_RUN_EXTRA_KEYS)
        all_rows = []
        row_source_idx = []
        for src_idx, (ds, path, data) in enumerate(loaded):
            r = run_summary_row(path, data, results_dir, ds, stat_keys)
            if r:
                all_rows.append(r)
                row_source_idx.append(src_idx)
        if not all_rows:
            raise SystemExit("No run-level rows built.")
        row_fit_mask = np.array([fit_mask[i] for i in row_source_idx], dtype=bool)
        if row_fit_mask.sum() < 2:
            raise SystemExit("Not enough run rows for PCA after --exclude_fold.")
        feature_keys = collect_run_feature_keys(all_rows)
        meta_keys = [
            "run_id",
            "dataset",
            "exp_slug",
            "variant",
            "fold_idx",
            "seed",
            "test_acc",
            "n_epochs",
        ]
        X, feat_names, imputer = build_matrix(all_rows, feature_keys, row_fit_mask)

    scaler = StandardScaler()
    n_comp = min(args.n_components, int(row_fit_mask.sum()), X.shape[1])
    pca = PCA(n_components=n_comp)

    X_fit = X[row_fit_mask]
    X_fit_s = scaler.fit_transform(X_fit)
    pca.fit(X_fit_s)

    X_all_s = scaler.transform(X)
    scores = pca.transform(X_all_s)

    save_scores_csv(out_dir / "pca_scores.csv", all_rows, scores, meta_keys)
    save_pca_json(out_dir / "pca_model.json", feat_names, pca, scaler)

    # Small run manifest for reproducibility
    manifest = {
        "mode": args.mode,
        "requested_datasets": list(datasets),
        "requested_experiments": list(experiments),
        "paper_reported_ablations": bool(args.paper_reported_ablations),
        "paper_table2": bool(args.paper_table2),
        "allow_partial_scope": bool(args.allow_partial_scope),
        "include_qb_stats": bool(args.include_qb_stats),
        "n_files": len(loaded),
        "n_rows": len(all_rows),
        "feature_count": len(feat_names),
        "features": feat_names,
        "exclude_fold": args.exclude_fold,
        "matched_scope": [f"{ds}/{exp}" for ds, exp in matched_scope],
        "missing_scope": [f"{ds}/{exp}" for ds, exp in missing_scope],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    manifest.update(scope_summary_from_rows(all_rows))
    with open(out_dir / "pca_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {out_dir / 'pca_scores.csv'} ({len(all_rows)} rows, {scores.shape[1]} PCs)")
    print(f"Wrote {out_dir / 'pca_model.json'}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Loaded variants: {', '.join(manifest['loaded_variants']) or 'none'}")
    if len(manifest["loaded_variants"]) < 2:
        print(
            "[warn] fewer than two variants were loaded; interpret this PCA as "
            "within-variant descriptive structure, not an ablation comparison."
        )

    if args.plot:
        if plt is None:
            print("matplotlib missing; skip plot")
        else:
            title = f"PCA training dynamics ({args.mode})"
            plot_pc_scatter(all_rows, scores, out_dir / "pca_pc1_pc2.png", title)
            print(f"Wrote {out_dir / 'pca_pc1_pc2.png'}")


if __name__ == "__main__":
    main()
