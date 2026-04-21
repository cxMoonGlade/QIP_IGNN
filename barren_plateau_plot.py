#!/usr/bin/env python3
"""Plot barren-plateau diagnostic results produced by `barren_plateau_analysis.py`.

Reads `bp_grid.csv` and writes three figures:
    bp_vs_n.pdf   log10 Var[grad] vs n_qubits       (axis=width)
    bp_vs_R.pdf   log10 Var[grad] vs circuit_reps   (axis=depth)
    bp_vs_T.pdf   log10 Var[grad] vs max_iter       (axis=T)

Each figure shows one line per variant with a log-linear fit annotated
(slope s such that Var ~ c * base^{-s * x}). All figures use the "ALL" group
aggregate by default; other groups can be selected via --group.

Usage
-----
    python barren_plateau_plot.py \
    --csv results/barren_plateau/NCI1/main_torchdeq/bp_grid.csv \
    --solver_in_filename \
    --axes width depth T

example output:
output_dir/
  bp_vs_n.pdf                     ← default
  bp_vs_n_torchdeq.pdf            ← --solver_in_filename
  bp_vs_n_smoke.pdf               ← --tag=smoke
  bp_vs_n_torchdeq_smoke.pdf      ← --solver_in_filename --tag=smoke
  bp_vs_R.pdf  bp_vs_T.pdf        ← same for R and T axes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AXIS_META = {
    "width": {"x_col": "n_qubits",    "xlabel": "PQC qubits n", "fname": "bp_vs_n.pdf"},
    "depth": {"x_col": "circuit_reps", "xlabel": "Circuit reps R",    "fname": "bp_vs_R.pdf"},
    "T":     {"x_col": "max_iter",     "xlabel": "Solver iterations T","fname": "bp_vs_T.pdf"},
}

VARIANT_STYLE = {
    "IN": {"color": "tab:blue",   "marker": "o", "label": "IN (independent)"},
    "SD": {"color": "tab:red",    "marker": "s", "label": "SD (state-dependent)"},
    "BD": {"color": "tab:green",  "marker": "^", "label": "BD (backbone-dependent)"},
}


REQUIRED_CSV_COLUMNS = {"variant", "axis", "group", "var_grad_mean"}


def load_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        raise SystemExit(f"[plot] --csv not found: {csv_path}")
    if csv_path.suffix.lower() == ".json":
        raise SystemExit(
            f"[plot] --csv got a .json path ({csv_path}). Pass the "
            f"bp_grid.csv file, not bp_manifest.json.")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = set(reader.fieldnames or [])
    missing = REQUIRED_CSV_COLUMNS - header
    if missing:
        raise SystemExit(
            f"[plot] CSV at {csv_path} is missing required columns: {sorted(missing)}.\n"
            f"  Columns found: {sorted(header) or '(none)'}.\n"
            f"  Make sure you are pointing at bp_grid.csv produced by "
            f"barren_plateau_analysis.py, not bp_manifest.json or some other file.")
    int_keys = (
        "n_qubits", "circuit_reps", "max_iter",
        "n_params_pqc", "n_params_in_group", "n_resamples",
        "n_resamples_completed",
        "n_resamples_finite", "n_resamples_diverged", "n_loss_diverged",
    )
    float_keys = (
        "var_grad_mean", "var_grad_median", "var_grad_max",
        "grad_log10_abs_median",
        "var_output_mean", "var_loss", "loss_mean", "point_wall_s",
    )
    bool_keys = ("timed_out",)
    for r in rows:
        for k in int_keys:
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = int(r[k])
                except ValueError:
                    r[k] = 0
        for k in float_keys:
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = float("nan")
        for k in bool_keys:
            if k in r and r[k] not in ("", None):
                r[k] = str(r[k]).strip().lower() in ("1", "true", "t", "yes")
    return rows


def select(rows: List[Dict], axis: str, variant: str, group: str) -> List[Dict]:
    out = [r for r in rows
           if r["axis"] == axis and r["variant"] == variant and r["group"] == group]
    out.sort(key=lambda r: r[AXIS_META[axis]["x_col"]])
    return out


def log_linear_slope(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Fit y = c * base^{-s * x} in log space; return (slope_s, intercept_log10c).

    We fit in natural log on x (log-linear exponential decay). When the axis is
    already an integer like n_qubits / circuit_reps, we fit log10(Var) vs x
    (not log10(x)) since BP theory gives exponential decay in those variables.
    For the T axis, we use the same linear-in-T fit so a horizontal (zero-slope)
    line for IN is visually obvious.
    """
    xs_a = np.asarray(xs, dtype=np.float64)
    ys_a = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(ys_a) & (ys_a > 0) & np.isfinite(xs_a)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    log_y = np.log10(ys_a[ok])
    m, b = np.polyfit(xs_a[ok], log_y, 1)
    slope = -float(m)
    return slope, float(b)


def plot_axis(rows: List[Dict], axis: str, group: str, output_dir: Path,
              title_extra: str = "",
              slope_in: str = "legend",
              show_fit_line: bool = True,
              bp_solver: Optional[str] = None) -> Path:
    """Plot one axis of the BP scan.

    slope_in:
        "legend"  : annotate fitted slope inside each variant's legend label (default,
                    avoids overlap even when variant lines pass through identical points).
        "line"    : annotate at the right edge of each line at its last data point;
                    may overlap when lines are close.
        "none"    : do not annotate slopes at all.
    show_fit_line:
        if True, overlay a faint dashed line showing the log-linear fit for each variant.
    bp_solver:
        Optional solver label to show in the title; if None, try to infer from rows.

    Timed-out points (``timed_out=True``, typically <3 completed resamples) are
    always plotted, but as open markers with a ``completed/requested`` annotation,
    are not connected by the line, and are excluded from the log-linear fit.
    """
    meta = AXIS_META[axis]
    x_col = meta["x_col"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)

    legend_handles = []
    legend_labels = []
    diverged_notes: List[str] = []
    timed_out_notes: List[str] = []

    for variant, style in VARIANT_STYLE.items():
        pts = select(rows, axis, variant, group)
        if not pts:
            continue
        xs_all = np.asarray([p[x_col] for p in pts], dtype=np.float64)
        ys_all = np.asarray([p["var_grad_mean"] for p in pts], dtype=np.float64)
        div_counts = np.asarray(
            [p.get("n_resamples_diverged", 0) or 0 for p in pts], dtype=np.int64)
        is_timedout = np.asarray(
            [bool(p.get("timed_out")) for p in pts], dtype=bool)
        is_finite = np.isfinite(ys_all) & (ys_all > 0)
        # ``good`` rows connect into the line and drive the log-linear fit.
        # Timed-out points are excluded from both because their variance is
        # estimated from too few resamples to be a meaningful BP signal.
        good = is_finite & ~is_timedout
        xs = xs_all[good]
        ys = ys_all[good]

        (line,) = ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                          linestyle="-", markersize=6, linewidth=1.6)

        # Non-finite points (inf/nan variance): mark with 'x' on top edge.
        bad_idx = np.where(~is_finite)[0]
        if bad_idx.size:
            ax.scatter(
                xs_all[bad_idx],
                np.full(bad_idx.size, ax.get_ylim()[1] if ax.has_data() else 1.0),
                color=style["color"], marker="x", s=36, linewidths=1.2, zorder=5)
            diverged_notes.append(
                f"{variant}: {bad_idx.size} non-finite point(s) at "
                f"{x_col}={list(xs_all[bad_idx].astype(int))}")

        # Timed-out points with a finite (low-sample) variance: open marker +
        # ``n_completed/n_requested`` annotation, not connected to line.
        to_idx = np.where(is_timedout & is_finite)[0]
        if to_idx.size:
            ax.scatter(xs_all[to_idx], ys_all[to_idx],
                       facecolors="white", edgecolors=style["color"],
                       marker=style["marker"], s=80, linewidths=1.4, zorder=4)
            for i in to_idx:
                p = pts[i]
                n_done = p.get("n_resamples_completed")
                n_req = p.get("n_resamples")
                if n_done is not None and n_req is not None:
                    ax.annotate(f" {n_done}/{n_req}",
                                xy=(float(xs_all[i]), float(ys_all[i])),
                                xytext=(6, 0), textcoords="offset points",
                                fontsize=7, color=style["color"],
                                va="center", clip_on=False)
            timed_out_notes.append(
                f"{variant} at {x_col}={list(xs_all[to_idx].astype(int))}")

        # Partial divergence (some of the K resamples produced inf grads but the
        # aggregate variance is still finite and based on enough samples): subtle
        # open marker, no annotation, included in the line and fit.
        partial = np.where((div_counts > 0) & is_finite & ~is_timedout)[0]
        if partial.size:
            ax.scatter(xs_all[partial], ys_all[partial],
                       facecolors="white", edgecolors=style["color"],
                       marker=style["marker"], s=70, linewidths=1.0, zorder=3)

        slope, intercept = log_linear_slope(xs.tolist(), ys.tolist())

        # Optional fit line overlay.
        if show_fit_line and not np.isnan(slope) and len(xs) >= 2:
            xs_fit = np.linspace(float(xs.min()), float(xs.max()), 64)
            ys_fit = 10 ** (-slope * xs_fit + intercept)
            ax.plot(xs_fit, ys_fit, color=style["color"],
                    linestyle="--", linewidth=0.9, alpha=0.55)

        # Legend label with slope baked in to avoid on-plot overlap.
        if slope_in == "legend":
            if np.isnan(slope):
                label = style["label"]
            else:
                label = f"{style['label']} (slope = {slope:+.2f})"
        else:
            label = style["label"]
            if slope_in == "line" and not np.isnan(slope) and xs.size:
                ax.annotate(
                    f" slope={slope:+.2f}",
                    xy=(float(xs[-1]), float(ys[-1])),
                    xytext=(4, 0), textcoords="offset points",
                    color=style["color"], fontsize=8, va="center",
                    clip_on=False,
                )
        legend_handles.append(line)
        legend_labels.append(label)

    ax.set_yscale("log")
    ax.set_xlabel(meta["xlabel"])
    ax.set_ylabel(r"$\mathrm{Var}_{\theta}[\partial L / \partial \theta]$ (mean over PQC params)")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    title = f"Barren-plateau diagnostic (axis={axis}, group={group})"
    if bp_solver:
        title += f"  [solver = {bp_solver}]"
    if title_extra:
        title += f"  [{title_extra}]"
    ax.set_title(title, fontsize=10)

    # Place the legend outside the axes when annotations would otherwise crowd it.
    ax.legend(legend_handles, legend_labels,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False, borderaxespad=0.0)

    # Footer: document flagged points so the figure stays self-describing.
    footer_parts: List[str] = []
    if timed_out_notes:
        footer_parts.append(
            "timed out (open marker + completed/requested annotation; "
            "excluded from fit): "
            + "; ".join(timed_out_notes))
    if diverged_notes:
        footer_parts.append(
            "non-finite variance (excluded from line and fit): "
            + "; ".join(diverged_notes))
    if footer_parts:
        fig.text(0.01, -0.02, " | ".join(footer_parts),
                 fontsize=7, color="0.25")

    fig.canvas.draw()

    out_path = output_dir / meta["fname"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot barren-plateau results")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to bp_grid.csv produced by barren_plateau_analysis.py")
    parser.add_argument("--group", type=str, default="ALL",
                        help="Parameter group to plot (default: ALL aggregate).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write PDFs (default: same as CSV).")
    parser.add_argument("--axes", type=str, nargs="+",
                        default=["width", "depth", "T"],
                        choices=list(AXIS_META.keys()))
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to figure filenames.")
    parser.add_argument("--slope_in", type=str, default="legend",
                        choices=["legend", "line", "none"],
                        help="Where to display fitted slopes (default: legend, "
                             "avoids overlapping annotations).")
    parser.add_argument("--no_fit_line", action="store_true",
                        help="Do not overlay dashed log-linear fit lines.")
    parser.add_argument("--solver_in_filename", action="store_true",
                        help="Append solver tag (e.g. _unroll, _torchdeq) to output filenames "
                             "so dual-solver runs do not clobber each other.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)

    # Detect the solver used (when CSV has bp_solver column). Assume the scan
    # is solver-homogeneous; if mixed, the first row wins and we warn.
    solvers_seen = sorted({str(r.get("bp_solver", "")) for r in rows if r.get("bp_solver")})
    bp_solver: Optional[str] = solvers_seen[0] if solvers_seen else None
    if len(solvers_seen) > 1:
        print(f"[plot] [warn] CSV mixes multiple solvers {solvers_seen}; "
              f"using '{bp_solver}' in title. Filter CSV first for clean figures.")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally suffix filenames with tag (e.g. "smoke") and/or solver name.
    suffix_parts: List[str] = []
    if args.solver_in_filename and bp_solver:
        suffix_parts.append(bp_solver)
    if args.tag:
        suffix_parts.append(args.tag)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    for axis in args.axes:
        out = plot_axis(
            rows, axis, args.group, output_dir,
            title_extra=args.tag or "",
            slope_in=args.slope_in,
            show_fit_line=not args.no_fit_line,
            bp_solver=bp_solver,
        )
        if suffix:
            new_name = out.with_name(out.stem + suffix + out.suffix)
            out.rename(new_name)
            out = new_name
        print(f"[plot] {axis} -> {out}")


if __name__ == "__main__":
    main()
