#!/usr/bin/env python3
"""
Combine BP diagnostic results across multiple datasets into a single panel
figure suitable for the paper main text or appendix.

Reuses the per-axis visual conventions of ``barren_plateau_plot.py``:
    - IN / BD / SD fixed colour+marker mapping
    - good points connected by a solid line; timed-out points drawn as open
      markers with a ``completed/requested`` annotation and excluded from fit;
      non-finite points marked with an ``x`` on the top edge and excluded
    - log-linear fit as a dashed line, slope printed in the per-panel legend

Default layout is 3 rows (datasets) x 2 columns (width / depth axis), sized
for a single IEEE double-column text column; the ``--layout 2x3`` variant
fits a two-column ``figure*`` span. The script also prints a "TODO numbers"
block at the end collecting every quantity the paper `\\textbf{[TODO:...]}`
placeholders ask for, so those can be filled by copy-paste.

Usage
-----
Main figure (matches paper \\S{sec:bp}):

    python combine_plots.py \\
        --csvs results/barren_plateau/NCI1/main_torchdeq/bp_grid.csv \\
               results/barren_plateau/PROTEINS/main_torchdeq/bp_grid.csv \\
               results/barren_plateau/MUTAG/main_torchdeq/bp_grid.csv \\
        --axes width depth \\
        --output ../paper_new/figures/bp_main_3x2.pdf

Shortcut using the conventional directory layout:

    python combine_plots.py --mode main \\
        --output ../paper_new/figures/bp_main_3x2.pdf

Appendix figure (BPTT T-axis, rendered 1x3):

    python combine_plots.py --mode appendix \\
        --output ../paper_new/figures/bp_appendix_bptt_1x3.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from barren_plateau_plot import (
    AXIS_META,
    VARIANT_STYLE,
    load_rows,
    log_linear_slope,
    select,
)


DATASETS_DEFAULT = ["NCI1", "PROTEINS", "MUTAG"]


# =============================================================================
# Panel renderer
# =============================================================================

def render_panel(
    ax,
    rows,
    axis: str,
    group: str,
    show_fit_line: bool = True,
    slope_in_legend: bool = True,
) -> Tuple[Dict, Dict, List[Tuple[str, List[int]]]]:
    """Draw IN/BD/SD lines+markers+fits onto ``ax``.

    Returns
    -------
    slopes : dict variant -> float  (NaN if not fittable)
    legend : dict variant -> (handle, label)
    flagged : list of (variant, kind, xs_list) where kind in {"timed_out","nonfinite"}
    """
    meta = AXIS_META[axis]
    x_col = meta["x_col"]
    slopes: Dict[str, float] = {}
    legend_entries: Dict[str, Tuple] = {}
    flagged: List[Tuple[str, str, List[int]]] = []

    for variant, style in VARIANT_STYLE.items():
        pts = select(rows, axis, variant, group)
        if not pts:
            continue
        xs_all = np.asarray([p[x_col] for p in pts], dtype=np.float64)
        ys_all = np.asarray([p["var_grad_mean"] for p in pts], dtype=np.float64)
        is_timedout = np.asarray(
            [bool(p.get("timed_out")) for p in pts], dtype=bool)
        is_finite = np.isfinite(ys_all) & (ys_all > 0)
        # Timed-out points with a finite variance are treated as ordinary data:
        # connected by the line, drawn with a solid marker, and included in the
        # log-linear fit. They are still reported in the ``flagged`` list so the
        # TODO numbers summary can call them out.
        good = is_finite
        xs = xs_all[good]
        ys = ys_all[good]

        (line,) = ax.plot(
            xs, ys, color=style["color"], marker=style["marker"],
            linestyle="-", markersize=5, linewidth=1.5)

        # Non-finite: small x on top edge
        bad_idx = np.where(~is_finite)[0]
        if bad_idx.size:
            ax.scatter(
                xs_all[bad_idx],
                np.full(bad_idx.size, ax.get_ylim()[1] if ax.has_data() else 1.0),
                color=style["color"], marker="x", s=30, linewidths=1.0, zorder=5)
            flagged.append((variant, "nonfinite",
                            list(xs_all[bad_idx].astype(int))))

        # Record timed-out points in the flagged list (no visual distinction
        # from regular points; reported textually in the TODO summary).
        to_idx = np.where(is_timedout & is_finite)[0]
        if to_idx.size:
            flagged.append((variant, "timed_out",
                            list(xs_all[to_idx].astype(int))))

        slope, intercept = log_linear_slope(xs.tolist(), ys.tolist())
        slopes[variant] = slope

        if show_fit_line and not np.isnan(slope) and len(xs) >= 2:
            xs_fit = np.linspace(float(xs.min()), float(xs.max()), 64)
            ys_fit = 10 ** (-slope * xs_fit + intercept)
            ax.plot(xs_fit, ys_fit, color=style["color"],
                    linestyle="--", linewidth=0.8, alpha=0.55)

        if slope_in_legend and not np.isnan(slope):
            label = f"{variant} ({slope:+.2f})"
        else:
            label = variant
        legend_entries[variant] = (line, label)

    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.25, alpha=0.4)
    return slopes, legend_entries, flagged


# =============================================================================
# Full grid figure
# =============================================================================

def plot_grid(
    csv_paths: Dict[str, Path],
    axes_list: List[str],
    output_path: Path,
    layout: str = "3x2",
    figure_class: str = "single",
    group: str = "ALL",
    font_base: Optional[float] = None,
) -> Dict:
    """Render a multi-dataset panel figure.

    Returns a dict of aggregated numbers suitable for filling paper TODOs.
    """
    datasets = list(csv_paths.keys())
    n_ds = len(datasets)
    n_ax = len(axes_list)

    # Decide grid orientation
    if layout == "3x2":
        nrows, ncols = n_ds, n_ax
        rows_are_datasets = True
    elif layout == "2x3":
        nrows, ncols = n_ax, n_ds
        rows_are_datasets = False
    elif layout == "1x3":
        nrows, ncols = 1, n_ds
        rows_are_datasets = False
    elif layout == "3x1":
        nrows, ncols = n_ds, 1
        rows_are_datasets = True
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Pick physical size based on target column class. Single-column IEEE
    # conference text width is ~3.4". Panel height and inter-row gap are split
    # explicitly so we can halve the gap without growing each panel.
    if figure_class == "single":
        fig_w = 3.4
        panel_h_abs = 1.115       # drawing area of a single panel, inches
        hspace_frac = 0.225       # gap between rows as fraction of panel height
        font_default = 6
    else:  # "wide"
        fig_w = 7.0
        panel_h_abs = 1.20
        hspace_frac = 0.225
        font_default = 6.5
    font_base = font_base if font_base is not None else font_default

    # Total data-grid height = nrows panels + (nrows-1) inter-row gaps.
    gap_h_abs = hspace_frac * panel_h_abs
    fig_h = nrows * panel_h_abs + (nrows - 1) * gap_h_abs

    plt.rcParams.update({
        "font.size": font_base,
        "axes.titlesize": font_base + 1,
        "axes.labelsize": font_base,
        "xtick.labelsize": font_base - 0.5,
        "ytick.labelsize": font_base - 0.5,
        "legend.fontsize": font_base,
    })

    # Layout (top to bottom), all heights in absolute inches:
    #   top_margin_abs       canvas top edge -> legend top
    #   legend_h             legend strip (IN / BD / SD stacked vertically)
    #   legend_to_panels     gap between legend bottom and first-row column titles
    #   fig_h                data panel grid (nrows x ncols, with hspace gaps
    #                        between rows and wspace gaps between columns)
    #   bottom_margin_abs    lowest xtick labels -> canvas bottom edge
    #
    # The legend is placed with fig.add_axes() in absolute figure fractions so
    # that its position is COMPLETELY decoupled from the data grid's hspace.
    # Raising hspace shrinks each data panel but does NOT widen the gap between
    # the legend and the first data row, which is controlled solely by
    # ``legend_to_panels`` below.
    sharey_mode = "row" if rows_are_datasets else "col"
    legend_rows = 3
    line_h_in = (font_base * 1.3) / 72.0
    legend_h = legend_rows * line_h_in + 0.08

    top_margin_abs    = 0.06
    legend_to_panels  = 0.10  # gap between legend and column titles (inches)
    bottom_margin_abs = 0.26

    total_h = (top_margin_abs + legend_h + legend_to_panels
               + fig_h + bottom_margin_abs)
    fig = plt.figure(figsize=(fig_w, total_h))

    left_frac = 0.20
    right_frac = 0.97

    # Legend axis: absolute fractions.
    legend_y_bottom = 1.0 - (top_margin_abs + legend_h) / total_h
    legend_y_height = legend_h / total_h
    legend_ax = fig.add_axes([
        left_frac, legend_y_bottom,
        right_frac - left_frac, legend_y_height,
    ])
    legend_ax.axis("off")

    # Data grid: starts legend_to_panels inches below the legend and extends
    # to the bottom margin.
    data_top_frac = 1.0 - (top_margin_abs + legend_h + legend_to_panels) / total_h
    data_bottom_frac = bottom_margin_abs / total_h
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=hspace_frac, wspace=0.25,
        left=left_frac, right=right_frac,
        top=data_top_frac, bottom=data_bottom_frac,
    )

    ax_grid = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            share_y = None
            share_x = None
            if sharey_mode == "row" and j > 0:
                share_y = ax_grid[i, 0]
            elif sharey_mode == "col" and i > 0:
                share_y = ax_grid[0, j]
            ax_grid[i, j] = fig.add_subplot(gs[i, j], sharey=share_y, sharex=share_x)
            # Hide inner tick labels to match plt.subplots(sharey=...) behaviour.
            if sharey_mode == "row" and j > 0:
                plt.setp(ax_grid[i, j].get_yticklabels(), visible=False)
            elif sharey_mode == "col" and i > 0:
                plt.setp(ax_grid[i, j].get_yticklabels(), visible=False)

    rows_by_ds = {ds: load_rows(p) for ds, p in csv_paths.items()}

    all_slopes: Dict[Tuple[str, str], Dict[str, float]] = {}
    all_flagged: List[Tuple[str, str, str, List[int]]] = []
    legend_handles_by_variant: Dict[str, Tuple] = {}

    for i in range(nrows):
        for j in range(ncols):
            if rows_are_datasets:
                ds = datasets[i]
                axis = axes_list[j]
            else:
                axis = axes_list[i] if layout != "1x3" else axes_list[0]
                ds = datasets[j]
            ax = ax_grid[i, j]
            slopes, legend, flagged = render_panel(ax, rows_by_ds[ds], axis, group)
            all_slopes[(ds, axis)] = slopes
            for (var, kind, xs) in flagged:
                all_flagged.append((ds, axis, var, kind, xs))
            for var, (h, l) in legend.items():
                # keep first-seen handle per variant; use bare name in shared legend
                if var not in legend_handles_by_variant:
                    legend_handles_by_variant[var] = (h, var)

            # Headers & labels
            axis_meta = AXIS_META[axis]
            if rows_are_datasets:
                # Top row: column headers = axis labels (these sit directly
                # below the legend strip -- reader order is legend -> column -> data).
                if i == 0:
                    ax.set_title(axis_meta["xlabel"], fontsize=font_base + 1,
                                 pad=4)
                # Left column: row labels = dataset names
                if j == 0:
                    ax.set_ylabel(
                        f"{ds}\n" + r"$\mathrm{Var}_{\theta}[\partial L/\partial\theta]$",
                        fontsize=font_base)
                # X axis labels only on the bottom row to avoid duplication.
                # We do NOT repeat the x-label on the bottom row either, since
                # the column title at the top already carries it.
                ax.set_xlabel("")
            else:
                # 2x3 layout: columns = datasets, rows = axes
                if i == 0:
                    ax.set_title(ds, fontsize=font_base + 1, pad=4)
                if j == 0:
                    ax.set_ylabel(
                        f"{axis_meta['xlabel']} sweep\n"
                        + r"$\mathrm{Var}_{\theta}[\partial L/\partial\theta]$",
                        fontsize=font_base)
                if i == nrows - 1:
                    ax.set_xlabel(axis_meta["xlabel"], fontsize=font_base)
                else:
                    ax.set_xlabel("")

    # Shared legend at the top, one row of handles
    # Include per-variant mean slope across panels for a compact summary
    mean_slope_by_variant: Dict[str, Tuple[float, float]] = {}
    for variant in VARIANT_STYLE.keys():
        collected = [s.get(variant, np.nan) for s in all_slopes.values()]
        collected = [c for c in collected if c is not None and not np.isnan(c)]
        if collected:
            mean_slope_by_variant[variant] = (float(np.mean(collected)),
                                              float(np.std(collected)))

    legend_labels = []
    legend_handles = []
    for variant in ["IN", "BD", "SD"]:
        if variant not in legend_handles_by_variant:
            continue
        h, _ = legend_handles_by_variant[variant]
        style = VARIANT_STYLE[variant]
        # Compact label: variant code only. Per-panel slopes would bloat the
        # legend strip and are anyway printed in the TODO numbers block.
        legend_labels.append(style["label"])
        legend_handles.append(h)

    # Put the legend INSIDE the reserved legend_ax with ncol=1 so the three
    # variant entries stack vertically. This (a) keeps the figure's width equal
    # to fig_w (no horizontal widening to fit a long legend row), and (b)
    # positions the legend directly above the column titles ("PQC qubits n" /
    # "Circuit reps R") so the reader's eye goes variant -> column -> data.
    legend_ax.legend(
        legend_handles, legend_labels,
        loc="center", ncol=1,
        frameon=False, fontsize=font_base,
        handlelength=1.8, handletextpad=0.5, labelspacing=0.3,
        borderaxespad=0.0,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" here: we want the final PDF size to match the
    # requested figsize exactly so it drops into the paper at its intended
    # physical width.
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "slopes": all_slopes,
        "flagged": all_flagged,
        "mean_slope_by_variant": mean_slope_by_variant,
        "output": str(output_path),
        "layout": layout,
    }


# =============================================================================
# TODO-numbers summary
# =============================================================================

def print_todo_numbers(
    summary: Dict,
    csv_paths: Dict[str, Path],
    axes_list: List[str],
    group: str = "ALL",
) -> None:
    """Print the concrete numbers the paper's TODO placeholders ask for."""
    datasets = list(csv_paths.keys())
    slopes = summary["slopes"]
    flagged = summary["flagged"]
    rows_by_ds = {ds: load_rows(p) for ds, p in csv_paths.items()}

    print()
    print("=" * 78)
    print("TODO NUMBERS — copy/paste targets for revised.tex")
    print("=" * 78)

    # ------- Slopes per (axis, dataset) -------
    # Convention: slope s is fitted as log10(Var) = -s * x + b, so
    # NEGATIVE slope = Var DECAYS with x (classical barren-plateau signature);
    # POSITIVE slope = Var GROWS with x (no-BP or even anti-BP regime).
    print("\nNote on signs: slope s is fitted as log10(Var) = -s*x + b. "
          "Negative s => Var grows with x (no BP decay); positive s => Var decays.")
    for axis in axes_list:
        axis_label = AXIS_META[axis]["xlabel"]
        print(f"\n--- Slopes on {axis_label} axis (log10 Var per unit of x) ---")
        for ds in datasets:
            s = slopes.get((ds, axis), {})
            parts = [f"{v}={s.get(v, float('nan')):+.2f}" for v in ["IN", "BD", "SD"]
                     if v in s]
            print(f"  {ds:10s}: " + "  ".join(parts))

    # ------- Variance ratios IN/SD at extreme of each axis -------
    print("\n--- Variance ratios IN/SD at extreme of each axis (log10 IN/SD) ---")
    for axis in axes_list:
        x_col = AXIS_META[axis]["x_col"]
        print(f"  axis={axis}:")
        for ds in datasets:
            rows = rows_by_ds[ds]
            # Extreme is the largest x that is NOT timed_out for both variants
            in_pts = [p for p in select(rows, axis, "IN", group)
                      if not bool(p.get("timed_out"))]
            sd_pts = [p for p in select(rows, axis, "SD", group)
                      if not bool(p.get("timed_out"))]
            common_xs = sorted(
                set(int(p[x_col]) for p in in_pts) &
                set(int(p[x_col]) for p in sd_pts))
            if not common_xs:
                print(f"    {ds:10s}: no common non-timed-out x")
                continue
            x_star = common_xs[-1]
            in_at = next(p for p in in_pts if int(p[x_col]) == x_star)
            sd_at = next(p for p in sd_pts if int(p[x_col]) == x_star)
            try:
                log_ratio = np.log10(float(in_at["var_grad_mean"])
                                     / float(sd_at["var_grad_mean"]))
                print(f"    {ds:10s}: at {x_col}={x_star}  "
                      f"IN/SD = 10^{log_ratio:+.2f}  "
                      f"(IN={float(in_at['var_grad_mean']):.2e}, "
                      f"SD={float(sd_at['var_grad_mean']):.2e})")
            except (ValueError, TypeError, ZeroDivisionError):
                print(f"    {ds:10s}: at {x_col}={x_star}  [could not compute ratio]")

    # ------- IN strict dominance check -------
    # Paper claim: IN above both BD and SD on every (dataset, axis) grid point.
    # This is the MAIN claim; BD vs SD is secondary.
    print("\n--- IN strict dominance (IN > BD and IN > SD at every x) ---")
    in_exceptions: List[str] = []
    total_pts = 0
    in_ok_pts = 0
    for axis in axes_list:
        for ds in datasets:
            rows = rows_by_ds[ds]
            x_col = AXIS_META[axis]["x_col"]
            in_pts = {int(p[x_col]): float(p["var_grad_mean"])
                      for p in select(rows, axis, "IN", group)
                      if not bool(p.get("timed_out"))
                      and np.isfinite(float(p["var_grad_mean"]))}
            bd_pts = {int(p[x_col]): float(p["var_grad_mean"])
                      for p in select(rows, axis, "BD", group)
                      if not bool(p.get("timed_out"))
                      and np.isfinite(float(p["var_grad_mean"]))}
            sd_pts = {int(p[x_col]): float(p["var_grad_mean"])
                      for p in select(rows, axis, "SD", group)
                      if not bool(p.get("timed_out"))
                      and np.isfinite(float(p["var_grad_mean"]))}
            common_xs = sorted(set(in_pts) & set(bd_pts) & set(sd_pts))
            for x in common_xs:
                total_pts += 1
                if in_pts[x] > bd_pts[x] and in_pts[x] > sd_pts[x]:
                    in_ok_pts += 1
                else:
                    reasons = []
                    if in_pts[x] <= bd_pts[x]:
                        reasons.append(f"IN<={in_pts[x]:.2e}<=BD={bd_pts[x]:.2e}")
                    if in_pts[x] <= sd_pts[x]:
                        reasons.append(f"IN={in_pts[x]:.2e}<=SD={sd_pts[x]:.2e}")
                    in_exceptions.append(
                        f"({ds}, {axis}, {x_col}={x}): " + "; ".join(reasons))
    print(f"  IN above both BD and SD at {in_ok_pts}/{total_pts} grid points")
    if in_exceptions:
        print("  Exceptions where IN is not strictly above both:")
        for e in in_exceptions:
            print(f"    - {e}")
    else:
        print("  (no exceptions)")

    # ------- BD vs SD panel-level and point-level ordering -------
    print("\n--- BD vs SD ordering ---")
    panel_summary: List[str] = []
    total_panels = 0
    strict_bd_below = 0
    majority_bd_below = 0
    total_bd_vs_sd_pts = 0
    bd_below_pts = 0
    for axis in axes_list:
        for ds in datasets:
            rows = rows_by_ds[ds]
            x_col = AXIS_META[axis]["x_col"]
            bd_pts = {int(p[x_col]): float(p["var_grad_mean"])
                      for p in select(rows, axis, "BD", group)
                      if not bool(p.get("timed_out"))
                      and np.isfinite(float(p["var_grad_mean"]))}
            sd_pts = {int(p[x_col]): float(p["var_grad_mean"])
                      for p in select(rows, axis, "SD", group)
                      if not bool(p.get("timed_out"))
                      and np.isfinite(float(p["var_grad_mean"]))}
            common_xs = sorted(set(bd_pts) & set(sd_pts))
            if not common_xs:
                continue
            total_panels += 1
            n_below = sum(1 for x in common_xs if bd_pts[x] < sd_pts[x])
            total_bd_vs_sd_pts += len(common_xs)
            bd_below_pts += n_below
            if n_below == len(common_xs):
                strict_bd_below += 1
            if n_below > len(common_xs) / 2:
                majority_bd_below += 1
            panel_summary.append(
                f"({ds}, {axis}): BD<SD at {n_below}/{len(common_xs)} x "
                f"(x={common_xs})")
    print(f"  Strict BD<SD (all x)     : {strict_bd_below}/{total_panels} panels")
    print(f"  Majority BD<SD (> 50% x) : {majority_bd_below}/{total_panels} panels")
    print(f"  Overall BD<SD point count: {bd_below_pts}/{total_bd_vs_sd_pts} points "
          f"({100*bd_below_pts/max(total_bd_vs_sd_pts,1):.0f}%)")
    print("  Per-panel breakdown:")
    for s in panel_summary:
        print(f"    - {s}")

    # ------- Timed-out / non-finite flagged points -------
    print("\n--- Flagged points (open markers / top-edge x) ---")
    if not flagged:
        print("  (none)")
    else:
        for (ds, axis, variant, kind, xs) in flagged:
            tag = "TIMEOUT " if kind == "timed_out" else "NONFINITE"
            print(f"  {tag}  {ds:10s} axis={axis:6s} variant={variant:3s}  "
                  f"{AXIS_META[axis]['x_col']}={xs}")

    print()
    print("=" * 78)


# =============================================================================
# CLI
# =============================================================================

MODE_DEFAULTS = {
    "main":     {"subdir": "main_torchdeq",   "axes": ["width", "depth"],
                 "layout": "3x2"},
    "appendix": {"subdir": "appendix_bptt",   "axes": ["T"],
                 "layout": "1x3"},
}


def main() -> None:
    p = argparse.ArgumentParser(description="Combine BP CSVs into a panel figure.")
    p.add_argument("--csvs", nargs="+", type=str, default=None,
                   help="Explicit list of bp_grid.csv paths, one per dataset.")
    p.add_argument("--datasets", nargs="+", type=str, default=None,
                   help="Dataset labels matching --csvs order. If omitted but "
                        "--csvs is given, tries to infer from path; if --mode "
                        "is used, defaults to NCI1/PROTEINS/MUTAG.")
    p.add_argument("--mode", type=str, default=None, choices=list(MODE_DEFAULTS),
                   help="Shortcut: --mode main uses "
                        "results/barren_plateau/<DS>/main_torchdeq/bp_grid.csv "
                        "for each dataset; --mode appendix uses appendix_bptt.")
    p.add_argument("--results_root", type=str, default="results/barren_plateau",
                   help="Root directory used by --mode to discover CSVs.")
    p.add_argument("--axes", nargs="+", type=str, default=None,
                   choices=list(AXIS_META.keys()),
                   help="Axes to plot. If omitted, derived from --mode or "
                        "defaults to width+depth.")
    p.add_argument("--layout", type=str, default=None,
                   choices=["3x2", "2x3", "1x3", "3x1"],
                   help="Panel layout. Default: 3x2 for main, 1x3 for appendix.")
    p.add_argument("--figure_class", type=str, default="single",
                   choices=["single", "wide"],
                   help="Target column class: 'single' (3.4\" wide, for IEEE "
                        "one-column figure) or 'wide' (7\" wide, for figure*).")
    p.add_argument("--group", type=str, default="ALL",
                   help="Parameter group to render (see barren_plateau_plot).")
    p.add_argument("--font_base", type=float, default=None,
                   help="Override base font size (default 9pt single / 9.5pt wide).")
    p.add_argument("--output", type=str, required=True,
                   help="Output figure path (.pdf recommended).")
    args = p.parse_args()

    # Resolve CSVs + dataset labels
    if args.csvs:
        csv_paths_list = [Path(c) for c in args.csvs]
        if args.datasets:
            if len(args.datasets) != len(csv_paths_list):
                raise SystemExit("--datasets must match --csvs length")
            labels = args.datasets
        else:
            # Infer from 3rd-to-last path component, e.g.
            # results/barren_plateau/NCI1/main_torchdeq/bp_grid.csv -> NCI1
            labels = []
            for c in csv_paths_list:
                parts = c.resolve().parts
                labels.append(parts[-3] if len(parts) >= 3 else c.stem)
    elif args.mode:
        spec = MODE_DEFAULTS[args.mode]
        labels = args.datasets or DATASETS_DEFAULT
        root = Path(args.results_root)
        csv_paths_list = [root / ds / spec["subdir"] / "bp_grid.csv"
                          for ds in labels]
    else:
        raise SystemExit("Either --csvs or --mode must be given.")

    missing = [str(p) for p in csv_paths_list if not p.exists()]
    if missing:
        raise SystemExit("Missing CSV paths:\n  " + "\n  ".join(missing))

    csv_paths = dict(zip(labels, csv_paths_list))

    # Resolve axes + layout defaults
    if args.axes:
        axes_list = args.axes
    elif args.mode:
        axes_list = MODE_DEFAULTS[args.mode]["axes"]
    else:
        axes_list = ["width", "depth"]
    layout = args.layout or (
        MODE_DEFAULTS[args.mode]["layout"] if args.mode else "3x2")

    print(f"[combine] datasets = {labels}")
    print(f"[combine] axes     = {axes_list}")
    print(f"[combine] layout   = {layout}  (figure_class={args.figure_class})")
    print(f"[combine] output   = {args.output}")

    summary = plot_grid(
        csv_paths=csv_paths,
        axes_list=axes_list,
        output_path=Path(args.output),
        layout=layout,
        figure_class=args.figure_class,
        group=args.group,
        font_base=args.font_base,
    )

    print(f"[combine] wrote {summary['output']}")
    print_todo_numbers(summary, csv_paths, axes_list, group=args.group)


if __name__ == "__main__":
    main()
