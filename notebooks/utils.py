"""Utility functions for RAG evaluation metric exploration notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Column groups (filter to present columns at call time)
# ---------------------------------------------------------------------------

LATENCY_COLS = [
    "latency_e2e_latency_ms",
    "latency_ttft_ms",
    "latency_llm_generation_latency_ms",
    "latency_retrieval_latency_ms",
    "latency_predicted_per_token_ms",
]

GENERATION_COLS = [
    "generation_tokens_per_second",
    "generation_completion_tokens",
    "generation_prompt_tokens",
]

RETRIEVAL_COLS = [
    "metrics_recall_at_k",
    "metrics_precision_at_k",
    "metrics_mrr",
]

HARDWARE_COLS = [
    "hardware_max_ram_usage_mb",
    "hardware_avg_cpu_utilization_pct",
    "hardware_swap_in_bytes",
    "hardware_swap_out_bytes",
]

GENERATION_QUALITY_COLS = [
    "score_Correctness",
    "score_Context Recall",
    "score_Evaluate Hallucination V2",
    "score_Faithfulness custom",
    "metrics_recall_at_k",
    "metrics_mrr",
]


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only columns from *cols* that exist in *df*."""
    return [c for c in cols if c in df.columns]


def _numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric columns that are numeric and non-boolean (safe for quantile ops)."""
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in ("metrics_", "latency_", "generation_", "hardware_", "score_"))
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].dtype != bool
    ]


# ---------------------------------------------------------------------------
# Outlier helpers
# ---------------------------------------------------------------------------

#: Identity columns shown alongside metric values in outlier inspection tables.
_ID_COLS = ["session_id", "run_id", "claim_id", "trace_id"]


def find_outliers(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Return a summary DataFrame of outlier rows detected across *cols*.

    Each outlier row from the source DataFrame is tagged with the column(s)
    that triggered the flag plus its actual value and the computed bounds.

    Args:
        df:        Source DataFrame.
        cols:      Columns to inspect. Defaults to all numeric metric cols.
        method:    ``"iqr"``  — flag values outside ``median ± threshold × IQR``.
                   ``"zscore"`` — flag values with ``|z| > threshold``.
        threshold: Multiplier for IQR or z-score cutoff.

    Returns:
        DataFrame with columns: ``outlier_col``, ``value``, ``lower``,
        ``upper``, plus the identity columns and all inspected metric cols.
    """
    if cols is None:
        cols = _numeric_metric_cols(df)
    cols = _present(df, cols)

    id_cols = _present(df, _ID_COLS)
    records = []

    for col in cols:
        series = df[col].dropna()
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        else:  # zscore
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                continue
            lower = mu - threshold * sigma
            upper = mu + threshold * sigma

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_rows = df.loc[mask & df[col].notna()].copy()
        if outlier_rows.empty:
            continue

        outlier_rows["outlier_col"] = col
        outlier_rows["value"] = outlier_rows[col]
        outlier_rows["lower"] = round(lower, 4)
        outlier_rows["upper"] = round(upper, 4)
        records.append(outlier_rows)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    front = ["outlier_col", "value", "lower", "upper"] + id_cols
    metric_display = [c for c in cols if c in result.columns]
    return result[front + [c for c in metric_display if c not in front]].sort_values(
        ["outlier_col", "value"], ascending=[True, False]
    )


def exclude_outliers(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Return *df* with rows that are outliers in **any** of *cols* removed.

    Args:
        df:        Source DataFrame.
        cols:      Columns to check. Defaults to all numeric metric cols.
        method:    ``"iqr"`` or ``"zscore"`` (see :func:`find_outliers`).
        threshold: IQR multiplier or z-score cutoff.

    Returns:
        Filtered DataFrame (copy). Rows flagged as an outlier in at least one
        column are dropped; all others are kept.
    """
    if cols is None:
        cols = _numeric_metric_cols(df)
    cols = _present(df, cols)

    keep_mask = pd.Series(True, index=df.index)

    for col in cols:
        series = df[col]
        valid = series.dropna()
        if method == "iqr":
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        else:  # zscore
            mu, sigma = valid.mean(), valid.std()
            if sigma == 0:
                continue
            lower = mu - threshold * sigma
            upper = mu + threshold * sigma

        outlier_mask = (series < lower) | (series > upper)
        keep_mask &= ~outlier_mask.fillna(False)

    n_dropped = (~keep_mask).sum()
    pct = round(n_dropped / len(df) * 100, 1)
    print(f"Excluded {n_dropped} / {len(df)} rows ({pct}%) as outliers "
          f"[method={method}, threshold={threshold}]")
    return df.loc[keep_mask].copy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_files(
    paths: list[str | Path] | None = None,
    folder: str | Path | None = None,
    pattern: str = "*langfuse_export*.parquet",
) -> pd.DataFrame:
    """Load one or more parquet export files into a single DataFrame.

    Args:
        paths:   Explicit list of file paths to load.
        folder:  Directory to glob all matching files from (uses *pattern*).
        pattern: Glob pattern applied when *folder* is given.

    At least one of *paths* or *folder* must be provided.
    When both are given they are combined.
    """
    if paths is None and folder is None:
        raise ValueError("Provide at least one of `paths` or `folder`.")

    resolved: list[Path] = []

    if paths:
        resolved += [Path(p) for p in paths]

    if folder:
        matched = sorted(Path(folder).glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
        resolved += matched

    if not resolved:
        raise FileNotFoundError("No files found to load.")

    frames = []
    for p in resolved:
        print(f"Loading: {p.name}")
        frames.append(pd.read_parquet(p))

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\nTotal shape: {df_all.shape}")
    return df_all


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_session_ids(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique session IDs present in *df*."""
    return sorted(df["session_id"].dropna().unique().tolist())


def get_run_ids(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique run IDs present in *df*."""
    if "run_id" not in df.columns:
        return []
    return sorted(df["run_id"].dropna().unique().tolist())


def print_dataset_info(df: pd.DataFrame) -> None:
    """Print shape, session IDs, run IDs, and all column names."""
    print(f"Shape: {df.shape}\n")

    sessions = get_session_ids(df)
    print(f"Session IDs ({len(sessions)}):")
    for s in sessions:
        print(f"  {s}")

    runs = get_run_ids(df)
    if runs:
        print(f"\nRun IDs ({len(runs)}):")
        for r in runs:
            print(f"  {r}")

    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  {c}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_labels(values: list, labels: dict | None) -> list[str]:
    """Map *values* through *labels* dict; fall back to str(value) if absent."""
    if not labels:
        return [str(v) for v in values]
    return [str(labels.get(v, v)) for v in values]


import re as _re
_STAT_RE = _re.compile(r"^p(\d+)$")


def _resolve_stat(stat: str):
    """Return a pandas-compatible agg function for *stat*.

    Supports ``"mean"``, ``"median"``, ``"std"``, and percentile notation
    ``"pN"`` where N is 0–100 (e.g. ``"p90"`` for the 90th percentile).
    """
    if stat in ("mean", "median", "std"):
        return stat
    m = _STAT_RE.match(stat)
    if m:
        n = int(m.group(1))
        if not 0 <= n <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {n}.")
        q = n / 100
        return lambda x: x.quantile(q)
    raise ValueError(
        f"Unknown stat '{stat}'. Use 'mean', 'median', 'std', or 'pN' (e.g. 'p90')."
    )


# ---------------------------------------------------------------------------
# Box plots
# ---------------------------------------------------------------------------


def plot_boxplot(
    df: pd.DataFrame,
    col: str,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    ax=None,
    figsize: tuple[int, int] = (10, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Box plot for one numeric column, one box per group.

    Args:
        df:          DataFrame with the *group_by* column.
        col:         Numeric column to plot.
        group_by:    Column to group by.
        group_order: Explicit x-axis order. Defaults to sorted unique values.
        ax:          Existing Axes to draw on. Creates a new figure when None.
        figsize:     Figure size when creating a standalone figure.
        title:       Override the subplot title (defaults to *col*).
        xlabel:      Override the x-axis label.
        ylabel:      Override the y-axis label (defaults to *col*).
        labels:      Dict mapping raw group values to display strings for ticks.

    Returns:
        ``(fig, ax)`` tuple.
    """
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    data = [df.loc[df[group_by] == g, col].dropna().values for g in groups]
    tick_labels = _apply_labels(groups, labels)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.boxplot(data, tick_labels=tick_labels, vert=True)
    ax.set_title(title if title is not None else col)
    ax.set_ylabel(ylabel if ylabel is not None else col)
    ax.set_xlabel(xlabel or "")
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_boxplots(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    layout: str = "cols",
    figsize: tuple[int, int] | None = None,
    axes=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Box plots for multiple numeric columns.

    Args:
        df:          DataFrame with the *group_by* column.
        cols:        Columns to plot, one subplot each.
        group_by:    Column to group by.
        group_order: Explicit x-axis order on every subplot.
        layout:      ``"cols"`` — one row (default). ``"rows"`` — one column.
        figsize:     Overall figure size. Auto-computed when None.
        axes:        Existing flat sequence of Axes (one per column in *cols*).
                     Creates a new figure when None.
        title:       Figure suptitle.
        xlabel:      X-axis label applied to every subplot.
        ylabel:      Y-axis label applied to every subplot (overrides metric name).
        labels:      Dict mapping raw group values to display strings.

    Returns:
        ``(fig, axes_list)`` tuple.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for boxplots.")
        return None, None
    n = len(cols)

    standalone = axes is None
    if standalone:
        if layout == "rows":
            nrows, ncols = n, 1
            if figsize is None:
                figsize = (10, max(4, 4 * n))
        else:
            nrows, ncols = 1, n
            if figsize is None:
                figsize = (max(6, 4 * n), 5)
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, sharey=False, squeeze=False)
        axes_flat = list(axes_arr.flatten())
    else:
        axes_flat = list(axes)
        fig = axes_flat[0].figure

    for ax, col in zip(axes_flat, cols):
        plot_boxplot(df, col, group_by=group_by, group_order=group_order, ax=ax,
                     labels=labels,
                     ylabel=ylabel,
                     xlabel=xlabel)

    if title:
        fig.suptitle(title)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, axes_flat[:n]


# ---------------------------------------------------------------------------
# Histogram grid (rows = groups, cols = metrics)
# ---------------------------------------------------------------------------


def plot_metrics_grid(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    bins: int = 30,
    figsize_per_panel: tuple[int, int] = (5, 3),
    axes=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Grid of histograms: rows = groups, columns = metrics.

    Args:
        df:               DataFrame with the *group_by* column.
        cols:             Metric columns to show (one column per grid column).
        group_by:         Column to group by.
        group_order:      Explicit row order. Defaults to sorted unique values.
        bins:             Number of histogram bins.
        figsize_per_panel: ``(width, height)`` per individual subplot panel.
        axes:             Existing 2-D array of Axes with shape
                          ``(n_groups, n_cols)``. Creates a new figure when None.
        title:            Figure suptitle.
        xlabel:           X-axis label override (applied to bottom row).
        ylabel:           Y-axis label override (applied to left column; default
                          is the shortened group name).
        labels:           Dict mapping raw group values to display strings for
                          the left-column y-axis labels.

    Returns:
        ``(fig, axes_2d)`` tuple where *axes_2d* has shape ``(n_groups, n_cols)``.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for histogram grid.")
        return None, None

    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    n_rows = len(groups)
    n_cols = len(cols)
    row_labels = _apply_labels(groups, labels)

    standalone = axes is None
    if standalone:
        fig, axes_2d = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
            squeeze=False,
        )
    else:
        import numpy as _np2
        axes_2d = _np2.asarray(axes).reshape(n_rows, n_cols)
        fig = axes_2d.flat[0].figure

    for row_idx, (group, row_label) in enumerate(zip(groups, row_labels)):
        group_df = df[df[group_by] == group]
        for col_idx, col in enumerate(cols):
            ax = axes_2d[row_idx][col_idx]
            data = group_df[col].dropna()
            ax.hist(data, bins=bins)
            if row_idx == 0:
                ax.set_title(col, fontsize=9)
            if col_idx == 0:
                yl = ylabel if ylabel is not None else row_label
                short = yl[:35] + ("…" if len(yl) > 35 else "")
                ax.set_ylabel(short, fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel or col, fontsize=8)

    if title:
        fig.suptitle(title)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, axes_2d


# ---------------------------------------------------------------------------
# Histogram-grid wrappers for named metric sets
# ---------------------------------------------------------------------------


def plot_latency_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Histogram grid for latency metrics."""
    return plot_metrics_grid(df, LATENCY_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_generation_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Histogram grid for generation throughput metrics."""
    return plot_metrics_grid(df, GENERATION_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_retrieval_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Histogram grid for retrieval quality metrics."""
    return plot_metrics_grid(df, RETRIEVAL_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_hardware_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Histogram grid for hardware resource metrics."""
    return plot_metrics_grid(df, HARDWARE_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_generation_quality_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> tuple:
    """Histogram grid for generation quality / scoring metrics."""
    return plot_metrics_grid(df, GENERATION_QUALITY_COLS, group_by=group_by, group_order=group_order, **kwargs)


# ---------------------------------------------------------------------------
# Scatter: two metrics
# ---------------------------------------------------------------------------


def plot_scatter_two_metrics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    figsize: tuple[int, int] = (8, 6),
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Scatter plot of two metrics, one colour per group.

    Args:
        df:          DataFrame with the *group_by* column.
        x_col:       Column for the x-axis.
        y_col:       Column for the y-axis.
        group_by:    Column to colour/group by.
        group_order: Legend order. Defaults to sorted unique values.
        figsize:     Figure size (ignored when *ax* is provided).
        ax:          Existing axes to draw into. When None a new figure is created.
        title:       Override plot title.
        xlabel:      Override x-axis label.
        ylabel:      Override y-axis label.
        labels:      Dict mapping raw group values to display names in the legend.
    """
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for grp in groups:
        group = df[df[group_by] == grp]
        legend_label = str(labels.get(grp, grp)) if labels else str(grp)
        ax.scatter(group[x_col], group[y_col], label=legend_label, alpha=0.6, s=20)
    ax.set_xlabel(xlabel if xlabel is not None else x_col)
    ax.set_ylabel(ylabel if ylabel is not None else y_col)
    ax.set_title(title if title is not None else f"{x_col}  vs  {y_col}")
    ax.legend(loc="best", fontsize=8)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Statistics summary & bar charts
# ---------------------------------------------------------------------------


def stats_by_group(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    group_by: str = "session_id",
) -> pd.DataFrame:
    """Return a DataFrame of descriptive statistics grouped by *group_by*.

    Columns: metric names.  Index: multi-index (group, statistic).
    """
    if cols is None:
        cols = _numeric_metric_cols(df)
    cols = _present(df, cols)
    return df.groupby(group_by)[cols].describe().round(3)


def plot_stats_bar(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    show_std: bool = False,
    show_band: bool = False,
    band_percentiles: tuple[float, float] = (25, 75),
    figsize_per_col: tuple[int, int] = (5, 4),
    axes=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Bar chart of a statistic for each group, one subplot per metric.

    Args:
        df:               DataFrame with the *group_by* column.
        cols:             Metric columns to plot.
        group_by:         Column to group by.
        group_order:      Explicit group order.
        stat:             Aggregation statistic: ``"mean"``, ``"std"``, ``"median"``.
        show_std:         Overlay capped error bars (±1 std) on each bar.
        show_band:        Overlay a translucent rectangle for the percentile
                          range defined by *band_percentiles*.
        band_percentiles: ``(lower, upper)`` percentile pair. Defaults to ``(25, 75)``.
        figsize_per_col:  ``(width, height)`` per subplot (ignored when *axes* provided).
        axes:             Sequence of existing axes to draw into. When None a new
                          figure is created. Must match ``len(cols)``.
        title:            Shared suptitle override.
        xlabel:           Override x-axis label (applied to all subplots).
        ylabel:           Override y-axis label (applied to all subplots).
        labels:           Dict mapping raw group values to display tick labels.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for stats bar chart.")
        return None, None
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    tick_labels = _apply_labels(groups, labels)
    agg_fn = _resolve_stat(stat)
    subset = df[df[group_by].isin(groups)]
    agg = subset.groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    std_agg = lo_agg = hi_agg = None
    if show_std:
        std_agg = subset.groupby(group_by)[cols].std().reindex(groups)
    if show_band:
        lo_q, hi_q = band_percentiles[0] / 100, band_percentiles[1] / 100
        lo_agg = subset.groupby(group_by)[cols].quantile(lo_q).reindex(groups)
        hi_agg = subset.groupby(group_by)[cols].quantile(hi_q).reindex(groups)

    n = len(cols)
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
        if n == 1:
            axes = [axes]
    else:
        fig = axes[0].figure

    x = list(range(len(groups)))
    for ax, col in zip(axes, agg.columns):
        bars = ax.bar(x, agg[col].values)

        if show_band:
            lo_vals = lo_agg[col].values
            hi_vals = hi_agg[col].values
            ax.bar(x, hi_vals - lo_vals, bottom=lo_vals,
                   color="grey", alpha=0.35,
                   label=f"p{int(band_percentiles[0])}–p{int(band_percentiles[1])}")

        if show_std:
            ax.errorbar(x, agg[col].values, yerr=std_agg[col].values,
                        fmt="none", capsize=6, color="black", linewidth=1.5,
                        label="±1 std")

        if show_std or show_band:
            ax.legend(loc="best", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{stat}({col})", fontsize=9)
        ax.set_ylabel(ylabel if ylabel is not None else col)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

    if title is not None:
        fig.suptitle(title)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, list(axes)


def plot_grouped_bars(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    error: str | None = "std",
    percentiles: tuple[float, float] = (25, 75),
    bar_width: float = 0.8,
    figsize: tuple[int, int] = (10, 5),
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Grouped bar chart: one cluster per metric, one bar per group.

    The metrics in *cols* are placed in order along the x-axis. Within each
    metric's cluster the bars are centered on the x tick. Error bars are drawn
    in the same colour as their bar (no grey).

    Args:
        df:           DataFrame with the *group_by* column.
        cols:         Metric columns — displayed left-to-right in this order.
        group_by:     Column that defines the bar groups (legend entries).
        group_order:  Explicit group order within each cluster. Defaults to
                      sorted unique values.
        stat:         Aggregation: ``"mean"``, ``"median"``, or ``"std"``.
        error:        Error bar style.
                      ``"std"`` — symmetric ±1 std bars.
                      ``"percentile"`` — asymmetric bars from *percentiles*
                      to the stat value; coloured to match each bar.
                      ``None`` — no error bars.
        percentiles:  ``(lower, upper)`` percentile pair used when
                      ``error="percentile"``. Defaults to ``(25, 75)``.
        bar_width:    Total width of each cluster (all bars combined).
                      Defaults to ``0.8``.
        figsize:      Figure size (ignored when *ax* is provided).
        ax:           Existing axes to draw into. When None a new figure is
                      created.
        title:        Override plot title.
        xlabel:       Override x-axis label.
        ylabel:       Override y-axis label.
        labels:       Dict mapping raw group values to legend display strings.

    Returns:
        ``(fig, ax)`` tuple.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for grouped bar chart.")
        return None, None

    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    n_groups = len(groups)
    subset = df[df[group_by].isin(groups)]

    agg_fn = _resolve_stat(stat)
    agg = subset.groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    std_agg = lo_agg = hi_agg = None
    if error == "std":
        std_agg = subset.groupby(group_by)[cols].std().reindex(groups)
    elif error == "percentile":
        lo_q, hi_q = percentiles[0] / 100, percentiles[1] / 100
        lo_agg = subset.groupby(group_by)[cols].quantile(lo_q).reindex(groups)
        hi_agg = subset.groupby(group_by)[cols].quantile(hi_q).reindex(groups)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = _np.arange(len(cols))
    w = bar_width / n_groups
    offsets = [(-n_groups / 2 + i + 0.5) * w for i in range(n_groups)]
    legend_labels = _apply_labels(groups, labels)

    for i, (grp, offset, legend_label) in enumerate(zip(groups, offsets, legend_labels)):
        heights = agg.loc[grp, cols].values.astype(float)
        bars = ax.bar(x + offset, heights, width=w, label=legend_label)
        bar_color = bars[0].get_facecolor()

        if error == "std":
            yerr = std_agg.loc[grp, cols].values.astype(float)
            ax.errorbar(
                x + offset, heights, yerr=yerr,
                fmt="none", capsize=4, color=bar_color, linewidth=1.2,
            )
        elif error == "percentile":
            lo = lo_agg.loc[grp, cols].values.astype(float)
            hi = hi_agg.loc[grp, cols].values.astype(float)
            yerr = _np.array([heights - lo, hi - heights])
            ax.errorbar(
                x + offset, heights, yerr=yerr,
                fmt="none", capsize=4, color=bar_color, linewidth=1.2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)

    _error_label = (
        f"  [error: ±1 std]" if error == "std"
        else f"  [error: p{int(percentiles[0])}–p{int(percentiles[1])}]" if error == "percentile"
        else ""
    )
    ax.set_title(title if title is not None else f"{stat} by metric{_error_label}")
    ax.set_ylabel(ylabel if ylabel is not None else stat)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend(loc="best", fontsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_stats_line(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    show_std: bool = False,
    show_band: bool = False,
    band_percentiles: tuple[float, float] = (25, 75),
    figsize_per_col: tuple[int, int] = (5, 4),
    axes=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Line plot of a statistic across groups, one subplot per metric.

    Each group is a point on the x-axis connected by a line, with a dot
    marker at every data point.

    Args:
        df:               DataFrame with the *group_by* column.
        cols:             Metric columns to plot.
        group_by:         Column to group by.
        group_order:      Explicit group order (x-axis sequence).
        stat:             Aggregation: ``"mean"``, ``"std"``, or ``"median"``.
        show_std:         Overlay capped error bars (±1 std) at every point.
        show_band:        Draw a shaded fill between *band_percentiles*.
        band_percentiles: ``(lower, upper)`` percentile pair. Defaults to ``(25, 75)``.
        figsize_per_col:  ``(width, height)`` per subplot (ignored when *axes* provided).
        axes:             Sequence of existing axes to draw into. When None a new
                          figure is created. Must match ``len(cols)``.
        title:            Shared suptitle override.
        xlabel:           Override x-axis label (applied to all subplots).
        ylabel:           Override y-axis label (applied to all subplots).
        labels:           Dict mapping raw group values to display tick labels.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for stats line plot.")
        return None, None
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    tick_labels = _apply_labels(groups, labels)
    agg_fn = _resolve_stat(stat)
    subset = df[df[group_by].isin(groups)]
    agg = subset.groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    std_agg = lo_agg = hi_agg = None
    if show_std:
        std_agg = subset.groupby(group_by)[cols].std().reindex(groups)
    if show_band:
        lo_q, hi_q = band_percentiles[0] / 100, band_percentiles[1] / 100
        lo_agg = subset.groupby(group_by)[cols].quantile(lo_q).reindex(groups)
        hi_agg = subset.groupby(group_by)[cols].quantile(hi_q).reindex(groups)

    n = len(cols)
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
        if n == 1:
            axes = [axes]
    else:
        fig = axes[0].figure

    x = list(range(len(groups)))
    for ax, col in zip(axes, agg.columns):
        line, = ax.plot(x, agg[col].values, marker="o", linestyle="-")
        c = line.get_color()

        if show_band:
            ax.fill_between(x, lo_agg[col].values, hi_agg[col].values,
                            alpha=0.2, color=c,
                            label=f"p{int(band_percentiles[0])}–p{int(band_percentiles[1])}")
        if show_std:
            ax.errorbar(x, agg[col].values, yerr=std_agg[col].values,
                        fmt="none", capsize=6, color=c, alpha=0.7, linewidth=1.5,
                        label="±1 std")
        if show_band or show_std:
            ax.legend(loc="best", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{stat}({col})", fontsize=9)
        ax.set_ylabel(ylabel if ylabel is not None else col)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

    if title is not None:
        fig.suptitle(title)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, list(axes)


def plot_stats_multi_line(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    figsize: tuple[int, int] = (10, 5),
    normalize: bool = False,
    show_std: bool = False,
    show_band: bool = False,
    band_percentiles: tuple[float, float] = (25, 75),
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Line plot with all metrics overlaid in a single axes, one line per metric.

    Useful for comparing the trend of several metrics across groups at once.

    Args:
        df:               DataFrame with the *group_by* column.
        cols:             Metric columns — each becomes one labelled line.
        group_by:         Column to group by (x-axis ticks).
        group_order:      Explicit x-axis sequence. Defaults to sorted unique values.
        stat:             Aggregation: ``"mean"``, ``"std"``, or ``"median"``.
        figsize:          Figure size (ignored when *ax* is provided).
        normalize:        When True, each metric is min-max scaled to [0, 1] so
                          metrics with very different magnitudes can be compared
                          on the same y-axis.
        show_std:         Overlay capped error bars (±1 std) at every point on
                          each line, using the same colour as the line.
        show_band:        Draw a shaded fill between *band_percentiles* for each
                          metric (same colour, lower opacity).
        band_percentiles: ``(lower, upper)`` percentile pair. Defaults to ``(25, 75)``.
        ax:               Existing axes to draw into. When None a new figure is created.
        title:            Override plot title.
        xlabel:           Override x-axis label.
        ylabel:           Override y-axis label.
        labels:           Dict mapping raw group values to display tick labels.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for multi-line stats plot.")
        return None, None
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    tick_labels = _apply_labels(groups, labels)
    agg_fn = _resolve_stat(stat)
    subset = df[df[group_by].isin(groups)]
    agg = subset.groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    std_agg = lo_agg = hi_agg = None
    if show_std:
        std_agg = subset.groupby(group_by)[cols].std().reindex(groups)
    if show_band:
        lo_q, hi_q = band_percentiles[0] / 100, band_percentiles[1] / 100
        lo_agg = subset.groupby(group_by)[cols].quantile(lo_q).reindex(groups)
        hi_agg = subset.groupby(group_by)[cols].quantile(hi_q).reindex(groups)

    if normalize:
        col_min = agg.min()
        col_max = agg.max()
        scale = (col_max - col_min).replace(0, 1)
        agg = (agg - col_min) / scale
        if show_band:
            lo_agg = (lo_agg - col_min) / scale
            hi_agg = (hi_agg - col_min) / scale
        if show_std:
            std_agg = std_agg / scale  # std scales but does not shift

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = list(range(len(groups)))
    for col in agg.columns:
        line, = ax.plot(x, agg[col].values, marker="o", linestyle="-", label=col)
        c = line.get_color()
        if show_band:
            ax.fill_between(x, lo_agg[col].values, hi_agg[col].values,
                            alpha=0.15, color=c)
        if show_std:
            ax.errorbar(x, agg[col].values, yerr=std_agg[col].values,
                        fmt="none", capsize=6, color=c, alpha=0.6, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    _ylabel = ylabel if ylabel is not None else (f"{stat} (normalized)" if normalize else stat)
    ax.set_ylabel(_ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    _title = title
    if _title is None:
        _title = f"{stat} of metrics by {group_by}"
        if normalize:
            _title += " (normalized)"
        extras = []
        if show_band:
            extras.append(f"band: p{int(band_percentiles[0])}–p{int(band_percentiles[1])}")
        if show_std:
            extras.append("bars: ±1 std")
        if extras:
            _title += f"  [{', '.join(extras)}]"
    ax.set_title(_title)
    ax.legend(loc="best", fontsize=8)
    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Heatmap: var1 × var2 for a single metric
# ---------------------------------------------------------------------------

import numpy as _np  # local alias to avoid polluting the module namespace


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    var1: str = "run_id",
    var2: str = "claim_id",
    agg: str = "mean",
    var1_order: list[str] | None = None,
    var2_order: list[str] | None = None,
    cmap: str = "viridis",
    annotate: bool | None = None,
    fmt: str = ".2f",
    figsize: tuple[int, int] | None = None,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Heatmap of *metric* aggregated over every (var1, var2) cell.

    Rows = *var2* values, columns = *var1* values.  Cell colour encodes the
    aggregated metric value; an optional text annotation shows the number.

    Args:
        df:         Source DataFrame.
        metric:     Numeric column to aggregate and display.
        var1:       Column for the x-axis (columns of the heatmap).
                    Defaults to ``"run_id"``.
        var2:       Column for the y-axis (rows of the heatmap).
                    Defaults to ``"claim_id"``.
        agg:        Aggregation function name: ``"mean"``, ``"median"``,
                    ``"std"``, ``"min"``, ``"max"``, ``"count"``.
                    Defaults to ``"mean"``.
        var1_order: Explicit column order. Defaults to sorted unique values.
        var2_order: Explicit row order. Defaults to sorted unique values.
        cmap:       Matplotlib colormap name. Defaults to ``"viridis"``.
        annotate:   Whether to print the value in each cell.  Defaults to
                    ``True`` when the grid has ≤ 400 cells, ``False`` otherwise.
        fmt:        Python format string for cell annotations. Defaults to
                    ``".2f"``.
        figsize:    Figure size. Auto-computed from grid dimensions when None
                    (ignored when *ax* is provided).
        ax:         Existing axes to draw into. When None a new figure is created.
        title:      Override plot title.
        xlabel:     Override x-axis label (default: *var1*).
        ylabel:     Override y-axis label (default: *var2*).
        labels:     Dict mapping raw *var1* column values to display tick labels.
    """
    if metric not in df.columns:
        print(f"Column '{metric}' not found in DataFrame.")
        return None, None

    agg_fn = {
        "mean": "mean", "median": "median", "std": "std",
        "min": "min", "max": "max", "count": "count",
    }.get(agg)
    if agg_fn is None:
        raise ValueError(f"Unknown agg '{agg}'. Choose from: mean, median, std, min, max, count.")

    # Build pivot: rows=var2, cols=var1
    grouped = df.groupby([var1, var2])[metric].agg(agg_fn).reset_index()
    pivot = grouped.pivot(index=var2, columns=var1, values=metric)

    # Apply ordering
    if var1_order:
        pivot = pivot.reindex(columns=[v for v in var1_order if v in pivot.columns])
    else:
        pivot = pivot.reindex(columns=sorted(pivot.columns))

    if var2_order:
        pivot = pivot.reindex(index=[v for v in var2_order if v in pivot.index])
    else:
        pivot = pivot.reindex(index=sorted(pivot.index))

    n_rows, n_cols = pivot.shape
    if annotate is None:
        annotate = n_rows * n_cols <= 400

    standalone = ax is None
    if standalone:
        if figsize is None:
            cell_w = max(0.6, min(2.0, 12 / n_cols))
            cell_h = max(0.3, min(0.8, 20 / n_rows))
            figsize = (max(6, cell_w * n_cols + 2), max(4, cell_h * n_rows + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mat = pivot.values.astype(float)
    vmin, vmax = _np.nanmin(mat), _np.nanmax(mat)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"{agg}({metric})", shrink=0.8)

    col_tick_labels = _apply_labels(list(pivot.columns), labels)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_xlabel(xlabel if xlabel is not None else var1)
    ax.set_ylabel(ylabel if ylabel is not None else var2)
    ax.set_title(title if title is not None else f"{agg}({metric})  —  {var2} × {var1}")

    if annotate:
        # Choose text colour (black/white) based on relative cell brightness
        text_thresh = (vmin + vmax) / 2
        font_size = max(5, min(9, 120 // max(n_rows, n_cols)))
        for row in range(n_rows):
            for col in range(n_cols):
                val = mat[row, col]
                if _np.isnan(val):
                    txt = "—"
                    color = "grey"
                else:
                    txt = format(val, fmt)
                    color = "white" if val < text_thresh else "black"
                ax.text(col, row, txt, ha="center", va="center",
                        fontsize=font_size, color=color)

    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Dot-line profile grid: one subplot per var2, line+dots across var1
# ---------------------------------------------------------------------------


def plot_dot_profiles(
    df: pd.DataFrame,
    metric: str,
    var1: str = "run_id",
    var2: str = "claim_id",
    agg: str = "mean",
    show_std: bool = False,
    var1_order: list[str] | None = None,
    var2_order: list[str] | None = None,
    sharey: bool = False,
    figsize_per_row: tuple[float, float] = (8, 1.5),
    axes=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
) -> tuple:
    """Profile grid: one subplot per *var2* value, line + dots across *var1*.

    Each row shows how *metric* varies across *var1* (e.g. run_id) for a
    single *var2* slice (e.g. claim_id). Optional black error bars show ±1 std
    computed from multiple observations per (var1, var2) cell.

    Args:
        df:              DataFrame containing the three relevant columns.
        metric:          Numeric column to plot.
        var1:            Column whose unique values become x-axis ticks within
                         every subplot (default ``"run_id"``).
        var2:            Column whose unique values each get their own subplot
                         row (default ``"claim_id"``).
        agg:             Aggregation per (var1, var2) cell: ``"mean"``,
                         ``"median"``, ``"min"``, or ``"max"``.
        show_std:        When True, overlay black ±1 std error bars on each dot.
        var1_order:      Explicit x-axis order. Defaults to sorted unique values.
        var2_order:      Explicit subplot row order. Defaults to sorted unique
                         values.
        sharey:          Share y-axis across all subplots (useful when metric
                         values are on the same scale). Defaults to False.
        figsize_per_row: ``(width, height)`` per subplot row. Total figure
                         height scales with the number of *var2* values.
                         Ignored when *axes* is provided.
        axes:            Flat sequence of existing Axes, one per *var2* value.
                         When None a new figure is created.
        title:           Figure suptitle override.
        xlabel:          Override x-axis label shown on the bottom subplot.
        ylabel:          Override y-axis label applied to every subplot
                         (defaults to the var2 value of that row).
        labels:          Dict mapping raw *var1* values to display tick labels.

    Returns:
        ``(fig, axes_list)`` tuple where *axes_list* has one entry per *var2*
        value in the order they appear.
    """
    if metric not in df.columns:
        print(f"Column '{metric}' not found in DataFrame.")
        return None, None

    agg_fns = {"mean": "mean", "median": "median", "min": "min", "max": "max"}
    if agg not in agg_fns:
        raise ValueError(f"Unknown agg '{agg}'. Choose from: {list(agg_fns)}")

    var1_vals = var1_order or sorted(df[var1].dropna().unique().tolist())
    var2_vals = var2_order or sorted(df[var2].dropna().unique().tolist())
    n_rows = len(var2_vals)
    x = _np.arange(len(var1_vals))
    tick_labels = _apply_labels(var1_vals, labels)

    # Aggregate metric per (var1, var2) cell
    grouped = df.groupby([var1, var2])[metric]
    agg_vals = grouped.agg(agg_fns[agg]).unstack(level=0)   # index=var2, cols=var1
    agg_vals = agg_vals.reindex(index=var2_vals, columns=var1_vals)

    std_vals = None
    if show_std:
        std_vals = grouped.std().unstack(level=0).reindex(index=var2_vals, columns=var1_vals)

    standalone = axes is None
    if standalone:
        fig, axes_arr = plt.subplots(
            n_rows, 1,
            figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
            sharey=sharey,
            squeeze=False,
        )
        axes_flat = list(axes_arr.flatten())
    else:
        axes_flat = list(axes)
        fig = axes_flat[0].figure

    for row_idx, var2_val in enumerate(var2_vals):
        ax = axes_flat[row_idx]
        y = agg_vals.loc[var2_val].values.astype(float)

        ax.plot(x, y, linestyle="-", linewidth=0.8, color="steelblue")
        ax.scatter(x, y, s=18, zorder=3, color="steelblue")

        if show_std and std_vals is not None:
            yerr = std_vals.loc[var2_val].values.astype(float)
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3,
                        color="black", linewidth=0.9, zorder=4)

        short = str(var2_val)
        short = short[:30] + ("…" if len(short) > 30 else "")
        ax.set_ylabel(
            (ylabel if ylabel is not None else short),
            fontsize=7, rotation=0, ha="right", va="center", labelpad=4,
        )
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelbottom=(row_idx == n_rows - 1), labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # x-axis labels only on the bottom subplot
    axes_flat[n_rows - 1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    axes_flat[n_rows - 1].set_xlabel(xlabel if xlabel is not None else var1)

    if title is not None:
        fig.suptitle(title)
    elif standalone:
        _std_note = "  [±1 std]" if show_std else ""
        fig.suptitle(f"{agg}({metric})  —  {var2} profiles across {var1}{_std_note}", fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()
    return fig, axes_flat[:n_rows]
