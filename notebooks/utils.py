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
# Box plots
# ---------------------------------------------------------------------------


def plot_boxplot(
    df: pd.DataFrame,
    col: str,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    ax=None,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Box plot for one numeric column, one box per group.

    Args:
        df:          DataFrame with the *group_by* column.
        col:         Numeric column to plot.
        group_by:    Column to group by (``"session_id"`` or ``"run_id"``).
        group_order: Explicit x-axis order. Defaults to sorted unique values.
        ax:          Existing Axes to draw on. Creates a new figure when None.
        figsize:     Figure size when creating a standalone figure.
    """
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    data = [df.loc[df[group_by] == g, col].dropna().values for g in groups]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(data, tick_labels=groups, vert=True)
    ax.set_title(col)
    ax.set_ylabel(col)
    ax.set_xticklabels(groups, rotation=45, ha="right")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    layout: str = "cols",
    figsize: tuple[int, int] | None = None,
) -> None:
    """Box plots for multiple numeric columns.

    Args:
        df:          DataFrame with the *group_by* column.
        cols:        Columns to plot, one subplot each.
        group_by:    Column to group by (``"session_id"`` or ``"run_id"``).
        group_order: Explicit x-axis order on every subplot.
        layout:      ``"cols"`` — all subplots in one row (default).
                     ``"rows"`` — all subplots in one column.
        figsize:     Overall figure size. Auto-computed when None.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for boxplots.")
        return
    n = len(cols)

    if layout == "rows":
        nrows, ncols = n, 1
        if figsize is None:
            figsize = (10, max(4, 4 * n))
    else:  # "cols" (default)
        nrows, ncols = 1, n
        if figsize is None:
            figsize = (max(6, 4 * n), 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False, squeeze=False)
    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, cols):
        plot_boxplot(df, col, group_by=group_by, group_order=group_order, ax=ax)

    plt.tight_layout()
    plt.show()


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
) -> None:
    """Grid of histograms: rows = groups, columns = metrics.

    Args:
        df:               DataFrame with the *group_by* column.
        cols:             Metric columns to show (one column per grid column).
        group_by:         Column to group by (``"session_id"`` or ``"run_id"``).
        group_order:      Explicit row order. Defaults to sorted unique values.
        bins:             Number of histogram bins.
        figsize_per_panel: ``(width, height)`` per individual subplot panel.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for histogram grid.")
        return

    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    n_rows = len(groups)
    n_cols = len(cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for row_idx, group in enumerate(groups):
        group_df = df[df[group_by] == group]
        for col_idx, col in enumerate(cols):
            ax = axes[row_idx][col_idx]
            data = group_df[col].dropna()
            ax.hist(data, bins=bins)
            if row_idx == 0:
                ax.set_title(col, fontsize=9)
            if col_idx == 0:
                label = str(group)
                ax.set_ylabel(label[:35] + ("…" if len(label) > 35 else ""), fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel(col, fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Histogram-grid wrappers for named metric sets
# ---------------------------------------------------------------------------


def plot_latency_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> None:
    """Histogram grid for latency metrics."""
    plot_metrics_grid(df, LATENCY_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_generation_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> None:
    """Histogram grid for generation throughput metrics."""
    plot_metrics_grid(df, GENERATION_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_retrieval_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> None:
    """Histogram grid for retrieval quality metrics."""
    plot_metrics_grid(df, RETRIEVAL_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_hardware_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> None:
    """Histogram grid for hardware resource metrics."""
    plot_metrics_grid(df, HARDWARE_COLS, group_by=group_by, group_order=group_order, **kwargs)


def plot_generation_quality_as_hist(
    df: pd.DataFrame,
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    **kwargs,
) -> None:
    """Histogram grid for generation quality / scoring metrics."""
    plot_metrics_grid(df, GENERATION_QUALITY_COLS, group_by=group_by, group_order=group_order, **kwargs)


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
) -> None:
    """Scatter plot of two metrics, one colour per group.

    Args:
        df:          DataFrame with the *group_by* column.
        x_col:       Column for the x-axis.
        y_col:       Column for the y-axis.
        group_by:    Column to colour/group by.
        group_order: Legend order. Defaults to sorted unique values.
        figsize:     Figure size.
    """
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=figsize)
    for label in groups:
        group = df[df[group_by] == label]
        ax.scatter(group[x_col], group[y_col], label=str(label), alpha=0.6, s=20)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col}  vs  {y_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


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
    figsize_per_col: tuple[int, int] = (5, 4),
) -> None:
    """Bar chart of a statistic for each group, one subplot per metric.

    Args:
        df:             DataFrame with the *group_by* column.
        cols:           Metric columns to plot.
        group_by:       Column to group by.
        group_order:    Explicit group order.
        stat:           Aggregation statistic: ``"mean"``, ``"std"``, ``"median"``.
        figsize_per_col: ``(width, height)`` per subplot.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for stats bar chart.")
        return
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    agg_fn = {"mean": "mean", "std": "std", "median": "median"}[stat]

    agg = df[df[group_by].isin(groups)].groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, agg.columns):
        ax.bar(range(len(groups)), agg[col].values)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{stat}({col})", fontsize=9)
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()


def plot_stats_line(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    figsize_per_col: tuple[int, int] = (5, 4),
) -> None:
    """Line plot of a statistic across groups, one subplot per metric.

    Each group is a point on the x-axis connected by a line, with a dot
    marker at every data point.

    Args:
        df:             DataFrame with the *group_by* column.
        cols:           Metric columns to plot.
        group_by:       Column to group by.
        group_order:    Explicit group order (x-axis sequence).
        stat:           Aggregation: ``"mean"``, ``"std"``, or ``"median"``.
        figsize_per_col: ``(width, height)`` per subplot.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for stats line plot.")
        return
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    agg_fn = {"mean": "mean", "std": "std", "median": "median"}[stat]
    agg = df[df[group_by].isin(groups)].groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
    if n == 1:
        axes = [axes]

    x = range(len(groups))
    for ax, col in zip(axes, agg.columns):
        ax.plot(x, agg[col].values, marker="o", linestyle="-")
        ax.set_xticks(list(x))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{stat}({col})", fontsize=9)
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()


def plot_stats_multi_line(
    df: pd.DataFrame,
    cols: list[str],
    group_by: str = "session_id",
    group_order: list[str] | None = None,
    stat: str = "mean",
    figsize: tuple[int, int] = (10, 5),
    normalize: bool = False,
) -> None:
    """Line plot with all metrics overlaid in a single axes, one line per metric.

    Useful for comparing the trend of several metrics across groups at once.

    Args:
        df:          DataFrame with the *group_by* column.
        cols:        Metric columns — each becomes one labelled line.
        group_by:    Column to group by (x-axis ticks).
        group_order: Explicit x-axis sequence. Defaults to sorted unique values.
        stat:        Aggregation: ``"mean"``, ``"std"``, or ``"median"``.
        figsize:     Figure size.
        normalize:   When True, each metric is min-max scaled to [0, 1] so
                     metrics with very different magnitudes can be compared
                     on the same y-axis.
    """
    cols = _present(df, cols)
    if not cols:
        print("No matching columns found for multi-line stats plot.")
        return
    groups = group_order or sorted(df[group_by].dropna().unique().tolist())
    agg_fn = {"mean": "mean", "std": "std", "median": "median"}[stat]
    agg = df[df[group_by].isin(groups)].groupby(group_by)[cols].agg(agg_fn).reindex(groups)

    if normalize:
        agg = (agg - agg.min()) / (agg.max() - agg.min()).replace(0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(groups))
    for col in agg.columns:
        ax.plot(x, agg[col].values, marker="o", linestyle="-", label=col)

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ylabel = f"{stat} (normalized)" if normalize else stat
    ax.set_ylabel(ylabel)
    ax.set_title(f"{stat} of metrics by {group_by}" + (" (normalized)" if normalize else ""))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
