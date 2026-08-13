#!/usr/bin/env python3
"""Public-data structural-similarity plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_structure_benchmarks import read_annotations


def prepare_sucos_data(
    sucos_df: pd.DataFrame,
    annotation_file: Path,
) -> pd.DataFrame:
    """Annotate external SuCOS table with fragment/scaffold/filter metadata."""
    ann = read_annotations(annotation_file)

    if "query" not in sucos_df.columns:
        raise KeyError("SuCOS table is missing required column: query")

    sucos_df = sucos_df.copy()

    sucos_df = sucos_df.merge(
        ann[
            [
                "complex_id",
                "fragment_screen",
                "filtered",
            ]
        ],
        left_on="query",
        right_on="complex_id",
        how="left",
    )

    if sucos_df["complex_id"].isna().any():
        missing = sucos_df.loc[sucos_df["complex_id"].isna(), "query"].unique()
        raise ValueError(
            "Some SuCOS query values were not found in the annotation file. "
            f"Examples: {missing[:10]}"
        )

    return sucos_df


def plot_sucos_histogram(
    df: pd.DataFrame,
    column: str = "sucos_shape_pocket_qcov",
    bins: int = 20,
    subset: str = "all",
    filtered: bool = False,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the SuCOS/public-data similarity distribution.

    For subset='all', scaffolds and fragments are shown as a stacked histogram,
    with scaffolds at the bottom and fragments stacked on top.
    """
    subset = subset.lower()
    if subset not in {"all", "fragment", "scaffold"}:
        raise ValueError(
            f"Unknown subset={subset!r}. Expected 'all', 'fragment', or 'scaffold'."
        )

    required_cols = [column, "fragment_screen", "filtered"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"SuCOS dataframe is missing required columns: {missing}")

    plot_df = df.copy()

    if filtered:
        plot_df = plot_df[~plot_df["filtered"]].copy()

    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])

    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)

    if subset == "all":
        scaffold_values = plot_df.loc[~plot_df["fragment_screen"], column]
        fragment_values = plot_df.loc[plot_df["fragment_screen"], column]

        scaffold_label = f"Follow-on compounds (n={len(scaffold_values)})"
        fragment_label = f"Fragments (n={len(fragment_values)})"

        ax.hist(
            [scaffold_values, fragment_values],
            bins=bins,
            stacked=True,
            label=[scaffold_label, fragment_label],
            edgecolor="black",
            linewidth=0.8,
            color=["#4C78A8", "#F58518"],
        )

        handles, labels = ax.get_legend_handles_labels()

        ax.legend(
            handles[::-1],
            labels[::-1],
            frameon=False,
        )

    else:
        if subset == "fragment":
            values = plot_df.loc[plot_df["fragment_screen"], column]
            label = f"Fragments (n={len(values)})"
            color = "#F58518"
        else:
            values = plot_df.loc[~plot_df["fragment_screen"], column]
            label = f"Follow-on compounds (n={len(values)})"
            color = "#4C78A8"

        ax.hist(
            values,
            bins=bins,
            edgecolor="black",
            linewidth=0.8,
            color=color,
            label=label,
        )

        ax.legend(frameon=False)

    ax.set_xlabel("Similarity to public data")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def build_sucos_summary_table(
    sucos_df: pd.DataFrame,
    column: str = "sucos_shape_pocket_qcov",
    filtered: bool = True,
) -> pd.DataFrame:
    """Build manuscript-ready summary statistics for SuCOS/public-data similarity.

    The returned table has metrics as rows and dataset subsets as columns.
    """
    required_cols = [column, "fragment_screen", "filtered"]
    missing = [c for c in required_cols if c not in sucos_df.columns]
    if missing:
        raise KeyError(f"SuCOS dataframe is missing required columns: {missing}")

    df = sucos_df.copy()

    if filtered:
        df = df[~df["filtered"]].copy()

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    subsets = {
        "Curated complexes": df,
        "Curated fragments": df[df["fragment_screen"]],
        "Curated follow-on compounds": df[~df["fragment_screen"]],
    }

    metric_order = [
        "n",
        "Min.",
        "Q1",
        "Median",
        "Q3",
        "IQR",
        "Mean",
        "Max.",
        "<30 (%)",
        "<40 (%)",
        "<50 (%)",
    ]

    rows = []
    for metric in metric_order:
        row = {"Metric": metric}

        for subset_name, sub in subsets.items():
            values = sub[column].dropna()

            if metric == "n":
                value = int(values.shape[0])
            elif metric == "Min.":
                value = values.min()
            elif metric == "Q1":
                value = values.quantile(0.25)
            elif metric == "Median":
                value = values.median()
            elif metric == "Q3":
                value = values.quantile(0.75)
            elif metric == "IQR":
                value = values.quantile(0.75) - values.quantile(0.25)
            elif metric == "Mean":
                value = values.mean()
            elif metric == "Max.":
                value = values.max()
            elif metric == "<30 (%)":
                value = 100 * (values < 30).mean()
            elif metric == "<40 (%)":
                value = 100 * (values < 40).mean()
            elif metric == "<50 (%)":
                value = 100 * (values < 50).mean()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            row[subset_name] = value

        rows.append(row)

    return pd.DataFrame(rows)
