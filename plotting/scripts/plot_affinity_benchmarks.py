#!/usr/bin/env python3
"""Affinity benchmark plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from plot_structure_benchmarks import PLOT_COLORS, add_panel_label

MAIN_AFFINITY_METHODS = {
    "molecular_weight",
    "clogp",
    "aev_plig",
    "gnina_redock",
    "smina_redock",
    "aqaffinity",
    "boltz_2",
}

AFFINITY_MODE_METHODS = {
    "gnina_crystal",
    "gnina_crystal_minimised",
    "gnina_redock",
    "gnina_fragment_crossdock",
    "smina_crystal",
    "smina_crystal_minimised",
    "smina_redock",
    "smina_fragment_crossdock",
}

REQUIRED_AFFINITY_METHODS = MAIN_AFFINITY_METHODS | AFFINITY_MODE_METHODS


def read_affinity_predictions(
    affinity_file: Path,
) -> pd.DataFrame:
    """Read the pre-filtered compound-level affinity prediction table."""
    affinity_file = Path(affinity_file)

    if not affinity_file.exists():
        raise FileNotFoundError(f"Affinity prediction table not found: {affinity_file}")

    df = pd.read_csv(affinity_file)

    required = {
        "method",
        "smiles",
        "experimental_pKD",
        "predicted_affinity",
        "fragalysis_codes",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Affinity prediction table is missing columns: {sorted(missing)}"
        )

    return df


def validate_affinity_dataset(
    affinity_df: pd.DataFrame,
    required_methods: set[str] = REQUIRED_AFFINITY_METHODS,
    expected_compounds: int = 490,
) -> None:
    """Validate the locked compound-level affinity benchmark.

    This table must already have been generated after structure-level curation.
    Plotting therefore does not re-filter Fragalysis codes. Every required method
    must contain exactly one row for each of the final 490 unique compounds.
    """
    present = set(affinity_df["method"].dropna().astype(str))
    missing = sorted(required_methods - present)
    if missing:
        available = ", ".join(sorted(present))
        raise ValueError(
            "Affinity input does not contain all methods required for the "
            "manuscript figures. Missing: "
            + ", ".join(missing)
            + ". Available methods: "
            + available
        )

    duplicated = affinity_df.duplicated(["method", "smiles"], keep=False)
    if duplicated.any():
        examples = (
            affinity_df.loc[duplicated, ["method", "smiles"]]
            .drop_duplicates()
            .head(10)
            .astype(str)
            .agg(" / ".join, axis=1)
            .tolist()
        )
        raise ValueError(
            "Affinity input contains duplicated method/compound rows. Examples: "
            + "; ".join(examples)
        )

    required_df = affinity_df.loc[affinity_df["method"].isin(required_methods)].copy()
    counts = required_df.groupby("method")["smiles"].nunique()
    wrong_counts = counts.loc[counts != expected_compounds]
    if not wrong_counts.empty:
        details = ", ".join(
            f"{method}={count}" for method, count in wrong_counts.items()
        )
        raise ValueError(
            f"Expected {expected_compounds} unique compounds for every required "
            f"affinity method; found {details}. The affinity table should be "
            "rebuilt after structure-level curation, not filtered at plotting time."
        )

    compound_sets = {
        method: set(group["smiles"]) for method, group in required_df.groupby("method")
    }
    reference_method = sorted(compound_sets)[0]
    reference_set = compound_sets[reference_method]
    inconsistent = [
        method
        for method, compounds in compound_sets.items()
        if compounds != reference_set
    ]
    if inconsistent:
        raise ValueError(
            "Required affinity methods do not contain the same compound set: "
            + ", ".join(sorted(inconsistent))
        )

    experimental_conflicts = required_df.groupby("smiles")["experimental_pKD"].nunique(
        dropna=True
    )
    experimental_conflicts = experimental_conflicts[experimental_conflicts > 1]
    if not experimental_conflicts.empty:
        raise ValueError(
            "Conflicting experimental_pKD values were found for "
            f"{len(experimental_conflicts)} compounds."
        )

    missing_values = (
        required_df[["experimental_pKD", "predicted_affinity"]].isna().any(axis=1)
    )
    if missing_values.any():
        bad = required_df.loc[missing_values, "method"].value_counts().to_dict()
        raise ValueError(
            "Required affinity methods contain missing experimental or predicted "
            f"values: {bad}"
        )


def validate_affinity_methods(
    affinity_df: pd.DataFrame,
    required_methods: set[str] = REQUIRED_AFFINITY_METHODS,
) -> None:
    """Backward-compatible method-only validation."""
    present = set(affinity_df["method"].dropna().astype(str))
    missing = sorted(required_methods - present)
    if missing:
        raise ValueError("Missing affinity methods: " + ", ".join(missing))


def summarize_affinity_method(
    affinity_df: pd.DataFrame,
    method: str,
    calculate_rmse: bool = True,
) -> dict[str, object]:
    """Calculate compound-level affinity metrics for one method."""
    method_df = affinity_df.loc[
        affinity_df["method"] == method,
        ["experimental_pKD", "predicted_affinity"],
    ].dropna()

    if method_df.empty:
        raise ValueError(f"No valid affinity predictions found for method: {method}")

    experimental = method_df["experimental_pKD"].to_numpy(dtype=float)
    predicted = method_df["predicted_affinity"].to_numpy(dtype=float)

    rmse = (
        float(np.sqrt(np.mean((predicted - experimental) ** 2)))
        if calculate_rmse
        else np.nan
    )

    rho = float(spearmanr(experimental, predicted).statistic)

    return {
        "method": method,
        "n": len(method_df),
        "RMSE": rmse,
        "Spearman rho": rho,
    }


def summarize_mean_pkd_baseline(
    affinity_df: pd.DataFrame,
) -> dict[str, object]:
    """Calculate RMSE for predicting the mean experimental pKD for every compound."""
    required = {
        "smiles",
        "experimental_pKD",
    }
    missing = required - set(affinity_df.columns)
    if missing:
        raise ValueError(
            f"Affinity prediction table is missing columns: {sorted(missing)}"
        )

    compound_df = (
        affinity_df[
            [
                "smiles",
                "experimental_pKD",
            ]
        ]
        .dropna()
        .drop_duplicates(subset="smiles")
        .copy()
    )

    experimental = compound_df["experimental_pKD"].to_numpy(dtype=float)

    mean_pkd = float(np.mean(experimental))

    predicted = np.full(
        shape=len(experimental),
        fill_value=mean_pkd,
        dtype=float,
    )

    rmse = float(np.sqrt(np.mean((predicted - experimental) ** 2)))

    return {
        "method": "mean_pkd",
        "n": len(compound_df),
        "RMSE": rmse,
        "Spearman rho": np.nan,
        "mean_prediction": mean_pkd,
    }


def build_affinity_metrics_table(
    affinity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main affinity benchmark metrics table."""
    rows = [
        summarize_mean_pkd_baseline(affinity_df),
        summarize_affinity_method(
            affinity_df,
            "molecular_weight",
            calculate_rmse=False,
        ),
        summarize_affinity_method(
            affinity_df,
            "clogp",
            calculate_rmse=False,
        ),
        summarize_affinity_method(
            affinity_df,
            "aev_plig",
            calculate_rmse=True,
        ),
        summarize_affinity_method(
            affinity_df,
            "gnina_redock",
            calculate_rmse=True,
        ),
        summarize_affinity_method(
            affinity_df,
            "smina_redock",
            calculate_rmse=True,
        ),
        summarize_affinity_method(
            affinity_df,
            "aqaffinity",
            calculate_rmse=True,
        ),
        summarize_affinity_method(
            affinity_df,
            "boltz_2",
            calculate_rmse=True,
        ),
    ]

    return pd.DataFrame(rows)


def draw_affinity_metric_bar(
    ax: plt.Axes,
    metrics_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    *,
    compact: bool = False,
) -> None:
    """Draw one affinity metric bar plot on an existing axis."""
    color_map = {
        "molecular_weight": "#8E44AD",
        "clogp": "#3FA34D",
        "aev_plig": "#4CC9F0",
        "gnina_redock": PLOT_COLORS["gnina"],
        "smina_redock": PLOT_COLORS["smina"],
        "aqaffinity": PLOT_COLORS["of3p2"],
        "boltz_2": PLOT_COLORS["boltz-2"],
        "mean_pkd": "#B0B0B0",
    }

    label_map = {
        "molecular_weight": "Molecular\nweight",
        "clogp": "cLogP",
        "aev_plig": "AEV-PLIG\n(crystal)",
        "gnina_redock": "GNINA\n(redock)",
        "smina_redock": "Smina\n(redock)",
        "aqaffinity": "AQAffinity",
        "boltz_2": "Boltz-2",
        "mean_pkd": "Mean pK$_D$",
    }

    df = metrics_df.dropna(subset=[metric_col]).copy()
    x = np.arange(len(df))

    colors = [color_map[m] for m in df["method"]]
    labels = [label_map[m] for m in df["method"]]

    value_fontsize = 11
    tick_fontsize = 11
    ylabel_fontsize = 14
    ytick_fontsize = 12

    ax.bar(
        x,
        df[metric_col],
        color=colors,
        edgecolor="black",
        linewidth=1,
        zorder=3,
    )

    text_offset = 0.03 if metric_col == "RMSE" else 0.015

    for i, value in enumerate(df[metric_col]):
        ax.text(
            i,
            value + text_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=0,
        ha="center",
        fontsize=tick_fontsize,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=ylabel_fontsize,
        fontweight="bold",
    )

    if metric_col == "Spearman rho":
        ax.set_ylim(0, 1)

    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_affinity_metrics(
    affinity_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (11, 4),
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plot affinity benchmark RMSE and Spearman bar charts."""
    metrics_df = build_affinity_metrics_table(affinity_df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=300,
    )

    draw_affinity_metric_bar(
        axes[0],
        metrics_df,
        metric_col="RMSE",
        ylabel=r"RMSE (pK$_{\mathbf{D}}$)",
        compact=False,
    )

    draw_affinity_metric_bar(
        axes[1],
        metrics_df,
        metric_col="Spearman rho",
        ylabel="Spearman correlation",
        compact=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, axes, metrics_df


def draw_affinity_scatterplots(
    axes: np.ndarray,
    affinity_df: pd.DataFrame,
    *,
    compact: bool = False,
    show_annotation: bool = True,
) -> None:
    """Draw affinity scatterplots on existing axes."""
    method_specs = [
        (
            "molecular_weight",
            "Molecular weight",
            "Molecular weight",
            "#8E44AD",
        ),
        (
            "gnina_redock",
            "GNINA (redock)",
            "Predicted affinity",
            PLOT_COLORS["gnina"],
        ),
        (
            "boltz_2",
            "Boltz-2",
            "Predicted affinity",
            PLOT_COLORS["boltz-2"],
        ),
    ]

    # Match the typography used in the affinity barplots.
    title_fontsize = 14 if compact else 16
    y_axis_label_fontsize = 14
    x_axis_label_fontsize = 14
    annotation_fontsize = 9 if compact else 10
    tick_fontsize = 12

    for ax, (method, title, ylabel, color) in zip(
        axes,
        method_specs,
    ):
        method_df = affinity_df.loc[
            affinity_df["method"] == method,
            [
                "experimental_pKD",
                "predicted_affinity",
            ],
        ].dropna()

        experimental = method_df["experimental_pKD"].to_numpy(dtype=float)
        predicted = method_df["predicted_affinity"].to_numpy(dtype=float)

        rho = float(
            spearmanr(
                experimental,
                predicted,
            ).statistic
        )

        ax.set_axisbelow(True)
        ax.grid(
            color="#C7CBD1",
            alpha=0.7,
            zorder=0,
        )

        ax.scatter(
            experimental,
            predicted,
            s=24 if compact else 24,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
            zorder=3,
        )

        slope, intercept = np.polyfit(
            experimental,
            predicted,
            1,
        )

        fit_x = np.linspace(
            experimental.min(),
            experimental.max(),
            200,
        )

        ax.plot(
            fit_x,
            slope * fit_x + intercept,
            color="black",
            linewidth=3,
            zorder=3,
        )
        ax.plot(
            fit_x,
            slope * fit_x + intercept,
            color=color,
            linewidth=2,
            zorder=4,
        )

        if method != "molecular_weight":
            rmse = float(np.sqrt(np.mean((predicted - experimental) ** 2)))

            ax.plot(
                [2.5, 8.0],
                [2.5, 8.0],
                linestyle=":",
                color="#3C4A5A",
                linewidth=1.5,
                zorder=2,
            )

            annotation = f"RMSE (pK$_D$) = {rmse:.2f}\nSpearman correlation = {rho:.2f}"

            ax.set_ylim(
                2.5,
                8.0,
            )
        else:
            annotation = f"Spearman correlation = {rho:.2f}"

        if show_annotation:
            ax.text(
                0.94,
                0.03,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=annotation_fontsize,
                zorder=5,
            )

        ax.set_xlim(
            2.5,
            8.0,
        )

        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight="bold",
            loc="center",
        )

        ax.set_xlabel(
            r"Experimental affinity (pK$_D$)",
            fontsize=x_axis_label_fontsize,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=y_axis_label_fontsize,
            fontweight="bold",
        )

        ax.tick_params(
            axis="both",
            labelsize=tick_fontsize,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_affinity_scatterplots(
    affinity_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (14, 4.4),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot molecular-weight, GNINA-redock, and Boltz-2 relationships."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        dpi=300,
    )

    draw_affinity_scatterplots(
        axes,
        affinity_df,
        compact=False,
        show_annotation=True,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.98,
        bottom=0.20,
        wspace=0.28,
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, axes


def plot_affinity_composite(
    affinity_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12.5, 8.0),
) -> tuple[plt.Figure, dict[str, object], pd.DataFrame]:
    """Create composite affinity figure with RMSE, Spearman, and scatterplots."""
    metrics_df = build_affinity_metrics_table(affinity_df)

    fig = plt.figure(
        figsize=figsize,
        dpi=300,
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.92, 1.08],
        hspace=0.28,
        wspace=0.14,
    )

    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_spearman = fig.add_subplot(gs[0, 1])

    scatter_gs = gs[1, :].subgridspec(
        1,
        3,
        wspace=0.14,
    )
    scatter_axes = np.array([fig.add_subplot(scatter_gs[0, i]) for i in range(3)])

    draw_affinity_metric_bar(
        ax_rmse,
        metrics_df,
        metric_col="RMSE",
        ylabel=r"RMSE (pK$_{\mathbf{D}}$)",
        compact=True,
    )

    draw_affinity_metric_bar(
        ax_spearman,
        metrics_df,
        metric_col="Spearman rho",
        ylabel="Spearman correlation",
        compact=True,
    )

    draw_affinity_scatterplots(
        scatter_axes,
        affinity_df,
        compact=True,
        show_annotation=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        top=0.97,
        bottom=0.11,
    )

    add_panel_label(
        fig,
        ax_rmse,
        "A",
        x_offset=0.020,
        y_offset=0.010,
        fontsize=16,
    )

    add_panel_label(
        fig,
        ax_spearman,
        "B",
        x_offset=0.020,
        y_offset=0.010,
        fontsize=16,
    )

    add_panel_label(
        fig,
        scatter_axes[0],
        "C",
        x_offset=0.020,
        y_offset=0.005,
        fontsize=16,
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    axes_dict = {
        "rmse": ax_rmse,
        "spearman": ax_spearman,
        "scatter": scatter_axes,
    }

    return fig, axes_dict, metrics_df


def build_affinity_mode_metrics_table(
    affinity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build GNINA and Smina metrics across structural input modes."""
    rows = []

    mode_specs = {
        "GNINA": [
            ("Crystal", "gnina_crystal"),
            ("Crystal\n(minimised)", "gnina_crystal_minimised"),
            ("Redock", "gnina_redock"),
            ("Fragment\nCross-dock", "gnina_fragment_crossdock"),
        ],
        "Smina": [
            ("Crystal", "smina_crystal"),
            ("Crystal\n(minimised)", "smina_crystal_minimised"),
            ("Redock", "smina_redock"),
            ("Fragment\nCross-dock", "smina_fragment_crossdock"),
        ],
    }

    mean_baseline = summarize_mean_pkd_baseline(affinity_df)

    for family, specs in mode_specs.items():
        rows.append(
            {
                **mean_baseline,
                "family": family,
                "mode": "Mean pK$_D$",
            }
        )

        for mode, method in specs:
            row = summarize_affinity_method(
                affinity_df,
                method,
                calculate_rmse=True,
            )
            row["family"] = family
            row["mode"] = mode
            rows.append(row)

    return pd.DataFrame(rows)


def plot_affinity_mode_comparison(
    affinity_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 7),
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plot GNINA and Smina affinity metrics across structural modes."""
    metrics_df = build_affinity_mode_metrics_table(affinity_df)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        dpi=300,
    )

    family_colors = {
        "GNINA": PLOT_COLORS["gnina"],
        "Smina": PLOT_COLORS["smina"],
    }

    # Use one shared RMSE scale for GNINA and Smina.
    rmse_max = float(metrics_df["RMSE"].max())

    shared_rmse_ylim = (
        0,
        rmse_max * 1.15,
    )

    for row_idx, family in enumerate(
        [
            "GNINA",
            "Smina",
        ]
    ):
        family_df = metrics_df[metrics_df["family"] == family].reset_index(drop=True)

        # --------------------------------------------------------------
        # RMSE panel, including the mean-pKD baseline
        # --------------------------------------------------------------
        rmse_df = family_df.dropna(subset=["RMSE"]).reset_index(drop=True)

        x_rmse = np.arange(len(rmse_df))

        rmse_colors = [
            ("#B0B0B0" if method == "mean_pkd" else family_colors[family])
            for method in rmse_df["method"]
        ]

        rmse_ax = axes[
            row_idx,
            0,
        ]

        rmse_ax.bar(
            x_rmse,
            rmse_df["RMSE"],
            color=rmse_colors,
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )

        for i, value in enumerate(rmse_df["RMSE"]):
            rmse_ax.text(
                i,
                value + 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        rmse_ax.set_xticks(x_rmse)

        rmse_ax.set_xticklabels(
            rmse_df["mode"],
            rotation=0,
            ha="center",
            fontsize=10,
        )

        rmse_ax.set_ylabel(
            r"RMSE (pK$_{\mathbf{D}}$)",
            fontsize=14,
            fontweight="bold",
        )

        rmse_ax.set_ylim(*shared_rmse_ylim)

        rmse_ax.tick_params(
            axis="y",
            labelsize=12,
        )

        rmse_ax.set_axisbelow(True)

        rmse_ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5,
            zorder=0,
        )

        rmse_ax.spines["top"].set_visible(False)

        rmse_ax.spines["right"].set_visible(False)

        # --------------------------------------------------------------
        # Spearman panel, excluding the mean-pKD baseline
        # --------------------------------------------------------------
        spearman_df = family_df.dropna(subset=["Spearman rho"]).reset_index(drop=True)

        x_spearman = np.arange(len(spearman_df))

        spearman_ax = axes[
            row_idx,
            1,
        ]

        spearman_ax.bar(
            x_spearman,
            spearman_df["Spearman rho"],
            color=family_colors[family],
            edgecolor="black",
            linewidth=1,
            zorder=3,
        )

        for i, value in enumerate(spearman_df["Spearman rho"]):
            spearman_ax.text(
                i,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        spearman_ax.set_xticks(x_spearman)

        spearman_ax.set_xticklabels(
            spearman_df["mode"],
            rotation=0,
            ha="center",
            fontsize=11,
        )

        spearman_ax.set_ylabel(
            "Spearman correlation",
            fontsize=14,
            fontweight="bold",
        )

        spearman_ax.set_ylim(
            0,
            1,
        )

        spearman_ax.tick_params(
            axis="y",
            labelsize=12,
        )

        spearman_ax.set_axisbelow(True)

        spearman_ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5,
            zorder=0,
        )

        spearman_ax.spines["top"].set_visible(False)

        spearman_ax.spines["right"].set_visible(False)

    fig.tight_layout(
        h_pad=2.5,
        w_pad=2.0,
        rect=[
            0.03,
            0.0,
            1.0,
            0.92,
        ],
    )

    # Add headings after tight_layout has fixed the axes positions.
    for row_idx, family in enumerate(
        [
            "GNINA",
            "Smina",
        ]
    ):
        left = (
            axes[
                row_idx,
                0,
            ]
            .get_position()
            .x0
        )

        right = (
            axes[
                row_idx,
                1,
            ]
            .get_position()
            .x1
        )

        top = max(
            axes[
                row_idx,
                0,
            ]
            .get_position()
            .y1,
            axes[
                row_idx,
                1,
            ]
            .get_position()
            .y1,
        )

        heading_y = top + 0.02

        fig.text(
            (left + right) / 2,
            heading_y,
            family,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

        fig.text(
            left - 0.040,
            heading_y,
            chr(ord("A") + row_idx),
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, axes, metrics_df
