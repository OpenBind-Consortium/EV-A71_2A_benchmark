#!/usr/bin/env python3
"""
Generate benchmark figures and summary tables from prepared plotting inputs.

Author: Jochem Nelen (jochem.nelen@stats.ox.ac.uk)

Expected layout:
  analysis/plotting/
    plot_benchmark_figures.py
    figure_data/
      annotated_complexes.csv
      final_docking_pose_data.parquet
      final_cofolding_pose_data.parquet
      tsv_similarity_data_2021-09-30_v2.tsv
    figures/
    tables/

Inputs:
  - annotated_complexes.csv
  - final_docking_pose_data.parquet
  - final_cofolding_pose_data.parquet
  - optional SuCOS/public-data similarity table

Outputs:
  - PNG figures written to figures/
  - CSV summary tables written to tables/

Notes:
  The input tables are filtered and standardized versions of the original docking
  and co-folding results, prepared to be consistent and directly usable for plotting.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", "{:.3f}".format)


# =============================================================================
# Paths and constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Expected layout:
# analysis/plotting/
#   make_benchmark_plots.py
#   figure_data/
#   figures/
#   tables/

base_dir = SCRIPT_DIR
figure_data_dir = base_dir / "figure_data"
out_dir = base_dir / "figures"
tables_dir = base_dir / "tables"

annotation_file = figure_data_dir / "annotated_complexes.csv"
sucos_file = figure_data_dir / "tsv_similarity_data_2021-09-30_v2.tsv"

prepared_docking_file = figure_data_dir / "final_docking_pose_data.parquet"
prepared_cofolding_file = figure_data_dir / "final_cofolding_pose_data.parquet"

out_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

RMSD_THRESHOLD = 2.0
LDDT_PLI_THRESHOLD = 0.8

PLOT_COLORS = {
    "gnina_multi": "#E63232",
    "smina_multi": "#F07C12",
    "diffdock": "#FFC200",
    "af3": "#90BC1A",
    "boltz-1": "#21B534",
    "boltz-2": "#0095AC",
    "of3p2": "#1F64AD",
    "of3p2_ft": "#5DA5DA",
    "protenix": "#4040A0",
    "rf3": "#903498",
    "cofold_best": "#D3D3D3",
}

CLASSICAL_METHODS = [
    "gnina_multi",
    "smina_multi",
    "diffdock",
]

METHOD_LABELS = {
    "gnina_multi": "GNINA",
    "smina_multi": "Smina",
    "diffdock": "DiffDock",
    "af3": "AlphaFold3",
    "boltz-1": "Boltz-1",
    "boltz-2": "Boltz-2",
    "of3p2": "OpenFold3-p2",
    "of3p2_ft": "OpenFold3-p2\nfine-tuned",
    "protenix": "Protenix",
    "rf3": "RosettaFold 3",
    "cofold_best": "Best\nco-folding",
}

METHOD_ORDER = [
    "gnina_multi",
    "smina_multi",
    "diffdock",
    "af3",
    "boltz-1",
    "boltz-2",
    "of3p2",
    "protenix",
    "rf3",
]

COFOLD_METHOD_ORDER = [
    "af3",
    "boltz-1",
    "boltz-2",
    "of3p2",
    "protenix",
    "rf3",
]

DOCK_PROT_ORDER = [
    "redock",
    "A71EV2A_AF_prepared_aligned",
    "A71EV2A_8POA_prepared_aligned",
    "fragment_crossdock",
]

DOCK_PROT_LABELS = {
    "redock": "Redocking",
    "fragment_crossdock": "Cross-docking\nFragment screen",
    "A71EV2A_AF_prepared_aligned": "Cross-docking\nApo (AF2)",
    "A71EV2A_8POA_prepared_aligned": "Cross-docking\nApo (8POA)",
}


# =============================================================================
# Basic utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots and summary tables."
    )

    parser.add_argument(
        "--figure-data-dir",
        type=Path,
        default=figure_data_dir,
        help="Directory containing prepared input tables.",
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=out_dir,
        help="Output directory for figures.",
    )

    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=tables_dir,
        help="Output directory for summary tables.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top-ranked poses to consider (default: 25).",
    )

    parser.add_argument(
        "--skip-sucos",
        action="store_true",
        help="Skip SuCOS histogram plotting.",
    )

    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    """Read a Parquet, CSV, or TSV table based on file extension."""
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    raise ValueError(f"Unknown file type: {path}")


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean encodings to a boolean Series."""
    if series.dtype == "bool":
        return series.fillna(False)

    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def read_annotations(annotation_file: Path) -> pd.DataFrame:
    """Read complex annotations and standardize filtering columns."""
    ann = read_table(annotation_file)

    ann = ann.rename(
        columns={
            "complex_name": "complex_id",
            "smiles": "ligand_smiles",
            "pb_valid": "pb_valid_groundtruth",
        }
    )

    if "pb_valid_groundtruth" in ann.columns:
        ann["pb_valid_groundtruth"] = to_bool(ann["pb_valid_groundtruth"])
    else:
        ann["pb_valid_groundtruth"] = True

    if "artefact" in ann.columns:
        ann["artefact"] = to_bool(ann["artefact"])
    else:
        ann["artefact"] = False

    if "fragment_screen" in ann.columns:
        ann["fragment_screen"] = to_bool(ann["fragment_screen"])
    else:
        ann["fragment_screen"] = False

    ann["filtered"] = (~ann["pb_valid_groundtruth"]) | ann["artefact"]

    keep_cols = [
        "complex_id",
        "ligand_smiles",
        "fragment_screen",
        "artefact",
        "pb_valid_groundtruth",
        "filtered",
    ]

    ann = ann[[c for c in keep_cols if c in ann.columns]].copy()

    if ann["complex_id"].duplicated().any():
        dupes = ann.loc[ann["complex_id"].duplicated(), "complex_id"].unique()
        raise ValueError(f"Duplicate complex_id entries found: {dupes[:10]}")

    return ann


def get_denominator_complex_ids(
    annotation_file: Path,
    filtered: bool = False,
    scaffold_only: bool = False,
) -> list[str]:
    """Return complex IDs used as the denominator for success-rate calculations."""
    ann = read_annotations(annotation_file)

    if filtered:
        ann = ann[~ann["filtered"]].copy()

    if scaffold_only:
        ann = ann[~ann["fragment_screen"]].copy()

    return ann["complex_id"].dropna().drop_duplicates().sort_values().tolist()


def add_percent_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage columns derived from rate columns."""
    summary_df = summary_df.copy()
    summary_df["rmsd_valid_pct"] = 100 * summary_df["rmsd_valid_rate"]
    summary_df["success_valid_pct"] = 100 * summary_df["success_valid_rate"]
    summary_df["rmsd_only_pct"] = (
        summary_df["rmsd_valid_pct"] - summary_df["success_valid_pct"]
    )
    return summary_df


def sort_by_order(
    df: pd.DataFrame,
    column: str,
    order: list[str],
) -> pd.DataFrame:
    """Sort a dataframe by a predefined categorical order."""
    df = df.copy()
    present = [x for x in order if x in set(df[column].dropna())]
    df[column] = pd.Categorical(df[column], categories=present, ordered=True)
    return df.sort_values(column).reset_index(drop=True)


# =============================================================================
# Prepared data loading
# =============================================================================


def read_prepared_pose_data(
    prepared_docking_file: Path,
    prepared_cofolding_file: Path,
) -> pd.DataFrame:
    """Read, combine, and validate prepared docking and co-folding pose tables."""
    prepared_docking_file = Path(prepared_docking_file)
    prepared_cofolding_file = Path(prepared_cofolding_file)

    if not prepared_docking_file.exists():
        raise FileNotFoundError(
            f"Prepared docking file not found: {prepared_docking_file}"
        )

    if not prepared_cofolding_file.exists():
        raise FileNotFoundError(
            f"Prepared cofolding file not found: {prepared_cofolding_file}"
        )

    docking_df = read_table(prepared_docking_file)
    cofolding_df = read_table(prepared_cofolding_file)

    pose_df = pd.concat(
        [docking_df, cofolding_df],
        ignore_index=True,
        sort=False,
    )

    required_cols = [
        "source",
        "method",
        "dock_prot",
        "complex_id",
        "rank",
        "rank_score",
        "lig_rmsd",
        "lddt_pli",
        "pb_valid",
        "rmsd_valid",
        "lddt_pli_valid",
        "success_valid",
        "fragment_screen",
        "filtered",
    ]

    missing = [c for c in required_cols if c not in pose_df.columns]
    if missing:
        raise ValueError(f"Prepared pose data is missing required columns: {missing}")

    numeric_cols = [
        "seed",
        "sample",
        "rank",
        "rank_score",
        "rank_by_pair_iptm",
        "pair_iptm",
        "lig_rmsd",
        "lddt_pli",
        "pocket_qcov",
        "sucos_shape",
        "sucos_shape_pocket_qcov",
        "pocket_recall",
    ]

    for col in numeric_cols:
        if col in pose_df.columns:
            pose_df[col] = pd.to_numeric(pose_df[col], errors="coerce")

    bool_cols = [
        "pb_valid",
        "rmsd_valid",
        "lddt_pli_valid",
        "success_valid",
        "fragment_screen",
        "artefact",
        "pb_valid_groundtruth",
        "filtered",
    ]

    for col in bool_cols:
        if col in pose_df.columns:
            pose_df[col] = to_bool(pose_df[col])

    if pose_df["rank"].isna().any():
        bad = (
            pose_df[pose_df["rank"].isna()]
            .groupby(["source", "method"], dropna=False)
            .size()
            .reset_index(name="n_missing_rank")
        )

        raise ValueError(
            "Prepared pose data has missing `rank` values:\n"
            + bad.to_string(index=False)
        )

    return pose_df


# =============================================================================
# Selection and summarization
# =============================================================================


def select_predictions(
    pose_df: pd.DataFrame,
    source: str | None = None,
    method: str | None = None,
    dock_prot: str | None = None,
    top_n: int | None = 25,
    scaffold_only: bool = False,
    filtered: bool = False,
) -> pd.DataFrame:
    """Select pose-level predictions matching method, rank, and filtering criteria."""
    df = pose_df
    mask = pd.Series(True, index=df.index)

    if source is not None:
        mask = mask & (df["source"] == source)

    if method is not None:
        mask = mask & (df["method"] == method)

    if dock_prot is not None:
        mask = mask & (df["dock_prot"] == dock_prot)

    if top_n is not None:
        mask = mask & (df["rank"] < top_n)

    if filtered:
        mask = mask & (~df["filtered"])

    if scaffold_only:
        mask = mask & (~df["fragment_screen"])

    return df[mask].copy()


def summarize_oracle_any_pose(
    selected_df: pd.DataFrame,
    annotation_file: Path,
    label: str,
    method: str,
    dock_prot: str,
    filtered: bool = False,
    scaffold_only: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    """
    Summarize oracle success over selected poses.

    Per complex:
      rmsd_valid is true if any selected pose has RMSD <= 2 and is PB-valid
      success_valid is true if any selected pose has RMSD <= 2, LDDT-PLI >= 0.8,
      and is PB-valid

    Complexes missing predictions are counted as failures.
    """
    all_complex_ids = get_denominator_complex_ids(
        annotation_file,
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    if selected_df.empty:
        complex_df = pd.DataFrame(
            {
                "complex_id": all_complex_ids,
                "rmsd_valid": False,
                "success_valid": False,
                "n_poses": 0,
                "n_seeds": 0,
                "best_rmsd": np.nan,
                "best_lddt_pli": np.nan,
            }
        )
    else:
        complex_df = selected_df.groupby("complex_id", as_index=False).agg(
            rmsd_valid=("rmsd_valid", "any"),
            success_valid=("success_valid", "any"),
            n_poses=("complex_id", "size"),
            n_seeds=("seed", "nunique"),
            best_rmsd=("lig_rmsd", "min"),
            best_lddt_pli=("lddt_pli", "max"),
        )

        complex_df = (
            complex_df.set_index("complex_id").reindex(all_complex_ids).reset_index()
        )

        complex_df["rmsd_valid"] = (
            complex_df["rmsd_valid"].replace({np.nan: False}).astype(bool)
        )
        complex_df["success_valid"] = (
            complex_df["success_valid"].replace({np.nan: False}).astype(bool)
        )
        complex_df["n_poses"] = complex_df["n_poses"].fillna(0).astype(int)
        complex_df["n_seeds"] = complex_df["n_seeds"].fillna(0).astype(int)

    summary = {
        "label": label,
        "method": method,
        "dock_prot": dock_prot,
        "filtered": filtered,
        "scaffold_only": scaffold_only,
        "n_total": complex_df["complex_id"].nunique(),
        "n_with_predictions": int((complex_df["n_poses"] > 0).sum()),
        "mean_n_poses": complex_df["n_poses"].mean(),
        "n_rmsd_valid": int(complex_df["rmsd_valid"].sum()),
        "n_success_valid": int(complex_df["success_valid"].sum()),
        "rmsd_valid_rate": complex_df["rmsd_valid"].mean(),
        "success_valid_rate": complex_df["success_valid"].mean(),
    }

    return summary, complex_df


def build_docking_summaries(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    top_n: int = 25,
    filtered: bool = False,
    scaffold_only: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Build docking summary tables and complex-level result tables."""

    summaries = []
    complex_results = {}

    present_dock_prots = set(
        pose_df.loc[pose_df["source"] == "docking", "dock_prot"].dropna().unique()
    )

    dock_prots = [d for d in DOCK_PROT_ORDER if d in present_dock_prots]

    for dock_prot in dock_prots:
        for method in CLASSICAL_METHODS:
            selected = select_predictions(
                pose_df,
                source="docking",
                method=method,
                dock_prot=dock_prot,
                top_n=top_n,
                filtered=filtered,
                scaffold_only=scaffold_only,
            )

            summary, complex_df = summarize_oracle_any_pose(
                selected,
                annotation_file,
                label=f"{method} {dock_prot} top{top_n}",
                method=method,
                dock_prot=dock_prot,
                filtered=filtered,
                scaffold_only=scaffold_only,
            )

            summaries.append(summary)
            complex_results[(method, dock_prot)] = complex_df

    summary_df = add_percent_columns(pd.DataFrame(summaries))
    return summary_df, complex_results


def build_cofold_summaries(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    top_n: int = 25,
    filtered: bool = False,
    scaffold_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build co-folding summary tables and complex-level result tables."""
    summaries = []
    complex_results = {}

    present_methods = set(
        pose_df.loc[pose_df["source"] == "cofolding", "method"].dropna().unique()
    )

    cofold_methods = [m for m in COFOLD_METHOD_ORDER if m in present_methods]

    for method in cofold_methods:
        selected = select_predictions(
            pose_df,
            source="cofolding",
            method=method,
            dock_prot="cofold",
            top_n=top_n,
            filtered=filtered,
            scaffold_only=scaffold_only,
        )

        summary, complex_df = summarize_oracle_any_pose(
            selected,
            annotation_file,
            label=f"{method} top{top_n}",
            method=method,
            dock_prot="cofold",
            filtered=filtered,
            scaffold_only=scaffold_only,
        )

        summaries.append(summary)
        complex_results[method] = complex_df

    summary_df = add_percent_columns(pd.DataFrame(summaries))
    return summary_df, complex_results


# =============================================================================
# Plotting
# =============================================================================


def plot_sucos_histogram(
    df: pd.DataFrame,
    column: str | None = None,
    bins: int = 20,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the SuCOS/public-data similarity distribution."""
    if column is None:
        column = df.columns[-1]

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)

    ax.hist(
        values,
        bins=bins,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel("Similarity to public data")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_scaffold_dockprot_bars(
    scaffold_summary_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 5),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot scaffold-only docking success rates by docking protein preparation."""
    df = add_percent_columns(scaffold_summary_df)

    methods = [m for m in CLASSICAL_METHODS if m in set(df["method"])]
    dock_prots = [d for d in DOCK_PROT_ORDER if d in set(df["dock_prot"])]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    group_x = np.arange(len(dock_prots))
    bar_width = 0.22
    total_width = bar_width * len(methods)
    start = -total_width / 2 + bar_width / 2

    zero_bar_height = 0.1

    for j, method in enumerate(methods):
        method_df = (
            df[df["method"] == method]
            .set_index("dock_prot")
            .reindex(dock_prots)
            .reset_index()
        )

        success = method_df["success_valid_pct"].fillna(0).to_numpy()
        rmsd_total = method_df["rmsd_valid_pct"].fillna(0).to_numpy()
        rmsd_only = rmsd_total - success

        x = group_x + start + j * bar_width
        color = PLOT_COLORS.get(method, f"C{j}")

        success_plot = success.copy()
        rmsd_only_plot = rmsd_only.copy()

        zero_mask = rmsd_total == 0
        rmsd_only_plot[zero_mask] = zero_bar_height

        ax.bar(
            x,
            success_plot,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            label=METHOD_LABELS.get(method, method),
            zorder=3,
        )

        ax.bar(
            x,
            rmsd_only_plot,
            width=bar_width,
            bottom=success_plot,
            color="white",
            edgecolor=color,
            hatch="///",
            linewidth=1.2,
            zorder=3,
        )

        for xi, s, r in zip(x, success, rmsd_total):
            if s > 2:
                ax.text(
                    xi,
                    s / 2,
                    f"{s:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                    fontweight="bold",
                )

            if r > 0:
                ax.text(
                    xi,
                    r + 1,
                    f"{r:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
            else:
                ax.text(
                    xi,
                    zero_bar_height + 1,
                    "0%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="black",
                )

    xticklabels = [DOCK_PROT_LABELS.get(d, d) for d in dock_prots]

    ax.set_xticks(group_x)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 100)

    if title:
        ax.set_title(title)

    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    method_handles = [
        Patch(
            facecolor=PLOT_COLORS.get(method, f"C{i}"),
            edgecolor=PLOT_COLORS.get(method, f"C{i}"),
            label=METHOD_LABELS.get(method, method),
        )
        for i, method in enumerate(methods)
    ]

    metric_handles = [
        Patch(
            facecolor="gray",
            edgecolor="black",
            label=f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid & LDDT-PLI ≥ {LDDT_PLI_THRESHOLD:g}",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid",
        ),
    ]

    legend1 = ax.legend(
        handles=method_handles,
        frameon=False,
        loc="upper right",
        alignment="left",
        ncols=len(methods),
        bbox_to_anchor=(1.0, 1.0),
        fontsize=12,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=metric_handles,
        frameon=False,
        loc="upper right",
        alignment="left",
        ncols=1,
        bbox_to_anchor=(1.0, 0.92),
        fontsize=12,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_allmethods_grouped_bars(
    plot_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot co-folding and selected GNINA docking results in one grouped figure."""
    df = add_percent_columns(plot_df).reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for i, row in df.iterrows():
        method = str(row["method"])

        if row["plot_group"] == "GNINA":
            color = PLOT_COLORS["gnina_multi"]
        else:
            color = PLOT_COLORS.get(method, f"C{i}")

        success = row["success_valid_pct"]
        rmsd_only = row["rmsd_only_pct"]
        rmsd_total = row["rmsd_valid_pct"]

        ax.bar(
            i,
            success,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )

        ax.bar(
            i,
            rmsd_only,
            bottom=success,
            color="white",
            edgecolor=color,
            hatch="///",
            linewidth=1.2,
            zorder=3,
        )

        if success > 2:
            ax.text(
                i,
                success / 2,
                f"{success:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )

        if rmsd_total > 0:
            ax.text(
                i,
                rmsd_total + 1,
                f"{rmsd_total:.0f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["plot_label"], fontsize=12)

    n_cofold = int((df["plot_group"] == "Co-folding").sum())

    if 0 < n_cofold < len(df):
        ax.axvline(
            n_cofold - 0.5,
            color="black",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )

    trans = ax.get_xaxis_transform()

    if n_cofold > 0:
        ax.text(
            (n_cofold - 1) / 2,
            -0.12,
            "Co-folding",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=trans,
        )

    n_gnina = int((df["plot_group"] == "GNINA").sum())
    if n_gnina > 0:
        gnina_center = (n_cofold + len(df) - 1) / 2
        ax.text(
            gnina_center,
            -0.12,
            "GNINA",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            transform=trans,
        )

    legend_handles = [
        Patch(
            facecolor="gray",
            edgecolor="black",
            label=f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid & LDDT-PLI ≥ {LDDT_PLI_THRESHOLD:g}",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid",
        ),
    ]

    ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=12)

    ax.set_ylabel("Success rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_affinity_metrics(
    save_path: Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Plot affinity benchmark metrics and return the underlying metrics table."""
    affinity_df = pd.DataFrame(
        [
            {
                "method": "molecular_weight",
                "n": 494,
                "RMSE": np.nan,
                "Spearman rho": 0.483,
            },
            {"method": "clogp", "n": 494, "RMSE": np.nan, "Spearman rho": 0.174},
            {"method": "aev-plig", "n": 494, "RMSE": 1.090, "Spearman rho": 0.227},
            {"method": "gnina", "n": 494, "RMSE": 1.528, "Spearman rho": 0.453},
            {"method": "smina", "n": 494, "RMSE": 1.530, "Spearman rho": 0.255},
            {"method": "aqaffinity", "n": 494, "RMSE": 1.632, "Spearman rho": 0.117},
            {"method": "boltz-2", "n": 494, "RMSE": 1.091, "Spearman rho": 0.397},
        ]
    )

    color_map = {
        **PLOT_COLORS,
        "gnina": PLOT_COLORS["gnina_multi"],
        "smina": PLOT_COLORS["smina_multi"],
        "aqaffinity": PLOT_COLORS["of3p2"],
        "boltz-2": PLOT_COLORS["boltz-2"],
        "molecular_weight": "#8E44AD",
        "clogp": "#3FA34D",
        "aev-plig": "#4CC9F0",
    }

    label_map = {
        "molecular_weight": "Molecular\nweight",
        "clogp": "cLogP",
        "aev-plig": "AEV-PLIG",
        "gnina": "GNINA",
        "smina": "Smina",
        "aqaffinity": "AQAffinity",
        "boltz-2": "Boltz-2",
    }

    methods = affinity_df["method"].tolist()
    x = np.arange(len(methods))
    labels = [label_map.get(m, m) for m in methods]
    colors = [color_map[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)

    rmse_df = affinity_df.dropna(subset=["RMSE"]).copy()
    x_rmse = np.arange(len(rmse_df))
    rmse_colors = [color_map[m] for m in rmse_df["method"]]
    rmse_labels = [label_map.get(m, m) for m in rmse_df["method"]]

    axes[0].bar(
        x_rmse,
        rmse_df["RMSE"],
        color=rmse_colors,
        edgecolor=rmse_colors,
        linewidth=1.2,
        zorder=3,
    )

    for i, value in enumerate(rmse_df["RMSE"]):
        axes[0].text(
            i,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes[0].set_xticks(x_rmse)
    axes[0].set_xticklabels(rmse_labels, rotation=35, ha="right", fontsize=11)
    axes[0].set_ylabel("RMSE", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="y", labelsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].bar(
        x,
        affinity_df["Spearman rho"],
        color=colors,
        edgecolor=colors,
        linewidth=1.2,
        zorder=3,
    )

    for i, value in enumerate(affinity_df["Spearman rho"]):
        axes[1].text(
            i,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    axes[1].set_ylabel("Spearman ρ", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, affinity_df


# =============================================================================
# Figure construction
# =============================================================================


def make_docking_figure_variant(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    tables_dir: Path,
    figures_dir: Path,
    filtered: bool = False,
    suffix: str = "scaffold_unfiltered",
    title_suffix: str = "unfiltered",
    top_n: int = 25,
) -> pd.DataFrame:
    """Create the docking figure variant and write its source table."""
    scaffold_docking_summary_df, _ = build_docking_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        filtered=filtered,
        scaffold_only=True,
    )

    scaffold_docking_summary_df = scaffold_docking_summary_df[
        scaffold_docking_summary_df["method"].isin(CLASSICAL_METHODS)
    ].copy()

    scaffold_docking_summary_df.to_csv(
        tables_dir / f"figure_docking_{suffix}_summary.csv",
        index=False,
    )

    print(f"\nDocking figure variant: {suffix}")
    print(
        scaffold_docking_summary_df[
            [
                "method",
                "dock_prot",
                "n_total",
                "n_with_predictions",
                "n_rmsd_valid",
                "rmsd_valid_pct",
                "n_success_valid",
                "success_valid_pct",
            ]
        ].to_string(index=False)
    )

    plot_scaffold_dockprot_bars(
        scaffold_docking_summary_df,
        save_path=figures_dir / f"docking_{suffix}_stacked.png",
        figsize=(12, 5),
    )

    return scaffold_docking_summary_df


def make_allmethods_figure_variant(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    tables_dir: Path,
    figures_dir: Path,
    scaffold_only: bool = False,
    filtered: bool = False,
    suffix: str = "all",
    title_suffix: str = "Fragments and Scaffolds, unfiltered",
    top_n: int = 25,
) -> pd.DataFrame:
    """Create the all-methods comparison figure and write its source table."""
    docking_summary_df, _ = build_docking_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    cofold_summary_df, _ = build_cofold_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    selected_all = select_predictions(
        pose_df,
        source="cofolding",
        method=None,
        dock_prot="cofold",
        top_n=top_n,
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    selected_all = selected_all[selected_all["method"].isin(COFOLD_METHOD_ORDER)].copy()

    best_summary, _ = summarize_oracle_any_pose(
        selected_all,
        annotation_file,
        label="Best\nco-folding",
        method="cofold_best",
        dock_prot="cofold",
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    best_summary_df = add_percent_columns(pd.DataFrame([best_summary]))

    cofold_summary_df = pd.concat(
        [cofold_summary_df, best_summary_df],
        ignore_index=True,
        sort=False,
    )

    cofold_plot_df = cofold_summary_df.copy()
    cofold_plot_df["plot_group"] = "Co-folding"
    cofold_plot_df["plot_label"] = cofold_plot_df["method"].map(
        lambda x: METHOD_LABELS.get(x, x)
    )

    cofold_order = [
        m for m in COFOLD_METHOD_ORDER if m in set(cofold_plot_df["method"])
    ]

    if "cofold_best" in set(cofold_plot_df["method"]):
        cofold_order.append("cofold_best")

    cofold_plot_df["plot_order"] = cofold_plot_df["method"].map(
        {m: i for i, m in enumerate(cofold_order)}
    )

    gnina_plot_df = docking_summary_df[
        (docking_summary_df["method"] == "gnina_multi")
        & (
            docking_summary_df["dock_prot"].isin(
                [
                    "redock",
                    "fragment_crossdock",
                ]
            )
        )
    ].copy()

    gnina_plot_df["plot_group"] = "GNINA"
    gnina_plot_df["plot_label"] = gnina_plot_df["dock_prot"].map(
        {
            "redock": "Redocking",
            "fragment_crossdock": "Cross-docking\nFragment screen",
        }
    )

    gnina_dock_order = [
        "redock",
        "fragment_crossdock",
    ]

    gnina_plot_df["plot_order"] = gnina_plot_df["dock_prot"].map(
        {d: len(cofold_order) + i for i, d in enumerate(gnina_dock_order)}
    )

    allmethod_df = pd.concat(
        [cofold_plot_df, gnina_plot_df],
        ignore_index=True,
        sort=False,
    )

    allmethod_df = allmethod_df.sort_values("plot_order").reset_index(drop=True)

    allmethod_df.to_csv(
        tables_dir / f"allmethods_plot_data_{suffix}.csv",
        index=False,
    )

    print(f"\nAll methods figure variant: {suffix}")
    print(
        allmethod_df[
            [
                "plot_group",
                "method",
                "dock_prot",
                "plot_label",
                "n_total",
                "n_with_predictions",
                "n_rmsd_valid",
                "rmsd_valid_pct",
                "n_success_valid",
                "success_valid_pct",
            ]
        ].to_string(index=False)
    )

    plot_allmethods_grouped_bars(
        allmethod_df,
        save_path=figures_dir / f"allmethods_{suffix}_stacked.png",
        figsize=(max(12, 0.85 * len(allmethod_df)), 5),
    )

    return allmethod_df


def make_ft_comparison_figure(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    figures_dir: Path,
    filtered: bool = True,
    scaffold_only: bool = True,
    suffix: str = "of3p2_ft_comparison_scaffolds_filtered",
) -> tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Create the OpenFold3-p2 fine-tuning comparison figure."""
    rows = []

    comparisons = [
        {
            "source": "cofolding",
            "method": "of3p2",
            "dock_prot": "cofold",
            "plot_label": METHOD_LABELS["of3p2"],
        },
        {
            "source": "cofolding",
            "method": "of3p2_ft",
            "dock_prot": "cofold",
            "plot_label": METHOD_LABELS["of3p2_ft"],
        },
        {
            "source": "docking",
            "method": "gnina_multi",
            "dock_prot": "redock",
            "plot_label": "GNINA\nredock",
        },
    ]

    for top_n in [1, 25]:
        for item in comparisons:
            selected = select_predictions(
                pose_df,
                source=item["source"],
                method=item["method"],
                dock_prot=item["dock_prot"],
                top_n=top_n,
                filtered=filtered,
                scaffold_only=scaffold_only,
            )

            summary, _ = summarize_oracle_any_pose(
                selected,
                annotation_file,
                label=f"{item['method']} {item['dock_prot']} top{top_n}",
                method=item["method"],
                dock_prot=item["dock_prot"],
                filtered=filtered,
                scaffold_only=scaffold_only,
            )

            summary["top_n"] = top_n
            summary["plot_label"] = item["plot_label"]
            rows.append(summary)

    plot_df = add_percent_columns(pd.DataFrame(rows))

    plot_df["method_order"] = plot_df["method"].map(
        {
            "of3p2": 0,
            "of3p2_ft": 1,
            "gnina_multi": 2,
        }
    )

    plot_df = plot_df.sort_values(["top_n", "method_order"]).reset_index(drop=True)

    print(f"\nOF3p2 fine-tuning comparison: {suffix}")
    print(
        plot_df[
            [
                "top_n",
                "method",
                "dock_prot",
                "n_total",
                "n_with_predictions",
                "n_success_valid",
                "success_valid_pct",
            ]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    methods = ["of3p2", "of3p2_ft", "gnina_multi"]
    labels = [
        METHOD_LABELS["of3p2"],
        METHOD_LABELS["of3p2_ft"],
        "GNINA\nredock",
    ]

    x = np.arange(len(methods))
    bar_width = 0.55

    top1 = (
        (plot_df[plot_df["top_n"] == 1].set_index("method").reindex(methods))[
            "success_valid_pct"
        ]
        .fillna(0)
        .to_numpy()
    )

    top25 = (
        (plot_df[plot_df["top_n"] == 25].set_index("method").reindex(methods))[
            "success_valid_pct"
        ]
        .fillna(0)
        .to_numpy()
    )

    for i, method in enumerate(methods):
        color = PLOT_COLORS.get(method, f"C{i}")

        success_top1 = top1[i]
        success_top25 = top25[i]
        remainder = max(success_top25 - success_top1, 0)

        # Top 1 (solid)
        ax.bar(
            x[i],
            success_top1,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )

        # Remaining (Top25 - Top1) with hatch
        ax.bar(
            x[i],
            remainder,
            width=bar_width,
            bottom=success_top1,
            color="white",
            edgecolor=color,
            hatch=r"\\",
            linewidth=1.2,
            zorder=3,
        )

        # Top 1 label (inside)
        if success_top1 > 2:
            ax.text(
                x[i],
                success_top1 / 2,
                f"{success_top1:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

        # Top 25 label (top)
        if success_top25 > 0:
            ax.text(
                x[i],
                success_top25 + 1,
                f"{success_top25:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylabel("Success rate (%)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    topn_handles = [
        Patch(
            facecolor="gray",
            edgecolor="black",
            label="Top 1",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=r"\\",
            label="Top 25",
        ),
    ]

    ax.legend(
        handles=topn_handles,
        frameon=False,
        loc="upper left",
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        figures_dir / f"{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )

    return plot_df, fig, ax


# =============================================================================
# Diagnostics
# =============================================================================


def print_prepared_data_summary(
    pose_df: pd.DataFrame,
    annotation_file: Path,
) -> None:
    """Print row counts and method coverage for the prepared pose table."""
    print("\nPrepared data:")
    print("Docking rows:", int((pose_df["source"] == "docking").sum()))
    print("Cofolding rows:", int((pose_df["source"] == "cofolding").sum()))
    print("Combined rows:", len(pose_df))
    print("Denominator complexes:", len(get_denominator_complex_ids(annotation_file)))

    print("\nDocking method/dock_prot coverage:")
    print(
        pose_df[pose_df["source"] == "docking"]
        .groupby(["method", "dock_prot"])
        .agg(
            n_complexes=("complex_id", "nunique"),
            n_rows=("complex_id", "size"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .sort_values(["method", "dock_prot"])
        .to_string()
    )

    print("\nCofolding method coverage:")
    print(
        pose_df[pose_df["source"] == "cofolding"]
        .groupby("method")
        .agg(
            n_complexes=("complex_id", "nunique"),
            n_rows=("complex_id", "size"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .sort_values("method")
        .to_string()
    )


def print_summary_table(title: str, summary_df: pd.DataFrame) -> None:
    """Print a compact summary table to stdout."""
    print(f"\n{title}")
    cols = [
        "method",
        "dock_prot",
        "n_total",
        "n_with_predictions",
        "n_rmsd_valid",
        "rmsd_valid_pct",
        "n_success_valid",
        "success_valid_pct",
    ]

    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[cols].to_string(index=False))


# =============================================================================
# Main script
# =============================================================================


def main() -> None:
    """Generate all benchmark figures and summary tables."""

    args = parse_args()

    figure_data_dir = args.figure_data_dir
    figures_dir = args.figures_dir
    tables_dir = args.tables_dir
    top_n = args.top_n

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = figure_data_dir / "annotated_complexes.csv"
    sucos_file = figure_data_dir / "tsv_similarity_data_2021-09-30_v2.tsv"

    prepared_docking_file = figure_data_dir / "final_docking_pose_data.parquet"
    prepared_cofolding_file = figure_data_dir / "final_cofolding_pose_data.parquet"

    pose_df = read_prepared_pose_data(
        prepared_docking_file=prepared_docking_file,
        prepared_cofolding_file=prepared_cofolding_file,
    )

    print_prepared_data_summary(pose_df, annotation_file)

    # -------------------------------------------------------------------------
    # SuCOS pocket similarity
    # -------------------------------------------------------------------------
    if (not args.skip_sucos) and sucos_file.exists():
        sucos_df = read_table(sucos_file)
        plot_sucos_histogram(
            sucos_df,
            save_path=figures_dir / "SuCOS_pocket_AF3_similarity_cutoff.png",
        )
    elif args.skip_sucos:
        print("\nSkipping SuCOS histogram.")
    else:
        print(f"\nWarning: SuCOS file not found, skipping histogram: {sucos_file}")

    # -------------------------------------------------------------------------
    # Metrics for all data
    # -------------------------------------------------------------------------
    docking_summary_df, docking_complex_results = build_docking_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        scaffold_only=False,
        filtered=False,
    )

    cofold_summary_df, cofold_complex_results = build_cofold_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        scaffold_only=False,
        filtered=False,
    )

    docking_summary_df.to_csv(
        tables_dir / f"docking_summary_all_top{top_n}.csv",
        index=False,
    )

    cofold_summary_df.to_csv(
        tables_dir / f"cofolding_summary_all_top{top_n}.csv",
        index=False,
    )

    print_summary_table("Docking summary, all complexes:", docking_summary_df)
    print_summary_table("Cofolding summary, all complexes:", cofold_summary_df)

    # -------------------------------------------------------------------------
    # Docking figure: scaffold data only, filtered
    # -------------------------------------------------------------------------
    make_docking_figure_variant(
        pose_df,
        annotation_file,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        filtered=True,
        suffix=f"scaffolds_filtered_top{top_n}",
        top_n=top_n,
    )

    # -------------------------------------------------------------------------
    # All methods figure: cofolding + selected GNINA settings
    # -------------------------------------------------------------------------
    make_allmethods_figure_variant(
        pose_df,
        annotation_file,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        scaffold_only=True,
        filtered=True,
        suffix=f"scaffolds_filtered_top{top_n}",
        title_suffix="Scaffolds only, filtered",
        top_n=top_n,
    )

    # -------------------------------------------------------------------------
    # Affinity metrics
    # -------------------------------------------------------------------------
    _, _, affinity_df = plot_affinity_metrics(
        save_path=figures_dir / "affinity_rmse_spearman_bars.png"
    )

    affinity_df.to_csv(
        tables_dir / "affinity_metrics.csv",
        index=False,
    )

    ft_comparison_df, _, _ = make_ft_comparison_figure(
        pose_df,
        annotation_file,
        figures_dir=figures_dir,
        filtered=True,
        scaffold_only=True,
        suffix="of3p2_ft_vs_of3p2_vs_gnina_redock_scaffolds_filtered",
    )

    ft_comparison_df.to_csv(
        tables_dir / "of3p2_ft_vs_of3p2_vs_gnina_redock_scaffolds_filtered.csv",
        index=False,
    )

    print("\nDone.")
    print(f"Figures written to: {figures_dir}")
    print(f"Tables written to: {tables_dir}")


if __name__ == "__main__":
    main()
