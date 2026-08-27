#!/usr/bin/env python3
"""Structural docking and cofolding benchmark plotting utilities.

This module contains the reusable calculations and plotting functions used by
``plot_figures.py``. It is intentionally importable so individual analyses can
still be run interactively while keeping the manuscript runner fixed to the published analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

pd.set_option("display.float_format", "{:.3f}".format)

RMSD_THRESHOLD = 2.0
LDDT_PLI_THRESHOLD = 0.8
STANDARD_TOP_NS = (1, 25)

PLOT_COLORS = {
    "gnina": "#E63232",
    "smina": "#F07C12",
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

CLASSICAL_METHODS = ["gnina", "smina", "diffdock"]

METHOD_LABELS = {
    "gnina": "GNINA",
    "smina": "Smina",
    "diffdock": "DiffDock",
    "af3": "AlphaFold3",
    "boltz-1": "Boltz-1",
    "boltz-2": "Boltz-2",
    "of3p2": "OpenFold3-p2",
    "of3p2_ft": "OpenFold3-p2\nfine-tuned",
    "protenix": "Protenix",
    "rf3": "RosettaFold 3",
    "cofold_best": "Best\ncofolding",
}

METHOD_ORDER = [
    "gnina",
    "smina",
    "diffdock",
    "af3",
    "boltz-1",
    "boltz-2",
    "of3p2",
    "protenix",
    "rf3",
]
COFOLD_METHOD_ORDER = ["af3", "boltz-1", "boltz-2", "of3p2", "protenix", "rf3"]
DOCK_PROT_ORDER = ["redock", "A71EV2A_AF", "A71EV2A_8POA", "fragment_crossdock"]
DOCK_PROT_LABELS = {
    "redock": "Redocking",
    "fragment_crossdock": "Cross-docking\nfragment screen",
    "A71EV2A_AF": "Cross-docking\napo (AF2)",
    "A71EV2A_8POA": "Cross-docking\napo (8POA)",
}


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


def add_threshold_validity_columns(
    pose_df: pd.DataFrame,
    rmsd_threshold: float = RMSD_THRESHOLD,
    lddt_pli_threshold: float = LDDT_PLI_THRESHOLD,
) -> pd.DataFrame:
    """Compute threshold-dependent benchmark validity columns from raw metrics.

    The prepared pose table should contain raw `lig_rmsd`, `lddt_pli`, and
    `pb_valid` values. This function derives `rmsd_valid`, `lddt_pli_valid`,
    and `success_valid` at plotting time so the same prepared data can be
    reused with alternative cutoffs.
    """
    df = pose_df.copy()

    required = ["pb_valid", "lig_rmsd", "lddt_pli"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Pose data is missing required raw metric columns: {missing}")

    df["pb_valid"] = to_bool(df["pb_valid"])
    df["lig_rmsd"] = pd.to_numeric(df["lig_rmsd"], errors="coerce")
    df["lddt_pli"] = pd.to_numeric(df["lddt_pli"], errors="coerce")

    df["rmsd_valid"] = (df["lig_rmsd"] <= rmsd_threshold) & df["pb_valid"]
    df["lddt_pli_valid"] = (df["lddt_pli"] >= lddt_pli_threshold) & df["pb_valid"]
    df["success_valid"] = (
        (df["lig_rmsd"] <= rmsd_threshold)
        & (df["lddt_pli"] >= lddt_pli_threshold)
        & df["pb_valid"]
    )

    return df


def _format_threshold_value(value: float, scale: int | None = None) -> str:
    if scale is not None:
        return f"{int(round(value * scale)):03d}"

    text = f"{value:g}"
    return text.replace(".", "p")


def format_metric_suffix(rmsd_threshold: float, lddt_pli_threshold: float) -> str:
    """Return a filename-safe suffix describing benchmark thresholds."""
    rmsd_text = _format_threshold_value(rmsd_threshold)
    lddt_text = _format_threshold_value(lddt_pli_threshold, scale=100)
    return f"RMSD_{rmsd_text}A_lddt-pli_{lddt_text}"


def read_annotations(annotation_file: Path) -> pd.DataFrame:
    """Read complex annotations and standardise filtering columns."""
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


def read_prepared_pose_data(
    prepared_docking_file: Path,
    prepared_cofolding_file: Path,
) -> pd.DataFrame:
    """Read, combine, and validate prepared docking and cofolding pose tables."""
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
      rmsd_valid is true if any selected pose satisfies the configured RMSD
      cutoff and is PB-valid. success_valid additionally requires the configured
      LDDT-PLI cutoff.

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
            complex_df["rmsd_valid"].astype("boolean").fillna(False).astype(bool)
        )

        complex_df["success_valid"] = (
            complex_df["success_valid"].astype("boolean").fillna(False).astype(bool)
        )

        complex_df["n_poses"] = complex_df["n_poses"].fillna(0).astype(int)
        complex_df["n_seeds"] = complex_df["n_seeds"].fillna(0).astype(int)

    summary = {
        "label": label,
        "method": method,
        "dock_prot": dock_prot,
        "filtered": filtered,
        "scaffold_only": scaffold_only,
        "rmsd_threshold": RMSD_THRESHOLD,
        "lddt_pli_threshold": LDDT_PLI_THRESHOLD,
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
    """Build cofolding summary tables and complex-level result tables."""
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


def draw_allmethods_grouped_bars(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    top_n: int,
    *,
    show_legend: bool = True,
    show_title: bool = False,
    panel_label: str | None = None,
    compact: bool = False,
) -> None:
    """Draw the cofolding and selected GNINA comparison on an existing axis.

    The same drawing function is used for standalone figures and the manuscript
    composite, which keeps the visual encoding and layout consistent.
    """
    df = add_percent_columns(plot_df).reset_index(drop=True)
    x = np.arange(len(df))

    if compact:
        value_fontsize = 8.0
        tick_fontsize = 8.0
        group_fontsize = 9.0
        legend_fontsize = 8.0
        title_fontsize = 12.0
        ylabel_fontsize = 11.0
        ylabel_tick_fontsize = 9.0
        group_y = -0.17
        bar_width = 0.76
    else:
        value_fontsize = 11
        tick_fontsize = 12
        group_fontsize = 14
        legend_fontsize = 12
        title_fontsize = 16
        ylabel_fontsize = 16
        ylabel_tick_fontsize = 14
        group_y = -0.14
        bar_width = 0.80

    for i, row in df.iterrows():
        method = str(row["method"])
        color = (
            PLOT_COLORS["gnina"]
            if row["plot_group"] == "GNINA"
            else PLOT_COLORS.get(method, f"C{i}")
        )

        success = float(row["success_valid_pct"])
        rmsd_only = float(row["rmsd_only_pct"])
        rmsd_total = float(row["rmsd_valid_pct"])

        ax.bar(
            i,
            success,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )

        ax.bar(
            i,
            rmsd_only,
            width=bar_width,
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
                fontsize=value_fontsize,
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
                fontsize=value_fontsize,
                color="black",
            )

    display_labels = df["plot_label"].copy()
    if compact:
        display_labels = display_labels.replace(
            {"Cross-docking\nFragment screen": "Fragment\ncross-docking"}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=tick_fontsize)

    n_cofold = int((df["plot_group"] == "Cofolding").sum())

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
            group_y,
            "Cofolding",
            ha="center",
            va="top",
            fontsize=group_fontsize,
            fontweight="bold",
            transform=trans,
            clip_on=False,
        )

    n_gnina = int((df["plot_group"] == "GNINA").sum())
    if n_gnina > 0:
        gnina_center = (n_cofold + len(df) - 1) / 2
        ax.text(
            gnina_center,
            group_y,
            "Docking\n(GNINA)",
            ha="center",
            va="top",
            fontsize=group_fontsize,
            fontweight="bold",
            transform=trans,
            clip_on=False,
        )

    if show_legend:
        legend_handles = [
            Patch(
                facecolor="gray",
                edgecolor="black",
                label=(
                    f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid & "
                    f"LDDT-PLI ≥ {LDDT_PLI_THRESHOLD:g}"
                ),
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label=f"RMSD ≤ {RMSD_THRESHOLD:g} Å & PB-valid",
            ),
        ]

        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper left",
            fontsize=legend_fontsize,
        )

    if show_title:
        panel_title = "Top 1 pose" if top_n == 1 else f"Top {top_n} poses"

        ax.set_title(
            panel_title,
            loc="left",
            fontsize=title_fontsize,
            fontweight="bold",
            pad=8 if compact else 10,
        )

    ax.set_ylabel(
        "Success rate (%)",
        fontsize=ylabel_fontsize,
        fontweight="bold",
    )
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=ylabel_tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_allmethods_grouped_bars(
    plot_df: pd.DataFrame,
    top_n: int,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Create a standalone all-methods comparison figure.

    Standalone Top-1 and Top-25 figures deliberately omit a panel title; the
    Top-N information is encoded in the filename and source table. Panel titles
    are added only in the combined manuscript figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    draw_allmethods_grouped_bars(
        ax,
        plot_df,
        top_n=top_n,
        show_legend=True,
        show_title=False,
        compact=False,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def add_panel_label(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
    *,
    x_offset: float = 0.060,
    y_offset: float = 0.010,
    fontsize: float = 14,
) -> None:
    """Place a panel label just above and to the left of an axis."""
    bbox = ax.get_position()

    fig.text(
        bbox.x0 - x_offset,
        bbox.y1 + y_offset,
        label,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def plot_allmethods_composite(
    top25_df: pd.DataFrame,
    top1_df: pd.DataFrame,
    save_path: Path,
    figsize: tuple[float, float] = (7.4, 6.6),
) -> tuple[plt.Figure, np.ndarray]:
    """Create the manuscript composite with Top-25 and Top-1 panels."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        dpi=600,
        sharey=True,
    )

    draw_allmethods_grouped_bars(
        axes[0],
        top25_df,
        top_n=25,
        show_legend=True,
        show_title=True,
        compact=True,
    )

    draw_allmethods_grouped_bars(
        axes[1],
        top1_df,
        top_n=1,
        show_legend=False,
        show_title=True,
        compact=True,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.965,
        bottom=0.09,
        hspace=0.34,
    )

    add_panel_label(
        fig,
        axes[0],
        "A",
    )

    add_panel_label(
        fig,
        axes[1],
        "B",
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    return fig, axes


def make_docking_figure_variant(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    tables_dir: Path,
    figures_dir: Path,
    scaffold_only: bool = True,
    filtered: bool = True,
    suffix: str = "scaffolds_filtered",
    top_n: int = 25,
    output_stem: str | None = None,
    save_figure: bool = True,
) -> pd.DataFrame:
    """Create docking source data and optionally write the corresponding figure."""

    scaffold_docking_summary_df, _ = build_docking_summaries(
        pose_df,
        annotation_file,
        top_n=top_n,
        filtered=filtered,
        scaffold_only=scaffold_only,
    )

    scaffold_docking_summary_df = scaffold_docking_summary_df[
        scaffold_docking_summary_df["method"].isin(CLASSICAL_METHODS)
    ].copy()

    table_stem = output_stem or f"figure_docking_{suffix}_summary"
    scaffold_docking_summary_df.to_csv(
        tables_dir / f"{table_stem}.csv",
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

    if save_figure:
        figure_stem = output_stem or f"docking_{suffix}_stacked"
        fig, _ = plot_scaffold_dockprot_bars(
            scaffold_docking_summary_df,
            save_path=figures_dir / f"{figure_stem}.png",
            figsize=(12, 5),
        )
        plt.close(fig)

    return scaffold_docking_summary_df


def make_allmethods_figure_variant(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    tables_dir: Path,
    figures_dir: Path,
    scaffold_only: bool = True,
    filtered: bool = True,
    suffix: str = "scaffolds_filtered",
    top_n: int = 25,
    output_stem: str | None = None,
    save_figure: bool = True,
) -> pd.DataFrame:
    """Create cofolding/docking comparison data and optionally its standalone figure."""
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
        label="Best\nCofolding",
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
    cofold_plot_df["plot_group"] = "Cofolding"
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
        (docking_summary_df["method"] == "gnina")
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

    table_stem = output_stem or f"allmethods_plot_data_{suffix}"
    allmethod_df.to_csv(
        tables_dir / f"{table_stem}.csv",
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

    if save_figure:
        figure_stem = output_stem or f"allmethods_{suffix}_stacked"
        fig, _ = plot_allmethods_grouped_bars(
            allmethod_df,
            top_n=top_n,
            save_path=figures_dir / f"{figure_stem}.png",
            figsize=(max(12, 0.85 * len(allmethod_df)), 5),
        )
        plt.close(fig)

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
            "max_top_n": 25,
        },
        {
            "source": "cofolding",
            "method": "of3p2_ft",
            "dock_prot": "cofold",
            "plot_label": METHOD_LABELS["of3p2_ft"],
            "max_top_n": 25,
        },
    ]

    for item in comparisons:
        for top_n in [1, item["max_top_n"]]:
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
            summary["max_top_n"] = item["max_top_n"]
            summary["plot_label"] = item["plot_label"]
            rows.append(summary)

    plot_df = add_percent_columns(pd.DataFrame(rows))

    comparison_order = {
        ("of3p2", "cofold"): 0,
        ("of3p2_ft", "cofold"): 1,
    }
    plot_df["comparison_order"] = [
        comparison_order[(method, dock_prot)]
        for method, dock_prot in zip(plot_df["method"], plot_df["dock_prot"])
    ]
    plot_df = plot_df.sort_values(["comparison_order", "top_n"]).reset_index(drop=True)

    print(f"\nOF3p2 fine-tuning comparison: {suffix}")
    print(
        plot_df[
            [
                "top_n",
                "max_top_n",
                "method",
                "dock_prot",
                "n_total",
                "n_with_predictions",
                "n_success_valid",
                "success_valid_pct",
            ]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(4.8, 4), dpi=300)

    comparison_keys = [
        ("of3p2", "cofold"),
        ("of3p2_ft", "cofold"),
    ]
    labels = [item["plot_label"] for item in comparisons]
    max_top_ns = [item["max_top_n"] for item in comparisons]

    x = np.arange(len(comparison_keys))
    bar_width = 0.6
    hatch_pattern = "xx"  # or ".." / "||" if you want something cleaner

    for i, ((method, dock_prot), max_top_n) in enumerate(
        zip(comparison_keys, max_top_ns)
    ):
        comparison_df = plot_df[
            (plot_df["method"] == method) & (plot_df["dock_prot"] == dock_prot)
        ].set_index("top_n")

        success_top1 = float(comparison_df["success_valid_pct"].get(1, 0.0))
        success_topn = float(comparison_df["success_valid_pct"].get(max_top_n, 0.0))
        remainder = max(success_topn - success_top1, 0)

        color = PLOT_COLORS.get(method, f"C{i}")

        ax.bar(
            x[i],
            success_top1,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )

        ax.bar(
            x[i],
            remainder,
            width=bar_width,
            bottom=success_top1,
            color="white",
            edgecolor=color,
            hatch=hatch_pattern,
            linewidth=1.2,
            zorder=3,
        )

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

        if success_topn > 0:
            ax.text(
                x[i],
                success_topn + 1,
                f"{success_topn:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

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
            hatch=hatch_pattern,
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
