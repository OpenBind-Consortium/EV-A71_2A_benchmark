#!/usr/bin/env python3
"""Plot the OpenBind virtual-screening ROC benchmark from distilled score files.

The input directory should contain:

    smina.csv
    gnina_cnnscore.csv
    gnina_cnn_vs.csv
    gnina_cnnaffinity.csv
    molecular_weight.csv
    negative_clogp.csv
    openfold3_p2.csv
    openfold3_p2_ft.csv

Each file must contain:

    Name,is_binder,score

All scores must already be oriented so that higher is better.

Each method is evaluated on all benchmark compounds available for that method.
Final manuscript analyses use the 2,084-compound virtual-screening benchmark
comprising 566 binders and 1,518 suspected non-binders.

Outputs:
    vs_roc.png
    vs_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


METHODS = {
    "smina": {
        "filename": "smina.csv",
        "label": "Smina (Vina)",
        "family": "Docking",
        "metric": "minimizedAffinity",
        "plot": True,
    },
    "gnina_cnnscore": {
        "filename": "gnina_cnnscore.csv",
        "label": "GNINA (CNNscore)",
        "family": "Docking",
        "metric": "CNNscore",
        "plot": False,
    },
    "gnina_cnn_vs": {
        "filename": "gnina_cnn_vs.csv",
        "label": "GNINA (CNN_VS)",
        "family": "Docking",
        "metric": "CNN_VS",
        "plot": True,
    },
    "gnina_cnnaffinity": {
        "filename": "gnina_cnnaffinity.csv",
        "label": "GNINA (CNNaffinity)",
        "family": "Docking",
        "metric": "CNNaffinity",
        "plot": False,
    },
    "openfold3_p2": {
        "filename": "openfold3_p2.csv",
        "label": "OpenFold3-p2 (Pair ipTM)",
        "family": "Co-folding",
        "metric": "pair_iptm",
        "plot": True,
    },
    "openfold3_p2_ft": {
        "filename": "openfold3_p2_ft.csv",
        "label": "OpenFold3-p2-FT (Pair ipTM)",
        "family": "Co-folding",
        "metric": "pair_iptm",
        "plot": True,
    },
    "molecular_weight": {
        "filename": "molecular_weight.csv",
        "label": "Molecular weight",
        "family": "Baseline",
        "metric": "molecular_weight",
        "plot": True,
    },
    "negative_clogp": {
        "filename": "negative_clogp.csv",
        "label": "-cLogP",
        "family": "Baseline",
        "metric": "negative_clogp",
        "plot": False,
    },
}

PLOT_COLORS = {
    "molecular_weight": "#8E44AD",
    "gnina": "#E63232",
    "smina": "#F07C12",
    "of3p2": "#1F64AD",
    "of3p2_ft": "#5DA5DA",
}


def read_score_file(path: Path, label: str) -> pd.DataFrame:
    """Read and validate one distilled score file."""
    df = pd.read_csv(path)

    required = {"Name", "is_binder", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[["Name", "is_binder", "score"]].copy()
    df["Name"] = df["Name"].astype("string").str.strip()
    df["is_binder"] = pd.to_numeric(df["is_binder"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["Name", "is_binder", "score"])

    if df["Name"].eq("").any():
        raise ValueError(f"{path} contains empty compound IDs.")

    if df["Name"].duplicated().any():
        duplicated = sorted(
            df.loc[df["Name"].duplicated(keep=False), "Name"].astype(str).unique()
        )
        raise ValueError(
            f"{path} contains duplicated compound IDs: " + ", ".join(duplicated[:10])
        )

    labels = df["is_binder"].astype(int)
    bad_labels = sorted(set(labels) - {0, 1})
    if bad_labels:
        raise ValueError(f"{path} contains non-binary labels: {bad_labels}")

    df["is_binder"] = labels

    n_binders = int(df["is_binder"].sum())
    n_nonbinders = int(len(df) - n_binders)

    print(
        f"{label}: {len(df):,} compounds "
        f"({n_binders:,} binders, {n_nonbinders:,} non-binders)"
    )

    return df.sort_values("Name", kind="stable").reset_index(drop=True)


def validate_labels_across_methods(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """Ensure compounds shared across methods have consistent labels."""
    seen: dict[str, int] = {}

    for key, df in datasets.items():
        for row in df[["Name", "is_binder"]].itertuples(index=False):
            label = int(row.is_binder)
            previous = seen.get(row.Name)

            if previous is None:
                seen[row.Name] = label
            elif previous != label:
                raise ValueError(
                    f"Conflicting binder label for {row.Name!r} in method {key!r}."
                )


def enrichment_factor(
    labels: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> float:
    """Calculate EF with fractional handling of score ties at the cutoff."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    n = len(labels)
    n_binders = int(labels.sum())

    if n == 0 or n_binders == 0:
        raise ValueError("Enrichment factor requires at least one binder.")

    top_n = max(1, int(np.ceil(fraction * n)))
    order = np.argsort(-scores, kind="stable")
    cutoff_score = scores[order[top_n - 1]]

    above = scores > cutoff_score
    tied = scores == cutoff_score

    n_above = int(above.sum())
    slots_at_cutoff = top_n - n_above

    expected_binders = float(labels[above].sum())
    if slots_at_cutoff > 0:
        expected_binders += slots_at_cutoff * float(labels[tied].mean())

    observed_fraction = expected_binders / top_n
    background_fraction = n_binders / n
    return float(observed_fraction / background_fraction)


def calculate_metrics(
    df: pd.DataFrame,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Calculate ROC-AUC, AP, EF values, and ROC coordinates."""
    labels = df["is_binder"].to_numpy(dtype=int)
    scores = df["score"].to_numpy(dtype=float)

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both binder and non-binder classes.")

    fpr, tpr, _ = roc_curve(labels, scores)

    metrics = {
        "n_compounds": int(len(df)),
        "n_binders": int(labels.sum()),
        "n_nonbinders": int(len(labels) - labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "ef_1pct": enrichment_factor(labels, scores, 0.01),
        "ef_2pct": enrichment_factor(labels, scores, 0.02),
        "ef_5pct": enrichment_factor(labels, scores, 0.05),
    }

    return metrics, fpr, tpr


def make_roc_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Draw the final ROC figure."""
    fig, ax = plt.subplots(figsize=(8.5, 7.0), tight_layout=True)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.7,
        color="black",
        alpha=0.7,
        label="Random",
    )

    plot_specs = [
        ("molecular_weight", PLOT_COLORS["molecular_weight"]),
        ("smina", PLOT_COLORS["smina"]),
        ("gnina_cnn_vs", PLOT_COLORS["gnina"]),
        ("openfold3_p2", PLOT_COLORS["of3p2"]),
        ("openfold3_p2_ft", PLOT_COLORS["of3p2_ft"]),
    ]

    for key, color in plot_specs:
        spec = METHODS[key]
        fpr, tpr, auc_value = curves[key]

        ax.plot(
            fpr,
            tpr,
            linewidth=3.0,
            color=color,
            label=f"{spec['label']} (AUC = {auc_value:.3f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate", fontsize=18)
    ax.set_ylabel("True positive rate", fontsize=18)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=15,
    )
    ax.legend(
        loc="lower right",
        fontsize=12.5,
        frameon=False,
    )
    ax.grid(False)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_virtual_screening_performance(
    scores_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Generate the VS ROC figure and metrics table."""
    print("Virtual-screening ROC benchmark")

    datasets: dict[str, pd.DataFrame] = {}

    for key, spec in METHODS.items():
        path = scores_dir / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing required score file: {path}")

        datasets[key] = read_score_file(
            path,
            spec["label"],
        )

    validate_labels_across_methods(datasets)

    sizes = {key: len(df) for key, df in datasets.items()}
    if len(set(sizes.values())) > 1:
        print(
            "\nWARNING: methods are evaluated on their available compound "
            "sets; denominators differ."
        )

    rows: list[dict[str, object]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    print("\nPerformance")
    for key, spec in METHODS.items():
        evaluated = datasets[key]

        metrics, fpr, tpr = calculate_metrics(evaluated)

        rows.append(
            {
                "family": spec["family"],
                "method": spec["label"],
                "metric": spec["metric"],
                **metrics,
            }
        )

        curves[key] = (
            fpr,
            tpr,
            float(metrics["roc_auc"]),
        )

        print(
            f"  {spec['label']}: "
            f"ROC-AUC={metrics['roc_auc']:.3f}, "
            f"AP={metrics['average_precision']:.3f}, "
            f"EF1%={metrics['ef_1pct']:.2f}, "
            f"EF2%={metrics['ef_2pct']:.2f}, "
            f"EF5%={metrics['ef_5pct']:.2f}, "
            f"n={metrics['n_compounds']:,}"
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figures_dir / "vs_roc.png"
    table_path = tables_dir / "vs_metrics.csv"

    make_roc_plot(
        curves,
        figure_path,
    )

    pd.DataFrame(rows).to_csv(
        table_path,
        index=False,
        float_format="%.6f",
    )

    print(f"\nFigure: {figure_path}")
    print(f"Table:  {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir",
        required=True,
        type=Path,
        help="Directory containing the eight distilled VS score CSV files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("plotting/figures"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("plotting/tables"),
    )
    args = parser.parse_args()

    generate_virtual_screening_performance(
        scores_dir=args.scores_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )


if __name__ == "__main__":
    main()
