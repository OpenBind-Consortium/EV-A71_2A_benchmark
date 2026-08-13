#!/usr/bin/env python3
"""Reproduce the main manuscript figures for the OpenBind enteroviral 2A benchmark.

Run from the repository root with::

    python plotting/plot_figures.py

The runner uses the final curated structural, affinity, virtual-screening,
fragment-series, and public-data similarity benchmark inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTTING_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PLOTTING_DIR / "scripts"
REPO_ROOT = PLOTTING_DIR.parent

# Keep the plotting helpers as standalone scripts while allowing this runner to
# import them without requiring plotting/scripts to be an installed package.
sys.path.insert(0, str(SCRIPTS_DIR))

import plot_fragment_series as fragment_series  # noqa: E402
import plot_structure_benchmarks as structure  # noqa: E402
import plot_virtual_screening_characterisation as vs_characterisation  # noqa: E402
import plot_virtual_screening_performance as vs_performance  # noqa: E402
from plot_affinity_benchmarks import (  # noqa: E402
    plot_affinity_composite,
    plot_affinity_mode_comparison,
    read_affinity_predictions,
    validate_affinity_dataset,
)
from plot_public_data_similarity import (  # noqa: E402
    build_sucos_summary_table,
    plot_sucos_histogram,
    prepare_sucos_data,
)

DEFAULT_PROCESSED_DATA_DIR = REPO_ROOT / "structure" / "processed_outputs"
DEFAULT_FIGURES_DIR = PLOTTING_DIR / "figures"
DEFAULT_TABLES_DIR = PLOTTING_DIR / "tables"
DEFAULT_FRAGMENT_SIMILARITY = (
    DEFAULT_PROCESSED_DATA_DIR / "fragment_followon_similarity.csv"
)
DEFAULT_AFFINITY_FILE = (
    REPO_ROOT / "affinity" / "outputs" / "compound_level_prediction_analysis.csv"
)
DEFAULT_SUCOS_FILES = (
    REPO_ROOT / "similarity_metrics" / "tsv_similarity_data_2021-09-30_v2.tsv",
    REPO_ROOT / "similarity_metrics" / "tsv_similarity_data_2023-06-01_v2.tsv",
)
DEFAULT_VS_BENCHMARK_FILE = (
    REPO_ROOT / "virtual_screening" / "benchmark" / "virtual_screening_benchmark.csv"
)
DEFAULT_VS_SCORES_DIR = REPO_ROOT / "virtual_screening" / "results"

# Locked manuscript denominators. The figure runner is intentionally specific
# to this benchmark and should fail if the underlying curated dataset changes.
EXPECTED_CURATED_COMPLEXES = 881
EXPECTED_CURATED_FRAGMENTS = 79
EXPECTED_CURATED_FOLLOWONS = 802
EXPECTED_AFFINITY_COMPOUNDS = 490


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the main OpenBind enteroviral 2A manuscript figures."
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help="Directory containing annotated complexes and prepared pose tables.",
    )
    parser.add_argument(
        "--fragment-similarity",
        type=Path,
        default=DEFAULT_FRAGMENT_SIMILARITY,
        help="Fragment-to-follow-on similarity table produced during input preparation.",
    )
    parser.add_argument(
        "--affinity-file",
        type=Path,
        default=DEFAULT_AFFINITY_FILE,
        help="Final compound-level affinity prediction table.",
    )
    parser.add_argument(
        "--sucos-files",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SUCOS_FILES),
        help="Public-data similarity TSV files to plot.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for manuscript figures.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Output directory for source and summary tables.",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=2.0,
        help="Ligand heavy-atom RMSD success threshold in Angstrom.",
    )
    parser.add_argument(
        "--lddt-pli-threshold",
        type=float,
        default=0.8,
        help="LDDT-PLI success threshold.",
    )
    parser.add_argument(
        "--vs-benchmark-file",
        type=Path,
        default=DEFAULT_VS_BENCHMARK_FILE,
        help="Final virtual-screening benchmark CSV.",
    )
    parser.add_argument(
        "--vs-scores-dir",
        type=Path,
        default=DEFAULT_VS_SCORES_DIR,
        help="Directory containing processed virtual-screening score CSV files.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def threshold_tag(rmsd: float, lddt_pli: float) -> str:
    """Tag only non-standard structural thresholds in output filenames."""
    if abs(rmsd - 2.0) < 1e-12 and abs(lddt_pli - 0.8) < 1e-12:
        return ""
    rmsd_text = f"{rmsd:g}".replace(".", "p")
    lddt_text = f"{lddt_pli:g}".replace(".", "p")
    return f"_rmsd{rmsd_text}_lddtpli{lddt_text}"


def sucos_date_label(path: Path) -> str:
    stem = path.stem
    prefix = "tsv_similarity_data_"
    suffix = "_v2"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def validate_structural_denominators(annotation_file: Path) -> None:
    annotations = structure.read_annotations(annotation_file)
    curated = annotations.loc[~annotations["filtered"]].copy()
    fragments = curated.loc[curated["fragment_screen"]]
    followons = curated.loc[~curated["fragment_screen"]]

    observed = {
        "curated complexes": len(curated),
        "curated fragments": len(fragments),
        "curated follow-on complexes": len(followons),
    }
    expected = {
        "curated complexes": EXPECTED_CURATED_COMPLEXES,
        "curated fragments": EXPECTED_CURATED_FRAGMENTS,
        "curated follow-on complexes": EXPECTED_CURATED_FOLLOWONS,
    }

    mismatches = [
        f"{label}: expected {expected[label]}, found {observed[label]}"
        for label in expected
        if observed[label] != expected[label]
    ]
    if mismatches:
        raise ValueError(
            "Structural manuscript denominators do not match the locked benchmark:\n  "
            + "\n  ".join(mismatches)
        )


def generate_public_similarity_outputs(
    annotation_file: Path,
    sucos_files: list[Path],
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    for sucos_arg in sucos_files:
        sucos_path = require_file(sucos_arg, "Public-data similarity table")
        sucos_df = prepare_sucos_data(
            structure.read_table(sucos_path),
            annotation_file,
        )
        label = sucos_date_label(sucos_path)
        stem = f"public_data_similarity_{label}"

        fig, _ = plot_sucos_histogram(
            sucos_df,
            subset="all",
            filtered=True,
            save_path=figures_dir / f"{stem}.png",
        )
        plt.close(fig)

        summary = build_sucos_summary_table(
            sucos_df,
            column="sucos_shape_pocket_qcov",
            filtered=True,
        )
        summary.to_csv(tables_dir / f"{stem}.csv", index=False)


def generate_structure_outputs(
    pose_df: pd.DataFrame,
    annotation_file: Path,
    figures_dir: Path,
    tables_dir: Path,
    tag: str,
) -> None:
    allmethods_by_topn: dict[int, pd.DataFrame] = {}
    for top_n in structure.STANDARD_TOP_NS:
        output_stem = f"cofolding_performance_top{top_n}{tag}"
        allmethods_by_topn[top_n] = structure.make_allmethods_figure_variant(
            pose_df,
            annotation_file,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            scaffold_only=True,
            filtered=True,
            suffix=output_stem,
            top_n=top_n,
            output_stem=output_stem,
            save_figure=False,
        )

    fig, _ = structure.plot_allmethods_composite(
        top25_df=allmethods_by_topn[25],
        top1_df=allmethods_by_topn[1],
        save_path=figures_dir / f"cofolding_performance_top1_top25{tag}.png",
    )
    plt.close(fig)

    structure.make_docking_figure_variant(
        pose_df,
        annotation_file,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        scaffold_only=True,
        filtered=True,
        suffix=f"top25{tag}",
        top_n=25,
        output_stem=f"docking_performance_top25{tag}",
        save_figure=True,
    )

    ft_df, ft_fig, _ = structure.make_ft_comparison_figure(
        pose_df,
        annotation_file,
        figures_dir=figures_dir,
        filtered=True,
        scaffold_only=True,
        suffix=f"of3_finetuning_comparison{tag}",
    )
    plt.close(ft_fig)
    ft_df.to_csv(tables_dir / f"of3_finetuning_comparison{tag}.csv", index=False)


def generate_affinity_outputs(
    affinity_file: Path,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    affinity_df = read_affinity_predictions(affinity_file)
    validate_affinity_dataset(
        affinity_df,
        expected_compounds=EXPECTED_AFFINITY_COMPOUNDS,
    )

    fig, _, metrics = plot_affinity_composite(
        affinity_df,
        save_path=figures_dir / "affinity_prediction_performance.png",
    )
    plt.close(fig)
    metrics.to_csv(tables_dir / "affinity_prediction_metrics.csv", index=False)

    fig, _, mode_metrics = plot_affinity_mode_comparison(
        affinity_df,
        save_path=figures_dir / "affinity_docking_mode_comparison.png",
    )
    plt.close(fig)
    mode_metrics.to_csv(
        tables_dir / "affinity_docking_mode_metrics.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()

    processed_data_dir = args.processed_data_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    tables_dir = args.tables_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = require_file(
        processed_data_dir / "annotated_complexes.csv",
        "Annotation table",
    )
    docking_file = require_file(
        processed_data_dir / "final_docking_pose_data.parquet",
        "Prepared docking table",
    )
    cofolding_file = require_file(
        processed_data_dir / "final_cofolding_pose_data.parquet",
        "Prepared cofolding table",
    )
    fragment_similarity = require_file(
        args.fragment_similarity,
        "Fragment-follow-on similarity table",
    )
    affinity_file = require_file(args.affinity_file, "Affinity prediction table")
    vs_benchmark_file = require_file(
        args.vs_benchmark_file,
        "Virtual-screening benchmark",
    )
    vs_scores_dir = args.vs_scores_dir.resolve()
    if not vs_scores_dir.is_dir():
        raise FileNotFoundError(
            f"Virtual-screening scores directory not found: {vs_scores_dir}"
        )

    validate_structural_denominators(annotation_file)

    structure.RMSD_THRESHOLD = args.rmsd_threshold
    structure.LDDT_PLI_THRESHOLD = args.lddt_pli_threshold
    tag = threshold_tag(args.rmsd_threshold, args.lddt_pli_threshold)

    pose_df = structure.read_prepared_pose_data(docking_file, cofolding_file)
    pose_df = structure.add_threshold_validity_columns(
        pose_df,
        rmsd_threshold=args.rmsd_threshold,
        lddt_pli_threshold=args.lddt_pli_threshold,
    )

    print(
        f"Generating manuscript figures "
        f"(RMSD <= {args.rmsd_threshold:g} A, LDDT-PLI >= {args.lddt_pli_threshold:g})..."
    )

    generate_public_similarity_outputs(
        annotation_file=annotation_file,
        sucos_files=args.sucos_files,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    fragment_series.generate_fragment_series_outputs(
        annotated_path=annotation_file,
        similarity_path=fragment_similarity,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    vs_characterisation.generate_virtual_screening_characterisation(
        benchmark_path=vs_benchmark_file,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    vs_performance.generate_virtual_screening_performance(
        scores_dir=vs_scores_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    generate_structure_outputs(
        pose_df=pose_df,
        annotation_file=annotation_file,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        tag=tag,
    )
    generate_affinity_outputs(
        affinity_file=affinity_file,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )

    print("\nGenerated manuscript figures:")
    for path in sorted(figures_dir.glob("*.png")):
        print(f"  {path.name}")
    print(f"\nFigures: {figures_dir}")
    print(f"Tables:  {tables_dir}")


if __name__ == "__main__":
    main()
