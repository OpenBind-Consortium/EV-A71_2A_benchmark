# Manuscript plotting

This directory contains the plotting workflow used to reproduce the main figures and summary tables for the OpenBind enteroviral 2A benchmark.

## Run

From the repository root:

```bash
conda activate openbind-analysis
python plotting/plot_figures.py
```

The standard run uses the final curated benchmark inputs and writes PNG figures to:

```text
plotting/figures/
```

and CSV summary/source tables to:

```text
plotting/tables/
```

## Inputs

The plotting workflow combines processed results from the structural, affinity, virtual-screening, fragment-series, and public-data similarity analyses.

The principal inputs are:

```text
structure/processed_outputs/annotated_complexes.csv
structure/processed_outputs/final_docking_pose_data.parquet
structure/processed_outputs/final_cofolding_pose_data.parquet
structure/processed_outputs/fragment_followon_similarity.csv

similarity_metrics/tsv_similarity_data_2021-09-30_v2.tsv
similarity_metrics/tsv_similarity_data_2023-06-01_v2.tsv

virtual_screening/benchmark/virtual_screening_benchmark.csv
virtual_screening/results/

affinity/outputs/compound_level_prediction_analysis.csv
```

The structural benchmark uses 881 curated complexes, comprising 79 fragment-screen complexes and 802 follow-on complexes. The final affinity benchmark contains 490 compounds after structure-level quality filtering and compound-level aggregation.

## Plotting modules

Individual plotting and analysis functions are stored under:

```text
plotting/scripts/
```

The top-level `plot_figures.py` runner provides the manuscript analysis configuration and calls these modules using the final curated inputs.

Structural success is defined using ligand heavy-atom RMSD, LDDT-PLI, and PoseBusters validity. The default manuscript thresholds are RMSD ≤ 2 Å and LDDT-PLI ≥ 0.8.

Alternative input paths or structural thresholds can be inspected with:

```bash
python plotting/plot_figures.py --help
```

## Environment

The plotting dependencies are included in the repository-level environment:

```bash
conda env create -f environment.yml
conda activate openbind-analysis
```