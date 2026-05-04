# Benchmark plotting

This directory contains the plotting code and prepared input tables used to reproduce the benchmark figures.

## Directory structure

```text
analysis/plotting/
  README.md
  plot_benchmark_figures.py
  figure_data/
  figures/
  tables/
```

## Dependencies

- `numpy` – numerical operations  
- `pandas` – table manipulation and I/O (CSV, TSV, Parquet)  
- `matplotlib` – plotting  
- `pyarrow` (or `fastparquet`) – required for reading Parquet files  

## Inputs

The `figure_data/` directory contains the prepared tables used as plotting inputs.

Required files:

- `annotated_complexes.csv`: complex-level annotations, including fragment-screen, artefact, and PoseBusters-validity metadata.
- `final_docking_pose_data.parquet`: prepared pose-level benchmark table for docking methods.
- `final_cofolding_pose_data.parquet`: prepared pose-level benchmark table for co-folding methods.

Optional files:

- `tsv_similarity_data_2021-09-30_v2.tsv`: SuCOS similarity data used for the public-data similarity histogram.

The Parquet files are filtered and standardized versions of the original docking and co-folding results. They contain the same benchmark information in a format that is consistent across methods and directly usable for plotting.

## Outputs

Running the plotting script writes:

- figures to `figures/`
- summary tables to `tables/`

The tables in `tables/` are the source data for the plotted results and should be regenerated whenever the plotting inputs or benchmark settings change.

## Usage

From the repository root:

```bash
python analysis/plotting/plot_benchmark_figures.py
```

Optional arguments:

```bash
python analysis/plotting/plot_benchmark_figures.py \
  --figure-data-dir analysis/plotting/figure_data \
  --figures-dir analysis/plotting/figures \
  --tables-dir analysis/plotting/tables \
  --top-n 25
```

To skip the SuCOS/public-data similarity histogram:

```bash
python analysis/plotting/plot_benchmark_figures.py --skip-sucos
```

## Benchmark settings

The default plotting workflow uses:

- top 25 poses per method
- RMSD threshold: 2.0 Å
- LDDT-PLI threshold: 0.8
- filtered scaffold-only subsets for the main comparison figures

The RMSD and LDDT-PLI thresholds are currently encoded in the prepared input tables through the `rmsd_valid`, `lddt_pli_valid`, and `success_valid` columns. Changing these thresholds requires regenerating the prepared tables.

