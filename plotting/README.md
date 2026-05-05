# Benchmark plotting

This directory contains the plotting code used to generate the EV-A71 2A benchmark figures.

The plotting script reads processed structure data and produces figures and summary tables.

## Directory structure

```text
plotting/
  plot_figures.py
  requirements.txt
  figures/
  tables/
```

Input data is stored outside this directory:

```text
structure/
  processed_outputs/
```

## Dependencies

Install required packages with:

```bash
pip install -r plotting/requirements.txt
```

Main dependencies:

- numpy  
- pandas  
- matplotlib  
- pyarrow or fastparquet  

## Inputs

By default, the script expects the following files in:

```text
structure/processed_outputs/
```

- `annotated_complexes.csv`  
- `final_docking_pose_data.parquet`  
- `final_cofolding_pose_data.parquet`  

The SuCOS/public-data similarity file is read from:

```text
similarity_metrics/tsv_similarity_data_2021-09-30_v2.tsv
```

This can be overridden using `--sucos-file`.

## Outputs

Generated files are written to:

```text
plotting/figures/
plotting/tables/
```

- `figures/`: PNG plots  
- `tables/`: CSV summary tables used to generate figures  

## Usage

Run from the repository root:

```bash
python plotting/plot_figures.py
```

Optional arguments:

```bash
python plotting/plot_figures.py \
  --processed-data-dir structure/processed_outputs \
  --figures-dir plotting/figures \
  --tables-dir plotting/tables \
  --top-n 25
```

To skip the similarity plot:

```bash
python plotting/plot_figures.py --skip-sucos
```

## Benchmark settings

The default plotting workflow uses:

- top 25 poses per method
- RMSD threshold: 2.0 Å
- LDDT-PLI threshold: 0.8
- filtered scaffold-only subsets for the main comparison figures

The pocket-aligned RMSD and LDDT-PLI metrics were computed against the ground-truth structures using the [OST compare-ligand-structures](https://openstructure.org/docs/2.11/actions/#comparing-two-structures-with-ligands) command. The thresholds applied to these metrics are encoded in the prepared input tables as the `rmsd_valid`, `lddt_pli_valid`, and `success_valid columns`.