# Structure benchmark data

This directory contains processed structure-level benchmark data for the EV-A71 2A dataset.

## Directory structure

```text
structure/
  processed_outputs/
    annotated_complexes.csv
    final_docking_pose_data.parquet
    final_cofolding_pose_data.parquet
    tsv_similarity_data_2021-09-30_v2.tsv
```

## Processed outputs

The `processed_outputs/` directory contains the inputs used by the plotting workflow.

### `annotated_complexes.csv`

Complex-level annotations for the dataset.

Includes:

- complex identifiers  
- fragment screen labels  
- artefact flags  
- PoseBusters validity  
- filtering indicators used in benchmark subsets  

This file defines which complexes are included in different evaluation settings.

---

### `final_docking_pose_data.parquet`

Prepared docking benchmark data.

Contains:

- pose-level results for docking methods  
- ranks and scores  
- RMSD and lDDT-PLI metrics  
- validity flags used for evaluation  

---

### `final_cofolding_pose_data.parquet`

Prepared co-folding benchmark data.

Structured in the same format as the docking table, allowing direct comparison between docking and co-folding methods.

---

## Relationship to other directories

- `plotting/` consumes these processed tables to generate figures  
- `similarity_metrics/` contains the code used to compute similarity features  
- `affinity/` contains separate affinity benchmark data  

