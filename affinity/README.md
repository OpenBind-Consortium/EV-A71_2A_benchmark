# Affinity benchmark

This directory contains the experimental affinity measurements and processed prediction results used for the OpenBind enteroviral 2A affinity benchmark.

## Contents

```text
affinity/
├── all_affinity_data_release_v1.csv
├── reference/
├── predictions/
├── scripts/
└── outputs/
```

`all_affinity_data_release_v1.csv` contains the row-level experimental affinity measurements.

`reference/` contains the processed experimental reference information used to connect affinity measurements to the structural dataset.

`predictions/` contains structure-level affinity predictions. GNINA and Smina predictions from the evaluated docking protocols are stored together in `docking_affinity_predictions.csv`; other prediction methods and descriptor baselines are stored separately.

## Compound-level benchmark

Affinity measurements and predictions are cross-referenced against:

```text
structure/processed_outputs/annotated_complexes.csv
```

Binding events annotated as PoseBusters-invalid or suspected crystallographic artefacts are excluded before compound-level aggregation.

For compounds with multiple retained crystallographic binding events, structure-level predictions are averaged to obtain one prediction per compound.

The final curated affinity benchmark contains 490 compounds.

Regenerate the compound-level analysis table from the repository root with:

```bash
python affinity/scripts/build_compound_level_affinity_predictions.py \
    affinity/predictions \
    --reference affinity/reference/fragalysis_compound_reference.csv \
    --annotated-complexes structure/processed_outputs/annotated_complexes.csv \
    --output affinity/outputs/compound_level_prediction_analysis.csv
```

The resulting table:

```text
affinity/outputs/compound_level_prediction_analysis.csv
```

is used directly by the manuscript plotting workflow.