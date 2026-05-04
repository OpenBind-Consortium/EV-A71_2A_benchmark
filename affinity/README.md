# Affinity Benchmark Data

## Structure

```text
affinity/
  all_affinity_data_release_v1.csv
  README.md
  reference/
  predictions/
```

## Experimental Data

`all_affinity_data_release_v1.csv` contains row-level affinity measurements.

`Used in analysis` denotes the entries that were included in the analysis based on several filters. Reference pKD values were calculated from included rows only.

## Reference

`reference/experimental_reference_ground_truth.csv` has one row per Fragalysis structure:

Blank `experimental_pKD` means no affinity value was available for that structure after filtering.

## Predictions

Each file in `predictions/` has:

```text
method
fragalysis_code
predicted_affinity
```

`molecular_weight` and `clogp` are raw property values.

## Score Conversions

In cases where the raw scores were not in pK units, we converted them as follows:

```text
smina:      predicted_affinity = -affinity / (R * T * ln(10)), T = 298.15 K
AQAffinity: predicted_affinity = predicted_affinity_score + 6
Boltz-2:    predicted_affinity = 6 - predicted_affinity_score
```

# Analysis

Analysis was conducted at the compound-level, i.e. predictions from multiple structures for the same compound were averaged.
