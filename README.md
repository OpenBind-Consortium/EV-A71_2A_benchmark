# OpenBind EV-A71 2A dataset and benchmarks

This repository contains data and reference benchmarks for the first OpenBind release: a structure–affinity dataset for structure-based AI.

The release includes experimentally determined protein–ligand complexes, affinity measurements, and benchmark evaluations for docking, cofolding, and affinity prediction methods.

## Overview

The dataset focuses on EV-A71 2A protease and contains:

- 925 crystallographic binding events  
- 699 compounds  
- 601 compounds with affinity measurements  

This is a dense, single-target dataset designed for:
- model training and fine-tuning  
- benchmarking and comparison  
- error analysis and method development  

Unlike many public resources, it provides both structure and affinity across a coherent compound series, making it valuable for studying local structure–activity relationships.

## What’s included

- Curated structure and affinity data (see data links below)
- Prepared benchmark tables for plotting and analysis
- Reference evaluations across:
  - docking (classical and ML-based)
  - cofolding methods
  - affinity prediction


## Data and external resources

- Dataset: [Zenodo](https://doi.org/10.5281/zenodo.20026661) / [Fragalysis](https://fragalysis.diamond.ac.uk/viewer/react/landing)  
- Benchmarks: this repository  
- Experimental protocols: [OpenBind protocols.io workspace ](https://doi.org/10.5281/zenodo.20026661) 

## Usage

To reproduce benchmark figures:

```bash
python analysis/plotting/plot_benchmark_figures.py
```

See `analysis/plotting/README.md` for details.


## Citation and license

- Data license: CC0 1.0 Universal  
- DOI: [10.5281/zenodo.20026661](https://doi.org/10.5281/zenodo.20026661)  
