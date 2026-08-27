# OpenBind Enteroviral 2A Protease dataset and benchmarks

This repository contains processed data, benchmark results, and analysis code for the first [OpenBind](https://openbind.uk/) release: a structure–affinity dataset for structure-based AI and drug discovery.

The release combines crystallographic fragment screening, follow-on compound optimisation, and affinity measurements for enteroviral 2A protease, together with reference benchmarks for docking, co-folding, virtual screening, and affinity prediction.

The accompanying preprint provides a detailed description of the dataset, experimental provenance, benchmark construction, and results: [OpenBind's first release: a structure–affinity dataset for structure-based AI](https://doi.org/10.64898/2026.08.27.747600).

The release was first introduced in an [OpenBind blog post](https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/), which provides a shorter overview of the dataset and project.

## Dataset overview

The dataset focuses on CVA16 / EV-A71 2A protease and contains 925 crystallographic binding events from 699 compounds, together with affinity measurements for 601 compounds.

<p align="center">
  <img width="750" height="360" alt="EV2A_Overview_Figure_small"
       src="https://github.com/user-attachments/assets/96059865-b839-41ea-a618-c055976234fc" />
</p>

The dataset provides dense structural and affinity information from a coherent single-target discovery campaign, enabling evaluation of model behaviour across fragment-to-lead progression, local structure–activity relationships, protein–ligand pose prediction, virtual screening, and affinity prediction.

## Repository structure

```text
EV-A71_2A_benchmark/
├── structure/            Processed structural benchmark data
├── affinity/             Affinity measurements and prediction benchmark
├── virtual_screening/    Virtual-screening benchmark and processed scores
├── similarity_metrics/   Public-data structural similarity analysis
└── plotting/             Manuscript plotting and summary-table generation
```

The main manuscript figures are generated from the processed data stored in this repository. The underlying docking and co-folding calculations were produced using separate workflows:

- [OpenBind docking](https://github.com/OpenBind-Consortium/openbind-docking) — docking preparation, execution, and structural evaluation
- [cofolding_evaluation](https://github.com/OmeirK/cofolding_evaluation) — processing and evaluation of co-folding predictions

## Data and external resources

- **Preprint:** [bioRxiv](https://doi.org/10.64898/2026.08.27.747600)
- **Experimental dataset:** [Zenodo](https://doi.org/10.5281/zenodo.20026660) and [Fragalysis](https://fragalysis.diamond.ac.uk/viewer/react/preview/target/A71EV2A/tas/lb42888-1)
- **Prepared dataset, docking outputs, and cofolding predictions:** [Zenodo](https://zenodo.org/records/20798527)
- **Blog post:** [OpenBind’s first release: A structure–affinity dataset for structure-based AI](https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/)
- **Experimental protocols:** [OpenBind protocols.io workspace](https://www.protocols.io/workspaces/openbind)
- **Fine-tuned OpenFold3-p2 model:** [of3p2-ft-ev2a.ckpt](https://openfold3-data.s3.amazonaws.com/openfold3-parameters/openbind/of3p2-ft-ev2a.ckpt)

The OpenFold3 model was fine-tuned on fragment-bound enteroviral 2A protease structures for target-specific follow-on compound prediction.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark.git
cd EV-A71_2A_benchmark
```

Create and activate the main analysis environment:

```bash
conda env create -f environment.yml
conda activate openbind-analysis
```

The public-data similarity workflow under `similarity_metrics/scripts/` uses a separate environment because it depends on additional structural-search software.

## Reproducing the analysis

The final manuscript figures and summary tables can be regenerated from the repository root with:

```bash
python plotting/plot_figures.py
```

Figures are written to `plotting/figures/` and summary/source tables to `plotting/tables/`.

See [`plotting/README.md`](plotting/README.md) for details of the plotting workflow and its inputs.

## Citation and licence

- Preprint DOI: [10.64898/2026.08.27.747600](https://doi.org/10.64898/2026.08.27.747600)
- Dataset DOI: [10.5281/zenodo.20026660](https://doi.org/10.5281/zenodo.20026660)
- Repository licence: [Apache 2.0](LICENSE)
- Data licence: CC0 1.0 Universal