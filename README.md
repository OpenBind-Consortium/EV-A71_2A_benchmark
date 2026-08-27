# OpenBind Enteroviral 2A Protease dataset and benchmarks

This repository contains processed data, benchmark results, and analysis code for the first [OpenBind](https://openbind.uk/) release: a structure–affinity dataset for structure-based AI and drug discovery.

The release combines crystallographic fragment screening, follow-on compound optimisation, and affinity measurements for enteroviral 2A protease, together with reference benchmarks for docking, co-folding, virtual screening, and affinity prediction.

For more background, see our accompanying blog post, [OpenBind’s first release: A structure–affinity dataset for structure-based AI](https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/), which discusses the release, the enteroviral 2A protease target, and the benchmark results in more detail.

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

- **Experimental dataset:** [Zenodo](https://doi.org/10.5281/zenodo.20026661) and [Fragalysis](https://fragalysis.diamond.ac.uk/viewer/react/preview/target/A71EV2A/tas/lb42888-1)
- **Blog post:** [OpenBind’s first release: A structure–affinity dataset for structure-based AI](https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/)
- **Experimental protocols:** [OpenBind protocols.io workspace](https://www.protocols.io/workspaces/openbind)
- **Fine-tuned OpenFold3-p2 model:** [of3p2-ft-ev2a.ckpt](https://openfold3-data.s3.amazonaws.com/openfold3-parameters/openbind/of3p2-ft-ev2a.ckpt)

The OpenFold3 model was fine-tuned on fragment-bound enteroviral 2A protease structures for target-specific follow-on compound prediction.

## Environment

Create the main analysis environment from the repository root:

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

- Repository licence: [Apache 2.0](LICENSE)
- Data licence: CC0 1.0 Universal
- Dataset DOI: [10.5281/zenodo.20026661](https://doi.org/10.5281/zenodo.20026661)