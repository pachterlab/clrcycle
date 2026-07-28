# clrcycle

`clrcycle` is a centered-log-ratio circular projection method for cyclic
compositional data. This repository accompanies *Circular Projection for Cyclic
Compositional Data* and
repository reproduces the circadian analyses of the GSE54650 mouse tissue atlas:

- a supervised reference, selecting 240 probes by their known-time harmonic
  association; and
- a label-free analysis, selecting 96 probes by two-cycle repeatability.

The latter uses no circadian-time values during feature selection. In both
cases, time is used after fitting only to orient and evaluate the circular
projection.

## Reproduce the analysis

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/download_circadian_data.sh
python scripts/run_all_tissues_clr_acs.py
```

The download script retrieves the GSE54650 series matrix and GPL6246 probe
annotation directly from GEO. No expression data are versioned in this
repository. The final command writes the two manuscript analyses to:

```
results/all_tissues/
results/all_tissues_unsupervised_repeat_periodic_96/
```

Run one analysis only with `--analysis supervised` or `--analysis label-free`.
Use `--results-dir PATH` to write results outside the repository.

## Main files

- `scripts/run_all_tissues_clr_acs.py`: all-tissue implementation and figure
  generation.
- `scripts/run_liver_clr_acs.py`: liver-only supervised example and shared
  clrcycle routines.
- `scripts/download_circadian_data.sh`: official GEO data download.

The computations use fixed random seeds. The clrcycle order is initialized from the
leading covariance eigenvectors and refined by 4,000 improving pairwise swaps.

## Data and attribution

GSE54650 is the mammalian circadian atlas reported by Zhang et al. (2014),
*PNAS*, DOI: [10.1073/pnas.1408886111](https://doi.org/10.1073/pnas.1408886111).
Please cite the atlas when using the downloaded data.

## Repository scope

This repository deliberately contains the circadian workflow only. Generated
data, plots, and LaTeX intermediates are ignored so a clone stays lightweight
and all analysis outputs can be regenerated from the public source data.
