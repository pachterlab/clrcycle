# clrcycle

`clrcycle` is a circular projection method for nonnegative, compositional data.
It applies a centered log-ratio transform, learns a cyclic ordering of features,
and projects samples onto the first circular Fourier mode. The result is an
interpretable phase--amplitude view: sample angle describes position around the
dominant cycle, sample radius measures the strength of the corresponding
compositional contrast, and the learned feature circle orders features by their
contribution to that cycle.

## Install

Python 3.10 or later is required.

```bash
git clone https://github.com/pachterlab/clrcycle.git
cd clrcycle
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For an editable development installation, use `python -m pip install -e .`.

## Run clrcycle

Supply a CSV with samples as rows, features as columns, and sample identifiers
in the first column. Values must be finite and nonnegative.

```bash
python -m clrcycle path/to/matrix.csv --output-dir clrcycle_output
```

For wide matrices, clrcycle selects the 240 features with largest log variance
by default, without labels. Use `--max-features N` to change the panel size or
`--all-features` to fit all nonconstant features. The command writes:

- `sample_coordinates.csv`: cosine/sine coordinates, phase, and radius;
- `feature_order.csv`: the learned cyclic feature order; and
- `projection.png` and `projection.svg`: the sample projection and learned
  feature circle in a single figure.

The core API is also available from Python:

```python
import pandas as pd
from clrcycle import fit, plot

result = fit(pd.read_csv("matrix.csv", index_col=0))
print(result.coordinates)
print(result.feature_order)

figure = plot(result)
figure.savefig("clrcycle.png", dpi=220)
```

Pass a short list of feature names with
`plot(result, feature_labels=["feature_a", "feature_b"])` to label them on the
circle. The plotting helper is deliberately minimal and dataset-independent.
For metadata-aware and publication-style figures, use the returned tables as a
starting point; examples are available in the
[clrcycle paper repository](https://github.com/pachterlab/SEP_2026).

## Tutorial

The Colab-ready
[`tutorial/circadian_liver.ipynb`](tutorial/circadian_liver.ipynb) notebook is
the supported end-to-end example. It downloads the public GSE54650 mouse liver
data, clearly separates data processing and feature selection from the two
clrcycle commands, and explains how to interpret the resulting sample
projection and learned gene circle.

Open it directly in
[Google Colab](https://colab.research.google.com/github/pachterlab/clrcycle/blob/main/tutorial/circadian_liver.ipynb)
or run it from a local checkout. The notebook saves its tables and PNG/SVG
figure under `tutorial/results/`. Downloaded data and generated results are not
tracked by Git.

## Development

Run the dataset-independent API, plotting, and command-line regression tests
with:

```bash
python -m unittest discover -s tests -v
```

The GitHub Action runs these tests on Python 3.10 through 3.13. Input data and
generated outputs are excluded from version control so clones remain
lightweight.

## Data attribution

The tutorial downloads GSE54650 from GEO. Please cite Zhang et al. (2014),
*PNAS*, DOI:
[10.1073/pnas.1408886111](https://doi.org/10.1073/pnas.1408886111), when using
those data.
