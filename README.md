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
- `projection.png` and `projection.svg`: the sample projection.

The core API is also available from Python:

```python
import pandas as pd
from clrcycle import fit

result = fit(pd.read_csv("matrix.csv", index_col=0))
print(result.coordinates)
print(result.feature_order)
```

## Hogenesch circadian example

The repository includes an example using the GSE54650 mouse tissue atlas from
Hogenesch and colleagues. It creates the supervised reference and the label-free
two-cycle-repeatability projection used in the accompanying study; the example
is not required to run clrcycle on new data.

```bash
bash scripts/download_circadian_data.sh
python scripts/run_all_tissues_clr_acs.py --analysis label-free
```

The label-free outputs are written to
`results/all_tissues_unsupervised_repeat_periodic_96/`. Run the supervised
example with `--analysis supervised`, or both with the default command.

## Development

The GitHub Action checks the generic command-line interface and the Hogenesch
label-free example. Input data and generated outputs are excluded from version
control so clones remain lightweight.

## Data attribution

The optional Hogenesch example downloads GSE54650 from GEO. Please cite Zhang
et al. (2014), *PNAS*, DOI:
[10.1073/pnas.1408886111](https://doi.org/10.1073/pnas.1408886111), when using
those data.
