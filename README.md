# clrcycle

`clrcycle` is a circular projection method for nonnegative, compositional data.
It applies a centered log-ratio transformation, learns a cyclic feature ordering by maximizing variance captured by the first Fourier mode, and projects samples onto the corresponding cosine and sine coordinates. The resulting two-dimensional representation assigns each sample a phase and a radius measuring the magnitude of its projection onto the learned first-Fourier plane.

Immediately after the radius, clrcycle decomposes it into per-feature weights,
$w_{ig} = Z_{ig}(c_g\cos(\phi_i) + s_g\sin(\phi_i))$. `feature_weight` is
feature *g*'s contribution to sample *i*'s radial first-Fourier projection:
positive weights reinforce the sample's projected direction, negative weights
oppose it, and the weights sum exactly to the sample radius. They are derived
from centered CLR values and the fitted first-Fourier geometry, not expression
values or differential-expression statistics.

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
- `feature_order.csv`: the learned cyclic feature order;
- `feature_weights.csv`: per-sample feature contributions to the radius; and
- `projection.png` and `projection.svg`: the sample projection and learned
  feature circle in a single figure.

### Command-line modes and defaults

| Option | Default | Meaning |
|---|---:|---|
| `input_csv` | required | Samples-by-features CSV; the first column contains sample identifiers. |
| `--output-dir` | `clrcycle_output` | Directory for the result tables and figures. |
| `--max-features N` | `240` | Retain the `N` nonconstant features with greatest log variance. Selection does not use sample labels. |
| `--all-features` | off | Use every nonconstant feature, overriding `--max-features`. |
| `--seed` | `17` | Seed for the random improving-swap proposals. |
| `--n-swaps` | `4000` | Number of candidate pair swaps evaluated after spectral initialization. |

The default mode is label-free high-variance selection followed by a
fixed-seed, 4,000-swap optimization. Use `--all-features` when the input is
already a selected panel. Increasing `--n-swaps` explores more candidate orders
but takes longer. The same input, seed, and swap count are reproducible.

Run `python -m clrcycle --help` for the current command-line help.

## Python API

The core API is also available from Python:

```python
import pandas as pd
from clrcycle import fit, plot

result = fit(pd.read_csv("matrix.csv", index_col=0))
result.coordinates
result.feature_order
result.feature_weights

figure = plot(result)
figure.savefig("clrcycle.png", dpi=220)
```

### `fit`

```python
fit(data, *, max_features=240, seed=17, n_swaps=4000)
```

Fits clrcycle to a nonnegative pandas data frame with samples in rows and
features in columns. `max_features=None` uses every nonconstant feature;
otherwise the highest-log-variance features are retained. It returns a
`ClrCycleResult` and does not modify the input data frame.

Zeros are handled with a small data-derived pseudocount. Missing, infinite, or
negative values are rejected. At least three samples and three varying features
are required.

### `plot`

```python
plot(result, *, feature_labels=())
```

Creates the standard sample-projection and learned-feature-circle figure from a
`ClrCycleResult`. It returns a Matplotlib `Figure`; displaying or saving it is
left to the caller. `feature_labels` controls which feature names are annotated
and has no effect on the fit.

Pass a short list of feature names with
`plot(result, feature_labels=["feature_a", "feature_b"])` to label them on the
circle. The plotting helper is deliberately minimal and dataset-independent.
For metadata-aware and publication-style figures, use the returned tables as a
starting point; examples are available in the
[clrcycle paper repository](https://github.com/pachterlab/SEP_2026).

### `ClrCycleResult`

The object returned by `fit` contains:

| Attribute | Contents |
|---|---|
| `coordinates` | One row per sample with `sample`, `clrcycle_cosine`, `clrcycle_sine`, `clrcycle_angle`, and `clrcycle_radius`. Angles are in radians on `[0, 2π)`. |
| `feature_order` | One row per retained feature with `feature`, zero-based `clrcycle_position`, `clrcycle_angle`, and `log_variance`. |
| `feature_weights` | One row per sample and retained feature with its centered CLR value and contribution to the sample's radial projection. |
| `objective` | Variance captured by the learned first circular Fourier mode. |
| `rho` | Objective divided by the sum of the two largest covariance eigenvalues. |

The absolute rotation and reflection of an unsupervised circle are arbitrary.
Use external metadata after fitting if an application requires alignment to an
absolute phase reference.

### `clr_transform`

```python
clr_transform(values)
```

Applies the centered log-ratio transform to a nonnegative two-dimensional NumPy
array, treating rows as samples. Zeros receive the same small, data-dependent
pseudocount used by `fit`.

### `circular_order`

```python
circular_order(covariance, *, seed=17, n_swaps=4000)
```

Learns a cyclic order directly from a square covariance matrix. It returns
`(order, objective)`, where `order` is an integer NumPy array. Most users should
call `fit`; this lower-level function is useful when a covariance matrix has
already been prepared.

## Tutorial

The Colab-ready
[`tutorial/circadian_liver.ipynb`](tutorial/circadian_liver.ipynb) notebook is
the supported end-to-end example. It downloads the public GSE54650 mouse liver
data, clearly separates data processing and feature selection from the two
clrcycle commands, and explains how to interpret the resulting sample
projection and learned gene circle.

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
