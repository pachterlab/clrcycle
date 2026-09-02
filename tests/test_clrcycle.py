"""Minimal tests for the public clrcycle workflow."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clrcycle import clr_transform, fit, plot


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "toy_cycle.csv"


class ClrCycleTests(unittest.TestCase):
    def test_fit_and_plot(self):
        data = pd.read_csv(DATA, index_col=0)
        result = fit(data, max_features=None, n_swaps=20)

        self.assertEqual(len(result.coordinates), len(data))
        self.assertEqual(set(result.feature_order["feature"]), set(data.columns))
        self.assertTrue(np.isfinite(result.coordinates.iloc[:, 1:].to_numpy()).all())

        weights = result.feature_weights
        n_features = len(result.feature_order)
        self.assertEqual(len(weights), len(data) * n_features)
        self.assertEqual(
            weights["sample"].tolist(), np.repeat(data.index.astype(str), n_features).tolist()
        )
        self.assertEqual(
            weights["feature"].tolist(),
            np.tile(result.feature_order["feature"], len(data)).tolist(),
        )
        self.assertEqual(
            weights["clrcycle_position"].tolist(),
            np.tile(np.arange(n_features), len(data)).tolist(),
        )
        np.testing.assert_allclose(
            weights.groupby("sample", sort=False)["feature_weight"].sum().to_numpy(),
            result.coordinates["clrcycle_radius"],
            rtol=1e-12,
            atol=1e-12,
        )
        transformed = clr_transform(data.to_numpy())
        centered = transformed - transformed.mean(axis=0, keepdims=True)
        order = [data.columns.get_loc(name) for name in result.feature_order["feature"]]
        ordered = centered[:, order]
        np.testing.assert_allclose(weights["centered_clr"], ordered.ravel())

        figure = plot(result)
        self.addCleanup(plt.close, figure)
        self.assertGreaterEqual(len(figure.axes), 2)

    def test_command_line_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clrcycle",
                    str(DATA),
                    "--output-dir",
                    directory,
                    "--all-features",
                    "--n-swaps",
                    "20",
                ],
                cwd=ROOT,
                check=True,
            )
            for name in (
                "sample_coordinates.csv",
                "feature_order.csv",
                "feature_weights.csv",
                "projection.png",
                "projection.svg",
            ):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
