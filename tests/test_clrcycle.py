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

from clrcycle import fit, plot


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "toy_cycle.csv"


class ClrCycleTests(unittest.TestCase):
    def test_fit_and_plot(self):
        data = pd.read_csv(DATA, index_col=0)
        result = fit(data, max_features=None, n_swaps=20)

        self.assertEqual(len(result.coordinates), len(data))
        self.assertEqual(set(result.feature_order["feature"]), set(data.columns))
        self.assertTrue(np.isfinite(result.coordinates.iloc[:, 1:].to_numpy()).all())

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
                "projection.png",
                "projection.svg",
            ):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
