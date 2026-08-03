import unittest

import pandas as pd

from clrcycle import fit


class TestClrCycleFit(unittest.TestCase):
    def test_scale_only_samples_have_no_compositional_variation(self):
        data = pd.DataFrame(
            [[1, 2, 4], [10, 20, 40], [100, 200, 400]],
            columns=["feature_a", "feature_b", "feature_c"],
        )

        with self.assertRaisesRegex(ValueError, "no compositional variation"):
            fit(data, n_swaps=0)


if __name__ == "__main__":
    unittest.main()
