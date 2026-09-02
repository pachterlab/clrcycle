"""Command-line entry point for clrcycle."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import fit, plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a clrcycle projection to a samples-by-features CSV matrix.")
    parser.add_argument("input_csv", type=Path, help="CSV with samples as rows and features as columns; first column is sample IDs.")
    parser.add_argument("--output-dir", type=Path, default=Path("clrcycle_output"))
    parser.add_argument("--max-features", type=int, default=240, help="Number of label-free high-variance features to retain (default: 240).")
    parser.add_argument("--all-features", action="store_true", help="Use every nonconstant feature instead of feature selection.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-swaps", type=int, default=4000)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv, index_col=0)
    result = fit(
        data,
        max_features=None if args.all_features else args.max_features,
        seed=args.seed,
        n_swaps=args.n_swaps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.coordinates.to_csv(args.output_dir / "sample_coordinates.csv", index=False)
    result.feature_order.to_csv(args.output_dir / "feature_order.csv", index=False)
    result.feature_weights.to_csv(args.output_dir / "feature_weights.csv", index=False)

    figure = plot(result)
    figure.savefig(args.output_dir / "projection.png", dpi=220)
    figure.savefig(args.output_dir / "projection.svg")
    plt.close(figure)
    print(f"Selected features: {len(result.feature_order)}")
    print(f"clrcycle objective: {result.objective:.6f}")
    print(f"rho_clrcycle: {result.rho:.6f}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
