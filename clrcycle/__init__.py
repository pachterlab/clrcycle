"""clrcycle: circular projection for compositional data."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["ClrCycleResult", "circular_order", "clr_transform", "fit", "plot"]


@dataclass
class ClrCycleResult:
    """Result including per-sample feature contributions to the radial projection."""

    coordinates: pd.DataFrame
    feature_order: pd.DataFrame
    objective: float
    rho: float
    feature_weights: pd.DataFrame = field(default_factory=pd.DataFrame)


def clr_transform(values: np.ndarray) -> np.ndarray:
    """Apply the centered log-ratio transform to a nonnegative matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("values must be a nonempty two-dimensional matrix.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("values must be finite and nonnegative.")
    if np.any(values <= 0):
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError("at least one input value must be positive.")
        values = values + 1e-3 * positive.min()
    log_values = np.log(values)
    return log_values - log_values.mean(axis=1, keepdims=True)


def _objective(covariance: np.ndarray, order: np.ndarray) -> float:
    d = len(order)
    angles = 2.0 * np.pi * np.arange(d) / d
    cosine = np.sqrt(2.0 / d) * np.cos(angles)
    sine = np.sqrt(2.0 / d) * np.sin(angles)
    ordered = covariance[np.ix_(order, order)]
    return float(cosine @ ordered @ cosine + sine @ ordered @ sine)


def circular_order(covariance: np.ndarray, *, seed: int = 17, n_swaps: int = 4000) -> tuple[np.ndarray, float]:
    """Learn a cyclic feature order by spectral initialization and improving swaps."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    if covariance.shape[0] < 3:
        raise ValueError("clrcycle requires at least three features.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    xy = eigenvectors[:, -2:] * np.sqrt(np.maximum(eigenvalues[-2:], 0.0))
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    best = _objective(covariance, order)
    rng = np.random.default_rng(seed)
    for _ in range(n_swaps):
        left, right = rng.choice(len(order), size=2, replace=False)
        candidate = order.copy()
        candidate[left], candidate[right] = candidate[right], candidate[left]
        score = _objective(covariance, candidate)
        if score > best:
            order, best = candidate, score
    return order, best


def fit(
    data: pd.DataFrame,
    *,
    max_features: Optional[int] = 240,
    seed: int = 17,
    n_swaps: int = 4000,
) -> ClrCycleResult:
    """Fit clrcycle to a samples-by-features nonnegative data frame.

    When ``max_features`` is set, features are selected without labels by log
    variance. This keeps the optimization practical for wide matrices.
    """
    if data.shape[0] < 3 or data.shape[1] < 3:
        raise ValueError("clrcycle requires at least three samples and three features.")
    numeric = data.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.to_numpy() < 0).any():
        raise ValueError("input values must be finite and nonnegative.")

    positive = numeric.to_numpy()[numeric.to_numpy() > 0]
    if positive.size == 0:
        raise ValueError("at least one input value must be positive.")
    stabilized = numeric + (1e-3 * positive.min() if (numeric.to_numpy() == 0).any() else 0.0)
    log_variance = np.log(stabilized).var(axis=0, ddof=1)
    nonconstant = log_variance[log_variance > 0]
    if len(nonconstant) < 3:
        raise ValueError("fewer than three features vary across samples.")
    if max_features is not None:
        selected_names = nonconstant.nlargest(min(max_features, len(nonconstant))).index
    else:
        selected_names = nonconstant.index
    selected = numeric.loc[:, selected_names]

    transformed = clr_transform(selected.to_numpy())
    centered = transformed - transformed.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    order, objective = circular_order(covariance, seed=seed, n_swaps=n_swaps)
    d = len(order)
    angles = 2.0 * np.pi * np.arange(d) / d
    cosine = np.sqrt(2.0 / d) * np.cos(angles)
    sine = np.sqrt(2.0 / d) * np.sin(angles)
    cosine_coordinate = centered[:, order] @ cosine
    sine_coordinate = centered[:, order] @ sine
    eigenvalues = np.linalg.eigvalsh(covariance)

    coordinates = pd.DataFrame(
        {
            "sample": selected.index.astype(str),
            "clrcycle_cosine": cosine_coordinate,
            "clrcycle_sine": sine_coordinate,
            "clrcycle_angle": np.mod(np.arctan2(sine_coordinate, cosine_coordinate), 2.0 * np.pi),
            "clrcycle_radius": np.hypot(cosine_coordinate, sine_coordinate),
        }
    )
    feature_order = pd.DataFrame(
        {
            "feature": selected.columns.to_numpy()[order],
            "clrcycle_position": np.arange(d),
            "clrcycle_angle": angles,
            "log_variance": log_variance.loc[selected.columns.to_numpy()[order]].to_numpy(),
        }
    )
    sample_angles = coordinates["clrcycle_angle"].to_numpy()
    radii = coordinates["clrcycle_radius"].to_numpy()
    ordered_centered = centered[:, order]
    weights = ordered_centered * (
        np.cos(sample_angles[:, None]) * cosine + np.sin(sample_angles[:, None]) * sine
    )
    feature_weights = pd.DataFrame(
        {
            "sample": np.repeat(coordinates["sample"].to_numpy(), d),
            "feature": np.tile(feature_order["feature"].to_numpy(), len(coordinates)),
            "clrcycle_position": np.tile(
                feature_order["clrcycle_position"].to_numpy(), len(coordinates)
            ),
            "feature_angle": np.tile(feature_order["clrcycle_angle"].to_numpy(), len(coordinates)),
            "sample_angle": np.repeat(sample_angles, d),
            "clrcycle_radius": np.repeat(radii, d),
            "centered_clr": ordered_centered.ravel(),
            "feature_weight": weights.ravel(),
        }
    )
    if not np.allclose(weights.sum(axis=1), radii, rtol=1e-12, atol=1e-12):
        raise RuntimeError("feature weights do not reconstruct clrcycle_radius.")
    return ClrCycleResult(
        coordinates=coordinates,
        feature_order=feature_order,
        objective=objective,
        rho=float(objective / (eigenvalues[-1] + eigenvalues[-2])),
        feature_weights=feature_weights,
    )


def plot(result: ClrCycleResult, *, feature_labels: Iterable[str] = ()) -> "Figure":
    """Plot the sample projection and learned feature circle in one figure."""
    import matplotlib.pyplot as plt

    coordinates = result.coordinates
    features = result.feature_order
    labels = {str(label) for label in feature_labels}
    cmap = "twilight_shifted"

    figure, (sample_axis, feature_axis) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True
    )
    sample_axis.scatter(
        coordinates["clrcycle_cosine"],
        coordinates["clrcycle_sine"],
        c=coordinates["clrcycle_angle"],
        cmap=cmap,
        vmin=0,
        vmax=2.0 * np.pi,
        s=46,
        edgecolor="black",
        linewidth=0.5,
    )
    sample_axis.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    sample_axis.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    sample_axis.set_aspect("equal", adjustable="datalim")
    sample_axis.set_xlabel("clrcycle cosine coordinate")
    sample_axis.set_ylabel("clrcycle sine coordinate")
    sample_axis.set_title("Sample projection")

    angles = features["clrcycle_angle"].to_numpy(float)
    x = np.cos(angles)
    y = np.sin(angles)
    points = feature_axis.scatter(
        x,
        y,
        c=angles,
        cmap=cmap,
        vmin=0,
        vmax=2.0 * np.pi,
        s=24,
    )
    feature_axis.add_patch(
        plt.Circle((0, 0), 1.0, fill=False, color="0.75", linewidth=0.9)
    )
    for name, angle in zip(features["feature"].astype(str), angles):
        if name in labels:
            label_x, label_y = 1.1 * np.cos(angle), 1.1 * np.sin(angle)
            feature_axis.text(
                label_x,
                label_y,
                name,
                fontsize=7,
                ha="left" if label_x >= 0 else "right",
                va="center",
            )
    feature_axis.set_aspect("equal")
    feature_axis.set_xlim(-1.3, 1.3)
    feature_axis.set_ylim(-1.3, 1.3)
    feature_axis.set_xticks([])
    feature_axis.set_yticks([])
    for spine in feature_axis.spines.values():
        spine.set_visible(False)
    feature_axis.set_title("Learned feature circle")

    colorbar = figure.colorbar(points, ax=[sample_axis, feature_axis], shrink=0.8)
    colorbar.set_label("clrcycle angle")
    colorbar.set_ticks([0, np.pi, 2.0 * np.pi], labels=["0", "π", "2π"])
    return figure
