"""clrcycle: circular projection for compositional data."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ClrCycleResult:
    """Result of a clrcycle fit."""

    coordinates: pd.DataFrame
    feature_order: pd.DataFrame
    objective: float
    rho: float


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
    return ClrCycleResult(
        coordinates=coordinates,
        feature_order=feature_order,
        objective=objective,
        rho=float(objective / (eigenvalues[-1] + eigenvalues[-2])),
    )
