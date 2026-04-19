"""Shared metric helpers."""

from __future__ import annotations

import numpy as np


def _validate_pair_inputs(genuine: np.ndarray, imposter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genuine_arr = np.asarray(genuine, dtype=float).reshape(-1)
    imposter_arr = np.asarray(imposter, dtype=float).reshape(-1)

    if genuine_arr.size == 0 or imposter_arr.size == 0:
        raise ValueError("genuine and imposter must both be non-empty")

    return genuine_arr, imposter_arr


def compute_roc_eer(
    genuine: np.ndarray,
    imposter: np.ndarray,
    step: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FAR, GAR, and EER using a fixed threshold sweep."""

    genuine_arr, imposter_arr = _validate_pair_inputs(genuine, imposter)

    thresholds = np.arange(genuine_arr.min() - step, imposter_arr.max() + step + step, step, dtype=float)
    far = np.empty(thresholds.shape, dtype=float)
    frr = np.empty(thresholds.shape, dtype=float)

    for idx, threshold in enumerate(thresholds):
        far[idx] = np.sum(imposter_arr <= threshold) / imposter_arr.size * 100.0
        frr[idx] = np.sum(genuine_arr > threshold) / genuine_arr.size * 100.0

    gar = 100.0 - frr
    min_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[min_idx] + frr[min_idx]) / 2.0)
    return far, gar, eer


def compute_dprime(genuine: np.ndarray, imposter: np.ndarray) -> float:
    """Compute d-prime separation between genuine and imposter scores."""

    genuine_arr, imposter_arr = _validate_pair_inputs(genuine, imposter)

    mu_g = float(np.mean(genuine_arr))
    mu_i = float(np.mean(imposter_arr))
    var_g = float(np.var(genuine_arr))
    var_i = float(np.var(imposter_arr))
    pooled = 0.5 * (var_g + var_i)

    if pooled <= 0.0:
        return 0.0 if mu_g == mu_i else float("inf")

    return abs(mu_g - mu_i) / float(np.sqrt(pooled))


def compute_distribution(data: np.ndarray, gap: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Convert scores into a percentage distribution curve."""

    data_arr = np.asarray(data, dtype=float).reshape(-1)
    if data_arr.size == 0:
        raise ValueError("data must be non-empty")

    x = np.arange(data_arr.min() - gap, data_arr.max() + gap + gap, gap, dtype=float)
    if x.size < 2:
        x = np.array([data_arr[0] - gap, data_arr[0] + gap], dtype=float)

    hist, _ = np.histogram(data_arr, bins=x)
    y = np.concatenate([hist.astype(float), np.array([0.0])]) / data_arr.size * 100.0
    return x, y
