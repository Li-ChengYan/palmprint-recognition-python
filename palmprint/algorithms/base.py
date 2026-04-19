"""Common algorithm helpers."""

from __future__ import annotations

import numpy as np


def ensure_same_shape(feature_a: np.ndarray, feature_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate that two array-like features have the same shape."""

    left = np.asarray(feature_a)
    right = np.asarray(feature_b)
    if left.shape != right.shape:
        raise ValueError(f"feature shape mismatch: {left.shape} != {right.shape}")
    return left, right
