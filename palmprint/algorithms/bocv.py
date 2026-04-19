"""BOCV implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import mean_per_channel_shifted_hamming


class BOCVAlgorithm:
    """Binary orientation co-occurrence vector extractor and matcher."""

    name = "BOCV"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        features = []
        for kernel in self.kernels:
            response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
            features.append(response[::4, ::4] > 0)
        return np.stack(features, axis=0)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        return mean_per_channel_shifted_hamming(left, right, shift=4)
