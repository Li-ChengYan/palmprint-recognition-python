"""Fast-CC implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import minimum_shifted_hamming


class FastCCAlgorithm:
    """Two-direction Fast-CC feature extractor and matcher."""

    name = "FastCC"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        response_a = convolve2d(image_arr, self.kernels[0], mode="same", boundary="fill", fillvalue=0)[::4, ::4]
        response_b = convolve2d(image_arr, self.kernels[3], mode="same", boundary="fill", fillvalue=0)[::4, ::4]
        return response_a > response_b

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        return minimum_shifted_hamming(left, right, shift=4)
