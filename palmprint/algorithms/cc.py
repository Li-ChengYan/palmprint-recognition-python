"""CompCode implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import minimum_shifted_angular_distance


class CCAlgorithm:
    """Competitive code feature extractor and matcher."""

    name = "CC"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        responses = []
        for kernel in self.kernels:
            response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
            responses.append(response[::4, ::4])
        response_stack = np.stack(responses, axis=0)
        return np.argmin(response_stack, axis=0).astype(np.uint8)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        return minimum_shifted_angular_distance(left, right, shift=2, period=6)
