"""PalmCode implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_pc_gabor
from palmprint.core.matching import iter_shifted_overlap


class PCAlgorithm:
    """PalmCode feature extractor and matcher."""

    name = "PC"

    def __init__(self) -> None:
        self.kernel = generate_pc_gabor(np.pi / 4.0)

    def extract(self, image: np.ndarray) -> np.ndarray:
        response = convolve2d(np.asarray(image, dtype=float), self.kernel, mode="same", boundary="fill", fillvalue=0)
        downsampled = response[::4, ::4]
        real_bits = np.real(downsampled) > 0
        imag_bits = np.imag(downsampled) > 0
        return np.concatenate([real_bits, imag_bits], axis=1)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        left_real, left_imag = left[:, :32], left[:, 32:]
        right_real, right_imag = right[:, :32], right[:, 32:]

        distances: list[float] = []
        real_windows = iter_shifted_overlap(left_real, right_real, shift=2)
        imag_windows = iter_shifted_overlap(left_imag, right_imag, shift=2)
        for (real_a, real_b), (imag_a, imag_b) in zip(real_windows, imag_windows, strict=True):
            real_distance = np.mean(np.logical_xor(real_a, real_b))
            imag_distance = np.mean(np.logical_xor(imag_a, imag_b))
            distances.append(float((real_distance + imag_distance) / 2.0))

        return min(distances)
