"""RLOC and Fast-RLOC implementations."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_rloc_filter_bank
from palmprint.core.matching import maximum_shifted_similarity, iter_shifted_overlap


def _extract_rloc_code(image: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    image_arr = np.asarray(image, dtype=float)
    responses = []
    for kernel in kernels:
        response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
        responses.append(response[::4, ::4])
    return np.argmin(np.stack(responses, axis=0), axis=0).astype(np.uint8)


class RLOCAlgorithm:
    """RLOC with pixel-to-area matching."""

    name = "RLOC"

    def __init__(self) -> None:
        self.kernels = generate_rloc_filter_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        return _extract_rloc_code(image, self.kernels)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        scores: list[float] = []

        for left_window, right_window in iter_shifted_overlap(left, right, shift=2):
            padded_right = np.pad(right_window, pad_width=1, mode="constant", constant_values=6)
            padded_left = np.pad(left_window, pad_width=1, mode="constant", constant_values=6)

            left_hits = np.zeros(left_window.shape, dtype=bool)
            right_hits = np.zeros(right_window.shape, dtype=bool)

            for row in range(left_window.shape[0]):
                for col in range(left_window.shape[1]):
                    right_cross = (
                        padded_right[row, col + 1],
                        padded_right[row + 1, col],
                        padded_right[row + 1, col + 1],
                        padded_right[row + 1, col + 2],
                        padded_right[row + 2, col + 1],
                    )
                    left_cross = (
                        padded_left[row, col + 1],
                        padded_left[row + 1, col],
                        padded_left[row + 1, col + 1],
                        padded_left[row + 1, col + 2],
                        padded_left[row + 2, col + 1],
                    )
                    left_hits[row, col] = int(left_window[row, col]) in right_cross
                    right_hits[row, col] = int(right_window[row, col]) in left_cross

            scores.append(float(max(np.mean(left_hits), np.mean(right_hits))))

        return 1.0 - max(scores)


class FastRLOCAlgorithm:
    """Fast-RLOC with one-to-one matching."""

    name = "FastRLOC"

    def __init__(self) -> None:
        self.kernels = generate_rloc_filter_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        return _extract_rloc_code(image, self.kernels)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        return 1.0 - maximum_shifted_similarity(left, right, shift=4)
