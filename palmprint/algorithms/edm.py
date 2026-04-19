"""EDM implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import mean_per_channel_shifted_hamming


class EDMAlgorithm:
    """Extreme downsampling matcher using shared best-impact pixels."""

    name = "EDM"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        responses = []
        for kernel in self.kernels:
            responses.append(convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0))

        response_stack = np.stack(responses, axis=0)
        channels, rows, cols = response_stack.shape
        if rows % 4 != 0 or cols % 4 != 0:
            raise ValueError("EDM response maps must be divisible by 4")

        out = np.zeros((channels, rows // 4, cols // 4), dtype=bool)
        for row in range(rows // 4):
            row_slice = slice(row * 4, (row + 1) * 4)
            for col in range(cols // 4):
                col_slice = slice(col * 4, (col + 1) * 4)
                block = response_stack[:, row_slice, col_slice]
                impact = np.sum(np.abs(block), axis=0)
                best_index = int(np.argmax(impact))
                best_row, best_col = divmod(best_index, 4)
                out[:, row, col] = block[:, best_row, best_col] > 0

        return out

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        return mean_per_channel_shifted_hamming(left, right, shift=4)
