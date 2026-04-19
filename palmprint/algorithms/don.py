"""DoN implementation."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape


@lru_cache(maxsize=1)
def _load_don_template() -> np.ndarray:
    resource = files("palmprint.resources").joinpath("don_template.npy")
    with resource.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


class DoNAlgorithm:
    """DoN feature extractor and matcher."""

    name = "DoN"

    def __init__(self) -> None:
        self.kernel = _load_don_template()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        inverted = 255.0 - image_arr
        response = convolve2d(inverted, self.kernel, mode="same", boundary="fill", fillvalue=0)

        code_1 = response > 0
        code_2 = binary_opening(code_1)
        code_3 = binary_closing(code_1)
        return np.stack([code_1, code_2, code_3], axis=0)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        if left.ndim != 3 or left.shape[0] != 3:
            raise ValueError("DoN features must have shape (3, rows, cols)")

        channels, rows, cols = left.shape
        shift = 7
        cropped_left = left[:, shift:-shift, shift:-shift]
        crop_rows = cropped_left.shape[1]
        crop_cols = cropped_left.shape[2]

        scores = np.zeros((shift * 2, shift * 2), dtype=float)
        weights = np.array([0.6, 0.2, 0.2], dtype=float)
        denominator = float(channels * rows * cols)

        for row_offset in range(shift * 2):
            for col_offset in range(shift * 2):
                target = right[:, row_offset : row_offset + crop_rows, col_offset : col_offset + crop_cols]
                xor_counts = np.logical_xor(cropped_left, target).reshape(channels, -1).sum(axis=1)
                scores[row_offset, col_offset] = float(np.sum(xor_counts * weights) / denominator)

        return float(np.min(scores))
