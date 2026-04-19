"""DOC implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import iter_shifted_overlap


class DOCAlgorithm:
    """Double-orientation code extractor and matcher."""

    name = "DOC"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        responses = []
        for kernel in self.kernels:
            response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
            responses.append(response[::4, ::4])

        response_stack = np.stack(responses, axis=0)
        sorted_indices = np.argsort(response_stack, axis=0)
        primary = sorted_indices[0].astype(np.uint8)
        secondary = sorted_indices[1].astype(np.uint8)
        return np.stack([primary, secondary], axis=0)

    def match(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        left, right = ensure_same_shape(feature_a, feature_b)
        if left.shape != (2, 32, 32):
            raise ValueError("DOC features must have shape (2, 32, 32)")

        scores = []
        primary_windows = iter_shifted_overlap(left[0], right[0], shift=2)
        secondary_windows = iter_shifted_overlap(left[1], right[1], shift=2)
        cross_a_windows = iter_shifted_overlap(left[0], right[1], shift=2)
        cross_b_windows = iter_shifted_overlap(left[1], right[0], shift=2)

        for (p1_a, p1_b), (p2_a, p2_b), (c1_a, c1_b), (c2_a, c2_b) in zip(
            primary_windows,
            secondary_windows,
            cross_a_windows,
            cross_b_windows,
            strict=True,
        ):
            dis_1 = np.minimum(np.abs(p1_a.astype(np.int16) - p1_b.astype(np.int16)), 12 - np.abs(p1_a.astype(np.int16) - p1_b.astype(np.int16)))
            dis_2 = np.minimum(np.abs(c1_a.astype(np.int16) - c1_b.astype(np.int16)), 12 - np.abs(c1_a.astype(np.int16) - c1_b.astype(np.int16)))
            dis_3 = np.minimum(np.abs(c2_a.astype(np.int16) - c2_b.astype(np.int16)), 12 - np.abs(c2_a.astype(np.int16) - c2_b.astype(np.int16)))
            dis_4 = np.minimum(np.abs(p2_a.astype(np.int16) - p2_b.astype(np.int16)), 12 - np.abs(p2_a.astype(np.int16) - p2_b.astype(np.int16)))

            p1_score = np.exp(-dis_1) + np.exp(-dis_4)
            p2_score = np.exp(-dis_2) + np.exp(-dis_3)
            p_score = np.maximum(p1_score, p2_score)
            scores.append(float(np.mean(p_score) / 2.0))

        return 1.0 - max(scores)
