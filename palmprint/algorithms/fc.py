"""FusionCode implementation."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from palmprint.algorithms.base import ensure_same_shape
from palmprint.core.kernels import generate_fc_gabor_bank
from palmprint.core.matching import iter_shifted_overlap


class FCAlgorithm:
    """FusionCode feature extractor and matcher."""

    name = "FC"

    def __init__(self) -> None:
        self.kernels = generate_fc_gabor_bank()

    def extract(self, image: np.ndarray) -> np.ndarray:
        image_arr = np.asarray(image, dtype=float)
        magnitudes = []
        phases = []

        for kernel in self.kernels:
            response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
            magnitudes.append(np.abs(response)[::4, ::4])
            phases.append(np.angle(response)[::4, ::4])

        magnitude_stack = np.stack(magnitudes, axis=0)
        phase_stack = np.stack(phases, axis=0)

        orientation = np.argmax(magnitude_stack, axis=0)
        selected_phase = np.take_along_axis(phase_stack, orientation[None, :, :], axis=0)[0]
        selected_phase = np.where(selected_phase < 0, selected_phase + 2.0 * np.pi, selected_phase)

        real_bits = np.logical_or(
            np.logical_and(selected_phase >= 0.0, selected_phase < np.pi / 2.0),
            np.logical_and(selected_phase >= 3.0 * np.pi / 2.0, selected_phase < 2.0 * np.pi),
        )
        imag_bits = np.logical_and(selected_phase >= 0.0, selected_phase < np.pi)
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
