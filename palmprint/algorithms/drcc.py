"""DRCC implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d

from palmprint.core.kernels import generate_drcc_filter_bank
from palmprint.core.matching import iter_shifted_overlap


@dataclass(frozen=True)
class DRCCFeature:
    """Dominant-direction and neighbor-order representation."""

    dominant: np.ndarray
    neighbor: np.ndarray


class DRCCAlgorithm:
    """Discriminative and robust competitive code extractor and matcher."""

    name = "DRCC"

    def __init__(self) -> None:
        filters = generate_drcc_filter_bank()
        self.kernels = filters[:6]
        self.prefilter = filters[6]

    def extract(self, image: np.ndarray) -> DRCCFeature:
        image_arr = np.asarray(image, dtype=float)
        smoothed = convolve2d(image_arr, self.prefilter, mode="same", boundary="fill", fillvalue=0)

        responses = []
        for kernel in self.kernels:
            response = convolve2d(smoothed, kernel, mode="same", boundary="fill", fillvalue=0)
            responses.append(response[::4, ::4])

        response_stack = np.stack(responses, axis=0)
        dominant = np.argmax(response_stack, axis=0).astype(np.uint8)
        neighbor = np.zeros(dominant.shape, dtype=bool)

        for direction in range(6):
            mask = dominant == direction
            if not np.any(mask):
                continue
            left_direction = (direction + 1) % 6
            right_direction = (direction - 1) % 6
            neighbor[mask] = response_stack[left_direction][mask] >= response_stack[right_direction][mask]

        return DRCCFeature(dominant=dominant, neighbor=neighbor)

    def match(self, feature_a: DRCCFeature, feature_b: DRCCFeature) -> float:
        if feature_a.dominant.shape != feature_a.neighbor.shape:
            raise ValueError("DRCC feature_a has mismatched dominant and neighbor shapes")
        if feature_b.dominant.shape != feature_b.neighbor.shape:
            raise ValueError("DRCC feature_b has mismatched dominant and neighbor shapes")
        if feature_a.dominant.shape != feature_b.dominant.shape:
            raise ValueError("DRCC feature shape mismatch")

        distances = []
        dominant_windows = iter_shifted_overlap(feature_a.dominant, feature_b.dominant, shift=2)
        neighbor_windows = iter_shifted_overlap(feature_a.neighbor, feature_b.neighbor, shift=2)
        for (dom_a, dom_b), (nei_a, nei_b) in zip(dominant_windows, neighbor_windows, strict=True):
            delta = np.abs(dom_a.astype(np.int16) - dom_b.astype(np.int16))
            delta = np.minimum(delta, 6 - delta)
            dominant_penalty = delta / 3.0
            neighbor_penalty = np.logical_and(delta == 0, np.logical_xor(nei_a, nei_b)).astype(float)
            pixel_penalty = (dominant_penalty + neighbor_penalty) / 2.0
            distances.append(float(np.mean(pixel_penalty)))

        return min(distances)
