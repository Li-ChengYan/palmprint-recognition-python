"""E-BOCV implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import convolve2d

from palmprint.core.kernels import generate_np_gabor_bank
from palmprint.core.matching import iter_shifted_overlap


@dataclass(frozen=True)
class EBOCVFeature:
    """BOCV bits plus fragile-bit masks."""

    bocv: np.ndarray
    mask: np.ndarray


class EBOCVAlgorithm:
    """Enhanced BOCV with fragile-bit masks."""

    name = "EBOCV"

    def __init__(self) -> None:
        self.kernels = generate_np_gabor_bank()

    def extract(self, image: np.ndarray) -> EBOCVFeature:
        image_arr = np.asarray(image, dtype=float)
        bocv_maps = []
        mask_maps = []

        for kernel in self.kernels:
            response = convolve2d(image_arr, kernel, mode="same", boundary="fill", fillvalue=0)
            bocv_maps.append(response[::4, ::4] > 0)

            abs_response = np.abs(response)
            sorted_values = np.sort(abs_response, axis=None)
            threshold_index = max(0, math.floor(sorted_values.size * 0.08) - 1)
            threshold = sorted_values[threshold_index]
            mask_maps.append(abs_response[::4, ::4] >= threshold)

        return EBOCVFeature(
            bocv=np.stack(bocv_maps, axis=0),
            mask=np.stack(mask_maps, axis=0),
        )

    def match(self, feature_a: EBOCVFeature, feature_b: EBOCVFeature) -> float:
        if feature_a.bocv.shape != feature_b.bocv.shape or feature_a.mask.shape != feature_b.mask.shape:
            raise ValueError("EBOCV feature shape mismatch")

        hdm_scores = []
        fpd_scores = []
        for channel in range(feature_a.bocv.shape[0]):
            channel_hdm = []
            channel_fpd = []

            data_windows = iter_shifted_overlap(feature_a.bocv[channel], feature_b.bocv[channel], shift=4)
            mask_windows = iter_shifted_overlap(feature_a.mask[channel], feature_b.mask[channel], shift=4)
            for (data_a, data_b), (mask_a, mask_b) in zip(data_windows, mask_windows, strict=True):
                overlap = np.logical_and(mask_a, mask_b)
                overlap_count = int(np.sum(overlap))
                if overlap_count == 0:
                    continue

                channel_hdm.append(float(np.sum(np.logical_and(np.logical_xor(data_a, data_b), overlap)) / overlap_count))
                channel_fpd.append(float(np.sum(np.logical_xor(mask_a, mask_b)) / overlap_count))

            if not channel_hdm or not channel_fpd:
                raise ValueError("EBOCV masks have zero valid overlap for at least one channel")

            hdm_scores.append(min(channel_hdm))
            fpd_scores.append(min(channel_fpd))

        hdm = float(np.mean(hdm_scores))
        fpd = float(np.mean(fpd_scores))
        return (0.45 * hdm) + (0.55 * fpd)
