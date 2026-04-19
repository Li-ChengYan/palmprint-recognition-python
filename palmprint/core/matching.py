"""Shared shifted-overlap matching helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _crop_for_shift(array: np.ndarray, row_shift: int, col_shift: int, take_left: bool) -> np.ndarray:
    rows, cols = array.shape[-2:]

    if take_left:
        row_start = (abs(row_shift) - row_shift) // 2
        row_end = rows - (abs(row_shift) + row_shift) // 2
        col_start = (abs(col_shift) - col_shift) // 2
        col_end = cols - (abs(col_shift) + col_shift) // 2
    else:
        row_start = (abs(row_shift) + row_shift) // 2
        row_end = rows - (abs(row_shift) - row_shift) // 2
        col_start = (abs(col_shift) + col_shift) // 2
        col_end = cols - (abs(col_shift) - col_shift) // 2

    return array[..., row_start:row_end, col_start:col_end]


def iter_shifted_overlap(left: np.ndarray, right: np.ndarray, shift: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield all overlapping windows under symmetric shift cropping."""

    left_arr = np.asarray(left)
    right_arr = np.asarray(right)

    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right must have the same shape")

    for row_shift in range(-shift, shift + 1):
        for col_shift in range(-shift, shift + 1):
            yield (
                _crop_for_shift(left_arr, row_shift, col_shift, True),
                _crop_for_shift(right_arr, row_shift, col_shift, False),
            )


def minimum_shifted_hamming(left: np.ndarray, right: np.ndarray, shift: int) -> float:
    """Return the minimum normalized Hamming distance over all shifts."""

    distances: list[float] = []
    for left_window, right_window in iter_shifted_overlap(left, right, shift):
        distances.append(float(np.mean(np.logical_xor(left_window, right_window))))
    return min(distances)


def mean_per_channel_shifted_hamming(left: np.ndarray, right: np.ndarray, shift: int) -> float:
    """Average the best shifted Hamming distance over the leading channel axis."""

    left_arr = np.asarray(left)
    right_arr = np.asarray(right)

    if left_arr.shape != right_arr.shape:
        raise ValueError("left and right must have the same shape")
    if left_arr.ndim != 3:
        raise ValueError("left and right must have shape (channels, rows, cols)")

    per_channel = [minimum_shifted_hamming(left_arr[idx], right_arr[idx], shift) for idx in range(left_arr.shape[0])]
    return float(np.mean(per_channel))


def minimum_shifted_angular_distance(left: np.ndarray, right: np.ndarray, shift: int, period: int = 6) -> float:
    """Return the minimum shifted angular distance for orientation codes."""

    distances: list[float] = []
    for left_window, right_window in iter_shifted_overlap(left, right, shift):
        delta = np.abs(left_window.astype(np.int16) - right_window.astype(np.int16))
        wrapped = np.minimum(delta, period - delta)
        distances.append(float(np.mean(wrapped / 3.0)))
    return min(distances)


def maximum_shifted_similarity(left: np.ndarray, right: np.ndarray, shift: int) -> float:
    """Return the maximum equality ratio over all shifts."""

    similarities: list[float] = []
    for left_window, right_window in iter_shifted_overlap(left, right, shift):
        similarities.append(float(np.mean(left_window == right_window)))
    return max(similarities)
