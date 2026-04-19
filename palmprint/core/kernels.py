"""Kernel generation helpers for built-in palmprint algorithms."""

from __future__ import annotations

import math

import numpy as np


def generate_pc_gabor(theta: float) -> np.ndarray:
    """Build the complex Gabor kernel used by PalmCode and FusionCode."""

    u = 0.0916
    sigma = 5.6179
    n = 8
    kernel = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex128)

    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            gaussian = (1.0 / (2.0 * math.pi * sigma**2)) * math.exp(-0.5 * ((x**2 + y**2) / sigma**2))
            sinusoid = np.exp(2j * math.pi * (u * x * math.cos(theta) + u * y * math.sin(theta)))
            kernel[x + n, y + n] = gaussian * sinusoid

    kernel -= np.mean(kernel)
    return kernel


def generate_fc_gabor_bank() -> list[np.ndarray]:
    """Build the four-orientation Gabor bank for FusionCode."""

    return [generate_pc_gabor(k * math.pi / 4.0) for k in range(4)]


def generate_np_gabor(theta: float) -> np.ndarray:
    """Build the real-valued neurophysiology-inspired Gabor kernel."""

    n = 8
    omega = 0.5
    kappa = 2.0
    kernel = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)

    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            dx = x * math.cos(theta) + y * math.sin(theta)
            dy = y * math.cos(theta) - x * math.sin(theta)
            envelope = -omega / (math.sqrt(2.0 * math.pi) * kappa)
            envelope *= math.exp(-(omega**2) * (4.0 * dx**2 + dy**2) / (8.0 * kappa**2))
            carrier = math.cos(omega * dx) - math.exp(-(kappa**2) / 2.0)
            kernel[x + n, y + n] = envelope * carrier

    kernel -= np.mean(kernel)
    return kernel


def generate_np_gabor_bank() -> list[np.ndarray]:
    """Build the six-orientation bank shared by CC-family algorithms."""

    return [generate_np_gabor(k * math.pi / 6.0) for k in range(6)]


def _gaussian_oc_single(theta: float) -> np.ndarray:
    width = 25
    height = 7
    m = width // 2
    n = height // 2
    kernel = np.zeros((width, height), dtype=np.float64)

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            first = ((x * math.cos(theta) + y * math.sin(theta)) / width) ** 2
            second = ((-x * math.sin(theta) + y * math.cos(theta)) / height) ** 2
            kernel[x + m, y + n] = math.exp(-(first + second))

    kernel -= np.mean(kernel)
    return kernel


def generate_oc_filter_bank() -> list[np.ndarray]:
    """Build the three ordinal-code Gaussian difference filters."""

    return [_gaussian_oc_single(i * math.pi / 6.0) - _gaussian_oc_single(i * math.pi / 6.0 + math.pi / 2.0) for i in range(3)]


def generate_drcc_filter_bank() -> list[np.ndarray]:
    """Build the six Gabor filters plus the Gaussian prefilter used by DRCC."""

    gaussian = np.zeros((5, 5), dtype=np.float64)
    sigma = 1.0
    n = 2
    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            gaussian[x + n, y + n] = (1.0 / (2.0 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    gaussian /= np.sum(gaussian)
    return [*generate_np_gabor_bank(), gaussian]


def _mfrat(theta: float) -> np.ndarray:
    n = 8
    size = 2 * n
    core = n
    mask = np.zeros((size, size), dtype=np.float64)

    for i in range(1, size + 1):
        if theta <= math.pi / 4.0 or theta >= 3.0 * math.pi / 4.0:
            j = int(core - math.tan(theta) * (i - core))
            rows = range(max(1, j - 1), min(size, j + 2) + 1)
            for row in rows:
                mask[row - 1, i - 1] = 1.0
        else:
            j = int(core - math.tan(math.pi / 2.0 - theta) * (i - core))
            cols = range(max(1, j - 1), min(size, j + 2) + 1)
            for col in cols:
                mask[i - 1, col - 1] = 1.0

    return mask


def generate_rloc_filter_bank() -> list[np.ndarray]:
    """Build the six modified finite Radon transform masks."""

    return [_mfrat(k * math.pi / 6.0) for k in range(6)]
