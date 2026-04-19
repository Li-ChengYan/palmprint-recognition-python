"""Generic image loading for baseline-style runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from palmprint.data.naming import extract_class_no_from_name


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load one image as a grayscale uint8 array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def load_images_and_labels(img_path: str | Path, img_form: str, dataset_type: str) -> tuple[list[np.ndarray], list[int]]:
    """Load all images for one dataset folder and infer class labels from filenames."""

    root = Path(img_path)
    files = sorted(root.glob(f"*.{img_form}"), key=lambda item: item.name)
    if not files:
        raise ValueError(f"No {img_form} images found under {root}.")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for file_path in files:
        images.append(load_grayscale_image(file_path))
        labels.append(extract_class_no_from_name(file_path.name, dataset_type))

    return images, labels
