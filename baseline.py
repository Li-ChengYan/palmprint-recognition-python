"""Baseline workflow orchestration."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from palmprint.algorithms.registry import get_algorithm
from palmprint.data.loader import load_images_and_labels


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline score generation."""

    img_path: Path
    img_form: str
    dataset_type: str
    algorithm_name: str
    output_file: Path | None = None


@dataclass(frozen=True)
class BaselineResult:
    """Output for baseline score generation."""

    config: BaselineConfig
    genuine: np.ndarray
    imposter: np.ndarray


def compute_genuine_imposter(
    img_path: str | Path,
    img_form: str,
    dataset_type: str,
    algorithm_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and split pairwise scores into genuine and imposter sets."""

    algorithm = get_algorithm(algorithm_name)
    images, labels = load_images_and_labels(img_path, img_form, dataset_type)
    features = [algorithm.extract(image) for image in images]

    genuine: list[float] = []
    imposter: list[float] = []
    for left_index in range(len(features)):
        for right_index in range(left_index + 1, len(features)):
            score = float(algorithm.match(features[left_index], features[right_index]))
            if labels[left_index] == labels[right_index]:
                genuine.append(score)
            else:
                imposter.append(score)

    return np.asarray(genuine, dtype=float), np.asarray(imposter, dtype=float)


def run_baseline(
    img_path: str | Path,
    img_form: str,
    dataset_type: str,
    algorithm_name: str,
    output_file: str | Path | None = None,
) -> BaselineResult:
    """Run the baseline flow and optionally save scores."""

    config = BaselineConfig(
        img_path=Path(img_path),
        img_form=img_form,
        dataset_type=dataset_type,
        algorithm_name=algorithm_name,
        output_file=Path(output_file) if output_file is not None else None,
    )
    genuine, imposter = compute_genuine_imposter(
        img_path=config.img_path,
        img_form=config.img_form,
        dataset_type=config.dataset_type,
        algorithm_name=config.algorithm_name,
    )

    if config.output_file is not None:
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(config.output_file, genuine=genuine, imposter=imposter)

    return BaselineResult(config=config, genuine=genuine, imposter=imposter)


def run_baseline_command(args: Namespace) -> BaselineResult:
    """Execute the baseline command from parsed arguments."""

    return run_baseline(
        img_path=args.img_path,
        img_form=args.img_form,
        dataset_type=args.dataset_type,
        algorithm_name=args.algorithm,
        output_file=args.output_file,
    )
