"""PolyU benchmark workflow orchestration."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from palmprint.algorithms.registry import get_algorithm
from palmprint.core.metrics import compute_dprime, compute_roc_eer
from palmprint.data.loader import load_grayscale_image
from palmprint.data.polyu import SelectionRecord, select_polyu_subset


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the deterministic benchmark flow."""

    img_path: Path
    subset_codes: tuple[str, ...] = ("F",)
    num_classes: int = 100
    images_per_class: int = 10
    algorithms: tuple[str, ...] = ("PC", "DRCC", "MTCC")
    output_file: Path | None = None
    roc_output_file: Path | None = None


@dataclass(frozen=True)
class AlgorithmSummary:
    """Per-algorithm benchmark summary."""

    algorithm: str
    eer: float
    d_prime: float
    extract_ms: float
    match_ms: float
    far: np.ndarray
    gar: np.ndarray
    num_images: int
    num_genuine: int
    num_imposter: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Output for one benchmark run."""

    config: BenchmarkConfig
    selection: tuple[SelectionRecord, ...] = ()
    summaries: tuple[AlgorithmSummary, ...] = ()
    markdown: str = ""

_METADATA = {
    "PC": ("[1]", 2003, "PalmCode", "Gabor filtering", "2 x 32 x 32, B"),
    "FC": ("[3]", 2004, "FusionCode", "Gabor filtering at four directions, fusion strategy", "2 x 32 x 32, B"),
    "CC": ("[2]", 2004, "CompCode", "Competitive coding in six directions", "1 x 32 x 32, I"),
    "FastCC": ("[8]", 2015, "Fast-CC", "Two-direction CompCode variant", "1 x 32 x 32, I"),
    "OC": ("[4]", 2005, "OrdinalCode", "Gaussian filtering in three groups", "3 x 32 x 32, B"),
    "RLOC": ("[5]", 2008, "RLOC", "Radon competitive coding with pixel-to-area matching", "1 x 32 x 32, I"),
    "FastRLOC": ("[8]", 2015, "Fast-RLOC", "RLOC with one-to-one matching", "1 x 32 x 32, I*"),
    "BOCV": ("[6]", 2009, "BOCV", "Six-direction binary orientation coding", "6 x 32 x 32, B"),
    "EBOCV": ("[7]", 2012, "E-BOCV", "BOCV with fragile-bit masks", "12 x 32 x 32, B"),
    "DOC": ("[10]", 2016, "DOC", "Top-2 competitive codes", "2 x 32 x 32, I"),
    "DRCC": ("[11]", 2016, "DRCC", "Competitive code and neighbor ordinal feature", "1 x 32 x 32, I"),
    "EDM": ("[13]", 2020, "EDM", "Joint best-impact downsampling", "6 x 32 x 32, B"),
    "MTCC": ("[15]", 2023, "MTCC", "Multi-order Gabor features", "12 x 32 x 32, B"),
    "DoN": ("[9]", 2016, "DoN", "3D descriptor recovered from one 2D palmprint", "3 x 128 x 128, I"),
}

_LINE_STYLES = ("-", "--", "-.", ":")
_MARKERS = (None, "o", "s", "^", "D", "v", "P", "X", "*", "<", ">")


def run_polyu_benchmark(
    img_path: str | Path,
    subset_codes: tuple[str, ...] | list[str] | str = ("F",),
    num_classes: int = 100,
    images_per_class: int = 10,
    algorithms: tuple[str, ...] | list[str] = ("PC", "DRCC", "MTCC"),
    output_file: str | Path | None = None,
    roc_output_file: str | Path | None = None,
) -> BenchmarkResult:
    """Run the deterministic PolyU benchmark for one or more algorithms."""

    config = BenchmarkConfig(
        img_path=Path(img_path),
        subset_codes=tuple(subset_codes) if not isinstance(subset_codes, str) else tuple(subset_codes),
        num_classes=num_classes,
        images_per_class=images_per_class,
        algorithms=tuple(algorithms),
        output_file=Path(output_file) if output_file is not None else None,
        roc_output_file=Path(roc_output_file) if roc_output_file is not None else None,
    )

    selection = tuple(
        select_polyu_subset(
            img_path=config.img_path,
            subset_codes=config.subset_codes,
            num_classes=config.num_classes,
            images_per_class=config.images_per_class,
        )
    )
    images = [load_grayscale_image(record.path) for record in selection]
    labels = [record.class_no for record in selection]

    summaries: list[AlgorithmSummary] = []
    for algorithm_name in config.algorithms:
        algorithm = get_algorithm(algorithm_name)

        features = []
        extract_times = []
        for image in images:
            start = perf_counter()
            features.append(algorithm.extract(image))
            extract_times.append(perf_counter() - start)

        genuine = []
        imposter = []
        match_time_sum = 0.0
        for left_index in range(len(features)):
            for right_index in range(left_index + 1, len(features)):
                start = perf_counter()
                score = float(algorithm.match(features[left_index], features[right_index]))
                match_time_sum += perf_counter() - start
                if labels[left_index] == labels[right_index]:
                    genuine.append(score)
                else:
                    imposter.append(score)

        genuine_arr = np.asarray(genuine, dtype=float)
        imposter_arr = np.asarray(imposter, dtype=float)
        far, gar, eer = compute_roc_eer(genuine_arr, imposter_arr)
        d_prime = compute_dprime(genuine_arr, imposter_arr)
        num_pairs = len(features) * (len(features) - 1) // 2

        summaries.append(
            AlgorithmSummary(
                algorithm=algorithm_name,
                eer=eer,
                d_prime=d_prime,
                extract_ms=float(np.mean(extract_times) * 1000.0),
                match_ms=float((match_time_sum / num_pairs) * 1000.0),
                far=far,
                gar=gar,
                num_images=len(features),
                num_genuine=genuine_arr.size,
                num_imposter=imposter_arr.size,
            )
        )

    return BenchmarkResult(config=config, selection=selection, summaries=tuple(summaries))


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render a markdown table for benchmark summaries."""

    header = [
        "| Ref. | Year | Usual name | Code name | Method summary | Template size / format | EER (%) | d-prime | Extract (ms/img) | Match (ms/pair) |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    rows = []
    for summary in result.summaries:
        ref, year, usual_name, method_summary, template_format = _METADATA[summary.algorithm]
        rows.append(
            "| "
            f"{ref} | {year} | {usual_name} | `{summary.algorithm}` | {method_summary} | {template_format} | "
            f"{summary.eer:.3f} | {summary.d_prime:.3f} | {summary.extract_ms:.3f} | {summary.match_ms:.3f} |"
        )

    return "\n".join([*header, *rows])


def save_benchmark_npz(path: str | Path, result: BenchmarkResult) -> None:
    """Save benchmark summaries into one NPZ bundle."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        algorithms=np.asarray([summary.algorithm for summary in result.summaries], dtype=object),
        eer=np.asarray([summary.eer for summary in result.summaries], dtype=float),
        d_prime=np.asarray([summary.d_prime for summary in result.summaries], dtype=float),
        extract_ms=np.asarray([summary.extract_ms for summary in result.summaries], dtype=float),
        match_ms=np.asarray([summary.match_ms for summary in result.summaries], dtype=float),
        far=np.asarray([summary.far for summary in result.summaries], dtype=object),
        gar=np.asarray([summary.gar for summary in result.summaries], dtype=object),
    )


def save_benchmark_markdown(path: str | Path, markdown: str) -> None:
    """Write benchmark markdown to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")


def _build_curve_style(index: int) -> dict[str, object]:
    marker = _MARKERS[index % len(_MARKERS)]
    style: dict[str, object] = {
        "linestyle": _LINE_STYLES[index % len(_LINE_STYLES)],
        "marker": marker,
        "linewidth": 1.8,
    }
    if marker is not None:
        style["markersize"] = 4.0
    return style


def _compute_gar_axis_limits(result: BenchmarkResult) -> tuple[float, float]:
    if not result.summaries:
        return 0.99, 1.0

    all_gar = np.concatenate([summary.gar for summary in result.summaries]) / 100.0
    observed_lower = float(np.min(all_gar))
    worst_eer = max(summary.eer for summary in result.summaries) / 100.0

    zoom_floor = 1.0 - max(0.004, worst_eer * 3.0)
    lower = max(observed_lower - 0.001, zoom_floor)
    lower = min(lower, 0.999)
    lower = max(lower, 0.0)
    lower = np.floor(lower * 1000.0) / 1000.0

    return float(lower), 1.0


def plot_roc_overview(result: BenchmarkResult, output_path: str | Path, title: str = "PolyU ROC Overview") -> None:
    """Plot all benchmark ROC curves into one image."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for index, summary in enumerate(result.summaries):
        far_ratio = summary.far / 100.0
        gar_ratio = summary.gar / 100.0
        style = _build_curve_style(index)
        if style["marker"] is not None:
            style["markevery"] = max(1, len(far_ratio) // 12)
        ax.semilogx(far_ratio, gar_ratio, label=summary.algorithm, **style)

    ax.set_title(title)
    ax.set_xlabel("FAR")
    ax.set_ylabel("GAR")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(*_compute_gar_axis_limits(result))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)


def write_benchmark_artifacts(result: BenchmarkResult, output_dir: str | Path) -> None:
    """Write benchmark markdown, npz, and ROC overview artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    markdown = render_benchmark_markdown(result)
    save_benchmark_markdown(target / "benchmark_results.md", markdown)
    save_benchmark_npz(target / "benchmark_results.npz", result)
    plot_roc_overview(result, target / "roc_overview.png")


def print_benchmark_summary(result: BenchmarkResult) -> None:
    """Print per-algorithm summary lines."""

    for summary in result.summaries:
        print(
            f"{summary.algorithm}: "
            f"EER={summary.eer:.3f}% "
            f"d-prime={summary.d_prime:.3f} "
            f"extract_ms={summary.extract_ms:.3f} "
            f"match_ms={summary.match_ms:.3f}"
        )


def run_benchmark_command(args: Namespace) -> BenchmarkResult:
    """Execute the benchmark command from parsed arguments."""

    result = run_polyu_benchmark(
        img_path=args.img_path,
        subset_codes=("F",),
        num_classes=args.num_classes,
        images_per_class=args.images_per_class,
        algorithms=args.algorithms,
    )
    write_benchmark_artifacts(result, args.output_dir)
    print_benchmark_summary(result)
    return result
