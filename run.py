"""Direct script entrypoint for local execution."""

from __future__ import annotations

import argparse


def _run_baseline(args: argparse.Namespace):
    from baseline import run_baseline_command

    return run_baseline_command(args)


def _run_benchmark(args: argparse.Namespace):
    from benchmark import run_benchmark_command

    return run_benchmark_command(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the direct script entrypoint."""

    parser = argparse.ArgumentParser(prog="run.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--img-path", required=True)
    baseline.add_argument("--img-form", required=True)
    baseline.add_argument("--dataset-type", default="PolyU_CF")
    baseline.add_argument("--algorithm", default="MTCC")
    baseline.add_argument("--output-file")
    baseline.set_defaults(handler=_run_baseline)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--img-path", required=True)
    benchmark.add_argument("--output-dir", default="artifacts/python_benchmark")
    benchmark.add_argument("--num-classes", type=int, default=100)
    benchmark.add_argument("--images-per-class", type=int, default=10)
    benchmark.add_argument("--algorithms", nargs="+", default=["PC", "DRCC", "MTCC"])
    benchmark.set_defaults(handler=_run_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to one workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
