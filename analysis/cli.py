from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import InfluenceAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/results_run1",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyzer = InfluenceAnalyzer.from_directory(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir) / "figures"
    saved = analyzer.write_default_artifacts(
        output_dir=output_dir,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        top_pairs=args.top_pairs,
    )
    for path in saved:
        print(f"  saved {path}")
    if args.print_report:
        print((output_dir / "report.txt").read_text())
    else:
        print(f"\nAll figures saved in {output_dir.resolve()}/")
    return 0
