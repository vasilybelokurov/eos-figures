#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, DEFAULT_OUTDIR
from eos_figures.figures import plot_vphi


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sorted and/or unsorted Eos Vtan figure variants.")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Matched catalogue cache")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for PDFs")
    parser.add_argument(
        "--variant",
        choices=("both", "unsorted", "sorted"),
        default="both",
        help="Which point draw-order variant to generate",
    )
    args = parser.parse_args()

    variants = []
    if args.variant in ("both", "unsorted"):
        variants.append((False, "eos_vphi_unsorted"))
    if args.variant in ("both", "sorted"):
        variants.append((True, "eos_vphi_sorted"))

    for sort_by_color, output_name in variants:
        path = plot_vphi(args.cache, args.outdir, sort_by_color=sort_by_color, output_name=output_name)
        print(path)


if __name__ == "__main__":
    main()
