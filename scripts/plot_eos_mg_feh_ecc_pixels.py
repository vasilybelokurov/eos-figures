#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, DEFAULT_OUTDIR
from eos_figures.figures import plot_mg_feh_ecc_pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pixelated mean-property maps for eos_mg_feh_ecc.")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Matched catalogue cache")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for PDFs")
    parser.add_argument("--xbins", type=int, default=71, help="Number of [Fe/H] bins")
    parser.add_argument("--ybins", type=int, default=71, help="Number of abundance bins")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum stars per pixel to display")
    parser.add_argument("--name", default="eos_mg_feh_ecc_pixels", help="Output PDF stem")
    args = parser.parse_args()

    path = plot_mg_feh_ecc_pixels(
        args.cache,
        args.outdir,
        bins=(args.xbins, args.ybins),
        min_count=args.min_count,
        output_name=args.name,
    )
    print(path)


if __name__ == "__main__":
    main()
