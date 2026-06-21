#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, DEFAULT_OUTDIR
from eos_figures.figures import FIGURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Eos paper figures from the matched APOGEE DR17 cache.")
    parser.add_argument("figure", choices=sorted([*FIGURES, "all"]), help="Figure name to generate")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Matched catalogue cache")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for PDFs")
    args = parser.parse_args()
    names = sorted(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        path = FIGURES[name](args.cache, args.outdir)
        print(path)


if __name__ == "__main__":
    main()

