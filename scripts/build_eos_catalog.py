#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eos_figures.config import DEFAULT_APOGEE, DEFAULT_ASTRONN, DEFAULT_CACHE
from eos_figures.data import build_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the matched APOGEE DR17 + AstroNN Eos catalogue cache.")
    parser.add_argument("--apogee", default=DEFAULT_APOGEE, help="Path to allStarLite-dr17-synspec_rev1.fits")
    parser.add_argument("--astronn", default=DEFAULT_ASTRONN, help="Path to apogee_astroNN-DR17.fits")
    parser.add_argument("--output", default=DEFAULT_CACHE, help="Output compact FITS cache")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache")
    args = parser.parse_args()
    path = build_cache(args.apogee, args.astronn, args.output, overwrite=args.overwrite)
    print(path)


if __name__ == "__main__":
    main()
