#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, DEFAULT_LIST_DIR
from eos_figures.data import gc_member_mask, satellite_out_mask


DEFAULT_GC_MEMBERS = Path(
    "/Users/vasilybelokurov/IoA Dropbox/Dr V.A. Belokurov/data/catalogues/gc_members_gaia_vasiliev.fits"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add satellite and GC-member masks to the compact Eos cache.")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Input cache to update")
    parser.add_argument("--output", default=DEFAULT_CACHE, help="Output cache path")
    parser.add_argument("--satellite-list-dir", default=DEFAULT_LIST_DIR, help="Directory containing CompiledSatCatalogv2_gabriel.csv")
    parser.add_argument("--gc-members", default=DEFAULT_GC_MEMBERS, help="Vasiliev GC member FITS catalogue")
    parser.add_argument("--gc-match-tol", type=float, default=1.0, help="GC member sky-match tolerance in arcsec")
    parser.add_argument("--gc-min-prob", type=float, default=0.5, help="Minimum Vasiliev membership probability")
    args = parser.parse_args()

    table = Table.read(args.cache)
    satellite_out = satellite_out_mask(np.asarray(table["ra"]), np.asarray(table["dec"]), list_dir=args.satellite_list_dir)
    gc_member = gc_member_mask(
        np.asarray(table["ra"]),
        np.asarray(table["dec"]),
        args.gc_members,
        match_tol_arcsec=args.gc_match_tol,
        min_prob=args.gc_min_prob,
    )
    table["satellite_out"] = satellite_out
    table["gc_member"] = gc_member
    table.meta["SATMASK"] = "CompiledSatCatalogv2_gabriel.csv, radius=1 deg"
    table.meta["GCMEM"] = f"{Path(args.gc_members).name}, tol={args.gc_match_tol} arcsec, prob>={args.gc_min_prob}"
    table.write(args.output, overwrite=True)
    print(args.output)
    print(f"satellite_out=False: {(~satellite_out).sum()}")
    print(f"gc_member=True: {gc_member.sum()}")
    print(f"gc_member surviving satellite mask: {(gc_member & satellite_out).sum()}")


if __name__ == "__main__":
    main()
