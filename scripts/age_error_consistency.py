#!/usr/bin/env python
"""Additive-noise consistency test for AstroNN ages: Var(age_obs) - <sigma^2> per [Fe/H] bin.

Under age_obs = age_true + noise (noise independent of age_true, unselected sample),
Var(age_obs) >= <sigma^2> in every bin. Reported for raw, linear-corrected and
LOWESS-corrected AstroNN ages; errors are propagated through the (local) slope
of each correction. Needs a cache with age_linear_correct and age_lowess_correct.

Usage
-----
  python scripts/age_error_consistency.py --cache products/cache_agecols/eos_agecols.fits
  python scripts/age_error_consistency.py --cache ... --with-age-cut   # apply sigma_model/age < 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.data import load_catalog, make_masks

LINEAR_SLOPE = 0.88723217  # DR17 astroNN: age_linear_correct = (age - 0.23834204) / 0.88723217


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--with-age-cut", action="store_true")
    a = p.parse_args()
    cat = load_catalog(a.cache)
    m = make_masks(cat)
    raw = np.asarray(cat["age"], float)
    lin = np.asarray(cat["age_linear_correct"], float)
    low = np.asarray(cat["age_lowess_correct"], float)
    st, sm = np.asarray(cat["age_total_error"], float), np.asarray(cat["age_model_error"], float)
    feh = np.asarray(cat["fe_h"], float)
    fin = np.isfinite(raw) & np.isfinite(st) & np.isfinite(low) & (raw > 0)

    # LOWESS map is a deterministic function of raw age: take its local slope
    o = np.argsort(raw[fin])
    grid = np.linspace(*np.percentile(raw[fin], [0.5, 99.5]), 60)
    slope = np.interp(raw, grid, np.gradient(np.interp(grid, raw[fin][o], low[fin][o]), grid))
    versions = {
        "raw": (raw, st, sm),
        "linear": (lin, st / LINEAR_SLOPE, sm / LINEAR_SLOPE),
        "lowess": (low, st * slope, sm * slope),
    }
    for name in ("base", "thick", "thin"):
        w = m[name] & fin
        if a.with_age_cut:
            w &= m["base_age"]
        print(f"\n== {name}{' + age cut' if a.with_age_cut else ', no age-error cut'}: N={w.sum()}")
        print(f"{'[Fe/H]':12s} {'N':>6s} | " + " | ".join(f"{v}: std, rms_tot, V-<tot2>, V-<mod2>" for v in versions))
        for lo in np.arange(-1.0, 0.41, 0.2):
            s = w & (feh >= lo) & (feh < lo + 0.2)
            if s.sum() < 200:
                continue
            cells = []
            for x, et, em in versions.values():
                v = x[s].var()
                cells.append(f"{np.sqrt(v):5.2f} {np.sqrt(np.mean(et[s]**2)):5.2f} {v - np.mean(et[s]**2):+6.2f} {v - np.mean(em[s]**2):+6.2f}")
            print(f"{lo:+.1f}..{lo + 0.2:+.1f} {s.sum():7d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
