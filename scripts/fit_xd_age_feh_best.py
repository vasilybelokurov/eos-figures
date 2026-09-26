#!/usr/bin/env python
"""Final full-sample XD fit of Age-[Fe/H] at the (coord, s, K) chosen by Stage B.

Reads ``products/xd_noise_scan/summary.json`` (from ``xd_noise_scan.py summary``)
unless --coord/--scale/--k are given, fits all ``base_agefin`` stars with
several random starts in parallel, keeps the highest training likelihood, and
writes ``data/xd_age_feh_model_best.json`` and ``figures/eos_age_feh_xd_best.pdf``.

Usage
-----
  python scripts/fit_xd_age_feh_best.py
  python scripts/fit_xd_age_feh_best.py --coord log --scale 0.3 --k 16
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, REPO
from eos_figures.xd import aic, bic, fit_xd, save_model
from eos_figures.xd_age_feh import load_age_feh

MASK = "base_agefin"
FIT_KW = dict(max_iter=3000, tol=3e-7, window=50, init_method="random")
MODEL = REPO / "data" / "xd_age_feh_model_best.json"


def one_fit(cache, coord, scale, k, restart):
    t0 = time.time()
    x, cov, _ = load_age_feh(cache, MASK, coord=coord, scale=scale)
    res = fit_xd(x, cov, n_components=k, seed=5000 + 1000 * restart + k, **FIT_KW)
    return restart, res, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--coord", default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--restarts", type=int, default=3)
    a = p.parse_args()

    summ = json.loads((REPO / "products" / "xd_noise_scan" / "summary.json").read_text())
    coord = a.coord or summ["best_coord"]
    scale = a.scale if a.scale is not None else summ["best_scale"]
    k = a.k or summ["best_k"]
    print(f"fitting coord={coord} s={scale} K={k} with {a.restarts} restarts", flush=True)

    with ProcessPoolExecutor(max_workers=a.restarts) as ex:
        results = list(ex.map(one_fit, *zip(*[(a.cache, coord, scale, k, r) for r in range(a.restarts)])))
    for r, res, dt in results:
        print(f"restart {r}: {res.n_iter} it, conv={res.converged}, {dt:.0f}s, train ll={res.mean_loglike:.5f}")
    _, best, _ = max(results, key=lambda t: t[1].mean_loglike)

    x, _, meta = load_age_feh(a.cache, MASK, coord=coord, scale=scale)
    meta.pop("log_jacobian")
    n = len(x)
    row = next((r for r in summ["rows"] if r["coord"] == coord and r["scale"] == scale and r.get("k") == k), None)
    meta.update({"k": k, "train_mean_loglike": best.mean_loglike, "converged": best.converged, "n_iter": best.n_iter,
                 "bic": bic(best, n), "aic": aic(best, n), "fit_kw": FIT_KW, "restarts": a.restarts,
                 "restart_loglikes": [res.mean_loglike for _, res, _ in results],
                 "selection": "max 5-fold held-out ln p(age,[Fe/H]) over (s, K) grid" if a.k is None else "user choice",
                 "cv_row": row})
    print("wrote", save_model(MODEL, best.mixture, meta))

    from eos_figures.figures import plot_age_feh_xd

    note = (rf"ln(age) fit, $\sigma_{{\rm age}}={scale:g}\times$age_total_error, K={k}, all finite-age stars"
            if coord == "log" else rf"linear age, s={scale:g}, K={k}")
    print(plot_age_feh_xd(model_path=MODEL, output_name="eos_age_feh_xd_best", note=note))


if __name__ == "__main__":
    main()
