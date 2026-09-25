#!/usr/bin/env python
"""Stage A: scan the age-noise scale s and the age coordinate at fixed K.

For each (coord, s) the XD model is fitted by 5-fold cross-validation on the
no-cut sample ``base_agefin``. Per-star held-out log densities are stored so that
configurations can be compared with *paired* standard errors. Scores are
converted to the density in (age [Gyr], [Fe/H]) so linear and log fits compare
directly: ln p(age, feh) = ln p(ln age, feh) - ln age.

Usage
-----
  python scripts/xd_noise_scan.py run               # all jobs, parallel, resumable
  python scripts/xd_noise_scan.py summary           # table + diagnostics
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.config import DEFAULT_CACHE, REPO, Cuts
from eos_figures.stats import hist2d
from eos_figures.xd import GaussianMixture, fit_xd
from eos_figures.xd_age_feh import convolved_counts, load_age_feh

OUT = REPO / "products" / "xd_noise_scan"
MASK = "base_agefin"
K = 12
N_FOLDS = 5
SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
COORDS = ("lin", "log")
FIT_KW = dict(max_iter=3000, tol=3e-7, window=50, init_method="random")

_CACHE: dict = {}


def data(cache, coord, s):
    key = (coord, s)
    if key not in _CACHE:
        _CACHE[key] = load_age_feh(cache, MASK, coord=coord, scale=s)
    return _CACHE[key]


def folds(n):
    return np.random.default_rng(0).permutation(n) % N_FOLDS


def tag(coord, s):
    return f"{coord}_s{s:.2f}"


def job(cache, coord, s, fold):
    t0 = time.time()
    x, cov, meta = data(cache, coord, s)
    f = folds(len(x))
    tr, va = f != fold, f == fold
    res = fit_xd(x[tr], cov[tr], n_components=K, seed=17 * fold + K, **FIT_KW)
    lp = res.mixture.log_prob(x[va], cov[va]) + meta["log_jacobian"][va]
    # names contain dots (s0.40): build paths explicitly, never with Path.with_suffix
    name = f"{tag(coord, s)}_f{fold}"
    np.savez_compressed(OUT / f"{name}.npz", idx=np.flatnonzero(va), lp=lp)
    out = {"coord": coord, "scale": s, "fold": fold, "k": K, "n_iter": res.n_iter, "converged": res.converged,
           "train_mean_loglike": res.mean_loglike, "val_mean_logp_age_feh": float(lp.mean()),
           "mixture": res.mixture.to_dict(), "seconds": time.time() - t0}
    (OUT / f"{name}.json").write_text(json.dumps(out))
    return name, out


def run(cache, workers, coords=COORDS, scales=SCALES):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(cache, c, s, f) for c in coords for s in scales for f in range(N_FOLDS)
            if not (OUT / f"{tag(c, s)}_f{f}.json").exists()]
    n_all = len(coords) * len(scales) * N_FOLDS
    print(f"{n_all} jobs, {n_all - len(jobs)} done, running {len(jobs)} on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, o = fut.result()
            print(f"[{i}/{len(jobs)}] {name}: {o['n_iter']} it, conv={o['converged']}, {o['seconds']:.0f}s, "
                  f"val lnp(age,feh)={o['val_mean_logp_age_feh']:.5f}", flush=True)


def diagnostics(cache, coord, s, mix):
    """chi2/bin and per-[Fe/H] age widths of fold-0 model vs all data, on the linear plotting grid."""
    c = Cuts()
    x, cov, _ = data(cache, coord, s)
    age = np.exp(x[:, 0]) if coord == "log" else x[:, 0]
    xe = np.linspace(*c.ager, c.nage + 1)
    ye = np.linspace(*c.fehr_age, c.nfeh_age + 1)
    h, _, _ = hist2d(age, x[:, 1], c.ager, c.fehr_age, c.nage, c.nfeh_age)
    xe_m = np.log(np.maximum(xe, 0.05)) if coord == "log" else xe
    hc = convolved_counts(mix, cov, xe_m, ye)
    good = hc > 5
    chi2 = float(((h - hc) ** 2 / hc)[good].sum() / good.sum())
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])

    def sd(col):
        w = col / col.sum()
        mu = (w * xc).sum()
        return float(np.sqrt((w * (xc - mu) ** 2).sum()))

    widths = {}
    for fe in (-0.8, -0.5, -0.2, 0.1):
        j = int(np.argmin(abs(yc - fe)))
        widths[f"{yc[j]:+.2f}"] = (sd(h[:, j]), sd(hc[:, j]))
    return chi2, widths


def summary(cache):
    cfgs = sorted({(j.name.split("_s")[0], float(j.name.split("_s")[1].split("_f")[0]))
                   for j in OUT.glob("*_f0.json")})
    lp_all, info = {}, {}
    for c, s in cfgs:
        files = [OUT / f"{tag(c, s)}_f{f}.npz" for f in range(N_FOLDS)]
        if not all(p.exists() for p in files):
            continue
        n = sum(len(np.load(p)["idx"]) for p in files)
        lp = np.empty(n)
        for p in files:
            d = np.load(p)
            lp[d["idx"]] = d["lp"]
        lp_all[(c, s)] = lp
        js = [json.loads((OUT / f"{tag(c, s)}_f{f}.json").read_text()) for f in range(N_FOLDS)]
        info[(c, s)] = {"converged": all(j["converged"] for j in js), "mix0": GaussianMixture.from_dict(js[0]["mixture"])}
    if not lp_all:
        raise SystemExit("no complete configurations yet")
    best = max(lp_all, key=lambda k: lp_all[k].mean())
    rows = []
    print(f"best: {tag(*best)}   (paired SE of the difference to best)")
    print(f"{'config':12s} {'CV lnp':>9s} {'d vs best':>10s} {'paired SE':>9s} {'conv':>5s} {'chi2/bin':>8s}  age std data/model at [Fe/H]")
    for key in sorted(lp_all):
        d = lp_all[key] - lp_all[best]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        chi2, widths = diagnostics(cache, *key, info[key]["mix0"])
        wtxt = "  ".join(f"{k}:{v[0]:.2f}/{v[1]:.2f}" for k, v in widths.items())
        print(f"{tag(*key):12s} {lp_all[key].mean():9.5f} {d.mean():+10.5f} {se:9.5f} {str(info[key]['converged']):>5s} {chi2:8.2f}  {wtxt}")
        rows.append({"coord": key[0], "scale": key[1], "cv_mean": float(lp_all[key].mean()), "d_vs_best": float(d.mean()),
                     "paired_se": se, "converged": info[key]["converged"], "chi2_per_bin_fold0": chi2, "age_std_data_model": widths})
    (OUT / "summary.json").write_text(json.dumps({"best": tag(*best), "k": K, "mask": MASK, "rows": rows}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "summary"])
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--coords", nargs="+", default=list(COORDS))
    p.add_argument("--scales", nargs="+", type=float, default=list(SCALES))
    a = p.parse_args()
    run(a.cache, a.workers, a.coords, a.scales) if a.command == "run" else summary(a.cache)


if __name__ == "__main__":
    main()
