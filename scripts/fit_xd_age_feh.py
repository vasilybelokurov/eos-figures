#!/usr/bin/env python
"""Extreme-deconvolution Gaussian mixture of the Age-[Fe/H] distribution.

Sample: ``base_age`` stars (base sample with finite, positive AstroNN age and
age_model_error/age < 0.2). Data vector (age [Gyr], [Fe/H] [dex]); per-star noise
covariance diag(age_total_error^2, fe_h_err^2).

Subcommands (all jobs run in parallel, one JSON per job, existing files skipped):

  init-test  K=8 on all stars, k-means vs random starts, 3 seeds each
  cv         K-fold held-out log density for each K (model selection)
  full       fits of all stars for each K (BIC/AIC and the final model)
  select     aggregate cv + full, choose K, write the model JSON to data/

Usage
-----
  python scripts/fit_xd_age_feh.py init-test
  python scripts/fit_xd_age_feh.py cv --kmax 20
  python scripts/fit_xd_age_feh.py full --kmax 20
  python scripts/fit_xd_age_feh.py select
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

from eos_figures.config import DEFAULT_CACHE, REPO
from eos_figures.xd import aic, bic, fit_xd, save_model
from eos_figures.xd_age_feh import DEFAULT_MODEL, load_age_feh

OUT = REPO / "products" / "xd_age_feh"
N_FOLDS = 5
FOLD_SEED = 0
# Per-star relative tolerance: 3e-7 * |ll|~2.25 * n~1.6e5 ~ 0.1 in total log-likelihood
# per iteration, averaged over WINDOW iterations.
FIT_KW = dict(max_iter=3000, tol=3e-7, window=50)

_DATA: dict = {}


def _data(cache):
    if "x" not in _DATA:
        _DATA["x"], _DATA["cov"], _DATA["meta"] = load_age_feh(cache)
    return _DATA["x"], _DATA["cov"]


def _folds(n):
    return np.random.default_rng(FOLD_SEED).permutation(n) % N_FOLDS


def _result_dict(res, extra):
    return {
        **extra,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "train_mean_loglike": res.mean_loglike,
        "history_tail": res.history[-5:],
        "mixture": res.mixture.to_dict(),
    }


def job(kind, cache, k, fold, restart, init_method, path):
    t0 = time.time()
    x, cov = _data(cache)
    seed = 1000 * restart + 17 * fold + k
    if kind == "cv":
        f = _folds(len(x))
        tr, va = f != fold, f == fold
        res = fit_xd(x[tr], cov[tr], n_components=k, seed=seed, init_method=init_method, **FIT_KW)
        lp = res.mixture.log_prob(x[va], cov[va])
        out = _result_dict(res, {"k": k, "fold": fold, "restart": restart, "n_train": int(tr.sum()),
                                 "n_val": int(va.sum()), "val_mean_logp": float(lp.mean()),
                                 "val_sum_logp": float(lp.sum()), "val_sumsq_logp": float((lp**2).sum())})
    else:
        res = fit_xd(x, cov, n_components=k, seed=seed, init_method=init_method, **FIT_KW)
        n = len(x)
        out = _result_dict(res, {"k": k, "restart": restart, "init_method": init_method, "n": n,
                                 "bic": bic(res, n), "aic": aic(res, n), "n_params": res.mixture.n_params})
    out["seconds"] = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path, out


def run_jobs(jobs, workers):
    todo = [j for j in jobs if not j[-1].exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} on {workers} workers", flush=True)
    # longest (largest K) first for better load balancing
    todo.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, *j) for j in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            path, out = fut.result()
            print(f"[{i}/{len(todo)}] {path.name}: {out['n_iter']} it, conv={out['converged']}, "
                  f"{out['seconds']:.0f}s, train ll={out['train_mean_loglike']:.5f}", flush=True)


def select(kmax_hint=None):
    cv = [json.loads(p.read_text()) for p in sorted((OUT / "cv").glob("*.json"))]
    full = [json.loads(p.read_text()) for p in sorted((OUT / "full").glob("*.json"))]
    ks = sorted({r["k"] for r in cv})
    table = {}
    for k in ks:
        rows = [r for r in cv if r["k"] == k]
        # per fold keep the restart with the best *training* likelihood (no peeking at validation)
        best = {}
        for r in rows:
            if r["fold"] not in best or r["train_mean_loglike"] > best[r["fold"]]["train_mean_loglike"]:
                best[r["fold"]] = r
        if len(best) < N_FOLDS:
            continue
        s = sum(r["val_sum_logp"] for r in best.values())
        ss = sum(r["val_sumsq_logp"] for r in best.values())
        n = sum(r["n_val"] for r in best.values())
        mean = s / n
        se = np.sqrt(max(ss / n - mean**2, 0) / (n - 1))
        fr = [r for r in full if r["k"] == k]
        fb = max(fr, key=lambda r: r["train_mean_loglike"]) if fr else None
        table[k] = {
            "cv_mean": mean, "cv_se": float(se),
            "cv_folds": [best[f]["val_mean_logp"] for f in sorted(best)],
            "cv_all_converged": all(r["converged"] for r in best.values()),
            "full_loglike": fb["train_mean_loglike"] if fb else None,
            "bic": fb["bic"] if fb else None, "aic": fb["aic"] if fb else None,
            "full_converged": fb["converged"] if fb else None,
        }
    k_best = max(table, key=lambda k: table[k]["cv_mean"])
    thr = table[k_best]["cv_mean"] - table[k_best]["cv_se"]
    k_1se = min(k for k in table if table[k]["cv_mean"] >= thr)
    have_bic = [k for k in table if table[k]["bic"] is not None]
    k_bic = min(have_bic, key=lambda k: table[k]["bic"]) if have_bic else None
    k_aic = min(have_bic, key=lambda k: table[k]["aic"]) if have_bic else None

    print(f"{'K':>3} {'CV mean logp':>13} {'SE':>8} {'dCV vs best':>11} {'conv':>5} {'BIC':>12} {'AIC':>12}")
    for k in sorted(table):
        t = table[k]
        b = f"{t['bic']:12.1f}" if t["bic"] is not None else f"{'-':>12}"
        a = f"{t['aic']:12.1f}" if t["aic"] is not None else f"{'-':>12}"
        print(f"{k:3d} {t['cv_mean']:13.5f} {t['cv_se']:8.5f} {t['cv_mean'] - table[k_best]['cv_mean']:11.5f} "
              f"{str(t['cv_all_converged']):>5} {b} {a}")
    print(f"CV max: K={k_best}; CV 1-SE rule: K={k_1se}; BIC min: K={k_bic}; AIC min: K={k_aic}")
    summary = {"table": table, "k_cv_max": k_best, "k_cv_1se": k_1se, "k_bic": k_bic, "k_aic": k_aic,
               "n_folds": N_FOLDS, "fit_kw": FIT_KW}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init-test", "cv", "full", "select"])
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--kmin", type=int, default=1)
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--init", default="kmeans", choices=["kmeans", "random"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--k", type=int, default=None, help="select: override the chosen K")
    a = p.parse_args()
    ks = range(a.kmin, a.kmax + 1)

    if a.command == "init-test":
        jobs = [("full", a.cache, 8, 0, r, m, OUT / "init_test" / f"K8_{m}_r{r}.json")
                for m in ("kmeans", "random") for r in range(3)]
        run_jobs(jobs, a.workers)
        for m in ("kmeans", "random"):
            for r in range(3):
                d = json.loads((OUT / "init_test" / f"K8_{m}_r{r}.json").read_text())
                print(f"{m:7s} r{r}: {d['n_iter']:5d} it  conv={d['converged']}  {d['seconds']:5.0f}s  ll={d['train_mean_loglike']:.5f}")
    elif a.command == "cv":
        jobs = [("cv", a.cache, k, f, r, a.init, OUT / "cv" / f"K{k:02d}_f{f}_r{r}.json")
                for k in ks for f in range(N_FOLDS) for r in range(a.restarts)]
        run_jobs(jobs, a.workers)
    elif a.command == "full":
        jobs = [("full", a.cache, k, 0, r, a.init, OUT / "full" / f"K{k:02d}_r{r}.json")
                for k in ks for r in range(a.restarts)]
        run_jobs(jobs, a.workers)
    else:
        s = select()
        k = a.k or s["k_cv_max"]
        fr = [json.loads(q.read_text()) for q in (OUT / "full").glob(f"K{k:02d}_r*.json")]
        if not fr:
            raise SystemExit(f"no full-sample fit for K={k}; run: fit_xd_age_feh.py full --kmin {k} --kmax {k}")
        best = max(fr, key=lambda r: r["train_mean_loglike"])
        from eos_figures.xd import GaussianMixture

        _, _, meta = load_age_feh(a.cache)
        meta.pop("log_jacobian", None)
        meta.update({"k": k, "selection": "max K-fold held-out log density" if a.k is None else "user override",
                     "k_cv_max": s["k_cv_max"], "k_cv_1se": s["k_cv_1se"], "k_bic": s["k_bic"], "k_aic": s["k_aic"],
                     "train_mean_loglike": best["train_mean_loglike"], "converged": best["converged"],
                     "n_iter": best["n_iter"], "bic": best["bic"], "aic": best["aic"], "fit_kw": FIT_KW,
                     "cv_table": s["table"]})
        path = save_model(DEFAULT_MODEL, GaussianMixture.from_dict(best["mixture"]), meta)
        print(f"wrote {path} (K={k})")


if __name__ == "__main__":
    main()
