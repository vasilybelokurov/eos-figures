"""Checks that extreme deconvolution deconvolves (ported from qso_p_color tests/test_xd.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.xd import GaussianMixture, bic, fit_xd, select_n_components


def test_xd_recovers_intrinsic_width():
    rng = np.random.default_rng(0)
    n, truth_var, noise_var = 20000, 0.09, 0.25
    obs = rng.normal(0.0, np.sqrt(truth_var), (n, 1)) + rng.normal(0.0, np.sqrt(noise_var), (n, 1))
    cov = np.full((n, 1, 1), noise_var)
    res = fit_xd(obs, cov, n_components=1, tol=1e-9)
    assert res.converged
    assert res.mixture.covs[0, 0, 0] == pytest.approx(truth_var, rel=0.08)
    naive = fit_xd(obs, None, n_components=1, tol=1e-9)
    assert naive.mixture.covs[0, 0, 0] == pytest.approx(truth_var + noise_var, rel=0.05)


def test_xd_recovers_a_two_component_mixture():
    rng = np.random.default_rng(1)
    truth = GaussianMixture(
        np.array([0.35, 0.65]),
        np.array([[-1.5, 0.5], [1.0, -0.8]]),
        np.stack([np.diag([0.2, 0.1]), np.diag([0.15, 0.3])]),
    )
    n = 30000
    obs = truth.sample(n, rng) + rng.normal(0.0, np.sqrt(0.2), (n, 2))
    res = fit_xd(obs, np.broadcast_to(0.2 * np.eye(2), (n, 2, 2)), n_components=2, seed=3, tol=1e-9)
    o = np.argsort(res.mixture.means[:, 0])
    assert np.allclose(res.mixture.means[o], truth.means, atol=0.05)
    assert np.allclose(res.mixture.weights[o], truth.weights, atol=0.03)
    assert np.allclose(res.mixture.covs[o], truth.covs, atol=0.05)


def test_xd_handles_heteroscedastic_correlated_noise():
    """Per-star anisotropic noise, as for (age, [Fe/H]) with very different errors."""
    rng = np.random.default_rng(2)
    n = 40000
    tv = np.array([[1.0, 0.3], [0.3, 0.5]])
    latent = rng.multivariate_normal([0, 0], tv, n)
    sa = rng.uniform(0.2, 1.5, n)
    sb = rng.uniform(0.05, 0.4, n)
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = sa**2, sb**2
    obs = latent + np.column_stack([sa * rng.normal(size=n), sb * rng.normal(size=n)])
    res = fit_xd(obs, cov, n_components=1, tol=1e-9)
    assert np.allclose(res.mixture.covs[0], tv, atol=0.04)


def test_loglikelihood_increases_monotonically():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(-2, 0.5, (3000, 2)), rng.normal(1, 0.8, (3000, 2))])
    res = fit_xd(x, np.broadcast_to(0.1 * np.eye(2), (6000, 2, 2)), n_components=3, tol=1e-10, max_iter=200)
    assert np.all(np.diff(res.history) > -1e-9)


def test_select_n_components_prefers_the_true_complexity():
    rng = np.random.default_rng(7)
    truth = GaussianMixture(np.array([0.5, 0.5]), np.array([[-2.0], [2.0]]), np.full((2, 1, 1), 0.2))
    x = truth.sample(6000, rng) + rng.normal(0, np.sqrt(0.05), (6000, 1))
    cov = np.full((6000, 1, 1), 0.05)
    best, scores = select_n_components(x, cov, [1, 2, 4], n_folds=3, n_init=1, tol=1e-7)
    assert best in (2, 4)
    assert scores[2]["cv_mean"] > scores[1]["cv_mean"] + 0.2


def test_bic_penalises_extra_components():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, (5000, 2))
    cov = np.broadcast_to(0.1 * np.eye(2), (5000, 2, 2))
    b1 = bic(fit_xd(x, cov, n_components=1, tol=1e-9), 5000)
    b3 = bic(fit_xd(x, cov, n_components=3, tol=1e-9), 5000)
    assert b1 < b3
