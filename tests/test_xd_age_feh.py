"""Grid evaluation of intrinsic and noise-convolved XD models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eos_figures.xd import GaussianMixture
from eos_figures.xd_age_feh import convolved_counts, intrinsic_counts

MIX = GaussianMixture(
    np.array([0.6, 0.4]),
    np.array([[4.0, -0.1], [8.5, -0.5]]),
    np.array([[[2.0, -0.1], [-0.1, 0.04]], [[1.0, -0.05], [-0.05, 0.06]]]),
)
XE, YE = np.linspace(-6, 20, 131), np.linspace(-2.5, 1.5, 81)


def _noise(n, rng):
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = rng.uniform(0.3, 3.0, n) ** 2
    cov[:, 1, 1] = rng.uniform(0.006, 0.04, n) ** 2
    return cov


def test_intrinsic_counts_integrate_to_n():
    h = intrinsic_counts(MIX, 1000, XE, YE)
    assert abs(h.sum() - 1000) < 1.0


def test_convolved_counts_integrate_to_n_and_match_exact_sum():
    rng = np.random.default_rng(0)
    cov = _noise(300, rng)
    xe, ye = np.linspace(-6, 20, 53), np.linspace(-2.5, 1.5, 41)
    fast = convolved_counts(MIX, cov, xe, ye, sub=3, n_err_bins=40)
    assert abs(fast.sum() / 300 - 1) < 2e-3
    # exact: one group per star
    exact = sum(convolved_counts(MIX, cov[i:i + 1], xe, ye, sub=3, n_err_bins=1) for i in range(300))
    assert np.max(np.abs(fast - exact)) < 0.01 * exact.max()


def test_convolved_counts_match_monte_carlo():
    rng = np.random.default_rng(1)
    n = 200000
    cov = _noise(n, rng)
    obs = MIX.sample(n, rng) + np.column_stack([np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1])]) * rng.normal(size=(n, 2))
    xe, ye = np.linspace(-2, 14, 33), np.linspace(-1.5, 0.7, 23)
    h, _, _ = np.histogram2d(obs[:, 0], obs[:, 1], bins=[xe, ye])
    model = convolved_counts(MIX, cov, xe, ye)
    good = model > 100
    pull = (h[good] - model[good]) / np.sqrt(model[good])
    assert abs(pull.mean()) < 0.3 and 0.7 < pull.std() < 1.3
