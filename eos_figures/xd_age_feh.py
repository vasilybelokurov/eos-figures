"""Data vector, noise model and grid evaluation for the Age-[Fe/H] XD model."""
from __future__ import annotations

import numpy as np

from .config import DEFAULT_CACHE, REPO, Cuts
from .data import load_catalog, make_masks
from .xd import GaussianMixture

DEFAULT_MODEL = REPO / "data" / "xd_age_feh_model.json"
LABELS = ("age", "fe_h")


def load_age_feh(cache=DEFAULT_CACHE, mask_name: str = "base_age"):
    """Return x (n, 2) = (age [Gyr], [Fe/H]), noise cov (n, 2, 2) and metadata.

    Noise: diag(age_total_error^2, fe_h_err^2); AstroNN total age error
    (model + propagated measurement uncertainty), APOGEE [Fe/H] error, no
    covariance between the two.
    """
    cat = load_catalog(cache)
    w = make_masks(cat, Cuts())[mask_name]
    x = np.column_stack([cat["age"][w], cat["fe_h"][w]]).astype(float)
    cov = np.zeros((len(x), 2, 2))
    cov[:, 0, 0] = np.asarray(cat["age_total_error"][w], float) ** 2
    cov[:, 1, 1] = np.asarray(cat["fe_h_err"][w], float) ** 2
    good = np.isfinite(x).all(1) & np.isfinite(cov).all((1, 2)) & (cov[:, 0, 0] > 0) & (cov[:, 1, 1] > 0)
    meta = {"mask": mask_name, "n": int(good.sum()), "n_dropped_nonfinite": int((~good).sum()),
            "labels": list(LABELS), "noise": "diag(age_total_error^2, fe_h_err^2)"}
    return x[good], cov[good], meta


def _supersampled_centres(xe, ye, sub):
    """Sub-bin centres, shape (nx, ny, sub*sub, 2)."""
    fx = (np.arange(sub) + 0.5) / sub
    xs = xe[:-1, None] + np.diff(xe)[:, None] * fx[None]                  # (nx, sub)
    ys = ye[:-1, None] + np.diff(ye)[:, None] * fx[None]                  # (ny, sub)
    gx = np.broadcast_to(xs[:, None, :, None], (len(xs), len(ys), sub, sub))
    gy = np.broadcast_to(ys[None, :, None, :], (len(xs), len(ys), sub, sub))
    return np.stack([gx, gy], -1).reshape(len(xs), len(ys), sub * sub, 2)


def intrinsic_counts(mix: GaussianMixture, n: int, xe, ye, sub: int = 5) -> np.ndarray:
    """Expected counts per bin, n * integral of the deconvolved density (midpoint rule on sub x sub)."""
    pts = _supersampled_centres(xe, ye, sub)
    area = np.diff(xe)[:, None] * np.diff(ye)[None, :]
    p = np.exp(mix.log_prob(pts.reshape(-1, 2))).reshape(pts.shape[:3]).mean(-1)
    return n * p * area


def convolved_counts(mix: GaussianMixture, cov: np.ndarray, xe, ye, sub: int = 3, n_err_bins: int = 40) -> np.ndarray:
    """Expected counts per bin of the noisy data: sum_i integral of sum_k a_k N(x | mu_k, V_k + S_i).

    Stars are grouped by their diagonal errors on a n_err_bins^2 grid in
    log(sigma); each group uses its mean variances and its star count. The
    grouping only affects the result at the level of the within-group error
    spread (checked against the exact sum in tests).
    """
    sa, sf = np.sqrt(cov[:, 0, 0]), np.sqrt(cov[:, 1, 1])
    la, lf = np.log(sa), np.log(sf)
    ia = np.minimum(((la - la.min()) / (np.ptp(la) + 1e-12) * n_err_bins).astype(int), n_err_bins - 1)
    jf = np.minimum(((lf - lf.min()) / (np.ptp(lf) + 1e-12) * n_err_bins).astype(int), n_err_bins - 1)
    key = ia * n_err_bins + jf
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    va = np.bincount(inv, cov[:, 0, 0]) / cnt
    vf = np.bincount(inv, cov[:, 1, 1]) / cnt

    pts = _supersampled_centres(xe, ye, sub).reshape(-1, 2)
    area = np.diff(xe)[:, None] * np.diff(ye)[None, :]
    dens = np.zeros(len(pts))
    for g in range(len(uk)):
        s = np.array([[va[g], 0.0], [0.0, vf[g]]])
        dens += cnt[g] * np.exp(mix.log_prob(pts, np.broadcast_to(s, (len(pts), 2, 2))))
    dens = dens.reshape(len(xe) - 1, len(ye) - 1, sub * sub).mean(-1)
    return dens * area
