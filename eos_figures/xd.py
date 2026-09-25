"""Extreme deconvolution: EM fit of an intrinsic Gaussian mixture to noisy data.

Implements Bovy, Hogg & Roweis (2011), arXiv:0905.2979,
https://ui.adsabs.harvard.edu/abs/2011AnApS...5.1657B . Adapted from the numpy
implementation in qso_p_color (https://github.com/vasilybelokurov/qso_p_color,
``src/qso_pcolor/xd.py`` and ``gaussmix.py``), reduced to fully observed data.

For star :math:`i` with measurement covariance :math:`\\mathbf{S}_i` and
component :math:`j`,

.. math::
    \\mathbf{T}_{ij} &= \\mathbf{V}_j + \\mathbf{S}_i \\\\
    q_{ij} &\\propto \\alpha_j\\,\\mathcal{N}(\\mathbf{w}_i \\mid
        \\boldsymbol{\\mu}_j, \\mathbf{T}_{ij}) \\\\
    \\mathbf{b}_{ij} &= \\boldsymbol{\\mu}_j + \\mathbf{V}_j\\mathbf{T}_{ij}^{-1}
        (\\mathbf{w}_i - \\boldsymbol{\\mu}_j) \\\\
    \\mathbf{B}_{ij} &= \\mathbf{V}_j - \\mathbf{V}_j\\mathbf{T}_{ij}^{-1}\\mathbf{V}_j

With :math:`\\mathbf{S}_i = 0` this reduces to ordinary Gaussian-mixture EM.

The number of components is chosen by K-fold held-out log density of the
*observed* data under the noise-convolved model, as in qso_p_color. BIC is also
reported, but for mixtures it rests on regularity conditions that fail
(the model is singular), so it is not used for the decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

log = logging.getLogger(__name__)

_LOG2PI = float(np.log(2.0 * np.pi))


def log_gauss(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a multivariate normal, broadcast over leading axes.

    Parameters
    ----------
    x, mean : ndarray, shape (..., d)
    cov : ndarray, shape (..., d, d), positive definite.

    Returns
    -------
    ndarray, shape (...)
    """
    delta = np.asarray(x, float) - np.asarray(mean, float)
    cov = np.asarray(cov, float)
    d = delta.shape[-1]
    if d == 2:
        # closed form for 2x2: ~10x faster than batched Cholesky, identical result
        a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
        det = a * c - b * b
        if np.any(det <= 0) or np.any(a <= 0):
            raise np.linalg.LinAlgError("covariance not positive definite")
        dx, dy = delta[..., 0], delta[..., 1]
        maha = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
        return -0.5 * (2 * _LOG2PI + np.log(det) + maha)
    shape = np.broadcast_shapes(delta.shape[:-1], cov.shape[:-2])
    delta = np.broadcast_to(delta, shape + (d,)).reshape(-1, d)
    cov = np.broadcast_to(cov, shape + (d, d)).reshape(-1, d, d)
    chol = np.linalg.cholesky(cov)
    y = np.linalg.solve(chol, delta[..., None])[..., 0]
    maha = np.einsum("ni,ni->n", y, y)
    logdet = 2.0 * np.log(np.einsum("nii->ni", chol)).sum(axis=-1)
    return (-0.5 * (d * _LOG2PI + logdet + maha)).reshape(shape)


@dataclass(frozen=True)
class GaussianMixture:
    """Intrinsic (noise-deconvolved) Gaussian mixture in ``d`` dimensions."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.means.ndim != 2 or self.covs.shape != self.means.shape + (self.means.shape[1],):
            raise ValueError("means must be (K, d) and covs (K, d, d)")
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-8):
            raise ValueError("weights must sum to 1")

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def n_dim(self) -> int:
        return self.means.shape[1]

    @property
    def n_params(self) -> int:
        """Free parameters: K-1 weights, K*d means, K*d(d+1)/2 covariances."""
        k, d = self.n_components, self.n_dim
        return (k - 1) + k * d + k * d * (d + 1) // 2

    def component_log_prob(self, x: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        """log(alpha_k N(x | mu_k, V_k + S)), shape (n, K)."""
        x = np.atleast_2d(np.asarray(x, float))
        n, d = x.shape
        t = self.covs[None] if cov is None else self.covs[None] + np.broadcast_to(np.asarray(cov, float).reshape(-1, d, d), (n, d, d))[:, None]
        return log_gauss(x[:, None, :], self.means[None], t) + np.log(self.weights)[None]

    def log_prob(self, x: np.ndarray, cov: np.ndarray | None = None) -> np.ndarray:
        """log p(x); with ``cov`` the density of noisy observations (convolved model)."""
        return logsumexp(self.component_log_prob(x, cov), axis=1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.empty((n, self.n_dim))
        for j in range(self.n_components):
            sel = k == j
            if sel.any():
                out[sel] = rng.multivariate_normal(self.means[j], self.covs[j], size=int(sel.sum()))
        return out

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianMixture":
        covs = np.asarray(d["covs"], float)
        np.linalg.cholesky(covs)  # fail at load time if not positive definite
        return cls(np.asarray(d["weights"], float), np.asarray(d["means"], float), covs, tuple(d.get("labels", ())))


@dataclass
class XDFitResult:
    """Outcome of one XD fit; ``mean_loglike`` is for the returned mixture."""

    mixture: GaussianMixture
    n_iter: int
    mean_loglike: float
    converged: bool
    history: list[float] = field(default_factory=list)


def _gain(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """V_k T_nk^-1 for V (K, d, d) and T (n, K, d, d)."""
    if v.shape[-1] == 2:
        a, b, c = t[..., 0, 0], t[..., 0, 1], t[..., 1, 1]
        det = a * c - b * b
        tinv = np.empty_like(t)
        tinv[..., 0, 0], tinv[..., 1, 1] = c / det, a / det
        tinv[..., 0, 1] = tinv[..., 1, 0] = -b / det
        return np.einsum("kde,nkef->nkdf", v, tinv)
    return np.swapaxes(np.linalg.solve(t, np.broadcast_to(v[None], t.shape)), -1, -2)


def _e_step(x: np.ndarray, s: np.ndarray, mix: GaussianMixture):
    """E step and M-step sufficient statistics.

    Returns per-star log-likelihood ``ll`` (n,), and summed over stars:
    ``acc_q`` = sum q (K,), ``acc_dmu`` = sum q (b - mu) (K, d), and
    ``acc_v`` = sum q [B + (b - mu)(b - mu)^T] (K, d, d).
    """
    lw = np.log(np.maximum(mix.weights, 1e-300))[None]
    if x.shape[1] == 2:
        # element-wise 2x2 algebra over (n, K)
        va, vb, vc = (mix.covs[:, i, j][None] for i, j in ((0, 0), (0, 1), (1, 1)))
        ta, tb, tc = va + s[:, 0, 0, None], vb + s[:, 0, 1, None], vc + s[:, 1, 1, None]
        det = ta * tc - tb * tb
        rx = x[:, 0, None] - mix.means[None, :, 0]
        ry = x[:, 1, None] - mix.means[None, :, 1]
        maha = (tc * rx * rx - 2 * tb * rx * ry + ta * ry * ry) / det
        lq = -0.5 * (2 * _LOG2PI + np.log(det) + maha) + lw
        ll = logsumexp(lq, axis=1)
        q = np.exp(lq - ll[:, None])
        # G = V T^-1
        g00, g01 = (va * tc - vb * tb) / det, (vb * ta - va * tb) / det
        g10, g11 = (vb * tc - vc * tb) / det, (vc * ta - vb * tb) / det
        dbx, dby = g00 * rx + g01 * ry, g10 * rx + g11 * ry
        # B = V - G V, plus db db^T
        m00 = va - (g00 * va + g01 * vb) + dbx * dbx
        m01 = vb - (g00 * vb + g01 * vc) + dbx * dby
        m11 = vc - (g10 * vb + g11 * vc) + dby * dby
        acc_v = np.empty((mix.n_components, 2, 2))
        acc_v[:, 0, 0], acc_v[:, 1, 1] = (q * m00).sum(0), (q * m11).sum(0)
        acc_v[:, 0, 1] = acc_v[:, 1, 0] = (q * m01).sum(0)
        acc_dmu = np.column_stack([(q * dbx).sum(0), (q * dby).sum(0)])
        return ll, q.sum(0), acc_dmu, acc_v
    t = mix.covs[None] + s[:, None]                                      # (n, K, d, d)
    lq = log_gauss(x[:, None, :], mix.means[None], t) + lw
    ll = logsumexp(lq, axis=1)
    q = np.exp(lq - ll[:, None])
    resid = x[:, None, :] - mix.means[None]
    gain = _gain(mix.covs, t)
    db = np.einsum("nkde,nke->nkd", gain, resid)                         # b - mu_old
    bmat = mix.covs[None] - np.einsum("nkde,kef->nkdf", gain, mix.covs)
    acc_v = np.einsum("nk,nkde->kde", q, bmat) + np.einsum("nk,nkd,nke->kde", q, db, db)
    return ll, q.sum(0), np.einsum("nk,nkd->kd", q, db), acc_v


def _init_kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> GaussianMixture:
    """k-means++ clustering of the standardised data; per-cluster moments."""
    from scipy.cluster.vq import kmeans2

    mu, sd = x.mean(0), x.std(0)
    z = (x - mu) / sd
    _, lab = kmeans2(z, k, minit="++", seed=rng, iter=30)
    var0 = x.var(axis=0)
    w, means, covs = np.empty(k), np.empty((k, x.shape[1])), np.empty((k, x.shape[1], x.shape[1]))
    for j in range(k):
        sel = lab == j
        if sel.sum() > x.shape[1] + 1:
            w[j], means[j], covs[j] = sel.mean(), x[sel].mean(0), np.cov(x[sel].T) + 1e-6 * np.diag(var0)
        else:  # empty or tiny cluster: restart at a random row
            w[j], means[j], covs[j] = 1.0 / x.shape[0], x[rng.integers(x.shape[0])], np.diag(0.01 * var0)
    return GaussianMixture(w / w.sum(), means, covs)


def _init_mixture(x: np.ndarray, k: int, rng: np.random.Generator) -> GaussianMixture:
    """Centres at jittered random data rows; covariances at the global variance."""
    var0 = x.var(axis=0)
    idx = rng.choice(x.shape[0], size=k, replace=x.shape[0] < k)
    means = x[idx] + rng.normal(scale=0.1 * np.sqrt(var0), size=(k, x.shape[1]))
    covs = np.repeat(np.diag(var0)[None], k, axis=0)
    return GaussianMixture(np.full(k, 1.0 / k), means, covs)


def fit_xd(
    x: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    n_components: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-7,
    window: int = 1,
    regularization: float = 0.0,
    init: GaussianMixture | None = None,
    init_method: str = "kmeans",
    seed: int = 0,
    labels: tuple[str, ...] = (),
) -> XDFitResult:
    """Fit an intrinsic Gaussian mixture by extreme deconvolution.

    Parameters
    ----------
    x : ndarray, shape (n, d)
        Observed values.
    cov : ndarray, shape (n, d, d) or (d, d), optional
        Per-object measurement covariance. ``None`` fits an ordinary GMM.
    n_components : int
        Number of components K; choose it with :func:`select_n_components`.
    max_iter, tol, window : int, float, int
        Stop when the mean per-star log-likelihood has changed by less than
        ``tol * max(1, |ll|)`` per iteration, averaged over the last ``window``
        iterations. ``window > 1`` avoids stopping on the plateaus that XD
        crosses slowly before a component rearranges.
    regularization : float
        Variance floor added to each component covariance diagonal (feature units^2).
    init : GaussianMixture, optional
        Starting point; ``None`` initialises from the data with ``seed``.
    init_method : {"kmeans", "random"}
        ``"kmeans"``: k-means++ clusters of the standardised data (default).
        ``"random"``: jittered random rows with the global variance, as in qso_p_color.

    Returns
    -------
    XDFitResult
    """
    x = np.atleast_2d(np.asarray(x, float))
    n, d = x.shape
    s = np.zeros((n, d, d)) if cov is None else np.broadcast_to(np.asarray(cov, float).reshape(-1, d, d), (n, d, d))
    if init is None:
        make = _init_kmeans if init_method == "kmeans" else _init_mixture
        init = make(x, n_components, np.random.default_rng(seed))
    mix = init
    k = mix.n_components

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        ll, acc_q, acc_dmu, acc_v = _e_step(x, s, mix)

        # M step; second moments are about the old mean for numerical stability
        alive = acc_q > 1e-10
        if not alive.all():
            log.warning("K=%d: %d empty component(s) kept frozen", k, int((~alive).sum()))
        qs = np.maximum(acc_q, 1e-300)
        shift = acc_dmu / qs[:, None]
        new_covs = acc_v / qs[:, None, None] - np.einsum("kd,ke->kde", shift, shift)
        new_covs = 0.5 * (new_covs + np.swapaxes(new_covs, -1, -2)) + regularization * np.eye(d)
        new_covs = np.where(alive[:, None, None], new_covs, mix.covs)
        new_means = np.where(alive[:, None], mix.means + shift, mix.means)
        mix = GaussianMixture(acc_q / acc_q.sum(), new_means, new_covs, labels)

        mean_ll = float(ll.mean())
        history.append(mean_ll)
        if it > window and abs(mean_ll - history[-1 - window]) / window < tol * max(1.0, abs(mean_ll)):
            converged = True
            break

    # the loop's likelihood belongs to the pre-M-step mixture; recompute for the returned one
    final_ll = float(mix.log_prob(x, None if cov is None else s).mean())
    return XDFitResult(mix, it, final_ll, converged, history)


def fit_xd_restarts(x, cov, n_components: int, n_init: int = 3, seed: int = 0, **kwargs) -> XDFitResult:
    """Best (highest training log-likelihood) of ``n_init`` randomly initialised fits."""
    fits = [fit_xd(x, cov, n_components=n_components, seed=seed + 1000 * r, **kwargs) for r in range(n_init)]
    return max(fits, key=lambda f: f.mean_loglike)


def select_n_components(
    x: np.ndarray,
    cov: np.ndarray | None,
    candidates: list[int],
    *,
    n_folds: int = 5,
    n_init: int = 3,
    seed: int = 0,
    **fit_kwargs,
) -> tuple[int, dict[int, dict]]:
    """Choose K by K-fold held-out log density of the observed data.

    Returns
    -------
    best_k : int
        K with the highest mean held-out log density per star.
    scores : dict
        ``K -> {"cv_mean", "cv_se", "cv_folds"}``; ``cv_se`` is the standard
        error of the per-star mean over all held-out stars.
    """
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[0]
    fold_of = np.random.default_rng(seed).permutation(n) % n_folds
    scores: dict[int, dict] = {}
    for k in candidates:
        lp_all = np.empty(n)
        folds = []
        for f in range(n_folds):
            tr, va = fold_of != f, fold_of == f
            res = fit_xd_restarts(x[tr], None if cov is None else cov[tr], k, n_init=n_init, seed=seed + f, **fit_kwargs)
            lp_all[va] = res.mixture.log_prob(x[va], None if cov is None else cov[va])
            folds.append(float(lp_all[va].mean()))
        scores[k] = {"cv_mean": float(lp_all.mean()), "cv_se": float(lp_all.std(ddof=1) / np.sqrt(n)), "cv_folds": folds}
        log.info("K=%d held-out mean log density %.5f +- %.5f", k, scores[k]["cv_mean"], scores[k]["cv_se"])
    best_k = max(scores, key=lambda kk: scores[kk]["cv_mean"])
    return best_k, scores


def bic(result: XDFitResult, n: int) -> float:
    """Bayesian information criterion, -2 ln L + p ln n (lower is better)."""
    return -2.0 * result.mean_loglike * n + result.mixture.n_params * np.log(n)


def aic(result: XDFitResult, n: int) -> float:
    """Akaike information criterion, -2 ln L + 2p (lower is better)."""
    return -2.0 * result.mean_loglike * n + 2.0 * result.mixture.n_params


def save_model(path: Path, mixture: GaussianMixture, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mixture": mixture.to_dict(), "meta": meta}, indent=1))
    return path


def load_model(path: Path) -> tuple[GaussianMixture, dict]:
    d = json.loads(Path(path).read_text())
    return GaussianMixture.from_dict(d["mixture"]), d["meta"]
