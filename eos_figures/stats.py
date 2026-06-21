from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d, gaussian_kde


def hist2d(x, y, xr, yr, nx, ny, normalize: str | None = None):
    h, xe, ye = np.histogram2d(x, y, bins=[nx, ny], range=[xr, yr])
    if normalize == "x":
        denom = h.sum(axis=1, keepdims=True)
        h = np.divide(h, denom, out=np.zeros_like(h), where=denom > 0)
    elif normalize == "y":
        denom = h.sum(axis=0, keepdims=True)
        h = np.divide(h, denom, out=np.zeros_like(h), where=denom > 0)
    return h, xe, ye


def stat2d(x, y, values, xr, yr, nx, ny, statistic="median"):
    stat, xe, ye, _ = binned_statistic_2d(
        x, y, values, statistic=statistic, bins=[nx, ny], range=[xr, yr]
    )
    return stat, xe, ye


def finite_percentile(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([np.nan] * len(q))
    return np.nanpercentile(values, q)


def log_image(h):
    out = np.full_like(h, np.nan, dtype=float)
    good = h > 0
    out[good] = np.log10(h[good])
    return out


def scale01(values, vmin, vmax):
    return np.clip((values - vmin) / (vmax - vmin), 0, 1)


def bin_percentile(x, y, xr, nbins, percentile):
    stat, edges, _ = binned_statistic(x, y, statistic=lambda z: np.nanpercentile(z, percentile), bins=nbins, range=xr)
    med, _, _ = binned_statistic(x, x, statistic="median", bins=nbins, range=xr)
    std, _, _ = binned_statistic(x, y, statistic="std", bins=nbins, range=xr)
    count, _, _ = binned_statistic(x, y, statistic="count", bins=nbins, range=xr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, med, stat, std, count


def kde_curve(values, xr, n=300):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    grid = np.linspace(xr[0], xr[1], n)
    if values.size < 2:
        return grid, np.zeros_like(grid)
    den = gaussian_kde(values)(grid)
    area = np.trapz(den, grid)
    if area > 0:
        den = den / area
    return grid, den

