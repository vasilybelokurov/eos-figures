from __future__ import annotations

import os
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_repo / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_repo / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(_repo / ".mplconfig").mkdir(exist_ok=True)
(_repo / ".cache").mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .stats import finite_percentile, log_image


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }
)


def setup_axes(ncols, nrows=1, figsize=(10, 3), sharex=False, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey, constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel()


def density_panel(ax, h, xe, ye, percentiles=(2, 98), vmin=None, vmax=None, cmap="Greys", **kwargs):
    im = log_image(h)
    if vmin is None or vmax is None:
        vals = im[np.isfinite(im)]
        mm = finite_percentile(vals, percentiles)
        vmin = mm[0] if vmin is None else vmin
        vmax = mm[1] if vmax is None else vmax
    return ax.imshow(
        im.T,
        origin="lower",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )


def value_panel(ax, values, xe, ye, vmin, vmax, mask=None, cmap="viridis_r", colorbar_label=None):
    img = np.array(values, dtype=float)
    if mask is not None:
        img = img.copy()
        img[mask] = np.nan
    im = ax.imshow(
        img.T,
        origin="lower",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if colorbar_label:
        plt.colorbar(im, ax=ax, pad=0.02, label=colorbar_label)
    return im


def smoothed_scatter_panel(ax, x, y, c=None, xr=None, yr=None, cmap="viridis_r", vmin=None, vmax=None, s=1.2, sort_by_color=True):
    if c is None:
        ax.scatter(x, y, s=s, c="0.1", rasterized=True, linewidths=0)
        return None
    order = np.argsort(c) if sort_by_color else np.arange(len(c))
    sc = ax.scatter(x[order], y[order], c=c[order], s=s, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True, linewidths=0)
    return sc


def filtered_image_from_points(x, y, c, xr, yr, bins=(800, 800), vmin=None, vmax=None):
    h_sum, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr], weights=c)
    h_count, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    img = np.divide(h_sum, h_count, out=np.full_like(h_sum, np.nan, dtype=float), where=h_count > 0)
    img = gaussian_filter(np.nan_to_num(img, nan=0.0), sigma=1.2)
    return img, xe, ye


def label_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
