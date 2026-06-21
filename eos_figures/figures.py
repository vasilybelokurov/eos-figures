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

from .config import Cuts, DEFAULT_CACHE, DEFAULT_OUTDIR
from .data import load_catalog, make_masks
from .plotting import density_panel, label_axes, setup_axes, smoothed_scatter_panel, value_panel
from .stats import bin_percentile, finite_percentile, hist2d, kde_curve, log_image, stat2d


FIGURES = {}


def figure(name):
    def decorate(func):
        FIGURES[name] = func
        return func

    return decorate


def load_context(cache=DEFAULT_CACHE):
    cat = load_catalog(cache)
    cuts = Cuts()
    masks = make_masks(cat, cuts)
    return cat, cuts, masks


def output_path(outdir, name):
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{name}.pdf"


def save(fig, outdir, name):
    path = output_path(outdir, name)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def _idl_low_density_mask(hist, percentiles, white_lim):
    im = log_image(hist)
    mm = finite_percentile(im, percentiles)
    if not np.all(np.isfinite(mm)) or mm[1] <= mm[0]:
        return ~np.isfinite(im)
    scaled = 255 * np.clip((im - mm[0]) / (mm[1] - mm[0]), 0, 1)
    return (~np.isfinite(im)) | ((255 - scaled) > white_lim)


@figure("eos_energy_mg_al")
def plot_energy_mg_al(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))

    h, xe, ye = hist2d(cat["lz"][m["base"]], 1e-5 * cat["energy"][m["base"]], c.lzr, c.enr, c.nlz, c.nen)
    density_panel(ax[0], h, xe * 1e-3, ye, percentiles=(1, 95), vmin=-0.3)
    ax[0].axvline(0, color="k", lw=0.8)
    label_axes(ax[0], r"$L_z\times 10^{-3}$", r"$E\times 10^{-5}$", "Energy, Lz")

    h, xe, ye = hist2d(cat["fe_h"][m["base"]], cat["mg_fe"][m["base"]], c.fehr, c.mgfer, c.nfeh, c.nmg)
    density_panel(ax[1], h, xe, ye, percentiles=(2, 98))
    xx = np.array(c.fehr)
    ax[1].plot(xx, c.slope_acc * xx + c.inter_acc, "w-", lw=1.1)
    ax[1].plot(xx, c.slope_acc * xx + c.inter_acc, "k--", lw=0.8)
    ax[1].plot(xx, c.slope_acc2 * xx + c.inter_acc2, "w-", lw=1.1)
    ax[1].plot(xx, c.slope_acc2 * xx + c.inter_acc2, "k:", lw=1.0)
    ax[1].text(-1.8, -0.02, "accreted", fontsize=8)
    ax[1].text(-0.8, 0.32, r"high-$\alpha$", color="w", fontsize=8)
    ax[1].text(-0.35, 0.06, r"low-$\alpha$", color="w", fontsize=8, rotation=-30)
    label_axes(ax[1], "[Fe/H]", "[Mg/Fe]", "Magnesium")

    h, xe, ye = hist2d(cat["fe_h"][m["base"]], cat["al_fe"][m["base"]], c.fehr, c.alfer, c.nfeh, c.nal)
    density_panel(ax[2], h, xe, ye, percentiles=c.perc1)
    ax[2].axhline(c.alfe_cut, color="k", ls="--", lw=0.8)
    label_axes(ax[2], "[Fe/H]", "[Al/Fe]", "Aluminium")
    return save(fig, outdir, "eos_energy_mg_al")


@figure("eos_alfe_pops")
def plot_alfe_pops(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, nrows=2, figsize=(10, 6))
    specs = [
        ("acc", "accreted", c.perc),
        ("thick", r"high-$\alpha$", c.perc2),
        ("thin", r"low-$\alpha$", c.perc2),
    ]
    hist_cache = {}
    mask_cache = {}
    for i, (mask_name, title, perc) in enumerate(specs):
        h, xe, ye = hist2d(cat["fe_h"][m[mask_name]], cat["al_fe"][m[mask_name]], c.fehr, c.alfer, c.nfeh, c.nal2, normalize="x")
        hist_cache[mask_name] = (h, xe, ye)
        mask_cache[mask_name] = _idl_low_density_mask(h, perc, c.white_lim)
        density_panel(ax[i], h, xe, ye, percentiles=perc)
        ax[i].plot(c.fehr, np.array(c.fehr) * c.kalfe + c.offalfe, "k--", lw=0.8)
        ax[i].axhline(c.alfe_cut, color="k", ls="--", lw=0.8)
        ax[i].set_xlim(c.fehr)
        ax[i].set_ylim(c.alfer)
        label_axes(ax[i], "[Fe/H]", "[Al/Fe]", title)

    for i, (mask_name, title, _) in enumerate(specs, start=3):
        h, xe, ye = hist_cache[mask_name]
        vmask = (
            m[mask_name]
            & np.isfinite(cat["galvt"])
            & (cat["galvt"] >= c.vtanr[0])
            & (cat["galvt"] <= c.vtanr[1])
        )
        med, _, _ = stat2d(cat["fe_h"][vmask], cat["al_fe"][vmask], cat["galvt"][vmask], c.fehr, c.alfer, c.nfeh, c.nal2)
        h_med, _, _ = hist2d(cat["fe_h"][vmask], cat["al_fe"][vmask], c.fehr, c.alfer, c.nfeh, c.nal2)
        med = np.nan_to_num(med, nan=0.0)
        med[h_med <= 2] = 0.0
        value_panel(
            ax[i],
            med,
            xe,
            ye,
            *c.mm_vtan,
            mask=mask_cache[mask_name],
            cmap="RdYlBu_r",
            colorbar_label=r"$V_{\rm tan}$ [km/s]" if i == 4 else None,
        )
        ax[i].plot(c.fehr, np.array(c.fehr) * c.kalfe + c.offalfe, "k--", lw=0.8)
        ax[i].axhline(c.alfe_cut, color="k", ls="--", lw=0.8)
        ax[i].set_xlim(c.fehr)
        ax[i].set_ylim(c.alfer)
        label_axes(ax[i], "[Fe/H]", "[Al/Fe]", title)
    ax[3].text(-1.5, 0.01, "GS/E", fontsize=9)
    ax[4].text(-1.75, 0.0, "Aurora", fontsize=9, rotation=60)
    ax[4].text(-1.25, 0.44, r"Splash+high-$\alpha$ disk", fontsize=8)
    ax[5].text(-1.3, -0.1, "Eos", fontsize=9, rotation=60)
    ax[5].text(-0.75, 0.3, r"low-$\alpha$ disk", fontsize=8)
    return save(fig, outdir, "eos_alfe_pops")


@figure("eos_energy_pops")
def plot_energy_pops(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))
    for axis, mask_name, title in zip(ax, ["acc_al", "thick_al", "thin_al"], ["accreted", r"high-$\alpha$", r"low-$\alpha$"]):
        h, xe, ye = hist2d(cat["lz"][m[mask_name]], 1e-5 * cat["energy"][m[mask_name]], c.lzr, c.enr, c.nlz2, c.nen2)
        density_panel(axis, h, xe * 1e-3, ye, percentiles=c.perc_elz, vmin=-0.3)
        axis.axvline(0, color="k", lw=0.8)
        label_axes(axis, r"$L_z\times 10^{-3}$", r"$E\times 10^{-5}$", title)
    ax[2].text(-1.4, -0.55, "Eos", fontsize=9)
    return save(fig, outdir, "eos_energy_pops")


@figure("eos_alfe_3pops")
def plot_alfe_3pops(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))
    h0, xe, ye = hist2d(cat["fe_h"][m["base_en"]], cat["mg_fe"][m["base_en"]], c.fehr2, c.mgfer2, c.nfeh2, c.nmg2)
    h, _, _ = hist2d(cat["fe_h"][m["base_en"]], cat["mg_fe"][m["base_en"]], c.fehr2, c.mgfer2, c.nfeh2, c.nmg2, normalize="x")
    density_panel(ax[0], h, xe, ye, percentiles=c.perc_mgfe)
    xx = np.array(c.fehr2)
    ax[0].plot(xx, c.slope_acc * xx + c.inter_acc, "w--", lw=0.9)
    ax[0].plot(xx, c.slope_acc2 * xx + c.inter_acc2, "w:", lw=1.0)
    ax[0].set_xlim(c.fehr2)
    ax[0].set_ylim(c.mgfer2)
    label_axes(ax[0], "[Fe/H]", "[Mg/Fe]", "Column-normalised density")

    mean_al, xe, ye = stat2d(cat["fe_h"][m["base_en"]], cat["mg_fe"][m["base_en"]], cat["al_fe"][m["base_en"]], c.fehr2, c.mgfer2, c.nfeh2, c.nmg2, statistic="mean")
    im_mid = value_panel(ax[1], mean_al, xe, ye, -0.2, 0.27, mask=h0 <= 1, cmap="RdYlBu_r")
    cax = ax[1].inset_axes([0.12, 0.10, 0.56, 0.035])
    cb = fig.colorbar(im_mid, cax=cax, orientation="horizontal")
    cb.set_label("[Al/Fe]", fontsize=8)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=7, length=2)
    ax[1].plot(xx, c.slope_acc * xx + c.inter_acc, "k--", lw=0.8)
    ax[1].plot(xx, c.slope_acc2 * xx + c.inter_acc2, "k:", lw=1.0)
    ax[1].set_xlim(c.fehr2)
    ax[1].set_ylim(c.mgfer2)
    label_axes(ax[1], "[Fe/H]", "[Mg/Fe]", "Mean [Al/Fe]")

    h0, xe, ye = hist2d(cat["al_fe"][m["tri"]], cat["mg_fe"][m["tri"]], c.alfer2, c.mgfer2, 21, 20)
    h, _, _ = hist2d(cat["al_fe"][m["tri"]], cat["mg_fe"][m["tri"]], c.alfer2, c.mgfer2, 21, 20, normalize="y")
    mm = finite_percentile(h[h > 0], (35, 93))
    scaled = 255 * np.clip((h - mm[0]) / (mm[1] - mm[0]), 0, 1)
    scaled[h0 <= 1] = np.nan
    ax[2].imshow(
        scaled.T,
        origin="lower",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=255,
    )
    prof = np.nansum(scaled, axis=1)
    if np.nanmax(prof) > 0:
        xcent = 0.5 * (xe[:-1] + xe[1:])
        yprof = c.mgfer2[0] + 0.2 * prof / np.nanmax(prof)
        ax[2].step(xcent, yprof, where="mid", color="k", lw=0.9)
    ax[2].text(-0.5, 0.1, "GS/E", fontsize=8)
    ax[2].text(-0.17, 0.22, "Eos", fontsize=8)
    ax[2].text(-0.02, 0.33, "disc", fontsize=8)
    ax[2].set_xlim(c.alfer2)
    ax[2].set_ylim(c.mgfer2)
    label_axes(ax[2], "[Al/Fe]", "[Mg/Fe]", f"{c.feh_tri_cut[0]:.1f}<[Fe/H]<{c.feh_tri_cut[1]:.1f}")
    return save(fig, outdir, "eos_alfe_3pops")


@figure("eos_vphi")
def plot_vphi(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR, sort_by_color=False, output_name="eos_vphi"):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))
    panels = [
        (m["thin_al"], cat["rap"], c.mm_rapo, r"$r_{\rm apo}$", np.linspace(c.mm_rapo[0], c.mm_rapo[1], 5)),
        (m["thin_al"], cat["rperi"], c.mm_rperi, r"$r_{\rm peri}$", np.linspace(c.mm_rperi[0], c.mm_rperi[1], 5)),
        (m["thin_al_age"], cat["age"], c.mm_age, "Age", np.linspace(c.mm_age[0], c.mm_age[1], 5)),
    ]
    point_size = 4.5
    for axis, (mask, color, mm, label, ticks) in zip(ax, panels):
        clipped_color = np.clip(color[mask], mm[0], mm[1])
        sc = smoothed_scatter_panel(
            axis,
            cat["fe_h"][mask],
            cat["galvt"][mask],
            clipped_color,
            c.fehr_plot,
            c.vphir_plot2,
            cmap="RdYlBu_r",
            vmin=mm[0],
            vmax=mm[1],
            s=point_size,
            sort_by_color=sort_by_color,
        )
        axis.axhline(0, color="k", ls="--", lw=0.8)
        axis.set_xlim(c.fehr_plot)
        axis.set_ylim(c.vphir_plot2)
        axis.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5])
        axis.set_yticks([-200, -100, 0, 100, 200, 300])
        cax = axis.inset_axes([0.08, 0.07, 0.78, 0.035])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=8)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.tick_params(labelsize=7, length=2)
        label_axes(axis, "[Fe/H]", r"$V_{\rm tan}$")
    return save(fig, outdir, output_name)


def plot_vphi_pixels(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR, bins=(70, 70), min_count=1, output_name="eos_vphi_pixels"):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))
    panels = [
        (m["thin_al"], cat["rap"], c.mm_rapo, r"$r_{\rm apo}$", np.linspace(c.mm_rapo[0], c.mm_rapo[1], 5)),
        (m["thin_al"], cat["rperi"], c.mm_rperi, r"$r_{\rm peri}$", np.linspace(c.mm_rperi[0], c.mm_rperi[1], 5)),
        (m["thin_al_age"], cat["age"], c.mm_age, "Age", np.linspace(c.mm_age[0], c.mm_age[1], 5)),
    ]
    for axis, (mask, values, mm, label, ticks) in zip(ax, panels):
        mean, xe, ye = stat2d(
            cat["fe_h"][mask],
            cat["galvt"][mask],
            values[mask],
            c.fehr_plot,
            c.vphir_plot2,
            bins[0],
            bins[1],
            statistic="mean",
        )
        counts, _, _ = hist2d(cat["fe_h"][mask], cat["galvt"][mask], c.fehr_plot, c.vphir_plot2, bins[0], bins[1])
        im = value_panel(axis, mean, xe, ye, mm[0], mm[1], mask=counts < min_count, cmap="RdYlBu_r")
        axis.axhline(0, color="k", ls="--", lw=0.8)
        axis.set_xlim(c.fehr_plot)
        axis.set_ylim(c.vphir_plot2)
        axis.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5])
        axis.set_yticks([-200, -100, 0, 100, 200, 300])
        cax = axis.inset_axes([0.08, 0.07, 0.78, 0.035])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=8)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.tick_params(labelsize=7, length=2)
        label_axes(axis, "[Fe/H]", r"$V_{\rm tan}$", f"Mean {label}")
    return save(fig, outdir, output_name)


@figure("eos_age")
def plot_age(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(3, figsize=(10, 3))

    for mask_name, color, label in [("thin_al_age", "tab:red", r"low-$\alpha$"), ("thick_age", "tab:blue", r"high-$\alpha$")]:
        xcen, _, med, std, n = bin_percentile(cat["fe_h"][m[mask_name]], cat["age"][m[mask_name]], c.fehr_plot, c.nbins_age, 50)
        _, _, p5, _, _ = bin_percentile(cat["fe_h"][m[mask_name]], cat["age"][m[mask_name]], c.fehr_plot, c.nbins_age, 5)
        _, _, p95, _, _ = bin_percentile(cat["fe_h"][m[mask_name]], cat["age"][m[mask_name]], c.fehr_plot, c.nbins_age, 95)
        err = np.divide(std, np.sqrt(n), out=np.full_like(std, np.nan), where=n > 0)
        good = np.isfinite(med)
        ax[0].fill_between(xcen[good], med[good] - err[good], med[good] + err[good], color=color, alpha=0.25)
        ax[0].plot(xcen, med, color=color, label=label)
        ax[0].plot(xcen, p5, color=color, ls=":", lw=0.9)
        ax[0].plot(xcen, p95, color=color, ls=":", lw=0.9)
    ax[0].set_xlim(c.fehr_plot)
    ax[0].set_ylim(c.ager)
    ax[0].legend(frameon=False, fontsize=8)
    label_axes(ax[0], "[Fe/H]", "Age, Gyr")

    sc = smoothed_scatter_panel(
        ax[1],
        cat["fe_h"][m["thin_al_vt_age"]],
        cat["age"][m["thin_al_vt_age"]],
        cat["rap"][m["thin_al_vt_age"]],
        c.fehr_plot,
        c.ager,
        cmap="RdYlBu_r",
        vmin=c.mm_rapo[0],
        vmax=c.mm_rapo[1],
        s=13,
    )
    ax[1].set_xlim(c.fehr_plot)
    ax[1].set_ylim(c.ager)
    cax = ax[1].inset_axes([0.08, 0.08, 0.36, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("r apo", fontsize=8)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=7, length=2)
    label_axes(ax[1], "[Fe/H]", "Age", r"low-$\alpha$ and $V_{\rm tan}<80$ km/s")

    curves = [
        ("thin_al_vt_rap_age", "tab:red", "Eos", "-"),
        ("thick_al_splash_age", "tab:blue", "Splash", "-"),
        ("thin_al_age", "tab:red", r"low-$\alpha$ disc", "--"),
    ]
    for mask_name, color, label, ls in curves:
        x, y = kde_curve(cat["age"][m[mask_name]], c.ager)
        ax[2].plot(x, y, color=color, ls=ls, label=label)
    ax[2].set_xlim(c.ager)
    ax[2].set_ylim(0, None)
    ax[2].legend(frameon=False, fontsize=8)
    label_axes(ax[2], "Age, Gyr", "Density")
    return save(fig, outdir, "eos_age")


@figure("eos_mg_feh_ecc")
def plot_mg_feh_ecc(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(4, nrows=2, figsize=(12.75, 6))
    w = m["ecc"]
    high_vtan = m["base"] & (cat["galvt"] > 220)
    point_size = 6.0

    panels = [
        (cat["fe_h"], cat["mg_fe"], None, c.fehr3, c.mgfer_plot, None, "[Mg/Fe]"),
        (cat["fe_h"], cat["mg_fe"], cat["al_fe"], c.fehr3, c.mgfer_plot, c.mm_al, "[Mg/Fe]"),
        (cat["fe_h"], cat["mg_fe"], cat["rap"], c.fehr3, c.mgfer_plot, c.mm_rapo2, "[Mg/Fe]"),
        (cat["fe_h"], cat["mg_fe"], None, c.fehr3, c.mgfer_plot, None, "[Mg/Fe]"),
        (cat["fe_h"], cat["al_fe"], None, c.fehr3, c.alfer_plot, None, "[Al/Fe]"),
        (cat["fe_h"], cat["al_fe"], cat["mg_fe"], c.fehr3, c.alfer_plot, c.mm_mg, "[Al/Fe]"),
        (cat["fe_h"], cat["al_fe"], cat["rap"], c.fehr3, c.alfer_plot, c.mm_rapo2, "[Al/Fe]"),
        (cat["fe_h"], cat["al_fe"], None, c.fehr3, c.alfer_plot, None, "[Al/Fe]"),
    ]
    for i, (x, y, color, xr, yr, mm, ylabel) in enumerate(panels):
        axis = ax[i]
        if color is None:
            axis.scatter(x[w], y[w], s=point_size, c="0.0" if i not in (3, 7) else "0.65", rasterized=True, linewidths=0)
        else:
            sc = smoothed_scatter_panel(
                axis,
                x[w],
                y[w],
                color[w],
                xr,
                yr,
                cmap="RdYlBu_r",
                vmin=mm[0],
                vmax=mm[1],
                s=point_size,
            )
            cax = axis.inset_axes([0.08, 0.08, 0.82, 0.035])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cb.set_label("[Al/Fe]" if i == 1 else ("[Mg/Fe]" if i == 5 else "r apo"), fontsize=8)
            cb.ax.xaxis.set_label_position("top")
            cb.ax.xaxis.set_ticks_position("bottom")
            cb.ax.tick_params(labelsize=7, length=2)
        if i in (3, 7):
            hh, xe, ye = hist2d(x[high_vtan], y[high_vtan], xr, yr, c.npix_ecc, c.npix_ecc)
            lev = finite_percentile(hh[hh > 0], (95, 99))
            if np.all(np.isfinite(lev)):
                axis.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hh.T, levels=lev, colors="k", linewidths=0.7)
        axis.set_xlim(xr)
        axis.set_ylim(yr)
        label_axes(axis, "[Fe/H]", ylabel)
    right_label = {"fontsize": 9.5, "fontweight": "semibold"}
    ax[3].text(-0.9, 0.335, "Splash", **right_label)
    ax[3].text(-1.52, 0.34, "Aurora", **right_label)
    ax[3].text(-1.42, 0.15, "GS/E", **right_label)
    ax[3].text(-0.81, 0.2, "Eos", **right_label)
    ax[7].text(-1.02, 0.26, "Splash", **right_label)
    ax[7].text(-1.55, 0.075, "Aurora", **right_label)
    ax[7].text(-1.45, -0.27, "GS/E", **right_label)
    ax[7].text(-0.93, -0.07, "Eos", **right_label)
    return save(fig, outdir, "eos_mg_feh_ecc")


def plot_mg_feh_ecc_pixels(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR, bins=None, min_count=1, output_name="eos_mg_feh_ecc_pixels"):
    cat, c, m = load_context(cache)
    if bins is None:
        bins = (c.npix_ecc, c.npix_ecc)
    fig, ax = setup_axes(4, nrows=2, figsize=(12.75, 6))
    w = m["ecc"]
    high_vtan = m["base"] & (cat["galvt"] > 220)
    point_size = 6.0

    panels = [
        (cat["fe_h"], cat["mg_fe"], None, c.fehr3, c.mgfer_plot, None, "[Mg/Fe]", None),
        (cat["fe_h"], cat["mg_fe"], cat["al_fe"], c.fehr3, c.mgfer_plot, c.mm_al, "[Mg/Fe]", "[Al/Fe]"),
        (cat["fe_h"], cat["mg_fe"], cat["rap"], c.fehr3, c.mgfer_plot, c.mm_rapo2, "[Mg/Fe]", "r apo"),
        (cat["fe_h"], cat["mg_fe"], None, c.fehr3, c.mgfer_plot, None, "[Mg/Fe]", None),
        (cat["fe_h"], cat["al_fe"], None, c.fehr3, c.alfer_plot, None, "[Al/Fe]", None),
        (cat["fe_h"], cat["al_fe"], cat["mg_fe"], c.fehr3, c.alfer_plot, c.mm_mg, "[Al/Fe]", "[Mg/Fe]"),
        (cat["fe_h"], cat["al_fe"], cat["rap"], c.fehr3, c.alfer_plot, c.mm_rapo2, "[Al/Fe]", "r apo"),
        (cat["fe_h"], cat["al_fe"], None, c.fehr3, c.alfer_plot, None, "[Al/Fe]", None),
    ]
    for i, (x, y, color, xr, yr, mm, ylabel, cbar_label) in enumerate(panels):
        axis = ax[i]
        if color is None:
            axis.scatter(x[w], y[w], s=point_size, c="0.0" if i not in (3, 7) else "0.65", rasterized=True, linewidths=0)
        else:
            mean, xe, ye = stat2d(x[w], y[w], color[w], xr, yr, bins[0], bins[1], statistic="mean")
            counts, _, _ = hist2d(x[w], y[w], xr, yr, bins[0], bins[1])
            mean = np.clip(mean, mm[0], mm[1])
            im = value_panel(axis, mean, xe, ye, mm[0], mm[1], mask=counts < min_count, cmap="RdYlBu_r")
            cax = axis.inset_axes([0.08, 0.08, 0.82, 0.035])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_label(cbar_label, fontsize=8)
            cb.ax.xaxis.set_label_position("top")
            cb.ax.xaxis.set_ticks_position("bottom")
            cb.ax.tick_params(labelsize=7, length=2)
        if i in (3, 7):
            hh, xe, ye = hist2d(x[high_vtan], y[high_vtan], xr, yr, c.npix_ecc, c.npix_ecc)
            lev = finite_percentile(hh[hh > 0], (95, 99))
            if np.all(np.isfinite(lev)):
                axis.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hh.T, levels=lev, colors="k", linewidths=0.7)
        axis.set_xlim(xr)
        axis.set_ylim(yr)
        label_axes(axis, "[Fe/H]", ylabel)
    right_label = {"fontsize": 9.5, "fontweight": "semibold"}
    ax[3].text(-0.9, 0.335, "Splash", **right_label)
    ax[3].text(-1.52, 0.34, "Aurora", **right_label)
    ax[3].text(-1.42, 0.15, "GS/E", **right_label)
    ax[3].text(-0.81, 0.2, "Eos", **right_label)
    ax[7].text(-1.02, 0.26, "Splash", **right_label)
    ax[7].text(-1.55, 0.075, "Aurora", **right_label)
    ax[7].text(-1.45, -0.27, "GS/E", **right_label)
    ax[7].text(-0.93, -0.07, "Eos", **right_label)
    return save(fig, outdir, output_name)


@figure("eos_mg_rap_slice")
def plot_mg_rap_slice(cache=DEFAULT_CACHE, outdir=DEFAULT_OUTDIR):
    cat, c, m = load_context(cache)
    fig, ax = setup_axes(2, figsize=(12, 4))
    wf = (cat["fe_h"] > c.feh_tri_cut[0]) & (cat["fe_h"] < c.feh_tri_cut[1])
    ww = m["base"] & (cat["galvt"] < c.vt_sep) & wf
    wf2 = (cat["fe_h"] > -1.5) & (cat["fe_h"] < -1.1)
    ww2 = m["base"] & (cat["galvt"] < c.vt_sep) & wf2

    point_size = 5.0
    ax[0].scatter(cat["mg_fe"][ww], cat["rap"][ww], s=point_size, c="0.0", rasterized=True, linewidths=0)
    x, _, p95, _, _ = bin_percentile(cat["mg_fe"][ww], cat["rap"][ww], c.mgfer, 14, 95)
    use_curve = x >= -0.1
    ax[0].plot(x[use_curve], p95[use_curve], "k-", lw=1)
    ax[0].set_xlim(c.mgfer)
    ax[0].set_ylim(c.rapor)
    slice_label = {"fontsize": 11, "fontweight": "semibold"}
    ax[0].text(0.015, 33, "GS/E", **slice_label)
    ax[0].text(0.17, 24, "Eos", **slice_label)
    ax[0].text(0.39, 16.5, "Splash", **slice_label)
    label_axes(ax[0], "[Mg/Fe]", r"$r_{\rm apo}$ [kpc]", f"{c.feh_tri_cut[0]:.1f}<[Fe/H]<{c.feh_tri_cut[1]:.1f}")

    ax[1].scatter(cat["al_fe"][ww2], cat["rap"][ww2], s=point_size, c="0.0", rasterized=True, linewidths=0)
    x, _, p95, _, _ = bin_percentile(cat["al_fe"][ww2], cat["rap"][ww2], c.alfer, 12, 95)
    use_curve = x >= -0.5
    ax[1].plot(x[use_curve], p95[use_curve], "k-", lw=1)
    ax[1].set_xlim(c.alfer)
    ax[1].set_ylim(c.rapor)
    ax[1].text(-0.13, 38, "GS/E", **slice_label)
    ax[1].text(0.05, 20, "Aurora", **slice_label)
    label_axes(ax[1], "[Al/Fe]", r"$r_{\rm apo}$ [kpc]", "-1.5<[Fe/H]<-1.1")
    return save(fig, outdir, "eos_mg_rap_slice")
