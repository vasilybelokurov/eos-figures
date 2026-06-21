from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
import astropy.units as u

from .config import DEFAULT_ASTRONN, DEFAULT_APOGEE, DEFAULT_CACHE, DEFAULT_LIST_DIR, Cuts


APOGEE_COLUMNS = [
    "APOGEE_ID",
    "RA",
    "DEC",
    "PROGRAMNAME",
    "FE_H",
    "FE_H_ERR",
    "MG_FE",
    "MG_FE_ERR",
    "MN_FE_ERR",
    "AL_FE",
    "AL_FE_ERR",
    "C_FE_ERR",
    "CR_FE_ERR",
    "O_FE_ERR",
    "N_FE_ERR",
    "NI_FE_ERR",
    "SI_FE_ERR",
    "LOGG",
]

ASTRONN_COLUMNS = [
    ("GALVR_ERR", "galvr_err"),
    ("GALVT_ERR", "galvt_err"),
    ("GALVZ_ERR", "galvz_err"),
    ("WEIGHTED_DIST", "weighted_dist"),
    ("ENERGY", "Energy"),
    ("LZ", "Lz"),
    ("AGE", "age"),
    ("AGE_MODEL_ERROR", "age_model_error"),
    ("GALVT", "galvt"),
    ("RAP", "rap"),
    ("RPERI", "rperi"),
]


def _contains(strings: np.ndarray, token: str) -> np.ndarray:
    return np.char.find(strings.astype(str), token) >= 0


def apogee_quality_mask(table) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    for token in ("STAR_BAD", "TEFF_BAD", "LOGG_BAD"):
        mask &= ~_contains(table["ASPCAPFLAGS"], token)
    for token in (
        "VERY_BRIGHT_NEIGHBOR",
        "LOW_SNR",
        "PERSIST_HIGH",
        "PERSIST_JUMP_POS",
        "PERSIST_JUMP_NEG",
        "SUSPECT_RV_COMBINATION",
    ):
        mask &= ~_contains(table["STARFLAGS"], token)
    mask &= (np.asarray(table["EXTRATARG"]) & (1 << 4)) == 0
    mask &= np.isfinite(table["RA"]) & np.isfinite(table["DEC"])
    return mask


def build_cache(
    apogee_path: Path = DEFAULT_APOGEE,
    astronn_path: Path = DEFAULT_ASTRONN,
    cache_path: Path = DEFAULT_CACHE,
    match_tol_arcsec: float = 0.001,
    overwrite: bool = False,
) -> Path:
    """Build the compact matched catalogue used by the plotting scripts."""
    apogee_path = Path(apogee_path).expanduser()
    astronn_path = Path(astronn_path).expanduser()
    cache_path = Path(cache_path).expanduser()
    if cache_path.exists() and not overwrite:
        return cache_path
    if not apogee_path.exists():
        raise FileNotFoundError(f"Missing APOGEE allStarLite file: {apogee_path}")
    if not astronn_path.exists():
        raise FileNotFoundError(f"Missing AstroNN file: {astronn_path}")

    with fits.open(apogee_path, memmap=True) as hdul_d, fits.open(astronn_path, memmap=True) as hdul_a:
        d = hdul_d[1].data
        ann = hdul_a[1].data
        keep = apogee_quality_mask(d)
        d_idx = np.flatnonzero(keep)

        ann_good = np.isfinite(ann["RA_APOGEE"]) & np.isfinite(ann["DEC_APOGEE"])
        ann_idx = np.flatnonzero(ann_good)
        c_d = SkyCoord(d["RA"][d_idx] * u.deg, d["DEC"][d_idx] * u.deg)
        c_a = SkyCoord(ann["RA_APOGEE"][ann_idx] * u.deg, ann["DEC_APOGEE"][ann_idx] * u.deg)
        a_match, sep, _ = match_coordinates_sky(c_d, c_a)
        matched = sep.arcsec < match_tol_arcsec
        d_idx = d_idx[matched]
        a_idx = ann_idx[a_match[matched]]

        cols = []
        for name in APOGEE_COLUMNS:
            arr = np.asarray(d[name][d_idx])
            cols.append(fits.Column(name=name.lower(), array=arr, format=_fits_format(arr)))
        for out_name, in_name in ASTRONN_COLUMNS:
            arr = np.asarray(ann[in_name][a_idx])
            cols.append(fits.Column(name=out_name.lower(), array=arr, format=_fits_format(arr)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["APOGEE"] = str(apogee_path)
    hdu.header["ASTRONN"] = str(astronn_path)
    hdu.header["MATCHTOL"] = (match_tol_arcsec, "arcsec")
    hdu.writeto(cache_path, overwrite=overwrite)
    return cache_path


def _fits_format(arr: np.ndarray) -> str:
    if arr.dtype.kind in "SU":
        width = max(1, int(arr.dtype.itemsize // (4 if arr.dtype.kind == "U" else 1)))
        return f"{width}A"
    if arr.dtype.kind == "i":
        return "K" if arr.dtype.itemsize > 4 else "J"
    if arr.dtype.kind == "f":
        return "D" if arr.dtype.itemsize > 4 else "E"
    raise TypeError(f"Unsupported FITS column dtype: {arr.dtype}")


def load_catalog(path: Path = DEFAULT_CACHE):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing matched catalogue cache: {path}. Run scripts/build_eos_catalog.py first."
        )
    with fits.open(path, memmap=False) as hdul:
        return hdul[1].data


def _read_compiled_satellites(list_dir: Path = DEFAULT_LIST_DIR) -> np.ndarray:
    path = Path(list_dir).expanduser() / "CompiledSatCatalogv2_gabriel.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.rec.fromarrays([data["ra"], data["dec"]], names="ra,dec")


def satellite_out_mask(ra: np.ndarray, dec: np.ndarray, radius_deg: float = 1.0) -> np.ndarray:
    """Mimic cutradec(..., /gcs, /dwa, /sdss, /comp), where /comp overwrites loc."""
    sats = _read_compiled_satellites()
    stars = SkyCoord(ra * u.deg, dec * u.deg)
    locs = SkyCoord(sats.ra * u.deg, sats.dec * u.deg)
    idx_star, _, _, _ = locs.search_around_sky(stars, radius_deg * u.deg)
    mask = np.ones(len(ra), dtype=bool)
    mask[np.unique(idx_star)] = False
    return mask


def make_masks(cat, cuts: Cuts = Cuts()) -> dict[str, np.ndarray]:
    c = cuts
    finite_core = np.isfinite(cat["fe_h"]) & np.isfinite(cat["mg_fe"]) & np.isfinite(cat["al_fe"])
    noclouds = np.char.lower(cat["programname"].astype(str)) != "magclouds"
    velerr = (
        (cat["galvr_err"] < c.velerr_lim)
        & (cat["galvt_err"] < c.velerr_lim)
        & (cat["galvz_err"] < c.velerr_lim)
    )
    chemerr = (
        (cat["fe_h_err"] < c.feh_err_lim)
        & (cat["mg_fe_err"] < c.chem_err_lim)
        & (cat["mn_fe_err"] < c.chem_err_lim)
        & (cat["al_fe_err"] < c.chem_err_lim)
        & (cat["c_fe_err"] < c.chem_err_lim)
        & (cat["cr_fe_err"] < c.chem_err_lim)
        & (cat["o_fe_err"] < c.chem_err_lim)
        & (cat["n_fe_err"] < c.chem_err_lim)
        & (cat["ni_fe_err"] < c.chem_err_lim)
        & (cat["si_fe_err"] < c.chem_err_lim)
    )
    logg = cat["logg"] < c.logg_lim
    dist = cat["weighted_dist"] < 15e3
    out = satellite_out_mask(cat["ra"], cat["dec"])
    base = finite_core & velerr & chemerr & logg & out & dist & noclouds

    encut = (cat["energy"] > c.en_lim[0]) & (cat["energy"] < c.en_lim[1])
    encut_acc = cat["energy"] > c.en_lim_acc
    lz_acc = np.abs(cat["lz"]) < c.lz_lim_acc
    age_err = (cat["age_model_error"] / cat["age"]) < c.age_err_frac

    mg_in = cat["mg_fe"] > c.slope_acc * cat["fe_h"] + c.inter_acc
    al_thin = cat["al_fe"] < c.kalfe * cat["fe_h"] + c.offalfe
    al_insitu = cat["al_fe"] > c.alfe_cut
    al_acc = cat["al_fe"] < c.alfe_cut
    feh_acc = cat["fe_h"] < c.feh_acc_cut
    mg_acc = cat["mg_fe"] < c.slope_acc * cat["fe_h"] + c.inter_acc
    mg_thick = cat["mg_fe"] > c.slope_acc2 * cat["fe_h"] + c.inter_acc2
    mg_thin = cat["mg_fe"] < c.slope_acc2 * cat["fe_h"] + c.inter_acc2

    masks = {
        "base": base,
        "base_en": base & encut,
        "base_en_age": base & encut & age_err,
        "acc": base & mg_acc & mg_thin & encut_acc & lz_acc,
        "acc_al": base & mg_acc & mg_thin & al_acc & feh_acc,
        "thin": base & mg_in & mg_thin & al_thin,
        "thick": base & mg_thick,
    }
    masks["thin_en"] = masks["thin"] & encut
    masks["thick_en"] = masks["thick"] & encut
    masks["acc_en"] = masks["acc"] & encut
    masks["thin_al"] = base & mg_in & mg_thin & al_thin & al_insitu
    masks["thick_al"] = base & mg_thick & al_insitu
    masks["thin_al_age"] = masks["thin_al"] & age_err
    masks["thin_age"] = masks["thin"] & age_err
    masks["thick_age"] = masks["thick"] & age_err
    masks["thick_al_age"] = masks["thick_al"] & age_err
    masks["tri"] = masks["base_en"] & (cat["fe_h"] > c.feh_tri_cut[0]) & (cat["fe_h"] < c.feh_tri_cut[1])
    vt = cat["galvt"] < c.vt_sep
    rap = cat["rap"] > c.rap_min
    rap_age = cat["rap"] > c.rap_min_age
    masks["thin_al_vt_age"] = masks["thin_al_age"] & vt
    masks["thin_al_vt_rap_age"] = masks["thin_al_vt_age"] & rap_age
    feh_splash = cat["fe_h"] > c.feh_splash_min
    masks["thick_al_splash_age"] = masks["thick_al_age"] & vt & feh_splash
    masks["ecc"] = base & vt & rap
    return masks
