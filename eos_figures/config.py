from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_APOGEE = Path("/Users/vasilybelokurov/data/apogee/dr17/allStarLite-dr17-synspec_rev1.fits")
DEFAULT_ASTRONN = Path("/Users/vasilybelokurov/data/apogee/dr17/apogee_astroNN-DR17.fits")
DEFAULT_CACHE = REPO / "data" / "eos_apogee_dr17_lite_ann.fits.gz"
DEFAULT_OUTDIR = REPO / "figures"
DEFAULT_LIST_DIR = REPO / "data"


@dataclass(frozen=True)
class Cuts:
    white_lim: float = 200
    vtanr: tuple[float, float] = (-200, 400)
    mm_vtan: tuple[float, float] = (0, 250)
    ager: tuple[float, float] = (0, 12)
    age_bin: float = 0.7
    alfe_cut: float = -0.12
    feh_acc_cut: float = -0.5
    kalfe: float = 0.9
    offalfe: float = 0.9
    vt_sep: float = 80
    nbins_age: int = 15
    rap_min: float = 6
    rap_min_age: float = 9
    en_lim: tuple[float, float] = (-0.75e5, -0.4e5)
    en_lim_acc: float = -2e5
    lz_lim_acc: float = 0.5e3
    lzr: tuple[float, float] = (-2.8e3, 2.8e3)
    enr: tuple[float, float] = (-1.5, 0.25)
    age_err_frac: float = 0.2
    feh_splash_min: float = -0.9
    fehr: tuple[float, float] = (-2.1, 0.4)
    mgfer: tuple[float, float] = (-0.15, 0.5)
    mgfer2: tuple[float, float] = (-0.15, 0.4)
    mgfer_plot: tuple[float, float] = (-0.25, 0.5)
    alfer: tuple[float, float] = (-0.6, 0.6)
    alfer2: tuple[float, float] = (-0.6, 0.5)
    alfer_plot: tuple[float, float] = (-0.85, 0.65)
    mm_al: tuple[float, float] = (-0.4, 0.4)
    mm_mg: tuple[float, float] = (0.0, 0.4)
    nfeh: int = 55
    nmg: int = 70
    nal: int = 55
    nfe: int = 27
    nal2: int = 27
    fehr2: tuple[float, float] = (-1.7, 0.3)
    nfeh2: int = 51
    nmg2: int = 40
    nlz: int = 75
    nen: int = 125
    nlz2: int = 85
    nen2: int = 85
    velerr_lim: float = 50.0
    logg_lim: float = 3.0
    feh_err_lim: float = 0.2
    chem_err_lim: float = 0.2
    slope_acc: float = -0.3
    inter_acc: float = -0.1
    slope_acc2: float = -0.14
    inter_acc2: float = 0.135
    perc: tuple[float, float] = (40, 94)
    perc1: tuple[float, float] = (10, 99)
    perc2: tuple[float, float] = (50, 98)
    perc_elz: tuple[float, float] = (5, 95)
    perc_mgfe: tuple[float, float] = (3, 97)
    feh_tri_cut: tuple[float, float] = (-0.8, -0.6)
    fehr3: tuple[float, float] = (-2.1, 0.1)
    fehr_plot: tuple[float, float] = (-1.5, 0.55)
    vphir_plot2: tuple[float, float] = (-200, 370)
    mm_en: tuple[float, float] = (-1.3, -0.1)
    mm_age: tuple[float, float] = (3.0, 10.0)
    mm_rperi: tuple[float, float] = (1, 17)
    mm_rapo: tuple[float, float] = (1, 17)
    rapor: tuple[float, float] = (0, 45)
    zmaxr: tuple[float, float] = (0, 35)
    mm_rapo2: tuple[float, float] = (7, 15)
    npix_ecc: int = 71
