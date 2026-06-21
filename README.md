# Eos Figure Scripts

Reusable Python conversion of the IDL plotting workflow used for the Eos paper figures.

This repository includes:

- `eos_figures/`: reusable plotting and data helpers;
- `scripts/`: command-line wrappers for building the catalogue and regenerating figures;
- `data/eos_apogee_dr17_lite_ann.fits.gz`: compact matched APOGEE DR17 + AstroNN catalogue used by the plotting scripts;
- `figures/`: generated PDF figures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Included Data

The included cache is:

```text
data/eos_apogee_dr17_lite_ann.fits.gz
```

It contains 562,603 matched rows and 29 columns. The cache is gzipped so it can be stored in a normal GitHub repository without Git LFS; Astropy reads it directly.

The original full input catalogues used to build the cache were APOGEE allStarLite DR17 and AstroNN DR17. If those files are available locally, the cache can be rebuilt with:

```bash
python scripts/build_eos_catalog.py \
  --apogee /path/to/allStarLite-dr17-synspec_rev1.fits \
  --astronn /path/to/apogee_astroNN-DR17.fits \
  --output data/eos_apogee_dr17_lite_ann.fits.gz \
  --overwrite
```

## Generate Figures

Generate all converted paper-style figure PDFs:

```bash
python scripts/plot_eos_figure.py all
```

Generate one figure:

```bash
python scripts/plot_eos_energy_mg_al.py
```

Generate sorted and unsorted `eos_vphi` variants:

```bash
python scripts/plot_eos_vphi.py
```

Generate pixelated mean-property maps in `[Fe/H]`-`Vtan` space:

```bash
python scripts/plot_eos_vphi_pixels.py
```

Generate the pixelated mean-property variant of `eos_mg_feh_ecc`:

```bash
python scripts/plot_eos_mg_feh_ecc_pixels.py
```

All scripts write to `figures/` by default. Use `--outdir <path>` to write somewhere else.
