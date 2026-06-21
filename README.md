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

It contains 562,603 matched rows and 31 columns. The cache is gzipped so it can be stored in a normal GitHub repository without Git LFS; Astropy reads it directly.

The cache also includes two boolean selection columns used by the plotting masks:

- `satellite_out`: false for stars within 1 degree of objects in `CompiledSatCatalogv2_gabriel.csv`, mimicking the original IDL `cutradec(..., /gcs, /dwa, /sdss, /comp)` removal;
- `gc_member`: true for likely Vasiliev globular-cluster members, crossmatched within 1 arcsec and requiring membership probability `PROB >= 0.5`.

The base plotting mask requires `satellite_out` and `~gc_member`.

The original full input catalogues used to build the cache were APOGEE allStarLite DR17 and AstroNN DR17. If those files are available locally, the cache can be rebuilt with:

```bash
python scripts/build_eos_catalog.py \
  --apogee /path/to/allStarLite-dr17-synspec_rev1.fits \
  --astronn /path/to/apogee_astroNN-DR17.fits \
  --output data/eos_apogee_dr17_lite_ann.fits.gz \
  --overwrite
```

If rebuilding the cache from scratch, add the public-repo selection flags with:

```bash
python scripts/add_eos_cache_masks.py --cache data/eos_apogee_dr17_lite_ann.fits.gz
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
