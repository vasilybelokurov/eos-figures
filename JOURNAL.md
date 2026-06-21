# JOURNAL

## Rules of Engagement

- Prefer writing reusable Python scripts for analyses, data processing, and other repeatable work instead of one-off shell snippets or notebook-only logic.
- Keep this `JOURNAL.md` regularly updated with detailed notes on what was done, why it was done, commands or scripts used, outputs produced, and any open follow-up items.

## 2026-06-19

### Session: Establish Working Rules

- Recorded standing project rules of engagement:
  - Write Python scripts for future reuse wherever practical.
  - Maintain a detailed `JOURNAL.md` and update it regularly as work proceeds.
- Confirmed that no existing `JOURNAL.md` was present before creating this file.

### Session: Start IDL-to-Python Conversion for Eos Paper Figures

- Used the `overleaf-sync` skill because the target paper is an Overleaf project.
- Cloned the Overleaf project `https://www.overleaf.com/project/623c784beb3193c951d1a693` into `overleaf-paper/`.
- Inspected `overleaf-paper/nrla.tex` and confirmed that the paper currently includes six figure PDFs:
  - `img/eos_energy_mg_al.pdf`
  - `img/eos_alfe_pops.pdf`
  - `img/eos_energy_pops.pdf`
  - `img/eos_alfe_3pops.pdf`
  - `img/eos_vphi.pdf`
  - `img/eos_age.pdf`
- Inspected the IDL source:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
  - Found eight output figures in the IDL workflow, including the six paper figures plus `eos_mg_feh_ecc` and `eos_mg_rap_slice`.
- Located the real data required for testing:
  - APOGEE allStarLite DR17: `/Users/vasilybelokurov/data/apogee/dr17/allStarLite-dr17-synspec_rev1.fits`
  - AstroNN DR17: `/Users/vasilybelokurov/data/apogee/dr17/apogee_astroNN-DR17.fits`
  - Existing smaller combined files were checked but are not sufficient for exact reproduction because they omit several fields used by the IDL cuts and plots.
- Added reusable Python package files:
  - `eos_figures/config.py`
  - `eos_figures/data.py`
  - `eos_figures/stats.py`
  - `eos_figures/plotting.py`
  - `eos_figures/figures.py`
- Added reusable scripts:
  - `scripts/build_eos_catalog.py`
  - `scripts/plot_eos_figure.py`
  - one wrapper script per IDL output figure.
- Built the matched real-data cache:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/build_eos_catalog.py --overwrite`
  - Output: `data/eos_apogee_dr17_lite_ann.fits`
  - Size: 119 MB
  - Matched rows: 562,603
  - Columns: 29
- Validated the IDL-style selection counts from the cache:
  - `base`: 210,392
  - `acc`: 2,584
  - `acc_al`: 3,127
  - `thin`: 143,018
  - `thick`: 60,513
  - `thin_al`: 141,288
  - `thick_al`: 60,301
  - `base_en`: 78,827
  - `base_en_age`: 67,786
  - `thin_al_age`: 107,906
  - `ecc`: 5,724
- Generated all eight converted figures from real data into `products/test_figures`:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_figure.py all --outdir products/test_figures`
  - Produced all expected PDFs successfully.
  - Runtime warnings from NumPy/SciPy appeared for empty or non-finite binned-statistic cells, but the command completed.
- Verified one per-figure wrapper:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_energy_pops.py --outdir products/test_wrapper`
  - Produced `products/test_wrapper/eos_energy_pops.pdf`.
- Added `.gitignore` entries for generated cache, products, Python bytecode, and local Matplotlib/font caches.
- Added `README.md` with data paths and commands.
- Did not overwrite the existing PDFs in `overleaf-paper/img` during this first validation run. The plotting scripts default to `overleaf-paper/img` when `--outdir` is omitted.

### Session: Fix `eos_alfe_pops` Python/IDL Mismatch

- Compared `products/test_figures/eos_alfe_pops.pdf` against the IDL/Overleaf counterpart `overleaf-paper/img/eos_alfe_pops.pdf`.
- Confirmed two problems in the Python version:
  - Matplotlib autoscaled the y-range after drawing the diagonal selection cut, unlike IDL which clips overlays to the pre-set plot range.
  - The bottom-row median-velocity maps only masked empty bins, while the IDL code masks low-density bins using the top-row density image and `white_lim = 200`.
- Inspected the local IDL helper files:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/lib/vasily/hist2d.pro`
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/lib/vasily/med2d.pro`
- Updated `eos_figures/figures.py`:
  - Reapplied `xlim = fehr` and `ylim = alfer` after overlays in `plot_alfe_pops`.
  - Added `_idl_low_density_mask()` to mimic the IDL `w0_*` masks generated from scaled density images.
  - Clipped median-map input velocities to `vtanr = [-200, 400]`, matching `med2d(..., vr=vtanr)`.
  - Required more than two velocity samples per bin before accepting the median, matching the IDL `med2d` behaviour.
  - Switched the median-map colour table to a closer blue-to-red Matplotlib palette.
- Regenerated the corrected PDF:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_alfe_pops.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_alfe_pops.pdf`
- Rendered the corrected PDF and checked visually against the Overleaf reference.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_alfe_3pops` Colour Map and Right Panel Density

- Compared `products/test_figures/eos_alfe_3pops.pdf` against `overleaf-paper/img/eos_alfe_3pops.pdf`.
- User noted that:
  - the middle panel should use the same blue-to-red palette as the bottom row of `eos_alfe_pops`;
  - the rightmost 2D density looked wrong;
  - the 1D histogram/profile underneath the rightmost density was missing.
- Inspected the relevant IDL block in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Found that the rightmost panel uses:
  - `den_mg_al = hist2d(..., xnorm = 1)`, i.e. density normalized across `[Al/Fe]` at fixed `[Mg/Fe]`;
  - linear scaled density, not log density;
  - `den_mg_al0 <= 1` masking;
  - a 1D profile made from `total((255-im_byt), 2)`.
- Updated `eos_figures/figures.py`:
  - Changed the middle panel colour map to `RdYlBu_r`, matching the fixed `eos_alfe_pops` velocity-map style.
  - Changed the middle panel colour bar to a small horizontal inset, closer to the IDL/Overleaf version.
  - Changed the rightmost density panel from log density to linearly scaled, IDL-style `xnorm` density.
  - Rebuilt the 1D black profile from the scaled right-panel image.
  - Reapplied the intended `alfer2` and `mgfer2` axis limits.
- Regenerated the corrected PDF:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_alfe_3pops.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_alfe_3pops.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_alfe_3pops` First-Two-Panel Y-Axis Ranges

- User noted that the y-axis range in the first two panels of `products/test_figures/eos_alfe_3pops.pdf` was wrong.
- Inspected the IDL `eos_alfe_3pops` plotting block in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Confirmed that IDL uses `yr = mgfer2` for the first two `[Fe/H]` vs `[Mg/Fe]` panels, with `mgfer2 = [-0.15, 0.4]`.
- Found that the Python histograms already used `c.mgfer2`, but Matplotlib autoscaled the first two panels after drawing the separator lines, because one line extends below the lower y-limit.
- Updated `eos_figures/figures.py`:
  - Reapplied `xlim = c.fehr2` and `ylim = c.mgfer2` after the separator-line overlays in panels 1 and 2.
  - This mirrors the IDL `xs=1, ys=1` fixed-axis behaviour.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_alfe_3pops.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_alfe_3pops.pdf`
- Rendered the regenerated PDF to PNG and checked that the first two panels now use the intended `[Mg/Fe]` range.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_vphi` Axis Ranges, Point Size, and Palette

- User requested fixes to `products/test_figures/eos_vphi.pdf`:
  - axis ranges/ticks;
  - point size;
  - colour palette.
- Inspected the IDL `eos_vphi` block in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Confirmed that the IDL figure uses:
  - `fehr_plot = [-1.5, 0.55]`;
  - `vphir_plot2 = [-200, 370]`;
  - colour table 72 reversed for the `r apo`, `r peri`, and `Age` panels;
  - horizontal in-panel colour bars.
- Updated `eos_figures/figures.py`:
  - Increased the `eos_vphi` scatter point size from `s = 1.2` to `s = 4.5`.
  - Changed the colour map from the default `viridis_r` to `RdYlBu_r`, matching the established IDL-style blue-to-red palette used elsewhere in the converted figures.
  - Replaced vertical colour bars with horizontal inset colour bars in each panel.
  - Reapplied the IDL x/y ranges and explicit ticks: `[Fe/H]` ticks at `-1.5, -1.0, -0.5, 0.0, 0.5`, and `Vtan` ticks at `-200, -100, 0, 100, 200, 300`.
- Follow-up correction after user feedback:
  - Checked `/Users/vasilybelokurov/Dropbox (Personal)/idl/lib/vasily/sc01.pro`.
  - Confirmed that `sc01(value, mm=...)` linearly maps to `[0, 1]` and clips values outside the supplied `mm` interval.
  - Explicitly clipped each Python colour array to the corresponding IDL interval before plotting:
    - `r apo`: `mm_rapo = [1, 17]`;
    - `r peri`: `mm_rperi = [1, 17]`;
    - `Age`: `mm_age = [3, 10]`.
  - Forced colour-bar ticks to the IDL endpoint-inclusive five-tick layout:
    - `1.0, 5.0, 9.0, 13.0, 17.0` for `r apo` and `r peri`;
    - `3.0, 4.8, 6.5, 8.2, 10.0` for `Age`.
- Follow-up correction for point draw order:
  - Confirmed that the IDL `plots` calls do not sort by colour; they plot the selected catalogue rows in their existing order.
  - The Python helper `smoothed_scatter_panel()` had been sorting by colour with `np.argsort(c)`, which changed which points appeared on top.
  - Added a reusable `sort_by_color` argument to `smoothed_scatter_panel()`, defaulting to the old behaviour for other figures.
  - Set `sort_by_color=False` for `eos_vphi` to match the IDL point ordering.
- Added explicit sorted/unsorted figure variants after user requested both versions:
  - Updated `plot_vphi()` to accept `sort_by_color` and `output_name`, while keeping the generic `eos_vphi` default as IDL-like unsorted.
  - Replaced `scripts/plot_eos_vphi.py` with a reusable CLI supporting `--variant both`, `--variant unsorted`, and `--variant sorted`.
  - The default wrapper command now writes both `eos_vphi_unsorted.pdf` and `eos_vphi_sorted.pdf`.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_vphi.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_vphi_unsorted.pdf`
  - Output: `products/test_figures/eos_vphi_sorted.pdf`
- Rendered the regenerated PDFs to PNG and checked that the only intended visual difference is point draw order.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Trim `eos_mg_rap_slice` Percentile Curves

- User requested changing the minimum plotted x value of the percentile curves in `products/test_figures/eos_mg_rap_slice.pdf`:
  - left panel curve should start at `[Mg/Fe] = -0.1`;
  - right panel curve should start at `[Al/Fe] = -0.5`.
- Updated `eos_figures/figures.py`:
  - Added a curve mask `x >= -0.1` before plotting the left-panel 95th-percentile `r apo` curve.
  - Added a curve mask `x >= -0.5` before plotting the right-panel 95th-percentile `r apo` curve.
  - Left the scatter data and panel axis limits unchanged.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_rap_slice.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_mg_rap_slice.pdf`
- Rendered the regenerated PDF to PNG and checked that the curve starts match the requested x values.

### Session: Emphasize `eos_mg_rap_slice` Substructure Labels

- User requested larger and bolder substructure labels in both panels of `products/test_figures/eos_mg_rap_slice.pdf`.
- Updated `eos_figures/figures.py`:
  - Added shared `slice_label` styling with `fontsize = 11` and `fontweight = semibold`.
  - Applied it to `GS/E`, `Eos`, `Splash`, and `Aurora` labels across both panels.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_rap_slice.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_mg_rap_slice.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Align `eos_mg_rap_slice` Left-Panel Metallicity Slice with `eos_alfe_3pops`

- Revisited the left-panel `[Fe/H]` selection for `products/test_figures/eos_mg_rap_slice.pdf`.
- User noted that the selection should likely be `-0.8 < [Fe/H] < -0.6` because this matches the right panel of `eos_alfe_3pops.pdf`.
- Agreed that using the shared slice is more internally consistent, even though the current IDL source has an older `-0.8 < [Fe/H] < -0.5` line.
- Updated `eos_figures/figures.py`:
  - Changed the left-panel selection to use `c.feh_tri_cut`, i.e. `[-0.8, -0.6]`.
  - Updated the left-panel title to derive from `c.feh_tri_cut`.
  - Kept the physically meaningful `[Mg/Fe]` x-axis and did not reproduce the old IDL display-range bug.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_rap_slice.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_mg_rap_slice.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_mg_rap_slice` Point Size and Left-Panel Range

- Compared `products/test_figures/eos_mg_rap_slice.pdf` against `overleaf-paper/img/eos_mg_rap_slice.pdf`.
- User noted that point sizes needed updating and asked whether the left-panel `[Fe/H]` range was correct.
- Inspected the relevant IDL block in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Confirmed from the current IDL source that the left-panel stellar selection is:
  - `-0.8 < [Fe/H] < -0.5`
- Noted that the old Overleaf/reference PDF shows `-0.8 < [Fe/H] < -0.6` and also has an IDL display-range issue on the left panel x-axis. User explicitly said not to reproduce the wrong IDL x-axis range.
- Updated `eos_figures/figures.py`:
  - Increased the point size in both panels from `s=1.2` to `s=5.0`.
  - Kept the physically meaningful left-panel `[Mg/Fe]` x-axis range `mgfer = [-0.15, 0.5]`.
  - Kept the current IDL-source left-panel metallicity slice and title `-0.8<[Fe/H]<-0.5`.
- Regenerated the corrected PDF:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_rap_slice.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_mg_rap_slice.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_mg_feh_ecc` Point Size, Palette, and Right-Column Labels

- Compared `products/test_figures/eos_mg_feh_ecc.pdf` against `overleaf-paper/img/eos_mg_feh_ecc.pdf`.
- User noted that the point sizes and colour palette needed to be fixed, then requested slightly larger and bolder labels in the rightmost column.
- Inspected the corresponding IDL blocks in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Confirmed that IDL uses larger rasterized points (`sz_rast2 = 1.5`) and colour table 72 reversed for `[Al/Fe]`, `[Mg/Fe]`, and `r apo` panels.
- Updated `eos_figures/figures.py`:
  - Increased point size in `plot_mg_feh_ecc`.
  - Switched coloured panels to `RdYlBu_r`.
  - Replaced vertical colour bars with horizontal inset colour bars.
  - Made rightmost-column component labels slightly larger and semibold.
- Regenerated the corrected PDF:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_feh_ecc.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_mg_feh_ecc.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

### Session: Fix `eos_age` Middle Panel Marker Size and Palette

- Compared `products/test_figures/eos_age.pdf` against `overleaf-paper/img/eos_age.pdf`.
- User noted that the middle panel points should be larger and should use the correct colour palette.
- Inspected the IDL age-figure block in:
  - `/Users/vasilybelokurov/Dropbox (Personal)/idl/work/apogee_dr17_eos/eos_first_figures.pro`
- Confirmed that the middle panel colour-codes `r apo` with IDL colour table 72 reversed and uses visibly larger rasterized points (`symsize = 2.2` before image filtering).
- Updated `eos_figures/figures.py`:
  - Changed the middle-panel scatter colour map to `RdYlBu_r`, so small `r apo` is blue and large `r apo` is red/yellow.
  - Increased the Matplotlib marker size from `s=2.2` to `s=13`.
  - Replaced the vertical colour bar with a small horizontal inset colour bar labelled `r apo`, closer to the IDL/Overleaf panel.
- Regenerated the corrected PDF:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_age.py --outdir products/test_figures`
  - Output: `products/test_figures/eos_age.pdf`
- Rendered the regenerated PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.

## 2026-06-21

### Session: Tidy Generated Products Folder

- User requested tidying the `products/` folder because many test folders were no longer needed.
- Inspected the generated-output tree and found:
  - current working figure set in `products/test_figures/`;
  - stale one-figure comparison folders: `products/compare_*`;
  - obsolete wrapper-test folder: `products/test_wrapper/`;
  - local macOS metadata file: `products/.DS_Store`.
- Renamed the active output directory:
  - from `products/test_figures/`;
  - to `products/figures/`.
- Updated `README.md` so example generation commands now use `--outdir products/figures`.
- Planned cleanup of stale generated folders:
  - `products/compare_age`;
  - `products/compare_alfe_3pops`;
  - `products/compare_alfe_3pops_v2`;
  - `products/compare_alfe_3pops_v3`;
  - `products/compare_alfe_pops`;
  - `products/compare_alfe_pops_v2`;
  - `products/compare_mg_feh_ecc`;
  - `products/compare_mg_rap_slice`;
  - `products/compare_mg_rap_slice_v2`;
  - `products/test_wrapper`;
  - `products/.DS_Store`.
- Removed those stale folders/files after approval.
- Verified the cleaned `products/` tree now contains only:
  - `products/figures/`;
  - current generated PDFs for the converted paper figures, including sorted and unsorted `eos_vphi` variants.
- Current disk usage:
  - `products/`: 3.3 MB;
  - `products/figures/`: 3.3 MB.

### Session: Move Generated Figures to Root-Level `figures/`

- User noted that `products/figures` was unnecessary and requested making figures in the repository root.
- Checked that no root-level `figures/` directory already existed.
- Moved:
  - from `products/figures/`;
  - to `figures/`.
- Removed the leftover `products/.DS_Store` and the now-empty `products/` directory.
- Updated `README.md` examples so generated figures use `--outdir figures`.
- Updated `.gitignore` to ignore `figures/` instead of `products/`.

### Session: Add Pixelated Mean-Property Version of `eos_vphi`

- User requested a pixelated version of `figures/eos_vphi_sorted.pdf` showing mean properties per pixel in `[Fe/H]`-`Vtan` space.
- Added reusable plotting function `plot_vphi_pixels()` in `eos_figures/figures.py`.
- Implementation choices:
  - Uses the same three panel selections as `eos_vphi`:
    - `thin_al` for mean `r apo`;
    - `thin_al` for mean `r peri`;
    - `thin_al_age` for mean `Age`.
  - Uses `stat2d(..., statistic="mean")` to compute binned mean values in fixed `[Fe/H]` and `Vtan` bins.
  - Uses `hist2d()` counts to mask empty pixels.
  - Uses default `70 x 70` bins over the same axis ranges as `eos_vphi`:
    - `[Fe/H] = [-1.5, 0.55]`;
    - `Vtan = [-200, 370]`.
  - Uses unsmoothed nearest-neighbour `imshow()` rendering so the output is genuinely pixelated.
  - Reuses the same IDL-style colour ranges and ticks:
    - `r apo`: `[1, 17]`;
    - `r peri`: `[1, 17]`;
    - `Age`: `[3, 10]`.
- Added reusable script:
  - `scripts/plot_eos_vphi_pixels.py`
  - Supports `--xbins`, `--ybins`, `--min-count`, and `--name`.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_vphi_pixels.py --outdir figures`
  - Output: `figures/eos_vphi_pixels.pdf`
- Rendered the PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.
- Updated `README.md` with the new pixel-map generation command.

### Session: Add Pixelated Mean-Property Version of `eos_mg_feh_ecc`

- User requested a pixelated version of `figures/eos_mg_feh_ecc.pdf`, with the second and third panels in each row showing pixelated means.
- Added reusable plotting function `plot_mg_feh_ecc_pixels()` in `eos_figures/figures.py`.
- Implementation choices:
  - Preserves the original `2 x 4` layout.
  - Keeps columns 1 and 4 in each row as context panels:
    - black/grey scatter in columns 1 and 4;
    - high-`Vtan` contours and substructure labels in column 4.
  - Replaces columns 2 and 3 in each row with binned mean maps:
    - top row, column 2: mean `[Al/Fe]` in `[Fe/H]`-`[Mg/Fe]` pixels;
    - top row, column 3: mean `r apo` in `[Fe/H]`-`[Mg/Fe]` pixels;
    - bottom row, column 2: mean `[Mg/Fe]` in `[Fe/H]`-`[Al/Fe]` pixels;
    - bottom row, column 3: mean `r apo` in `[Fe/H]`-`[Al/Fe]` pixels.
  - Uses the same `ecc` mask as the original figure.
  - Uses `stat2d(..., statistic="mean")` for mean values and `hist2d()` counts to mask empty pixels.
  - Uses default `71 x 71` bins, matching the existing `npix_ecc` setting.
  - Uses unsmoothed nearest-neighbour `imshow()` rendering through `value_panel()`.
  - Reuses the original colour ranges:
    - `[Al/Fe]`: `[-0.4, 0.4]`;
    - `[Mg/Fe]`: `[0.0, 0.4]`;
    - `r apo`: `[7, 15]`.
- Added reusable script:
  - `scripts/plot_eos_mg_feh_ecc_pixels.py`
  - Supports `--xbins`, `--ybins`, `--min-count`, and `--name`.
- Regenerated:
  - Command: `/Users/vasilybelokurov/Work/venvs/.venv/bin/python scripts/plot_eos_mg_feh_ecc_pixels.py --outdir figures`
  - Output: `figures/eos_mg_feh_ecc_pixels.pdf`
- Rendered the PDF to PNG and checked the result visually.
- Re-ran Python bytecode compilation for all package and script files successfully.
- Updated `README.md` with the new pixel-map generation command.
