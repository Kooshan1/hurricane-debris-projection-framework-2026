# Revision-figure generation scripts

These scripts post-process the existing Monte-Carlo and trained-NN outputs
into the new figures called for by the major-revision plan
(`prompts/responses/revision_plan_*.md` in the paper repository).

**No ADCIRC-SWAN, no LCM, and no NN re-training are performed.** Only the
already-saved 700-sample CSVs, the 250-m grid shapefiles, the hazard
rasters, and the saved NN checkpoint are read.

## Order to run

```powershell
$py = "python"   # use the conda env from environment.yml
cd "C:\...\NSF_Debris_Project"

& $py figure_generation\01_compute_uncertainty_metrics.py
& $py figure_generation\02_make_uncertainty_maps.py
& $py figure_generation\03_make_trajectory_plot.py
& $py figure_generation\04_make_amplification_plot.py
& $py figure_generation\05_make_box_violin.py
& $py figure_generation\06_make_nn_sensitivity.py
& $py figure_generation\07_make_elasticity_map.py
```

`01` writes derived per-tract / per-link metric CSVs into
`outputs/debris_impact_output/monte_carlo_result/derived_uncertainty/`.
Steps `02`-`07` consume those derived CSVs (or the original artefacts) and
write PNG building blocks into
`outputs/figure/paper_figures/<artefact>/`. See the README in that
folder for the figure ID-to-file mapping.

## Style conformance

`_style.py` mirrors the colormaps, font, and rendering choices used by
`utils/PlotGenerator.py` and `utils/clr_summary_plots.py`, so the new
figures are visually interchangeable with the existing paper figures.
Maps are saved with hidden axes and no titles; legends are saved as
separate horizontal-colorbar PNGs. The user finalises composition (panel
labels, titles, legend placement) in Inkscape.

## Display vs file year

The on-disk files use 2019 for the baseline scenarios; the paper reports
this as 2020. The mapping `2020 → 2019, 2030 → 2030, 2040 → 2040` is
defined once in `_style.YEAR_DISPLAY_TO_FILE` and used by all scripts.
All output filenames use the **display year** (2020 / 2030 / 2040).

## Environment

Tested with the `2026_debris_prediction` conda env (PyTorch 2.3.1,
GeoPandas 1.1.2, scikit-learn 1.7.2, rasterio, joblib, pyyaml, matplotlib).
