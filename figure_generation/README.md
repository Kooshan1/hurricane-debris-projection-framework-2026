# Figure-generation scripts

These scripts post-process the published Monte Carlo results and
debris-volume predictions into the manuscript and supplementary figures.
No hazard simulation, land-cover modeling, or model training is performed —
only the published result tables, the 250 m grid shapefiles, and the saved
model checkpoint are read (see the root README's data-staging table).

## Order to run

```bash
python figure_generation/01_compute_uncertainty_metrics.py
python figure_generation/02_make_uncertainty_maps.py
python figure_generation/03_make_trajectory_plot.py
python figure_generation/04_make_amplification_plot.py
python figure_generation/05_make_box_violin.py
python figure_generation/07_make_elasticity_map.py
python figure_generation/08_make_feature_importance.py
python figure_generation/09_make_dev_cluster_diagnostics.py
```

`01` derives per-tract / per-link metric CSVs from the Monte Carlo sample
tables; the later steps consume those derived CSVs (or the published
artefacts directly) and write PNG building blocks into
`outputs/figure/paper_figures/<artefact>/`. `make_clr_vector_panels.py`
builds the merged vector/raster map panels of the supplement, and the
repository-root `generate_paper_map_figures.py` regenerates the map-style
figures (debris volume, road closure, CLR, surge).

## Style conformance

`_style.py` centralizes the colormaps, fonts (STIX Two Text), DPI, scenario
codes (storms ike / fema33 / fema36; years 2020 / 2030 / 2040), and shared
paths, so every figure is visually consistent. Maps are saved with hidden
axes and no titles; legends are saved as separate horizontal-colorbar PNGs.
Final panel composition (labels, titles, legend placement) is done in a
vector-graphics editor.

## Environment

Use the pinned conda environment from the repository root
(`environment.yml`).
