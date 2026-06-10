# Hurricane Debris Projection Framework (2026)

Code companion for the paper:

> Amini, K., Meads, M. M., Padgett, J. E., Highfield, W., Gonzalez-Duenas, C.,
> Pachev, B., and Dawson, C. (2026). "Evolving threats: Understanding the
> coupled impact of sea level and land-cover changes on hurricane-induced
> debris risk." *Natural Hazards Review* (in review).

The framework projects hurricane-induced debris generation and its cascading
impact on transportation-network connectivity under coupled sea-level-rise
(SLR) and land-cover-change (LCC) future scenarios. It couples ADCIRC+SWAN
hazard simulations with a physics-informed (monotonicity-constrained) neural
network for debris volume, a conditional-random-field debris-dispersion model,
and a Monte Carlo road-closure / connectivity-loss analysis for Galveston
County, Texas.

## Data availability

Input datasets, the supplementary document, and example outputs are published
in the NHERI DesignSafe Data Depot:

> **DesignSafe Data Depot DOI/URL: _to be inserted upon publication of the
> data repository_**

After downloading, place the data as follows (paths relative to this
repository root; both locations are `.gitignore`d):

| DesignSafe folder                     | Place at                                            |
|---------------------------------------|-----------------------------------------------------|
| Scenario model-input tables           | `outputs/final_input_for_debris_volume_model/<storm>/<year>/final_input_parameters.csv` |
| Debris-volume training target         | `outputs/debris_volume/Grid_250_with_debris_volume.shp` |
| v7d debris-volume predictions (CSV + shapefile) | `outputs/final_debris_volume_output/<storm>/<year>/V14_physics_v7d_predictions.csv` (figure scripts) and `inputs/predictions_v7d/<storm>/<year>/V14_physics_v7d_predictions.shp` (Monte Carlo) |
| Network / facilities / tracts shapefiles | `inputs/debris_impact_data/...` |
| Land-cover scenario features (LCM)    | `inputs/land_cover_scenarios/MM_Input<year>F.csv`   |
| Monte Carlo result tables (1,000 samples) | `debris_volume_model/outputs/v7d_runs/`          |

Storm/year codes: storms `ike`, `fema33`, `fema36`; file years `2019`
(displayed as the 2020 baseline in the paper), `2030`, `2040`. Scenario
scoring (`score_all_scenarios.py`) additionally expects the Ike training-time
inputs at `outputs/final_input_for_debris_volume_model/ike/2008/`.

## Environment

```bash
conda env create -f environment.yml
conda activate debris_projection
```

Python 3.10; key packages: PyTorch 2.3, GeoPandas 1.1, scikit-learn 1.7,
GSTools 1.7 (random fields), NetworkX 3.4, rasterio 1.4.

## Repository layout

```
config/                  YAML configurations (data prep, Monte Carlo, plotting)
utils/                   Shared library (hazard extraction, geoprocessing,
                         debris dispersion, network analysis, plotting)
debris_volume_model/     Physics-informed debris-volume model (v7d production)
  model.py               3-branch monotonicity-constrained DNN definition
  train.py               Training driver (CLI)
  prepare_features.py    Feature engineering for the training table
  evaluate.py / evaluate_v7plus.py
                         Evaluation: R^2/MAE, bootstrap CIs, permutation and
                         group importance, PDP/ALE diagnostics
  score_all_scenarios.py Apply the trained model to all storm x year scenarios
  generate_scenario_shapefiles.py
                         Export per-scenario prediction shapefiles
  feature_groups_v7d.json  30-feature manifest (18 monotone-increasing,
                         2 monotone-decreasing, 10 unconstrained)
  engineered_input_2026_05_04_no_outlier_v7d.csv
                         Training table (Hurricane Ike 2008; 12,048 cells)
  outputs/
    run_v7d_hazard_interactions/   Production checkpoint (seed 7): model_best.pth,
                         scaler.pkl, manifest.json, evaluation/ diagnostics
    run_v7d_seed{98,42,2024}/      Seed-stability checkpoints (paper Table S4)
    v7d_runs/            Monte Carlo post-processing (aggregation, convergence,
                         tract-FID mapping) + published aggregate tables
    v7d_*.py             Model diagnostics (ALE, collinearity, joint importance,
                         PDP sign audit, decreasing-cells analysis)
figure_generation/       Paper/supplementary figure scripts (run in numeric order)
main.py                  Stage 0: hazard-intensity extraction from ADCIRC+SWAN HDF5
generate_hazard_rasters.py           Stage 0b: gridded hazard rasters
generate_final_input_for_debris_volume_model.py
                         Stage 1: assemble per-scenario model-input tables
run_v7d_monte_carlo_cached.py        Stage 5: dispersion + road-closure Monte Carlo
generate_v7d_paper_figures.py        Stage 6b: map-style paper figures
```

## Pipeline

Stages 0-1 require the raw hazard simulations and are normally not re-run;
their products (the per-scenario `final_input_parameters.csv` tables) are
published on DesignSafe. Stages 2-6 reproduce every number and figure in the
paper from those tables.

**Stage 0 — hazard extraction** (raw ADCIRC+SWAN HDF5 -> intensity measures):
`python main.py` with `config/config_data_processing.yaml` (set
`folder_path` to the raw simulation bundles), then
`python generate_hazard_rasters.py`.

**Stage 1 — model-input assembly**:
`python generate_final_input_for_debris_volume_model.py` (merges hazard,
built-environment, and land-cover scenario features onto the 250 m grid).

**Stage 2 — train the debris-volume model** (production run = seed 7):

```bash
python debris_volume_model/train.py \
  --csv debris_volume_model/engineered_input_2026_05_04_no_outlier_v7d.csv \
  --manifest debris_volume_model/feature_groups_v7d.json \
  --out-dir debris_volume_model/outputs/run_v7d_hazard_interactions \
  --epochs 300 --batch-size 64 --lr 5e-4 --weight-decay 1e-4 \
  --patience 30 --lr-patience 8 --seed 7 --no-wandb
```

**Stage 3 — evaluate**:
`python debris_volume_model/evaluate_v7plus.py --run-dir
debris_volume_model/outputs/run_v7d_hazard_interactions` (bootstrap CIs,
permutation/group importance, PDPs; ALE and collinearity diagnostics via the
`debris_volume_model/outputs/v7d_*.py` scripts).

**Stage 4 — score scenarios and export shapefiles**:

```bash
python debris_volume_model/score_all_scenarios.py \
  --run-dir debris_volume_model/outputs/run_v7d_hazard_interactions \
  --manifest debris_volume_model/feature_groups_v7d.json --calibrate
python debris_volume_model/generate_scenario_shapefiles.py \
  --run-dir debris_volume_model/outputs/run_v7d_hazard_interactions \
  --manifest debris_volume_model/feature_groups_v7d.json \
  --output-prefix V14_physics_v7d
```

**Stage 5 — Monte Carlo impact analysis** (1,000 samples per scenario; the
published run used seeds 0-999):

```bash
python run_v7d_monte_carlo_cached.py --scenario ike --year 2019 \
  --start-seed 0 --n-samples 1000 --n-parallel 20
```

Post-process with `debris_volume_model/outputs/v7d_runs/v7d_aggregate.py`
(county/tract/link summaries) and `v7d_convergence_v2.py` (CV diagnostics).

**Stage 6 — figures**: run `figure_generation/01_...py` through `09_...py`
in order (see `figure_generation/README.md`), plus
`generate_v7d_paper_figures.py` for the map-style figures.

## Trained model summary (v7d production, seed 7)

- 30 input features in 3 branches: 18 monotone-increasing (incl. 3 explicit
  hazard x building interaction terms), 2 monotone-decreasing, 10 free;
  ~2,340 trainable parameters; softplus output (non-negative volume).
- Held-out test set: R^2 ~ 0.66 (calibrated), MAE ~ 102 m^3/cell; bootstrap
  95% CI for R^2 = [0.576, 0.734]; post-hoc calibration factor c = 1.482.
- Monotonicity audit: all 20 constrained features have the prescribed PDP
  sign by construction; seed stability across 4 seeds in Table S4 of the
  paper supplement.

## Notes

- Experiment tracking (Weights & Biases) is optional; pass `--no-wandb` to
  `train.py`. API keys are not stored in this repository (`wandb_key` fields
  in run manifests are nulled).
- `config/config_debris_v7d.yaml` paths are relative to the repository root.
- The Monte Carlo stage writes ~113 MB per scenario of per-link samples;
  these are `.gitignore`d and published on DesignSafe instead.
