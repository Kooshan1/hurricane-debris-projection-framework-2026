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

This repository ships the **code, configuration, and trained model** only; the
datasets it consumes are published in the DesignSafe Data Depot and are **not
duplicated here** (every target path below is `.gitignore`d). After downloading,
place the data as follows (paths relative to this repository root):

| DesignSafe folder | Place at |
|---|---|
| `Model_Training_Data/` | `debris_volume_model/training_data.csv`, `debris_volume_model/grid_static_features.csv`, and `outputs/debris_volume/Grid_250_with_debris_volume.shp` |
| `Scenario_Model_Inputs/` | `outputs/final_input_for_debris_volume_model/<storm>/<year>/final_input_parameters.csv` |
| `Debris_Volume_Predictions/` | `outputs/final_debris_volume_output/<storm>/<year>/debris_volume_predictions.csv` (figure scripts) and `inputs/debris_volume_predictions/<storm>/<year>/debris_volume_predictions.shp` (Monte Carlo) |
| `Network_and_Facilities/` | `inputs/debris_impact_data/...` |
| `Land_Cover_Scenarios/` | `inputs/land_cover_scenarios/MM_Input<year>F.csv` |
| `Monte_Carlo_Results/` (samples + aggregates) | `debris_volume_model/monte_carlo_results/` |

(The model architecture manifest `debris_volume_model/feature_groups.json` ships
with the code; a copy is also included in `Model_Training_Data/` for reference.)

Storm codes: `ike` (Hurricane Ike), `fema33` (moderate synthetic storm),
`fema36` (high-intensity synthetic storm). Scenario years: `2020` (baseline),
`2030`, `2040` (coupled SLR + LCC projections).

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
debris_volume_model/     Physics-informed debris-volume model
  model.py               3-branch monotonicity-constrained DNN definition
  train.py               Training driver (CLI)
  prepare_features.py    Feature engineering for the training table
  evaluate.py / evaluate_extended.py
                         Evaluation: R^2/MAE, bootstrap CIs, permutation and
                         group importance, PDP/ALE diagnostics
  score_all_scenarios.py Apply the trained model to all storm x year scenarios
  generate_scenario_shapefiles.py
                         Export per-scenario prediction shapefiles
  feature_groups.json    30-feature manifest (18 monotone-increasing,
                         2 monotone-decreasing, 10 unconstrained)
  training_data.csv      Training table (Hurricane Ike 2008; 17,961 grid
                         cells, of which 12,048 with complete features)
  grid_static_features.csv
                         Static per-cell lookup (FID, distance to coastline)
  MODEL_CARD.md          Model card for the trained production model
  trained_model/         Production checkpoint (seed 7): model_best.pth,
                         scaler.pkl, manifest.json, evaluation/ diagnostics
  seed_stability/        Seed-stability checkpoints (paper Table S4)
  diagnostics/           Model diagnostics: ALE, developed-land-cluster
                         collinearity and joint importance, PDP sign audit,
                         decreasing-cells analysis
  monte_carlo_results/   Monte Carlo post-processing (aggregation,
                         convergence, tract-FID mapping) + published
                         aggregate tables; bulk sample CSVs staged here
figure_generation/       Paper/supplementary figure scripts (run in numeric order)
main.py                  Stage 0: hazard-intensity extraction from ADCIRC+SWAN HDF5
generate_hazard_rasters.py           Stage 0b: gridded hazard rasters
generate_final_input_for_debris_volume_model.py
                         Stage 1: assemble per-scenario model-input tables
run_monte_carlo.py       Stage 5: dispersion + road-closure Monte Carlo
generate_paper_map_figures.py        Stage 6b: map-style paper figures
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
  --csv debris_volume_model/training_data.csv \
  --manifest debris_volume_model/feature_groups.json \
  --out-dir debris_volume_model/trained_model \
  --epochs 300 --batch-size 64 --lr 5e-4 --weight-decay 1e-4 \
  --patience 30 --lr-patience 8 --seed 7 --no-wandb
```

**Stage 3 — evaluate**:
`python debris_volume_model/evaluate_extended.py --run-dir
debris_volume_model/trained_model` (bootstrap CIs, permutation/group
importance, PDPs); ALE and collinearity diagnostics via the scripts in
`debris_volume_model/diagnostics/`.

**Stage 4 — score scenarios and export shapefiles**:

```bash
python debris_volume_model/score_all_scenarios.py \
  --run-dir debris_volume_model/trained_model \
  --manifest debris_volume_model/feature_groups.json --calibrate
python debris_volume_model/generate_scenario_shapefiles.py \
  --run-dir debris_volume_model/trained_model \
  --manifest debris_volume_model/feature_groups.json \
  --output-prefix debris_volume
```

**Stage 5 — Monte Carlo impact analysis** (1,000 samples per scenario; the
published run used seeds 0-999):

```bash
python run_monte_carlo.py --scenario ike --year 2020 \
  --start-seed 0 --n-samples 1000 --n-parallel 20
```

Post-process with
`debris_volume_model/monte_carlo_results/fix_summary_fids.py` (tract-FID
mapping), `aggregate_results.py` (county/tract/link summaries), and
`convergence_diagnostics.py` (CV diagnostics).

**Stage 6 — figures**: run `figure_generation/01_...py` through `09_...py`
in order (see `figure_generation/README.md`), plus
`generate_paper_map_figures.py` for the map-style figures.

## Trained model summary (production run, seed 7)

- 30 input features in 3 branches: 18 monotone-increasing (incl. 3 explicit
  hazard x building interaction terms), 2 monotone-decreasing, 10 free;
  2,340 trainable parameters; softplus output (non-negative volume).
- Held-out test set: R^2 ~ 0.66 (calibrated), MAE ~ 102 m^3/cell; bootstrap
  95% CI for R^2 = [0.576, 0.734]; post-hoc calibration factor c = 1.482.
- Monotonicity audit: all 20 constrained features have the prescribed PDP
  sign by construction; seed stability across 4 seeds in Table S4 of the
  paper supplement.

See `debris_volume_model/MODEL_CARD.md` for the full model card.

## Notes

- Experiment tracking (Weights & Biases) is optional; pass `--no-wandb` to
  `train.py`. No API keys are stored in this repository.
- `config/config_monte_carlo.yaml` paths are relative to the repository root.
- The Monte Carlo stage writes ~113 MB per scenario of per-link samples;
  these are `.gitignore`d and published on DesignSafe instead.
