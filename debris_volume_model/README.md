# Physics-informed debris-volume model

The debris-volume model of the paper: a partially-monotone neural network
that maps 30 grid-cell features to hurricane-induced debris volume
(m^3 per 250 m cell). See `MODEL_CARD.md` for the model card and the
repository-root `README.md` for the full pipeline.

## Contents

```
model.py                  3-branch monotonicity-constrained DNN definition
train.py                  Training driver (CLI; optional W&B logging)
prepare_features.py       Builds the engineered training table from the
                          Stage-1 inputs (mean_bldg_size_m2, dist_to_coast_m)
evaluate.py               Basic post-training evaluation
evaluate_extended.py      Bootstrap CIs, permutation/group importance, PDPs
score_all_scenarios.py    Score all storm x year scenarios; verifies the
                          county totals reported in Table 2 of the paper
generate_scenario_shapefiles.py
                          Export per-scenario prediction shapefiles for the
                          Monte Carlo stage
feature_groups.json       30-feature manifest (branch assignment + target)
training_data.csv         Hurricane Ike (2008) training table
grid_static_features.csv  Static per-cell lookup (FID, dist_to_coast_m)
trained_model/            Production checkpoint (seed 7) + evaluation
seed_stability/           Seed-stability runs (paper Table S4)
diagnostics/              ALE / collinearity / PDP-sign / decreasing-cells
monte_carlo_results/      MC post-processing scripts + published aggregates
```

## Architecture summary

Inputs are split into three branches:

| Branch | n | Constraint |
|---|---|---|
| Monotone-increasing | 18 | non-negative hidden weights via softplus(W) + ReLU |
| Monotone-decreasing | 2 | sign-flipped variant of the increasing branch |
| Free | 10 | unconstrained MLP (BatchNorm + dropout) |

```
y = softplus( bias + branch_inc(x_inc) + branch_dec(x_dec) + branch_free(x_free) )
```

softplus enforces y >= 0 with no post-hoc clipping; the monotonicity
constraints are guaranteed by construction.

## Reproduce the production model

```bash
python debris_volume_model/train.py \
  --csv debris_volume_model/training_data.csv \
  --manifest debris_volume_model/feature_groups.json \
  --out-dir debris_volume_model/trained_model \
  --epochs 300 --batch-size 64 --lr 5e-4 --weight-decay 1e-4 \
  --patience 30 --lr-patience 8 --seed 7 --no-wandb

python debris_volume_model/evaluate_extended.py \
  --run-dir debris_volume_model/trained_model

python debris_volume_model/score_all_scenarios.py \
  --run-dir debris_volume_model/trained_model \
  --manifest debris_volume_model/feature_groups.json --calibrate
```

Scenario scoring requires the per-scenario input tables from the DesignSafe
data publication (see the root README's data-staging table).
