# Model Card — Physics-Informed Debris-Volume Model

Production model of the paper "Evolving Threats: Understanding the Coupled
Impact of Sea Level and Land-Cover Changes on Hurricane-Induced Debris Risk"
(*Natural Hazards Review*, in review).

## Task

Regress hurricane-induced debris volume (m^3) per 250 m grid cell from 30
hazard, built-environment, natural-environment, and socio-economic features.
Study area: the contiguous mainland and Galveston Island portions of
Galveston County, Texas.

## Architecture

Three parallel MLP branches whose scalar outputs are summed with a global
bias and passed through a softplus output activation (predictions are
non-negative by construction):

| Branch | Features | Constraint |
|---|---|---|
| Monotone-increasing | 18 (incl. the SD_NumB, WH_NumB, MF_NumB hazard x building interactions) | hidden weights reparameterized with softplus(W) + ReLU, so the output is non-decreasing in every branch input |
| Monotone-decreasing | 2 (minimum elevation, distance to coastline) | sign-flipped variant of the increasing branch |
| Unconstrained (free) | 10 | standard MLP with BatchNorm + dropout |

2,340 trainable parameters. The monotonicity constraints hold by
construction, independent of the training outcome (Sill 1998; You et al.
2017; Wehenkel & Louppe 2019). The full feature list with branch assignment
is in `feature_groups.json`; definitions, units, and training-distribution
statistics are in Table S2 of the paper supplement and in the
variable-definitions table of the DesignSafe data publication.

## Training data

`training_data.csv`: the Hurricane Ike (2008) event for Galveston County on
the 250 m grid — 17,961 cells (one spatial outlier cell removed), of which
12,048 have complete features and contribute to the train/validation/test
split (60/20/20, random). Target: observed debris volume per cell
(`volume_m3`); the corresponding shapefile is published with the data
repository. Two engineered spatial features replace region-locked
coordinates: `mean_bldg_size_m2` (= TBFA / NumB, breaking their
collinearity) and `dist_to_coast_m` (translation-invariant distance to the
nearest open-water cell).

## Training configuration (production run)

- Optimizer AdamW, lr 5e-4, weight decay 1e-4; Huber loss; batch size 64;
  up to 300 epochs with early stopping (patience 30) and LR-on-plateau
  (patience 8); seed 7. See `trained_model/manifest.json` for the exact
  arguments and feature ordering, and `train.py --help` for the CLI.
- Post-hoc multiplicative calibration c = mean(y_train) / mean(pred_train)
  = 1.482 applied to all reported predictions (preserves the spatial
  pattern; matches the predicted county total to the observed total).

## Performance (held-out test set)

- R^2 ~ 0.66 calibrated, MAE ~ 102 m^3/cell calibrated, RMSE ~ 339 m^3/cell;
  bootstrap 95% CI for R^2 = [0.576, 0.734] (`trained_model/evaluation/`).
- Seed stability: 4 random seeds (7, 98, 42, 2024) in `seed_stability/`;
  calibrated R^2 = 0.598 +/- 0.060 across seeds (paper Table S4), with the
  projected year-over-year debris increases preserved by every seed.

## Diagnostics

- `trained_model/evaluation/`: permutation and group importance, per-bin
  MAE, PDPs of the top features, ALE for the developed-land cluster,
  bootstrap confidence intervals, test-set scatter plots.
- `diagnostics/`: PDP sign audit for all 30 features (every constrained
  feature carries the prescribed sign), developed-land cluster collinearity
  (DT = DM + DO exactly, so importance is evaluated jointly), joint
  permutation importance, on-manifold joint-effect analysis, and the
  decreasing-cells analysis (cells with projected debris decreases under
  the future scenarios).

## Intended use and limitations

Intended for projecting county-scale debris generation under coupled
SLR + LCC scenarios in the Galveston County study area, and as a template
for re-instantiation in other coastal regions (with local training data and
recalibration). The model is trained on a single historical event
(Hurricane Ike); transfer to other regions or storm climatologies requires
retraining, as discussed in the paper's "Scope of Generalizability" and
"Limitations and Future Work" sections.
