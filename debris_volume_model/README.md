# Revision test — physics-informed debris-volume model

## Why this exists

The trained NN documented in the manuscript has three weaknesses that
became apparent during the revision-figure interpretability analysis
(see `prompts/responses/weakness_assessment_2026_05_04_*.md`):

1. **`centroid_x` is the strongest single feature.** Predicted debris
   triples (~95 → ~370 m³/cell) just by sliding longitude across
   Galveston, before any physical input changes. The model is
   memorising Hurricane-Ike-specific geography.
2. **`TBFA` and `NumB` have opposite-sign 1-D PDPs** (TBFA slopes
   downward, NumB slopes upward), even though they correlate at
   r = 0.74 in the training data. A reviewer can rightly call this
   "non-physical."
3. The architecture (32 → 512 → 256 → 128 → 64 → 1, ~190 k parameters)
   is **far too large** for the ~7 200 training cells. There is no
   built-in physical prior; non-negativity of the output is not
   guaranteed.

This directory contains a **test model** that addresses all three
weaknesses while remaining defensible in a journal paper. It is **not**
intended to replace the manuscript's main model — it is an additional
test of an alternative formulation that we may cite or migrate to in a
follow-on study.

> No file in the original `outputs/` tree is modified. New artefacts
> live under `debris_volume_model/` and on the remote server at
> `~/nsf_debris_project/`.

## Design

### Feature engineering (32 → 31 features)

| Action | Variable | Reason |
|---|---|---|
| **Drop** | `centroid_x`, `centroid_y` | region-specific position memorisation (4× swing in PDP) |
| **Replace** | `TBFA` → `mean_bldg_size_m2 = TBFA / max(NumB, 1)` | breaks the r = 0.74 collinearity; physically meaningful |
| **Add** | `dist_to_coast_m` | translation-invariant spatial feature; replaces what `centroid_x` was implicitly proxying |

`dist_to_coast_m` is computed locally (in `prepare_features.py`) by
re-projecting the 250-m grid to UTM zone 15 N (meters), declaring any
cell whose `OW` (open-water fraction) exceeds 0.3 to be a "water cell,"
and then taking the Euclidean distance from each cell's centroid to the
nearest water cell using `scipy.spatial.cKDTree`. The threshold 0.3 is
reported in the manifest JSON so the supplement can document it.

After engineering: r(`mean_bldg_size_m2`, `NumB`) = 0.034 (from 0.742),
and `dist_to_coast_m` spans 0 – 7 558 m with median 750 m.

### Architecture (`model.py`) — partially-monotone NN

Inputs are split into three blocks:

| Branch | Inputs | Constraint |
|---|---|---|
| Monotone-INCREASING (`n_inc = 11`) | `SD, WH, WS, MF, WPF_1, WPF_2, NumB, mean_bldg_size_m2, NAS, TAAS, NumMH` | non-negative weights via `softplus(W)` + ReLU activations → output non-decreasing in each input |
| Monotone-DECREASING (`n_dec = 2`) | `ME, dist_to_coast_m` | wrapper `f(x) = -inc(x)` of the monotone-inc block → output non-increasing in each input |
| Free (`n_free = 18`) | directional / categorical / socioeconomic features | unconstrained 2-hidden-layer MLP |

Final output:

```
y = softplus( bias + branch_inc(x_inc) + branch_dec(x_dec) + branch_free(x_free) )
```

`softplus` enforces `y ≥ 0` regardless of the linear combination —
no post-hoc clipping is needed.

Two parameter budgets were tested:

| Variant | `h_inc` | `h_dec` | `h_free` | total params |
|---|---|---|---|---|
| `physics_v1` | (16, 8) | (8, 4) | (16, 8) | 900 |
| `physics_v2_larger` | (32, 16) | (16, 8) | (32, 16) | 2372 |

Original NN (paper): ~190 k parameters (~80–210× larger than these).
Training-set size after `dropna`: 7 229 cells. The variants above sit
at **8:1** and **3:1** sample-to-parameter ratios, both more
appropriate for tabular data of this scale than the original 0.04:1.

### Justifications for the journal supplement

1. **Constrained NN as a class.** Monotonic neural networks via
   non-negative weights are a standard, well-cited construction
   (Wehenkel & Louppe, 2019; You et al., 2017 *Deep Lattice Networks*;
   Sill, 1998). The constraints are *guaranteed by construction*, not
   merely encouraged through regularisation.
2. **Physical priors selected.** Monotonicity is asserted only where
   physics demands it: hazard-intensity → debris (non-decreasing),
   built-environment density → debris (non-decreasing), elevation →
   debris (non-increasing), distance-to-coast → debris (non-increasing).
   Directional, socioeconomic, and development-class variables are
   left in the unconstrained "free" branch — a defensible compromise
   between physical defensibility and data-driven flexibility.
3. **Non-negativity by construction.** The original NN can produce
   negative debris predictions which are clipped to zero in
   post-processing; here `softplus` removes the need for any clipping.
4. **Decoupled features.** Replacing `TBFA` with `mean_bldg_size_m2`
   preserves the same physical information (mass at risk) while
   eliminating the multicollinearity that produced opposite-sign PDPs
   in the original.
5. **Translation-invariant spatial proxy.** `dist_to_coast_m` is
   computed from a coast definition (cells with high open-water
   fraction) and is meaningful for any coastal county. Unlike
   `centroid_x`, it does not require retraining when the model is
   applied to a new region.

## Files

```
debris_volume_model/
├── prepare_features.py             local: builds engineered_input_<DATE>.csv
├── model.py                        partially-monotone NN definition
├── train.py                        training loop + W&B logging
├── evaluate.py                     post-training perm. importance + PDPs
├── engineered_input_2026_05_04.csv [generated]
├── feature_groups_2026_05_04.json  [generated]
├── outputs/
│   ├── run_v1/                     [pulled from remote]
│   └── run_v2/                     [pulled from remote]
└── README.md                       this file
```

## How to reproduce

### Local (one-time)

```powershell
$py = "python"   # use the conda env from environment.yml
cd "C:\...\NSF_Debris_Project"
& $py debris_volume_model\prepare_features.py
```

### Remote (`ai4resilience`, `debris_volume_project` conda env)

```bash
# transfer (already done by the assistant)
scp debris_volume_model/{model.py,train.py,evaluate.py} ai4resilience:~/nsf_debris_project/scripts/
scp debris_volume_model/{engineered_input_*.csv,feature_groups_*.json} ai4resilience:~/nsf_debris_project/data/

# launch
ssh ai4resilience "cd ~/nsf_debris_project && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate debris_volume_project && \
  python scripts/train.py \
    --csv data/engineered_input_2026_05_04.csv \
    --manifest data/feature_groups_2026_05_04.json \
    --out-dir outputs/run_v2 \
    --epochs 300 --batch-size 64 \
    --wandb-project NSF_Debris_Project_Revision_Test \
    --run-name physics_v2_larger"

# evaluate
ssh ai4resilience "cd ~/nsf_debris_project && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate debris_volume_project && \
  python scripts/evaluate.py --run-dir outputs/run_v2 \
    --csv data/engineered_input_2026_05_04.csv \
    --manifest data/feature_groups_2026_05_04.json"
```

### W&B project

`https://wandb.ai/kooshan1/NSF_Debris_Project_Revision_Test`

Logged metrics include train / val / test MAE, RMSE, R²; epoch time;
running average loss & gradient norm; mean output of each branch
(`branch/h_inc_mean`, `branch/h_dec_mean`, `branch/h_free_mean`); and
a final test-set predicted-vs-actual scatter image.

## Results (compared to the original NN)

See `outputs/run_v*/evaluation_summary.json` and
`outputs/run_v*/final_metrics.json`. Headline comparison filled in
once both runs complete.
