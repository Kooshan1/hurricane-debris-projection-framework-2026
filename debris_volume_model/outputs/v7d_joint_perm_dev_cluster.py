"""
Joint permutation importance for the v7d dev cluster (DT, DM, DO).

Why this exists
---------------
Single-feature permutation importance is misleading under perfect rank
deficiency. In v7d, DT = DM + DO exactly (R^2 = 1.000), so when we
permute DT alone the model can still recover the dev-cluster signal
from DM and DO -- the single-feature importance therefore *under*-states
the cluster's true predictive contribution. The statistically valid
remedy under collinearity is to permute the entire correlated block
together (Strobl et al. 2008, Hooker & Mentch 2021).

This script:
  1. Loads the v7d production model + test split.
  2. Computes baseline MAE on the calibrated test predictions.
  3. Permutes (DT, DM, DO) JOINTLY: each row's three values are reassigned
     to the row's three values from a randomly drawn donor row. The
     within-cluster correlation structure is preserved; only the
     cluster's link to the OTHER features is broken.
  4. Repeats 10 times, reports mean / std of the delta-MAE.

Writes evaluation/joint_perm_dev_cluster.csv with a single row for the
combined "Developed land" feature group, to be picked up by the M-F3
figure generator.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs

CSV = THIS_DIR / "engineered_input_2026_05_04_no_outlier_v7d.csv"
MAN = THIS_DIR / "feature_groups_v7d.json"
RUN_DIR = THIS_DIR / "outputs" / "run_v7d_hazard_interactions"
EVAL_DIR = RUN_DIR / "evaluation"
SEED = 7
N_REPEATS = 10
CLUSTER = ["DT", "DM", "DO"]

with open(MAN) as f:
    m = json.load(f)
feat_order = m["monotone_inc"] + m["monotone_dec"] + m["free"]
n_inc, n_dec = len(m["monotone_inc"]), len(m["monotone_dec"])
n_free = len(feat_order) - n_inc - n_dec
name_to_idx = {n: i for i, n in enumerate(feat_order)}
cluster_idx = [name_to_idx[c] for c in CLUSTER]

df = pd.read_csv(CSV).dropna(subset=feat_order + ["volume_m3"]).reset_index(drop=True)
X = df[feat_order].to_numpy(dtype=np.float32)
y = df["volume_m3"].to_numpy(dtype=np.float32).reshape(-1, 1)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

scaler = joblib.load(RUN_DIR / "scaler.pkl")
X_te_s = scaler.transform(X_te).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhysicsInformedDebrisNN(n_inc=n_inc, n_dec=n_dec, n_free=n_free).to(device)
ckpt = torch.load(RUN_DIR / "model_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open(EVAL_DIR / "evaluation_summary.json") as f:
    c_factor = float(json.load(f)["calibration_factor"])
print(f"Loaded v7d (epoch {ckpt['epoch']}), c = {c_factor:.4f}")


@torch.no_grad()
def predict(X_scaled, batch=4096):
    out = []
    for i in range(0, len(X_scaled), batch):
        xb = torch.tensor(X_scaled[i:i + batch], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        out.append(model(x_inc, x_dec, x_free).cpu().numpy().flatten())
    return np.concatenate(out)


# Baseline MAE
base_pred = predict(X_te_s) * c_factor
base_mae = float(mean_absolute_error(y_te.flatten(), base_pred))
print(f"Baseline test MAE = {base_mae:.4f}")

# Joint permutation: shuffle (DT, DM, DO) as a 3-tuple block.
# i.e. for each row pick a random donor row and copy its DT/DM/DO triple in.
rng = np.random.default_rng(42)
deltas = []
n_te = len(X_te_s)
for r in range(N_REPEATS):
    perm_order = rng.permutation(n_te)
    X_perm = X_te_s.copy()
    X_perm[:, cluster_idx] = X_te_s[perm_order][:, cluster_idx]
    pred = predict(X_perm) * c_factor
    deltas.append(float(mean_absolute_error(y_te.flatten(), pred)) - base_mae)
    print(f"  rep {r+1}/{N_REPEATS}: delta-MAE = {deltas[-1]:+.4f}")

imp_mean = float(np.mean(deltas))
imp_std = float(np.std(deltas))
print(f"\nJoint perm importance for {CLUSTER}: {imp_mean:.4f} +/- {imp_std:.4f} m^3/cell")

# For reference, sum of single-feature importances from existing CSV
perm = pd.read_csv(EVAL_DIR / "permutation_importance.csv")
single_sum = float(perm[perm["feature"].isin(CLUSTER)]["importance_mean"].sum())
print(f"For comparison, sum of single-feature perm importances "
      f"({CLUSTER}): {single_sum:.4f} m^3/cell")
print("(Joint > sum-of-singletons under collinearity is the expected pattern: "
      "single-feature perm is masked by the redundant partners.)")

# Save result
out_csv = EVAL_DIR / "joint_perm_dev_cluster.csv"
pd.DataFrame([{
    "feature": "DT|DM|DO",
    "members": "+".join(CLUSTER),
    "n_repeats": N_REPEATS,
    "importance_mean": imp_mean,
    "importance_std": imp_std,
    "sum_singleton_importance": single_sum,
    "joint_minus_singleton": imp_mean - single_sum,
    "baseline_mae": base_mae,
}]).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
