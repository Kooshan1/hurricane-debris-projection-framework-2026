"""
Joint dev-cluster ALE for the production model -- DT, DM, DO moving TOGETHER along the
empirical correlation manifold.

Why this exists
---------------
Single-feature PDP / ALE answer "what if we move feature j alone?". But DT,
DM, DO are LCC subcategories that always move together in real inputs
(every LCC projection update perturbs all three). The decision-relevant
question is therefore: "When the dev cluster moves from its low-development
regime to its high-development regime, what does the model predict?".

Implementation
--------------
1. Rank observations by the SUM (DT+DM+DO) -- the empirical "cluster axis".
2. Bin the cluster axis on quantiles.
3. For each bin k, compute the mean prediction f(x) over the observations
   actually falling in that bin (no off-manifold inputs anywhere).
4. Report the slope from cluster-p5 -> cluster-p95.

This is on-manifold by construction: every (DT, DM, DO) triple evaluated is
a real observed triple. The result is the "true predictive effect" of the
dev cluster on debris, as seen by the production model in the data the model was trained on.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs

CSV = THIS_DIR / "training_data.csv"
MAN = THIS_DIR / "feature_groups.json"
RUN_DIR = THIS_DIR / "trained_model"
EVAL_DIR = RUN_DIR / "evaluation"
SEED = 7
N_BINS = 20

with open(MAN) as f:
    m = json.load(f)
feat_order = m["monotone_inc"] + m["monotone_dec"] + m["free"]
n_inc, n_dec = len(m["monotone_inc"]), len(m["monotone_dec"])
n_free = len(feat_order) - n_inc - n_dec
name_to_idx = {n: i for i, n in enumerate(feat_order)}

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
print(f"Loaded production model (epoch {ckpt['epoch']}), c = {c_factor:.4f}")


@torch.no_grad()
def predict(X_scaled, batch=4096):
    out = []
    for i in range(0, len(X_scaled), batch):
        xb = torch.tensor(X_scaled[i:i + batch], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        out.append(model(x_inc, x_dec, x_free).cpu().numpy().flatten())
    return np.concatenate(out)


# Cluster axis = sum of DT + DM + DO (original scale)
idx_DT = name_to_idx["DT"]
idx_DM = name_to_idx["DM"]
idx_DO = name_to_idx["DO"]
cluster_axis = X_te[:, idx_DT] + X_te[:, idx_DM] + X_te[:, idx_DO]
preds_te = predict(X_te_s) * c_factor

# Bin observations by cluster axis on quantiles
edges = np.quantile(cluster_axis, np.linspace(0, 1, N_BINS + 1))
edges = np.unique(edges)
K = len(edges) - 1
rows = []
for k in range(K):
    lo, hi = edges[k], edges[k + 1]
    if k == 0:
        mask = (cluster_axis >= lo) & (cluster_axis <= hi)
    else:
        mask = (cluster_axis > lo) & (cluster_axis <= hi)
    n = int(mask.sum())
    if n == 0:
        continue
    rows.append({
        "bin": k,
        "cluster_axis_lo": float(lo),
        "cluster_axis_hi": float(hi),
        "cluster_axis_mid": float(0.5 * (lo + hi)),
        "n_obs": n,
        "mean_pred_m3": float(preds_te[mask].mean()),
        "mean_DT": float(X_te[mask, idx_DT].mean()),
        "mean_DM": float(X_te[mask, idx_DM].mean()),
        "mean_DO": float(X_te[mask, idx_DO].mean()),
    })
clu = pd.DataFrame(rows)
out_csv = EVAL_DIR / "dev_cluster_joint_effect.csv"
clu.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")
print(clu[["cluster_axis_mid", "n_obs", "mean_pred_m3", "mean_DT", "mean_DM", "mean_DO"]].to_string(index=False))

# Headline number: cluster-p5 -> cluster-p95 slope of the mean prediction
p5_axis, p95_axis = np.percentile(cluster_axis, [5, 95])
pred_at_p5 = float(np.interp(p5_axis, clu["cluster_axis_mid"], clu["mean_pred_m3"]))
pred_at_p95 = float(np.interp(p95_axis, clu["cluster_axis_mid"], clu["mean_pred_m3"]))
joint_slope = pred_at_p95 - pred_at_p5
print(f"\nCluster axis (DT+DM+DO) p5..p95 = {p5_axis:.3f} .. {p95_axis:.3f}")
print(f"Mean pred at cluster p5  = {pred_at_p5:.2f} m^3")
print(f"Mean pred at cluster p95 = {pred_at_p95:.2f} m^3")
print(f"Joint dev-cluster slope  = {joint_slope:+.2f} m^3")

# Plot
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(clu["cluster_axis_mid"], clu["mean_pred_m3"],
        marker="o", color="#1b7837", linewidth=1.4)
ax.set_xlabel("Dev-cluster axis (DT + DM + DO)")
ax.set_ylabel(r"Mean predicted debris (m$^3$/cell)")
ax.set_title(f"Joint dev-cluster effect (on-manifold)\n"
             f"p5..p95 of cluster -> slope = {joint_slope:+.0f} m$^3$/cell",
             fontsize=10)
ax.grid(linestyle=":", alpha=0.35)
ax.axhline(pred_at_p5,  color="#888", linewidth=0.4, linestyle=":")
ax.axhline(pred_at_p95, color="#888", linewidth=0.4, linestyle=":")
ax.axvline(p5_axis,  color="#888", linewidth=0.4, linestyle="--")
ax.axvline(p95_axis, color="#888", linewidth=0.4, linestyle="--")
ax.set_facecolor("none")
fig.patch.set_alpha(0)
fig.tight_layout()
out_png = EVAL_DIR / "dev_cluster_joint_effect.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True, facecolor="none")
plt.close(fig)
print(f"Wrote {out_png}")
