"""
Accumulated Local Effects (ALE) -- Apley & Zhu (2020), JRSS-B -- for the production model.

Motivation
----------
PDP for DT in the production model is -59 m^3 (p5 -> p95), contradicting the physical prior
that more development -> more debris. The 1-D PDP holds DM, DO, RD fixed
while varying DT, but in real LCC inputs DT, DM, DO are highly correlated
and never appear independently. PDP therefore probes off-manifold inputs
the model was never trained on.

ALE fixes this by:
  1. Binning the feature on its empirical quantiles.
  2. For each bin, computing local differences using ONLY observations
     whose other features are consistent with that bin's neighbourhood.
  3. Accumulating the bin-averaged local effects.

This is the canonical correction for the PDP-collinearity artefact;
Apley & Zhu introduced it specifically because Friedman PDPs mislead
under correlated features.

Output: ALE curves for DT, DM, DO in trained_model/evaluation/.
Also writes a CSV summarising the p5->p95 ALE slope alongside the
matching PDP slope from pdp_signs.csv for direct comparison.
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
TARGETS = ["DT", "DM", "DO"]

# Existing PDP slopes for the same features (from pdp_signs.csv) so
# the comparison table is one-shot rather than requiring a join.
PDP_SLOPES = {"DT": -59.353, "DM": 50.678, "DO": 131.140}


# ---------------------------------------------------------------------------
# Load model, scaler, manifest, test split (matches evaluate_extended.py)
# ---------------------------------------------------------------------------
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

# Calibration factor -- same convention as evaluate_extended.py
with open(EVAL_DIR / "evaluation_summary.json") as f:
    c_factor = float(json.load(f)["calibration_factor"])
print(f"Loaded production model (best epoch {ckpt['epoch']}), calibration c = {c_factor:.4f}")


@torch.no_grad()
def predict(X_scaled: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, len(X_scaled), batch):
        xb = torch.tensor(X_scaled[i:i + batch], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        out.append(model(x_inc, x_dec, x_free).cpu().numpy().flatten())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# 1-D ALE (Apley & Zhu 2020)
# ---------------------------------------------------------------------------
def ale_1d(X_scaled: np.ndarray, feat_idx: int, n_bins: int = 20,
           calibration: float = 1.0):
    """Return (bin_midpoints_orig_scale, ale_centred_m3, bin_counts).

    Algorithm:
      1. Bin feature j on quantiles (0, 1/K, 2/K, ..., 1).
      2. For each bin k:
           N_k = observations with x_j in (z_{k-1}, z_k]
           local_k = mean over N_k of [ f(z_k, x_{-j}) - f(z_{k-1}, x_{-j}) ]
      3. ALE_k = sum_{l<=k} local_l (accumulate).
      4. Centre so weighted mean (by bin count) is zero.
    """
    feat_scaled = X_scaled[:, feat_idx]
    edges = np.unique(np.quantile(feat_scaled, np.linspace(0, 1, n_bins + 1)))
    K = len(edges) - 1
    local = np.zeros(K, dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        z_lo, z_hi = edges[k], edges[k + 1]
        if k == 0:
            mask = (feat_scaled >= z_lo) & (feat_scaled <= z_hi)
        else:
            mask = (feat_scaled > z_lo) & (feat_scaled <= z_hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        counts[k] = n_k
        X_sub = X_scaled[mask].copy()
        X_hi = X_sub.copy(); X_hi[:, feat_idx] = z_hi
        X_lo = X_sub.copy(); X_lo[:, feat_idx] = z_lo
        diff = (predict(X_hi) - predict(X_lo)) * calibration
        local[k] = diff.mean()

    ale = np.cumsum(local)
    # Centre: weighted mean over observations (each obs lives in one bin)
    weights = counts / counts.sum()
    ale_centred = ale - np.sum(weights * ale)

    # Bin midpoints in original feature units
    mid_scaled = 0.5 * (edges[:-1] + edges[1:])
    feat_mean = scaler.mean_[feat_idx]
    feat_std = scaler.scale_[feat_idx]
    mid_orig = mid_scaled * feat_std + feat_mean
    return mid_orig, ale_centred, counts, edges


# ---------------------------------------------------------------------------
# Run ALE for DT, DM, DO + a joint cluster check
# ---------------------------------------------------------------------------
EVAL_DIR.mkdir(parents=True, exist_ok=True)

summary_rows = []
ale_curves = {}
for name in TARGETS:
    idx = name_to_idx[name]
    mid, ale, counts, edges = ale_1d(X_te_s, idx, n_bins=N_BINS, calibration=c_factor)
    ale_curves[name] = (mid, ale, counts)
    # p5 -> p95 slope in original units, on the same observations as PDP
    feat_orig = X_te[:, idx]
    p5 = float(np.percentile(feat_orig, 5))
    p95 = float(np.percentile(feat_orig, 95))
    # Linear interpolation of the centred ALE at these two anchor points
    ale_p5 = float(np.interp(p5, mid, ale))
    ale_p95 = float(np.interp(p95, mid, ale))
    ale_slope = ale_p95 - ale_p5
    summary_rows.append({
        "feature": name,
        "p5": round(p5, 4),
        "p95": round(p95, 4),
        "pdp_slope_m3": PDP_SLOPES[name],
        "ale_p5_m3": round(ale_p5, 3),
        "ale_p95_m3": round(ale_p95, 3),
        "ale_slope_m3": round(ale_slope, 3),
        "sign_change_vs_pdp": ("YES" if np.sign(ale_slope) != np.sign(PDP_SLOPES[name])
                                else "no"),
        "n_bins": N_BINS,
        "n_test_rows": int(len(X_te)),
    })

summary_df = pd.DataFrame(summary_rows)
out_csv = EVAL_DIR / "ale_dev_cluster.csv"
summary_df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Plot: three subplots, one per feature, ALE vs original feature value
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, name in zip(axes, TARGETS):
    mid, ale, _ = ale_curves[name]
    ax.plot(mid, ale, color="#1b7837", linewidth=1.6, label="ALE (Apley-Zhu 2020)")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    # Mark p5 and p95 with vertical lines + the ALE slope as annotation
    feat_orig = X_te[:, name_to_idx[name]]
    p5, p95 = np.percentile(feat_orig, [5, 95])
    ax.axvline(p5, color="#888", linewidth=0.4, linestyle="--")
    ax.axvline(p95, color="#888", linewidth=0.4, linestyle="--")
    slope = next(r for r in summary_rows if r["feature"] == name)["ale_slope_m3"]
    pdp = PDP_SLOPES[name]
    sign_match = "matches +" if slope > 0 else "still -"
    ax.set_title(f"{name}: ALE slope = {slope:+.1f} m$^3$  ({sign_match})\n"
                 f"PDP slope (off-manifold) = {pdp:+.1f} m$^3$",
                 fontsize=9)
    ax.set_xlabel(name)
    ax.set_ylabel(r"Centred ALE (m$^3$ / cell)")
    ax.grid(linestyle=":", alpha=0.35)
    ax.set_facecolor("none")

fig.suptitle("Production-model ALE for the dev cluster (free branch) -- on-manifold uncentred-PDP correction",
             fontsize=10)
fig.patch.set_alpha(0)
fig.tight_layout()
out_png = EVAL_DIR / "ale_dev_cluster.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight",
            transparent=True, facecolor="none")
plt.close(fig)
print(f"Wrote {out_png}")


# ---------------------------------------------------------------------------
# Save raw ALE curves as CSV for transparency / paper supplement
# ---------------------------------------------------------------------------
for name, (mid, ale, counts) in ale_curves.items():
    pd.DataFrame({
        "bin_midpoint_original_scale": mid,
        "ale_centred_m3": ale,
        "bin_count": counts,
    }).to_csv(EVAL_DIR / f"ale_{name}.csv", index=False)
print(f"Wrote ale_{{DT,DM,DO}}.csv\n")
