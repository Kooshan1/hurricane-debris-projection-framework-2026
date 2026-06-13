"""
Compute PDP slope (5th -> 95th percentile range in m^3) for EVERY feature
in the training data, and compare to expected physical sign.

For each feature:
  - sign_INC (+): debris should INCREASE with feature (more buildings, more hazard, etc.)
  - sign_DEC (-): debris should DECREASE with feature (more open water, farther from coast)
  - sign_FREE (?): no a-priori expectation
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs

CSV = "training_data.csv"
MAN = "feature_groups.json"
RUN_DIR = "trained_model"
SEED = 7

# Expected physical signs (paper-defensible reasoning)
EXPECTED_SIGN = {
    # Monotone-INC (architecturally enforced)
    "SD": "+ (surge depth -> more debris; arch-enforced INC)",
    "WH": "+ (wave height -> more debris; arch-enforced INC)",
    "WS": "+ (wind steadiness -> more debris; arch-enforced INC)",
    "MF": "+ (momentum flux -> more debris; arch-enforced INC)",
    "WPF_1": "+ (failure prob -> more debris; arch-enforced INC)",
    "WPF_2": "+ (failure prob -> more debris; arch-enforced INC)",
    "NumB": "+ (more buildings -> more debris; arch-enforced INC)",
    "mean_bldg_size_m2": "+ (larger buildings -> more debris; arch-enforced INC)",
    "NAS": "+ (more accessory structures -> more debris; arch-enforced INC)",
    "TAAS": "+ (more accessory area -> more debris; arch-enforced INC)",
    "NumMH": "+ (more mobile homes (vulnerable) -> more debris; arch-enforced INC)",
    "NumHU": "+ (more housing units -> more debris; arch-enforced INC)",
    "OHU": "+ (more occupied housing -> more debris; arch-enforced INC)",
    "VHU": "+ (more vacant housing -> more debris; arch-enforced INC)",
    "PD": "+ (more population -> more debris; arch-enforced INC)",
    "SD_NumB": "+ (surge x buildings exposed; arch-enforced INC)",
    "WH_NumB": "+ (wave x buildings exposed; arch-enforced INC)",
    "MF_NumB": "+ (momentum flux x buildings exposed; arch-enforced INC)",
    # Monotone-DEC (architecturally enforced)
    "ME": "- (higher elevation -> less surge -> less debris; arch-enforced DEC)",
    "dist_to_coast_m": "- (further from coast -> less surge -> less debris; arch-enforced DEC)",
    # Free (sign learned from data; we have priors)
    "WV_X": "? (signed water velocity x; either sign physically possible)",
    "WV_Y": "? (signed water velocity y; either sign physically possible)",
    "OW": "- (more open water -> less land -> less debris)",
    "DO": "+ (more dev-open -> modest debris)",
    "DM": "+ (more dev-medium -> more debris)",
    "DT": "+ (more dev-total -> more debris)",
    "RD": "+ (more roads -> more developed area -> more debris)",
    "ADS": "- (further from seawall -> less surge exposure -> less debris)",
    "MHI": "? (richer areas have more expensive but better-built structures)",
    "PR": "? (renters vs owners - debris quality vs maintenance trade-off)",
}

with open(MAN) as f:
    m = json.load(f)
feat = m["monotone_inc"] + m["monotone_dec"] + m["free"]
n_inc, n_dec = len(m["monotone_inc"]), len(m["monotone_dec"])
n_free = len(feat) - n_inc - n_dec

df = pd.read_csv(CSV).dropna(subset=feat + ["volume_m3"]).reset_index(drop=True)
X = df[feat].to_numpy(dtype=np.float32)
y = df["volume_m3"].to_numpy(dtype=np.float32).reshape(-1, 1)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED)
Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED)

scaler = joblib.load(Path(RUN_DIR) / "scaler.pkl")
Xte_s = scaler.transform(Xte).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhysicsInformedDebrisNN(n_inc=n_inc, n_dec=n_dec, n_free=n_free).to(device)
ckpt = torch.load(Path(RUN_DIR) / "model_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

c_factor = float(np.array(ytr).flatten().mean() /
                 max(model_pred_mean := 1.0, 1e-9))  # placeholder; we use raw PDP slope, sign is independent of c

@torch.no_grad()
def predict(X_scaled):
    out = []
    for i in range(0, len(X_scaled), 4096):
        xb = torch.tensor(X_scaled[i:i+4096], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        out.append(model(x_inc, x_dec, x_free).cpu().numpy())
    return np.concatenate(out).flatten()

# 1-D PDP for every feature
N_GRID = 40
results = []
for j, name in enumerate(feat):
    feat_vals_orig = Xte[:, j]
    p5  = float(np.percentile(feat_vals_orig, 5))
    p95 = float(np.percentile(feat_vals_orig, 95))
    if abs(p95 - p5) < 1e-12:  # constant feature
        results.append({"feature": name, "p5": p5, "p95": p95, "pdp_p5": np.nan,
                        "pdp_p95": np.nan, "pdp_range": np.nan,
                        "expected_sign": EXPECTED_SIGN.get(name, "?"),
                        "observed_sign": "constant", "verdict": "n/a"})
        continue
    # Need to scale grid in scaled space
    grid_orig = np.linspace(p5, p95, N_GRID)
    # mean of feature in scaled space (for setting other rows constant) - we vary just column j
    base = Xte_s.copy()
    pdp = np.empty(N_GRID, dtype=np.float32)
    feat_mean = scaler.mean_[j]; feat_std = scaler.scale_[j]
    for k, v in enumerate(grid_orig):
        v_scaled = (v - feat_mean) / feat_std
        base[:, j] = v_scaled
        pdp[k] = predict(base).mean()
    pdp_range = float(pdp[-1] - pdp[0])
    obs_sign = "+" if pdp_range > 1.0 else ("-" if pdp_range < -1.0 else "~0")
    exp = EXPECTED_SIGN.get(name, "?")
    if exp.startswith("+"):
        verdict = "OK" if pdp_range > 0 else ("MILD" if pdp_range > -10 else "WRONG")
    elif exp.startswith("-"):
        verdict = "OK" if pdp_range < 0 else ("MILD" if pdp_range < 10 else "WRONG")
    else:
        verdict = "n/a (free)"
    results.append({
        "feature": name, "p5": round(p5, 4), "p95": round(p95, 4),
        "pdp_p5":  round(float(pdp[0]),  3), "pdp_p95": round(float(pdp[-1]), 3),
        "pdp_range": round(pdp_range, 3),
        "expected_sign": exp,
        "observed_sign": obs_sign, "verdict": verdict,
    })

df_out = pd.DataFrame(results).sort_values("pdp_range", key=lambda s: s.abs(), ascending=False)
out_csv = "outputs/pdp_signs.csv"
df_out.to_csv(out_csv, index=False)
print(f"Wrote {out_csv}\n")
print(df_out.to_string(index=False))
