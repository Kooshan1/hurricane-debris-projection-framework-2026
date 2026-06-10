"""
Diagnostic: is DT a near-linear function of DM + DO in the v7d data?
If so, the per-feature coefficients on the trio are non-identifiable
and what is statistically identifiable is the *joint* effect along the
cluster axis. This is the root reason DT alone shows a negative slope
under both PDP and ALE: the model can put any decomposition of the
three coefficients that sums correctly along the manifold, and the one
that gradient descent settled on has DT carrying a large negative
correction to keep the sum honest.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent.parent
CSV = THIS_DIR / "engineered_input_2026_05_04_no_outlier_v7d.csv"

df = pd.read_csv(CSV)
DT = df["DT"].values
DM = df["DM"].values
DO = df["DO"].values

# Pairwise correlations
corr = pd.DataFrame({"DT": DT, "DM": DM, "DO": DO}).corr()
print("Pearson correlation matrix:")
print(corr.round(4).to_string())

# OLS: DT ~ DM + DO  (does DM + DO explain almost all variance in DT?)
X = np.column_stack([np.ones_like(DM), DM, DO])
beta, *_ = np.linalg.lstsq(X, DT, rcond=None)
pred = X @ beta
resid = DT - pred
ss_res = float((resid ** 2).sum())
ss_tot = float(((DT - DT.mean()) ** 2).sum())
R2 = 1 - ss_res / ss_tot
print(f"\nOLS: DT  ~  {beta[0]:+.4f} + {beta[1]:+.4f} * DM + {beta[2]:+.4f} * DO")
print(f"     R^2 = {R2:.4f}   (1.0 = DT is exactly a linear combination of DM, DO)")

# Variance Inflation Factor for DT in the trio (regress DT on DM, DO)
print(f"\nVIF(DT | DM, DO)  =  1 / (1 - R^2)  =  {1 / max(1 - R2, 1e-12):.2f}")
print("(VIF > 10 is the canonical multicollinearity threshold; > 100 = severe.)")

# Equivalent OLS on the centred / scaled triple
print("\n--- Quick descriptive stats of the trio ---")
desc = pd.DataFrame({"DT": DT, "DM": DM, "DO": DO}).describe().round(4)
print(desc.to_string())

# Verify: in the cells where DT is highest (top 5 %), how does DT compare
# to DM + DO?
mask = DT > np.percentile(DT, 95)
print(f"\nIn the highest-DT cells (top 5 %; n = {int(mask.sum())}):")
print(f"  mean DT      = {DT[mask].mean():.4f}")
print(f"  mean DM + DO = {(DM[mask] + DO[mask]).mean():.4f}")
print(f"  mean (DT - DM - DO) = {(DT[mask] - DM[mask] - DO[mask]).mean():+.4f}")
print(f"  (positive residual = additional developed area NOT in DM or DO)")

# Save numbers
out_csv = THIS_DIR / "outputs" / "v7d_dev_cluster_collinearity.csv"
pd.DataFrame({
    "metric": [
        "corr(DT, DM)", "corr(DT, DO)", "corr(DM, DO)",
        "R^2 of DT ~ DM + DO",
        "VIF(DT | DM, DO)",
        "intercept of OLS",
        "slope on DM",
        "slope on DO",
        "mean(DT - DM - DO) in top-5% DT cells",
    ],
    "value": [
        float(corr.loc["DT", "DM"]), float(corr.loc["DT", "DO"]),
        float(corr.loc["DM", "DO"]),
        float(R2), float(1 / max(1 - R2, 1e-12)),
        float(beta[0]), float(beta[1]), float(beta[2]),
        float((DT[mask] - DM[mask] - DO[mask]).mean()),
    ],
}).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
