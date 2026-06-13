"""
Aggregate Monte Carlo results into paper-ready summaries.

For each of the 9 scenarios (3 storms x 3 years), we have:
  - result_summary_<storm>_<year>_seeds0-999.csv  (100 tracts x 1000 samples)
    Per-sample CL_fire_stations ratio per census tract
  - result_network_<storm>_<year>_seeds0-999.csv  (58813 links x 1000 samples)
    Per-sample road_closure (0/1) per network link

This script produces:
  1. tract_summary.csv  (n_tracts x [scenario] x [stat])
       Per-tract mean / std / SEM / 95% CI for each scenario
  2. link_summary.csv   (n_links x [scenario] x [stat])
       Per-link closure-rate p / std / SEM for each scenario
  3. county_aggregate.csv
       County-wide aggregate: mean-of-tract-means, max, etc., per scenario
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

RDIR = Path("debris_volume_model/monte_carlo_results")

SCENARIOS = [
    ("ike", 2020), ("ike", 2030), ("ike", 2040),
    ("fema33", 2020), ("fema33", 2030), ("fema33", 2040),
    ("fema36", 2020), ("fema36", 2030), ("fema36", 2040),
]

DISPLAY_YEAR = {2020: 2020, 2030: 2030, 2040: 2040}


def aggregate_per_tract():
    print("=" * 60)
    print("Aggregating per-tract CL results")
    print("=" * 60)
    rows = {}
    for storm, year in SCENARIOS:
        f = RDIR / f"result_summary_{storm}_{year}_seeds0-999.csv"
        df = pd.read_csv(f, index_col=0)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        arr = df[sample_cols].to_numpy(dtype=np.float32)
        means = arr.mean(axis=1)
        stds = arr.std(axis=1, ddof=1)
        sems = stds / np.sqrt(arr.shape[1])
        ci_lo = means - 1.96 * sems
        ci_hi = means + 1.96 * sems
        tag = f"{storm}_{DISPLAY_YEAR[year]}"
        rows[f"{tag}_mean"] = means
        rows[f"{tag}_sem"] = sems
        rows[f"{tag}_ci_lo"] = ci_lo
        rows[f"{tag}_ci_hi"] = ci_hi
        print(f"  {tag:18s}  mean_CL = {means.mean():.4f}  max_tract = {means.max():.4f}  "
              f"max_sem = {sems.max():.4f}")
    out = pd.DataFrame(rows)
    out.index.name = "tract_id"
    out_path = RDIR / "tract_summary.csv"
    out.to_csv(out_path)
    print(f"\nWrote {out_path} ({out.shape})")


def aggregate_per_link():
    print()
    print("=" * 60)
    print("Aggregating per-link road-closure rates")
    print("=" * 60)
    rows = {}
    for storm, year in SCENARIOS:
        f = RDIR / f"result_network_{storm}_{year}_seeds0-999.csv"
        df = pd.read_csv(f, index_col=0)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        arr = df[sample_cols].to_numpy(dtype=np.float32)
        p_close = arr.mean(axis=1)
        sems = np.sqrt(p_close * (1 - p_close) / arr.shape[1])
        n_active = int((p_close > 0.05).sum())
        tag = f"{storm}_{DISPLAY_YEAR[year]}"
        rows[f"{tag}_p_close"] = p_close
        rows[f"{tag}_sem"] = sems
        print(f"  {tag:18s}  mean_p_close = {p_close.mean():.4f}  "
              f"max_p_close = {p_close.max():.4f}  active_links(>5%) = {n_active}")
    out = pd.DataFrame(rows)
    out.index.name = "link_id"
    out_path = RDIR / "link_summary.csv"
    out.to_csv(out_path)
    print(f"\nWrote {out_path} ({out.shape})")


def county_aggregate():
    print()
    print("=" * 60)
    print("County-wide aggregate (mean-of-tract-means per sample, then summarized)")
    print("=" * 60)
    rows = []
    for storm, year in SCENARIOS:
        f = RDIR / f"result_summary_{storm}_{year}_seeds0-999.csv"
        df = pd.read_csv(f, index_col=0)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        arr = df[sample_cols].to_numpy(dtype=np.float32)
        # County aggregate per sample = mean across tracts
        county_per_sample = arr.mean(axis=0)  # 1000 values
        rows.append({
            "storm": storm,
            "display_year": DISPLAY_YEAR[year],
            "county_mean_cl": float(county_per_sample.mean()),
            "county_std_cl": float(county_per_sample.std(ddof=1)),
            "county_sem_cl": float(county_per_sample.std(ddof=1) / np.sqrt(len(county_per_sample))),
            "county_ci95_lo": float(county_per_sample.mean() - 1.96 * county_per_sample.std(ddof=1) / np.sqrt(len(county_per_sample))),
            "county_ci95_hi": float(county_per_sample.mean() + 1.96 * county_per_sample.std(ddof=1) / np.sqrt(len(county_per_sample))),
        })
    out = pd.DataFrame(rows)
    out_path = RDIR / "county_aggregate.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    aggregate_per_tract()
    aggregate_per_link()
    county_aggregate()
