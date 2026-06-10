"""
Step 1 of the revision-figure pipeline.

Reads the per-sample Monte-Carlo CSVs that were already produced by the main
pipeline (700 samples for each of the nine storm x year scenarios) and writes
small derived CSVs containing per-tract / per-link uncertainty metrics:

    mean, std, p05, p95, p_exceed_0.1, p_exceed_0.3, p_exceed_0.5

The derived CSVs are written to:

    outputs/debris_impact_output/monte_carlo_result/derived_uncertainty/
        clr_metrics_<storm>_<file_year>.csv
        network_metrics_<storm>_<file_year>.csv

These files are then consumed by 02_make_uncertainty_maps.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (HURRICANES, MC_DIR_700, YEAR_DISPLAY_TO_FILE,
                    DISPLAY_YEARS)

DERIVED_DIR = MC_DIR_700 / "derived_uncertainty"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def _summarise(per_sample: pd.DataFrame, id_col: str, thresholds=(0.1, 0.3, 0.5)) -> pd.DataFrame:
    """Compute per-row uncertainty metrics across the sample columns."""
    sample_cols = [c for c in per_sample.columns if c != id_col]
    arr = per_sample[sample_cols].to_numpy(dtype=float)
    out = pd.DataFrame({id_col: per_sample[id_col].astype(np.int64).values})
    out["mean"] = np.nanmean(arr, axis=1)
    out["std"] = np.nanstd(arr, axis=1, ddof=1)
    out["p05"] = np.nanpercentile(arr, 5, axis=1)
    out["p95"] = np.nanpercentile(arr, 95, axis=1)
    out["ci90_width"] = out["p95"] - out["p05"]
    for t in thresholds:
        out[f"p_exceed_{t:.2f}"] = np.nanmean(arr > t, axis=1)
    return out


def derive_clr_metrics():
    print("\n--- CLR per-tract metrics ---")
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv_in = MC_DIR_700 / f"result_summary_{storm}_{file_year}_v7d_seeds0-999.csv"
            if not csv_in.is_file():
                print(f"  [skip] {csv_in.name} not found")
                continue
            df = pd.read_csv(csv_in)
            id_col = df.columns[0]
            if str(id_col).upper() != "FID":
                df = df.rename(columns={id_col: "FID"})
                id_col = "FID"
            metrics = _summarise(df, id_col=id_col)
            out_path = DERIVED_DIR / f"clr_metrics_{storm}_{file_year}.csv"
            metrics.to_csv(out_path, index=False)
            print(f"  wrote {out_path.name}  ({len(metrics)} tracts)")


def derive_network_metrics():
    print("\n--- Network per-link metrics ---")
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv_in = MC_DIR_700 / f"result_network_{storm}_{file_year}_v7d_seeds0-999.csv"
            if not csv_in.is_file():
                print(f"  [skip] {csv_in.name} not found")
                continue
            df = pd.read_csv(csv_in)
            id_col = df.columns[0]
            metrics = _summarise(df, id_col=id_col, thresholds=(0.1, 0.3, 0.5))
            out_path = DERIVED_DIR / f"network_metrics_{storm}_{file_year}.csv"
            metrics.to_csv(out_path, index=False)
            print(f"  wrote {out_path.name}  ({len(metrics)} links)")


if __name__ == "__main__":
    derive_clr_metrics()
    derive_network_metrics()
    print(f"\nAll derived metrics saved under: {DERIVED_DIR}")
