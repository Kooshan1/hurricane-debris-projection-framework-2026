"""
Convergence diagnostics for the Monte Carlo results.

Reports two metrics per scenario:

  (1) Tract-level CL ratio convergence (the paper-relevant aggregate):
      - For each tract, sample mean of CL across N samples
      - Standard error of the mean (SEM) at N=1000
      - Worst-case (max over tracts) and aggregate (mean over tracts) SEM
      - 'Converged' if max-tract SEM < threshold (default 0.005 = 0.5 % CL units)

  (2) Network-link closure-rate convergence:
      - For each link, fraction of samples where road_closure = 1
      - Standard error sqrt(p*(1-p)/N) per link
      - Same threshold-based verdict

Outputs a summary table.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str,
                   default="debris_volume_model/monte_carlo_results")
    p.add_argument("--summary-tract-threshold", type=float, default=0.005,
                   help="Max-tract SEM threshold for tract-level CL convergence (default 0.005)")
    p.add_argument("--network-link-threshold", type=float, default=0.01,
                   help="Max-active-link SEM threshold for closure-rate convergence (default 0.01)")
    p.add_argument("--min-active-link-mean", type=float, default=0.05,
                   help="Skip links with mean closure < this for network conv check")
    return p.parse_args()


def check_summary(csv_path: Path, threshold: float) -> dict:
    df = pd.read_csv(csv_path, index_col=0)
    sample_cols = [c for c in df.columns if c.startswith("sample_")]
    n = len(sample_cols)
    arr = df[sample_cols].to_numpy(dtype=np.float32)  # n_tracts x n_samples

    means = arr.mean(axis=1)
    stds = arr.std(axis=1, ddof=1)
    sems = stds / np.sqrt(n)
    return {
        "file": csv_path.name,
        "n_samples": n,
        "n_tracts": arr.shape[0],
        "mean_cl_overall": float(means.mean()),
        "max_tract_mean": float(means.max()),
        "mean_tract_sem": float(sems.mean()),
        "max_tract_sem": float(sems.max()),
        "p95_tract_sem": float(np.percentile(sems, 95)),
        "verdict": "CONVERGED" if sems.max() < threshold else "NEEDS_MORE",
    }


def check_network(csv_path: Path, threshold: float, min_mean: float) -> dict:
    df = pd.read_csv(csv_path, index_col=0)
    sample_cols = [c for c in df.columns if c.startswith("sample_")]
    n = len(sample_cols)
    arr = df[sample_cols].to_numpy(dtype=np.float32)  # n_links x n_samples

    p_close = arr.mean(axis=1)
    sems = np.sqrt(p_close * (1 - p_close) / n)
    active = p_close > min_mean
    n_active = int(active.sum())
    if n_active == 0:
        return {"file": csv_path.name, "n_samples": n, "n_links": arr.shape[0],
                "n_active_links": 0, "mean_closure_rate": float(p_close.mean()),
                "max_active_sem": np.nan, "verdict": "NO_ACTIVE_LINKS"}
    sems_active = sems[active]
    return {
        "file": csv_path.name,
        "n_samples": n,
        "n_links": arr.shape[0],
        "n_active_links": n_active,
        "mean_closure_rate": float(p_close.mean()),
        "active_link_mean_p_close": float(p_close[active].mean()),
        "mean_active_sem": float(sems_active.mean()),
        "max_active_sem": float(sems_active.max()),
        "p95_active_sem": float(np.percentile(sems_active, 95)),
        "verdict": "CONVERGED" if sems_active.max() < threshold else "NEEDS_MORE",
    }


def main():
    args = parse_args()
    rdir = Path(args.results_dir)

    print("=" * 80)
    print("(1) TRACT-LEVEL CL-RATIO CONVERGENCE (paper aggregate)")
    print("=" * 80)
    summary_rows = []
    for f in sorted(rdir.glob("result_summary_*_seeds*.csv")):
        row = check_summary(f, args.summary_tract_threshold)
        summary_rows.append(row)
        print(f"{row['file']:<55s} {row['verdict']:<12s} "
              f"max_sem={row['max_tract_sem']:.4f}  "
              f"mean_sem={row['mean_tract_sem']:.4f}  "
              f"mean_CL={row['mean_cl_overall']:.4f}")
    pd.DataFrame(summary_rows).to_csv(rdir / "convergence_summary_tracts.csv", index=False)

    print()
    print("=" * 80)
    print("(2) NETWORK-LINK CLOSURE-RATE CONVERGENCE (active links only)")
    print("=" * 80)
    network_rows = []
    for f in sorted(rdir.glob("result_network_*_seeds*.csv")):
        row = check_network(f, args.network_link_threshold, args.min_active_link_mean)
        network_rows.append(row)
        print(f"{row['file']:<55s} {row['verdict']:<12s} "
              f"max_sem={row.get('max_active_sem', float('nan')):.4f}  "
              f"n_active={row.get('n_active_links', 0):>5d}  "
              f"mean_closure={row['mean_closure_rate']:.4f}")
    pd.DataFrame(network_rows).to_csv(rdir / "convergence_summary_network.csv", index=False)


if __name__ == "__main__":
    main()
