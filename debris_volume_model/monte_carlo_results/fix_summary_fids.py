"""
Fix the FID indexing in the 9 per-tract result_summary CSVs.

Bug recap:
  - During the Monte Carlo run, each per-sample DataFrame inherited
    `summary_gdf.index` (an integer permutation of 0..99) as its row index,
    instead of the true tract FID from summary_gdf['FID'].
  - When PlotGenerator merges the resulting CSV's "FID" column with the tract
    shapefile by VALUE, ~98 out of 100 tract values are misattributed and 2
    are dropped entirely.

Fix:
  - Use the position-to-FID mapping (`position_to_fid.csv`) produced by
    a dedicated FID diagnostic run, which captured the exact
    summary_gdf.index -> true_tract_FID mapping (deterministic, identical
    across all 9 scenarios because clip + sjoin + groupby are deterministic
    given the same tract + network shapefiles).
  - For each result_summary_*_seeds0-999.csv:
      * Read the CSV
      * Replace the first column (the bad "FID") with the true tract FID via
        the mapping
      * Re-save with the correct "FID" column name

Idempotent: if a CSV already has the correct FIDs (e.g., {1..101} - {34}), the
script reports "ALREADY FIXED" and skips it.

Outputs are written IN PLACE (with a one-shot backup copy of each file before
overwrite).
"""
from __future__ import annotations
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = Path("debris_volume_model/monte_carlo_results")
MAPPING_CSV = RUNS_DIR / "position_to_fid.csv"

SCENARIOS = [
    ("ike", 2020), ("ike", 2030), ("ike", 2040),
    ("fema33", 2020), ("fema33", 2030), ("fema33", 2040),
    ("fema36", 2020), ("fema36", 2030), ("fema36", 2040),
]


def main():
    # Load + sanity-check the mapping
    m = pd.read_csv(MAPPING_CSV)
    assert len(m) == 100, f"Mapping must have 100 rows, got {len(m)}"
    assert m["csv_index_value"].nunique() == 100, "Mapping has duplicate csv indices"
    assert m["true_tract_FID"].nunique() == 100, "Mapping has duplicate true FIDs"
    true_fids_sorted = sorted(m["true_tract_FID"].tolist())
    # The 100 true tract FIDs should be [1..101] except 34
    expected = list(range(1, 34)) + list(range(35, 102))
    assert true_fids_sorted == expected, (
        f"Mapping's true FIDs don't match shapefile expectation. "
        f"Got first 5: {true_fids_sorted[:5]}, expected first 5: {expected[:5]}")
    print("Mapping is valid: 100 unique csv indices -> 100 unique true tract FIDs in [1..101]\\{34}")

    mapping = dict(zip(m["csv_index_value"], m["true_tract_FID"]))

    backup_dir = RUNS_DIR / "pre_fid_fix_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    fixed = 0
    skipped = 0
    for storm, year in SCENARIOS:
        csv_path = RUNS_DIR / f"result_summary_{storm}_{year}_seeds0-999.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found")
            continue

        df = pd.read_csv(csv_path)
        first_col_name = df.columns[0]
        first_col_vals = df[first_col_name].to_list()

        # Idempotency: if the first col already matches the expected true FIDs
        # (in some order), skip.
        if set(first_col_vals) == set(expected):
            print(f"  [skip] {csv_path.name} - first column already contains true FIDs.")
            skipped += 1
            continue

        # Sanity: first col should be exactly the 100 csv index values
        if set(first_col_vals) != set(mapping.keys()):
            extra = set(first_col_vals) - set(mapping.keys())
            missing = set(mapping.keys()) - set(first_col_vals)
            raise RuntimeError(
                f"{csv_path.name}: first-col values don't match the mapping keys. "
                f"Extra in CSV: {sorted(extra)[:10]}...  Missing from CSV: {sorted(missing)[:10]}...")

        # Backup once
        bp = backup_dir / csv_path.name
        if not bp.exists():
            shutil.copy2(csv_path, bp)
            print(f"  backup: {bp.name}")

        # Apply mapping
        df_new = df.copy()
        df_new[first_col_name] = df_new[first_col_name].map(mapping)
        # Rename first column to "FID" (matches original 700-sample CSV convention)
        df_new = df_new.rename(columns={first_col_name: "FID"})
        # Optional: sort by FID for clean downstream reads (matches original CSV layout)
        df_new = df_new.sort_values("FID").reset_index(drop=True)

        df_new.to_csv(csv_path, index=False)
        print(f"  fixed: {csv_path.name}  (n_rows={len(df_new)}, FID range {df_new['FID'].min()}..{df_new['FID'].max()})")
        fixed += 1

    print(f"\nDone. Fixed {fixed}, skipped {skipped}, total scenarios {len(SCENARIOS)}.")


if __name__ == "__main__":
    main()
