"""
Local one-shot script that produces the engineered input CSV used by the
new physics-informed test model.

Produces a single self-contained CSV (FID, all 31 input features, target
volume_m3) so the remote training environment does not need geopandas /
rasterio / shapely.

Reads:
    outputs/final_input_for_debris_volume_model/ike/2008/final_input_parameters.csv
    outputs/debris_volume/Grid_250_with_debris_volume.shp

Writes:
    debris_volume_model/training_data.csv
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---- paths --------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # NSF_Debris_Project/
RAW_INPUT_CSV = (PROJECT_ROOT / "outputs" / "final_input_for_debris_volume_model"
                 / "ike" / "2008" / "final_input_parameters.csv")
TARGET_SHP = PROJECT_ROOT / "outputs" / "debris_volume" / "Grid_250_with_debris_volume.shp"

# ---- feature engineering ------------------------------------------------
ORIGINAL_FEATURES = [
    'SD', 'WH', 'WaveD', 'WV_X', 'WV_Y', 'WindD', 'WS', 'MF',
    'NumB', 'TBFA', 'NAS', 'TAAS', 'NumMH', 'WPF_1', 'WPF_2',
    'OW', 'DO', 'DM', 'DT', 'RD', 'ME', 'ADS', 'UL',
    'PD', 'NumH', 'MHI', 'NumHU', 'OHU', 'VHU', 'PR',
]


def main():
    print("Loading raw input CSV ...")
    df = pd.read_csv(RAW_INPUT_CSV)
    df["FID"] = df["FID"].astype(int)

    print("Loading target shapefile (volumes + geometry) ...")
    gdf_y = gpd.read_file(TARGET_SHP)
    gdf_y["FID"] = gdf_y["FID"].astype(int)

    # Project to a *projected* CRS (UTM zone 15N, EPSG:32615) so distance
    # computations are in metres rather than degrees. Galveston is in UTM 15N.
    print(f"Reprojecting to EPSG:32615 (UTM 15N, units = metres) for accurate distances ...")
    gdf_y_utm = gdf_y.to_crs("EPSG:32615")
    centroids_utm = gdf_y_utm.geometry.centroid
    cx = centroids_utm.x.values
    cy = centroids_utm.y.values

    merged = df.merge(gdf_y_utm[["FID", "volume_m3"]], on="FID", how="left")
    merged["centroid_x_utm_m"] = cx
    merged["centroid_y_utm_m"] = cy

    # Drop rows with missing target (shouldn't happen but defensive)
    merged = merged.dropna(subset=["volume_m3"])
    n = len(merged)
    print(f"  rows after merge / dropna: {n}")

    # ---- engineered feature 1: mean building size --------------------
    # mean_bldg_size_m2 = TBFA / max(NumB, 1).  When NumB == 0, set to 0
    # (matches "no buildings -> no characteristic size"; the model can still
    # use NumB=0 as the dominant signal).
    nb = merged["NumB"].astype(float).values
    tbfa = merged["TBFA"].astype(float).values
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_size = np.where(nb > 0, tbfa / np.maximum(nb, 1.0), 0.0)
    merged["mean_bldg_size_m2"] = mean_size
    print(f"  mean_bldg_size_m2: min={mean_size.min():.1f}, "
          f"median={np.median(mean_size):.1f}, max={mean_size.max():.1f}")

    # Sanity: correlation of (TBFA, NumB) and (mean_bldg_size, NumB)
    corr_tbfa_numb = np.corrcoef(tbfa, nb)[0, 1]
    corr_meansize_numb = np.corrcoef(mean_size, nb)[0, 1]
    print(f"  Pearson r(TBFA, NumB) = {corr_tbfa_numb:.3f}  (this is what we replace)")
    print(f"  Pearson r(mean_bldg_size, NumB) = {corr_meansize_numb:.3f}  (decorrelated)")

    # ---- engineered feature 2: distance to coast ---------------------
    # We define "coast" as the centroid of any grid cell whose 'OW' (open
    # water fraction) is >= 0.3.  These are the water-dominated cells; the
    # land-water interface is then implicit, and distance to that interface
    # is a translation-invariant proxy for distance-to-coast.  This uses
    # NO storm-specific information.
    OW_THRESHOLD = 0.3
    is_water = merged["OW"].astype(float).values >= OW_THRESHOLD
    n_water = int(is_water.sum())
    print(f"  Cells with OW >= {OW_THRESHOLD} (used as coast proxy): {n_water}")

    if n_water == 0:
        print("  No water cells found! falling back to coastline = bbox edge.")
        merged["dist_to_coast_m"] = 0.0
    else:
        water_xy = np.column_stack([cx[is_water], cy[is_water]])
        tree = cKDTree(water_xy)
        all_xy = np.column_stack([cx, cy])
        dists, _ = tree.query(all_xy, k=1)
        # cells that are themselves water get dist=0
        merged["dist_to_coast_m"] = dists

    d = merged["dist_to_coast_m"].values
    print(f"  dist_to_coast_m: min={d.min():.0f}, median={np.median(d):.0f}, "
          f"p95={np.percentile(d, 95):.0f}, max={d.max():.0f}")

    # ---- final feature list ------------------------------------------
    feature_cols_dropped = ["centroid_x", "centroid_y", "TBFA"]
    feature_cols_added = ["mean_bldg_size_m2", "dist_to_coast_m"]

    # Order of features (stable, used by training script):
    monotone_inc = [
        "SD", "WH", "WS", "MF", "WPF_1", "WPF_2",
        "NumB", "mean_bldg_size_m2", "NAS", "TAAS", "NumMH",
    ]
    monotone_dec = ["ME", "dist_to_coast_m"]
    free = [
        "WaveD", "WindD", "WV_X", "WV_Y",
        "OW", "DO", "DM", "DT", "RD",
        "ADS", "UL",
        "PD", "NumH", "MHI", "NumHU", "OHU", "VHU", "PR",
    ]
    feature_order = monotone_inc + monotone_dec + free

    n_inc, n_dec, n_free = len(monotone_inc), len(monotone_dec), len(free)
    print(f"\nFeature groups: inc={n_inc}, dec={n_dec}, free={n_free}, total={n_inc+n_dec+n_free}")
    assert len(feature_order) == 31, f"Expected 31 features, got {len(feature_order)}"

    # Verify all features exist
    missing = [c for c in feature_order if c not in merged.columns]
    if missing:
        print(f"!! MISSING columns: {missing}")
        sys.exit(1)

    # ---- write self-contained CSV ------------------------------------
    out_cols = ["FID"] + feature_order + ["volume_m3",
                                          "centroid_x_utm_m", "centroid_y_utm_m"]
    out = merged[out_cols].copy()
    timestamp = datetime.now().strftime("%Y_%m_%d")
    out_path = THIS_DIR / "training_data.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote engineered CSV: {out_path}")
    print(f"  rows: {len(out)}, columns: {len(out.columns)}")

    # Save the feature-group manifest for the training script
    manifest_path = THIS_DIR / f"feature_groups_{timestamp}.json"
    import json
    manifest = {
        "monotone_inc": monotone_inc,
        "monotone_dec": monotone_dec,
        "free": free,
        "all_features": feature_order,
        "target": "volume_m3",
        "n_features": len(feature_order),
        "csv_file": out_path.name,
        "n_rows": len(out),
        "feature_dropped": feature_cols_dropped,
        "feature_added": feature_cols_added,
        "ow_threshold_for_coast": OW_THRESHOLD,
        "n_water_cells": n_water,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote feature manifest: {manifest_path}")


if __name__ == "__main__":
    main()
