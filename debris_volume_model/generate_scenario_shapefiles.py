"""
For a chosen trained model + scaler + manifest + calibration constant,
re-predict debris volume on the FULL 250-m grid for each of the nine
storm * year forward scenarios and save the result as a shapefile that
matches the format of the original NN's outputs:

    final_debris_volume_output/<storm>/<year>/V13_NN_all_parameters_nn_model_epoch_100_predictions.{shp,csv}

The new outputs go to the *parallel* directory:

    final_debris_volume_output/<storm>/<year>/<output_prefix>_predictions.{shp,csv}

so the existing files are not overwritten.  The downstream dispersion
model and network analysis can then be re-run against the new
prediction shapefiles.

Usage:
    python generate_scenario_shapefiles.py --run-dir outputs/run_v8a \
        --manifest feature_groups_v3.json --output-prefix V14_physics_v8a \
        [--no-calibrate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from model import PhysicsInformedDebrisNN, split_inputs

PROJECT_ROOT = THIS_DIR.parent
INPUT_PARAMS_DIR = PROJECT_ROOT / "outputs" / "final_input_for_debris_volume_model"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_debris_volume_output"
ENGINEERED_CSV = THIS_DIR / "engineered_input_2026_05_04.csv"

SCENARIOS = [
    ("ike",    2019), ("ike",    2030), ("ike",    2040),
    ("fema33", 2019), ("fema33", 2030), ("fema33", 2040),
    ("fema36", 2019), ("fema36", 2030), ("fema36", 2040),
]
# Optional: include the in-sample sanity scenario
SANITY_SCENARIO = ("ike", 2008)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output-prefix", type=str, required=True,
                   help="filename prefix for the new shapefiles")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip post-hoc multiplicative calibration")
    p.add_argument("--include-sanity", action="store_true",
                   help="also generate predictions for the Ike-2008 in-sample case")
    return p.parse_args()


def _load_args_from_run(run_dir: Path) -> dict:
    with open(run_dir / "manifest.json") as f:
        m = json.load(f)
    return m.get("args", {})


def load_model_from_run(run_dir, n_inc, n_dec, n_free, device):
    saved = _load_args_from_run(run_dir)
    h_inc = tuple(saved.get("h_inc", [32, 16]))
    h_dec = tuple(saved.get("h_dec", [16, 8]))
    h_free = tuple(saved.get("h_free", [32, 16]))
    dropout = saved.get("dropout", 0.1)
    print(f"  arch h_inc={h_inc} h_dec={h_dec} h_free={h_free} dropout={dropout}")
    model = PhysicsInformedDebrisNN(
        n_inc=n_inc, n_dec=n_dec, n_free=n_free,
        h_inc=h_inc, h_dec=h_dec, h_free=h_free, dropout=dropout,
    ).to(device)
    ckpt = torch.load(run_dir / "model_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def get_dist_to_coast_mapping():
    df = pd.read_csv(ENGINEERED_CSV)[["FID", "dist_to_coast_m"]]
    df["FID"] = df["FID"].astype(int)
    return dict(zip(df["FID"].values, df["dist_to_coast_m"].values))


def engineer_scenario(df_raw, dist_map):
    df = df_raw.copy()
    df["FID"] = df["FID"].astype(int)
    nb = df["NumB"].astype(float).values
    tbfa = df["TBFA"].astype(float).values
    df["mean_bldg_size_m2"] = np.where(nb > 0, tbfa / np.maximum(nb, 1.0), 0.0)
    df["dist_to_coast_m"] = df["FID"].map(dist_map).astype(float)
    # v7d: hazard x building interaction features (computed on the fly so v7d
    # can score per-scenario CSVs without needing to pre-edit them)
    for haz in ("SD", "WH", "MF"):
        col = f"{haz}_NumB"
        if col not in df.columns and haz in df.columns:
            df[col] = df[haz].astype(float).values * nb
    return df


@torch.no_grad()
def predict_full(model, X_scaled, n_inc, n_dec, device, batch_size=4096):
    model.eval()
    preds = []
    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32, device=device)
        x_inc, x_dec, x_free = split_inputs(xb, n_inc, n_dec)
        preds.append(model(x_inc, x_dec, x_free).cpu().numpy())
    return np.concatenate(preds)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.manifest)

    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_order = manifest["monotone_inc"] + manifest["monotone_dec"] + manifest["free"]
    n_inc, n_dec = len(manifest["monotone_inc"]), len(manifest["monotone_dec"])
    n_free = len(feature_order) - n_inc - n_dec

    scaler = joblib.load(run_dir / "scaler.pkl")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model_from_run(run_dir, n_inc, n_dec, n_free, device)
    print(f"Loaded model best epoch={ckpt['epoch']} from {run_dir}")

    cal = 1.0
    if not args.no_calibrate:
        with open(run_dir / "final_metrics.json") as f:
            fm = json.load(f)
        cal = fm["train"]["mean_true"] / max(fm["train"]["mean_pred"], 1e-6)
        print(f"Calibration c = {cal:.4f}")

    dist_map = get_dist_to_coast_mapping()
    scenarios = list(SCENARIOS)
    if args.include_sanity:
        scenarios = [SANITY_SCENARIO] + scenarios

    for storm, year in scenarios:
        print(f"\n--- {storm} / {year} ---")
        csv_path = INPUT_PARAMS_DIR / storm / str(year) / "final_input_parameters.csv"
        if not csv_path.is_file():
            print(f"  [skip] {csv_path} not found")
            continue
        df_raw = pd.read_csv(csv_path)
        df_eng = engineer_scenario(df_raw, dist_map)
        n_total = len(df_eng)
        valid_mask = ~df_eng[feature_order].isna().any(axis=1)
        valid = df_eng.loc[valid_mask].copy()

        X = valid[feature_order].to_numpy(dtype=np.float32)
        X_s = scaler.transform(X).astype(np.float32)
        pred = predict_full(model, X_s, n_inc, n_dec, device) * cal
        valid["pred_m3"] = pred

        # Re-attach to full grid (cells with NaN inputs get pred=0)
        full = df_raw[["FID"]].copy()
        full = full.merge(valid[["FID", "pred_m3"]], on="FID", how="left")
        full["pred_m3"] = full["pred_m3"].fillna(0.0)
        # Also include the original 30 raw input columns + geometry so the
        # output is a drop-in replacement for V13_NN_*.shp.
        full = full.merge(df_raw, on="FID", how="left", suffixes=("", "_dup"))
        full = full.drop(columns=[c for c in full.columns if c.endswith("_dup")])
        # WKT geometry -> Shapely
        from shapely import wkt as _wkt
        if "geometry" in full.columns and full["geometry"].dtype == object:
            full["geometry"] = full["geometry"].apply(_wkt.loads)
        gdf = gpd.GeoDataFrame(full, geometry="geometry", crs="EPSG:4326")

        out_dir = OUTPUT_DIR / storm / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        shp_path = out_dir / f"{args.output_prefix}_predictions.shp"
        csv_path_out = out_dir / f"{args.output_prefix}_predictions.csv"

        # Save (truncate column names for ESRI shapefile constraints)
        # All our column names are <= 10 chars except 'mean_bldg_size_m2'
        # which would be added by engineer step -- but we keep the raw input
        # only here, so no conflict.  Just in case, let geopandas handle it.
        gdf_to_save = gdf.copy()
        # ESRI shapefile column names must be <= 10 chars; rename if needed
        renames = {c: c[:10] for c in gdf_to_save.columns
                   if len(c) > 10 and c != gdf_to_save.geometry.name}
        if renames:
            gdf_to_save = gdf_to_save.rename(columns=renames)
        gdf_to_save.to_file(shp_path)
        # CSV (full names)
        full.drop(columns=["geometry"]).to_csv(csv_path_out, index=False)
        n_valid = len(valid)
        total = float(full["pred_m3"].sum())
        print(f"  cells valid/total = {n_valid}/{n_total}, "
              f"sum predicted = {total:,.0f} m^3")
        print(f"  saved {shp_path.name}, {csv_path_out.name}")

    print(f"\nAll scenario shapefiles written under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
