"""
Investigate cells where predicted debris volume DECREASES across the 2020 -> 2030 -> 2040
horizon, and identify which input-feature changes drive the decrease.

Methodology:
  1. For each storm in {ike, fema33, fema36}, load the per-cell debris-volume predictions for 2020/2030/2040.
  2. Compute per-cell deltas: D_30_20 = pred_30 - pred_20 ;  D_40_30 = pred_40 - pred_30
  3. Tag cells as "decreasing" if either delta is < -1 m^3 (small threshold to avoid float noise).
  4. For decreasing cells, load the matching per-cell input CSV and compute the input-feature
     change between the two years.
  5. Identify which features have the LARGEST signed change for the decreasing cells, and
     compare to the trends for non-decreasing cells. This tells us which inputs drove the
     drops.

We focus on free-branch features (DT, DM, DO, RD, ADS, OW, MHI, PR, WV_X, WV_Y) because the
monotone-INC / monotone-DEC features are architecturally CONSTRAINED: their increases can
only push predictions UP. A net DECREASE in a cell's prediction must come from changes in
the FREE branch (or from a rare combination where a monotone-DEC feature decreases).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

PRED_DIR = Path("outputs/final_debris_volume_output")
INPUT_DIR = Path("outputs/final_input_for_debris_volume_model")
OUT_DIR = Path("debris_volume_model/outputs/decreasing_cells")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["ike", "fema33", "fema36"]
FILE_YEARS = [2020, 2030, 2040]      # disk
DISP_YEARS = [2020, 2030, 2040]      # paper

# Free-branch features (where decreases CAN come from architecturally)
FREE_FEATS = ["DT", "DM", "DO", "RD", "ADS", "OW", "MHI", "PR", "WV_X", "WV_Y"]
# Monotone-INC features that should drive INCREASES if they grow year-over-year
MONO_INC_FEATS = ["SD", "WH", "WS", "MF", "WPF_1", "WPF_2", "NumB", "NAS", "TAAS",
                  "NumMH", "NumHU", "OHU", "VHU", "PD"]


def load_pred(storm, file_year):
    p = PRED_DIR / storm / str(file_year) / "debris_volume_predictions.shp"
    g = gpd.read_file(p)[["FID", "pred_m3"]]
    g["FID"] = g["FID"].astype(int)
    return g.rename(columns={"pred_m3": f"pred_{file_year}"})


def load_input(storm, file_year):
    p = INPUT_DIR / storm / str(file_year) / "final_input_parameters.csv"
    df = pd.read_csv(p)
    df["FID"] = df["FID"].astype(int)
    return df


def analyse_storm(storm):
    print(f"\n===== {storm.upper()} =====")
    # Load all 3 years of preds
    p20 = load_pred(storm, 2020)
    p30 = load_pred(storm, 2030)
    p40 = load_pred(storm, 2040)
    df = p20.merge(p30, on="FID", how="outer").merge(p40, on="FID", how="outer")

    # Drop NaNs
    df = df.dropna(subset=["pred_2020", "pred_2030", "pred_2040"]).reset_index(drop=True)
    df["d_30_20"] = df["pred_2030"] - df["pred_2020"]
    df["d_40_30"] = df["pred_2040"] - df["pred_2030"]
    df["d_total"] = df["pred_2040"] - df["pred_2020"]

    n = len(df)
    n_dec_30_20 = (df["d_30_20"] < -1).sum()
    n_dec_40_30 = (df["d_40_30"] < -1).sum()
    n_dec_either = ((df["d_30_20"] < -1) | (df["d_40_30"] < -1)).sum()
    n_dec_total = (df["d_total"] < -1).sum()

    n_active = (df["pred_2020"] > 1).sum()
    print(f"  Total cells with predictions: {n}")
    print(f"  Cells with active baseline (pred_2020 > 1 m3): {n_active}")
    print(f"  Cells decreasing 2020->2030: {n_dec_30_20} ({100*n_dec_30_20/n:.1f}%)")
    print(f"  Cells decreasing 2030->2040: {n_dec_40_30} ({100*n_dec_40_30/n:.1f}%)")
    print(f"  Cells decreasing in EITHER step: {n_dec_either} ({100*n_dec_either/n:.1f}%)")
    print(f"  Cells with NET decrease 2020->2040: {n_dec_total} ({100*n_dec_total/n:.1f}%)")

    # Bin: how big are the decreases?
    print("\n  Distribution of d_total (m3) across cells:")
    print(df["d_total"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2))

    # Where ARE the decreases concentrated? Among the active or inactive cells?
    df["active"] = df["pred_2020"] > 1
    print("\n  Decreasing cells split by active/inactive baseline:")
    print(df.groupby(["active"])["d_total"].agg(["count", "mean", "min", "max"]).round(2))

    # For decreasing cells, what input features changed?
    # Load inputs for 2020 and 2040
    in20 = load_input(storm, 2020)
    in40 = load_input(storm, 2040)
    common = sorted(set(in20.columns) & set(in40.columns) - {"FID", "geometry"})
    common = [c for c in common if c in FREE_FEATS + MONO_INC_FEATS or c in
              ("dist_to_coast_m", "mean_bldg_size_m2", "ME", "WaveD", "WindD")]

    df_in = in20[["FID"] + common].merge(in40[["FID"] + common], on="FID",
                                         suffixes=("_20", "_40"))
    for c in common:
        df_in[f"d_{c}"] = df_in[f"{c}_40"] - df_in[f"{c}_20"]

    df_full = df.merge(df_in, on="FID", how="inner")

    dec = df_full[(df_full["d_total"] < -10) & (df_full["pred_2020"] > 1)]
    inc = df_full[(df_full["d_total"] > 10) & (df_full["pred_2020"] > 1)]
    print(f"\n  Working sets: {len(dec)} decreasing cells (d_total < -10 m3, active baseline), "
          f"{len(inc)} increasing cells (d_total > +10 m3, active baseline)")

    if len(dec) == 0:
        print("  No active-baseline decreasing cells found.")
        return df_full

    # For each feature, compare the MEAN signed change in decreasing vs increasing cells.
    print("\n  Mean signed input-feature change (2040 - 2020), decreasing vs increasing cells:")
    rows = []
    for c in common:
        col = f"d_{c}"
        m_dec = dec[col].mean()
        m_inc = inc[col].mean()
        rows.append({"feature": c, "mean_change_decreasing": round(m_dec, 4),
                     "mean_change_increasing": round(m_inc, 4),
                     "diff": round(m_dec - m_inc, 4),
                     "branch": ("FREE" if c in FREE_FEATS else
                                ("MONO_INC" if c in MONO_INC_FEATS else "MONO_DEC/OTHER"))})
    feat_df = pd.DataFrame(rows).sort_values("diff", key=lambda s: s.abs(), ascending=False)
    print(feat_df.to_string(index=False))

    # Save outputs
    out_csv_cells = OUT_DIR / f"{storm}_per_cell_deltas.csv"
    df_full[["FID", "pred_2020", "pred_2030", "pred_2040",
             "d_30_20", "d_40_30", "d_total"] + [f"d_{c}" for c in common]].to_csv(
        out_csv_cells, index=False)
    out_csv_feats = OUT_DIR / f"{storm}_feature_drivers.csv"
    feat_df.to_csv(out_csv_feats, index=False)
    print(f"\n  Wrote: {out_csv_cells} and {out_csv_feats}")

    return df_full


def main():
    summary_rows = []
    for storm in STORMS:
        df = analyse_storm(storm)
        n = len(df)
        n_dec = (df["d_total"] < -1).sum()
        n_dec_active = ((df["d_total"] < -10) & (df["pred_2020"] > 1)).sum()
        net_change_dec_cells = df.loc[df["d_total"] < -1, "d_total"].sum()
        net_change_inc_cells = df.loc[df["d_total"] > 1, "d_total"].sum()
        summary_rows.append({
            "storm": storm,
            "n_cells": n,
            "n_decreasing": int(n_dec),
            "pct_decreasing": round(100 * n_dec / n, 1),
            "n_decreasing_active": int(n_dec_active),
            "net_volume_decrease_m3": round(net_change_dec_cells, 0),
            "net_volume_increase_m3": round(net_change_inc_cells, 0),
            "net_balance_m3": round(net_change_inc_cells + net_change_dec_cells, 0),
        })

    summary = pd.DataFrame(summary_rows)
    print("\n\n===== SUMMARY ACROSS STORMS =====")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "summary_across_storms.csv", index=False)
    print(f"\nWrote summary: {OUT_DIR / 'summary_across_storms.csv'}")


if __name__ == "__main__":
    main()
