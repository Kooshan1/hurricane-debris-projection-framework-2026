"""
S-F7: hazard-to-impact amplification figure (compound-effect evidence for R2.5).

For each storm at years 2030 and 2040, compute % change relative to the 2020
baseline of three quantities:

    (1) county-mean of max surge depth (m)            — hazard
    (2) total predicted debris volume (m^3)           — intermediate
    (3) county-wide mean CLR                          — downstream impact

Cell (i,j) in the figure is the bar group for storm j at year i. Within each
group, three coloured bars show (1)/(2)/(3).

The empirical signature is that CLR % change > Volume % change > Surge %
change. Because CLR and volume are derived from the *same* coupled SLR × LCC
inputs, a super-additive cascade is direct evidence that the compound effect
is not captured by a simple sum of marginal hazard and land-cover effects.

Outputs (both PNG and PDF):
    S-F9_amplification/S-F9_amplification_merged.{png,pdf}
    S-F9_amplification/amplification_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DPI, FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE,
                    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
                    HURRICANE_LABELS, HURRICANES, MC_DIR_700, OUTPUTS_ROOT,
                    REVISION_FIG_ROOT, YEAR_DISPLAY_TO_FILE, ensure_font,
                    save_figure)

OUT_DIR = REVISION_FIG_ROOT / "S-F9_amplification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _county_mean_surge(storm: str, file_year: int) -> float:
    """Mean surge depth (m) over inundated cells, from V14_physics_v7d shp."""
    shp = (OUTPUTS_ROOT / "final_debris_volume_output" / storm
           / str(file_year) / "V14_physics_v7d_predictions.shp")
    if not shp.is_file():
        return float("nan")
    g = gpd.read_file(shp, columns=["SD"])
    sd = g["SD"].astype(float).values
    sd = sd[np.isfinite(sd) & (sd > 0)]
    return float(np.mean(sd)) if sd.size else float("nan")


def _total_debris(storm: str, file_year: int) -> float:
    """Total debris volume (m^3), from V14_physics_v7d (calibrated)."""
    csv = (OUTPUTS_ROOT / "final_debris_volume_output" / storm
           / str(file_year) / "V14_physics_v7d_predictions.csv")
    if csv.is_file():
        df = pd.read_csv(csv)
        return float(np.nansum(df["pred_m3"].astype(float).values))
    shp = csv.with_suffix(".shp")
    if shp.is_file():
        g = gpd.read_file(shp, columns=["pred_m3"])
        return float(np.nansum(g["pred_m3"].astype(float).values))
    return float("nan")


def _county_mean_clr(storm: str, file_year: int) -> float:
    """County-wide mean CLR across all MC samples and all tracts."""
    csv = MC_DIR_700 / f"result_summary_{storm}_{file_year}_v7d_seeds0-999.csv"
    if not csv.is_file():
        return float("nan")
    df = pd.read_csv(csv)
    arr = df.iloc[:, 1:].to_numpy(dtype=float)
    return float(np.nanmean(arr))


def build_amplification_table() -> pd.DataFrame:
    rows = []
    for storm in HURRICANES:
        baseline_year = YEAR_DISPLAY_TO_FILE[2020]
        b_surge = _county_mean_surge(storm, baseline_year)
        b_vol = _total_debris(storm, baseline_year)
        b_clr = _county_mean_clr(storm, baseline_year)
        for disp_year in (2030, 2040):
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            c_surge = _county_mean_surge(storm, file_year)
            c_vol = _total_debris(storm, file_year)
            c_clr = _county_mean_clr(storm, file_year)
            rows.append({
                "storm": storm, "year": disp_year,
                "pct_change_surge": 100.0 * (c_surge - b_surge) / b_surge if b_surge else np.nan,
                "pct_change_volume": 100.0 * (c_vol - b_vol) / b_vol if b_vol else np.nan,
                "pct_change_clr": 100.0 * (c_clr - b_clr) / b_clr if b_clr else np.nan,
                "baseline_surge_m": b_surge, "current_surge_m": c_surge,
                "baseline_volume_m3": b_vol, "current_volume_m3": c_vol,
                "baseline_clr": b_clr, "current_clr": c_clr,
            })
    return pd.DataFrame(rows)


def _plot_merged(df: pd.DataFrame, out_path: Path):
    """Single full-width panel: grouped bar chart over (storm, year) pairs.

    Layout: x-axis has 6 groups arranged as 3 major storm blocks (Ike,
    FEMA33, FEMA36), each block containing 2 sub-positions for the two
    horizons (2030, 2040). Each sub-position has 3 coloured bars (surge,
    volume, CLR % change). Storms are visually separated by extra
    horizontal padding and a hairline tick mark on the x-axis.
    """
    ensure_font()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.0))
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.22)
    ax.set_facecolor("none")

    metric_cols = [
        ("pct_change_surge", "Mean surge depth",   "#9ecae1"),
        ("pct_change_volume","Total debris volume","#fdae6b"),
        ("pct_change_clr",   "Mean CLR",           "#cb181d"),
    ]
    storms_in_order = list(HURRICANES)
    years_in_order = (2030, 2040)
    bar_width = 0.22

    # Position layout: storms separated by 1.5 bar-widths of gap, years
    # within a storm separated by 0.0 (sub-positions touching).
    pair_positions = []
    pair_labels = []
    minor_ticks = []   # for year sub-labels
    storm_centers = []
    x = 0.0
    storm_gap = bar_width * 4.0     # gap between storm blocks
    year_step = bar_width * 3.4     # step between 2030 and 2040 within a storm
    for storm in storms_in_order:
        storm_first_x = x
        for year in years_in_order:
            pair_positions.append(x)
            pair_labels.append(str(year))
            minor_ticks.append(x)
            x += year_step
        storm_last_x = x - year_step
        storm_centers.append(0.5 * (storm_first_x + storm_last_x))
        x = storm_last_x + storm_gap

    # Draw the three coloured bars at each (storm, year) position.
    for i, (col, _, color) in enumerate(metric_cols):
        offset = (i - 1) * bar_width
        vals = []
        for storm in storms_in_order:
            for year in years_in_order:
                sub = df[(df.storm == storm) & (df.year == year)]
                vals.append(float(sub[col].iloc[0]) if not sub.empty else np.nan)
        ax.bar(np.array(pair_positions) + offset, vals, width=bar_width,
               color=color, edgecolor="black", linewidth=0.3)

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xticks(minor_ticks)
    ax.set_xticklabels(pair_labels, fontsize=FONT_SIZE_SMALL)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("% change vs 2020 baseline", fontsize=FONT_SIZE_NORMAL)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_NORMAL)

    # Storm-name annotation BELOW the year tick labels.
    ymin, _ = ax.get_ylim()
    for sc, storm in zip(storm_centers, storms_in_order):
        ax.annotate(HURRICANE_LABELS[storm], xy=(sc, ymin),
                    xytext=(0, -16), textcoords="offset points",
                    ha="center", va="top", fontsize=FONT_SIZE_NORMAL,
                    fontweight="bold")

    # Legend below the chart.
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black",
                                    linewidth=0.3, label=lab)
                      for _, lab, c in metric_cols]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=FONT_SIZE_SMALL,
               bbox_to_anchor=(0.5, 0.0))

    save_figure(fig, out_path)


def main():
    print("Computing amplification table ...")
    df = build_amplification_table()
    csv_path = OUT_DIR / "amplification_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  data CSV: {csv_path}")
    print(df.to_string(index=False))

    out_path = OUT_DIR / "S-F9_amplification_merged.png"
    _plot_merged(df, out_path)
    print(f"\nS-F7 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
