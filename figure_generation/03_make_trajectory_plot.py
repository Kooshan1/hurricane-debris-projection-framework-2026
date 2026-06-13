"""
M-F2: Three-point time-horizon trajectory plots (merged two-panel figure).

Panel (a): Total predicted debris volume across the county (m^3).
Panel (b): County-wide mean CLR with IQR uncertainty band.

Professional format: STIX Two Text, 7" width (double-column), 600 DPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DISPLAY_YEARS, DPI, FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE,
                    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
                    HURRICANE_COLORS, HURRICANE_LABELS, HURRICANES,
                    MC_DIR, OUTPUTS_ROOT, REVISION_FIG_ROOT,
                    YEAR_DISPLAY_TO_FILE, ensure_font, save_figure)

OUT_DIR = REVISION_FIG_ROOT / "M-F2_trajectory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NN_RESIDUAL_STD_PER_CELL = np.sqrt(130000.0)
Z_IQR = 0.6745


def _total_debris_per_scenario():
    rows = []
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv = (OUTPUTS_ROOT / "final_debris_volume_output" / storm
                   / str(file_year) / "debris_volume_predictions.csv")
            if not csv.is_file():
                shp = csv.with_suffix(".shp")
                if shp.is_file():
                    g = gpd.read_file(shp)
                    total = float(np.nansum(g["pred_m3"].astype(float).values))
                    n_cells = int((g["pred_m3"].astype(float) > 0).sum())
                else:
                    continue
            else:
                df = pd.read_csv(csv)
                total = float(np.nansum(df["pred_m3"].astype(float).values))
                n_cells = int((df["pred_m3"].astype(float) > 0).sum())
            ci_half = Z_IQR * NN_RESIDUAL_STD_PER_CELL * np.sqrt(max(n_cells, 1))
            rows.append({"storm": storm, "year": disp_year, "total": total,
                         "ci_half": ci_half})
    return pd.DataFrame(rows)


def _mean_clr_with_band():
    rows = []
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv = MC_DIR / f"result_summary_{storm}_{file_year}_seeds0-999.csv"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            sample_cols = df.columns[1:]
            arr = df[sample_cols].to_numpy(dtype=float)
            sample_means = np.nanmean(arr, axis=0)
            mean_v = float(np.nanmean(sample_means))
            lo_v = float(np.nanpercentile(sample_means, 25))
            hi_v = float(np.nanpercentile(sample_means, 75))
            rows.append({"storm": storm, "year": disp_year,
                         "mean": mean_v, "lo": lo_v, "hi": hi_v})
    return pd.DataFrame(rows)


def _plot_merged(volume_df, clr_df, out_path: Path):
    """Merged two-panel figure: full page width (7"), side by side."""
    ensure_font()
    fig, (ax_vol, ax_clr) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_DOUBLE, 2.5),
        gridspec_kw={"wspace": 0.28},
    )
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.30)

    # --- Panel (a): Volume ---
    ax_vol.set_facecolor("none")
    for storm in HURRICANES:
        sub = volume_df[volume_df.storm == storm].sort_values("year")
        if sub.empty:
            continue
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        ax_vol.plot(sub["year"], sub["total"] / 1e6, marker="o", markersize=4.5,
                    color=color, linewidth=1.5, label=label,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax_vol.set_xticks(list(DISPLAY_YEARS))
    ax_vol.set_xlabel("Year", fontsize=FONT_SIZE_NORMAL)
    ax_vol.set_ylabel(r"Total debris volume ($\times 10^{6}$ m$^{3}$)",
                      fontsize=FONT_SIZE_NORMAL)
    ax_vol.grid(axis="y", linestyle=":", alpha=0.35)
    ax_vol.spines[["top", "right"]].set_visible(False)
    ax_vol.tick_params(labelsize=FONT_SIZE_NORMAL)
    ax_vol.text(0.02, 0.95, "(a)", transform=ax_vol.transAxes,
                fontsize=FONT_SIZE_LARGE, fontweight="bold", va="top")

    # --- Panel (b): CLR ---
    ax_clr.set_facecolor("none")
    for storm in HURRICANES:
        sub = clr_df[clr_df.storm == storm].sort_values("year")
        if sub.empty:
            continue
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        ax_clr.fill_between(sub["year"], sub["lo"], sub["hi"],
                            color=color, alpha=0.16, linewidth=0, zorder=1)
        ax_clr.plot(sub["year"], sub["mean"], marker="o", markersize=4.5,
                    color=color, linewidth=1.5, label=label,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax_clr.set_xticks(list(DISPLAY_YEARS))
    ax_clr.set_xlabel("Year", fontsize=FONT_SIZE_NORMAL)
    ax_clr.set_ylabel("County-wide mean CLR", fontsize=FONT_SIZE_NORMAL)
    ax_clr.grid(axis="y", linestyle=":", alpha=0.35)
    ax_clr.spines[["top", "right"]].set_visible(False)
    ax_clr.tick_params(labelsize=FONT_SIZE_NORMAL)
    ax_clr.text(0.02, 0.95, "(b)", transform=ax_clr.transAxes,
                fontsize=FONT_SIZE_LARGE, fontweight="bold", va="top")

    # --- Shared legend with uncertainty band indicator ---
    legend_handles = []
    for storm in HURRICANES:
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        line = plt.Line2D([0], [0], color=color, linewidth=1.2, marker="o",
                          markersize=4, label=label)
        legend_handles.append(line)
    band_patch = mpatches.Patch(facecolor="#888888", alpha=0.18, edgecolor="none",
                                label="25th–75th percentile")
    legend_handles.append(band_patch)

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False,
               fontsize=FONT_SIZE_SMALL, bbox_to_anchor=(0.5, 0.02))

    save_figure(fig, out_path, pad_inches=0.03)


def _plot_single_volume(volume_df, out_path: Path):
    """Single-column volume panel for flexibility."""
    ensure_font()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, 2.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for storm in HURRICANES:
        sub = volume_df[volume_df.storm == storm].sort_values("year")
        if sub.empty:
            continue
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        ax.plot(sub["year"], sub["total"] / 1e6, marker="o", markersize=4,
                color=color, linewidth=1.2, label=label)
    ax.set_xticks(list(DISPLAY_YEARS))
    ax.set_xlabel("Year", fontsize=FONT_SIZE_NORMAL)
    ax.set_ylabel(r"Total debris volume ($\times 10^{6}$ m$^{3}$)",
                  fontsize=FONT_SIZE_NORMAL)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_NORMAL)

    legend_handles = []
    for storm in HURRICANES:
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.2,
                                         marker="o", markersize=4, label=label))
    ax.legend(handles=legend_handles, framealpha=0.0, loc="upper left",
              fontsize=FONT_SIZE_SMALL)
    fig.tight_layout()
    save_figure(fig, out_path, pad_inches=0.03)


def _plot_single_clr(clr_df, out_path: Path):
    """Single-column CLR panel for flexibility."""
    ensure_font()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE, 2.4))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for storm in HURRICANES:
        sub = clr_df[clr_df.storm == storm].sort_values("year")
        if sub.empty:
            continue
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        ax.plot(sub["year"], sub["mean"], marker="o", markersize=4,
                color=color, linewidth=1.2, label=label)
        ax.fill_between(sub["year"], sub["lo"], sub["hi"],
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xticks(list(DISPLAY_YEARS))
    ax.set_xlabel("Year", fontsize=FONT_SIZE_NORMAL)
    ax.set_ylabel("County-wide mean CLR", fontsize=FONT_SIZE_NORMAL)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_NORMAL)

    legend_handles = []
    for storm in HURRICANES:
        label = HURRICANE_LABELS[storm]
        color = HURRICANE_COLORS[label]
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.2,
                                         marker="o", markersize=4, label=label))
    legend_handles.append(mpatches.Patch(facecolor="#888888", alpha=0.18,
                                         edgecolor="none", label="25th–75th pctl"))
    ax.legend(handles=legend_handles, framealpha=0.0, loc="upper left",
              fontsize=FONT_SIZE_SMALL)
    fig.tight_layout()
    save_figure(fig, out_path, pad_inches=0.03)


def main():
    print("Computing per-scenario aggregates ...")
    vol = _total_debris_per_scenario()
    clr = _mean_clr_with_band()

    vol.to_csv(OUT_DIR / "trajectory_total_debris_volume_data.csv", index=False)
    clr.to_csv(OUT_DIR / "trajectory_mean_clr_data.csv", index=False)

    print("Rendering merged two-panel M-F2 ...")
    _plot_merged(vol, clr, OUT_DIR / "M-F2_trajectory_merged.png")

    print("Rendering individual panels ...")
    _plot_single_volume(vol, OUT_DIR / "trajectory_total_debris_volume.png")
    _plot_single_clr(clr, OUT_DIR / "trajectory_mean_clr.png")

    print(f"\nM-F2 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
