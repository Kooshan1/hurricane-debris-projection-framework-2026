"""
S-F1: Box plot of tract-level CLR distributions across all 9 scenarios.

For each (storm, year) we have ~100 tract-level CLR averages. Grouped by
year, colored by storm. Professional format: STIX Two Text, 7" width, 600 DPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DPI, DISPLAY_YEARS, FIG_WIDTH_DOUBLE, FONT_SIZE_LARGE,
                    FONT_SIZE_NORMAL, FONT_SIZE_SMALL, HURRICANE_COLORS,
                    HURRICANE_LABELS, HURRICANES, MC_DIR_700,
                    REVISION_FIG_ROOT, YEAR_DISPLAY_TO_FILE, ensure_font,
                    save_figure)

OUT_DIR = REVISION_FIG_ROOT / "S-F1_box_plot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_long_df():
    records = []
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv = MC_DIR_700 / f"result_summary_{storm}_{file_year}_v7d_seeds0-999.csv"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            sample_cols = df.columns[1:]
            tract_means = df[sample_cols].astype(float).mean(axis=1).values
            for v in tract_means:
                if pd.notna(v):
                    records.append({"Year": disp_year,
                                    "Hurricane": HURRICANE_LABELS[storm],
                                    "CLR": float(v)})
    return pd.DataFrame.from_records(records)


def _group_positions(years, hurricane_labels):
    dx, group_gap = 0.22, 1.2
    x_positions, group_centers = {}, {}
    for i, y in enumerate(years):
        base = i * group_gap
        offsets = np.linspace(-dx, +dx, len(hurricane_labels))
        for h, off in zip(hurricane_labels, offsets):
            x_positions[(y, h)] = base + off
        group_centers[y] = base
    return x_positions, group_centers


def plot_box(long_df, out_path: Path):
    ensure_font()
    storm_labels = list(HURRICANE_LABELS.values())
    x_pos, centers = _group_positions(list(DISPLAY_YEARS), storm_labels)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 2.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    positions, datas, colors = [], [], []
    for y in DISPLAY_YEARS:
        for h in storm_labels:
            sub = long_df[(long_df.Year == y) & (long_df.Hurricane == h)]["CLR"]
            sub = sub.clip(lower=1e-4, upper=1).values
            positions.append(x_pos[(y, h)])
            datas.append(sub)
            colors.append(HURRICANE_COLORS[h])

    bp = ax.boxplot(datas, positions=positions, widths=0.18, patch_artist=True,
                    manage_ticks=False, whis=1.5, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
        patch.set_edgecolor(c)
    for elem in ("whiskers", "caps", "medians"):
        for line in bp[elem]:
            line.set_color("#444444")
            line.set_alpha(0.8)
            line.set_linewidth(0.7)

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xticks([centers[y] for y in DISPLAY_YEARS])
    ax.set_xticklabels([str(y) for y in DISPLAY_YEARS], fontsize=FONT_SIZE_NORMAL)
    ax.set_ylabel("Tract-level connectivity loss ratio",
                  fontsize=FONT_SIZE_NORMAL)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_NORMAL)

    handles = [plt.Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=HURRICANE_COLORS[h],
                          markeredgecolor="none", markersize=7, label=h)
               for h in HURRICANE_LABELS.values()]
    ax.legend(handles=handles, loc="upper left", framealpha=0.0,
              fontsize=FONT_SIZE_SMALL, title="Hurricane",
              title_fontsize=FONT_SIZE_NORMAL)

    fig.tight_layout()
    save_figure(fig, out_path, pad_inches=0.03)


def main():
    long_df = _load_long_df()
    if long_df.empty:
        print("No CLR samples found; aborting.")
        return
    print(f"Loaded {len(long_df)} (year, hurricane, tract) records.")
    plot_box(long_df, OUT_DIR / "S-F1_clr_box_plot.png")
    print(f"\nS-F1 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
