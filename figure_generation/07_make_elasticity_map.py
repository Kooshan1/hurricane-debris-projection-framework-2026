"""
S-F4: Per-tract sensitivity (CLR vs surge) map.

sensitivity_tract = (CLR_2040 - CLR_2020) / (SD_2040 - SD_2020)

Tracts where |Delta SD| < 0.02 m are shown in light gray ("no surge change"
at this location) rather than left transparent.

Professional format: STIX Two Text, map transform applied, 600 DPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DPI, FIG_WIDTH_SINGLE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
                    HURRICANES, HURRICANE_LABELS, MC_DIR_700,
                    OUTPUTS_ROOT, REVISION_FIG_ROOT, YEAR_DISPLAY_TO_FILE,
                    MidpointNormalize, boundary_extent, ensure_font,
                    load_county_gdf, load_tract_gdf, make_merged_map_panel,
                    register_custom_cmaps, save_horizontal_legend)

OUT_DIR = REVISION_FIG_ROOT / "S-F8_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_SD_MIN = 0.02
SENS_VLIM = 1.0


def _tract_mean_clr(storm: str, file_year: int, tract_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    csv = MC_DIR_700 / f"result_summary_{storm}_{file_year}_v7d_seeds0-999.csv"
    df = pd.read_csv(csv)
    df = df.rename(columns={df.columns[0]: "FID"})
    df["FID"] = df["FID"].astype(np.int64)
    sample_cols = df.columns[1:]
    df["mean_clr"] = df[sample_cols].astype(float).mean(axis=1)
    return df[["FID", "mean_clr"]]


def _tract_mean_surge(storm: str, file_year: int, tract_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    shp = (OUTPUTS_ROOT / "final_debris_volume_output" / storm / str(file_year)
           / "V14_physics_v7d_predictions.shp")
    grid = gpd.read_file(shp)
    if str(grid.crs).lower() != str(tract_gdf.crs).lower():
        grid = grid.to_crs(tract_gdf.crs)
    if not grid.geometry.is_valid.all():
        grid["geometry"] = grid.geometry.buffer(0)
    joined = gpd.sjoin(grid[["SD", grid.geometry.name]],
                       tract_gdf[["FID", tract_gdf.geometry.name]],
                       how="inner", predicate="intersects")
    fid_col = "FID" if "FID" in joined.columns else "FID_right"
    agg = joined.groupby(fid_col)["SD"].mean().reset_index()
    agg = agg.rename(columns={fid_col: "FID"})
    agg["FID"] = agg["FID"].astype(np.int64)
    return agg.rename(columns={"SD": "mean_surge_m"})


def _render_sensitivity_map(gdf, *, value_col, out_path, tract_gdf, county_gdf):
    """Render sensitivity map with gray fill for tracts without surge change."""
    register_custom_cmaps()
    ensure_font()

    figsize = (6, 6)

    cmap = plt.get_cmap("change_rdbu_9_r")
    norm = MidpointNormalize(vmin=-SENS_VLIM, vmax=SENS_VLIM, midpoint=0.0)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # First: plot ALL tracts as light gray background (no-data fill)
    tract_gdf.plot(ax=ax, facecolor="#e0e0e0", edgecolor="black",
                   linewidth=0.15, zorder=0)

    # Then: overlay tracts that HAVE valid sensitivity
    valid = gdf[gdf[value_col].notna()]
    if not valid.empty:
        valid.plot(column=value_col, cmap=cmap, norm=norm,
                   linewidth=0.2, edgecolor="black", ax=ax,
                   legend=False, zorder=1)

    # Set extent from tract boundary
    minx, miny, maxx, maxy = boundary_extent(tract_gdf)
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if county_gdf is not None:
        county_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                        linewidth=0.2, zorder=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                transparent=True, facecolor="none", edgecolor="none",
                pad_inches=0.01)
    plt.close(fig)


def main():
    register_custom_cmaps()
    print("Loading tract / county boundaries ...")
    tract_gdf = load_tract_gdf()
    county_gdf = load_county_gdf()

    n_empty_total = 0
    for storm in HURRICANES:
        baseline_year = YEAR_DISPLAY_TO_FILE[2020]
        future_year = YEAR_DISPLAY_TO_FILE[2040]

        print(f"\n--- {storm}: 2020 -> 2040 ---")
        clr_2020 = _tract_mean_clr(storm, baseline_year, tract_gdf)
        clr_2040 = _tract_mean_clr(storm, future_year, tract_gdf)
        sd_2020 = _tract_mean_surge(storm, baseline_year, tract_gdf)
        sd_2040 = _tract_mean_surge(storm, future_year, tract_gdf)

        merged = (clr_2020.rename(columns={"mean_clr": "clr_2020"})
                  .merge(clr_2040.rename(columns={"mean_clr": "clr_2040"}), on="FID")
                  .merge(sd_2020.rename(columns={"mean_surge_m": "sd_2020"}), on="FID")
                  .merge(sd_2040.rename(columns={"mean_surge_m": "sd_2040"}), on="FID"))
        merged["delta_clr"] = merged["clr_2040"] - merged["clr_2020"]
        merged["delta_sd"] = merged["sd_2040"] - merged["sd_2020"]
        merged["sensitivity"] = np.where(np.abs(merged["delta_sd"]) >= DELTA_SD_MIN,
                                          merged["delta_clr"] / merged["delta_sd"],
                                          np.nan)

        n_empty = int(merged["sensitivity"].isna().sum())
        n_total = len(merged)
        n_empty_total += n_empty
        print(f"  {n_empty}/{n_total} tracts have |delta_SD| < {DELTA_SD_MIN} m "
              f"(shown as gray on map)")

        merged.to_csv(OUT_DIR / f"sensitivity_data_{storm}.csv", index=False)

        gdf = (tract_gdf[["FID", tract_gdf.geometry.name]]
               .merge(merged[["FID", "sensitivity"]], on="FID", how="left")
               .rename(columns={"sensitivity": "value"}))

        out_map = OUT_DIR / "maps" / f"sensitivity_clr_per_msurge_{storm}_2020_2040.png"
        _render_sensitivity_map(
            gdf, value_col="value", out_path=out_map,
            tract_gdf=tract_gdf, county_gdf=county_gdf,
        )
        print(f"  wrote {out_map.name}")

    # Shared legend
    cmap = plt.get_cmap("change_rdbu_9_r")
    norm = MidpointNormalize(vmin=-SENS_VLIM, vmax=SENS_VLIM, midpoint=0.0)
    leg_path = OUT_DIR / "legends" / "sensitivity_clr_per_msurge_legend.png"
    save_horizontal_legend(cmap, norm, leg_path,
                           label=r"$\Delta$CLR / $\Delta$Surge (1/m)")
    print(f"\n  wrote shared legend {leg_path.name}")
    print(f"  Total empty (gray) tracts across all storms: {n_empty_total}")

    # Merged panel: all 3 storms side by side
    png_paths = [OUT_DIR / "maps" / f"sensitivity_clr_per_msurge_{s}_2020_2040.png"
                 for s in HURRICANES]
    labels = [f"{HURRICANE_LABELS[s]}" for s in HURRICANES]
    merged_path = OUT_DIR / "merged" / "S-F8_sensitivity_merged.png"
    make_merged_map_panel(png_paths, labels, merged_path,
                          ncols=3, legend_path=leg_path)
    print(f"\nS-F4 outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
