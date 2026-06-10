"""
Step 2: Uncertainty maps (CLR, road-closure) for supplementary materials.

All CLR maps go to supplementary only (M-F1 removed from main paper — the
baseline CLR maps are already in the original paper's Fig. 6 and the v7d
regenerated maps in generated_maps_v7d/).

Layout:
    S-F2_supplementary/         <- Individual maps for all metrics × scenarios
    S-F2_supplementary/merged/  <- Grouped merged panels with map transform applied
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _style import (DPI, DISPLAY_YEARS, HURRICANES, HURRICANE_LABELS,
                    MC_DIR_700, REVISION_FIG_ROOT, YEAR_DISPLAY_TO_FILE,
                    load_county_gdf, load_network_gdf, load_tract_gdf,
                    make_merged_map_panel, register_custom_cmaps,
                    render_choropleth_map, save_horizontal_legend,
                    MidpointNormalize)
import matplotlib.colors as mcolors

DERIVED_DIR = MC_DIR_700 / "derived_uncertainty"

OUT_SF2 = REVISION_FIG_ROOT / "S-F2_supplementary"
OUT_MERGED = OUT_SF2 / "merged"


# ---------------------------------------------------------------------------
# Year-to-year contrast for the CLR metrics (S-F2..S-F5).
#
# The CLR mean / std / exceedance fields are heavily right-skewed: ~75-80% of
# tracts sit at <= 0.05 while a handful of coastal tracts reach ~0.96. A linear
# 0..1 (or 0..0.5) scale therefore squashes almost every populated tract into a
# narrow band of the ramp, so the genuine but small decadal changes
# (2020 -> 2030 -> 2040) are nearly invisible. To make those differences
# perceptible WITHOUT changing the (restored) colormap and WITHOUT breaking
# cross-panel comparability, every panel of a given metric shares ONE norm:
#   * vmax capped at the pooled ~97.5th percentile across all 9 scenarios
#     (instead of the global max ~0.96, which washes everything out), and
#   * a mild PowerNorm (gamma 0.6) that expands the populated low range.
# The identical norm is used for all 9 maps and for the shared legend.
CLR_NORM_GAMMA = 0.6
CLR_VMAX_PCTL = 97.5


def _pooled_clr_vmax(value_col: str, pctl: float = CLR_VMAX_PCTL) -> float:
    """Pooled high-percentile of a CLR metric across all storms x years."""
    vals = []
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            csv = DERIVED_DIR / f"clr_metrics_{storm}_{file_year}.csv"
            if csv.is_file():
                vals.append(pd.read_csv(csv)[value_col].dropna().values)
    if not vals:
        return 1.0
    return float(np.percentile(np.concatenate(vals), pctl))


def _clr_norm(value_col: str):
    """Shared PowerNorm for one CLR metric (same across all 9 panels)."""
    vmax = _pooled_clr_vmax(value_col)
    # Guard against a degenerate all-zero column.
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return mcolors.PowerNorm(gamma=CLR_NORM_GAMMA, vmin=0.0, vmax=vmax)


def _write_clr_map(tract_gdf, county_gdf, csv_path, value_col, scen_tag,
                   out_dir, prefix, cmap, vmin, vmax, midpoint=None,
                   legend_label="", norm=None):
    df = pd.read_csv(csv_path)[["FID", value_col]].rename(columns={value_col: "value"})
    df["FID"] = df["FID"].astype(np.int64)
    gdf = tract_gdf[["FID", tract_gdf.geometry.name]].merge(df, on="FID", how="left")
    out_map = out_dir / "maps" / f"{prefix}_{scen_tag}.png"
    render_choropleth_map(
        gdf, value_col="value", cmap_name=cmap,
        vmin=vmin, vmax=vmax, midpoint=midpoint,
        out_path=out_map, legend_path=None,
        legend_label=legend_label,
        county_gdf=county_gdf, tract_gdf=tract_gdf,
        norm=norm,
    )


def _write_network_map(network_gdf, county_gdf, tract_gdf, csv_path, value_col,
                       scen_tag, out_dir, prefix, cmap, vmin, vmax, midpoint=None,
                       legend_label=""):
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns and df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})
    if len(df) != len(network_gdf):
        n = min(len(df), len(network_gdf))
        df = df.iloc[:n].reset_index(drop=True)
        gdf = network_gdf.iloc[:n].reset_index(drop=True).copy()
    else:
        gdf = network_gdf.copy()
    gdf["value"] = df[value_col].astype(float).values
    out_map = out_dir / "maps" / f"{prefix}_{scen_tag}.png"
    render_choropleth_map(
        gdf, value_col="value", cmap_name=cmap,
        vmin=vmin, vmax=vmax, midpoint=midpoint,
        out_path=out_map, legend_path=None,
        legend_label=legend_label,
        county_gdf=county_gdf, tract_gdf=tract_gdf,
        is_line_geometry=True, line_width=0.4,
    )


def _write_legend_only(out_legend_path, cmap_name, vmin, vmax, midpoint, label,
                       norm=None):
    register_custom_cmaps()
    cmap = plt.get_cmap(cmap_name)
    if norm is None:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint) if midpoint is not None \
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
    save_horizontal_legend(cmap, norm, out_legend_path, label=label)


def _collect_merged_panel(map_dir: Path, prefix: str, storms, years,
                          out_path: Path, legend_path: Path | None = None,
                          ncols: int = 3):
    """Collect map PNGs and delegate to the shared make_merged_map_panel."""
    png_paths = []
    labels = []
    for storm in storms:
        for year in years:
            scen_tag = f"{storm}_{year}"
            png_paths.append(map_dir / f"{prefix}_{scen_tag}.png")
            labels.append(f"{HURRICANE_LABELS[storm]} {year}")
    make_merged_map_panel(png_paths, labels, out_path,
                          ncols=ncols, legend_path=legend_path)


def main():
    register_custom_cmaps()
    print("Loading boundaries / network ...")
    tract_gdf = load_tract_gdf()
    county_gdf = load_county_gdf()
    network_gdf = load_network_gdf()

    # Fixed scales for honest comparison
    clr_mean_vmin, clr_mean_vmax = 0.0, 1.0
    clr_std_vmin, clr_std_vmax = 0.0, 0.5
    clr_pexceed_vmin, clr_pexceed_vmax = 0.0, 1.0
    net_mean_vmin, net_mean_vmax = 0.0, 1.0
    net_std_vmin, net_std_vmax = 0.0, 0.5
    net_pexceed_vmin, net_pexceed_vmax = 0.0, 1.0

    # Shared per-metric PowerNorms for the CLR maps (one norm per metric, used
    # for every storm/year panel AND the legend) so the 9 panels stay directly
    # comparable while the skewed low range gets more of the colour ramp.
    clr_mean_norm = _clr_norm("mean")
    clr_std_norm = _clr_norm("std")
    clr_pex030_norm = _clr_norm("p_exceed_0.30")
    clr_pex050_norm = _clr_norm("p_exceed_0.50")
    print(f"  CLR norms (gamma={CLR_NORM_GAMMA}, vmax=p{CLR_VMAX_PCTL}): "
          f"mean->{clr_mean_norm.vmax:.3f}, std->{clr_std_norm.vmax:.3f}, "
          f"p030->{clr_pex030_norm.vmax:.3f}, p050->{clr_pex050_norm.vmax:.3f}")

    # --- Generate all individual maps ---
    print("\n--- Generating CLR maps ---")
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            scen_tag = f"{storm}_{disp_year}"
            csv = DERIVED_DIR / f"clr_metrics_{storm}_{file_year}.csv"
            if not csv.is_file():
                print(f"  [skip] {csv.name}")
                continue
            print(f"  {scen_tag}")

            # All CLR maps go to supplementary. Each metric uses its shared
            # PowerNorm (same across all 9 panels) to lift the skewed low range.
            _write_clr_map(tract_gdf, county_gdf, csv, "mean", scen_tag,
                           OUT_SF2, "CLR_mean", "baseline_reds_clr",
                           clr_mean_vmin, clr_mean_vmax,
                           legend_label="Mean CLR", norm=clr_mean_norm)
            _write_clr_map(tract_gdf, county_gdf, csv, "std", scen_tag,
                           OUT_SF2, "CLR_std", "uncertainty_purples",
                           clr_std_vmin, clr_std_vmax,
                           legend_label="Std. dev. of CLR", norm=clr_std_norm)
            _write_clr_map(tract_gdf, county_gdf, csv, "p_exceed_0.30", scen_tag,
                           OUT_SF2, "CLR_pexceed_030", "baseline_orrd_network",
                           clr_pexceed_vmin, clr_pexceed_vmax,
                           legend_label="P(CLR > 0.30)", norm=clr_pex030_norm)
            _write_clr_map(tract_gdf, county_gdf, csv, "p_exceed_0.50", scen_tag,
                           OUT_SF2, "CLR_pexceed_050", "baseline_orrd_network",
                           clr_pexceed_vmin, clr_pexceed_vmax,
                           legend_label="P(CLR > 0.50)", norm=clr_pex050_norm)

    print("\n--- Generating road-closure maps (S-F2) ---")
    for storm in HURRICANES:
        for disp_year in DISPLAY_YEARS:
            file_year = YEAR_DISPLAY_TO_FILE[disp_year]
            scen_tag = f"{storm}_{disp_year}"
            csv = DERIVED_DIR / f"network_metrics_{storm}_{file_year}.csv"
            if not csv.is_file():
                print(f"  [skip] {csv.name}")
                continue
            print(f"  {scen_tag}")
            _write_network_map(network_gdf, county_gdf, tract_gdf, csv, "mean",
                               scen_tag, OUT_SF2, "road_closure_mean",
                               "baseline_orrd_network",
                               net_mean_vmin, net_mean_vmax,
                               legend_label="Mean road-closure prob.")
            _write_network_map(network_gdf, county_gdf, tract_gdf, csv, "std",
                               scen_tag, OUT_SF2, "road_closure_std",
                               "uncertainty_purples",
                               net_std_vmin, net_std_vmax,
                               legend_label="Std. dev. of road-closure prob.")
            _write_network_map(network_gdf, county_gdf, tract_gdf, csv,
                               "p_exceed_0.50", scen_tag, OUT_SF2,
                               "road_closure_pexceed_050",
                               "baseline_orrd_network",
                               net_pexceed_vmin, net_pexceed_vmax,
                               legend_label="P(closure prob. > 0.50)")

    # --- Legends ---
    print("\n--- Saving shared legends ---")
    legend_specs = [
        (OUT_SF2 / "legends" / "CLR_mean_legend.png", "baseline_reds_clr",
            clr_mean_vmin, clr_mean_vmax, None, "Mean CLR", clr_mean_norm),
        (OUT_SF2 / "legends" / "CLR_std_legend.png", "uncertainty_purples",
            clr_std_vmin, clr_std_vmax, None, "Std. dev. of CLR", clr_std_norm),
        (OUT_SF2 / "legends" / "CLR_pexceed_030_legend.png", "baseline_orrd_network",
            clr_pexceed_vmin, clr_pexceed_vmax, None, "P(CLR > 0.30)", clr_pex030_norm),
        (OUT_SF2 / "legends" / "CLR_pexceed_050_legend.png", "baseline_orrd_network",
            clr_pexceed_vmin, clr_pexceed_vmax, None, "P(CLR > 0.50)", clr_pex050_norm),
        (OUT_SF2 / "legends" / "road_closure_mean_legend.png", "baseline_orrd_network",
            net_mean_vmin, net_mean_vmax, None, "Mean road-closure prob.", None),
        (OUT_SF2 / "legends" / "road_closure_std_legend.png", "uncertainty_purples",
            net_std_vmin, net_std_vmax, None, "Std. dev. of road-closure prob.", None),
        (OUT_SF2 / "legends" / "road_closure_pexceed_050_legend.png", "baseline_orrd_network",
            net_pexceed_vmin, net_pexceed_vmax, None, "P(closure prob. > 0.50)", None),
    ]
    for path, cmap_name, vmin, vmax, midpoint, label, norm in legend_specs:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_legend_only(path, cmap_name, vmin, vmax, midpoint, label, norm=norm)
        print(f"  {path.name}")

    # --- Merged grouped panels (with map transform applied) ---
    print("\n--- Creating merged panels ---")
    OUT_MERGED.mkdir(parents=True, exist_ok=True)
    storms = list(HURRICANES)
    years = list(DISPLAY_YEARS)

    merge_specs = [
        ("CLR_mean", OUT_SF2 / "legends" / "CLR_mean_legend.png"),
        ("CLR_std", OUT_SF2 / "legends" / "CLR_std_legend.png"),
        ("CLR_pexceed_030", OUT_SF2 / "legends" / "CLR_pexceed_030_legend.png"),
        ("CLR_pexceed_050", OUT_SF2 / "legends" / "CLR_pexceed_050_legend.png"),
    ]
    for prefix, leg_path in merge_specs:
        _collect_merged_panel(
            OUT_SF2 / "maps", prefix, storms, years,
            OUT_MERGED / f"merged_{prefix}_all_scenarios.png",
            legend_path=leg_path, ncols=3,
        )

    print(f"\nAll maps saved under {REVISION_FIG_ROOT}")


if __name__ == "__main__":
    main()
