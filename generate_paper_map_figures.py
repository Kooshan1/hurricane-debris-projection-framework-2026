"""
Regenerate the map-style paper figures from the Monte Carlo results and per-cell predictions.

Reads from:
  - debris_volume_model/monte_carlo_results/result_summary_*_seeds0-999.csv  (CLR)
  - debris_volume_model/monte_carlo_results/result_network_*_seeds0-999.csv  (NetworkCondition)
  - outputs/final_debris_volume_output/<storm>/<year>/debris_volume_predictions.shp  (pred_m3)
  - outputs/hazard_rasters/<storm>/<year>/<variable>.tif                              (max_surge_depth, identical)

Writes to:
  - outputs/figure/generated_maps/                                                 (all 36 maps + legends)
  - outputs/figure/generated_maps/config_plots_generated.yaml      (the config used)

Existing paper outputs at outputs/figure/generated_maps/ are NOT touched.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.PlotGenerator import PlotGenerator

# ---------------------------------------------------------------------------
# Build the config dict
# ---------------------------------------------------------------------------

OUTPUT_MAPS  = "./outputs/figure/generated_maps"
OUTPUT_LEGS  = "./outputs/figure/generated_maps/legends"

SCENARIOS = [
    ("ike", 2020), ("ike", 2030), ("ike", 2040),
    ("fema33", 2020), ("fema33", 2030), ("fema33", 2040),
    ("fema36", 2020), ("fema36", 2030), ("fema36", 2040),
]

GENERAL_SETTINGS = {
    "boundary_shapefile_path_tract": "../inputs/debris_impact_data/Tracts_Galveston_County_Without_Bolivar_Peninsula.shp",
    "boundary_shapefile_path_grid":  "../inputs/shapefiles/Grid_250_Without_Bolivar_Peninsula.shp",
    "boundary_shapefile_path_county":"../inputs/debris_impact_data/galveston_county_border.shp",
    "map_border": {"visible": True, "line_width": 0.2, "line_color": "black",
                    "line_style": "solid", "zorder": 10},
    "target_crs": "EPSG:3857",
    "output_directory_maps":   OUTPUT_MAPS,
    "output_directory_legends": OUTPUT_LEGS,
    "figure_defaults": {
        "dpi": 600, "transparent_background": True, "figsize": [6, 6],
        "axis_padding_factor": 0.05,
        "polygon_line_width": 0.2, "polygon_edge_color": "black", "polygon_linestyle": "solid",
        "network_line_width": 0.3,
        "grid_line_width": 0.05, "grid_edge_color": "grey", "grid_linestyle": "solid",
        "network_background_visible": True, "network_background_color": "#FFFFFF",
        "network_background_edge_color": "#000000",
        "network_background_line_width": 0.0, "network_background_linestyle": "dashed",
    },
    "font_defaults": {"family": "sans-serif", "label_size": 10, "tick_size": 8},
}

DATA_SOURCES = {
    "parameter_shapefile": {
        # per-cell debris-volume predictions
        "input_path_template": "./outputs/final_debris_volume_output/{hurricane_name}/{year}/debris_volume_predictions.shp",
        "aggregation_method": "mean",
        "variables": ["SD", "WH", "PD", "WPF_1", "pred_m3", "VHU", "NumHU"],
    },
    "hazard_raster": {
        "input_path_template": "./outputs/hazard_rasters/{hurricane_name}/{year}/{variable}.tif",
        "variables": ["max_surge_depth", "max_wave_height", "inundation_duration",
                      "max_wind_velocity", "max_wave_velocity", "momentum_flux"],
    },
    "clr": {
        # 1,000-sample Monte Carlo results
        "input_path_template": "./debris_volume_model/monte_carlo_results/result_summary_{hurricane_name}_{year}_seeds0-999.csv",
        "variables": ["CLR"],
    },
    "network_condition": {
        # Re-use the network geometry shapefile from the original run
        "network_shapefile_path": "./outputs/debris_impact_output/monte_carlo_result/Network_Condition_results.shp",
        "input_path_template": "./debris_volume_model/monte_carlo_results/result_network_{hurricane_name}_{year}_seeds0-999.csv",
        "variables": ["NetworkCondition"],
    },
}


def make_plot_def(plot_type: str, hurricane: str, year: int, mode: str,
                  variable: str = None, baseline_year: int = 2020,
                  output_level: str = None) -> dict:
    """Build a single plot definition matching original-paper styling."""
    common = {
        "plot_id": f"{plot_type}_{hurricane}_{year}_{mode}",
        "plot_type": plot_type,
        "scenario": {"hurricane_name": hurricane, "year": year},
        "map_mode": mode,
    }
    if mode in ("difference", "percent_change"):
        common["baseline_hurricane"] = hurricane
        common["baseline_year"] = baseline_year

    if plot_type == "parameter_shapefile":
        common["variable"] = variable or "pred_m3"
        common["output_level"] = output_level or "grid"
        if mode == "baseline":
            common["color_map"] = "baseline_oranges_debris"
            common["vmin"], common["vmax"] = 0, 5000
        else:  # difference
            common["color_map"] = "change_prgn_9_r"
            common["vmin"], common["vmax"], common["midpoint"] = -500, 500, 0
    elif plot_type == "hazard_raster":
        common["variable"] = variable or "max_surge_depth"
        if mode == "baseline":
            common["color_map"] = "baseline_blues_surge"
            common["vmin"], common["vmax"] = 0, 5
        else:  # percent_change
            common["color_map"] = "change_prgn_9_r"
            common["vmin"], common["vmax"], common["midpoint"] = -100, 100, 0
    elif plot_type == "clr":
        common["variable"] = "CLR"
        if mode == "baseline":
            common["color_map"] = "baseline_reds_clr"
            common["vmin"], common["vmax"] = 0, 1
            common["edge_color"] = "none"
        else:  # difference
            common["color_map"] = "change_prgn_9_r"
            common["vmin"], common["vmax"], common["midpoint"] = -0.1, 0.1, 0
            common["line_width"] = 0.1
    elif plot_type == "network_condition":
        common["variable"] = "NetworkCondition"
        if mode == "baseline":
            common["color_map"] = "baseline_orrd_network"
            common["vmin"], common["vmax"] = 0, 1
            common["line_width"] = 0.1
        else:  # difference
            common["color_map"] = "change_prgn_9_r"
            common["vmin"], common["vmax"], common["midpoint"] = -0.2, 0.2, 0
            common["line_width"] = 0.1
    return common


def build_plot_definitions() -> list:
    """Generate the full set of 36 plot definitions matching the original 36 figures.

    For each (plot_type) x (storm in {ike, fema33, fema36}):
      - 2020: baseline plot
      - 2030, 2040: difference (or percent_change for hazard) vs that storm's 2020

    Plot types covered: parameter_shapefile (pred_m3), hazard_raster (max_surge_depth),
                        clr, network_condition.
    """
    defs = []
    storms = ["ike", "fema33", "fema36"]
    years = [2020, 2030, 2040]

    plot_specs = [
        ("parameter_shapefile", "pred_m3", "difference"),
        ("hazard_raster",        "max_surge_depth", "percent_change"),
        ("clr",                   "CLR", "difference"),
        ("network_condition",     "NetworkCondition", "difference"),
    ]

    for plot_type, variable, future_mode in plot_specs:
        for storm in storms:
            # Baseline (2020)
            defs.append(make_plot_def(plot_type, storm, 2020, "baseline",
                                       variable=variable))
            # Future years (difference vs 2020)
            for year in [2030, 2040]:
                defs.append(make_plot_def(plot_type, storm, year, future_mode,
                                           variable=variable, baseline_year=2020))
    return defs


def main():
    config = {
        "general_settings": GENERAL_SETTINGS,
        "data_sources": DATA_SOURCES,
        "plot_definitions": build_plot_definitions(),
    }

    # Save the generated config for reproducibility
    out_config_dir = Path("outputs/figure/generated_maps")
    out_config_dir.mkdir(parents=True, exist_ok=True)
    out_config_path = out_config_dir / "config_plots_generated.yaml"
    with open(out_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote generated config: {out_config_path}")
    print(f"Total plot definitions: {len(config['plot_definitions'])}")

    # Run PlotGenerator
    print("\nLaunching PlotGenerator...")
    pg = PlotGenerator(config)
    pg.generate_all_maps()
    print("\nDone.")


if __name__ == "__main__":
    main()
