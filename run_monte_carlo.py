"""
Monte Carlo runner — CACHED-NETWORK version.

The original driver re-reads the road shapefile and re-runs all-source shortest
paths to fire stations for EVERY sample. Those operations are deterministic and
sample-invariant, so we run them ONCE and pickle the result, then each sample
loads the pickle and runs only the genuinely per-sample work.

The science is unchanged: the per-sample RNG sequence is identical (same
np.random.RandomState(seed) created fresh per sample, same order of
.randint / .uniform / .lognormal draws), and the SAME sample function calls
batch_extract_raster, generate_random_values, perform_scenaio_analysis,
estimate_CL_ratio, summarize_results_at_polygons, identify_flooded_roads in
the SAME order. Only the (deterministic, sample-invariant) network setup is
hoisted out of the per-sample loop.

Usage:
    python run_monte_carlo.py \
        --config config/config_monte_carlo.yaml \
        --scenario fema36 --year 2020 \
        --start-seed 0 --n-samples 1000 \
        --n-parallel 20
"""
from __future__ import annotations
import argparse
import copy
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.DebrisDispersionModel import debris_dispersion_model
from utils.SmartGeoProcess import GeoDataPoints
from utils.SmartNetworkAnalysis import NetworkAnalysis

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config_monte_carlo.yaml")
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--year", type=str, required=True)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-parallel", type=int, default=20)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--label", type=str, default="",
                   help="optional extra tag inserted into the result file names")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase A — sample-INVARIANT setup, run ONCE in main process and pickled.
# ---------------------------------------------------------------------------

def setup_network_once(config: dict, cache_path: Path) -> Path:
    """Build the NetworkAnalysis state that is identical for every sample,
    pickle it to disk, and return the cache path. This includes:

      - reading the road shapefile (NetworkAnalysis.fit())
      - reading critical-facility shapefiles + nearest-network-node mapping
      - all-source shortest-path 'Initial_dist_<facility>' (the slowest step)

    None of this depends on a sample seed or the debris field, so caching is
    safe (the per-sample RNG sequence is unchanged).
    """
    if cache_path.exists():
        print(f"Cache already exists at {cache_path}, reusing.")
        return cache_path

    print("Building network cache (one-time setup)...")
    crs = config["paths"]["crs"]
    network = NetworkAnalysis(path_network=config["paths"]["path_network"], crs=crs)
    network.fit()

    list_of_critical_facilities = config["paths"]["list_of_critical_facilities"]
    critical_facilities = {}
    for key in list_of_critical_facilities.keys():
        critical_facilities[key] = GeoDataPoints(
            path_geodata=list_of_critical_facilities[key], crs=crs)
        critical_facilities[key].read_geodata()
        critical_facilities[key].gdf = network.get_the_nearest_node_for_points(
            gdf=critical_facilities[key].gdf, label_facility=key)

    network.get_initial_distance()  # populates node_results['Initial_dist_<facility>']

    # Pickle the whole NetworkAnalysis instance. It's ~few hundred MB at most;
    # we trade disk I/O for hours of repeated CPU work.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Network cache written to {cache_path} ({cache_path.stat().st_size/1e6:.1f} MB)")
    return cache_path


# ---------------------------------------------------------------------------
# Phase B — per-sample work. Loads cache, deepcopies, runs identical steps to
# the original driver, and returns the per-sample columns.
# ---------------------------------------------------------------------------

def run_one_sample_cached(num_seed: int, config: dict, scenario: str, year: str,
                          random_fields_folder: str, cache_path: str):
    seed_simulation = np.random.RandomState(num_seed)

    # 1. Dispersion (uses the FRESH RNG -- 1st draw is .randint inside the model)
    grid_shapefile_path = config["paths"]["grid_shapefile_path"].format(
        hurricane_key_word=scenario, hurricane_year=year)
    debris_dispersion_model(
        grid_shapefile_path=grid_shapefile_path,
        building_data_path=config["paths"]["building_data_path"],
        grid_size=config["hyperparameters"]["grid_size"],
        num_seed=num_seed,
        seed_simulation=seed_simulation,
        num_inner_grids=config["hyperparameters"]["num_inner_grids_debris"],
        raster_output_path=random_fields_folder,
    )

    # 2. Load cached network setup and DEEPCOPY (we'll mutate per-sample)
    with open(cache_path, "rb") as f:
        cached_network = pickle.load(f)
    network = copy.deepcopy(cached_network)

    # 3. Extract debris-depth raster stats per road link (same as original)
    raster_path = f"{random_fields_folder}/output_merged_random_field_{num_seed}.tif"
    network.batch_extract_raster(
        list_of_raster_path=raster_path,
        buffer=config["hyperparameters"]["buffer"],
        roadway_inf_consideration=config["hyperparameters"]["roadway_inf_consideration"],
        fragility_type=config["hyperparameters"]["fragility_type"])
    if os.path.exists(raster_path):
        os.remove(raster_path)

    # 4. Random uniforms per road link (preserves RNG sequence even in 'debris' mode)
    network.generate_random_values(seed_simulation=seed_simulation)

    # 5. Ground clearance threshold (3rd RNG draw)
    ground_clearance_height = seed_simulation.lognormal(
        2.688439844526, 0.1980422004354) / 100

    # 6. Scenario analysis (Dijkstra per node x facility)
    network.perform_scenaio_analysis(
        link_removal_threshold=ground_clearance_height,
        analysis_type=config["hyperparameters"]["analysis_type"],
        seed_simulation=seed_simulation)
    network.estimate_CL_ratio()
    network.summarize_results_at_polygons(path_polygon=config["paths"]["path_polygon"])
    network.identify_flooded_roads(
        wading_height=ground_clearance_height,
        encode_no_data_as=0,
        analysis_type=config["hyperparameters"]["analysis_type"])

    # 7. Extract the per-sample output columns
    road_closure = pd.DataFrame(network.GeoData.gdf["road_closure"]).rename(
        columns={"road_closure": f"sample_{num_seed}"})
    cl_col = f"CL_fire_stations_max_output_merged_random_field_{num_seed}_mean"
    cl_summary = pd.DataFrame(network.summary_gdf[cl_col]).fillna(0).rename(
        columns={cl_col: f"sample_{num_seed}"})

    return road_closure, cl_summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["hyperparameters"]["hurricane_key_word"] = args.scenario
    config["hyperparameters"]["hurricane_year"] = args.year

    output_results_folder = args.output_dir or config["paths"]["output_results_folder"]
    Path(output_results_folder).mkdir(parents=True, exist_ok=True)

    random_fields_folder = (Path(config["paths"]["output_random_fields_folder"]) /
                            f"{args.scenario}_{args.year}_seeds{args.start_seed}")
    random_fields_folder.mkdir(parents=True, exist_ok=True)

    # Network cache is per-(roads,facilities,polygon) triple — these are the
    # same across ALL 9 scenarios so we use a single cache file.
    cache_dir = Path(output_results_folder).parent / "cache"
    cache_path = cache_dir / "network_setup.pkl"
    setup_network_once(config, cache_path)

    # Optional label segment (kept empty for the published file naming:
    # result_<kind>_<scenario>_<year>_seeds<a>-<b>.csv)
    label_part = f"{args.label}_" if args.label else ""
    network_csv = (Path(output_results_folder) /
                   f"result_network_{args.scenario}_{args.year}_{label_part}"
                   f"seeds{args.start_seed}-{args.start_seed + args.n_samples - 1}.csv")
    summary_csv = (Path(output_results_folder) /
                   f"result_summary_{args.scenario}_{args.year}_{label_part}"
                   f"seeds{args.start_seed}-{args.start_seed + args.n_samples - 1}.csv")

    seeds = list(range(args.start_seed, args.start_seed + args.n_samples))
    print(f"Scenario: {args.scenario}/{args.year}, seeds {seeds[0]}..{seeds[-1]} "
          f"(n={len(seeds)}), parallel workers: {args.n_parallel}")
    print(f"Network cache: {cache_path}")
    print(f"Network CSV  -> {network_csv}")
    print(f"Summary CSV  -> {summary_csv}")

    ts = time.time()
    out = Parallel(n_jobs=args.n_parallel, verbose=10)(
        delayed(run_one_sample_cached)(
            seed, config, args.scenario, args.year,
            str(random_fields_folder), str(cache_path))
        for seed in seeds)

    network_df = pd.concat([o[0] for o in out], axis=1)
    summary_df = pd.concat([o[1] for o in out], axis=1)
    network_df.to_csv(network_csv)
    summary_df.to_csv(summary_csv)

    elapsed = time.time() - ts
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(seeds):.2f}s/sample)")
    print(f"Wrote {network_csv} ({network_df.shape})")
    print(f"Wrote {summary_csv} ({summary_df.shape})")

    # Cleanup tmp folder
    try:
        if random_fields_folder.exists() and not any(random_fields_folder.iterdir()):
            random_fields_folder.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
