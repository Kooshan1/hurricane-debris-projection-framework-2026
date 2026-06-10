import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
import gstools as gs
import os
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import warnings

# Suppress the specific gstools warning
warnings.filterwarnings("ignore", message="gstools.RandMeth: \\*\\*kwargs are ignored")

def generate_random_field(i, polygon, debris_volume, cond_pos, cond_val, num_inner_grids, seed_random_field, grid_size):
    if debris_volume == 0:
        # Return zero array without further computation
        res_x = (polygon.bounds[2] - polygon.bounds[0]) / num_inner_grids
        res_y = (polygon.bounds[3] - polygon.bounds[1]) / num_inner_grids
        transform = rasterio.transform.from_origin(polygon.bounds[0], polygon.bounds[3], res_x, res_y)
        Z_flipped = np.zeros((num_inner_grids, num_inner_grids), dtype=np.float32)
        return Z_flipped, transform

    # Generate random field as before
    model = gs.Gaussian(dim=2, var=1e10 * debris_volume + 1, len_scale=0.0002)
    krige = gs.Krige(model, cond_pos=cond_pos, cond_val=cond_val)
    srf = gs.CondSRF(krige, mean=debris_volume / num_inner_grids ** 2, seed=seed_random_field)
    x = np.linspace(polygon.bounds[0], polygon.bounds[2], num_inner_grids)
    y = np.linspace(polygon.bounds[1], polygon.bounds[3], num_inner_grids)
    srf.field = srf.structured([x, y])

    srf.field = np.maximum(srf.field, 0)
    total_field_sum = np.sum(srf.field)
    if total_field_sum > 0:
        srf.field *= debris_volume / total_field_sum
    else:
        srf.field = np.zeros_like(srf.field)

    srf.field /= (grid_size ** 2) / (num_inner_grids ** 2)
    Z = srf.field.T
    res_x = (polygon.bounds[2] - polygon.bounds[0]) / num_inner_grids
    res_y = (polygon.bounds[3] - polygon.bounds[1]) / num_inner_grids
    transform = rasterio.transform.from_origin(polygon.bounds[0], polygon.bounds[3], res_x, res_y)

    # Flip Z along axis=0 to match rasterio's orientation
    Z_flipped = np.flip(Z, axis=0)

    return Z_flipped, transform

def process_and_merge(arrays_and_transforms, final_output_file):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import from_origin

    # Collect bounds and pixel sizes
    all_bounds = []
    pixel_sizes_x = []
    pixel_sizes_y = []

    for array, transform in arrays_and_transforms:
        minx, miny, maxx, maxy = rasterio.transform.array_bounds(array.shape[0], array.shape[1], transform)
        all_bounds.append((minx, miny, maxx, maxy))
        pixel_sizes_x.append(transform.a)
        pixel_sizes_y.append(-transform.e)  # Negative because transform.e is negative

    # Compute overall bounds
    minx = min(bound[0] for bound in all_bounds)
    miny = min(bound[1] for bound in all_bounds)
    maxx = max(bound[2] for bound in all_bounds)
    maxy = max(bound[3] for bound in all_bounds)

    # Decide on resolution (smallest pixel size)
    res_x = min(pixel_sizes_x)
    res_y = min(pixel_sizes_y)

    # Compute dimensions
    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))

    # Create final transform
    transform = from_origin(minx, maxy, res_x, res_y)

    # Create final raster dataset
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': arrays_and_transforms[0][0].dtype,
        'crs': 'epsg:4326',
        'transform': transform,
    }

    with rasterio.open(final_output_file, 'w', **profile) as dst:
        # Disable the progress bar here to prevent overlapping
        for array, src_transform in tqdm(arrays_and_transforms, desc="Writing arrays to final raster", leave=False):
            # Get bounds of the array
            minx_array, miny_array, maxx_array, maxy_array = rasterio.transform.array_bounds(
                array.shape[0], array.shape[1], src_transform)
            # Compute the window in the destination raster
            window = rasterio.windows.from_bounds(
                minx_array, miny_array, maxx_array, maxy_array,
                transform=dst.transform,
                height=dst.height,
                width=dst.width,
                precision=6  # Adjust precision if necessary
            )
            # Adjust window offsets and lengths
            window = window.round_offsets().round_lengths()

            # Ensure window dimensions match array shape
            if (int(window.height) != array.shape[0]) or (int(window.width) != array.shape[1]):
                # Adjust window size to match array shape
                window = rasterio.windows.Window(
                    int(window.col_off),
                    int(window.row_off),
                    array.shape[1],
                    array.shape[0]
                )

            # Write the array data into the window
            dst.write(array, 1, window=window)

    print(f"Final merged output saved to {final_output_file}")

def debris_dispersion_model(grid_shapefile_path, building_data_path, grid_size=500, num_seed=1,
                            seed_simulation=None, num_inner_grids=50, raster_output_path=None):
    grid_shapefile = gpd.read_file(grid_shapefile_path)
    building_data = gpd.read_file(building_data_path)

    # Ensure CRS match
    if grid_shapefile.crs != building_data.crs:
        building_data = building_data.to_crs(grid_shapefile.crs)

    # Spatial join to associate buildings with grid cells
    grid_shapefile = grid_shapefile.reset_index().rename(columns={'index': 'grid_index'})
    building_data = building_data.reset_index().rename(columns={'index': 'building_index'})

    # Perform spatial join
    building_data_with_grid = gpd.sjoin(
        building_data,
        grid_shapefile[['grid_index', 'geometry']],
        how='left',
        predicate='within'
    )

    parameters = []
    if seed_simulation is None:
        seed_simulation = np.random.RandomState()
    seed_random_field = seed_simulation.randint(1, 1_000_000_000)
    grid_pred_m3 = grid_shapefile["pred_m3"].fillna(0).clip(lower=0)

    for i, (polygon, debris_volume) in enumerate(zip(grid_shapefile.geometry, grid_pred_m3)):
        if debris_volume == 0:
            # Skip processing this grid cell
            continue

        # Get buildings in this grid cell
        buildings_in_polygon = building_data_with_grid[building_data_with_grid['grid_index'] == i]
        if not buildings_in_polygon.empty:
            coords = np.array([(geom.x, geom.y) for geom in buildings_in_polygon.geometry])
            cond_pos = [coords[:, 0].tolist(), coords[:, 1].tolist()]
            cond_val = [2_000_000 * debris_volume / num_inner_grids ** 2] * len(cond_pos[0])
        else:
            cond_pos = [[], []]
            cond_val = []

        parameters.append((i, polygon, debris_volume, cond_pos, cond_val,
                           num_inner_grids, seed_random_field, grid_size))

    # Default to all cores; can be capped with env var DEBRIS_DISPERSION_N_JOBS
    # so an outer Monte-Carlo loop can run many samples in parallel without each
    # sample's dispersion model trying to grab every core on the machine.
    num_cores = int(os.environ.get("DEBRIS_DISPERSION_N_JOBS", "-1"))

    # Use joblib's progress bar support
    print(f"Processing Grids (n_jobs={num_cores}):")
    with tqdm_joblib(tqdm(desc="Processing Grids", total=len(parameters))):
        with parallel_backend("loky", inner_max_num_threads=1):
            arrays_and_transforms = Parallel(n_jobs=num_cores, batch_size='auto', verbose=0)(
                delayed(generate_random_field)(*param_set) for param_set in parameters
            )

    # Final merging with the optimized process_and_merge function
    final_output_file = os.path.join(raster_output_path, f"output_merged_random_field_{num_seed}.tif")
    process_and_merge(arrays_and_transforms, final_output_file)
