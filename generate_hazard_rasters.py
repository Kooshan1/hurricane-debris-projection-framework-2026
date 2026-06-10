import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import griddata
from osgeo import gdal, osr
import os

hurricane_key_word = 'ike'

# Define the path where we want to get the data from
csv_file_path = f"./outputs/intensity_measures/{hurricane_key_word}/intensity_measures_data.csv"

# Load the CSV file back into a DataFrame
points_df = pd.read_csv(csv_file_path)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude)
)

# If necessary, set or correct the CRS (assuming it should be EPSG:4326)
gdf = gdf.set_crs("EPSG:4326", allow_override=True)

# Define the resolution of your raster
# The resolution should be chosen based on the density of your points and the desired detail of the raster
x_res = 0.001  # Example resolution for longitude
y_res = 0.001  # Example resolution for latitude

# Create a grid to interpolate your data
x_min, y_min, x_max, y_max = gdf.geometry.total_bounds
x_grid = np.arange(x_min, x_max, x_res)
y_grid = np.arange(y_min, y_max, y_res)
y_grid = y_grid[::-1]  # Inverse for later raster transform # TODO: Check this! It seems everything is correct now!
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Folder to save the raster files
output_folder = f"./outputs/hazard_rasters/{hurricane_key_word}"

# # Manually define the affine transform
# transform = Affine.translation(x_min - x_res / 2, y_min + y_res / 2) * Affine.scale(x_res, y_res)

# Interpolate and save each measure as a raster
for column in ['max_surge_depth', 'max_surge_level','wave_direction', 'max_wave_height', 'max_wave_velocity_x',
               'max_wave_velocity_y', 'max_wave_velocity', 'wind_steadiness',
               'wind_direction', 'max_wind_velocity', 'momentum_flux', 'inundation_duration']:
    # Interpolate the data
    z_interpolated = griddata(
        points=(gdf["longitude"].values, gdf["latitude"].values),
        values=gdf[column].values,
        xi=(x_mesh, y_mesh),
        method='linear'
    )

    # # Adjusted raster creation with the new transform
    # output_raster_path = f"{output_folder}/{column}.tif"
    # with rasterio.open(
    #         output_raster_path, 'w',
    #         driver='GTiff',
    #         height=z_interpolated.shape[0],
    #         width=z_interpolated.shape[1],
    #         count=1,
    #         dtype=z_interpolated.dtype,
    #         crs='EPSG:4326',
    #         transform=transform
    # ) as dst:
    #     dst.write(z_interpolated, 1)

    # Adjusted raster creation with the new transform
    output_raster_path = os.path.join(output_folder, f"{column}.tif")

    # Create the raster
    rows, cols = z_interpolated.shape
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(output_raster_path, cols, rows, 1, gdal.GDT_Float32)

    # Write the array to the file
    outRaster.GetRasterBand(1).WriteArray(z_interpolated)

    # Set the geotransform and projection
    outRaster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))  # (x_min, x_res, 0, y_min, 0, y_res)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

    # Flush data to disk, set the NoData value and then close the file
    outRaster.FlushCache()
    outRaster.GetRasterBand(1).SetNoDataValue(np.nan)
    outRaster = None

print("Raster files created and saved.")


