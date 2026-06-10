import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import utils.intensity_measure_processing as imp
import yaml
import numpy as np
from scipy.interpolate import griddata
from osgeo import gdal, osr
import os
from shapely import wkt

class HurricaneIntensityEvaluator:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.hurricane_key_word = self.config['hurricane_key_word']
        self.hurricane_year = self.config['hurricane_year']
        self.folder_path = f"{self.config['folder_path']}"
        self.results_file = f"{self.hurricane_key_word}/{self.hurricane_year}/{self.config['results_file']}"
        self.points_file = self.config['points_file']
        self.csv_file_path = self.config['csv_file_path'].format(hurricane_key_word=self.hurricane_key_word,
                                                                 hurricane_year=self.hurricane_year)
        self.grid_shapefile = self.config['grid_shapefile']
        self.output_shapefile_intensity = self.config['output_shapefile_intensity'].format(
            hurricane_key_word=self.hurricane_key_word, hurricane_year=self.hurricane_year)
        self.output_shapefile_final = self.config['output_shapefile_final'].format(
            hurricane_key_word=self.hurricane_key_word, hurricane_year=self.hurricane_year)
        self.building_damage_shapefile = self.config['building_damage_shapefile'].format(
            hurricane_key_word=self.hurricane_key_word, hurricane_year=self.hurricane_year)
        self.hazard_raster_folder = self.config['hazard_raster_folder'].format(
            hurricane_key_word=self.hurricane_key_word, hurricane_year=self.hurricane_year)
        self.x_res = self.config['x_res']
        self.y_res = self.config['y_res']
        self.merged_csv_path = self.config['merged_csv_path']
        self.additional_csv_path = self.config['additional_csv_path'].format(hurricane_key_word=self.hurricane_key_word,
                                                                             hurricane_year=self.hurricane_year)
        self.final_csv_path = self.config['final_csv_path'].format(hurricane_key_word=self.hurricane_key_word,
                                                                   hurricane_year=self.hurricane_year)
        self.final_shp_path = self.config['final_shp_path'].format(hurricane_key_word=self.hurricane_key_word,
                                                                   hurricane_year=self.hurricane_year)

    def load_datasets(self):
        self.swan_DIR = imp.load_data(self.folder_path, self.results_file, 'swan_DIR')
        self.swan_HS_max = imp.load_data(self.folder_path, self.results_file, 'swan_HS_max')
        self.zeta_max = imp.load_data(self.folder_path, self.results_file, 'zeta_max')
        self.zeta = imp.load_data(self.folder_path, self.results_file, 'zeta')
        self.u_vel = imp.load_data(self.folder_path, self.results_file, 'u-vel')
        self.v_vel = imp.load_data(self.folder_path, self.results_file, 'v-vel')
        self.windx = imp.load_data(self.folder_path, self.results_file, 'windx')
        self.windy = imp.load_data(self.folder_path, self.results_file, 'windy')
        self.depth = imp.load_data(self.folder_path, self.points_file, 'depth')
        self.latitude = imp.load_data(self.folder_path, self.points_file, 'latitude')
        self.longitude = imp.load_data(self.folder_path, self.points_file, 'longitude')
        self.node = imp.load_data(self.folder_path, self.points_file, 'node')

        print(f"All data parameters have been successfully loaded for hurricane {self.hurricane_key_word}.")

    def evaluate_intensity_measures(self):
        self.max_surge_depth = imp.calculate_max_surge_depth(self.zeta_max, self.depth)
        self.max_surge_level = imp.calculate_max_surge_level(self.zeta_max)
        self.wave_direction = imp.get_wave_direction(self.swan_DIR)
        self.max_wave_height = imp.get_wave_height_at_max_surge(self.swan_HS_max)
        self.max_wave_velocity_x = imp.get_max_wave_velocity_x(self.u_vel)
        self.max_wave_velocity_y = imp.get_max_wave_velocity_y(self.v_vel)
        self.max_wave_velocity = imp.get_max_wave_velocity(self.u_vel, self.v_vel)
        self.wind_dir = imp.calculate_wind_direction(self.windx, self.windy)
        self.wind_steadiness = imp.calculate_wind_steadiness(self.windx, self.windy)
        self.max_wind_velocity = imp.get_max_wind_velocity(self.windx, self.windy)
        self.momentum_flux = imp.calculate_momentum_flux(self.zeta, self.depth, self.u_vel, self.v_vel)
        self.inundation_duration = imp.calculate_inundation_duration(self.zeta, self.depth, 0.5)

        print(f"All intensity measures have been successfully loaded for hurricane {self.hurricane_key_word}.")

    def convert_points_to_polygons(self):
        self.grid_gdf = gpd.read_file(self.grid_shapefile)
        points_df = pd.DataFrame({
            'latitude': self.latitude,
            'longitude': self.longitude,
            'max_surge_depth': self.max_surge_depth,
            'max_surge_level': self.max_surge_level,
            'wave_direction': self.wave_direction,
            'max_wave_height': self.max_wave_height,
            'max_wave_velocity_x': self.max_wave_velocity_x,
            'max_wave_velocity_y': self.max_wave_velocity_y,
            'max_wave_velocity': self.max_wave_velocity,
            'wind_direction': self.wind_dir,
            'wind_steadiness': self.wind_steadiness,
            'max_wind_velocity': self.max_wind_velocity,
            'momentum_flux': self.momentum_flux,
            'inundation_duration': self.inundation_duration
        })
        os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
        points_df.to_csv(self.csv_file_path, index=False)
        geometry = [Point(xy) for xy in zip(points_df.longitude, points_df.latitude)]
        points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry)
        points_gdf.crs = self.grid_gdf.crs
        joined_gdf = gpd.sjoin(self.grid_gdf, points_gdf, how="left", predicate="contains")
        no_points = joined_gdf[joined_gdf['index_right'].isna()]
        if not no_points.empty:
            nearest_points_info = points_gdf.sindex.nearest(no_points.geometry, return_distance=False)
            nearest_points_indices = [info[0] for info in nearest_points_info]
            if len(no_points.index) == len(nearest_points_indices):
                nearest_points_mapping = pd.DataFrame({
                    'original_index': no_points.index.tolist(),
                    'nearest_point_index': nearest_points_indices
                })
                points_gdf_reset = points_gdf.reset_index()
                nearest_points_gdf = nearest_points_mapping.merge(points_gdf_reset, left_on='nearest_point_index',
                                                                  right_index=True)
                nearest_points_gdf.set_index('original_index', inplace=True)
                joined_gdf = pd.concat(
                    [joined_gdf.drop(index=no_points.index), nearest_points_gdf.drop(columns=['nearest_point_index'])])
        self.joined_gdf = joined_gdf

        print(f"All parameters have been processed before aggregating for hurricane {self.hurricane_key_word}.")

    def aggregate_intensity_measures(self):
        operation_for_columns = {
            'max_surge_depth': 'max',
            'max_surge_level': 'max',
            'wave_direction': 'mean',
            'max_wave_height': 'max',
            'max_wave_velocity_x': 'max',
            'max_wave_velocity_y': 'max',
            'max_wave_velocity': 'max',
            'wind_steadiness': 'mean',
            'wind_direction': 'mean',
            'max_wind_velocity': 'max',
            'momentum_flux': 'max',
            'inundation_duration': 'max'
        }
        columns_abbreviations = {
            'max_surge_depth': 'SD',
            'max_surge_level': 'SL',
            'wave_direction': 'WaveD',
            'max_wave_height': 'WH',
            'max_wave_velocity_x': 'WV_x',
            'max_wave_velocity_y': 'WV_y',
            'max_wave_velocity': 'WV',
            'wind_steadiness': 'WS',
            'wind_direction': 'WindD',
            'max_wind_velocity': 'WindV',
            'momentum_flux': 'MF',
            'inundation_duration': 'InunD'
        }
        for original_column, operation in operation_for_columns.items():
            if operation == 'max':
                aggregated_values = self.joined_gdf.groupby(self.joined_gdf.index).agg({original_column: 'max'})
            elif operation == 'mean':
                aggregated_values = self.joined_gdf.groupby(self.joined_gdf.index).agg({original_column: 'mean'})
            else:
                raise ValueError(f"Invalid operation for {original_column}. Choose 'max' or 'mean'.")
            abbreviation = columns_abbreviations[original_column]
            self.grid_gdf[f"{abbreviation}_{operation}"] = self.grid_gdf.index.map(aggregated_values[original_column])
        os.makedirs(os.path.dirname(self.output_shapefile_intensity), exist_ok=True)
        self.grid_gdf.to_file(self.output_shapefile_intensity)

        print(f"All parameters have been aggregated to polygon grids for hurricane {self.hurricane_key_word}.")

    def calculate_failure_probabilities(self):
        pf_gdf = gpd.read_file(self.building_damage_shapefile)
        joined_gdf = gpd.sjoin(self.grid_gdf, pf_gdf, how="left", predicate="contains")
        aggregated_data = joined_gdf.groupby(joined_gdf.index).apply(
            lambda g: pd.Series({
                'WeightPf_1': (g['sq_foot'] * g['PF']).sum() / 100,
                'WeightPf_2': (g['sq_foot'] * g['PF'] * g['no_stories']).sum() / 100
            })
        )
        self.grid_gdf = self.grid_gdf.join(aggregated_data, how="left")
        os.makedirs(os.path.dirname(self.output_shapefile_final), exist_ok=True)
        self.grid_gdf.to_file(self.output_shapefile_final)

        print(f"Failure probabilities have been added to input dataset for hurricane {self.hurricane_key_word}.")

    def interpolate_and_save_rasters(self):
        points_df = pd.read_csv(self.csv_file_path)
        gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude)
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        x_min, y_min, x_max, y_max = gdf.geometry.total_bounds
        x_grid = np.arange(x_min, x_max, self.x_res)
        y_grid = np.arange(y_min, y_max, self.y_res)
        y_grid = y_grid[::-1]
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        os.makedirs(self.hazard_raster_folder, exist_ok=True)

        for column in ['max_surge_depth', 'max_surge_level', 'wave_direction', 'max_wave_height', 'max_wave_velocity_x',
                       'max_wave_velocity_y', 'max_wave_velocity', 'wind_steadiness',
                       'wind_direction', 'max_wind_velocity', 'momentum_flux', 'inundation_duration']:
            z_interpolated = griddata(
                points=(gdf["longitude"].values, gdf["latitude"].values),
                values=gdf[column].values,
                xi=(x_mesh, y_mesh),
                method='linear'
            )
            output_raster_path = os.path.join(self.hazard_raster_folder, f"{column}.tif")
            rows, cols = z_interpolated.shape
            driver = gdal.GetDriverByName('GTiff')
            outRaster = driver.Create(output_raster_path, cols, rows, 1, gdal.GDT_Float32)
            outRaster.GetRasterBand(1).WriteArray(z_interpolated)
            outRaster.SetGeoTransform((x_min, self.x_res, 0, y_max, 0, -self.y_res))
            outRasterSRS = osr.SpatialReference()
            outRasterSRS.ImportFromEPSG(4326)
            outRaster.SetProjection(outRasterSRS.ExportToWkt())
            outRaster.FlushCache()
            outRaster.GetRasterBand(1).SetNoDataValue(np.nan)
            outRaster = None

        print(f"Raster files created and saved for hurricane {self.hurricane_key_word}.")

    def generate_final_inputs(self, consider_after_2008=True):
        shapefile_path = self.output_shapefile_final
        polygon_gdf = gpd.read_file(shapefile_path)
        polygon_gdf.crs = 'EPSG:4326'
        df = pd.read_csv(self.merged_csv_path)
        if not consider_after_2008:
            df = df[df['Year_built_Main_Area'] <= 2008]
        df['geometry'] = df['geometry'].apply(wkt.loads)
        point_gdf = gpd.GeoDataFrame(df, geometry='geometry')
        point_gdf.crs = 'EPSG:2278'
        point_gdf = point_gdf.to_crs('EPSG:4326')
        polygon_gdf = polygon_gdf.to_crs('EPSG:4326')
        joined_gdf = gpd.sjoin(point_gdf, polygon_gdf, how='inner', predicate='within')

        columns_to_sum = {
            'Sum_of_the_areas_of_Main_Areas': 'TBFA',
            'Count_of_accesory_structures': 'NAS',
            'Sum_of_the_area_of_the_accesory_structures': 'TAAS',
            'Count_Mobile_homes': 'NumMH'
        }

        missing_columns = [col for col in columns_to_sum.keys() if col not in joined_gdf.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in joined_gdf: {missing_columns}")

        for col in columns_to_sum.keys():
            joined_gdf[col] = joined_gdf[col].astype(float)

        joined_gdf.rename(columns=columns_to_sum, inplace=True)

        for abbreviation in columns_to_sum.values():
            polygon_gdf[abbreviation] = 0.0

        for index, polygon in polygon_gdf.iterrows():
            polygon_points = joined_gdf[joined_gdf.index_right == index]
            if not polygon_points.empty:
                polygon_gdf.at[index, 'NumB'] = len(polygon_points)
                for abbreviation in columns_to_sum.values():
                    polygon_gdf.at[index, abbreviation] = polygon_points[abbreviation].sum()
            else:
                polygon_gdf.at[index, 'NumB'] = 0

        additional_df = pd.read_csv(self.additional_csv_path)
        polygon_gdf['FID'] = polygon_gdf['FID'].astype(int)
        additional_df['fid_1'] = additional_df['fid_1'].astype(int)
        polygon_gdf = polygon_gdf.merge(additional_df, left_on='FID', right_on='fid_1', how='left')
        columns = ['FID', 'geometry'] + [c for c in polygon_gdf.columns if c not in ['FID', 'geometry']]
        polygon_gdf = polygon_gdf[columns]

        interested_columns = [
            'SD_max', 'WH_max', 'WaveD_mean', 'WV_x_max', 'WV_y_max', 'WindD_mean', 'WS_mean',
            'MF_max', 'NumB', 'TBFA', 'NAS', 'TAAS', 'NumMH', 'WeightPf_1', 'WeightPf_2', 'OW', 'DO', 'DM', 'DT', 'RD',
            'ME', 'ADS', 'UL', 'PD', 'NumH', 'MHI', 'NumHU', 'OHU', 'VHU', 'PR'
        ]

        new_names = [
            'SD', 'WH', 'WaveD', 'WV_X', 'WV_Y', 'WindD', 'WS', 'MF', 'NumB', 'TBFA', 'NAS', 'TAAS', 'NumMH',
            'WPF_1', 'WPF_2', 'OW', 'DO', 'DM', 'DT', 'RD', 'ME', 'ADS', 'UL', 'PD', 'NumH', 'MHI', 'NumHU',
            'OHU', 'VHU', 'PR'
        ]

        final_gdf = polygon_gdf[['FID', 'geometry'] + interested_columns]
        final_gdf.columns = ['FID', 'geometry'] + new_names

        os.makedirs(os.path.dirname(self.final_csv_path), exist_ok=True)
        final_gdf.to_csv(self.final_csv_path, index=False)

        os.makedirs(os.path.dirname(self.final_shp_path), exist_ok=True)
        final_gdf.to_file(self.final_shp_path)

        print("Final data processed and saved.")
