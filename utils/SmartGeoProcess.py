import math
import os

# import warnings
# warnings.filterwarnings("ignore")

# Rasterio
import rasterio
from rasterio import Affine as A
import rasterstats as rs
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform
# Geodata
import geopandas as gpd
#plotting
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from loky import get_reusable_executor
import rioxarray as rxr
from shapely.ops import unary_union
from shapely.geometry import LineString
from shapely.geometry import Point
import utm


class GeoDataShapes:
    """Master class for geodata"""
    def __init__(self, path_geodata=None, crs='epsg:26915'): #Changed crs='epsg:26915'
        self.path_geodata=path_geodata # Main geodata shapefile
        self.crs = crs # Web maps are usually provided in web Mercator 'epsg:3857'

    def read_geodata(self, path_geodata=None, keep_a_copy=True):
        """Read a file to geopandas"""

        if not path_geodata: path_geodata=self.path_geodata
        # Read the file
        gdf = gpd.read_file(path_geodata)
        # Convert the path to appropriate crs
        if gdf.crs != self.crs:
            print(f"Shape crs not matching the study crs, attempting conversion from {gdf.crs} to {self.crs}")
            gdf = gdf.to_crs(self.crs)

        # Return the file
        # saving a copy of the file to self.gdf
        if keep_a_copy: self.gdf = gdf

        return gdf

    def drop_duplicate_geometry(self, gdf=None):
        """Remove any duplicate geometry"""
        if gdf is None: gdf = self.gdf

        G = gdf["geometry"].apply(lambda geom: geom.wkb)
        return gdf.loc[G.drop_duplicates().index]

    def view(self):
        """View the values of geopandas as panda dataframe"""
        df = pd.DataFrame(self.gdf)
        pd.options.display.max_columns = 50
        print(df)
        return df

    def plot_geodata(self, ax=None):
        """Plot the geodataframe"""
        # Make sure that shape is not empty
        if self.gdf.shape[0]:
            self.gdf.plot(ax=ax)
            plt.show()

    def save_geodata(self, polygon=None, filename="GeoPolygon", format="ESRI Shapefile", path_to_save="./outputs/debris_impact_output/monte_carlo_result", file_extension=".shp"):
        print("Writing files to the disk...")
        if polygon is None: polygon = self.gdf

        # Convert tuple columns to strings
        for col_label in list(polygon.columns):
            if isinstance(polygon[col_label][0],tuple):
                polygon[col_label] = polygon[col_label].astype(str)

        # Make folder if it does not exist
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)

        polygon.to_file(f"{path_to_save}/{filename}{file_extension}", encoding='utf-8', driver=format)
        print("Done!")

    def clip(self, mask=None, shape_to_clip=None, buffer_dist=0, update_geometry=False, use_convex_hull=True):
        """
        Given a input shape (from_shape), a filter gdb and a buffer dist, this function returns
        geometry from 'from_shape' which lies within the bounds on the filter shape + buffer
        Args:
            shape_to_clip ():
            mask ():
            buffer_dist ():

        Returns:
        """
        if shape_to_clip is None:
            shape_to_clip = self.gdf

        if mask is None:
            mask = self.gdf

        import shapely
        if use_convex_hull==False:
            bound = shapely.geometry.box(*mask.total_bounds).buffer(buffer_dist)
            bound = gpd.GeoDataFrame(gpd.GeoSeries(bound), columns=['geometry'])
            bound = bound.set_crs(shape_to_clip.crs)
        else:
            bound = gpd.GeoDataFrame(geometry=gpd.GeoSeries(mask['geometry'].unary_union.convex_hull))
            bound.geometry = bound.geometry.buffer(buffer_dist)
            bound = bound.set_crs(shape_to_clip.crs)

        if update_geometry:
            self.gdf = gpd.overlay(shape_to_clip, bound, how='intersection')
        else:
            return gpd.overlay(shape_to_clip, bound, how='intersection')



    # def get_raster_stat_at_polygons(self, raster, polygon=None, stats=['min', 'max', 'median', 'majority', 'sum'], affine=None):
    #     raise NotImplemented("This function is not implemented here")

    def get_raster_stat_at_polygons(self, raster, polygon=None, stats=['min', 'max', 'median', 'majority', 'sum'],
                                    affine=None):
        """Always pass the tiff file"""
        """Obtaining raster statistics at polygon"""
        """Raster stat can be used on tiff as well as rasterio object"""
        # make a copy not to overwrite the input file


        if polygon is None:
            _polygon = self.gdf.copy()
        else:
            #_polygon = polygon.copy()  # Original
            _polygon = polygon # Kooshan - for segmentation (polygon instead of geodataframe)
        if isinstance(raster, str):
            """In this case the data passed is the tiff file path"""
            result = pd.DataFrame(rs.zonal_stats(vectors=_polygon, raster=raster, stats=stats, all_touched=True)) #all_touched=True added by Kooshan
        elif isinstance(raster.flat[0], np.floating):
            """Now the file is a rasterio input"""
            result = pd.DataFrame(rs.zonal_stats(_polygon, raster, affine=affine, all_touched=True)) #all_touched=True added by Kooshan

        # rdf = gpd.GeoDataFrame(pd.concat([_polygon, result], axis=1), crs=self.crs)
        # # Make sure that the last column is geometry
        # rdf = self._make_geometry_the_last_column(rdf)

        return result

    def extract_raster_stat(self,raster, polygon=None, stats=['min', 'max', 'median', 'majority', 'sum'], affine=None, suffix=None):
        raise NotImplemented("This function is not implemented here")

    def _make_geometry_the_last_column(self, gdf):
        """Make sure that the last column in geometry"""
        cols = list(gdf)
        cols.remove("geometry")
        cols.append('geometry')
        return gdf.reindex(columns=cols)

    # Kooshan
    # def prep_polygons_asarr(gs):
    #     def get_pts(poly):
    #         if isinstance(poly, Polygon):
    #             coords = np.array(poly.exterior.coords)
    #         elif isinstance(poly, MultiPolygon):
    #             coords = np.concatenate([get_pts(sp) for sp in poly.geoms])
    #         return coords
    #
    #     return [get_pts(poly) for poly in gs]

    # Kooshan
    # def get_nearest_poly(pt, polys):
    #     pt = np.array(pt.coords)
    #     dists = np.array([np.abs(np.linalg.norm(poly - pt, axis=1)).min() for poly in polys])
    #     return dists.argmin()

    def summarize_points_at_polygons(self, points, polygons=None, summary_statistics=['mean', 'std', 'max', 'min'],path_save_file=None):
        """This code will summarize the results of point files at polygons"""
        # Get the polygon
        if polygons is None: polygons = self.gdf

        polygons["Key_Geom_poly"] = polygons['geometry'].apply(lambda x: str(x))
        # Perform spatial joint
        points_polys = gpd.sjoin(points, polygons, how="left")
        # Get all numerical columns
        num_columns = [col for col in list(points.columns) if str(points[col][0]).replace('.', '', 1).isdigit()]
        # Set numerical cals to numerical type
        for col in num_columns:
            points_polys[col] = pd.to_numeric(points_polys[col])

        stats_pt = points_polys.groupby('Key_Geom_poly')[num_columns].agg(['mean', 'std', 'max', 'min'])
        stats_pt.columns = ["_".join(x) for x in stats_pt.columns.ravel()]

        # Kooshan - for polygons that have not any points in them, find the nearest polygon and copy that
        # polys_points = gpd.sjoin(polygons, points, how="left")
        #
        # for i in len(polys_points):
        #     if math.isnan(polys_points.index_right[i]):
        #         polygon_temp = polys_points.Key_Geom_poly[i]
        #         polys = prep_polygons_asarr([p1, p2, p3])
        #         get_nearest_poly(pt, polys)

        result = pd.merge(polygons, stats_pt, left_on='Key_Geom_poly', right_index=True, how='outer')
        # Remove the key
        result.drop('Key_Geom_poly', inplace=True, axis=1)

        if path_save_file:
            result.to_file(path_save_file)

        return result

class GeoDataPoints(GeoDataShapes):
    """A class for geodatabase line elements"""
    def __init__(self, path_geodata=None, crs='epsg:26915'): #Changed  crs='epsg:26915'
        """Initiate the global class"""
        GeoDataShapes.__init__(self, path_geodata=path_geodata, crs=crs)


class GeoDataLines(GeoDataShapes):
    """A class for geodatabase line elements"""
    def __init__(self, path_geodata=None, crs='epsg:26915'): #Changed  crs='epsg:26915'
        """Initiate the global class"""
        GeoDataShapes.__init__(self, path_geodata=path_geodata, crs=crs)

    def batch_estimate_raster_stat(self, list_of_raster_path, buffer=None, attach_to_file=False, roadway_inf_consideration=False, fragility_type="both"):

        # If string is passed, convert the string into raster
        list_of_raster_path = list_of_raster_path if isinstance(list_of_raster_path, list) else [list_of_raster_path]

        # Create a partial function for passing buffer and making the process concurrent
        func_ = partial(self.extract_raster_stat, buffer=buffer, concurrent=True, roadway_inf_consideration=roadway_inf_consideration, fragility_type=fragility_type)

        with get_reusable_executor() as executor:
            results = list(tqdm(executor.map(func_, list_of_raster_path), total=len(list_of_raster_path), desc="Extracting raster at GeoDataLines"))

        combined_df = pd.concat(results, axis=1)

        # If attached is true
        if attach_to_file:
            # _combined = pd.concat([self.gdf, combined_df], axis=1)
            _combined = pd.concat([self.gdf.reset_index(drop=True), combined_df.reset_index(drop=True)], axis=1)
            self.gdf = _combined
            return _combined

        return combined_df

    #TODO Check the accuracy of the files; the resutls seems to be off for the flood raster from Toni
    def extract_raster_stat(self, raster, polygon=None, buffer=None, stats=['max'], affine=None, suffix=None, concurrent=False, roadway_inf_consideration=False, fragility_type="both"):

        """['min', 'max', 'median', 'majority', 'sum']"""
        # Get the stat
        if polygon is None: polygon = self.gdf
        # Make a copy
        _polygon = polygon.copy()
        # get zonal statistics

        if roadway_inf_consideration:
            # constant values
            segmentation_threshold = 300  # meters
            segmentation_length = 200  # meters

            pf_final = np.zeros(len(_polygon))
            # Getting shoreline shapefile to evaluate distances to shoreline
            gdf_shoreline = gpd.read_file("./input_files/hazard_fema36/CUSPLine_Modified.shp")
            raster_wave_height_path = "./input_files/hazard_fema36/fema36_max_wave_height.tif"
            raster_surge_height_path = "./input_files/hazard_fema36/500yrMaxFloodDepth_WGS84.tif"
            gdf_shoreline = gdf_shoreline.to_crs(gdf_shoreline.estimate_utm_crs())

            # Getting the minimum distance to shoreline using the below method
            def min_distance(point, lines):
                return lines.distance(point).min()

            # Function for bridge fragility
            def bridge_fail_prob(Clear, Mass, Elev, Surge, Wave):

                pf_bridge_temp = 0.0;

                if Mass < 5:
                    a = 0.6468
                    b = 0.0406
                    c = -0.1376;
                elif Mass >= 5 and Mass < 10:
                    a = 0.4166
                    b = 0.0456
                    c = -0.2343
                elif Mass >= 10 and Mass < 15:
                    a = 0.3291
                    b = 0.0546
                    c = -0.2464
                elif Mass >= 15 and Mass < 20:
                    a = 0.33
                    b = 0.0576
                    c = -0.2444
                elif Mass >= 20 and Mass < 25:
                    a = 0.2843
                    b = 0.0512
                    c = -0.2421
                elif Mass >= 25 and Mass < 30:
                    a = 0.2865
                    b = 0.0881
                    c = -0.2391
                elif Mass >= 30 and Mass < 35:
                    a = 0.1870
                    b = 0.0782
                    c = -0.2618

                pf_bridge_temp = np.minimum(1, max(0, a + b * Wave + c * (Clear - Surge + Elev)))
                return pf_bridge_temp

            for i in range(0, len(_polygon)):
                # Redefining the road segments to be approximately 200 meters
                if fragility_type == "roadway" or fragility_type == "both":
                    pf_roadway = 1
                    #  _polygon.geometry[0].xy[0][0]
                    #  _polygon["length"][0]
                    # Redefine linestrings to get segments with proper length
                    if _polygon["length"][i] > segmentation_threshold:
                        num_nodes = np.rint(_polygon["length"][i] / segmentation_length) + 1
                        distances = np.linspace(0.0, _polygon.geometry[i].length, int(num_nodes))
                        points = [_polygon.geometry[i].interpolate(distance) for distance in distances]
                        multipoint = LineString(points)
                    elif _polygon["length"][i] < segmentation_length:
                        num_nodes = 2
                        distances = np.linspace(0.0, _polygon.geometry[i].length, int(num_nodes))
                        points = [_polygon.geometry[i].interpolate(distance) for distance in distances]
                        multipoint = LineString(points)
                    else:
                        multipoint = _polygon.geometry[i]

                    # Get each segment from redefined linestring
                    line_segments = list(map(LineString, zip(multipoint.coords[:-1], multipoint.coords[1:])))

                    for j in range(0, len(line_segments)):
                        if buffer is not None:
                            # Add buffer to each segment
                            polygon_segment = line_segments[j].buffer(buffer)
                        df_segment_stats = self.get_raster_stat_at_polygons(raster, polygon_segment, stats, affine)

                        polygon_centroid = utm.from_latlon(polygon_segment.centroid.y, polygon_segment.centroid.x)
                        polygon_centroid = Point(polygon_centroid[0], polygon_centroid[1])
                        min_dist_shore = min_distance(polygon_centroid, gdf_shoreline)
                        # Evaluating probability of failure using fragility model
                        pf = 1 / (1 + np.exp(-9.5877 + 7.2302 * np.log(min_dist_shore) - 7.1078 * np.log(df_segment_stats.values[0][0])))
                        pf_roadway *= (1 - pf)
                    pf_roadway = 1 - pf_roadway
                else:
                    pf_roadway = 0

                if fragility_type == "bridge" or fragility_type == "both":
                    if _polygon["bridge_inp"][i] == "yes":
                        if buffer is not None:
                            polygon_bridge = _polygon.geometry[i].buffer(buffer)
                        df_bridge_stats_wave_height = self.get_raster_stat_at_polygons(raster_wave_height_path, polygon_bridge, stats, affine)
                        df_bridge_stats_surge_height = self.get_raster_stat_at_polygons(raster_surge_height_path, polygon_bridge, stats, affine)
                        pf_bridge = bridge_fail_prob(_polygon["clearance"][i],_polygon["span_mass"][i],_polygon["g_elev"][i],df_bridge_stats_surge_height[stats[0]][0],df_bridge_stats_wave_height[stats[0]][0])
                    else:
                        pf_bridge = 0

                pf_final[i] = np.maximum(pf_roadway, pf_bridge)

            df_zonal_stats = pd.DataFrame(data = {'max': pf_final})
        else:
            if buffer is not None:
                # Add buffer
                _polygon = gpd.GeoDataFrame(_polygon, geometry=polygon.buffer(buffer))  # Buffer buffer on each side

            df_zonal_stats = self.get_raster_stat_at_polygons(raster, _polygon, stats, affine)


        # Rename the columns for all columns in the database by adding some suffix
        if (suffix is None) and (isinstance(raster, str)): suffix= os.path.basename(raster).split(".")[0]
        if suffix: df_zonal_stats.columns = [f"{x}_{suffix}" for x in df_zonal_stats.columns]

        if concurrent: return df_zonal_stats
        # combine the results and geodataframe
        rdf = gpd.GeoDataFrame(pd.concat([polygon, df_zonal_stats], axis=1), crs=self.crs)
        # Make sure that the last column is geometry
        rdf = self._make_geometry_the_last_column(rdf)

        return rdf


class GeoDataPolygons(GeoDataShapes):
    """A class for geodata polygons"""
    def __init__(self, path_geodata=None, crs='epsg:4326'):
        """Initiate the global class"""
        GeoDataShapes.__init__(self, path_geodata=path_geodata, crs=crs)


    def extract_raster_stat(self,raster, polygon=None, stats=['min', 'max', 'median', 'majority', 'sum'], affine=None, suffix=None):
        # Get the stat
        if polygon is None: polygon=self.gdf
        # Make a copy of polygon for ensure no overwriting
        _polygon = polygon.copy()
        # get zonal statistics
        df_zonal_stats = self.get_raster_stat_at_polygons(raster, _polygon, stats, affine)
        # Rename the columns for all columns in the database by adding some suffix
        if suffix: df_zonal_stats.columns = [f"{x}_{suffix}" for x in df_zonal_stats.columns]
        # combine the results and geodataframe
        rdf = gpd.GeoDataFrame(pd.concat([_polygon, df_zonal_stats], axis=1), crs=self.crs)
        # Make sure that the last column is geometry
        rdf = self._make_geometry_the_last_column(rdf)
        return rdf

    def spatial_join_polygon_to_polygon(self, gdf_main, gdf_merge):
        gdf_merged_to_main = gpd.sjoin(left_df=gdf_main,right_df=gdf_merge, how="left", op="intersects")
        return gdf_merged_to_main


class GeoDataRaster:
    """A class for handling raster data"""
    def __init__(self, path_data=None, crs='epsg:26915', load_file=True): #changed crs='epsg:26915'
        self.path_data=path_data # Main geodata shapefile
        self.crs = crs # Web maps are usually provided in web Mercator 'epsg:3857'

    def read_data(self, path_data=None, keep_a_copy=True, band_to_read=1):
        """Read a file to geopandas"""

        if not path_data: path_data = self.path_data
        # Read the file; srs is a reader element
        src = rasterio.open(path_data)
        self.dataset = src
        # Read the first band
        # more info https://rasterio.readthedocs.io/en/latest/topics/reading.html

        # Get the crs of the data
        raster_crs = src.crs.to_dict()["init"]

        if raster_crs != self.crs:
            print(f"Raster crs not matching the study crs, attempting conversion from {raster_crs} to {self.crs}")
            data, out_path, affine = self.reproject_raster()
            # Modify the reference to new path
            self.path_data = out_path
            self.affine = affine
        else:
            data = src.read(band_to_read)
            self.affine = src.transform

        if keep_a_copy:
            self.data = data

        return data

    def reproject_raster(self, in_path=None, band_to_read=1, out_folder="./raster_temp", destination_crs=None):

        """
        """
        if not destination_crs: destination_crs = self.crs
        if not in_path: in_path = self.path_data

        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        # reproject raster to project crs
        with rasterio.open(in_path) as src:
            src_crs = src.crs
            transform, width, height = calculate_default_transform(
                src.crs, destination_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': destination_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            out_path = f'{out_folder}/{src.name.split("/")[-1].split(".")[0]}.tif'
            #src.name.split("/")[-1].split(".")[0] gives the file name

            with rasterio.open(out_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=destination_crs,
                        resampling=Resampling.nearest)

        src = rasterio.open(out_path)
        self.dataset =src
        # Read the first band
        # more info https://rasterio.readthedocs.io/en/latest/topics/reading.html
        data = src.read(band_to_read)
        affine = src.transform

        return data, out_path, affine

    def plot_raster(self, ax=None, show_plot=False):

        if ax is None:
            ax = plt.imshow(self.data, cmap='pink')
        else:
            ax.imshow(self.data, cmap='pink')

        if show_plot: plt.show()
        return ax


    def plot_raster_for_paper(self, ax=None):
        """Prepare raster plot for publication"""
        # Prettier plotting with seaborn
        import seaborn as sns
        sns.set(font_scale=1.5, style="whitegrid")

        if not ax: plt.imshow(self.data, cmap='pink')
        if ax: ax.imshow(self.data, cmap='pink')


class SmartGeoProcess:
    """This is a class for smart geoprocessing"""
    @staticmethod
    def get_filepaths(path_directory, ends_with=".tiff", get_file_names_only=False, remove_extension=False, contains_the_string=None):
        """Gets all files with a specific extension from the provided folder"""

        file_paths = []  # List which will store all of the full filepaths.

        # Walk the tree.
        for root, directories, files in os.walk(path_directory):
            for filename in files:
                if filename.endswith(ends_with):
                    # Join the two strings in order to form the full filepath.
                    if not get_file_names_only:
                        filename = os.path.join(root, filename)

                    file_paths.append(filename)  # Add it to the list.

        if remove_extension:
            file_paths=[x.replace(ends_with, '') for x in file_paths]

        if contains_the_string:
            file_paths = [fp for fp in file_paths if contains_the_string in fp]

        return file_paths

    @staticmethod
    def batch_assign_crs(lst_path_raster, out_path, CRS):

        _func = partial(SmartGeoProcess.assign_crs_to_raster,out_path=out_path,CRS=CRS)

        with get_reusable_executor(max_workers=6) as executor:
            results = list(tqdm(executor.map(_func, lst_path_raster),
                                total=len(lst_path_raster),
                                desc="Extracting raster at GeoDataLines"))

        n_workers = len(set(results))
        print("Done! Number of used processes:", n_workers)

    @staticmethod
    def assign_crs_to_raster(path_raster, out_path, CRS):
        """Reads the raster assign it a projection and save it"""

        #Make path if it does not exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        lidar_dem = rxr.open_rasterio(path_raster)
        lidar_dem = lidar_dem.rio.set_crs(CRS, inplace=True)
        lidar_dem.rio.to_raster(os.path.join(out_path, os.path.basename(path_raster)))

    def get_raster_value_at_a_point(self):
        """Get the raster value at a point with a buffer"""

    def get_raster_value_at_a_point_with_buffer(self):
        """Get the raster value at a point with a buffer"""

    def get_raster_value_for_a_line(self):
        """Get the raster value at a point with a buffer"""

    def get_raster_value_for_a_line_gdb_with_buffer(self):
        """Get the raster value at a point with a buffer"""
