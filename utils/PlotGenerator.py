import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.ops import unary_union
from shapely.errors import GEOSException
from pathlib import Path
import traceback
import re

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*`use_inf_as_na` option is deprecated.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")

# Custom Colormap Definitions Helper Class (or dictionary)
class CustomColormaps:
    """
    Container for custom colormap definitions based on provided ColorBrewer schemes.
    Each key is the colormap name (string) used in the config file.
    Each value is a list of colors (hex strings).
    Includes both 11-color and 9-color diverging schemes (and their reversed versions),
    and 9-color sequential schemes.
    """
    # Base color definitions (keeps things cleaner)
    _prgn_11 = ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b']
    _prgn_9 = ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
    _piyg_11 = ['#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419']
    _piyg_9 = ['#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221']
    _rdbu_11 = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
    _rdbu_9 = ['#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac']
    _blues_9 = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    _oranges_9 = ['#FFFFFF', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
    # _reds_9 = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
    _reds_9 = ['#FFFFFF', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'] 
    # _orrd_9 = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000']
    _orrd_9 = ['#cccccc', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000']

    definitions = {
        # --- Diverging Schemes ---

        # PRGn (Purple-Green)
        'change_prgn_11': _prgn_11,
        'change_prgn_11_r': _prgn_11[::-1], # Reversed
        'change_prgn_9': _prgn_9,
        'change_prgn_9_r': _prgn_9[::-1],   # Reversed

        # PiYG (Pink-YellowGreen)
        'change_piyg_11': _piyg_11,
        'change_piyg_11_r': _piyg_11[::-1], # Reversed
        'change_piyg_9': _piyg_9,
        'change_piyg_9_r': _piyg_9[::-1],   # Reversed

        # RdBu (Red-Blue) - Recommended for Change/Difference
        'change_rdbu_11': _rdbu_11,
        'change_rdbu_11_r': _rdbu_11[::-1], # Reversed (Blue -> Red)
        'change_rdbu_9': _rdbu_9,
        'change_rdbu_9_r': _rdbu_9[::-1],   # Reversed (Blue -> Red)
        # Default points to 11-color Red->Blue version
        'change_rdbu': _rdbu_11,
        'change_rdbu_r': _rdbu_11[::-1],

        # --- Sequential Schemes (9 colors) ---
        'baseline_blues_surge': _blues_9,
        'baseline_oranges_debris': _oranges_9,
        'baseline_reds_clr': _reds_9,
        'baseline_orrd_network': _orrd_9,

    } # End of definitions dictionary

    # --- Static method to register colormaps (Keep as before) ---
    @staticmethod
    def register_colormaps():
        """Registers the custom colormaps with Matplotlib."""
        registered_count = 0
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        for name, colors in CustomColormaps.definitions.items():
            try:
                # Check if name already exists
                try:
                    plt.colormaps.get_cmap(name)
                    continue # Skip if already exists
                except ValueError:
                     pass # Doesn't exist, proceed

                # Parse hex string list into ListColormap
                hex_colors = []
                if isinstance(colors, str):
                    if len(colors) % 6 != 0: continue
                    hex_colors = [f"#{colors[i:i+6]}" for i in range(0, len(colors), 6)]
                elif isinstance(colors, list):
                     hex_colors = [c if c.startswith('#') else f'#{c}' for c in colors]
                else:
                    continue

                cmap = mcolors.ListedColormap(hex_colors, name=name)
                plt.colormaps.register(cmap=cmap)
                registered_count += 1
            except ValueError as e:
                print(f"Error creating/registering custom colormap '{name}': {e}")
            except Exception as e:
                print(f"Unexpected error registering custom colormap '{name}': {e}")
        if registered_count > 0:
            print(f"Registered {registered_count} custom colormaps.")


class MidpointNormalize(mcolors.Normalize):
    """Normalizes data so midpoint maps to 0.5 in colormap."""
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        vmin_valid = self.vmin is not None and np.isfinite(self.vmin)
        vmax_valid = self.vmax is not None and np.isfinite(self.vmax)
        midpoint_valid = self.midpoint is not None and np.isfinite(self.midpoint)

        if not (vmin_valid and vmax_valid):
            safe_vmin = self.vmin if vmin_valid else 0
            safe_vmax = self.vmax if vmax_valid else 1
            if safe_vmin == safe_vmax: safe_vmax = safe_vmin + 1e-6
            masked_result = np.ma.masked_invalid(result)
            interp_val = np.interp(masked_result.filled(np.nan), [safe_vmin, safe_vmax], [0, 1])
            return np.ma.masked_array(interp_val, mask=masked_result.mask)

        if midpoint_valid:
            if not (self.vmin <= self.midpoint <= self.vmax):
                if self.midpoint < self.vmin:
                    x_in, y_out = [self.vmin, self.vmax], [0.5, 1]
                else:
                    x_in, y_out = [self.vmin, self.vmax], [0, 0.5]
            else:
                x_in = [self.vmin, self.midpoint, self.vmax]
                y_out = [0, 0.5, 1]
        else:
             x_in = [self.vmin, self.vmax]
             y_out = [0, 1]

        masked_result = np.ma.masked_invalid(result)
        interp_val = np.interp(masked_result.filled(np.nan), x_in, y_out, left=y_out[0], right=y_out[-1])
        return np.ma.masked_array(interp_val, mask=masked_result.mask)


class PlotGenerator:
    """Generates maps based on configuration settings."""
    def __init__(self, config):
        """Initializes the PlotGenerator."""
        print("Initializing PlotGenerator...")
        self.config = config
        self.general_settings = config.get('general_settings', {})
        self.data_sources = config.get('data_sources', {})
        self.plot_definitions = config.get('plot_definitions', [])

        # --- Register Custom Colormaps ---
        print("Registering custom colormaps...")
        CustomColormaps.register_colormaps()
        # --- End Custom Colormap Registration ---

        self.target_crs = self.general_settings.get('target_crs', 'EPSG:3857')
        print(f" Config: Target CRS={self.target_crs}")

        # --- Load Boundaries ---
        self.boundary_gdf_tract = self._load_boundary(
            self.general_settings.get('boundary_shapefile_path_tract'),
            "Tract"
        )
        self.boundary_gdf_grid = self._load_boundary(
            self.general_settings.get('boundary_shapefile_path_grid'),
            "Grid"
        )
        self.boundary_gdf_county = self._load_boundary(
            self.general_settings.get('boundary_shapefile_path_county'),
            "County"
        )

        # --- Calculate Primary Boundary Union (using Tracts) ---
        self.boundary_union = None
        self.boundary_extent = None # Store the extent [minx, miny, maxx, maxy]
        if self.boundary_gdf_tract is not None:
            print(" Calculating primary boundary union and extent (from tracts)...")
            try:
                geoms_for_union = self.boundary_gdf_tract.geometry
                if not geoms_for_union.is_valid.all():
                    print(" Info: Fixing invalid geometries in tract boundary before union."); geoms_for_union = geoms_for_union.buffer(0)
                self.boundary_union = unary_union(geoms_for_union)
                if self.boundary_union is None or self.boundary_union.is_empty: print(" Warning: Tract boundary union is empty."); self.boundary_union = None
                elif not self.boundary_union.is_valid:
                    print(" Warning: Tract boundary union invalid. Attempting buffer(0)."); self.boundary_union = self.boundary_union.buffer(0)
                    if not self.boundary_union.is_valid: print(" Error: Tract boundary union remains invalid."); self.boundary_union = None

                if self.boundary_union and self.boundary_union.is_valid:
                    self.boundary_extent = self.boundary_union.bounds
                    print(f"  Boundary extent calculated from union: {self.boundary_extent}")
                else:
                     self.boundary_extent = self.boundary_gdf_tract.total_bounds
                     print(f"  Boundary extent calculated from GDF total_bounds: {self.boundary_extent}")
            except GEOSException as e: print(f" Error creating tract boundary union: {e}"); self.boundary_union = None; self.boundary_extent = self.boundary_gdf_tract.total_bounds if self.boundary_gdf_tract is not None else None
        else:
            print(" Warning: Tract boundary GDF not loaded, cannot calculate primary union or extent.")

        self._setup_output_dirs()
        print("Initialization complete.")

    def _load_boundary(self, path, boundary_type):
        """Loads, validates, and projects a boundary shapefile."""
        if not path:
            print(f" Info: {boundary_type} boundary path not provided in config.")
            return None
        path_obj = Path(path)
        if not path_obj.is_file():
            print(f" Warning: {boundary_type} boundary file not found at: {path_obj}. Proceeding without it.")
            return None

        print(f" Loading {boundary_type} boundary: {path_obj}")
        try:
            gdf = gpd.read_file(path_obj)
            if gdf.empty:
                print(f" Warning: {boundary_type} boundary file is empty: {path_obj}"); return None

            if gdf.crs is None:
                print(f" Warning: {boundary_type} boundary has no CRS defined. Assuming EPSG:4326."); gdf.set_crs(epsg=4326, inplace=True)

            if str(gdf.crs).lower() != str(self.target_crs).lower():
                print(f" Reprojecting {boundary_type} boundary from {gdf.crs} to {self.target_crs}"); gdf = gdf.to_crs(self.target_crs)

            if boundary_type == "Tract":
                if 'FID' not in gdf.columns:
                    print(f" Warning: {boundary_type} boundary missing 'FID'. Adding FID based on index."); gdf = gdf.reset_index().rename(columns={'index': 'FID'}); gdf['FID'] = gdf['FID'].astype(np.int64)
                else:
                    if not pd.api.types.is_integer_dtype(gdf['FID'].dtype):
                        try: gdf['FID'] = pd.to_numeric(gdf['FID']).astype(np.int64)
                        except Exception: print(f" Warning: Could not convert {boundary_type} FID to int64.")
                    if gdf['FID'].dtype != np.int64:
                        try: gdf['FID'] = gdf['FID'].astype(np.int64)
                        except Exception as e: print(f" Warning: Could not ensure {boundary_type} FID is int64: {e}")
                if not gdf['FID'].is_unique:
                    print(f" CRITICAL WARNING: {boundary_type} boundary 'FID' is NOT unique! Aggregation/merging results may be incorrect.")

            print(f" {boundary_type} boundary loaded. Features: {len(gdf)}, CRS: {gdf.crs}")
            return gdf
        except Exception as e:
            print(f" Error loading {boundary_type} boundary file {path_obj}: {e}"); traceback.print_exc(); return None

    def _setup_output_dirs(self):
        """Creates output directories defined in general_settings."""
        dirs_to_create = [
            self.general_settings.get('output_directory_maps'),
            self.general_settings.get('output_directory_legends'),
            Path(self.general_settings.get('output_directory_maps')) / 'shapefiles' if self.general_settings.get('output_directory_maps') else None
        ]
        created_count = 0
        for dir_path in dirs_to_create:
            if not dir_path:
                 print(" Warning: Output directory path is missing in general_settings.")
                 continue
            path = Path(dir_path)
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    print(f" Created output directory: {path}")
                    created_count +=1
            except Exception as e:
                print(f"  Error creating directory {path}: {e}")
        if created_count == 0 and all(dirs_to_create):
             print(" Output directories already exist.")

    def _get_data_path(self, plot_type, scenario, variable, is_network_geometry=False):
        """Constructs data path using templates."""
        source_config = self.data_sources.get(plot_type)
        if not source_config: raise ValueError(f"Data source config not found for plot_type: {plot_type}")

        if plot_type == 'network_condition' and is_network_geometry:
            network_shp_path = source_config.get('network_shapefile_path')
            if not network_shp_path: raise ValueError(f"Missing 'network_shapefile_path' for {plot_type}")
            return network_shp_path

        template = source_config.get('input_path_template')
        if not template: raise ValueError(f"Missing 'input_path_template' for {plot_type}")

        try:
             hurricane_name = scenario.get('hurricane_name')
             year = scenario.get('year')
             if hurricane_name is None or year is None:
                  raise KeyError(f"Scenario dictionary for {plot_type} must contain 'hurricane_name' and 'year'")
             path = template.format(hurricane_name=hurricane_name, year=year, variable=variable)
             return path
        except KeyError as e:
             raise ValueError(f"Missing placeholder '{e}' in input_path_template for {plot_type} ('{template}'). Check scenario keys and template string.")
        except Exception as e:
             raise ValueError(f"Error formatting path template for {plot_type}: {e}")

    def _load_data(self, plot_config, scenario, variable):
        """Loads and prepares data based on plot config, scenario, and variable."""
        plot_type = plot_config.get('plot_type')
        print(f"--- Loading data for: Type={plot_type}, Scenario={scenario}, Variable={variable} ---")
        source_config = self.data_sources.get(plot_type, {})
        data_path = None; gdf = None; geom_col_name = 'geometry'

        try:
            # Get path (different logic for network geometry)
            if plot_type == 'network_condition':
                data_path = self._get_data_path(plot_type, scenario, variable, is_network_geometry=True)
            else:
                data_path = self._get_data_path(plot_type, scenario, variable)

            data_path_obj = Path(data_path)
            if not data_path_obj.exists() and plot_type != 'network_condition': # Network CSV check is done later
                 print(f" Warning: Data file not found: {data_path_obj}"); return None, None, None

            # ---- Vector Data Types ----
            if plot_type in ['parameter_shapefile', 'clr', 'network_condition']:
                # --- Load Base Geometry/Data ---
                if plot_type == 'parameter_shapefile':
                    print(f" Reading parameter grid shapefile: {data_path_obj}")
                    gdf = gpd.read_file(data_path_obj)
                    if gdf.empty: print(f" Warning: Parameter shapefile is empty."); return None, None, None
                    geom_col_name = gdf.geometry.name
                    if variable not in gdf.columns: raise ValueError(f"Variable '{variable}' not found in columns: {gdf.columns.tolist()}")
                    cols_to_keep = [geom_col_name, variable]
                    # Preserve FID if present and numeric, else add one
                    if 'FID' in gdf.columns:
                        try: # Attempt conversion if not already int
                           if not pd.api.types.is_integer_dtype(gdf['FID'].dtype): gdf['FID'] = pd.to_numeric(gdf['FID']).astype(np.int64)
                           cols_to_keep.append('FID')
                        except Exception: print(f" Warning: Could not convert input parameter FID to int64. FID column will be ignored for joining/aggregation."); gdf = gdf.reset_index().rename(columns={'index': 'FID'}); gdf['FID'] = gdf['FID'].astype(np.int64); cols_to_keep.append('FID')
                    else:
                        gdf = gdf.reset_index().rename(columns={'index': 'FID'}); gdf['FID'] = gdf['FID'].astype(np.int64); cols_to_keep.append('FID')
                    if not gdf['FID'].is_unique: print(f" Warning: Input parameter shapefile 'FID' is not unique. Aggregation might be affected.");
                    gdf = gdf[cols_to_keep].rename(columns={variable: 'value'})


                elif plot_type == 'network_condition':
                    print(f" Reading base network shapefile: {data_path_obj}")
                    gdf = gpd.read_file(data_path_obj)
                    if gdf.empty: print(f" Warning: Network shapefile is empty."); return None, None, None
                    gdf = gdf.reset_index(drop=True) # Ensure simple index
                    geom_col_name = gdf.geometry.name
                    gdf['row_id'] = gdf.index # Add row_id for potential index-based merging if CSV fails
                    # Load CSV values
                    csv_path = self._get_data_path(plot_type, scenario, variable, is_network_geometry=False)
                    csv_path_obj = Path(csv_path)
                    if not csv_path_obj.is_file():
                        print(f" Warning: Network CSV not found ({csv_path_obj}). Network condition values will be NaN."); gdf['value'] = np.nan
                    else:
                        print(f" Reading network condition CSV: {csv_path_obj}")
                        df_values = pd.read_csv(csv_path_obj)
                        if df_values.empty: print(" Warning: Network CSV is empty."); gdf['value'] = np.nan
                        else:
                             value_cols = df_values.columns[1:]
                             if value_cols.empty: print(f" Warning: No value columns found (expected from index 1 onwards) in network CSV."); avg_values = pd.Series([np.nan] * len(df_values))
                             else:
                                 first_val_col = df_values[value_cols[0]]
                                 if not pd.api.types.is_numeric_dtype(first_val_col) and first_val_col.dtype != 'object':
                                     print(f" Warning: First value column '{value_cols[0]}' in network CSV is not numeric. Cannot calculate average."); avg_values = pd.Series([np.nan] * len(df_values))
                                 else:
                                     try:
                                         avg_values = df_values[value_cols].astype(float).mean(axis=1)
                                     except Exception as e: print(f" Warning: Could not convert network CSV values to float for averaging: {e}"); avg_values = pd.Series([np.nan] * len(df_values))

                             n_rows_geom, n_rows_csv = len(gdf), len(avg_values)
                             if n_rows_geom != n_rows_csv:
                                print(f" Warning: Network shapefile ({n_rows_geom} rows) and CSV ({n_rows_csv} rows) length mismatch. Aligning by index.")
                                avg_df = avg_values.to_frame(name='value')
                                gdf = gdf.merge(avg_df, left_index=True, right_index=True, how='left')
                                if gdf['value'].isnull().any(): print(f"  Info: {gdf['value'].isnull().sum()} network features have NaN 'value' after alignment.")
                             else:
                                gdf['value'] = avg_values.values

                elif plot_type == 'clr':
                    print(f" Reading CLR summary CSV: {data_path_obj}")
                    if self.boundary_gdf_tract is None: raise FileNotFoundError("Tract boundary file needed for CLR plots but not loaded.")
                    df_clr = pd.read_csv(data_path_obj)
                    if df_clr.empty: print(f" Warning: CLR CSV is empty."); return None, None, None
                    fid_col = df_clr.columns[0]; value_cols = df_clr.columns[1:]
                    if not fid_col.upper() == 'FID': print(f" Warning: Assuming first CLR column '{fid_col}' is the Tract FID."); df_clr = df_clr.rename(columns={fid_col: 'FID'})

                    try:
                        tract_fid_type = self.boundary_gdf_tract['FID'].dtype
                        if df_clr['FID'].dtype != tract_fid_type:
                           print(f" Info: Aligning CLR FID type ({df_clr['FID'].dtype}) to Tract FID type ({tract_fid_type}).")
                           df_clr['FID'] = df_clr['FID'].astype(tract_fid_type)
                    except Exception as e: print(f" Warning: Could not convert CLR FID type: {e}. Merge might fail or be incorrect.")

                    if value_cols.empty: print(f" Warning: No value columns found (expected from index 1 onwards) in CLR CSV."); avg_clr = pd.Series([np.nan] * len(df_clr))
                    else:
                        first_val_col = df_clr[value_cols[0]]
                        if not pd.api.types.is_numeric_dtype(first_val_col) and first_val_col.dtype != 'object':
                            print(f" Warning: First value column '{value_cols[0]}' in CLR CSV is not numeric. Cannot calculate average."); avg_clr = pd.Series([np.nan] * len(df_clr))
                        else:
                            try: avg_clr = df_clr[value_cols].astype(float).mean(axis=1)
                            except Exception as e: print(f" Warning: Could not convert CLR CSV values to float for averaging: {e}"); avg_clr = pd.Series([np.nan] * len(df_clr))

                    df_result = pd.DataFrame({'FID': df_clr['FID'], 'value': avg_clr})
                    geom_col_name = self.boundary_gdf_tract.geometry.name # CLR always merges to tracts
                    gdf = self.boundary_gdf_tract[['FID', geom_col_name]].merge(df_result, on='FID', how='left')
                    nan_count = gdf['value'].isnull().sum()
                    if nan_count > 0: print(f" Warning: {nan_count} tracts have NaN 'value' after CLR merge (check FID matching).")

                # --- Common Vector Processing: CRS ---
                if gdf.crs is None: print(f" Warning: Input vector {plot_type} has no CRS. Assuming EPSG:4326."); gdf.set_crs(epsg=4326, inplace=True)
                if str(gdf.crs).lower() != str(self.target_crs).lower(): print(f" Reprojecting input vector to {self.target_crs}"); gdf = gdf.to_crs(self.target_crs)
                geom_col_name = gdf.geometry.name # Update geom col name after reproject

                # --- Fix Invalid Geometries Before Spatial Ops ---
                if not gdf.geometry.is_valid.all():
                     invalid_count = (~gdf.geometry.is_valid).sum()
                     print(f" Info: Fixing {invalid_count} invalid geometries in loaded {plot_type} data using buffer(0).")
                     gdf[geom_col_name] = gdf.geometry.buffer(0)
                     if not gdf.geometry.is_valid.all(): print(f" Warning: Still {(~gdf.geometry.is_valid).sum()} invalid geometries after buffer(0). Clipping/aggregation might be affected.")

                # --- Specific Logic for Parameter Shapefile Output Level ---
                if plot_type == 'parameter_shapefile':
                    output_level = plot_config.get('output_level')
                    if output_level == 'tract':
                        print(" Aggregating parameter grid data to tracts...")
                        if self.boundary_gdf_tract is None: raise FileNotFoundError("Tract boundary file needed for aggregation but not loaded.")
                        agg_method = source_config.get('aggregation_method', 'mean')
                        if not self.boundary_gdf_tract['FID'].is_unique: print(" Warning: Tract boundary 'FID' not unique, aggregation results may be ambiguous.");
                        geom_col_boundary = self.boundary_gdf_tract.geometry.name
                        boundary_join_gdf = self.boundary_gdf_tract[['FID', geom_col_boundary]].copy()
                        if not boundary_join_gdf.geometry.is_valid.all(): print(" Info: Fixing invalid geometries in tract boundary before sjoin."); boundary_join_gdf[geom_col_boundary] = boundary_join_gdf[geom_col_boundary].buffer(0)

                        print(f" Performing spatial join (grid intersects tract)... Input grid features: {len(gdf)}")
                        joined = gpd.sjoin(gdf, boundary_join_gdf, how='inner', predicate='intersects')
                        print(f"  Features after join: {len(joined)}")
                        if joined.empty: print(" Warning: Spatial join resulted in zero matching features."); return None, None, None

                        if 'FID_right' in joined.columns: fid_col_agg = 'FID_right'
                        elif 'FID' in joined.columns and 'FID_left' not in joined.columns: fid_col_agg = 'FID'
                        else: raise KeyError(f"Could not determine tract boundary FID column in joined data after sjoin. Columns: {joined.columns.tolist()}")

                        print(f" Aggregating 'value' using '{agg_method}' grouped by '{fid_col_agg}'...")
                        aggregated = joined.groupby(fid_col_agg)['value'].agg(agg_method)
                        if isinstance(aggregated.index, pd.MultiIndex): print(f" Warning: Aggregation resulted in MultiIndex, check '{fid_col_agg}' column."); aggregated = aggregated.reset_index().set_index(fid_col_agg)
                        aggregated = aggregated.reset_index().rename(columns={fid_col_agg: 'FID'})
                        print(f"  Aggregated features: {len(aggregated)}")

                        print(" Merging aggregated values back to tract boundaries...")
                        boundary_fid_dtype = self.boundary_gdf_tract['FID'].dtype; aggregated_fid_dtype = aggregated['FID'].dtype
                        if boundary_fid_dtype != aggregated_fid_dtype:
                            print(f" Warning: Aligning FID types for merge (Tract: {boundary_fid_dtype}, Agg: {aggregated_fid_dtype}).")
                            try: aggregated['FID'] = aggregated['FID'].astype(boundary_fid_dtype)
                            except Exception as e: raise TypeError(f"Could not align FID types for merging aggregated data: {e}")

                        gdf = self.boundary_gdf_tract[['FID', geom_col_boundary]].merge(aggregated, on='FID', how='left')
                        nan_count = gdf['value'].isnull().sum()
                        if nan_count > 0: print(f" Warning: {nan_count} tracts have NaN 'value' after aggregation merge (check FIDs or aggregation result).")
                        geom_col_name = geom_col_boundary

                    elif output_level == 'grid':
                        print(" Clipping parameter grid data to grid boundary (if provided)...")
                        if self.boundary_gdf_grid is None:
                             print(" Info: Grid boundary file not provided. Outputting original parameter grid features within the tract union extent.")
                             if self.boundary_union and self.boundary_union.is_valid:
                                 print(f" Clipping {len(gdf)} input grid features to tract boundary union...")
                                 try: gdf = gdf.clip(self.boundary_union)
                                 except Exception as clip_err: print(f" Error during clipping with tract union: {clip_err}");
                                 print(f"  Features remaining after tract union clip: {len(gdf)}")
                             else: print(" Info: No valid boundary union available for clipping grid features.")
                        else:
                             grid_boundary_union = None
                             try:
                                 grid_geoms = self.boundary_gdf_grid.geometry
                                 if not grid_geoms.is_valid.all(): grid_geoms = grid_geoms.buffer(0)
                                 grid_boundary_union = unary_union(grid_geoms)
                                 if grid_boundary_union and grid_boundary_union.is_valid:
                                     original_count = len(gdf)
                                     print(f" Clipping {original_count} input grid features to grid boundary union...")
                                     gdf = gdf.clip(grid_boundary_union)
                                     print(f"  Features remaining after grid clip: {len(gdf)}")
                                 else: print(" Warning: Grid boundary union invalid or empty, skipping grid clip.")
                             except Exception as e: print(f" Error creating/using grid boundary union for clipping: {e}");

                    else:
                        raise ValueError(f"Invalid 'output_level' for parameter_shapefile: {output_level}. Use 'tract' or 'grid'.")

                # --- Clipping for CLR and Network ---
                elif plot_type in ['clr', 'network_condition']:
                    if self.boundary_union and self.boundary_union.is_valid and not self.boundary_union.is_empty:
                        print(f" Clipping {plot_type} data ({len(gdf)} features) to primary (tract) boundary union...")
                        original_count = len(gdf)
                        if gdf.geom_type.iloc[0] == 'LineString' or gdf.geom_type.iloc[0] == 'MultiLineString':
                            gdf_filtered = gdf[gdf.geometry.intersects(self.boundary_union)].copy()
                            print(f"  Features kept after intersects filter: {len(gdf_filtered)} / {original_count}")
                            gdf = gdf_filtered
                        else: # Polygons (CLR)
                             try:
                                 gdf = gdf.clip(self.boundary_union)
                                 print(f"  Features remaining after clip: {len(gdf)}")
                             except Exception as clip_err: print(f" Error during clipping {plot_type} data: {clip_err}");

                    else: print(" Info: Skipping vector clipping (primary boundary union invalid/empty or not applicable).")

                if gdf.empty: print(" Warning: GeoDataFrame empty after processing. Returning None."); return None, None, None
                print(f" Vector data loaded. Final features: {len(gdf)}, Geometry type: {gdf.geom_type.iloc[0] if len(gdf) > 0 else 'N/A'}")
                return gdf, 'vector', {'crs': gdf.crs}

            # ---- Raster Data Types ----
            elif plot_type == 'hazard_raster':
                 print(f" Reading hazard raster: {data_path_obj}")
                 with rasterio.open(data_path_obj) as src:
                    raster_crs = src.crs
                    if raster_crs is None:
                        print(f" Warning: Input raster has no CRS. Assuming EPSG:4326."); raster_crs = rasterio.crs.CRS.from_epsg(4326)

                    if self.boundary_gdf_tract is None: raise FileNotFoundError("Tract boundary required for raster masking but not loaded.")

                    print(f" Reprojecting tract boundary to raster CRS ({raster_crs}) for masking...")
                    try: boundary_for_mask = self.boundary_gdf_tract.to_crs(raster_crs)
                    except Exception as reproj_err: raise ValueError(f"Could not reproject tract boundary to raster CRS {raster_crs}: {reproj_err}")

                    if not boundary_for_mask.geometry.is_valid.all():
                        print(" Info: Fixing invalid tract boundary geometries before masking."); boundary_for_mask['geometry'] = boundary_for_mask.geometry.buffer(0)

                    valid_geoms = boundary_for_mask.geometry[boundary_for_mask.geometry.is_valid]
                    if valid_geoms.empty: raise ValueError("No valid geometries in tract boundary GDF (in raster CRS) for masking.")

                    masking_geom_union = unary_union(valid_geoms)
                    if not masking_geom_union.is_valid: masking_geom_union = masking_geom_union.buffer(0)
                    if not masking_geom_union or not masking_geom_union.is_valid: raise GEOSException("Tract boundary union for masking is invalid or empty.")

                    print(" Masking and cropping raster data...")
                    try:
                        out_image, out_transform = mask(src, [masking_geom_union], crop=True, filled=False)
                    except ValueError as e:
                        if "Input shapes do not overlap raster." in str(e): print(" Warning: Boundary does not overlap raster extent."); return None, None, None
                        else: raise e

                    clipped_data = out_image[0].astype(np.float32)
                    masked_array = np.ma.masked_array(clipped_data, mask=out_image.mask[0])

                    if masked_array.mask.all(): print(" Warning: Raster data entirely masked after clipping."); return None, None, None

                    print(f" Reprojecting masked raster to target CRS ({self.target_crs})...")
                    dst_crs = self.target_crs
                    left, bottom, right, top = rasterio.transform.array_bounds(masked_array.shape[0], masked_array.shape[1], out_transform)

                    transform, width, height = calculate_default_transform(
                         raster_crs, dst_crs, masked_array.shape[1], masked_array.shape[0], left, bottom, right, top)

                    reprojected_data = np.ma.empty((height, width), dtype=np.float32)
                    reprojected_data.mask = True

                    rasterio.warp.reproject(
                        source=masked_array.filled(src.nodata if src.nodata is not None else -9999),
                        destination=reprojected_data.data,
                        src_transform=out_transform,
                        src_crs=raster_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                        src_nodata=src.nodata if src.nodata is not None else -9999,
                        dst_nodata=np.nan
                    )
                    reprojected_data.mask = np.isnan(reprojected_data.data)
                    reprojected_data = np.ma.masked_invalid(reprojected_data)

                    if reprojected_data.mask.all(): print(" Warning: Raster data entirely masked after reprojection."); return None, None, None

                    print(f" Raster data loaded and reprojected. Final shape: {reprojected_data.shape}")
                    return (reprojected_data, transform), 'raster', {'crs': self.target_crs, 'nodata': np.nan, 'transform': transform}

            else:
                raise ValueError(f"Unsupported plot_type: {plot_type}")

        except Exception as e:
            print(f"!!! ERROR loading data for {plot_type}, Scenario={scenario}, Variable={variable} !!!")
            traceback.print_exc(); return None, None, None


    def _calculate_difference(self, current_data, baseline_data, data_type, plot_type):
        """Calculates absolute difference (current - baseline)."""
        print(" Calculating absolute difference...")
        if baseline_data is None:
            print(" Warning: Baseline data missing for difference calculation."); return None

        if data_type == 'vector':
            if plot_type == 'network_condition':
                print("  Using index alignment for network condition difference.")
                current_gdf = current_data.reset_index(drop=True); baseline_gdf = baseline_data.reset_index(drop=True)
                n_rows = min(len(current_gdf), len(baseline_gdf))
                if len(current_gdf) != len(baseline_gdf): print(f"  Warning: Length mismatch ({len(current_gdf)} vs {len(baseline_gdf)}). Using first {n_rows}.")
                current_gdf = current_gdf.iloc[:n_rows]; baseline_gdf = baseline_gdf.iloc[:n_rows]
                baseline_vals = baseline_gdf['value']; current_vals = current_gdf['value']
            else: # parameter_shapefile, clr (FID alignment)
                print("  Using FID alignment for difference.")
                if 'FID' not in current_data.columns or 'FID' not in baseline_data.columns: raise ValueError("FID column missing for difference calculation.")
                current_fid_type = current_data['FID'].dtype; baseline_fid_type = baseline_data['FID'].dtype
                if current_fid_type != baseline_fid_type:
                    print(f" Warning: Aligning FID types for difference ({current_fid_type} vs {baseline_fid_type}).")
                    try: baseline_data = baseline_data.copy(); baseline_data['FID'] = baseline_data['FID'].astype(current_fid_type)
                    except Exception as e: raise TypeError(f"Could not align FID types for difference: {e}")
                merged = current_data[['FID', current_data.geometry.name, 'value']].merge(
                    baseline_data[['FID', 'value']], on='FID', suffixes=('_current', '_baseline'), how='inner'
                )
                if merged.empty: print(" Warning: No common FIDs found for difference calculation."); return None
                print(f"  Found {len(merged)} common features based on FID for difference.")
                baseline_vals = merged['value_baseline']; current_vals = merged['value_current']

            diff = current_vals.astype(np.float64) - baseline_vals.astype(np.float64)
            diff[current_vals.isna() | baseline_vals.isna()] = np.nan
            print(f"  Difference calculated. NaN count: {diff.isna().sum()}")

            if plot_type == 'network_condition':
                result_gdf = current_gdf[[current_gdf.geometry.name]].copy()
                result_gdf['value'] = diff.astype(np.float32)
            else: # parameter_shapefile or clr
                 result_gdf = merged[['FID', merged.geometry.name]].copy()
                 result_gdf['value'] = diff.astype(np.float32)
            return result_gdf

        elif data_type == 'raster':
            print("  Calculating raster difference...")
            current_arr, current_trans = current_data; baseline_arr, baseline_trans = baseline_data
            if current_arr.shape != baseline_arr.shape or current_trans != baseline_trans:
                 print(" CRITICAL Error: Raster shapes/transforms differ unexpectedly for difference. Cannot proceed."); return None

            diff_arr = current_arr.astype(np.float64) - baseline_arr.astype(np.float64)
            combined_mask = np.logical_or(current_arr.mask, baseline_arr.mask)
            diff_arr = np.ma.masked_array(diff_arr, mask=combined_mask, dtype=np.float32)
            print(f"  Raster difference calculated. Valid pixels: {diff_arr.count()}")
            return (diff_arr, current_trans)
        else:
            raise ValueError(f"Unsupported data_type for difference calculation: {data_type}")

    def _calculate_percent_change(self, current_data, baseline_data, data_type, plot_type):
        """Calculates percent change ((current - baseline) / baseline) * 100."""
        print(" Calculating percent change...")
        if baseline_data is None:
             print(" Warning: Baseline data missing for percent change calculation."); return None

        if data_type == 'vector':
            if plot_type == 'network_condition':
                print("  Using index alignment for network condition percent change.")
                current_gdf = current_data.reset_index(drop=True); baseline_gdf = baseline_data.reset_index(drop=True)
                n_rows = min(len(current_gdf), len(baseline_gdf))
                if len(current_gdf) != len(baseline_gdf): print(f"  Warning: Length mismatch ({len(current_gdf)} vs {len(baseline_gdf)}). Using first {n_rows}.")
                current_gdf = current_gdf.iloc[:n_rows]; baseline_gdf = baseline_gdf.iloc[:n_rows]
                baseline_vals = baseline_gdf['value']; current_vals = current_gdf['value']
            else: # parameter_shapefile, clr (FID alignment)
                print("  Using FID alignment for percent change.")
                if 'FID' not in current_data.columns or 'FID' not in baseline_data.columns: raise ValueError("FID column missing for percent change.")
                current_fid_type = current_data['FID'].dtype; baseline_fid_type = baseline_data['FID'].dtype
                if current_fid_type != baseline_fid_type:
                    print(f" Warning: Aligning FID types for percent change ({current_fid_type} vs {baseline_fid_type}).")
                    try: baseline_data = baseline_data.copy(); baseline_data['FID'] = baseline_data['FID'].astype(current_fid_type)
                    except Exception as e: raise TypeError(f"Could not align FID types for percent change: {e}")
                merged = current_data[['FID', current_data.geometry.name, 'value']].merge(
                    baseline_data[['FID', 'value']], on='FID', suffixes=('_current', '_baseline'), how='inner'
                )
                if merged.empty: print(" Warning: No common FIDs found for percent change calculation."); return None
                print(f"  Found {len(merged)} common features based on FID for percent change.")
                baseline_vals = merged['value_baseline']; current_vals = merged['value_current']

            current_f = current_vals.astype(np.float64)
            baseline_f = baseline_vals.astype(np.float64)
            pct_change = pd.Series(np.nan, index=baseline_f.index)

            valid_mask = baseline_f.notna() & current_f.notna()
            nonzero_baseline_mask = valid_mask & (baseline_f != 0)
            zero_zero_mask = valid_mask & (baseline_f == 0) & (current_f == 0)

            pct_change.loc[nonzero_baseline_mask] = 100.0 * (current_f.loc[nonzero_baseline_mask] - baseline_f.loc[nonzero_baseline_mask]) / baseline_f.loc[nonzero_baseline_mask]
            pct_change.loc[zero_zero_mask] = 0.0
            pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)

            print(f"  Percent change calculated. NaN count: {pct_change.isna().sum()}")

            if plot_type == 'network_condition':
                result_gdf = current_gdf[[current_gdf.geometry.name]].copy()
                result_gdf['value'] = pct_change.astype(np.float32)
            else: # parameter_shapefile or clr
                 result_gdf = merged[['FID', merged.geometry.name]].copy()
                 result_gdf['value'] = pct_change.astype(np.float32)
            return result_gdf


        elif data_type == 'raster':
            print("  Calculating raster percent change...")
            current_arr, current_trans = current_data; baseline_arr, baseline_trans = baseline_data
            if current_arr.shape != baseline_arr.shape or current_trans != baseline_trans:
                 print(" CRITICAL Error: Raster shapes/transforms differ unexpectedly for pct change. Cannot proceed."); return None

            current_f = current_arr.astype(np.float64)
            baseline_f = baseline_arr.astype(np.float64)
            pct_change_arr = np.full(current_arr.shape, np.nan, dtype=np.float64)
            original_mask = np.logical_or(current_arr.mask, baseline_arr.mask)
            valid_calc_mask = ~original_mask
            valid_nonzero_baseline = valid_calc_mask & (baseline_f != 0)
            pct_change_arr[valid_nonzero_baseline] = 100.0 * (current_f[valid_nonzero_baseline] - baseline_f[valid_nonzero_baseline]) / baseline_f[valid_nonzero_baseline]
            valid_zero_baseline = valid_calc_mask & (baseline_f == 0)
            zero_zero_mask = valid_zero_baseline & (current_f == 0)
            pct_change_arr[zero_zero_mask] = 0.0
            result_masked_arr = np.ma.masked_array(pct_change_arr, mask=original_mask, dtype=np.float32)

            print(f"  Raster percent change calculated. Valid pixels: {result_masked_arr.count()}")
            return (result_masked_arr, current_trans)
        else:
             raise ValueError(f"Unsupported data_type for percent change: {data_type}")


    def _clean_filename(self, filename_part):
        """Removes potentially problematic characters for filenames."""
        if filename_part is None: return "None"
        cleaned = re.sub(r'[\\/*?:"<>|]', '_', str(filename_part))
        cleaned = cleaned.replace(' ', '_')
        return cleaned

    def _construct_output_paths(self, plot_config):
        """Generates output map, legend, and shapefile file paths dynamically."""
        map_dir = Path(self.general_settings.get('output_directory_maps', './outputs/figure/generated_maps'))
        legend_dir = Path(self.general_settings.get('output_directory_legends', './outputs/figure/generated_maps/legends'))
        shapefile_dir = map_dir / 'shapefiles'

        map_dir.mkdir(parents=True, exist_ok=True)
        legend_dir.mkdir(parents=True, exist_ok=True)
        shapefile_dir.mkdir(parents=True, exist_ok=True)

        var = self._clean_filename(plot_config.get('variable', 'novar'))
        h_name = self._clean_filename(plot_config.get('scenario', {}).get('hurricane_name', 'nohurr'))
        year = self._clean_filename(plot_config.get('scenario', {}).get('year', 'noyear'))
        mode = self._clean_filename(plot_config.get('map_mode', 'nomode'))
        plot_type = self._clean_filename(plot_config.get('plot_type', 'notype'))

        level_part = ""
        if plot_type == 'parameter_shapefile':
            level = self._clean_filename(plot_config.get('output_level', 'nolevel'))
            level_part = f"_{level}"

        base_filename = f"{var}_{h_name}_{year}_{mode}{level_part}"

        map_path = map_dir / f"{base_filename}.png"
        legend_path = legend_dir / f"{base_filename}_legend.png"
        shapefile_path = shapefile_dir / base_filename

        return map_path, legend_path, shapefile_path


    def _save_legend(self, cmap, norm, output_legend_path, label=""):
        """Saves a horizontal colorbar legend."""
        if not output_legend_path: return
        print(f" Saving legend to: {output_legend_path}")
        fig_legend = plt.figure(figsize=(5, 1)); ax_legend = fig_legend.add_axes([0.05, 0.4, 0.9, 0.2])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

        boundaries = None; extend = 'neither'; tick_values = None
        vmin_cb, vmax_cb = norm.vmin, norm.vmax

        valid_bounds = vmin_cb is not None and vmax_cb is not None and np.isfinite(vmin_cb) and np.isfinite(vmax_cb) and vmin_cb != vmax_cb
        if not valid_bounds:
             print(f" Warning: Invalid or equal vmin/vmax ({vmin_cb}, {vmax_cb}) for legend. Using default [0, 1].")
             vmin_cb, vmax_cb = 0, 1
             norm_cb = mcolors.Normalize(vmin=vmin_cb, vmax=vmax_cb)
             sm = plt.cm.ScalarMappable(norm=norm_cb, cmap=cmap); sm.set_array([])
        else:
            norm_cb = norm

        if isinstance(norm_cb, MidpointNormalize):
            midpoint_cb = getattr(norm_cb, 'midpoint', None)
            midpoint_valid = midpoint_cb is not None and np.isfinite(midpoint_cb)
            extend = 'both'

            if midpoint_valid and vmin_cb < midpoint_cb < vmax_cb:
                tick_values = sorted(list(set([vmin_cb, midpoint_cb, vmax_cb])))
            else:
                tick_values = [vmin_cb, vmax_cb]
            if len(tick_values) > 1:
                filtered_ticks=[tick_values[0]]; min_tick_sep=abs(vmax_cb-vmin_cb)*0.1
                for i in range(1,len(tick_values)):
                    if abs(tick_values[i]-filtered_ticks[-1]) > max(min_tick_sep, 1e-9):
                        filtered_ticks.append(tick_values[i])
                tick_values=filtered_ticks
            if vmin_cb not in tick_values and len(tick_values)<5: tick_values.insert(0,vmin_cb)
            if vmax_cb not in tick_values and len(tick_values)<5: tick_values.append(vmax_cb)
            tick_values=sorted(list(set(tick_values)))
        else:
            tick_values = [vmin_cb, vmax_cb]
            extend = 'both'

        try:
             cb = fig_legend.colorbar(sm, cax=ax_legend, orientation='horizontal', boundaries=boundaries, extend=extend, ticks=tick_values)
             if label: cb.set_label(label, size=self.general_settings.get('font_defaults', {}).get('label_size', 10))
             cb.ax.tick_params(labelsize=self.general_settings.get('font_defaults', {}).get('tick_size', 8))
             cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.2g}'))
             fig_legend.patch.set_facecolor('none')
             plt.savefig(output_legend_path, dpi=self.general_settings.get('figure_defaults', {}).get('dpi', 300), bbox_inches='tight', transparent=True, facecolor='none', edgecolor='none')
        except Exception as e: print(f" Error saving legend: {e}"); traceback.print_exc()
        finally: plt.close(fig_legend)

    def _save_shapefile(self, data, data_type, shapefile_path, metadata=None):
        """Saves data as a shapefile if possible."""
        if data is None:
            print(f" Warning: No data to save as shapefile.")
            return
            
        if data_type == 'vector':
            try:
                print(f" Saving vector data as shapefile to: {shapefile_path}")
                data.to_file(shapefile_path)
                print(f" Shapefile saved successfully.")
            except Exception as e:
                print(f" Error saving vector data as shapefile: {e}")
                traceback.print_exc()
        elif data_type == 'raster':
            try:
                print(f" Saving raster data as GeoTIFF to: {shapefile_path}.tif")
                raster_data, transform = data
                crs = metadata.get('crs') if metadata else None
                
                with rasterio.open(
                    f"{shapefile_path}.tif",
                    'w',
                    driver='GTiff',
                    height=raster_data.shape[0],
                    width=raster_data.shape[1],
                    count=1,
                    dtype=raster_data.dtype,
                    crs=crs,
                    transform=transform,
                    nodata=np.nan
                ) as dst:
                    dst.write(raster_data.filled(np.nan), 1)
                print(f" GeoTIFF saved successfully.")
            except Exception as e:
                print(f" Error saving raster data as GeoTIFF: {e}")
                traceback.print_exc()
        else:
            print(f" Warning: Unsupported data type '{data_type}' for shapefile output.")

    def generate_map(self, plot_config):
        """Generates a single map based on the provided plot definition."""
        plot_id = plot_config.get('plot_id', 'unnamed_plot')
        print(f"\n===== Generating Map: {plot_id} =====")

        plot_type = plot_config.get('plot_type'); scenario = plot_config.get('scenario'); variable = plot_config.get('variable'); map_mode = plot_config.get('map_mode', 'baseline')
        if not all([plot_type, scenario, variable, map_mode]): print(f" Error: Plot definition '{plot_id}' missing required fields (plot_type, scenario, variable, map_mode). Skipping."); return
        if map_mode not in ['baseline', 'percent_change', 'difference']: print(f" Error: Invalid map_mode '{map_mode}' for plot '{plot_id}'. Use 'baseline', 'percent_change', or 'difference'. Skipping."); return

        try: output_map_path, output_legend_path, output_shapefile_path = self._construct_output_paths(plot_config)
        except Exception as e: print(f" Error constructing output paths for '{plot_id}': {e}. Skipping."); return
        print(f"  Output Map Path: {output_map_path}")
        print(f"  Output Legend Path: {output_legend_path}")
        print(f"  Output Shapefile Path: {output_shapefile_path}")

        current_data, data_type, metadata = self._load_data(plot_config, scenario, variable)
        if current_data is None: print(f" Error: Failed to load data for {plot_id}. Skipping map generation."); return

        data_to_plot = None
        if map_mode == 'baseline':
            data_to_plot = current_data
        elif map_mode == 'percent_change' or map_mode == 'difference':
            baseline_hurricane = plot_config.get('baseline_hurricane')
            baseline_year = plot_config.get('baseline_year')
            if not baseline_hurricane or not baseline_year: print(f" Error: Plot '{plot_id}' requires 'baseline_hurricane' and 'baseline_year' for map_mode '{map_mode}'. Skipping."); return
            baseline_scenario = {'hurricane_name': baseline_hurricane, 'year': baseline_year}
            print(f" Loading baseline data ({baseline_hurricane}/{baseline_year}) for comparison...")
            baseline_plot_config = plot_config.copy(); baseline_plot_config['scenario'] = baseline_scenario
            baseline_data, baseline_data_type, _ = self._load_data(baseline_plot_config, baseline_scenario, variable)
            if baseline_data is None: print(f" Error: Failed to load baseline data for comparison. Skipping."); return
            if data_type != baseline_data_type: print(f" Error: Data type mismatch between current ({data_type}) and baseline ({baseline_data_type}). Skipping."); return

            if map_mode == 'percent_change':
                 data_to_plot = self._calculate_percent_change(current_data, baseline_data, data_type, plot_type)
            else: # map_mode == 'difference'
                 data_to_plot = self._calculate_difference(current_data, baseline_data, data_type, plot_type)
        else:
            print(f" Error: Invalid map_mode '{map_mode}'. Skipping."); return

        if data_to_plot is None: print(f" Error: Failed to obtain data_to_plot for {plot_id} (calculation failed or data missing). Skipping."); return

        # Save data as shapefile or GeoTIFF
        self._save_shapefile(data_to_plot, data_type, output_shapefile_path, metadata)

        # --- Configure Plot Aesthetics ---
        fig_defaults = self.general_settings.get('figure_defaults', {}); font_defaults = self.general_settings.get('font_defaults', {})
        effective_style = fig_defaults.copy()
        effective_style.update(plot_config)

        dpi = effective_style.get('dpi', 300); figsize = effective_style.get('figsize', [6, 6]); transparent = effective_style.get('transparent_background', True); bg_color = 'none' if transparent else 'white'
        cmap_name = effective_style.get('color_map', 'viridis'); vmin_cfg = effective_style.get('vmin'); vmax_cfg = effective_style.get('vmax'); midpoint = effective_style.get('midpoint', 0)
        axis_padding_factor = effective_style.get('axis_padding_factor', 0.05)

        line_width = None; edge_color = None; linestyle = None
        if data_type == 'vector':
             if plot_type == 'network_condition':
                 line_width = effective_style.get('network_line_width', 1.0)
                 edge_color = None
                 linestyle = effective_style.get('network_linestyle', 'solid')
             elif plot_type == 'parameter_shapefile' and plot_config.get('output_level') == 'grid':
                 line_width = effective_style.get('grid_line_width', 0.1)
                 edge_color = effective_style.get('grid_edge_color', 'grey')
                 linestyle = effective_style.get('grid_linestyle', 'solid')
             else: # Tracts
                 line_width = effective_style.get('polygon_line_width', 0.2)
                 edge_color = effective_style.get('polygon_edge_color', 'black')
                 linestyle = effective_style.get('polygon_linestyle', 'solid')

        # --- Determine Color Normalization & Colormap ---
        # Use plt.get_cmap, which now understands registered custom names
        try:
            cmap = plt.get_cmap(cmap_name)
            print(f" Using colormap: '{cmap_name}'")
        except ValueError:
            print(f"Error: Colormap '{cmap_name}' not found (neither built-in nor custom). Using 'viridis' as fallback.")
            cmap = plt.get_cmap('viridis')

        try: cmap.set_bad(alpha=0.0)
        except Exception as e: print(f" Warning: Could not set transparent color for invalid data on colormap '{cmap_name}': {e}")

        norm = None; legend_label = variable; vmin, vmax = None, None
        actual_values = None
        if data_type == 'vector':
            if 'value' in data_to_plot.columns and not data_to_plot['value'].isnull().all():
                actual_values = data_to_plot['value'].dropna()
        elif data_type == 'raster':
            raster_array, _ = data_to_plot
            if not raster_array.mask.all():
                 actual_values = raster_array.compressed()

        if actual_values is None or len(actual_values) == 0:
             print(" Warning: No valid data found for normalization range determination.")
             vmin = vmin_cfg if vmin_cfg is not None else 0
             vmax = vmax_cfg if vmax_cfg is not None else 1
             if vmin == vmax: vmax = vmin + 1e-6
        else:
             data_min_actual, data_max_actual = actual_values.min(), actual_values.max()
             print(f" Data range for normalization: min={data_min_actual:.3f}, max={data_max_actual:.3f}")
             vmin = vmin_cfg if vmin_cfg is not None else data_min_actual
             vmax = vmax_cfg if vmax_cfg is not None else data_max_actual
             if vmin > vmax: print(f" Warning: vmin ({vmin}) > vmax ({vmax}). Swapping."); vmin, vmax = vmax, vmin
             if vmin == vmax: vmax = vmin + 1e-6

        if map_mode == 'percent_change' or map_mode == 'difference':
            norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint)
            mode_label = "Difference" if map_mode == 'difference' else "% Change"
            legend_label = f"{mode_label} in {variable}"
        else: # Baseline
             norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
             legend_label = f"{variable}"

        # --- Create Plot ---
        print(" Creating figure...")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(bg_color); fig.patch.set_alpha(0.0 if transparent else 1.0)
        ax.set_facecolor(bg_color); ax.set_aspect('equal', adjustable='box'); ax.axis('off')

        # --- Plot Network Background ---
        if plot_type == 'network_condition' and effective_style.get('network_background_visible', False):
            if self.boundary_gdf_tract is not None and not self.boundary_gdf_tract.empty:
                print(" Plotting network background (tracts)...")
                bg_color = effective_style.get('network_background_color', '#EEEEEE')
                bg_ec = effective_style.get('network_background_edge_color', '#BBBBBB')
                bg_lw = effective_style.get('network_background_line_width', 0.3)
                bg_ls = effective_style.get('network_background_linestyle', 'dashed')
                try:
                    self.boundary_gdf_tract.plot(ax=ax, color=bg_color, edgecolor=bg_ec, linewidth=bg_lw, linestyle=bg_ls, zorder=0)
                except Exception as bg_plot_err:
                     print(f" Warning: Failed to plot network background: {bg_plot_err}")
            else:
                print(" Warning: Cannot plot network background - Tract boundary not loaded or empty.")

        # --- Plot Main Data ---
        plot_extent_source = None
        try:
            if data_type == 'vector':
                if not data_to_plot.empty and 'value' in data_to_plot.columns:
                    print(f" Plotting vector data ({len(data_to_plot)} features)...")
                    data_to_plot.plot(column='value', cmap=cmap, norm=norm,
                                      linewidth=line_width, edgecolor=edge_color, linestyle=linestyle,
                                      ax=ax, legend=False,
                                      missing_kwds={"color": "none", "edgecolor": "none", "hatch": "//"},
                                      zorder=1)
                    plot_extent_source = data_to_plot
                else: print(" Warning: No vector data or 'value' column to plot.")
            elif data_type == 'raster':
                raster_array, transform = data_to_plot
                if not raster_array.mask.all():
                    print(f" Plotting raster data (shape: {raster_array.shape})...")
                    show(raster_array, transform=transform, ax=ax, cmap=cmap, norm=norm, interpolation='none', zorder=1)
                    height, width = raster_array.shape
                    plot_extent_source = rasterio.transform.array_bounds(height, width, transform)
                else: print(" Warning: No raster data to plot (all masked).")
        except Exception as plot_err:
            print(f"!!! ERROR during plotting data for {plot_id} !!!"); traceback.print_exc(); plt.close(fig); return

        # --- Set Axis Limits based on OVERALL Boundary Extent ---
        print(" Setting axis limits based on overall boundary extent...")
        if self.boundary_extent is not None and all(np.isfinite(self.boundary_extent)):
            minx, miny, maxx, maxy = self.boundary_extent
            width = maxx - minx; height = maxy - miny
            if width <= 0 or height <= 0:
                print(f" Warning: Overall boundary extent invalid ({self.boundary_extent}). Falling back to data bounds or autoscale.")
                data_bounds = None
                if plot_extent_source is not None:
                     if isinstance(plot_extent_source, gpd.GeoDataFrame): data_bounds = plot_extent_source.total_bounds
                     else: data_bounds = plot_extent_source
                if data_bounds is not None and all(np.isfinite(data_bounds)):
                     minx, miny, maxx, maxy = data_bounds; width = maxx - minx; height = maxy - miny
                     pad_x = width * axis_padding_factor; pad_y = height * axis_padding_factor
                     ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
                     print(f"  Axis limits set from data bounds: xlim=({ax.get_xlim()}), ylim=({ax.get_ylim()})")
                else:
                     print("  Warning: No valid data bounds either. Using autoscale."); ax.autoscale_view()
            else:
                pad_x = width * axis_padding_factor; pad_y = height * axis_padding_factor
                ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
                print(f" Axis limits set from overall boundary: xlim=({ax.get_xlim()}), ylim=({ax.get_ylim()})")
        else:
            print(" Warning: Overall boundary extent not available. Using autoscale."); ax.autoscale_view()
            
        # --- Plot County Boundary as Border ---
        map_border_config = self.general_settings.get('map_border', {})
        # Allow plot-level override of border settings
        plot_border_config = plot_config.get('map_border', {})
        if plot_border_config:
            # Merge plot-specific border config with general settings, prioritizing plot-specific settings
            merged_border_config = map_border_config.copy()
            merged_border_config.update(plot_border_config)
            map_border_config = merged_border_config
        
        border_visible = map_border_config.get('visible', True)
        
        if border_visible and self.boundary_gdf_county is not None and not self.boundary_gdf_county.empty:
            print(" Adding county boundary border...")
            try:
                border_line_width = map_border_config.get('line_width', 1.0)
                border_line_color = map_border_config.get('line_color', 'black')
                border_line_style = map_border_config.get('line_style', 'solid')
                border_zorder = map_border_config.get('zorder', 10)
                
                # Plot county boundary with transparent fill and configurable border
                self.boundary_gdf_county.plot(
                    ax=ax,
                    facecolor='none',  # Transparent fill
                    edgecolor=border_line_color,
                    linewidth=border_line_width,
                    linestyle=border_line_style,
                    zorder=border_zorder  # Ensure border appears on top
                )
                print(f" County boundary border added successfully (style: {border_line_color}, width: {border_line_width}, type: {border_line_style})")
            except Exception as border_err:
                print(f" Warning: Failed to add county boundary border: {border_err}")
                traceback.print_exc()
        elif border_visible:
            print(" Warning: County boundary not available for border. Make sure boundary_shapefile_path_county is set correctly.")

        # --- Save Figure and Legend ---
        try:
            print(f" Saving map to: {output_map_path}")
            plt.savefig(output_map_path, dpi=dpi, bbox_inches='tight', transparent=transparent, facecolor=fig.get_facecolor() if not transparent else 'none', edgecolor='none', pad_inches=0.01)
            self._save_legend(cmap, norm, output_legend_path, label=legend_label)
        except Exception as e: print(f" Error saving map figure or legend: {e}"); traceback.print_exc()
        finally: plt.close(fig)

        print(f"===== Map Generation Complete: {plot_id} =====")


    def generate_all_maps(self):
        """Generates all maps defined in the 'plot_definitions' section."""
        if not self.plot_definitions: print("No plot definitions found."); return
        print(f"\n>>> Starting map generation for {len(self.plot_definitions)} definitions... <<<")
        success_count, fail_count = 0, 0
        for i, plot_def in enumerate(self.plot_definitions):
            plot_id = plot_def.get('plot_id', f'plot_{i+1}')
            try:
                self.generate_map(plot_def)
                success_count += 1
            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); print(f"CRITICAL ERROR generating map '{plot_id}': {e}"); traceback.print_exc(); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); fail_count += 1; print(f"--- Continuing with next plot ---")
        print(f"\n>>> Map Generation Summary <<<"); print(f"  Successfully generated: {success_count}"); print(f"  Failed to generate:   {fail_count}"); print("-----------------------------------------")