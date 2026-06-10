import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt

hurricane_key_word = 'ike'

def process_data(consider_after_2008=True):
    # Load the shapefile
    shapefile_path = f'./outputs/final_polygons_with_all_parameters/{hurricane_key_word}/Grid_250_all_parameters_Kooshan.shp'
    polygon_gdf = gpd.read_file(shapefile_path)

    # Set CRS for the shapefile if not already set
    polygon_gdf.crs = 'EPSG:4326'

    # Load the CSV file
    csv_path = './outputs/merged_dataset/merged_data_with_geometry.csv'
    df = pd.read_csv(csv_path)

    # Filter out rows with Year_built > 2008 if consider_after_2008 is False
    if not consider_after_2008:
        df = df[df['Year_built'] <= 2008]

    # Convert the 'geometry' column in the CSV to a GeoSeries
    df['geometry'] = df['geometry'].apply(wkt.loads)
    point_gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # Set CRS for the point GeoDataFrame to match the CSV file's original CRS
    point_gdf.crs = 'EPSG:2278'

    # Project the points to the same CRS as the shapefile (EPSG:4326)
    point_gdf = point_gdf.to_crs('EPSG:4326')

    # Ensure polygon_gdf is also in the same CRS (EPSG:4326)
    polygon_gdf = polygon_gdf.to_crs('EPSG:4326')

    # Perform spatial join - this adds the polygon attributes to each point within it
    joined_gdf = gpd.sjoin(point_gdf, polygon_gdf, how='inner', predicate='within')

    # Define abbreviations for the columns to aggregate
    columns_to_sum = {
        'Sum_of_the_areas_of_Main_Areas': 'TBFA',
        'Count_of_accesory_structures': 'NAS',
        'Sum_of_the_area_of_the_accesory_structures': 'TAAS',
        'Count_Mobile_homes': 'NumMH'
    }

    # Check if columns exist before conversion and aggregation
    missing_columns = [col for col in columns_to_sum.keys() if col not in joined_gdf.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in joined_gdf: {missing_columns}")

    # Convert specified columns to float to ensure proper summation
    for col in columns_to_sum.keys():
        joined_gdf[col] = joined_gdf[col].astype(float)

    # Renaming columns to match the abbreviated names
    joined_gdf.rename(columns=columns_to_sum, inplace=True)

    # Initialize columns in polygon_gdf for aggregated data
    for abbreviation in columns_to_sum.values():
        polygon_gdf[abbreviation] = np.nan  # Use np.nan instead of 0 for initialization

    # Aggregate data for each polygon
    for index, polygon in polygon_gdf.iterrows():
        polygon_points = joined_gdf[joined_gdf.index_right == index]
        if not polygon_points.empty:
            polygon_gdf.at[index, 'NumB'] = len(polygon_points)  # Number of Points in Polygon
            for original, abbreviation in columns_to_sum.items():
                polygon_gdf.at[index, abbreviation] = polygon_points[abbreviation].sum()  # Use abbreviated column names
        else:
            # If there are no points in the polygon, NumB (Number of Buildings) is set to NaN
            polygon_gdf.at[index, 'NumB'] = np.nan

    # Adding variables from Mitchell
    additional_csv_path = f'../data_raw/Mitchell/{hurricane_key_word}/Grid_2020_corrected.csv'
    additional_df = pd.read_csv(additional_csv_path)

    polygon_gdf['FID'] = polygon_gdf['FID'].astype(int)
    additional_df['FID_1'] = additional_df['FID_1'].astype(int)

    # Merge additional_df into polygon_gdf
    polygon_gdf = polygon_gdf.merge(additional_df, left_on='FID', right_on='FID_1', how='left')

    # Ensure 'geometry' is the second column
    columns = ['FID', 'geometry'] + [c for c in polygon_gdf.columns if c not in ['FID', 'geometry']]

    # Reorder polygon_gdf to ensure 'FID' is first and 'geometry' is second
    polygon_gdf = polygon_gdf[columns]

    # Define the columns you are interested in, in the desired order
    interested_columns = [
        'SD_max', 'WH_max', 'WaveD_mean', 'WV_x_max', 'WV_y_max', 'WindD_mean', 'WS_mean',
        'MF_max', 'NumB', 'TBFA', 'NAS', 'TAAS', 'NumMH', 'WeightPf_1', 'WeightPf_2', 'openwater2',
        'devlow20', 'devhigh20', 'devtotal20', 'road_densi', 'min_elevat', 'seawall_di', 'dev_lag20',
        'pop_densit', 'households', 'med_income', 'housing_un', 'occupied20', 'vacant2020', 'renters202'
    ]

    # Define the new names for these columns
    new_names = [
        'SD', 'WH', 'WaveD', 'WV_X', 'WV_Y', 'WindD', 'WS', 'MF', 'NumB', 'TBFA', 'NAS', 'TAAS', 'NumMH',
        'WPF_1', 'WPF_2', 'OW', 'DO', 'DM', 'DT', 'RD', 'ME', 'ADS', 'UL', 'PD', 'NumH', 'MHI', 'NumHU',
        'OHU', 'VHU', 'PR'
    ]

    # Create a new DataFrame with only the columns of interest
    final_gdf = polygon_gdf[['FID', 'geometry'] + interested_columns]

    # Rename the columns in final_gdf
    final_gdf.columns = ['FID', 'geometry'] + new_names

    # Define your save paths
    save_path_csv = f'./outputs/final_input_for_debris_volume_model/{hurricane_key_word}/final_input_parameters.csv'
    save_path_shp = f'./outputs/final_input_for_debris_volume_model/{hurricane_key_word}/final_input_parameters.shp'

    # Save final_gdf as a CSV
    final_gdf.to_csv(save_path_csv, index=False)

    # Save final_gdf as a Shapefile
    final_gdf.to_file(save_path_shp)

# Run the code
process_data(consider_after_2008=False)
