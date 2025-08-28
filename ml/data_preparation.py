# data_preparation.py
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from config import PATHS, TRAINING
from utils import setup_logging

def calculate_displacement(row):
    p_a = (row['lat'], row['lon'])
    p_b = (row['landing_lat'], row['landing_lon'])
    p_lat_only = (row['landing_lat'], row['lon'])
    disp_lat_m = geodesic(p_a, p_lat_only).meters
    if row['landing_lat'] < row['lat']: disp_lat_m *= -1
    p_lon_only = (row['lat'], row['landing_lon'])
    disp_lon_m = geodesic(p_a, p_lon_only).meters
    if row['landing_lon'] < row['lon']: disp_lon_m *= -1
    return disp_lat_m, disp_lon_m

def prepare_datasets():
    setup_logging()
    logging.info("--- Starting Data Preparation with Target Shaping & Scaling ---")
    df = pd.read_csv(PATHS['raw_dataset'])
    actual_landings = df.groupby('flight_id').last()[['lat', 'lon']].rename(columns={'lat': 'landing_lat', 'lon': 'landing_lon'})
    df = pd.merge(df, actual_landings, on='flight_id')
    logging.info("Calculating North/East displacement labels (Target Shaping)...")
    displacements = df.apply(calculate_displacement, axis=1)
    df[['landing_disp_north_m', 'landing_disp_east_m']] = pd.DataFrame(displacements.tolist(), index=df.index)
    unique_flight_ids = df['flight_id'].unique()
    train_flight_ids, meta_flight_ids = train_test_split(unique_flight_ids, test_size=TRAINING['meta_split_ratio'], random_state=TRAINING['random_state'])
    train_df = df[df['flight_id'].isin(train_flight_ids)]
    meta_df = df[df['flight_id'].isin(meta_flight_ids)]
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    meta_df_scaled = meta_df.copy()
    train_df_scaled[TRAINING['features']] = scaler.fit_transform(train_df[TRAINING['features']])
    meta_df_scaled[TRAINING['features']] = scaler.transform(meta_df[TRAINING['features']])
    train_df_scaled.to_csv(PATHS['processed_train_data'], index=False)
    meta_df_scaled.to_csv(PATHS['processed_meta_data'], index=False)
    joblib.dump(scaler, PATHS['scaler'])
    logging.info(f"Processed data and SCALER saved. Scaler is at: {PATHS['scaler']}")

if __name__ == "__main__":
    prepare_datasets()