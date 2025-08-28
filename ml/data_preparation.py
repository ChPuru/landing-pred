# data_preparation.py
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import PATHS, TRAINING
from utils import setup_logging

def prepare_datasets():
    setup_logging()
    logging.info("--- Starting Data Preparation and Feature Scaling ---")
    
    try:
        df = pd.read_csv(PATHS['raw_dataset'])
    except FileNotFoundError:
        logging.error(f"Raw dataset not found: {PATHS['raw_dataset']}. Please run a data generator.")
        return

    # --- 1. Prepare Labels ---
    actual_landings = df.groupby('flight_id').last()[['lat', 'lon']].rename(
        columns={'lat': 'landing_lat', 'lon': 'landing_lon'}
    )
    df = pd.merge(df, actual_landings, on='flight_id')

    # --- 2. CRITICAL FIX: Group-Aware Data Split ---
    # We split the FLIGHT IDs, not the rows, to prevent data leakage.
    unique_flight_ids = df['flight_id'].unique()
    train_flight_ids, meta_flight_ids = train_test_split(
        unique_flight_ids,
        test_size=TRAINING['meta_split_ratio'],
        random_state=TRAINING['random_state']
    )
    
    train_df = df[df['flight_id'].isin(train_flight_ids)]
    meta_df = df[df['flight_id'].isin(meta_flight_ids)]
    logging.info(f"Data split by flight ID: {len(train_flight_ids)} flights for training, {len(meta_flight_ids)} for meta-learning.")

    # --- 3. Feature Scaling ---
    # We fit the scaler ONLY on the training data to prevent leakage.
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    meta_df_scaled = meta_df.copy()
    
    train_df_scaled[TRAINING['features']] = scaler.fit_transform(train_df[TRAINING['features']])
    meta_df_scaled[TRAINING['features']] = scaler.transform(meta_df[TRAINING['features']]) # Use the same scaler
    logging.info("Features scaled successfully.")

    # --- 4. Save Artifacts ---
    train_df_scaled.to_csv(PATHS['processed_train_data'], index=False)
    meta_df_scaled.to_csv(PATHS['processed_meta_data'], index=False)
    joblib.dump(scaler, PATHS['scaler'])
    logging.info(f"Processed data and scaler saved. Training data: {PATHS['processed_train_data']}")

if __name__ == "__main__":
    prepare_datasets()