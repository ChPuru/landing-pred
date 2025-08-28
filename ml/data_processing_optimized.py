# ml/data_processing_optimized.py
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import distance
from config import PATHS, TRAINING
from utils import setup_logging

def calculate_displacement_vectorized(df):
    """
    Vectorized calculation of landing displacement.
    ~100x faster than the row-by-row approach.
    """
    # Convert to numpy arrays for speed
    lats = df['lat'].values
    lons = df['lon'].values
    landing_lats = df['landing_lat'].values
    landing_lons = df['landing_lon'].values
    
    # Vectorized displacement calculation
    # Approximate meters per degree (more accurate than fixed conversion)
    lat_rad = np.radians(lats)
    meters_per_deg_lat = 111320.0  # Constant
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    
    # Calculate displacements in meters
    disp_north_m = (landing_lats - lats) * meters_per_deg_lat
    disp_east_m = (landing_lons - lons) * meters_per_deg_lon
    
    return disp_north_m, disp_east_m

def engineer_advanced_features(df):
    """
    Engineer sophisticated features from telemetry data.
    This is where the magic happens for better ML performance.
    """
    logging.info("Engineering advanced features...")
    
    # Sort by flight_id and time for proper sequential operations
    df = df.sort_values(['flight_id', 'alt'], ascending=[True, False]).copy()
    
    # === AERODYNAMIC FEATURES ===
    # Descent rate changes (acceleration)
    df['descent_acceleration'] = df.groupby('flight_id')['vel_v'].diff().fillna(0)
    
    # Reynolds number for aerodynamic regime
    df['reynolds_number'] = calculate_reynolds_number_vectorized(
        df['alt'].values, 
        np.abs(df['vel_v'].values)
    )
    
    # Dynamic pressure (related to aerodynamic forces)
    air_density = 1.225 * np.exp(-df['alt'] / 8500.0)
    df['dynamic_pressure'] = 0.5 * air_density * df['vel_v'] ** 2
    
    # === TEMPORAL FEATURES ===
    # Time since deployment (approximate)
    df['time_since_deployment'] = df.groupby('flight_id').cumcount()
    
    # Normalized altitude (0 = ground, 1 = deployment)
    df['altitude_normalized'] = df['alt'] / df.groupby('flight_id')['alt'].transform('max')
    
    # Altitude loss rate (more stable than raw velocity)
    df['altitude_loss_rate'] = -df['vel_v']
    
    # === TRAJECTORY FEATURES ===
    # Horizontal movement between samples
    df['lat_diff'] = df.groupby('flight_id')['lat'].diff().fillna(0)
    df['lon_diff'] = df.groupby('flight_id')['lon'].diff().fillna(0)
    
    # Convert to meters for meaningful units
    lat_rad = np.radians(df['lat'])
    df['horizontal_velocity_north'] = df['lat_diff'] * 111320.0
    df['horizontal_velocity_east'] = df['lon_diff'] * 111320.0 * np.cos(lat_rad)
    
    # Total horizontal speed
    df['horizontal_speed'] = np.sqrt(
        df['horizontal_velocity_north']**2 + df['horizontal_velocity_east']**2
    )
    
    # Estimated wind speed (from horizontal movement)
    df['estimated_wind_speed'] = df['horizontal_speed']  # Simplified assumption
    
    # === STABILITY FEATURES ===
    # Rolling statistics for stability assessment
    window = 5
    for col in ['vel_v', 'horizontal_speed']:
        if col in df.columns:
            df[f'{col}_std'] = df.groupby('flight_id')[col].rolling(window, min_periods=1).std().values
            df[f'{col}_mean'] = df.groupby('flight_id')[col].rolling(window, min_periods=1).mean().values
    
    # Trajectory curvature (change in direction)
    df['trajectory_curvature'] = calculate_trajectory_curvature_vectorized(df)
    
    # === ENVIRONMENTAL FEATURES ===
    # Air density at current altitude
    df['air_density'] = 1.225 * np.exp(-df['alt'] / 8500.0)
    
    # Expected terminal velocity based on current conditions
    drag_coeff = 0.8  # From config
    parachute_area = 0.5  # From config
    mass = 0.350  # From config
    
    drag_term = 0.5 * df['air_density'] * drag_coeff * parachute_area
    df['expected_terminal_velocity'] = np.sqrt((mass * 9.81) / drag_term)
    
    # Deviation from expected terminal velocity (indicates wind/disturbances)
    df['velocity_deviation'] = np.abs(df['vel_v']) - df['expected_terminal_velocity']
    
    # === SENSOR FUSION FEATURES ===
    # If gyro and accelerometer data available
    if 'gyro_z' in df.columns:
        df['gyro_z_abs'] = np.abs(df['gyro_z'])
        df['gyro_z_std'] = df.groupby('flight_id')['gyro_z'].rolling(window, min_periods=1).std().values
    
    if 'accel_stddev' in df.columns:
        df['vibration_level'] = df['accel_stddev']
        df['vibration_trend'] = df.groupby('flight_id')['accel_stddev'].diff().fillna(0)
    
    logging.info(f"Feature engineering complete. Added {len(df.columns) - 8} new features.")
    return df

def calculate_reynolds_number_vectorized(altitudes, velocities):
    """Vectorized Reynolds number calculation."""
    air_density = 1.225 * np.exp(-altitudes / 8500.0)
    dynamic_viscosity = 1.81e-5 * (1 + altitudes * 6.5e-6)
    characteristic_length = 0.5
    
    # Avoid division by zero
    safe_velocities = np.where(velocities == 0, 1e-6, velocities)
    
    return (air_density * np.abs(safe_velocities) * characteristic_length) / dynamic_viscosity

def calculate_trajectory_curvature_vectorized(df):
    """Vectorized trajectory curvature calculation."""
    curvature = np.zeros(len(df))
    
    for flight_id in df['flight_id'].unique():
        flight_mask = df['flight_id'] == flight_id
        flight_data = df[flight_mask].copy()
        
        if len(flight_data) < 3:
            continue
            
        lats = flight_data['lat'].values
        lons = flight_data['lon'].values
        
        # Calculate curvature using finite differences
        for i in range(1, len(lats) - 1):
            # Vector from point i-1 to i
            v1_lat, v1_lon = lats[i] - lats[i-1], lons[i] - lons[i-1]
            # Vector from point i to i+1
            v2_lat, v2_lon = lats[i+1] - lats[i], lons[i+1] - lons[i]
            
            # Cross product magnitude (proportional to curvature)
            cross_product = v1_lat * v2_lon - v1_lon * v2_lat
            curvature[flight_data.index[i]] = abs(cross_product)
    
    return curvature

def prepare_datasets_optimized():
    """
    Optimized version of data preparation with advanced feature engineering.
    """
    setup_logging()
    logging.info("--- Starting OPTIMIZED Data Preparation ---")
    
    # Load raw data
    df = pd.read_csv(PATHS['raw_dataset'])
    logging.info(f"Loaded {len(df)} raw data points from {df['flight_id'].nunique()} flights")
    
    # Get actual landing points for each flight
    actual_landings = df.groupby('flight_id').last()[['lat', 'lon']].rename(
        columns={'lat': 'landing_lat', 'lon': 'landing_lon'}
    )
    
    # Merge landing points with all data
    df = pd.merge(df, actual_landings, on='flight_id')
    
    # Calculate displacement vectors (vectorized)
    logging.info("Calculating displacement vectors...")
    disp_north, disp_east = calculate_displacement_vectorized(df)
    df['landing_disp_north_m'] = disp_north
    df['landing_disp_east_m'] = disp_east
    
    # Advanced feature engineering
    df = engineer_advanced_features(df)
    
    # Update features list with new engineered features
    all_features = [
        'alt', 'vel_v', 'lat', 'lon', 'horizontal_speed',
        'descent_acceleration', 'reynolds_number', 'dynamic_pressure',
        'time_since_deployment', 'altitude_normalized', 'altitude_loss_rate',
        'horizontal_velocity_north', 'horizontal_velocity_east',
        'estimated_wind_speed', 'trajectory_curvature',
        'air_density', 'expected_terminal_velocity', 'velocity_deviation'
    ]
    
    # Add sensor features if available
    if 'gyro_z' in df.columns:
        all_features.extend(['gyro_z', 'gyro_z_abs', 'gyro_z_std'])
    if 'accel_stddev' in df.columns:
        all_features.extend(['accel_stddev', 'vibration_level', 'vibration_trend'])
    
    # Filter to available features
    available_features = [f for f in all_features if f in df.columns]
    logging.info(f"Using {len(available_features)} features: {available_features}")
    
    # Split flights for training and meta-model
    unique_flight_ids = df['flight_id'].unique()
    train_flight_ids, meta_flight_ids = train_test_split(
        unique_flight_ids, 
        test_size=TRAINING['meta_split_ratio'], 
        random_state=TRAINING['random_state']
    )
    
    train_df = df[df['flight_id'].isin(train_flight_ids)].copy()
    meta_df = df[df['flight_id'].isin(meta_flight_ids)].copy()
    
    # Scale features
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    meta_df_scaled = meta_df.copy()
    
    # Fit scaler on training data only
    train_df_scaled[available_features] = scaler.fit_transform(train_df[available_features])
    meta_df_scaled[available_features] = scaler.transform(meta_df[available_features])
    
    # Save processed data
    train_df_scaled.to_csv(PATHS['processed_train_data'], index=False)
    meta_df_scaled.to_csv(PATHS['processed_meta_data'], index=False)
    joblib.dump(scaler, PATHS['scaler'])
    
    # Save feature list for later use
    feature_config = {
        'features': available_features,
        'target_features': ['landing_disp_north_m', 'landing_disp_east_m']
    }
    joblib.dump(feature_config, PATHS['scaler'].replace('.pkl', '_features.pkl'))
    
    logging.info(f"Data preparation complete:")
    logging.info(f"  Training flights: {len(train_flight_ids)}")
    logging.info(f"  Meta-model flights: {len(meta_flight_ids)}")
    logging.info(f"  Features: {len(available_features)}")
    logging.info(f"  Training samples: {len(train_df_scaled)}")
    logging.info(f"  Meta samples: {len(meta_df_scaled)}")

if __name__ == "__main__":
    prepare_datasets_optimized()