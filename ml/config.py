# config.py

# --- 1. File Paths ---
# All file paths are defined in one place.
PATHS = {
    "raw_dataset": "master_flight_log.csv",
    "processed_train_data": "processed_train.csv",
    "processed_meta_data": "processed_meta.csv",
    "scaler": "scaler.pkl",
    "physics_model": "physics_engine_advanced.py", # For reference
    "xgboost_model": "ultimate_xgb_model.pkl",
    "lstm_model": "ultimate_lstm_model.keras", # Using the modern .keras format
    "meta_model": "ultimate_meta_model.pkl"
}

# --- 2. CanSat Physical Properties (for Advanced Physics Engine) ---
CANSAT_PHYSICS = {
    "mass_kg": 0.350,
    "parachute_area_m2": 0.5,
    "drag_coefficient": 0.8,
}

# --- 3. Layered Wind Model (for Advanced Physics Engine) ---
WIND_LAYERS = [
    (3000, 20.0, 290), (1500, 12.0, 280), (500,  5.0,  270),
]

# --- 4. Simulation & Data Generation Settings ---
SIMULATION = {
    "num_flights_to_generate": 100, # More data for better models
    "start_alt_m": 2000,
    "descent_rate_mps": 6.0,
    "start_lat": 40.7128,
    "start_lon": -74.0060,
    "physics_timestep_s": 1.0,
    "oscillation_mps": 0.2
}

# --- 5. ML Model Training Settings ---
TRAINING = {
    "features": ['alt', 'vel_v', 'lat', 'lon', 'horiz_speed', 'gyro_z', 'accel_stddev'],
    "labels": ['landing_lat', 'landing_lon'],
    "meta_split_ratio": 0.25, # 25% of flights reserved for meta-model training
    "lstm_sequence_length": 30,
    "random_state": 42
}

SIMULATION_SETTINGS = SIMULATION
FLIGHT_LOG_FILENAME = PATHS["raw_dataset"]

# Add paths for new models
PATHS["bayesian_meta_model"] = "bayesian_meta_model.pkl"
PATHS["wind_estimator"] = "wind_estimator.pkl"