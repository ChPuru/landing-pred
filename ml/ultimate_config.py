# ultimate_config.py

# --- File Naming ---
MASTER_LOG_FILENAME = "master_flight_log.csv"
RF_MODEL_FILENAME = "ultimate_rf_model.pkl"
LSTM_MODEL_FILENAME = "ultimate_lstm_model.keras"
LSTM_MODEL_FILENAME = "ultimate_lstm_model.h5"
META_MODEL_FILENAME = "ultimate_meta_model.pkl"

# --- CanSat Physical Properties (for Advanced Physics Engine) ---
CANSAT_PHYSICS = {
    "mass_kg": 0.350,
    "parachute_area_m2": 0.5,
    "drag_coefficient": 0.8,
}

# --- Layered Wind Model (for Advanced Physics Engine) ---
# Format: (Upper_Altitude_Limit_meters, Speed_mps, Direction_FROM_degrees)
WIND_LAYERS = [
    (3000, 20.0, 290),
    (1500, 12.0, 280),
    (500,  5.0,  270),
]

# --- Simulation Parameters ---
# For the data generator
SIMULATION_SETTINGS = {
    "num_flights_to_generate": 50,
    "start_alt_m": 2000,
    "descent_rate_mps": 6.0,
    "start_lat": 40.7128,
    "start_lon": -74.0060
}
# For the advanced physics engine
PHYSICS_ENGINE_SETTINGS = {
    "timestep_s": 1.0,
    "oscillation_mps": 0.2
}

# --- ML Model Training Settings ---
LSTM_SEQUENCE_LENGTH = 30 # Use the last 30 seconds of data

# --- Real-Time Prediction Settings ---
# How much to trust each model in the FUSION stage (must sum to 1.0)
# This is now superseded by the meta-model, but good for reference.
FUSION_WEIGHTS = {
    "physics": 0.3,
    "ml": 0.7
}
# How much uncertainty to add in the PROBABILISTIC stage
MONTE_CARLO_SETTINGS = {
    "num_simulations": 1000,
    "uncertainty_factor": 1.2
}