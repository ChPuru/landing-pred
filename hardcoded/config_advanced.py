# config_advanced.py

# --- File Naming & Dataset Generation ---
# These variables are needed by generate_dataset.py and replay_data_source.py
FLIGHT_LOG_FILENAME = "advanced_physics_flight_log.csv"
SIMULATION_SETTINGS = {
    "num_flights_to_generate": 50,
    "start_alt_m": 2000,
    "descent_rate_mps": 6.0,
    "start_lat": 40.7128,
    "start_lon": -74.0060
}

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

# --- Simulation Engine Parameters (for Advanced Physics Engine) ---
PHYSICS_ENGINE_SETTINGS = {
    "timestep_s": 1.0,
    "oscillation_mps": 0.2
}

# --- Serial Port (for real hardware) ---
# Used by real_hardware_source.py if you switch to it
SERIAL_PORT = 'COM4'
BAUD_RATE = 9600