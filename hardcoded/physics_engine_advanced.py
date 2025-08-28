# physics_engine_advanced.py
import math
import random
from prediction_engine import calculate_new_coords

# --- Atmospheric Model ---
def get_air_density(altitude_m):
    """
    Calculates air density (rho) using a simplified International Standard Atmosphere model.
    """
    # A good approximation for the troposphere
    return 1.225 * math.exp(-altitude_m / 8500.0)

def get_wind_for_altitude(altitude_m, wind_layers):
    """
    Finds the correct wind speed and direction for a given altitude from the layered model.
    """
    for upper_alt, speed, direction in wind_layers:
        if altitude_m >= upper_alt:
            return speed, direction
    # If below all layers, use the last one (or a default)
    return wind_layers[-1][1], wind_layers[-1][2]

# --- The Main Simulation Engine ---
def run_forward_simulation(current_telemetry, cansat_physics, wind_layers, timestep, oscillation):
    """
    Runs a high-speed, iterative simulation of the entire remaining descent.
    """
    # Initialize simulation state from the live telemetry
    sim_lat = current_telemetry['lat']
    sim_lon = current_telemetry['lon']
    sim_alt = current_telemetry['alt']

    while sim_alt > 0:
        # --- 1. Calculate Environment at Current Altitude ---
        rho = get_air_density(sim_alt)
        wind_speed, wind_dir = get_wind_for_altitude(sim_alt, wind_layers)
        
        # --- 2. Calculate Physics for this Timestep ---
        # Calculate terminal velocity (descent rate) based on air density
        # Drag Force = Gravitational Force -> 0.5 * rho * v^2 * Cd * A = m * g
        g = 9.81
        drag_term = 0.5 * rho * cansat_physics['drag_coefficient'] * cansat_physics['parachute_area_m2']
        if drag_term == 0: return None # Avoid division by zero
        
        terminal_velocity = math.sqrt((2 * cansat_physics['mass_kg'] * g) / drag_term)
        
        # --- 3. Update Position for this Timestep ---
        # a) Vertical movement
        alt_change = terminal_velocity * timestep
        sim_alt -= alt_change
        
        # b) Horizontal movement due to wind
        drift_distance = wind_speed * timestep
        drift_direction = (wind_dir + 180) % 360
        sim_lat, sim_lon = calculate_new_coords(sim_lat, sim_lon, drift_direction, drift_distance)
        
        # c) Horizontal movement due to parachute oscillation (random walk)
        osc_distance = random.uniform(-oscillation, oscillation) * timestep
        osc_direction = random.uniform(0, 360)
        sim_lat, sim_lon = calculate_new_coords(sim_lat, sim_lon, osc_direction, osc_distance)

    return {"pred_lat": sim_lat, "pred_lon": sim_lon}