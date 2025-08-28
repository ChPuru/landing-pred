# generate_rich_dataset.py
import pandas as pd
import random
import numpy as np
from config import SIMULATION_SETTINGS, FLIGHT_LOG_FILENAME
from prediction_engine import calculate_new_coords

def simulate_rich_flight(flight_id):
    """Simulates one flight with more realistic, dynamic data."""
    flight_data = []
    
    lat = SIMULATION_SETTINGS['start_lat'] + random.uniform(-0.1, 0.1)
    lon = SIMULATION_SETTINGS['start_lon'] + random.uniform(-0.1, 0.1)
    alt = float(SIMULATION_SETTINGS['start_alt_m'])
    
    # Store previous point to calculate speed
    prev_lat, prev_lon = lat, lon

    while alt > 0:
        # Calculate horizontal speed based on distance moved in the last second
        dist_moved = np.sqrt((lat - prev_lat)**2 + (lon - prev_lon)**2) * 111320 # Approx meters
        horiz_speed = dist_moved / 1.0 # Since timestep is 1s
        
        # Simulate some other interesting sensor data
        gyro_z = random.uniform(-30, 30) * (alt / SIMULATION_SETTINGS['start_alt_m']) # Spin faster at top
        accel_stddev = random.uniform(0.1, 1.5) # A measure of vibration/oscillation

        flight_data.append({
            "flight_id": flight_id,
            "lat": lat, "lon": lon, "alt": alt,
            "vel_v": -SIMULATION_SETTINGS['descent_rate_mps'],
            "horiz_speed": horiz_speed,
            "gyro_z": gyro_z,
            "accel_stddev": accel_stddev,
            "state": "DESCENT"
        })
        
        prev_lat, prev_lon = lat, lon
        
        # Update position with a slightly variable wind
        wind_speed = random.uniform(4.0, 8.0)
        wind_dir = random.uniform(260, 280)
        drift_dir = (wind_dir + 180) % 360
        lat, lon = calculate_new_coords(lat, lon, drift_dir, wind_speed)
        alt -= SIMULATION_SETTINGS['descent_rate_mps']
        
    return flight_data

def generate_master_log():
    """Generates a master log with rich, engineered features."""
    num_flights = SIMULATION_SETTINGS['num_flights_to_generate']
    print(f"--- Generating Rich Dataset with {num_flights} Flights ---")
    
    all_flights_data = []
    for i in range(num_flights):
        flight_path = simulate_rich_flight(i + 1)
        all_flights_data.extend(flight_path)
        
    df = pd.DataFrame(all_flights_data)
    df.to_csv(FLIGHT_LOG_FILENAME, index=False)
    print(f"Successfully created rich dataset: {FLIGHT_LOG_FILENAME}")

if __name__ == "__main__":
    generate_master_log()