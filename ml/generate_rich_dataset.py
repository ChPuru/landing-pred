# generate_rich_dataset.py
import pandas as pd
import random
import numpy as np
from config import SIMULATION, PATHS  # FIXED IMPORT
from prediction_engine import calculate_new_coords

def simulate_rich_flight(flight_id):
    """Simulates one flight with more realistic, dynamic data."""
    flight_data = []
    
    lat = SIMULATION['start_lat'] + random.uniform(-0.1, 0.1)  # FIXED
    lon = SIMULATION['start_lon'] + random.uniform(-0.1, 0.1)  # FIXED
    alt = float(SIMULATION['start_alt_m'])  # FIXED
    
    prev_lat, prev_lon = lat, lon

    while alt > 0:
        dist_moved = np.sqrt((lat - prev_lat)**2 + (lon - prev_lon)**2) * 111320
        horiz_speed = dist_moved / 1.0
        
        gyro_z = random.uniform(-30, 30) * (alt / SIMULATION['start_alt_m'])  # FIXED
        accel_stddev = random.uniform(0.1, 1.5)

        flight_data.append({
            "flight_id": flight_id,
            "lat": lat, "lon": lon, "alt": alt,
            "vel_v": -SIMULATION['descent_rate_mps'],  # FIXED
            "horiz_speed": horiz_speed,
            "gyro_z": gyro_z,
            "accel_stddev": accel_stddev,
            "state": "DESCENT"
        })
        
        prev_lat, prev_lon = lat, lon
        
        wind_speed = random.uniform(4.0, 8.0)
        wind_dir = random.uniform(260, 280)
        drift_dir = (wind_dir + 180) % 360
        lat, lon = calculate_new_coords(lat, lon, drift_dir, wind_speed)
        alt -= SIMULATION['descent_rate_mps']  # FIXED
        
    return flight_data

def generate_master_log():
    """Generates a master log with rich, engineered features."""
    num_flights = SIMULATION['num_flights_to_generate']  # FIXED
    print(f"--- Generating Rich Dataset with {num_flights} Flights ---")
    
    all_flights_data = []
    for i in range(num_flights):
        flight_path = simulate_rich_flight(i + 1)
        all_flights_data.extend(flight_path)
        
    df = pd.DataFrame(all_flights_data)
    df.to_csv(PATHS['raw_dataset'], index=False)  # FIXED
    print(f"Successfully created rich dataset: {PATHS['raw_dataset']}")
    
if __name__ == "__main__":
    generate_master_log()