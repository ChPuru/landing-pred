# generate_dataset.py
import pandas as pd
import random
from config_advanced import SIMULATION_SETTINGS, FLIGHT_LOG_FILENAME
from prediction_engine import calculate_new_coords

def simulate_single_flight(flight_id):
    """Simulates one complete flight with random variations."""
    flight_data = []
    
    # Start each flight at a slightly different location
    lat = SIMULATION_SETTINGS['start_lat'] + random.uniform(-0.1, 0.1)
    lon = SIMULATION_SETTINGS['start_lon'] + random.uniform(-0.1, 0.1)
    alt = float(SIMULATION_SETTINGS['start_alt_m'])
    
    # Each flight has slightly different (but constant) wind
    wind_speed = random.uniform(3.0, 10.0)
    wind_dir = random.uniform(0, 360)
    drift_dir = (wind_dir + 180) % 360

    while alt > 0:
        flight_data.append({
            "flight_id": flight_id,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "vel_v": -SIMULATION_SETTINGS['descent_rate_mps'],
            "state": "DESCENT"
        })
        
        # Update position based on this flight's unique wind
        lat, lon = calculate_new_coords(lat, lon, drift_dir, wind_speed)
        alt -= SIMULATION_SETTINGS['descent_rate_mps']
        
    return flight_data

def generate_master_log():
    """Generates and saves a master log of many simulated flights."""
    num_flights = SIMULATION_SETTINGS['num_flights_to_generate']
    print(f"--- Generating Master Dataset with {num_flights} Flights ---")
    
    all_flights_data = []
    for i in range(num_flights):
        print(f"Simulating flight {i+1}/{num_flights}...")
        flight_path = simulate_single_flight(i + 1)
        all_flights_data.extend(flight_path)
        
    # Use pandas to easily handle the data and save to CSV
    df = pd.DataFrame(all_flights_data)
    df.to_csv(FLIGHT_LOG_FILENAME, index=False)
    
    print("\n" + "="*40)
    print(f"Successfully created master dataset!")
    print(f"  > File: {FLIGHT_LOG_FILENAME}")
    print(f"  > Total Data Points: {len(df)}")
    print(f"  > Number of Flights: {num_flights}")
    print("="*40)

if __name__ == "__main__":
    generate_master_log()