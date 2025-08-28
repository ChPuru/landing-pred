# data_sources/replay_data_source.py
import pandas as pd
import time
import json
from config_advanced import FLIGHT_LOG_FILENAME

def get_telemetry_stream():
    """
    A generator that reads a pre-generated flight log CSV and replays it
    as if it were a live telemetry stream.
    """
    print(f"--- RUNNING IN REPLAY MODE (Reading from {FLIGHT_LOG_FILENAME}) ---")
    
    try:
        df = pd.read_csv(FLIGHT_LOG_FILENAME)
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset file not found: '{FLIGHT_LOG_FILENAME}'")
        print("  > Please run 'generate_dataset.py' first to create the flight log.")
        return # Stop the generator

    # Iterate through each row of the flight data
    for index, row in df.iterrows():
        # Convert the row (which is a pandas Series) to a dictionary
        telemetry_packet = row.to_dict()
        
        # Yield the data as a JSON string to perfectly mimic the real hardware
        yield json.dumps(telemetry_packet)
        
        # Wait for one second to simulate a real-time feed
        time.sleep(1)
        
    print("--- REPLAY FINISHED ---")