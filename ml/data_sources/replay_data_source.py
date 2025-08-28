# data_sources/replay_data_source.py
import pandas as pd
import time
import json
import logging

# --- THIS IS THE FIX ---
# Instead of importing the old variable, we import the new PATHS dictionary.
from config import PATHS

def get_telemetry_stream():
    """
    A generator that reads a pre-generated flight log CSV and replays it
    as if it were a live telemetry stream.
    """
    # Use the correct variable from the imported dictionary
    log_filename = PATHS['raw_dataset']
    
    logging.info(f"--- RUNNING IN REPLAY MODE (Reading from {log_filename}) ---")
    
    try:
        # Use the correct filename variable to open the file
        df = pd.read_csv(log_filename)
    except FileNotFoundError:
        logging.error(f"FATAL ERROR: Dataset file not found: '{log_filename}'")
        logging.error("  > Please run 'generate_rich_dataset.py' first to create the flight log.")
        return # Stop the generator

    # The rest of the file is perfect and does not need to change.
    for index, row in df.iterrows():
        telemetry_packet = row.to_dict()
        yield json.dumps(telemetry_packet)
        time.sleep(0.1) # Speed up replay slightly for faster testing
        
    logging.info("--- REPLAY FINISHED ---")