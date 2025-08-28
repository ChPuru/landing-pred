# utils.py
import logging
from geopy.distance import geodesic

def setup_logging():
    """Sets up a basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def calculate_geodesic_error(y_true, y_pred):
    """Calculates the landing error in meters using the precise geodesic formula."""
    errors = [
        geodesic((true_lat, true_lon), (pred_lat, pred_lon)).meters
        for (true_lat, true_lon), (pred_lat, pred_lon) in zip(y_true, y_pred)
    ]
    return sum(errors) / len(errors)

# --- THIS IS THE MOVED FUNCTION ---
def calculate_displacement(row):
    """
    Calculates the North (lat) and East (lon) displacement in meters.
    This is a reusable utility function.
    """
    p_a = (row['lat'], row['lon'])
    p_b = (row['landing_lat'], row['landing_lon'])
    
    p_lat_only = (row['landing_lat'], row['lon'])
    disp_lat_m = geodesic(p_a, p_lat_only).meters
    if row['landing_lat'] < row['lat']:
        disp_lat_m *= -1

    p_lon_only = (row['lat'], row['landing_lon'])
    disp_lon_m = geodesic(p_a, p_lon_only).meters
    if row['landing_lon'] < row['lon']:
        disp_lon_m *= -1
        
    return disp_lat_m, disp_lon_m