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

def calculate_haversine_error(y_true, y_pred):
    """Calculates the landing error in meters for a set of predictions."""
    errors = [
        geodesic((true_lat, true_lon), (pred_lat, pred_lon)).meters
        for (true_lat, true_lon), (pred_lat, pred_lon) in zip(y_true, y_pred)
    ]
    return sum(errors) / len(errors)