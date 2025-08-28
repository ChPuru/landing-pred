# prediction_engine.py
import math

# Physical constant for Earth's radius in meters
EARTH_RADIUS_M = 6378137.0

def calculate_new_coords(lat_deg, lon_deg, bearing_deg, distance_m):
    """
    Calculates a new GPS coordinate from a starting point, a direction (bearing),
    and a distance. This uses spherical trigonometry (Haversine formula principles).

    Args:
        lat_deg (float): Starting latitude in decimal degrees.
        lon_deg (float): Starting longitude in decimal degrees.
        bearing_deg (float): Bearing in degrees (0=North, 90=East, 180=South, 270=West).
        distance_m (float): Distance to travel in meters.

    Returns:
        A tuple containing the new (latitude, longitude) in decimal degrees.
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)
    
    angular_distance = distance_m / EARTH_RADIUS_M

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(angular_distance) +
                            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad))

    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
                                      math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad))

    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def predict_landing_point(current_telemetry, wind_info):
    """
    Predicts the landing point using a simple ballistic physics model with constant wind.
    
    Args:
        current_telemetry (dict): A dictionary containing 'lat', 'lon', 'alt', 'vel_v'.
        wind_info (dict): A dictionary containing 'speed_mps' and 'direction_deg'.
        
    Returns:
        A dictionary with the predicted coordinates and time, or None if prediction is not possible.
    """
    # --- Input Validation ---
    current_alt = current_telemetry.get('alt', 0.0)
    descent_rate = current_telemetry.get('vel_v', 0.0)

    if descent_rate >= 0:
        # We are not descending, so we can't predict a landing time.
        return None
    
    # Ensure we have a positive descent rate to avoid division by zero
    descent_rate = abs(descent_rate)
    if descent_rate < 0.1:
        return None

    # --- Core Calculation ---
    # 1. Calculate time remaining until landing (time = distance / speed)
    time_to_land_sec = current_alt / descent_rate

    # 2. Calculate how far the wind will push the object in that time
    drift_distance_m = wind_info['speed_mps'] * time_to_land_sec

    # 3. Calculate the direction of the drift.
    #    Wind direction is where it comes FROM. Drift is where it goes TO.
    drift_direction_deg = (wind_info['direction_deg'] + 180) % 360

    # 4. Calculate the final landing coordinates by projecting from the current point
    predicted_lat, predicted_lon = calculate_new_coords(
        current_telemetry['lat'],
        current_telemetry['lon'],
        drift_direction_deg,
        drift_distance_m
    )

    return {
        "pred_lat": predicted_lat,
        "pred_lon": predicted_lon,
        "time_to_land_sec": time_to_land_sec
    }