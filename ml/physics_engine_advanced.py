# physics_engine_advanced.py
import math
import random
import numpy as np
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
    FIXED: Properly handles altitude-based layer selection
    """
    # Sort layers by altitude (highest first) to ensure correct ordering
    sorted_layers = sorted(wind_layers, key=lambda x: x[0], reverse=True)
    
    # Find the appropriate layer for this altitude
    for upper_alt, speed, direction in sorted_layers:
        if altitude_m <= upper_alt:
            return speed, direction
    
    # If altitude is below all layers, use the lowest layer
    return sorted_layers[-1][1], sorted_layers[-1][2]

def calculate_reynolds_number(altitude, velocity):
    """Calculate Reynolds number for aerodynamic characterization."""
    air_density = get_air_density(altitude)
    dynamic_viscosity = 1.81e-5 * (1 + altitude * 6.5e-6)  # Temperature-adjusted viscosity
    characteristic_length = 0.5  # Parachute diameter in meters
    
    if velocity == 0:
        return 0
    
    return (air_density * abs(velocity) * characteristic_length) / dynamic_viscosity

def calculate_drag_coefficient_dynamic(reynolds_number, base_cd=0.8):
    """
    Calculate dynamic drag coefficient based on Reynolds number.
    More realistic than constant Cd.
    """
    # Simplified model for parachute drag variation
    if reynolds_number < 1e4:
        return base_cd * 1.2  # Higher drag at low Re
    elif reynolds_number > 1e6:
        return base_cd * 0.9  # Lower drag at high Re
    else:
        return base_cd

# --- The Main Simulation Engine ---
def run_forward_simulation(current_telemetry, cansat_physics, wind_layers, timestep, oscillation):
    """
    Runs a high-speed, iterative simulation of the entire remaining descent.
    ENHANCED: More realistic physics with dynamic drag and better numerical stability.
    """
    # Initialize simulation state from the live telemetry
    sim_lat = current_telemetry['lat']
    sim_lon = current_telemetry['lon']
    sim_alt = current_telemetry['alt']
    sim_velocity = abs(current_telemetry.get('vel_v', 6.0))  # Ensure positive descent rate
    
    # Trajectory tracking for stability analysis
    trajectory_points = [(sim_lat, sim_lon, sim_alt)]
    
    while sim_alt > 0:
        # --- 1. Calculate Environment at Current Altitude ---
        rho = get_air_density(sim_alt)
        wind_speed, wind_dir = get_wind_for_altitude(sim_alt, wind_layers)
        
        # --- 2. Enhanced Physics for this Timestep ---
        # Calculate Reynolds number for dynamic drag
        reynolds = calculate_reynolds_number(sim_alt, sim_velocity)
        drag_coefficient = calculate_drag_coefficient_dynamic(reynolds, cansat_physics['drag_coefficient'])
        
        # Calculate terminal velocity with updated drag coefficient
        g = 9.81
        drag_term = 0.5 * rho * drag_coefficient * cansat_physics['parachute_area_m2']
        
        if drag_term <= 0:
            # Fallback to prevent division by zero
            terminal_velocity = 6.0
        else:
            terminal_velocity = math.sqrt((cansat_physics['mass_kg'] * g) / drag_term)
        
        # Smooth velocity transition (prevents unrealistic jumps)
        velocity_change_rate = 0.1  # How quickly velocity can change
        target_velocity = terminal_velocity
        sim_velocity += (target_velocity - sim_velocity) * velocity_change_rate * timestep
        
        # Ensure minimum descent rate
        sim_velocity = max(sim_velocity, 1.0)
        
        # --- 3. Update Position for this Timestep ---
        # a) Vertical movement
        alt_change = sim_velocity * timestep
        sim_alt -= alt_change
        
        # Prevent going below ground
        if sim_alt < 0:
            sim_alt = 0
            break
            
        # b) Horizontal movement due to wind (with altitude-dependent scaling)
        altitude_factor = max(0.1, sim_alt / 2000.0)  # Wind effect reduces near ground
        effective_wind_speed = wind_speed * altitude_factor
        
        drift_distance = effective_wind_speed * timestep
        drift_direction = (wind_dir + 180) % 360  # Wind direction to drift direction
        
        sim_lat, sim_lon = calculate_new_coords(sim_lat, sim_lon, drift_direction, drift_distance)
        
        # c) Parachute oscillation with realistic damping
        if oscillation > 0:
            # Oscillation decreases with altitude (more stable near ground)
            effective_oscillation = oscillation * altitude_factor
            osc_distance = random.uniform(-effective_oscillation, effective_oscillation) * timestep
            osc_direction = random.uniform(0, 360)
            sim_lat, sim_lon = calculate_new_coords(sim_lat, sim_lon, osc_direction, abs(osc_distance))
        
        # Store trajectory point
        trajectory_points.append((sim_lat, sim_lon, sim_alt))
        
        # Prevent infinite loops
        if len(trajectory_points) > 10000:  # Max 10000 seconds of simulation
            break

    # Calculate trajectory stability score
    stability_score = calculate_trajectory_stability(trajectory_points)
    
    return {
        "pred_lat": sim_lat, 
        "pred_lon": sim_lon,
        "stability_score": stability_score,
        "trajectory_points": trajectory_points[-10:],  # Last 10 points for analysis
        "final_velocity": sim_velocity
    }

def calculate_trajectory_stability(trajectory_points):
    """
    Calculate a stability score based on trajectory smoothness.
    Higher scores indicate more stable/predictable trajectories.
    """
    if len(trajectory_points) < 3:
        return 0.5  # Default moderate stability
    
    # Calculate trajectory curvature
    curvatures = []
    for i in range(1, len(trajectory_points) - 1):
        p1 = trajectory_points[i-1]
        p2 = trajectory_points[i]
        p3 = trajectory_points[i+1]
        
        # Simple curvature approximation
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
        
        cross_product = dx1 * dy2 - dy1 * dx2
        curvatures.append(abs(cross_product))
    
    if not curvatures:
        return 0.5
    
    # Stability is inverse of average curvature (normalized)
    avg_curvature = np.mean(curvatures)
    stability = 1.0 / (1.0 + avg_curvature * 1000)  # Scale factor
    
    return max(0.1, min(1.0, stability))

# Enhanced validation function
def validate_simulation_inputs(current_telemetry, cansat_physics, wind_layers):
    """Validate inputs before running simulation to prevent errors."""
    required_fields = ['lat', 'lon', 'alt']
    
    for field in required_fields:
        if field not in current_telemetry:
            return False, f"Missing required field: {field}"
    
    if current_telemetry['alt'] <= 0:
        return False, "Altitude must be positive"
    
    if not (-90 <= current_telemetry['lat'] <= 90):
        return False, "Invalid latitude"
    
    if not (-180 <= current_telemetry['lon'] <= 180):
        return False, "Invalid longitude"
    
    # Validate physics parameters
    required_physics = ['mass_kg', 'parachute_area_m2', 'drag_coefficient']
    for param in required_physics:
        if param not in cansat_physics or cansat_physics[param] <= 0:
            return False, f"Invalid physics parameter: {param}"
    
    return True, "Valid"