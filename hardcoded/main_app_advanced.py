# main_app_advanced.py
import json
from config_advanced import CANSAT_PHYSICS, WIND_LAYERS, PHYSICS_ENGINE_SETTINGS
from physics_engine_advanced import run_forward_simulation
from data_sources.replay_data_source import get_telemetry_stream
def run_advanced_ground_station():
    """
    Main application that uses the iterative forward simulation engine.
    """
    print("--- CanSat Ground Station: Advanced Physics Predictor v1.5 ---")
    
    # Initialize the data source (replay or real)
    telemetry_stream = get_telemetry_stream()

    # Main Processing Loop
    for telemetry_string in telemetry_stream:
        try:
            telemetry_packet = json.loads(telemetry_string)
            
            if telemetry_packet['vel_v'] < 0: # Only predict when descending
                
                # --- Run the full forward simulation ---
                prediction = run_forward_simulation(
                    telemetry_packet,
                    CANSAT_PHYSICS,
                    WIND_LAYERS,
                    PHYSICS_ENGINE_SETTINGS['timestep_s'],
                    PHYSICS_ENGINE_SETTINGS['oscillation_mps']
                )
                
                if prediction:
                    print("\n" + "="*60)
                    print(f"Flight ID: {telemetry_packet.get('flight_id', 'N/A')} | Altitude: {telemetry_packet['alt']:.1f}m")
                    print(f"  Advanced Physics Prediction: Lat={prediction['pred_lat']:.4f}, Lon={prediction['pred_lon']:.4f}")

        except (json.JSONDecodeError, KeyError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    run_advanced_ground_station()