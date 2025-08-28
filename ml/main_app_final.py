# main_app_final.py
import json
import joblib
import pandas as pd
import numpy as np
import logging
from collections import deque
from tensorflow.keras.models import load_model
from geopy.point import Point
from geopy.distance import geodesic

# --- Import all our custom modules ---
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging, calculate_displacement
from physics_engine_advanced import run_forward_simulation
from data_sources.replay_data_source import get_telemetry_stream
# We no longer need the BayesianMetaModel class, as we are using a direct XGBoost model.

def project_from_displacement(lat, lon, disp_north_m, disp_east_m):
    """Converts a North/East displacement back to a final GPS coordinate."""
    start_point = Point(latitude=lat, longitude=lon)
    intermediate_point = geodesic(meters=disp_north_m).destination(start_point, bearing=0)
    final_point = geodesic(meters=disp_east_m).destination(intermediate_point, bearing=90)
    return final_point.latitude, final_point.longitude

def run_ultimate_predictor():
    setup_logging()
    logging.info("--- CanSat Ground Station: Professional Ensemble Predictor (FINAL) ---")
    
    # 1. Load all necessary artifacts
    try:
        scaler = joblib.load(PATHS['scaler'])
        xgb_model = joblib.load(PATHS['xgboost_model'])
        lstm_model = load_model(PATHS['lstm_model'], compile=False)
        
        # --- THIS IS THE FINAL FIX ---
        # We now load the XGBoost meta-model directly, just like the others.
        meta_model = joblib.load(PATHS['meta_model'])
        
        # Get feature names for validation
        base_model_features = xgb_model.feature_names_in_
        meta_model_features = meta_model.feature_names_in_
        
        logging.info("All models and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL ERROR loading models: {e}. Please run all training scripts first.")
        return

    # 2. Initialize real-time components
    telemetry_stream = get_telemetry_stream()
    history = deque(maxlen=TRAINING['lstm_sequence_length'])

    # 3. Main Real-Time Processing Loop
    for telemetry_string in telemetry_stream:
        try:
            telemetry_packet = json.loads(telemetry_string)
            history.append(telemetry_packet)
            
            if telemetry_packet.get('vel_v', 0) < 0 and len(history) == TRAINING['lstm_sequence_length']:
                # --- Prepare Live Data ---
                live_df = pd.DataFrame([telemetry_packet])
                live_df_scaled = scaler.transform(live_df[base_model_features])
                
                history_df = pd.DataFrame(list(history))
                history_df_scaled = scaler.transform(history_df[base_model_features])

                # --- Base Model Predictions ---
                phys_pred_coords = run_forward_simulation(telemetry_packet, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps'])
                xgb_pred_disp = xgb_model.predict(live_df_scaled)[0]
                
                lstm_input = history_df_scaled.reshape(1, TRAINING['lstm_sequence_length'], len(base_model_features))
                lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0]
                
                if phys_pred_coords:
                    phys_disp_n, phys_disp_e = calculate_displacement({'lat': telemetry_packet['lat'], 'lon': telemetry_packet['lon'], 'landing_lat': phys_pred_coords['pred_lat'], 'landing_lon': phys_pred_coords['pred_lon']})
                    
                    # --- Meta-Model Fusion ---
                    meta_input_df = pd.DataFrame(
                        [[
                            phys_disp_n, phys_disp_e,
                            xgb_pred_disp[0], xgb_pred_disp[1],
                            lstm_pred[0], lstm_pred[1]
                        ]],
                        columns=meta_model_features
                    )
                    
                    fused_disp = meta_model.predict(meta_input_df)[0]
                    
                    # --- Convert to Final Coords & Display ---
                    final_lat, final_lon = project_from_displacement(telemetry_packet['lat'], telemetry_packet['lon'], fused_disp[0], fused_disp[1])
                    logging.info(f"Altitude: {telemetry_packet['alt']:.1f}m | Final Prediction: Lat={final_lat:.4f}, Lon={final_lon:.4f}")
        except (json.JSONDecodeError, KeyError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    run_ultimate_predictor()