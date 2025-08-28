# main_app_ultimate.py
import json
import joblib
import pandas as pd
import numpy as np
import logging
from collections import deque
from tensorflow.keras.models import load_model
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging
from physics_engine_advanced import run_forward_simulation
from data_sources.replay_data_source import get_telemetry_stream

def run_ultimate_predictor():
    setup_logging()
    logging.info("--- CanSat Ground Station: Ultimate Predictor (Professional Version) ---")
    
    try:
        scaler = joblib.load(PATHS['scaler'])
        xgb_model = joblib.load(PATHS['xgboost_model'])
        lstm_model = load_model(PATHS['lstm_model'], compile=False)
        meta_model = joblib.load(PATHS['meta_model'])
        logging.info("All models and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"FATAL ERROR loading models: {e}. Please run data_preparation.py and train_ensemble.py first.")
        return

    telemetry_stream = get_telemetry_stream()
    history = deque(maxlen=TRAINING['lstm_sequence_length'])

    for telemetry_string in telemetry_stream:
        try:
            telemetry_packet = json.loads(telemetry_string)
            history.append(telemetry_packet)
            
            if telemetry_packet.get('vel_v', 0) < 0 and len(history) == TRAINING['lstm_sequence_length']:
                # --- 1. Prepare Live Data ---
                # Create a DataFrame for the single, most recent data point for XGBoost
                live_df = pd.DataFrame([telemetry_packet])
                live_df_scaled = scaler.transform(live_df[TRAINING['features']])
                
                # Create a DataFrame for the full history for the LSTM
                history_df = pd.DataFrame(list(history))
                history_df_scaled = scaler.transform(history_df[TRAINING['features']])

                # --- 2. Base Ensemble Predictions ---
                phys_pred = run_forward_simulation(telemetry_packet, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps'])
                
                # The XGBoost model expects a DataFrame, but we can use the scaled numpy array
                xgb_pred = xgb_model.predict(live_df_scaled)[0]
                
                # --- THIS IS THE CORRECTED LINE ---
                # history_df_scaled is already a NumPy array, so we remove .values
                lstm_input = history_df_scaled.reshape(1, TRAINING['lstm_sequence_length'], len(TRAINING['features']))
                lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0]

                # --- 3. Meta-Model Fusion ---
                if phys_pred:
                    meta_input = np.array([
                        phys_pred['pred_lat'], phys_pred['pred_lon'],
                        xgb_pred[0], xgb_pred[1],
                        lstm_pred[0], lstm_pred[1]
                    ]).reshape(1, -1)
                    
                    fused_pred = meta_model.predict(meta_input)[0]
                    
                    # --- 4. Display ---
                    logging.info(f"Altitude: {telemetry_packet['alt']:.1f}m | Fused Prediction: Lat={fused_pred[0]:.4f}, Lon={fused_pred[1]:.4f}")
                else:
                    logging.warning("Physics engine failed to predict. Skipping fusion.")

        except (json.JSONDecodeError, KeyError):
            logging.warning(f"Malformed data packet received: {telemetry_string}")
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user.")
            break

if __name__ == "__main__":
    run_ultimate_predictor()