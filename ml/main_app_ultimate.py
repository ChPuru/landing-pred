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

# NEW: Import enhanced components
try:
    from telemetry_buffer import TelemetryBuffer, FlightStateTracker
    from models.bayesian_ensemble import BayesianMetaModel
    ENHANCED_COMPONENTS = True
except ImportError:
    print("Enhanced components not available - using basic version")
    ENHANCED_COMPONENTS = False

def run_ultimate_predictor():
    setup_logging()
    logging.info("--- CanSat Ultimate Prediction System ---")
    
    # Load models and scaler
    try:
        scaler = joblib.load(PATHS['scaler'])
        xgb_model = joblib.load(PATHS['xgboost_model'])
        lstm_model = load_model(PATHS['lstm_model'], compile=False)
        
        # Try to load enhanced features
        try:
            feature_config = joblib.load(PATHS['scaler'].replace('.pkl', '_features.pkl'))
            features = feature_config['features']
            logging.info(f"Using {len(features)} enhanced features")
        except FileNotFoundError:
            features = TRAINING['features']
            logging.info(f"Using {len(features)} default features")
        
        # Load meta-model (try Bayesian first, fallback to traditional)
        meta_model = None
        try:
            bayesian_meta = BayesianMetaModel()
            bayesian_meta.load(PATHS['bayesian_meta_model'])
            meta_model = bayesian_meta
            logging.info("Bayesian meta-model loaded")
        except:
            meta_model = joblib.load(PATHS['meta_model'])
            logging.info("Traditional meta-model loaded")
        
        logging.info("All models loaded successfully.")
        
    except Exception as e:
        logging.error(f"FATAL ERROR loading models: {e}")
        return

    # Initialize enhanced components if available
    if ENHANCED_COMPONENTS:
        telemetry_buffer = TelemetryBuffer(max_size=TRAINING['lstm_sequence_length'])
        flight_tracker = FlightStateTracker()
        logging.info("Enhanced real-time processing enabled")
    else:
        # Fallback to basic approach
        history = deque(maxlen=TRAINING['lstm_sequence_length'])

    # Main prediction loop
    telemetry_stream = get_telemetry_stream()
    prediction_count = 0
    
    for telemetry_string in telemetry_stream:
        try:
            telemetry_packet = json.loads(telemetry_string)
            
            if ENHANCED_COMPONENTS:
                # === ENHANCED PROCESSING ===
                
                # Update flight tracking
                flight_phase = flight_tracker.update(telemetry_packet)
                
                # Add to buffer
                telemetry_buffer.add_sample(telemetry_packet)
                
                # Check if we should predict
                if not flight_tracker.should_predict():
                    continue
                
                # Get prediction-ready data
                sequence, engineered_features = telemetry_buffer.get_prediction_ready_data(scaler)
                
                if sequence is None:
                    continue
                
                # Get latest sample for physics
                current_state = telemetry_buffer.get_latest_sample()
                
            else:
                # === BASIC PROCESSING (Fallback) ===
                history.append(telemetry_packet)
                
                if (telemetry_packet.get('vel_v', 0) >= 0 or 
                    len(history) < TRAINING['lstm_sequence_length']):
                    continue
                
                # Prepare data
                current_state = telemetry_packet
                history_df = pd.DataFrame(list(history))
                
                # Handle missing features gracefully
                available_features = [f for f in features if f in history_df.columns]
                sequence = scaler.transform(history_df[available_features])
            
            # === MAKE PREDICTIONS ===
            
            # 1. Physics prediction
            phys_pred = run_forward_simulation(
                current_state, CANSAT_PHYSICS, WIND_LAYERS,
                SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps']
            )
            
            # 2. XGBoost prediction
            if ENHANCED_COMPONENTS:
                live_df = pd.DataFrame([current_state])
                available_features = [f for f in features if f in live_df.columns]
                live_scaled = scaler.transform(live_df[available_features])
            else:
                live_scaled = sequence[-1:] # Use last sequence element
            
            xgb_pred = xgb_model.predict(live_scaled)[0]
            
            # 3. LSTM prediction
            lstm_input = sequence.reshape(1, TRAINING['lstm_sequence_length'], -1)
            lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0]
            
            # === FUSION ===
            
            if phys_pred:
                # Convert physics prediction to displacement
                from data_preparation import calculate_displacement
                test_row = pd.Series({
                    'lat': current_state['lat'],
                    'lon': current_state['lon'], 
                    'pred_lat': phys_pred['pred_lat'],
                    'pred_lon': phys_pred['pred_lon']
                })
                
                phys_disp_n, phys_disp_e = calculate_displacement(test_row)
                
                if hasattr(meta_model, 'predict_with_uncertainty'):
                    # Bayesian meta-model
                    base_predictions = np.array([[
                        phys_disp_n, phys_disp_e,
                        xgb_pred[0], xgb_pred[1],
                        lstm_pred[0], lstm_pred[1]
                    ]])
                    
                    result = meta_model.predict_with_uncertainty(base_predictions)
                    fused_pred = result['prediction'][0]
                    uncertainty = result['total_uncertainty'][0]
                    
                    logging.info(f"Alt: {current_state['alt']:.1f}m | "
                               f"Pred: [{fused_pred[0]:.1f}m N, {fused_pred[1]:.1f}m E] | "
                               f"Uncertainty: ±{uncertainty[0]:.1f}m")
                    
                else:
                    # Traditional meta-model
                    meta_input_df = pd.DataFrame([[
                        phys_disp_n, phys_disp_e,
                        xgb_pred[0], xgb_pred[1],
                        lstm_pred[0], lstm_pred[1]
                    ]], columns=['phys_pred_north', 'phys_pred_east',
                               'xgb_pred_north', 'xgb_pred_east', 
                               'lstm_pred_north', 'lstm_pred_east'])
                    
                    fused_pred = meta_model.predict(meta_input_df)[0]
                    
                    logging.info(f"Alt: {current_state['alt']:.1f}m | "
                               f"Fused Prediction: [{fused_pred[0]:.1f}m N, {fused_pred[1]:.1f}m E]")
                
                # Update tracking if enhanced components available
                if ENHANCED_COMPONENTS:
                    flight_tracker.update_prediction_confidence({
                        'prediction': fused_pred,
                        'stability_score': phys_pred.get('stability_score', 0.5)
                    })
                
                prediction_count += 1
                
            else:
                logging.warning("Physics prediction failed - using model average")
                avg_pred = (xgb_pred + lstm_pred) / 2
                logging.info(f"Alt: {current_state['alt']:.1f}m | "
                           f"Average Prediction: [{avg_pred[0]:.1f}m N, {avg_pred[1]:.1f}m E]")

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Malformed data: {e}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user.")
            break
    
    logging.info(f"Session complete. Made {prediction_count} predictions.")

if __name__ == "__main__":
    run_ultimate_predictor()