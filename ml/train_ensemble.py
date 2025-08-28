import pandas as pd
import numpy as np
import joblib
import logging
import xgboost as xgb
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging
from physics_engine_advanced import run_forward_simulation

# NEW: Import Bayesian components
try:
    from models.bayesian_ensemble import BayesianMetaModel
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("Bayesian ensemble not available - using traditional meta-model")
    BAYESIAN_AVAILABLE = False

def create_sequences(data, features, labels=None):
    """Creates sequences for LSTM."""
    X, y = [], []
    has_labels = labels is not None and all(col in data.columns for col in labels)
    for i in range(len(data) - TRAINING['lstm_sequence_length']):
        X.append(data[features].iloc[i:(i + TRAINING['lstm_sequence_length'])].values)
        if has_labels:
            y.append(data[labels].iloc[i + TRAINING['lstm_sequence_length'] - 1].values)
    return (np.array(X), np.array(y)) if has_labels else np.array(X)

def calculate_displacement(row):
    """Calculate displacement for meta-model training."""
    from geopy.distance import geodesic
    
    p_a = (row['lat'], row['lon'])
    p_landing = (row['landing_lat'] if 'landing_lat' in row else row['pred_lat'], 
                 row['landing_lon'] if 'landing_lon' in row else row['pred_lon'])
    
    # North displacement
    p_north = (p_landing[0], p_a[1])
    disp_north = geodesic(p_a, p_north).meters
    if p_landing[0] < p_a[0]:
        disp_north *= -1
    
    # East displacement  
    p_east = (p_a[0], p_landing[1])
    disp_east = geodesic(p_a, p_east).meters
    if p_landing[1] < p_a[1]:
        disp_east *= -1
    
    return disp_north, disp_east

def train_all_models():
    setup_logging()
    logging.info("--- Training Enhanced Ensemble System ---")
    
    # Load processed data
    train_df = pd.read_csv(PATHS['processed_train_data'])
    meta_df = pd.read_csv(PATHS['processed_meta_data'])
    
    # NEW: Try to get features from optimized data preparation
    try:
        feature_config = joblib.load(PATHS['scaler'].replace('.pkl', '_features.pkl'))
        features = feature_config['features']
        logging.info(f"Using {len(features)} optimized features")
    except FileNotFoundError:
        features = TRAINING['features']
        logging.info(f"Using {len(features)} default features")
    
    labels = ['landing_disp_north_m', 'landing_disp_east_m']

    # === 1. Train Final Base Models ===
    logging.info("Training base models...")
    
    # XGBoost with enhanced parameters
    final_xgb_model = xgb.XGBRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=TRAINING['random_state']
    )
    final_xgb_model.fit(train_df[features], train_df[labels])
    joblib.dump(final_xgb_model, PATHS['xgboost_model'])
    logging.info("XGBoost model saved.")

    # Enhanced LSTM Model
    X_train_lstm, y_train_lstm = create_sequences(train_df, features, labels)
    
    final_lstm_model = Sequential([
        LayerNormalization(input_shape=(TRAINING['lstm_sequence_length'], len(features))),
        LSTM(128, activation='tanh', return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
        Dropout(0.2),
        LSTM(64, activation='tanh', dropout=0.1, recurrent_dropout=0.1),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(2)
    ])
    
    final_lstm_model.compile(
        optimizer='adam', 
        loss='mse',
        metrics=['mae']
    )
    
    history = final_lstm_model.fit(
        X_train_lstm, y_train_lstm, 
        epochs=50,  # Increased epochs
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    final_lstm_model.save(PATHS['lstm_model'])
    logging.info("LSTM model saved.")

    # === 2. Generate Predictions for Meta-Model ===
    logging.info("Generating meta-model training data...")
    
    # XGBoost predictions
    xgb_preds = final_xgb_model.predict(meta_df[features])
    
    # LSTM predictions
    X_meta_lstm = create_sequences(meta_df, features)
    lstm_preds = final_lstm_model.predict(X_meta_lstm, verbose=0)
    
    # Physics predictions
    phys_preds = []
    logging.info("Running physics simulations for meta-training...")
    
    for idx, row in meta_df.iterrows():
        try:
            telemetry = {key: row[key] for key in ['lat', 'lon', 'alt', 'vel_v']}
            pred = run_forward_simulation(
                telemetry, CANSAT_PHYSICS, WIND_LAYERS, 
                SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps']
            )
            
            if pred:
                # Convert to displacement
                test_row = row.copy()
                test_row['pred_lat'] = pred['pred_lat']
                test_row['pred_lon'] = pred['pred_lon']
                disp_n, disp_e = calculate_displacement(test_row)
                phys_preds.append([disp_n, disp_e])
            else:
                phys_preds.append([0.0, 0.0])
                
        except Exception as e:
            logging.warning(f"Physics prediction failed for row {idx}: {e}")
            phys_preds.append([0.0, 0.0])
    
    phys_preds = np.array(phys_preds)
    
    # === 3. Train Meta-Model ===
    # Align predictions (account for LSTM sequence length)
    seq_len = TRAINING['lstm_sequence_length']
    aligned_phys = phys_preds[seq_len:]
    aligned_xgb = xgb_preds[seq_len:]
    aligned_lstm = lstm_preds
    aligned_targets = meta_df[labels].iloc[seq_len:].values
    
    if BAYESIAN_AVAILABLE:
        # NEW: Train Bayesian Meta-Model
        logging.info("Training Bayesian meta-model...")
        
        # Combine predictions
        base_predictions = np.column_stack([
            aligned_phys, aligned_xgb, aligned_lstm
        ])
        
        bayesian_meta = BayesianMetaModel(n_base_models=3)
        bayesian_meta.fit(base_predictions, aligned_targets, epochs=1000)
        bayesian_meta.save(PATHS['bayesian_meta_model'])
        
        logging.info("Bayesian meta-model saved.")
    
    # Traditional Meta-Model (always train as backup)
    meta_features_df = pd.DataFrame({
        'phys_pred_north': aligned_phys[:, 0],
        'phys_pred_east': aligned_phys[:, 1],
        'xgb_pred_north': aligned_xgb[:, 0],
        'xgb_pred_east': aligned_xgb[:, 1],
        'lstm_pred_north': aligned_lstm[:, 0],
        'lstm_pred_east': aligned_lstm[:, 1],
    })
    
    meta_model = Ridge(alpha=0.1)
    meta_model.fit(meta_features_df, aligned_targets)
    joblib.dump(meta_model, PATHS['meta_model'])
    logging.info("Traditional meta-model saved as backup.")
    
    # === 4. Model Evaluation ===
    logging.info("Evaluating models...")
    
    # Calculate errors
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    xgb_mse = mean_squared_error(aligned_targets, aligned_xgb)
    lstm_mse = mean_squared_error(aligned_targets, aligned_lstm)
    phys_mse = mean_squared_error(aligned_targets, aligned_phys)
    
    logging.info(f"Model MSE Scores:")
    logging.info(f"  XGBoost: {xgb_mse:.4f}")
    logging.info(f"  LSTM: {lstm_mse:.4f}")
    logging.info(f"  Physics: {phys_mse:.4f}")
    
    # Test meta-model
    meta_pred = meta_model.predict(meta_features_df)
    meta_mse = mean_squared_error(aligned_targets, meta_pred)
    logging.info(f"  Meta-Model: {meta_mse:.4f}")
    
    logging.info("=== Training Complete ===")
