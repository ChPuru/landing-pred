# train_ensemble.py
import pandas as pd
import numpy as np
import joblib
import logging
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging, calculate_displacement
from physics_engine_advanced import run_forward_simulation

def create_sequences_optimized(data, features, labels=None):
    num_sequences = len(data) - TRAINING['lstm_sequence_length']
    if num_sequences <= 0: return (np.array([]), np.array([])) if labels else np.array([])
    X = np.zeros((num_sequences, TRAINING['lstm_sequence_length'], len(features)), dtype=np.float32)
    has_labels = labels is not None and all(col in data.columns for col in labels)
    if has_labels: y = np.zeros((num_sequences, len(labels)), dtype=np.float32)
    feature_data = data[features].values
    if has_labels: label_data = data[labels].values
    for i in range(num_sequences):
        X[i] = feature_data[i : i + TRAINING['lstm_sequence_length']]
        if has_labels: y[i] = label_data[i + TRAINING['lstm_sequence_length'] - 1]
    return (X, y) if has_labels else X

def train_all_models():
    setup_logging()
    logging.info("--- Training Ensemble with XGBoost Meta-Model (FINAL) ---")
    
    train_df = pd.read_csv(PATHS['processed_train_data'])
    meta_df = pd.read_csv(PATHS['processed_meta_data'])
    features, labels = TRAINING['features'], ['landing_disp_north_m', 'landing_disp_east_m']

    # --- 1. Train Final Base Models (No change here) ---
    logging.info("Training final base models...")
    final_xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=TRAINING['random_state'])
    final_xgb_model.fit(train_df[features], train_df[labels])
    joblib.dump(final_xgb_model, PATHS['xgboost_model'])
    logging.info(f"Final XGBoost model saved.")
    
    X_train_lstm, y_train_lstm = create_sequences_optimized(train_df, features, labels)
    final_lstm_model = Sequential([
        LayerNormalization(input_shape=(TRAINING['lstm_sequence_length'], len(features))),
        LSTM(64, activation='tanh', return_sequences=True), Dropout(0.2),
        LSTM(32, activation='tanh'), Dropout(0.2),
        Dense(2)
    ])
    final_lstm_model.compile(optimizer='adam', loss='mse')
    final_lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=64, verbose=1)
    final_lstm_model.save(PATHS['lstm_model'])
    logging.info(f"Final LSTM model saved.")

    # --- 2. Generate Predictions for Meta-Model Training (No change here) ---
    logging.info("Generating predictions for meta-model training...")
    xgb_preds = final_xgb_model.predict(meta_df[features])
    X_meta_lstm = create_sequences_optimized(meta_df, features)
    lstm_preds = final_lstm_model.predict(X_meta_lstm)
    phys_preds = []
    for _, row in meta_df.iterrows():
        pred = run_forward_simulation(row.to_dict(), CANSAT_PHYSICS, WIND_LAYERS, SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps'])
        disp_n, disp_e = calculate_displacement({'lat': row['lat'], 'lon': row['lon'], 'landing_lat': pred['pred_lat'], 'landing_lon': pred['pred_lon']}) if pred else (0,0)
        phys_preds.append([disp_n, disp_e])
    phys_preds = np.array(phys_preds)

    # --- 3. Train the Final Meta-Model ---
    meta_features_df = pd.DataFrame({
        'phys_pred_north': phys_preds[TRAINING['lstm_sequence_length']:, 0],
        'phys_pred_east': phys_preds[TRAINING['lstm_sequence_length']:, 1],
        'xgb_pred_north': xgb_preds[TRAINING['lstm_sequence_length']:, 0],
        'xgb_pred_east': xgb_preds[TRAINING['lstm_sequence_length']:, 1],
        'lstm_pred_north': lstm_preds[:, 0],
        'lstm_pred_east': lstm_preds[:, 1],
    })
    meta_labels_df = meta_df[labels].iloc[TRAINING['lstm_sequence_length']:]
    
    logging.info(f"Training final XGBoost meta-model on {len(meta_features_df)} samples...")
    
    meta_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=TRAINING['random_state'],
        n_jobs=-1
    )
    # ------------------------------------
    
    meta_model.fit(meta_features_df, meta_labels_df)
    joblib.dump(meta_model, PATHS['meta_model'])
    logging.info(f"Final Meta-Model saved to {PATHS['meta_model']}")

if __name__ == "__main__":
    train_all_models()