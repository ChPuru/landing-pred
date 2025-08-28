# train_ensemble.py
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

def create_sequences(data, features, labels=None):
    """Creates sequences for LSTM. Can be used for training (with labels) or prediction."""
    X, y = [], []
    has_labels = labels is not None and all(col in data.columns for col in labels)
    for i in range(len(data) - TRAINING['lstm_sequence_length']):
        X.append(data[features].iloc[i:(i + TRAINING['lstm_sequence_length'])].values)
        if has_labels:
            y.append(data[labels].iloc[i + TRAINING['lstm_sequence_length'] - 1].values)
    return (np.array(X), np.array(y)) if has_labels else np.array(X)

def train_all_models():
    setup_logging()
    logging.info("--- Training Ensemble (FINAL, CORRECTED VERSION) ---")
    
    train_df = pd.read_csv(PATHS['processed_train_data'])
    meta_df = pd.read_csv(PATHS['processed_meta_data'])
    features, labels = TRAINING['features'], ['landing_disp_north_m', 'landing_disp_east_m']

    # --- 1. Train Final Base Models on ALL Training Data ---
    logging.info("Training final base models on all available training data...")
    
    # Final XGBoost Model
    final_xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=TRAINING['random_state'])
    final_xgb_model.fit(train_df[features], train_df[labels])
    joblib.dump(final_xgb_model, PATHS['xgboost_model'])
    logging.info(f"Final XGBoost model saved.")

    # Final LSTM Model
    X_train_lstm, y_train_lstm = create_sequences(train_df, features, labels)
    
    # --- THIS IS THE CORRECTED SECTION ---
    # The placeholder [...] has been replaced with the actual model definition.
    final_lstm_model = Sequential([
        LayerNormalization(input_shape=(TRAINING['lstm_sequence_length'], len(features))),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(2)
    ])
    # ------------------------------------

    final_lstm_model.compile(optimizer='adam', loss='mse')
    final_lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=64, verbose=1)
    final_lstm_model.save(PATHS['lstm_model'])
    logging.info(f"Final LSTM model saved.")

    # --- 2. Generate Predictions for Meta-Model Training ---
    logging.info("Generating predictions for meta-model training...")
    
    xgb_preds = final_xgb_model.predict(meta_df[features])
    
    X_meta_lstm = create_sequences(meta_df, features)
    lstm_preds = final_lstm_model.predict(X_meta_lstm)
    
    phys_preds = []
    for _, row in meta_df.iterrows():
        pred = run_forward_simulation(row.to_dict(), CANSAT_PHYSICS, WIND_LAYERS, SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps'])
        disp_n, disp_e = calculate_displacement({'lat': row['lat'], 'lon': row['lon'], 'landing_lat': pred['pred_lat'], 'landing_lon': pred['pred_lon']}) if pred else (0,0)
        phys_preds.append([disp_n, disp_e])
    phys_preds = np.array(phys_preds)

    # --- 3. Train the Final Meta-Model ---
    # Align all predictions and create a DataFrame to preserve feature names
    meta_features_df = pd.DataFrame({
        'phys_pred_north': phys_preds[TRAINING['lstm_sequence_length']:, 0],
        'phys_pred_east': phys_preds[TRAINING['lstm_sequence_length']:, 1],
        'xgb_pred_north': xgb_preds[TRAINING['lstm_sequence_length']:, 0],
        'xgb_pred_east': xgb_preds[TRAINING['lstm_sequence_length']:, 1],
        'lstm_pred_north': lstm_preds[:, 0],
        'lstm_pred_east': lstm_preds[:, 1],
    })
    meta_labels_df = meta_df[labels].iloc[TRAINING['lstm_sequence_length']:]
    
    logging.info(f"Training final meta-model on {len(meta_features_df)} samples...")
    meta_model = Ridge()
    meta_model.fit(meta_features_df, meta_labels_df)
    joblib.dump(meta_model, PATHS['meta_model'])
    logging.info(f"Final Meta-Model saved.")

if __name__ == "__main__":
    from data_preparation import calculate_displacement
    train_all_models()