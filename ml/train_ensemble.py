# train_ensemble.py
import pandas as pd
import numpy as np
import joblib
import logging
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging, calculate_haversine_error
from physics_engine_advanced import run_forward_simulation

def create_sequences(data, features, labels):
    # (Same as before)
    X, y = [], []
    for i in range(len(data) - TRAINING['lstm_sequence_length']):
        X.append(data[features].iloc[i:(i + TRAINING['lstm_sequence_length'])].values)
        y.append(data[labels].iloc[i + TRAINING['lstm_sequence_length'] - 1].values)
    return np.array(X), np.array(y)

def train_all_models():
    setup_logging()
    logging.info("--- Training the Ultimate Ensemble Prediction System (Professional Version) ---")
    
    train_df = pd.read_csv(PATHS['processed_train_data'])
    meta_df = pd.read_csv(PATHS['processed_meta_data'])

    features = TRAINING['features']
    labels = TRAINING['labels']

    # --- 1. Train the XGBoost (Tabular) Model ---
    logging.info("Training XGBoost Model...")
    xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=TRAINING['random_state'], n_jobs=-1)
    xgb_model.fit(train_df[features], train_df[labels])
    joblib.dump(xgb_model, PATHS['xgboost_model'])
    logging.info(f"XGBoost model saved to {PATHS['xgboost_model']}")

    # --- 2. Train the LSTM (Sequential) Model ---
    logging.info("Training LSTM Model...")
    X_train_lstm, y_train_lstm = create_sequences(train_df, features, labels)
    
    lstm_model = Sequential([
        LayerNormalization(input_shape=(TRAINING['lstm_sequence_length'], len(features))),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(2)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=25, batch_size=64, verbose=0)
    lstm_model.save(PATHS['lstm_model'])
    logging.info(f"LSTM model saved to {PATHS['lstm_model']}")

    # --- 3. Train the Stacking Meta-Model ---
    logging.info("Training Stacking Meta-Model...")
    
    # Generate predictions from base models on the (unseen) meta dataset
    xgb_preds = xgb_model.predict(meta_df[features])
    
    X_meta_lstm, y_meta_true = create_sequences(meta_df, features, labels)
    lstm_preds = lstm_model.predict(X_meta_lstm)
    
    phys_preds = []
    for _, row in meta_df.iterrows():
        pred = run_forward_simulation(row.to_dict(), CANSAT_PHYSICS, WIND_LAYERS, SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps'])
        phys_preds.append([pred['pred_lat'], pred['pred_lon']] if pred else [row['lat'], row['lon']])
    phys_preds = np.array(phys_preds)
    
    # Align all predictions correctly
    meta_X = np.hstack([
        phys_preds[TRAINING['lstm_sequence_length']:],
        xgb_preds[TRAINING['lstm_sequence_length']:],
        lstm_preds
    ])
    meta_y = y_meta_true
    
    meta_model = LinearRegression()
    meta_model.fit(meta_X, meta_y)
    joblib.dump(meta_model, PATHS['meta_model'])
    logging.info(f"Meta-Model saved to {PATHS['meta_model']}")
    
    # --- 4. Final Evaluation ---
    final_preds = meta_model.predict(meta_X)
    error_meters = calculate_haversine_error(meta_y, final_preds)
    logging.info(f"--- Final Ensemble Evaluation | Average Landing Error: {error_meters:.2f} meters ---")

if __name__ == "__main__":
    train_all_models()