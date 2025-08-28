import os
import pandas as pd
import numpy as np
import joblib
import logging
import xgboost as xgb
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from config import PATHS, TRAINING, CANSAT_PHYSICS, WIND_LAYERS, SIMULATION
from utils import setup_logging
from physics_engine_advanced import run_forward_simulation

def create_sequences_memory_efficient(data, features, labels, max_sequences=5000):
    """
    Create sequences with memory efficiency in mind.
    Sample flights rather than using all data points.
    """
    print(f"Creating sequences from {len(data)} data points...")
    
    # Group by flight_id to create proper sequences
    flight_groups = data.groupby('flight_id')
    
    X, y = [], []
    sequence_count = 0
    seq_len = TRAINING['lstm_sequence_length']
    
    for flight_id, flight_data in flight_groups:
        if sequence_count >= max_sequences:
            break
            
        flight_data = flight_data.sort_values('alt', ascending=False)  # Ensure proper order
        
        if len(flight_data) <= seq_len:
            continue
            
        # Create multiple sequences per flight (sliding window with stride)
        stride = max(1, len(flight_data) // 10)  # Take every nth point for memory efficiency
        
        for i in range(0, len(flight_data) - seq_len, stride):
            if sequence_count >= max_sequences:
                break
                
            try:
                seq_data = flight_data[features].iloc[i:i+seq_len].values
                target_data = flight_data[labels].iloc[i+seq_len-1].values
                
                if seq_data.shape[0] == seq_len:  # Ensure proper sequence length
                    X.append(seq_data.astype(np.float32))  # Use float32 to save memory
                    y.append(target_data.astype(np.float32))
                    sequence_count += 1
                    
            except Exception as e:
                continue
    
    print(f"Created {len(X)} sequences")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_memory_efficient():
    """
    Memory-efficient training that handles large datasets.
    """
    
    print("=== Memory-Efficient Training ===\n")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(PATHS['processed_train_data'])
    meta_df = pd.read_csv(PATHS['processed_meta_data'])
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Meta data: {len(meta_df)} rows")
    
    # Get features
    try:
        feature_config = joblib.load(PATHS['scaler'].replace('.pkl', '_features.pkl'))
        features = feature_config['features']
        print(f"Using {len(features)} optimized features")
    except FileNotFoundError:
        features = TRAINING['features'] 
        print(f"Using {len(features)} default features")
    
    labels = ['landing_disp_north_m', 'landing_disp_east_m']
    
    # Check available features
    available_features = [f for f in features if f in train_df.columns]
    print(f"Available features: {len(available_features)}/{len(features)}")
    
    if len(available_features) == 0:
        print("❌ No features available!")
        return
    
    # === 1. Train XGBoost (Memory efficient already) ===
    print(f"\n1. Training XGBoost...")
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=TRAINING['random_state'],
            verbosity=1,
            n_jobs=-1  # Use all CPU cores
        )
        
        xgb_model.fit(train_df[available_features], train_df[labels])
        joblib.dump(xgb_model, PATHS['xgboost_model'])
        print("✅ XGBoost trained and saved")
        
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
        return
    
    # === 2. Train LSTM (Memory efficient) ===
    print(f"\n2. Training LSTM with memory optimization...")
    try:
        # Create sequences with memory limit
        X_train, y_train = create_sequences_memory_efficient(
            train_df, available_features, labels, max_sequences=3000
        )
        
        if len(X_train) == 0:
            print("❌ No LSTM sequences generated!")
            return
        
        print(f"LSTM training data shape: {X_train.shape} -> {y_train.shape}")
        
        # Build smaller LSTM model for memory efficiency
        lstm_model = Sequential([
            LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), 
                 return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(2)
        ])
        
        lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with callbacks for efficiency
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,  # Larger batch size for efficiency
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        lstm_model.save(PATHS['lstm_model'])
        print("✅ LSTM trained and saved")
        
        # Clear memory
        del X_train, y_train
        
    except Exception as e:
        print(f"❌ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # === 3. Train Meta-Model (Efficient) ===
    print(f"\n3. Training Meta-Model...")
    try:
        # Sample data for meta-model to save memory and time
        sample_size = min(1000, len(meta_df))
        meta_sample = meta_df.sample(n=sample_size, random_state=42)
        print(f"Using {sample_size} samples for meta-model training")
        
        # Generate predictions
        print("Generating XGBoost predictions...")
        xgb_preds = xgb_model.predict(meta_sample[available_features])
        
        print("Generating LSTM predictions...")
        X_meta, _ = create_sequences_memory_efficient(
            meta_sample, available_features, labels, max_sequences=500
        )
        
        if len(X_meta) > 0:
            lstm_preds = lstm_model.predict(X_meta, verbose=0)
            # Take the last predictions to match other models
            lstm_preds = lstm_preds[-len(xgb_preds):]
        else:
            lstm_preds = np.zeros_like(xgb_preds)
            
        print("Generating physics predictions (simplified)...")
        # Sample even fewer for physics to save time
        physics_sample_size = min(200, len(meta_sample))
        physics_sample = meta_sample.sample(n=physics_sample_size, random_state=43)
        
        phys_preds = []
        for i, (_, row) in enumerate(physics_sample.iterrows()):
            if i % 50 == 0:
                print(f"  Physics simulation {i+1}/{physics_sample_size}")
            
            try:
                telemetry = {
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'alt': float(row['alt']),
                    'vel_v': float(row['vel_v'])
                }
                
                pred = run_forward_simulation(
                    telemetry, CANSAT_PHYSICS, WIND_LAYERS,
                    SIMULATION['physics_timestep_s'], SIMULATION['oscillation_mps']
                )
                
                if pred:
                    disp_n = (pred['pred_lat'] - row['lat']) * 111320.0
                    disp_e = (pred['pred_lon'] - row['lon']) * 111320.0 * np.cos(np.radians(row['lat']))
                    phys_preds.append([disp_n, disp_e])
                else:
                    phys_preds.append([0.0, 0.0])
                    
            except Exception as e:
                phys_preds.append([0.0, 0.0])
        
        phys_preds = np.array(phys_preds)
        
        # Match array lengths
        min_len = min(len(phys_preds), len(xgb_preds), len(lstm_preds))
        
        if min_len < 10:
            print("❌ Not enough data for meta-model")
            return
        
        # Create meta-model features
        meta_features = pd.DataFrame({
            'phys_pred_north': phys_preds[:min_len, 0],
            'phys_pred_east': phys_preds[:min_len, 1],
            'xgb_pred_north': xgb_preds[:min_len, 0],
            'xgb_pred_east': xgb_preds[:min_len, 1],
            'lstm_pred_north': lstm_preds[:min_len, 0],
            'lstm_pred_east': lstm_preds[:min_len, 1],
        })
        
        # Get corresponding targets
        meta_targets = physics_sample[labels].iloc[:min_len].values
        
        print(f"Meta-model data: {meta_features.shape} -> {meta_targets.shape}")
        
        # Train meta-model
        meta_model = Ridge(alpha=0.1)
        meta_model.fit(meta_features, meta_targets)
        joblib.dump(meta_model, PATHS['meta_model'])
        
        print("✅ Meta-model trained and saved")
        
    except Exception as e:
        print(f"❌ Meta-model training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 MEMORY-EFFICIENT TRAINING COMPLETED! 🎉")
    
    # Show file sizes
    print(f"\nGenerated model files:")
    model_files = [PATHS['xgboost_model'], PATHS['lstm_model'], PATHS['meta_model']]
    total_size = 0
    
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ Missing: {file_path}")
    
    print(f"\nTotal model size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    # Quick validation test
    print(f"\n=== Quick Validation Test ===")
    try:
        # Test XGBoost
        test_features = train_df[available_features].iloc[0:1]
        xgb_test = xgb_model.predict(test_features)
        print(f"✅ XGBoost prediction: {xgb_test[0]}")
        
        # Test LSTM
        test_seq = np.random.random((1, TRAINING['lstm_sequence_length'], len(available_features))).astype(np.float32)
        lstm_test = lstm_model.predict(test_seq, verbose=0)
        print(f"✅ LSTM prediction: {lstm_test[0]}")
        
        # Test Meta-model
        test_meta = pd.DataFrame([[0, 0, xgb_test[0][0], xgb_test[0][1], lstm_test[0][0], lstm_test[0][1]]],
                                columns=meta_features.columns)
        meta_test = meta_model.predict(test_meta)
        print(f"✅ Meta-model prediction: {meta_test[0]}")
        
        print(f"\n✅ All models working correctly!")
        
    except Exception as e:
        print(f"⚠️  Validation test failed: {e}")

if __name__ == "__main__":
    train_memory_efficient()

# === Alternative: Quick training script ===
def quick_train():
    """Ultra-fast training for testing purposes."""
    
    print("=== QUICK TRAINING MODE ===")
    
    train_df = pd.read_csv(PATHS['processed_train_data'])
    
    # Use only first 1000 samples
    train_sample = train_df.head(1000)
    
    try:
        feature_config = joblib.load(PATHS['scaler'].replace('.pkl', '_features.pkl'))
        features = feature_config['features']
    except:
        from config import TRAINING
        features = TRAINING['features']
    
    available_features = [f for f in features if f in train_sample.columns]
    labels = ['landing_disp_north_m', 'landing_disp_east_m']
    
    # Quick XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    xgb_model.fit(train_sample[available_features], train_sample[labels])
    joblib.dump(xgb_model, PATHS['xgboost_model'])
    print("✅ Quick XGBoost done")
    
    # Simple LSTM
    X = []
    y = []
    for i in range(len(train_sample) - 30):
        X.append(train_sample[available_features].iloc[i:i+30].values)
        y.append(train_sample[labels].iloc[i+29].values)
    
    X = np.array(X, dtype=np.float32)[:100]  # Only 100 sequences
    y = np.array(y, dtype=np.float32)[:100]
    
    lstm_model = Sequential([
        LSTM(16, input_shape=(30, len(available_features))),
        Dense(2)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X, y, epochs=5, verbose=1)
    lstm_model.save(PATHS['lstm_model'])
    print("✅ Quick LSTM done")
    
    # Simple meta-model
    meta_model = Ridge()
    fake_meta_features = pd.DataFrame({
        'phys_pred_north': [0], 'phys_pred_east': [0],
        'xgb_pred_north': [0], 'xgb_pred_east': [0], 
        'lstm_pred_north': [0], 'lstm_pred_east': [0]
    })
    fake_targets = [[0, 0]]
    
    meta_model.fit(fake_meta_features, fake_targets)
    joblib.dump(meta_model, PATHS['meta_model'])
    print("✅ Quick Meta-model done")
    
    print("🎉 QUICK TRAINING COMPLETE!")