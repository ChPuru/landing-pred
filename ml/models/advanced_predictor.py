# ml/models/advanced_predictor.py
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import logging

class PhysicsInformedNN(tf.keras.layers.Layer):
    """
    Physics-informed neural network layer that incorporates aerodynamic principles.
    """
    
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
        # Physics-based feature extractors
        self.altitude_processor = Dense(8, activation='relu', name='altitude_physics')
        self.velocity_processor = Dense(8, activation='relu', name='velocity_physics')
        self.position_processor = Dense(8, activation='relu', name='position_physics')
        
        # Physics fusion layer
        self.physics_fusion = Dense(units, activation='relu', name='physics_fusion')
        
        # Physical constants (learnable parameters with physics constraints)
        self.drag_factor = self.add_weight(
            name='drag_factor',
            shape=(),
            initializer='ones',
            trainable=True,
            constraint=lambda w: tf.clip_by_value(w, 0.1, 2.0)
        )
        
        self.wind_factor = self.add_weight(
            name='wind_factor', 
            shape=(),
            initializer='ones',
            trainable=True,
            constraint=lambda w: tf.clip_by_value(w, 0.1, 2.0)
        )
    
    def call(self, inputs):
        # Extract physics-relevant features
        # inputs shape: (batch_size, sequence_length, features)
        
        # Get the latest timestep for physics calculations
        current_state = inputs[:, -1, :]  # (batch_size, features)
        
        # Assuming feature order: [alt, vel_v, lat, lon, ...]
        altitude = current_state[:, 0:1]
        velocity = current_state[:, 1:2]
        position = current_state[:, 2:4]  # lat, lon
        
        # Physics-informed feature processing
        alt_features = self.altitude_processor(altitude)
        vel_features = self.velocity_processor(velocity)
        pos_features = self.position_processor(position)
        
        # Apply physics-based transformations
        # Air density effect (exponential with altitude)
        air_density_effect = tf.exp(-altitude / 8500.0)
        physics_velocity = velocity * air_density_effect * self.drag_factor
        
        # Combine physics features
        physics_features = tf.concat([
            alt_features, 
            vel_features * air_density_effect,
            pos_features * self.wind_factor
        ], axis=1)
        
        return self.physics_fusion(physics_features)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Custom multi-head self-attention for temporal sequence processing.
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(concat_attention)


class AdvancedCanSatPredictor(tf.keras.Model):
    """
    Advanced neural network architecture for CanSat landing prediction.
    Combines physics-informed networks, attention mechanisms, and uncertainty quantification.
    """
    
    def __init__(self, sequence_length=30, num_features=17, **kwargs):
        super().__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        # Input normalization
        self.input_norm = LayerNormalization(name='input_norm')
        
        # Physics-informed processing
        self.physics_net = PhysicsInformedNN(units=64, name='physics_net')
        
        # Temporal sequence processing with attention
        self.lstm_1 = LSTM(128, return_sequences=True, name='lstm_1')
        self.dropout_1 = Dropout(0.2)
        
        self.attention = MultiHeadSelfAttention(d_model=128, num_heads=8, name='attention')
        self.attention_norm = LayerNormalization(name='attention_norm')
        
        self.lstm_2 = LSTM(64, return_sequences=False, name='lstm_2')
        self.dropout_2 = Dropout(0.2)
        
        # Feature fusion
        self.fusion_dense = Dense(128, activation='relu', name='fusion')
        self.fusion_dropout = Dropout(0.3)
        
        # Uncertainty estimation branch
        self.uncertainty_branch = tf.keras.Sequential([
            Dense(64, activation='relu', name='uncertainty_1'),
            Dense(32, activation='relu', name='uncertainty_2'),
            Dense(4, activation='softplus', name='uncertainty_output')  # [lat_std, lon_std, lat_bias, lon_bias]
        ], name='uncertainty_branch')
        
        # Main prediction branch
        self.prediction_branch = tf.keras.Sequential([
            Dense(64, activation='relu', name='prediction_1'),
            Dense(32, activation='relu', name='prediction_2'),
            Dense(2, activation='linear', name='prediction_output')  # [lat_displacement, lon_displacement]
        ], name='prediction_branch')
        
        # Environmental context processor (for additional features)
        self.env_processor = Dense(32, activation='relu', name='env_processor')
        
    def call(self, inputs, training=None):
        # inputs can be a single tensor or a list [sequence, environmental_features]
        if isinstance(inputs, list):
            sequence_input, env_input = inputs
        else:
            sequence_input = inputs
            env_input = None
        
        # Normalize input sequences
        x = self.input_norm(sequence_input)
        
        # Physics-informed processing
        physics_features = self.physics_net(x)
        
        # Temporal processing with LSTM
        lstm_out = self.lstm_1(x)
        lstm_out = self.dropout_1(lstm_out, training=training)
        
        # Self-attention mechanism
        attention_out = self.attention(lstm_out)
        attention_out = self.attention_norm(attention_out + lstm_out)  # Residual connection
        
        # Second LSTM layer
        temporal_features = self.lstm_2(attention_out)
        temporal_features = self.dropout_2(temporal_features, training=training)
        
        # Combine physics and temporal features
        combined_features = tf.concat([physics_features, temporal_features], axis=1)
        
        # Add environmental context if available
        if env_input is not None:
            env_features = self.env_processor(env_input)
            combined_features = tf.concat([combined_features, env_features], axis=1)
        
        # Final fusion
        fused_features = self.fusion_dense(combined_features)
        fused_features = self.fusion_dropout(fused_features, training=training)
        
        # Generate predictions and uncertainty estimates
        prediction = self.prediction_branch(fused_features)
        uncertainty = self.uncertainty_branch(fused_features)
        
        if training:
            return prediction, uncertainty
        else:
            return {
                'prediction': prediction,
                'uncertainty': uncertainty,
                'attention_weights': None  # Could return attention weights for interpretation
            }
    
    def predict_with_uncertainty(self, inputs, num_samples=100):
        """
        Monte Carlo prediction to estimate uncertainty.
        """
        predictions = []
        uncertainties = []
        
        for _ in range(num_samples):
            pred, unc = self(inputs, training=True)
            predictions.append(pred)
            uncertainties.append(unc)
        
        predictions = tf.stack(predictions, axis=0)  # (num_samples, batch_size, 2)
        uncertainties = tf.stack(uncertainties, axis=0)  # (num_samples, batch_size, 4)
        
        # Calculate statistics
        mean_prediction = tf.reduce_mean(predictions, axis=0)
        std_prediction = tf.math.reduce_std(predictions, axis=0)
        mean_uncertainty = tf.reduce_mean(uncertainties, axis=0)
        
        return {
            'prediction': mean_prediction,
            'prediction_std': std_prediction,
            'aleatoric_uncertainty': mean_uncertainty[:, :2],  # Data uncertainty
            'epistemic_uncertainty': std_prediction,  # Model uncertainty
            'total_uncertainty': tf.sqrt(std_prediction**2 + mean_uncertainty[:, :2]**2)
        }


class CustomLoss(tf.keras.losses.Loss):
    """
    Custom loss function that incorporates both prediction accuracy and uncertainty estimation.
    """
    
    def __init__(self, uncertainty_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.uncertainty_weight = uncertainty_weight
    
    def call(self, y_true, y_pred):
        # y_pred is a tuple: (prediction, uncertainty)
        prediction, uncertainty = y_pred
        
        # Main prediction loss (MSE)
        prediction_loss = tf.reduce_mean(tf.square(y_true - prediction))
        
        # Uncertainty loss - penalize overconfident wrong predictions
        prediction_error = tf.abs(y_true - prediction)
        uncertainty_std = uncertainty[:, :2]  # First 2 elements are std estimates
        
        # Negative log likelihood assuming Gaussian uncertainty
        nll = tf.reduce_mean(
            0.5 * tf.math.log(2 * np.pi * uncertainty_std**2) +
            0.5 * (prediction_error**2 / uncertainty_std**2)
        )
        
        return prediction_loss + self.uncertainty_weight * nll


def create_advanced_model(sequence_length=30, num_features=17):
    """
    Factory function to create and compile the advanced model.
    """
    model = AdvancedCanSatPredictor(
        sequence_length=sequence_length,
        num_features=num_features
    )
    
    # Custom optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    # Compile with custom loss
    model.compile(
        optimizer=optimizer,
        loss=CustomLoss(uncertainty_weight=0.1),
        metrics=['mse', 'mae']
    )
    
    return model


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions.
    """
    
    def __init__(self, models=None):
        self.models = models or []
        self.weights = None
        
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)
    
    def predict(self, inputs):
        """Make ensemble prediction."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        uncertainties = []
        
        for model in self.models:
            result = model(inputs, training=False)
            if isinstance(result, dict):
                predictions.append(result['prediction'])
                uncertainties.append(result['uncertainty'])
            else:
                pred, unc = result
                predictions.append(pred)
                uncertainties.append(unc)
        
        # Weighted average of predictions
        total_weight = sum(self.weights)
        weighted_predictions = []
        weighted_uncertainties = []
        
        for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
            weight = self.weights[i] / total_weight
            weighted_predictions.append(pred * weight)
            weighted_uncertainties.append(unc * weight)
        
        ensemble_prediction = tf.reduce_sum(tf.stack(weighted_predictions), axis=0)
        ensemble_uncertainty = tf.reduce_sum(tf.stack(weighted_uncertainties), axis=0)
        
        # Add ensemble diversity as additional uncertainty
        pred_stack = tf.stack(predictions)
        ensemble_diversity = tf.math.reduce_std(pred_stack, axis=0)
        
        return {
            'prediction': ensemble_prediction,
            'aleatoric_uncertainty': ensemble_uncertainty,
            'epistemic_uncertainty': ensemble_diversity,
            'total_uncertainty': tf.sqrt(ensemble_uncertainty**2 + ensemble_diversity**2)
        }


def train_advanced_model(train_data, val_data, model_path, epochs=100):
    """
    Train the advanced model with callbacks and monitoring.
    """
    model = create_advanced_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history