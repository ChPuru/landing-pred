# ml/models/bayesian_ensemble.py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, Tuple, Optional

tfd = tfp.distributions


class BayesianLinearRegression(tf.keras.Model):
    """
    Bayesian linear regression for ensemble weight learning with uncertainty.
    """
    
    def __init__(self, input_dim, output_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Prior distributions for weights and bias
        self.weight_prior = tfd.Normal(loc=0., scale=1.)
        self.bias_prior = tfd.Normal(loc=0., scale=1.)
        self.noise_prior = tfd.Gamma(concentration=1., rate=1.)
        
        # Variational parameters
        self.weight_loc = tf.Variable(
            tf.random.normal([input_dim, output_dim], stddev=0.1),
            name='weight_loc'
        )
        self.weight_scale = tf.Variable(
            tf.fill([input_dim, output_dim], -2.),
            name='weight_scale_raw'
        )
        
        self.bias_loc = tf.Variable(
            tf.zeros([output_dim]),
            name='bias_loc'
        )
        self.bias_scale = tf.Variable(
            tf.fill([output_dim], -2.),
            name='bias_scale_raw'
        )
        
        self.noise_concentration = tf.Variable(1., name='noise_concentration')
        self.noise_rate = tf.Variable(1., name='noise_rate')
    
    @property
    def weight_scale_constrained(self):
        return tf.nn.softplus(self.weight_scale) + 1e-5
    
    @property
    def bias_scale_constrained(self):
        return tf.nn.softplus(self.bias_scale) + 1e-5
    
    def weight_distribution(self):
        return tfd.Normal(loc=self.weight_loc, scale=self.weight_scale_constrained)
    
    def bias_distribution(self):
        return tfd.Normal(loc=self.bias_loc, scale=self.bias_scale_constrained)
    
    def noise_distribution(self):
        return tfd.Gamma(concentration=self.noise_concentration, rate=self.noise_rate)
    
    def call(self, inputs, num_samples=1):
        # Sample weights and bias
        weights = self.weight_distribution().sample(num_samples)
        bias = self.bias_distribution().sample(num_samples)
        
        # Compute predictions
        if num_samples == 1:
            predictions = tf.matmul(inputs, weights[0]) + bias[0]
        else:
            # Multiple samples for uncertainty estimation
            predictions = tf.stack([
                tf.matmul(inputs, weights[i]) + bias[i]
                for i in range(num_samples)
            ])
        
        return predictions
    
    def negative_log_likelihood(self, y_true, x_inputs, num_samples=10):
        """Compute negative log likelihood for training."""
        # Sample predictions
        predictions = self(x_inputs, num_samples=num_samples)
        
        # Noise variance
        noise_variance = 1.0 / self.noise_distribution().sample()
        
        # Likelihood
        if num_samples == 1:
            likelihood = tfd.Normal(loc=predictions, scale=tf.sqrt(noise_variance))
        else:
            # Average over samples
            mean_prediction = tf.reduce_mean(predictions, axis=0)
            likelihood = tfd.Normal(loc=mean_prediction, scale=tf.sqrt(noise_variance))
        
        log_likelihood = tf.reduce_sum(likelihood.log_prob(y_true))
        
        # KL divergence terms
        kl_weight = tf.reduce_sum(tfd.kl_divergence(
            self.weight_distribution(), self.weight_prior
        ))
        kl_bias = tf.reduce_sum(tfd.kl_divergence(
            self.bias_distribution(), self.bias_prior
        ))
        kl_noise = tfd.kl_divergence(
            self.noise_distribution(), self.noise_prior
        )
        
        return -(log_likelihood - kl_weight - kl_bias - kl_noise)


class BayesianMetaModel:
    """
    Sophisticated Bayesian meta-model for ensemble learning with full uncertainty quantification.
    """
    
    def __init__(self, n_base_models=3):
        self.n_base_models = n_base_models
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Model confidence tracking
        self.model_reliabilities = np.ones(n_base_models) / n_base_models
        self.prediction_history = []
        self.error_history = []
        
        logging.info(f"BayesianMetaModel initialized for {n_base_models} base models")
    
    def fit(self, base_predictions, targets, validation_split=0.2, epochs=1000):
        """
        Fit the Bayesian meta-model.
        
        Args:
            base_predictions: [N, n_base_models * 2] array of base model predictions
            targets: [N, 2] array of true targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
        """
        logging.info("Training Bayesian meta-model...")
        
        # Prepare data
        X = np.array(base_predictions, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Add interaction features
        X_enhanced = self._create_interaction_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        # Split data
        n_val = int(len(X_scaled) * validation_split)
        indices = np.random.permutation(len(X_scaled))
        
        X_train = X_scaled[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X_scaled[indices[:n_val]]
        y_val = y[indices[:n_val]]
        
        # Create Bayesian model
        input_dim = X_scaled.shape[1]
        self.model = BayesianLinearRegression(input_dim=input_dim, output_dim=2)
        
        # Training setup
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # Convert to tensors
        X_train_tensor = tf.constant(X_train, dtype=tf.float32)
        y_train_tensor = tf.constant(y_train, dtype=tf.float32)
        X_val_tensor = tf.constant(X_val, dtype=tf.float32)
        y_val_tensor = tf.constant(y_val, dtype=tf.float32)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.model.negative_log_likelihood(
                    y_train_tensor, X_train_tensor, num_samples=5
                )
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # Validation
            if epoch % 10 == 0:
                val_loss = self.model.negative_log_likelihood(
                    y_val_tensor, X_val_tensor, num_samples=5
                )
                
                if epoch % 100 == 0:
                    logging.info(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.is_fitted = True
        logging.info("Bayesian meta-model training completed")
        
        # Evaluate model reliability
        self._update_model_reliability(X_val, y_val)
        
        return self
    
    def predict_with_uncertainty(self, base_predictions, num_samples=100):
        """
        Make predictions with full uncertainty quantification.
        
        Args:
            base_predictions: [N, n_base_models * 2] array of base model predictions
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dict with prediction statistics and uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare input
        X = np.array(base_predictions, dtype=np.float32)
        X_enhanced = self._create_interaction_features(X)
        X_scaled = self.scaler.transform(X_enhanced)
        X_tensor = tf.constant(X_scaled, dtype=tf.float32)
        
        # Monte Carlo sampling
        predictions = self.model(X_tensor, num_samples=num_samples)
        predictions_np = predictions.numpy()
        
        # Calculate statistics
        mean_prediction = np.mean(predictions_np, axis=0)
        std_prediction = np.std(predictions_np, axis=0)
        
        # Percentile-based confidence intervals
        ci_lower = np.percentile(predictions_np, 2.5, axis=0)
        ci_upper = np.percentile(predictions_np, 97.5, axis=0)
        
        # Aleatoric uncertainty (data noise)
        noise_std = 1.0 / np.sqrt(self.model.noise_distribution().sample().numpy())
        aleatoric_uncertainty = np.full_like(mean_prediction, noise_std)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_prediction
        
        # Total uncertainty
        total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        results = {
            'prediction': mean_prediction,
            'prediction_std': std_prediction,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_samples': predictions_np
        }
        
        # Add model-specific insights
        results.update(self._analyze_base_model_contributions(base_predictions))
        
        return results
    
    def _create_interaction_features(self, X):
        """Create interaction features between base models."""
        n_samples, n_features = X.shape
        
        # Original features
        features = [X]
        
        # Pairwise differences (model disagreement features)
        for i in range(0, n_features, 2):  # Step by 2 for lat/lon pairs
            for j in range(i+2, n_features, 2):
                if j < n_features:
                    # Distance between model predictions
                    diff_lat = X[:, i] - X[:, j]
                    diff_lon = X[:, i+1] - X[:, j+1]
                    distance = np.sqrt(diff_lat**2 + diff_lon**2)
                    features.append(distance.reshape(-1, 1))
        
        # Model confidence features (based on historical performance)
        if len(self.model_reliabilities) == self.n_base_models:
            reliability_features = np.tile(
                self.model_reliabilities, 
                (n_samples, 1)
            )
            features.append(reliability_features)
        
        # Statistical features
        lat_predictions = X[:, ::2]  # Every other starting from 0
        lon_predictions = X[:, 1::2]  # Every other starting from 1
        
        # Mean and std of predictions
        lat_mean = np.mean(lat_predictions, axis=1, keepdims=True)
        lat_std = np.std(lat_predictions, axis=1, keepdims=True)
        lon_mean = np.mean(lon_predictions, axis=1, keepdims=True)
        lon_std = np.std(lon_predictions, axis=1, keepdims=True)
        
        features.extend([lat_mean, lat_std, lon_mean, lon_std])
        
        return np.concatenate(features, axis=1)
    
    def _update_model_reliability(self, X_val, y_val):
        """Update model reliability scores based on validation performance."""
        if len(X_val) == 0:
            return
        
        # Get individual model predictions from validation set
        n_models = self.n_base_models
        errors = np.zeros(n_models)
        
        for i in range(n_models):
            model_pred = X_val[:, i*2:(i+1)*2]  # Get lat/lon for model i
            model_error = np.mean(np.sqrt(np.sum((model_pred - y_val)**2, axis=1)))
            errors[i] = model_error
        
        # Convert errors to reliability scores (inverse relationship)
        reliabilities = 1.0 / (1.0 + errors)
        self.model_reliabilities = reliabilities / np.sum(reliabilities)
        
        logging.info(f"Updated model reliabilities: {self.model_reliabilities}")
    
    def _analyze_base_model_contributions(self, base_predictions):
        """Analyze which base models contribute most to the final prediction."""
        n_models = self.n_base_models
        contributions = {}
        
        # Calculate weighted contributions based on reliability
        for i in range(n_models):
            model_name = f'model_{i}'
            reliability = self.model_reliabilities[i] if len(self.model_reliabilities) == n_models else 1.0/n_models
            contributions[f'{model_name}_weight'] = reliability
            contributions[f'{model_name}_prediction'] = base_predictions[:, i*2:(i+1)*2].tolist()
        
        return contributions
    
    def save(self, filepath):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_dict = {
            'model_weights': [var.numpy() for var in self.model.trainable_variables],
            'scaler': self.scaler,
            'model_reliabilities': self.model_reliabilities,
            'n_base_models': self.n_base_models,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        logging.info(f"Bayesian meta-model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model."""
        save_dict = joblib.load(filepath)
        
        self.n_base_models = save_dict['n_base_models']
        self.scaler = save_dict['scaler']
        self.model_reliabilities = save_dict['model_reliabilities']
        self.is_fitted = save_dict['is_fitted']
        
        # Recreate model with correct dimensions
        input_dim = len(save_dict['scaler'].scale_)
        self.model = BayesianLinearRegression(input_dim=input_dim, output_dim=2)
        
        # Build model by doing a dummy forward pass
        dummy_input = tf.zeros((1, input_dim))
        _ = self.model(dummy_input)
        
        # Load weights
        for var, saved_weight in zip(self.model.trainable_variables, save_dict['model_weights']):
            var.assign(saved_weight)
        
        logging.info(f"Bayesian meta-model loaded from {filepath}")


class AdaptiveEnsembleManager:
    """
    Manages the ensemble of models with adaptive weighting and online learning.
    """
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.performance_history = {}
        self.recent_errors = {}
        self.adaptation_rate = 0.1
        
    def add_model(self, name: str, model, initial_weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = {
            'model': model,
            'weight': initial_weight,
            'error_history': [],
            'predictions': []
        }
        self.recent_errors[name] = []
        logging.info(f"Added model '{name}' to ensemble")
    
    def set_meta_model(self, meta_model: BayesianMetaModel):
        """Set the meta-model for ensemble combination."""
        self.meta_model = meta_model
    
    def predict(self, inputs, return_individual=False):
        """
        Make ensemble prediction with uncertainty quantification.
        """
        individual_predictions = {}
        base_predictions = []
        
        # Get predictions from all base models
        for name, model_info in self.models.items():
            try:
                if hasattr(model_info['model'], 'predict_with_uncertainty'):
                    result = model_info['model'].predict_with_uncertainty(inputs)
                    pred = result['prediction']
                elif hasattr(model_info['model'], 'predict'):
                    pred = model_info['model'].predict(inputs)
                else:
                    # Assume it's a callable
                    pred = model_info['model'](inputs)
                
                individual_predictions[name] = pred
                
                # Flatten prediction for meta-model input
                if pred.ndim > 1:
                    base_predictions.extend(pred.flatten())
                else:
                    base_predictions.extend([pred])
                    
                # Store for adaptation
                model_info['predictions'].append(pred)
                
            except Exception as e:
                logging.error(f"Error getting prediction from {name}: {e}")
                # Use zeros as fallback
                fallback_pred = np.zeros(2)
                individual_predictions[name] = fallback_pred
                base_predictions.extend(fallback_pred)
        
        # Use meta-model if available
        if self.meta_model and self.meta_model.is_fitted:
            try:
                base_pred_array = np.array(base_predictions).reshape(1, -1)
                ensemble_result = self.meta_model.predict_with_uncertainty(base_pred_array)
                ensemble_result['individual_predictions'] = individual_predictions
            except Exception as e:
                logging.error(f"Meta-model prediction failed: {e}")
                ensemble_result = self._fallback_ensemble(individual_predictions)
        else:
            ensemble_result = self._fallback_ensemble(individual_predictions)
        
        if return_individual:
            return ensemble_result, individual_predictions
        else:
            return ensemble_result
    
    def _fallback_ensemble(self, individual_predictions):
        """Simple weighted average fallback when meta-model is unavailable."""
        if not individual_predictions:
            return {'prediction': np.zeros(2), 'total_uncertainty': np.ones(2)}
        
        predictions = []
        weights = []
        
        for name, pred in individual_predictions.items():
            predictions.append(pred)
            weights.append(self.models[name]['weight'])
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        ensemble_uncertainty = np.std(predictions, axis=0)
        
        return {
            'prediction': ensemble_pred,
            'total_uncertainty': ensemble_uncertainty,
            'individual_predictions': individual_predictions
        }
    
    def update_performance(self, true_target, latest_predictions=None):
        """
        Update model performance tracking and adapt weights.
        """
        if latest_predictions is None:
            # Use the most recent predictions stored
            latest_predictions = {
                name: model_info['predictions'][-1] if model_info['predictions'] else np.zeros(2)
                for name, model_info in self.models.items()
            }
        
        # Calculate errors for each model
        for name, pred in latest_predictions.items():
            if name in self.models:
                error = np.linalg.norm(np.array(pred) - np.array(true_target))
                self.models[name]['error_history'].append(error)
                self.recent_errors[name].append(error)
                
                # Keep only recent errors (sliding window)
                if len(self.recent_errors[name]) > 50:
                    self.recent_errors[name] = self.recent_errors[name][-50:]
        
        # Adapt model weights based on recent performance
        self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt model weights based on recent performance."""
        if not any(self.recent_errors.values()):
            return
        
        # Calculate average recent errors
        avg_errors = {}
        for name, errors in self.recent_errors.items():
            if errors:
                avg_errors[name] = np.mean(errors)
            else:
                avg_errors[name] = float('inf')
        
        # Convert errors to weights (inverse relationship)
        total_inverse_error = sum(1.0 / (error + 1e-6) for error in avg_errors.values())
        
        for name in self.models:
            if name in avg_errors:
                new_weight = (1.0 / (avg_errors[name] + 1e-6)) / total_inverse_error
                # Smooth weight adaptation
                current_weight = self.models[name]['weight']
                self.models[name]['weight'] = (
                    (1 - self.adaptation_rate) * current_weight +
                    self.adaptation_rate * new_weight
                )
    
    def get_performance_summary(self):
        """Get performance summary for all models."""
        summary = {}
        for name, model_info in self.models.items():
            if model_info['error_history']:
                summary[name] = {
                    'current_weight': model_info['weight'],
                    'avg_error': np.mean(model_info['error_history']),
                    'recent_avg_error': np.mean(self.recent_errors[name]) if self.recent_errors[name] else 0,
                    'num_predictions': len(model_info['predictions'])
                }
            else:
                summary[name] = {
                    'current_weight': model_info['weight'],
                    'avg_error': 0,
                    'recent_avg_error': 0,
                    'num_predictions': 0
                }
        return summary