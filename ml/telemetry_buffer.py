# ml/telemetry_buffer.py
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

class TelemetryBuffer:
    """
    Memory-efficient circular buffer for real-time telemetry processing.
    Maintains a fixed-size window of recent telemetry data with automatic
    feature engineering and sequence preparation for ML models.
    """
    
    def __init__(self, max_size: int = 30, features: List[str] = None):
        """
        Initialize the telemetry buffer.
        
        Args:
            max_size: Maximum number of samples to store
            features: List of feature names to track
        """
        self.max_size = max_size
        self.features = features or ['alt', 'vel_v', 'lat', 'lon', 'horizontal_speed']
        
        # Core data storage (circular buffer)
        self.buffer = np.zeros((max_size, len(self.features)), dtype=np.float32)
        self.timestamps = np.zeros(max_size, dtype=np.float64)
        
        # Buffer state
        self.current_idx = 0
        self.is_full = False
        self.sample_count = 0
        
        # Feature engineering cache
        self._feature_cache = {}
        self._last_sample = None
        
        # Statistics tracking
        self.stats = {
            'samples_added': 0,
            'feature_computation_time': 0,
            'average_sample_rate': 0
        }
        
        logging.info(f"TelemetryBuffer initialized: {max_size} samples, {len(self.features)} features")
    
    def add_sample(self, telemetry_dict: Dict, timestamp: float = None) -> bool:
        """
        Add a new telemetry sample to the buffer.
        
        Args:
            telemetry_dict: Dictionary containing telemetry data
            timestamp: Optional timestamp, current time if None
            
        Returns:
            True if sample was added successfully
        """
        try:
            # Extract features and handle missing values
            feature_vector = np.zeros(len(self.features), dtype=np.float32)
            
            for i, feature_name in enumerate(self.features):
                if feature_name in telemetry_dict:
                    value = telemetry_dict[feature_name]
                    # Handle potential string/None values
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector[i] = float(value)
                    else:
                        # Use last known value or zero
                        if self.sample_count > 0:
                            prev_idx = (self.current_idx - 1) % self.max_size
                            feature_vector[i] = self.buffer[prev_idx, i]
                        else:
                            feature_vector[i] = 0.0
                else:
                    # Missing feature - use interpolation or default
                    feature_vector[i] = self._estimate_missing_feature(feature_name)
            
            # Store in circular buffer
            self.buffer[self.current_idx] = feature_vector
            self.timestamps[self.current_idx] = timestamp or self._get_current_time()
            
            # Update indices and state
            self.current_idx = (self.current_idx + 1) % self.max_size
            if self.current_idx == 0:
                self.is_full = True
            
            self.sample_count += 1
            self.stats['samples_added'] += 1
            
            # Cache this sample for feature engineering
            self._last_sample = telemetry_dict.copy()
            
            # Update statistics
            self._update_sample_rate()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to add telemetry sample: {e}")
            return False
    
    def get_sequence(self, length: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get the most recent sequence of samples for ML processing.
        
        Args:
            length: Sequence length (defaults to buffer size)
            
        Returns:
            Array of shape (length, n_features) or None if insufficient data
        """
        if not self.is_full and self.sample_count < (length or self.max_size):
            return None
        
        seq_length = length or self.max_size
        seq_length = min(seq_length, self.sample_count)
        
        if self.is_full:
            # Buffer is full - return properly ordered sequence
            sequence = np.concatenate([
                self.buffer[self.current_idx:],
                self.buffer[:self.current_idx]
            ])
            return sequence[-seq_length:]
        else:
            # Buffer not full - return available data
            return self.buffer[:self.sample_count][-seq_length:]
    
    def get_latest_sample(self) -> Optional[Dict]:
        """Get the most recently added sample as a dictionary."""
        if self.sample_count == 0:
            return None
        
        latest_idx = (self.current_idx - 1) % self.max_size
        latest_features = self.buffer[latest_idx]
        
        return {
            feature_name: float(value) 
            for feature_name, value in zip(self.features, latest_features)
        }
    
    def get_engineered_features(self) -> Dict[str, float]:
        """
        Calculate advanced features from the current buffer state.
        These are features that require temporal context.
        """
        if self.sample_count < 2:
            return {}
        
        sequence = self.get_sequence()
        if sequence is None:
            return {}
        
        features = {}
        
        try:
            # Get feature indices
            alt_idx = self.features.index('alt') if 'alt' in self.features else None
            vel_idx = self.features.index('vel_v') if 'vel_v' in self.features else None
            lat_idx = self.features.index('lat') if 'lat' in self.features else None
            lon_idx = self.features.index('lon') if 'lon' in self.features else None
            
            # Temporal derivatives
            if alt_idx is not None:
                alt_series = sequence[:, alt_idx]
                features['altitude_trend'] = np.polyfit(range(len(alt_series)), alt_series, 1)[0]
                features['altitude_acceleration'] = np.diff(alt_series, n=2).mean() if len(alt_series) > 2 else 0
            
            if vel_idx is not None:
                vel_series = sequence[:, vel_idx]
                features['velocity_stability'] = np.std(vel_series)
                features['velocity_trend'] = np.polyfit(range(len(vel_series)), vel_series, 1)[0]
            
            # Spatial features
            if lat_idx is not None and lon_idx is not None:
                lat_series = sequence[:, lat_idx]
                lon_series = sequence[:, lon_idx]
                
                # Calculate path curvature
                if len(lat_series) > 2:
                    lat_diff = np.diff(lat_series)
                    lon_diff = np.diff(lon_series)
                    curvature = np.diff(np.arctan2(lat_diff, lon_diff))
                    features['trajectory_curvature'] = np.mean(np.abs(curvature))
                else:
                    features['trajectory_curvature'] = 0
                
                # Horizontal velocity
                if len(lat_series) > 1:
                    lat_vel = np.diff(lat_series) * 111320  # Approximate meters
                    lon_vel = np.diff(lon_series) * 111320 * np.cos(np.radians(lat_series[:-1]))
                    horizontal_speed = np.sqrt(lat_vel**2 + lon_vel**2)
                    features['current_horizontal_speed'] = horizontal_speed[-1] if len(horizontal_speed) > 0 else 0
                    features['horizontal_speed_trend'] = np.polyfit(range(len(horizontal_speed)), horizontal_speed, 1)[0] if len(horizontal_speed) > 1 else 0
            
            # Stability indicators
            features['overall_stability'] = 1.0 / (1.0 + np.mean(np.std(sequence, axis=0)))
            
            # Time-based features
            if len(self.timestamps) > 1:
                recent_timestamps = self.timestamps[:self.sample_count] if not self.is_full else np.concatenate([
                    self.timestamps[self.current_idx:], self.timestamps[:self.current_idx]
                ])
                time_diffs = np.diff(recent_timestamps[-10:])  # Last 10 intervals
                features['sample_rate_consistency'] = 1.0 / (1.0 + np.std(time_diffs)) if len(time_diffs) > 0 else 1.0
            
        except Exception as e:
            logging.warning(f"Error calculating engineered features: {e}")
        
        return features
    
    def get_prediction_ready_data(self, scaler=None) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Get data ready for ML model prediction.
        
        Args:
            scaler: Optional sklearn scaler to apply
            
        Returns:
            Tuple of (scaled_sequence, engineered_features_dict)
        """
        sequence = self.get_sequence()
        if sequence is None:
            return None, {}
        
        # Apply scaling if provided
        if scaler is not None:
            try:
                sequence_scaled = scaler.transform(sequence)
            except Exception as e:
                logging.warning(f"Scaling failed: {e}")
                sequence_scaled = sequence
        else:
            sequence_scaled = sequence
        
        # Get engineered features
        engineered = self.get_engineered_features()
        
        return sequence_scaled, engineered
    
    def _estimate_missing_feature(self, feature_name: str) -> float:
        """Estimate value for missing feature based on context."""
        if self.sample_count == 0:
            # Default values for first sample
            defaults = {
                'alt': 2000.0,
                'vel_v': -6.0,
                'lat': 40.7128,
                'lon': -74.0060,
                'horizontal_speed': 5.0
            }
            return defaults.get(feature_name, 0.0)
        
        # Use last known value
        prev_idx = (self.current_idx - 1) % self.max_size
        try:
            feature_idx = self.features.index(feature_name)
            return self.buffer[prev_idx, feature_idx]
        except ValueError:
            return 0.0
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def _update_sample_rate(self):
        """Update sample rate statistics."""
        if self.sample_count > 1:
            recent_idx = (self.current_idx - 1) % self.max_size
            prev_idx = (self.current_idx - 2) % self.max_size
            
            time_diff = self.timestamps[recent_idx] - self.timestamps[prev_idx]
            if time_diff > 0:
                current_rate = 1.0 / time_diff
                # Exponential moving average
                alpha = 0.1
                self.stats['average_sample_rate'] = (
                    alpha * current_rate + 
                    (1 - alpha) * self.stats['average_sample_rate']
                )
    
    def get_buffer_info(self) -> Dict:
        """Get information about buffer state."""
        return {
            'max_size': self.max_size,
            'current_size': min(self.sample_count, self.max_size),
            'is_full': self.is_full,
            'sample_count': self.sample_count,
            'features': self.features,
            'stats': self.stats.copy()
        }
    
    def clear(self):
        """Clear the buffer and reset state."""
        self.buffer.fill(0)
        self.timestamps.fill(0)
        self.current_idx = 0
        self.is_full = False
        self.sample_count = 0
        self._feature_cache.clear()
        self._last_sample = None
        
        self.stats = {
            'samples_added': 0,
            'feature_computation_time': 0,
            'average_sample_rate': 0
        }
        
        logging.info("TelemetryBuffer cleared")


class FlightStateTracker:
    """
    Tracks flight state and provides context for predictions.
    Complements the TelemetryBuffer with higher-level state management.
    """
    
    def __init__(self):
        self.flight_phase = "UNKNOWN"
        self.deployment_altitude = None
        self.deployment_time = None
        self.prediction_confidence = 0.0
        self.last_valid_prediction = None
        self.telemetry_gaps = 0
        self.consecutive_good_samples = 0
        
        # State transition thresholds
        self.phase_thresholds = {
            'ASCENT_MIN_VELOCITY': 1.0,
            'DESCENT_MAX_VELOCITY': -0.5,
            'LANDED_MAX_ALTITUDE': 10.0,
            'LANDED_MIN_VELOCITY': 0.5
        }
        
        logging.info("FlightStateTracker initialized")
    
    def update(self, telemetry_dict: Dict) -> str:
        """
        Update flight state based on new telemetry.
        
        Returns:
            Current flight phase
        """
        try:
            alt = telemetry_dict.get('alt', 0)
            vel_v = telemetry_dict.get('vel_v', 0)
            
            # Detect phase transitions
            previous_phase = self.flight_phase
            
            if vel_v > self.phase_thresholds['ASCENT_MIN_VELOCITY']:
                self.flight_phase = "ASCENT"
            elif vel_v < self.phase_thresholds['DESCENT_MAX_VELOCITY']:
                if self.flight_phase != "DESCENT":
                    # First time entering descent - record deployment
                    self.deployment_altitude = alt
                    self.deployment_time = self._get_current_time()
                    logging.info(f"Descent phase detected at {alt:.1f}m")
                
                self.flight_phase = "DESCENT"
            elif (alt < self.phase_thresholds['LANDED_MAX_ALTITUDE'] and 
                  abs(vel_v) < self.phase_thresholds['LANDED_MIN_VELOCITY']):
                self.flight_phase = "LANDED"
            
            # Log phase transitions
            if previous_phase != self.flight_phase:
                logging.info(f"Flight phase transition: {previous_phase} -> {self.flight_phase}")
            
            # Update data quality metrics
            self._update_data_quality(telemetry_dict)
            
            return self.flight_phase
            
        except Exception as e:
            logging.warning(f"Error updating flight state: {e}")
            return self.flight_phase
    
    def should_predict(self) -> bool:
        """
        Determine if conditions are suitable for making predictions.
        """
        return (
            self.flight_phase == "DESCENT" and
            self.consecutive_good_samples >= 5 and
            self.prediction_confidence > 0.3
        )
    
    def get_flight_context(self) -> Dict:
        """
        Get contextual information about the current flight.
        """
        context = {
            'phase': self.flight_phase,
            'deployment_altitude': self.deployment_altitude,
            'deployment_time': self.deployment_time,
            'prediction_confidence': self.prediction_confidence,
            'data_quality': self.consecutive_good_samples,
            'telemetry_gaps': self.telemetry_gaps
        }
        
        # Calculate derived metrics
        if self.deployment_time:
            context['time_since_deployment'] = self._get_current_time() - self.deployment_time
        
        return context
    
    def update_prediction_confidence(self, prediction_result: Dict):
        """
        Update prediction confidence based on model outputs.
        """
        if prediction_result:
            # Simple confidence based on model agreement and stability
            stability = prediction_result.get('stability_score', 0.5)
            uncertainty = prediction_result.get('uncertainty', [0.5, 0.5])
            avg_uncertainty = np.mean(uncertainty) if hasattr(uncertainty, '__len__') else uncertainty
            
            # Lower uncertainty means higher confidence
            confidence = stability * (1.0 - min(avg_uncertainty, 1.0))
            
            # Smooth the confidence with exponential moving average
            alpha = 0.2
            self.prediction_confidence = (
                alpha * confidence + 
                (1 - alpha) * self.prediction_confidence
            )
            
            self.last_valid_prediction = prediction_result
    
    def _update_data_quality(self, telemetry_dict: Dict):
        """Update data quality metrics."""
        required_fields = ['lat', 'lon', 'alt', 'vel_v']
        
        # Check if sample has all required fields with valid values
        is_good_sample = True
        for field in required_fields:
            if field not in telemetry_dict:
                is_good_sample = False
                break
            
            value = telemetry_dict[field]
            if not isinstance(value, (int, float)) or np.isnan(value):
                is_good_sample = False
                break
        
        if is_good_sample:
            self.consecutive_good_samples += 1
            self.telemetry_gaps = 0
        else:
            self.consecutive_good_samples = 0
            self.telemetry_gaps += 1
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()