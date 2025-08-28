# ml/wind_estimation.py
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from prediction_engine import calculate_new_coords

@dataclass
class WindLayer:
    """Represents a wind layer with altitude bounds and wind vector."""
    altitude_min: float
    altitude_max: float
    wind_speed: float
    wind_direction: float  # Direction FROM (meteorological convention)
    confidence: float = 1.0
    last_update: float = 0.0


class ParticleFilter:
    """
    Particle filter for wind field estimation using trajectory observations.
    """
    
    def __init__(self, n_particles: int = 1000, altitude_layers: List[Tuple[float, float]] = None):
        """
        Initialize particle filter for wind estimation.
        
        Args:
            n_particles: Number of particles to use
            altitude_layers: List of (min_alt, max_alt) tuples defining wind layers
        """
        self.n_particles = n_particles
        
        # Define altitude layers if not provided
        if altitude_layers is None:
            altitude_layers = [
                (2000, 3000),  # High altitude
                (1000, 2000),  # Mid altitude  
                (500, 1000),   # Low-mid altitude
                (0, 500)       # Low altitude
            ]
        
        self.altitude_layers = altitude_layers
        self.n_layers = len(altitude_layers)
        
        # Initialize particles
        self.particles = self._initialize_particles()
        self.weights = np.ones(n_particles) / n_particles
        
        # State tracking
        self.effective_sample_size_threshold = n_particles / 2
        self.resampling_noise = 0.1
        
        # Wind field estimation
        self.wind_layers = [
            WindLayer(alt_min, alt_max, 5.0, 270.0)
            for alt_min, alt_max in altitude_layers
        ]
        
        logging.info(f"Initialized ParticleFilter with {n_particles} particles and {self.n_layers} altitude layers")
    
    def _initialize_particles(self) -> np.ndarray:
        """Initialize particles with random wind fields."""
        # Each particle represents a complete wind field
        # Shape: (n_particles, n_layers, 2) where last dimension is [speed, direction]
        particles = np.zeros((self.n_particles, self.n_layers, 2))
        
        for i in range(self.n_particles):
            for j in range(self.n_layers):
                # Random wind speed (0-20 m/s)
                particles[i, j, 0] = np.random.exponential(scale=8.0)  # Speed
                particles[i, j, 0] = np.clip(particles[i, j, 0], 0.5, 25.0)
                
                # Random wind direction (0-360 degrees)
                particles[i, j, 1] = np.random.uniform(0, 360)  # Direction
        
        return particles
    
    def predict(self, dt: float = 1.0):
        """
        Predict step: evolve particles forward in time.
        
        Args:
            dt: Time step in seconds
        """
        # Add process noise to wind field
        speed_noise = np.random.normal(0, 0.5 * dt, (self.n_particles, self.n_layers))
        direction_noise = np.random.normal(0, 5.0 * dt, (self.n_particles, self.n_layers))
        
        # Update particles
        self.particles[:, :, 0] += speed_noise
        self.particles[:, :, 1] += direction_noise
        
        # Apply constraints
        self.particles[:, :, 0] = np.clip(self.particles[:, :, 0], 0.1, 30.0)  # Speed bounds
        self.particles[:, :, 1] = self.particles[:, :, 1] % 360  # Wrap direction
    
    def update(self, telemetry_sequence: List[Dict]) -> Dict:
        """
        Update step: weight particles based on observed trajectory.
        
        Args:
            telemetry_sequence: Recent telemetry data (at least 2 points)
            
        Returns:
            Dictionary with current wind field estimate
        """
        if len(telemetry_sequence) < 2:
            logging.warning("Need at least 2 telemetry points for wind estimation")
            return self.get_wind_estimate()
        
        # Calculate observed horizontal movement
        observed_movement = self._calculate_observed_movement(telemetry_sequence)
        
        if observed_movement is None:
            return self.get_wind_estimate()
        
        # Update particle weights based on how well they predict the observed movement
        log_weights = np.zeros(self.n_particles)
        
        for i, particle in enumerate(self.particles):
            predicted_movement = self._predict_movement_for_particle(
                particle, telemetry_sequence
            )
            
            if predicted_movement is not None:
                # Calculate likelihood (Gaussian error model)
                error = np.linalg.norm(np.array(predicted_movement) - np.array(observed_movement))
                log_weights[i] = -0.5 * (error / 10.0) ** 2  # 10m standard deviation
        
        # Normalize weights
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        self.weights = weights / np.sum(weights)
        
        # Resample if effective sample size is too low
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size < self.effective_sample_size_threshold:
            self._resample()
        
        # Update wind layer estimates
        self._update_wind_layers()
        
        return self.get_wind_estimate()
    
    def _calculate_observed_movement(self, telemetry_sequence: List[Dict]) -> Optional[Tuple[float, float]]:
        """Calculate observed horizontal movement between telemetry points."""
        try:
            # Use last two points
            p1 = telemetry_sequence[-2]
            p2 = telemetry_sequence[-1]
            
            lat1, lon1 = p1['lat'], p1['lon']
            lat2, lon2 = p2['lat'], p2['lon']
            
            # Convert to meters (approximate)
            lat_diff_m = (lat2 - lat1) * 111320
            lon_diff_m = (lon2 - lon1) * 111320 * np.cos(np.radians((lat1 + lat2) / 2))
            
            return (lat_diff_m, lon_diff_m)
            
        except KeyError as e:
            logging.warning(f"Missing required telemetry field: {e}")
            return None
    
    def _predict_movement_for_particle(self, particle_wind_field: np.ndarray, 
                                     telemetry_sequence: List[Dict]) -> Optional[Tuple[float, float]]:
        """Predict horizontal movement using a particle's wind field."""
        try:
            current_state = telemetry_sequence[-1]
            altitude = current_state['alt']
            dt = 1.0  # Assume 1 second between samples
            
            # Find appropriate wind layer
            layer_idx = self._get_wind_layer_for_altitude(altitude)
            if layer_idx is None:
                return None
            
            wind_speed = particle_wind_field[layer_idx, 0]
            wind_direction = particle_wind_field[layer_idx, 1]
            
            # Calculate drift
            drift_direction = (wind_direction + 180) % 360  # Wind direction to drift direction
            drift_distance = wind_speed * dt
            
            # Convert to lat/lon movement
            drift_lat = drift_distance * np.cos(np.radians(drift_direction)) / 111320
            drift_lon = (drift_distance * np.sin(np.radians(drift_direction)) / 
                        (111320 * np.cos(np.radians(current_state['lat']))))
            
            # Convert to meters
            drift_lat_m = drift_lat * 111320
            drift_lon_m = drift_lon * 111320 * np.cos(np.radians(current_state['lat']))
            
            return (drift_lat_m, drift_lon_m)
            
        except Exception as e:
            logging.warning(f"Error predicting movement for particle: {e}")
            return None
    
    def _get_wind_layer_for_altitude(self, altitude: float) -> Optional[int]:
        """Find the wind layer index for a given altitude."""
        for i, (alt_min, alt_max) in enumerate(self.altitude_layers):
            if alt_min <= altitude <= alt_max:
                return i
        return None
    
    def _resample(self):
        """Resample particles based on weights."""
        # Systematic resampling
        n = self.n_particles
        indices = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        u = np.random.uniform(0, 1.0/n)
        
        for j in range(n):
            while cumulative_sum[i] < u:
                i += 1
            indices[j] = i
            u += 1.0/n
        
        # Resample particles
        self.particles = self.particles[indices].copy()
        
        # Add resampling noise
        noise_std = self.resampling_noise
        speed_noise = np.random.normal(0, noise_std, (n, self.n_layers))
        direction_noise = np.random.normal(0, noise_std * 10, (n, self.n_layers))
        
        self.particles[:, :, 0] += speed_noise
        self.particles[:, :, 1] += direction_noise
        
        # Apply constraints
        self.particles[:, :, 0] = np.clip(self.particles[:, :, 0], 0.1, 30.0)
        self.particles[:, :, 1] = self.particles[:, :, 1] % 360
        
        # Reset weights
        self.weights = np.ones(n) / n
        
        logging.debug("Resampled particles")
    
    def _update_wind_layers(self):
        """Update wind layer estimates based on weighted particles."""
        current_time = self._get_current_time()
        
        for layer_idx in range(self.n_layers):
            # Weighted average of particle values for this layer
            weighted_speeds = self.particles[:, layer_idx, 0] * self.weights
            weighted_directions = self.particles[:, layer_idx, 1] * self.weights
            
            estimated_speed = np.sum(weighted_speeds)
            
            # Handle circular mean for wind direction
            weighted_sin = np.sum(np.sin(np.radians(weighted_directions)))
            weighted_cos = np.sum(np.cos(np.radians(weighted_directions)))
            estimated_direction = np.degrees(np.arctan2(weighted_sin, weighted_cos)) % 360
            
            # Update wind layer
            alt_min, alt_max = self.altitude_layers[layer_idx]
            self.wind_layers[layer_idx] = WindLayer(
                altitude_min=alt_min,
                altitude_max=alt_max,
                wind_speed=estimated_speed,
                wind_direction=estimated_direction,
                confidence=self._calculate_layer_confidence(layer_idx),
                last_update=current_time
            )
    
    def _calculate_layer_confidence(self, layer_idx: int) -> float:
        """Calculate confidence in wind estimate for a layer."""
        # Confidence based on particle agreement
        speeds = self.particles[:, layer_idx, 0]
        directions = self.particles[:, layer_idx, 1]
        
        # Weighted standard deviations
        weighted_mean_speed = np.average(speeds, weights=self.weights)
        weighted_var_speed = np.average((speeds - weighted_mean_speed)**2, weights=self.weights)
        speed_uncertainty = np.sqrt(weighted_var_speed)
        
        # Direction uncertainty (circular statistics)
        sin_dirs = np.sin(np.radians(directions))
        cos_dirs = np.cos(np.radians(directions))
        weighted_sin = np.average(sin_dirs, weights=self.weights)
        weighted_cos = np.average(cos_dirs, weights=self.weights)
        direction_consistency = np.sqrt(weighted_sin**2 + weighted_cos**2)
        
        # Combine uncertainties into confidence (0-1 scale)
        speed_confidence = 1.0 / (1.0 + speed_uncertainty / 5.0)  # Normalize by 5 m/s
        direction_confidence = direction_consistency  # Already 0-1
        
        return (speed_confidence + direction_confidence) / 2.0
    
    def get_wind_estimate(self) -> Dict:
        """Get current wind field estimate."""
        return {
            'layers': [
                {
                    'altitude_range': (layer.altitude_min, layer.altitude_max),
                    'wind_speed': layer.wind_speed,
                    'wind_direction': layer.wind_direction,
                    'confidence': layer.confidence,
                    'last_update': layer.last_update
                }
                for layer in self.wind_layers
            ],
            'effective_sample_size': 1.0 / np.sum(self.weights**2),
            'total_particles': self.n_particles
        }
    
    def predict_trajectory_with_uncertainty(self, initial_state: Dict, timesteps: int = 60) -> Dict:
        """
        Predict future trajectory with uncertainty using current wind estimate.
        
        Args:
            initial_state: Current telemetry state
            timesteps: Number of future timesteps to predict
            
        Returns:
            Dictionary with trajectory predictions and uncertainty bounds
        """
        n_ensemble = min(100, self.n_particles)  # Use subset of particles for efficiency
        
        # Select particles based on weights
        selected_indices = np.random.choice(
            self.n_particles, 
            size=n_ensemble, 
            p=self.weights
        )
        
        trajectories = []
        
        for idx in selected_indices:
            particle_wind = self.particles[idx]
            trajectory = self._simulate_trajectory_with_wind(
                initial_state, particle_wind, timesteps
            )
            trajectories.append(trajectory)
        
        # Convert to numpy array for statistics
        trajectories = np.array(trajectories)  # Shape: (n_ensemble, timesteps, 3) [lat, lon, alt]
        
        # Calculate statistics
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)
        
        # Calculate confidence ellipses at different time points
        confidence_ellipses = []
        for t in range(timesteps):
            if mean_trajectory[t, 2] > 0:  # Only if still airborne
                positions = trajectories[:, t, :2]  # lat, lon positions
                confidence_ellipses.append(self._calculate_confidence_ellipse(positions))
            else:
                confidence_ellipses.append(None)
        
        return {
            'mean_trajectory': mean_trajectory,
            'std_trajectory': std_trajectory,
            'confidence_ellipses': confidence_ellipses,
            'individual_trajectories': trajectories,
            'landing_distribution': self._analyze_landing_distribution(trajectories)
        }
    
    def _simulate_trajectory_with_wind(self, initial_state: Dict, wind_field: np.ndarray, 
                                     timesteps: int) -> np.ndarray:
        """Simulate a single trajectory using a specific wind field."""
        trajectory = np.zeros((timesteps, 3))  # lat, lon, alt
        
        # Initialize state
        lat, lon, alt = initial_state['lat'], initial_state['lon'], initial_state['alt']
        velocity = abs(initial_state.get('vel_v', 6.0))  # Descent rate
        
        trajectory[0] = [lat, lon, alt]
        
        for t in range(1, timesteps):
            if alt <= 0:
                trajectory[t] = [lat, lon, 0]
                continue
            
            # Get wind for current altitude
            layer_idx = self._get_wind_layer_for_altitude(alt)
            if layer_idx is not None:
                wind_speed = wind_field[layer_idx, 0]
                wind_direction = wind_field[layer_idx, 1]
            else:
                wind_speed, wind_direction = 5.0, 270.0  # Default
            
            # Update altitude
            alt -= velocity  # Simple descent model
            alt = max(0, alt)
            
            # Apply wind drift
            if alt > 0:
                drift_direction = (wind_direction + 180) % 360
                drift_distance = wind_speed * 1.0  # 1 second timestep
                lat, lon = calculate_new_coords(lat, lon, drift_direction, drift_distance)
            
            trajectory[t] = [lat, lon, alt]
        
        return trajectory
    
    def _calculate_confidence_ellipse(self, positions: np.ndarray, confidence: float = 0.95) -> Dict:
        """Calculate confidence ellipse for a set of positions."""
        if len(positions) < 3:
            return None
        
        # Calculate covariance matrix
        cov_matrix = np.cov(positions.T)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
        
        # Calculate ellipse parameters
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence, df=2)
        
        # Semi-axes lengths
        a = np.sqrt(chi2_val * eigenvals[0])  # Major axis
        b = np.sqrt(chi2_val * eigenvals[1])  # Minor axis
        
        # Rotation angle
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Center
        center = np.mean(positions, axis=0)
        
        return {
            'center': center,
            'semi_major_axis': a,
            'semi_minor_axis': b,
            'rotation_angle': angle,
            'confidence_level': confidence
        }
    
    def _analyze_landing_distribution(self, trajectories: np.ndarray) -> Dict:
        """Analyze the distribution of predicted landing points."""
        # Find landing points (where altitude reaches 0)
        landing_points = []
        
        for trajectory in trajectories:
            # Find first point where altitude is 0
            ground_indices = np.where(trajectory[:, 2] <= 0)[0]
            if len(ground_indices) > 0:
                landing_idx = ground_indices[0]
                landing_points.append(trajectory[landing_idx, :2])  # lat, lon
            else:
                # Use last point if never reaches ground
                landing_points.append(trajectory[-1, :2])
        
        landing_points = np.array(landing_points)
        
        if len(landing_points) == 0:
            return {}
        
        # Calculate statistics
        mean_landing = np.mean(landing_points, axis=0)
        std_landing = np.std(landing_points, axis=0)
        
        # Calculate landing confidence ellipse
        landing_ellipse = self._calculate_confidence_ellipse(landing_points, confidence=0.95)
        
        # Calculate dispersion metrics
        distances_from_mean = np.linalg.norm(landing_points - mean_landing, axis=1)
        
        return {
            'mean_landing_point': mean_landing,
            'landing_std': std_landing,
            'confidence_ellipse': landing_ellipse,
            'max_dispersion': np.max(distances_from_mean),
            'mean_dispersion': np.mean(distances_from_mean),
            'landing_points': landing_points.tolist()
        }
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


class AdaptiveWindEstimator:
    """
    High-level interface for adaptive wind field estimation.
    Combines multiple estimation techniques for robust performance.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the adaptive wind estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize particle filter
        self.particle_filter = ParticleFilter(
            n_particles=self.config['n_particles'],
            altitude_layers=self.config['altitude_layers']
        )
        
        # Historical data for trend analysis
        self.telemetry_history = []
        self.wind_estimates_history = []
        
        # Performance tracking
        self.prediction_errors = []
        self.last_prediction = None
        
        logging.info("AdaptiveWindEstimator initialized")
    
    def update(self, telemetry_data: Dict) -> Dict:
        """
        Update wind estimate with new telemetry data.
        
        Args:
            telemetry_data: Current telemetry reading
            
        Returns:
            Updated wind field estimate
        """
        # Add to history
        self.telemetry_history.append(telemetry_data)
        
        # Keep only recent history
        max_history = self.config['max_history_length']
        if len(self.telemetry_history) > max_history:
            self.telemetry_history = self.telemetry_history[-max_history:]
        
        # Update particle filter
        wind_estimate = self.particle_filter.update(self.telemetry_history)
        
        # Store estimate
        self.wind_estimates_history.append({
            'timestamp': self._get_current_time(),
            'estimate': wind_estimate
        })
        
        # Validate prediction if we had a previous one
        if self.last_prediction is not None:
            self._validate_prediction(telemetry_data)
        
        return wind_estimate
    
    def predict_landing_with_uncertainty(self, current_telemetry: Dict) -> Dict:
        """
        Predict landing point with comprehensive uncertainty quantification.
        
        Args:
            current_telemetry: Current telemetry state
            
        Returns:
            Landing prediction with uncertainty bounds
        """
        # Get trajectory prediction from particle filter
        trajectory_prediction = self.particle_filter.predict_trajectory_with_uncertainty(
            current_telemetry,
            timesteps=int(current_telemetry['alt'] / 6.0) + 10  # Adaptive timesteps
        )
        
        # Store prediction for validation
        self.last_prediction = {
            'timestamp': self._get_current_time(),
            'prediction': trajectory_prediction,
            'initial_state': current_telemetry.copy()
        }
        
        # Extract landing prediction
        landing_dist = trajectory_prediction['landing_distribution']
        
        if landing_dist:
            result = {
                'predicted_landing': landing_dist['mean_landing_point'],
                'uncertainty_ellipse': landing_dist['confidence_ellipse'],
                'max_error_estimate': landing_dist['max_dispersion'],
                'mean_error_estimate': landing_dist['mean_dispersion'],
                'confidence_level': 0.95,
                'wind_field': self.get_current_wind_field(),
                'trajectory_prediction': trajectory_prediction
            }
        else:
            # Fallback prediction
            result = {
                'predicted_landing': [current_telemetry['lat'], current_telemetry['lon']],
                'uncertainty_ellipse': None,
                'max_error_estimate': 1000.0,  # High uncertainty
                'mean_error_estimate': 500.0,
                'confidence_level': 0.1,
                'wind_field': self.get_current_wind_field(),
                'trajectory_prediction': trajectory_prediction
            }
        
        return result
    
    def get_current_wind_field(self) -> Dict:
        """Get the current best estimate of the wind field."""
        return self.particle_filter.get_wind_estimate()
    
    def _validate_prediction(self, actual_telemetry: Dict):
        """Validate previous prediction against actual telemetry."""
        if not self.last_prediction:
            return
        
        # Calculate how far off our prediction was
        pred_trajectory = self.last_prediction['prediction']['mean_trajectory']
        initial_alt = self.last_prediction['initial_state']['alt']
        current_alt = actual_telemetry['alt']
        
        # Find predicted position at current altitude
        alt_diff = initial_alt - current_alt
        timestep_estimate = int(alt_diff / 6.0)  # Assuming 6 m/s descent
        
        if 0 <= timestep_estimate < len(pred_trajectory):
            predicted_pos = pred_trajectory[timestep_estimate, :2]  # lat, lon
            actual_pos = [actual_telemetry['lat'], actual_telemetry['lon']]
            
            # Calculate error in meters
            from geopy.distance import geodesic
            error_m = geodesic(predicted_pos, actual_pos).meters
            
            self.prediction_errors.append(error_m)
            
            # Keep only recent errors
            if len(self.prediction_errors) > 100:
                self.prediction_errors = self.prediction_errors[-100:]
            
            logging.debug(f"Prediction validation: {error_m:.1f}m error")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the wind estimator."""
        if not self.prediction_errors:
            return {'status': 'insufficient_data'}
        
        errors = np.array(self.prediction_errors)
        
        return {
            'mean_error_m': np.mean(errors),
            'std_error_m': np.std(errors),
            'median_error_m': np.median(errors),
            'max_error_m': np.max(errors),
            'min_error_m': np.min(errors),
            'num_predictions_validated': len(errors),
            'recent_performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate whether performance is improving, degrading, or stable."""
        if len(self.prediction_errors) < 10:
            return 'insufficient_data'
        
        recent_errors = self.prediction_errors[-5:]
        older_errors = self.prediction_errors[-10:-5]
        
        recent_mean = np.mean(recent_errors)
        older_mean = np.mean(older_errors)
        
        improvement_threshold = 10.0  # meters
        
        if recent_mean < older_mean - improvement_threshold:
            return 'improving'
        elif recent_mean > older_mean + improvement_threshold:
            return 'degrading'
        else:
            return 'stable'
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'n_particles': 1000,
            'altitude_layers': [
                (2000, 3000),
                (1000, 2000),
                (500, 1000),
                (0, 500)
            ],
            'max_history_length': 60,
            'resampling_threshold': 0.5
        }
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


# Example usage and integration helper
def integrate_wind_estimator_with_main_app():
    """
    Example of how to integrate the adaptive wind estimator with the main application.
    """
    
    # Initialize wind estimator
    wind_estimator = AdaptiveWindEstimator()
    
    # Example main loop integration
    def enhanced_prediction_loop(telemetry_stream):
        for telemetry_string in telemetry_stream:
            try:
                telemetry_packet = json.loads(telemetry_string)
                
                if telemetry_packet.get('vel_v', 0) < 0:  # Descending
                    
                    # Update wind field estimate
                    wind_estimate = wind_estimator.update(telemetry_packet)
                    
                    # Get enhanced prediction with wind uncertainty
                    enhanced_prediction = wind_estimator.predict_landing_with_uncertainty(
                        telemetry_packet
                    )
                    
                    # Log results
                    logging.info(f"Enhanced prediction at {telemetry_packet['alt']:.1f}m:")
                    logging.info(f"  Landing: {enhanced_prediction['predicted_landing']}")
                    logging.info(f"  Error estimate: ±{enhanced_prediction['mean_error_estimate']:.1f}m")
                    
                    # Get performance metrics periodically
                    if telemetry_packet.get('time_since_deployment', 0) % 30 == 0:
                        perf = wind_estimator.get_performance_metrics()
                        logging.info(f"Wind estimator performance: {perf}")
                
            except Exception as e:
                logging.error(f"Error in enhanced prediction loop: {e}")
    
    return enhanced_prediction_loop