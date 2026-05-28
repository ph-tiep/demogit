# -*- coding: utf-8 -*-
"""
Digital Twin Module based on Extended Kalman Filter (EKF)
Replaces Random Forest (classification) and Isolation Forest (anomaly detection)
with a physics-informed state estimation model.
Optimized for normalized [0,1] input data from MinMaxScaler.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EKFConfig:
    """Configuration for EKF Digital Twin"""
    state_dim: int = 3
    obs_dim: int = 5
    process_noise: float = 0.01
    measurement_noise: float = 0.05
    innovation_threshold: float = 3.5
    min_innovation_history: int = 50


class DigitalTwinEKF:
    """
    Digital Twin model using Extended Kalman Filter (EKF).

    State vector: x = [connection_quality, rssi_quality, bs_quality]
    Observation:   z = [mean_rssi, num_active_bs, lat, lon, hour]  (all normalized [0,1])

    Works with normalized data directly.
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        if config is None:
            config = EKFConfig()
        self.config = config

        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim

        # State: x = [quality, rssi_component, bs_component]
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)

        # Noise covariances tuned for normalized [0,1] data
        self.Q = np.diag([config.process_noise] * self.state_dim)
        self.R = np.diag([config.measurement_noise] * self.obs_dim)

        # Tracking
        self.step_count = 0
        self.innovation_history = []
        self.chi2_history = []
        self.quality_history = []

    def initialize(self, initial_observation: np.ndarray):
        """Initialize EKF with first observation (normalized data)"""
        # observation: [mean_rssi, num_active_bs, lat, lon, hour] - all in [0,1]
        mean_rssi = initial_observation[0]    # normalized
        num_active_bs = initial_observation[1]  # normalized
        # hour in [0,1]
        self.x[0] = 0.5 * mean_rssi + 0.4 * num_active_bs + 0.1  # initial quality estimate
        self.x[1] = mean_rssi  # rssi component
        self.x[2] = num_active_bs  # bs component
        self.P = np.eye(self.state_dim) * 0.01

    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        """
        State transition: quality slowly decays toward prior, trends decay.
        """
        F = np.array([
            [0.98, 0.02, 0.0],
            [0.0, 0.9,  0.0],
            [0.0, 0.0,  0.9]
        ])
        return F @ x

    def _state_jacobian(self) -> np.ndarray:
        """Jacobian of state transition"""
        return np.array([
            [0.98, 0.02, 0.0],
            [0.0,  0.9,  0.0],
            [0.0,  0.0,  0.9]
        ])

    def _observation_model(self, x: np.ndarray, obs_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Observation model for normalized [0,1] data.
        z = H @ x + observation noise
        Maps state back to normalized observations.
        """
        quality = np.clip(x[0], 0.0, 1.0)
        rssi_comp = np.clip(x[1], 0.0, 1.0)
        bs_comp = np.clip(x[2], 0.0, 1.0)

        # Expected observations based on state
        expected_rssi = rssi_comp
        expected_bs = bs_comp

        if obs_params is not None:
            expected_lat = obs_params[2]
            expected_lon = obs_params[3]
            expected_hour = obs_params[4]
        else:
            expected_lat = 0.5
            expected_lon = 0.5
            expected_hour = 0.5

        return np.array([
            expected_rssi,
            expected_bs,
            expected_lat,
            expected_lon,
            expected_hour
        ])

    def _observation_jacobian(self) -> np.ndarray:
        """
        Jacobian of observation model.
        Maps state to observation space.
        """
        H = np.zeros((self.obs_dim, self.state_dim))
        H[0, 1] = 1.0  # rssi_obs = rssi_component
        H[1, 2] = 1.0  # bs_obs = bs_component
        H[2, 0] = 0.0  # lat from params
        H[2, 1] = 0.0
        H[2, 2] = 0.0
        H[3, 0] = 0.0  # lon from params
        H[3, 1] = 0.0
        H[3, 2] = 0.0
        H[4, 0] = 0.0  # hour from params
        H[4, 1] = 0.0
        H[4, 2] = 0.0
        return H

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """EKF Predict step"""
        F = self._state_jacobian()
        self.x = self._state_transition(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, observation: np.ndarray, obs_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        EKF Update step.
        Returns: (state, covariance, innovation, mahalanobis)
        """
        H = self._observation_jacobian()
        expected_obs = self._observation_model(self.x, obs_params)

        innovation = observation - expected_obs
        S = H @ self.P @ H.T + self.R

        # Regularize S to prevent singularity
        S_reg = S + np.eye(self.obs_dim) * 1e-6

        try:
            S_inv = np.linalg.inv(S_reg)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            K = np.zeros((self.state_dim, self.obs_dim))

        self.x = self.x + K @ innovation
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        # Ensure P stays positive semi-definite
        eigvals = np.linalg.eigvalsh(self.P)
        if np.any(eigvals < 1e-10):
            self.P = np.eye(self.state_dim) * 1e-4

        # Mahalanobis distance
        try:
            mahalanobis = innovation @ S_inv @ innovation
        except (np.linalg.LinAlgError, ValueError):
            mahalanobis = np.linalg.norm(innovation)

        self.step_count += 1
        self.innovation_history.append(float(np.linalg.norm(innovation)))
        self.chi2_history.append(float(mahalanobis))
        self.quality_history.append(float(np.clip(self.x[0], 0.0, 1.0)))

        return self.x.copy(), self.P.copy(), innovation.copy(), mahalanobis

    def step(self, observation: np.ndarray, obs_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float, bool]:
        """
        Full EKF step: predict -> update.

        Returns:
            state: Updated state estimate
            quality: Predicted connection quality [0, 1]
            anomaly_score: Mahalanobis distance
            is_anomaly: Boolean flag
        """
        if self.step_count == 0:
            self.initialize(observation)

        self.predict()
        _, _, innovation, mahalanobis = self.update(observation, obs_params)

        quality = float(np.clip(self.x[0], 0.0, 1.0))

        # Adaptive threshold
        threshold = self._compute_adaptive_threshold()
        is_anomaly = bool(mahalanobis > threshold)

        # Slow noise adaptation every 500 steps
        if self.step_count % 500 == 0:
            self._adapt_noise()

        return self.x.copy(), quality, float(mahalanobis), is_anomaly

    def _compute_adaptive_threshold(self) -> float:
        """Compute adaptive anomaly threshold"""
        if len(self.chi2_history) < self.config.min_innovation_history:
            return self.config.innovation_threshold

        recent = np.array(self.chi2_history[-self.config.min_innovation_history:])
        mean_val = np.mean(recent)
        std_val = np.std(recent)

        threshold = mean_val + self.config.innovation_threshold * max(std_val, 0.1)
        return float(np.clip(threshold, 1.0, 50.0))

    def _adapt_noise(self):
        """Slowly adapt process noise based on innovation magnitude"""
        if len(self.innovation_history) < 100:
            return

        recent = np.array(self.innovation_history[-100:])
        mean_innov = np.mean(recent)

        # Adapt Q (process noise)
        q_scale = np.clip(mean_innov / 2.0, 0.5, 5.0)
        self.Q = np.diag([self.config.process_noise * q_scale] * self.state_dim)

    def predict_batch(self, observations: np.ndarray,
                     obs_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process batch of observations sequentially."""
        n = len(observations)
        qualities = np.zeros(n)
        anomaly_scores = np.zeros(n)
        is_anomalies = np.zeros(n, dtype=bool)

        for i in range(n):
            params_i = obs_params[i] if obs_params is not None else None
            _, quality, score, is_anom = self.step(observations[i], params_i)
            qualities[i] = quality
            anomaly_scores[i] = score
            is_anomalies[i] = is_anom

        return qualities, anomaly_scores, is_anomalies

    def get_state_summary(self) -> dict:
        """Get current state summary"""
        return {
            'state': self.x.tolist(),
            'quality_estimate': float(np.clip(self.x[0], 0.0, 1.0)),
            'rssi_component': float(np.clip(self.x[1], 0.0, 1.0)),
            'bs_component': float(np.clip(self.x[2], 0.0, 1.0)),
            'step_count': self.step_count,
            'mean_innovation': float(np.mean(self.innovation_history[-100:])) if self.innovation_history else 0.0,
            'adaptive_threshold': float(self._compute_adaptive_threshold()) if self.step_count > 0 else self.config.innovation_threshold
        }

    def reset(self):
        """Reset EKF state"""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)
        self.step_count = 0
        self.innovation_history = []
        self.chi2_history = []
        self.quality_history = []
        self.Q = np.diag([self.config.process_noise] * self.state_dim)
        self.R = np.diag([self.config.measurement_noise] * self.obs_dim)


def train_digital_twin(observations: np.ndarray,
                       obs_params: Optional[np.ndarray] = None,
                       config: Optional[EKFConfig] = None) -> Tuple[DigitalTwinEKF, float]:
    """
    Initialize and run Digital Twin on data.
    Returns trained DigitalTwinEKF instance and initial quality.
    """
    if config is None:
        config = EKFConfig()

    dt = DigitalTwinEKF(config)

    burn_in = min(100, len(observations) // 10)
    for i in range(burn_in):
        params_i = obs_params[i] if obs_params is not None else None
        dt.step(observations[i], params_i)

    quality = float(np.clip(dt.x[0], 0.0, 1.0))
    return dt, quality
