# -*- coding: utf-8 -*-
"""
Digital Twin for IoT networks using Extended Kalman Filter (EKF).
Replaces Random Forest (classification) and Isolation Forest (anomaly detection)
with a physics-inspired, adaptive state estimator.

State vector: x = [mean_rssi, num_active_bs]
Observation:  z = [mean_rssi, num_active_bs]   (direct measurement)

The EKF tracks the "true" network QoS state over time.
Anomalies are flagged by the Normalized Innovation Squared (NIS) test.
Quality classification is derived from the filtered state estimate.
"""

import numpy as np
from scipy.stats import chi2
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)


class EKF:
    """
    Extended Kalman Filter for IoT state estimation.

    In our linear observation model the EKF reduces to a standard KF,
    but the class is structured for nonlinear extensions (Jacobian-ready).
    """

    def __init__(self, state_dim=2, obs_dim=2,
                 process_noise=0.5, obs_noise=1.5):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Initial state & covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 10.0

        # State-transition (constant-state model: x_{t+1} ≈ x_t)
        self.F = np.eye(state_dim)

        # Observation matrix (direct measurement of state)
        self.H = np.eye(obs_dim, state_dim)

        # Noise covariances
        self.Q = np.eye(state_dim) * process_noise   # process noise
        self.R = np.eye(obs_dim) * obs_noise          # observation noise

    def reset(self, x0):
        self.x = np.array(x0, dtype=float)
        self.P = np.eye(self.state_dim) * 10.0

    def predict(self):
        """Predict step: propagate state and covariance forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        Update step: incorporate new measurement z.
        Returns filtered state, innovation vector, NIS scalar.
        """
        z = np.array(z, dtype=float)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        # Normalized Innovation Squared (chi-squared statistic)
        nis = float(y.T @ np.linalg.inv(S) @ y)
        return self.x.copy(), y, nis

    def step(self, z):
        """One full predict-update cycle."""
        self.predict()
        return self.update(z)


class DigitalTwin:
    """
    Digital Twin of an IoT wireless network.

    Wraps an EKF to provide:
      - Continuous state estimation (filtered mean_rssi, num_active_bs)
      - Quality classification: 1 = good connection, 0 = poor
      - Anomaly detection via NIS chi-squared test
    """

    def __init__(self,
                 state_dim=2, obs_dim=2,
                 rssi_threshold=-110.0,
                 bs_threshold=3,
                 process_noise=0.5,
                 obs_noise=1.5,
                 anomaly_confidence=0.95):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.rssi_threshold = rssi_threshold
        self.bs_threshold = bs_threshold
        # Chi-squared threshold for NIS anomaly test
        self.chi2_thresh = chi2.ppf(anomaly_confidence, df=obs_dim)
        self.ekf = EKF(state_dim, obs_dim, process_noise, obs_noise)

    def reset(self, first_obs):
        self.ekf.reset(np.array(first_obs[:self.state_dim]))

    def classify_quality(self, state):
        """Rule-based quality label consistent with baseline labelling."""
        return int(state[0] > self.rssi_threshold and
                   state[1] >= self.bs_threshold)

    def run_sequence(self, observations):
        """
        Process an ordered sequence of observations.

        Parameters
        ----------
        observations : np.ndarray, shape [N, obs_dim]
            Each row is [mean_rssi, num_active_bs].

        Returns
        -------
        pd.DataFrame with columns:
            est_rssi, est_num_bs, obs_rssi, obs_num_bs,
            innov_rssi, innov_num_bs, nis, is_anomaly, quality_label
        """
        N = len(observations)
        if N == 0:
            return pd.DataFrame()

        self.reset(observations[0])
        rows = []

        for z in observations:
            state, innov, nis = self.ekf.step(z)
            rows.append({
                'est_rssi':      state[0],
                'est_num_bs':    state[1],
                'obs_rssi':      z[0],
                'obs_num_bs':    z[1],
                'innov_rssi':    innov[0],
                'innov_num_bs':  innov[1],
                'nis':           nis,
                'is_anomaly':    int(nis > self.chi2_thresh),
                'quality_label': self.classify_quality(state),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_quality(self, results_df, y_true):
        """Compare Digital Twin quality predictions against ground truth."""
        y_pred = results_df['quality_label'].values
        acc = accuracy_score(y_true, y_pred)

        # Use inverse-NIS as soft confidence score for quality = 1
        conf = 1.0 / (1.0 + results_df['nis'].values)
        try:
            auc = roc_auc_score(y_true, conf)
            fpr, tpr, _ = roc_curve(y_true, conf)
        except ValueError:
            auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])

        report = classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        rssi_rmse = np.sqrt(np.mean(
            (results_df['est_rssi'].values - results_df['obs_rssi'].values) ** 2))
        bs_rmse = np.sqrt(np.mean(
            (results_df['est_num_bs'].values - results_df['obs_num_bs'].values) ** 2))

        return {
            'accuracy': acc,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'report': report,
            'confusion_matrix': cm,
            'rssi_rmse': rssi_rmse,
            'bs_rmse': bs_rmse,
        }

    def evaluate_anomaly(self, results_df, y_true_anomaly):
        """
        Evaluate anomaly detection.
        y_true_anomaly: 1 = anomaly, 0 = normal
        """
        nis_scores = results_df['nis'].values
        y_pred_anomaly = results_df['is_anomaly'].values
        acc = accuracy_score(y_true_anomaly, y_pred_anomaly)

        try:
            auc = roc_auc_score(y_true_anomaly, nis_scores)
            fpr, tpr, _ = roc_curve(y_true_anomaly, nis_scores)
        except ValueError:
            auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])

        return {
            'accuracy': acc,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
        }
