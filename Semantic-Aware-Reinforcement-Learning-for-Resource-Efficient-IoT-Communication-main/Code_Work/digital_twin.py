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
    Extended Kalman Filter for IoT channel state estimation.

    State       : x = [RSSI_dBm, num_active_bs]  (dBm domain)
    Observation : z = [RSSI_dBm, num_active_bs]  (direct measurement)

    Nonlinearity via mean-reverting transition for RSSI:
        x[0]_{t+1} = x[0]_t + alpha * tanh((mu - x[0]_t) / sigma_s) + w
    Jacobian F_t[0,0] = 1 - alpha / (sigma_s * cosh^2(...)) changes each
    step — this is the defining property of EKF vs standard KF.

    Physical motivation: RSSI in wireless channels exhibits mean-reversion
    (Ornstein-Uhlenbeck-like dynamics) toward the long-term path-loss mean.
    The tanh nonlinearity captures saturation effects at signal boundaries.
    """

    def __init__(self, state_dim=2, obs_dim=2,
                 process_noise=0.5, obs_noise=1.5,
                 rssi_mean=0.0, rssi_std=10.0, alpha=0.3):
        self.state_dim = state_dim
        self.obs_dim   = obs_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 10.0
        self.H = np.eye(obs_dim, state_dim)   # linear observation
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(obs_dim)   * obs_noise

        # Nonlinear transition parameters (set via set_rssi_stats)
        self.rssi_mean  = rssi_mean    # μ: long-term RSSI mean (dBm)
        self.rssi_std   = max(rssi_std, 1e-3)  # σ_s: scale
        self.alpha      = alpha        # mean-reversion strength

    def set_rssi_stats(self, mean, std):
        """Call before run_sequence to set dataset-specific RSSI statistics."""
        self.rssi_mean = float(mean)
        self.rssi_std  = max(float(std), 1e-3)

    # ------------------------------------------------------------------
    # Nonlinear transition f(x) and its Jacobian F_t
    # ------------------------------------------------------------------

    def _f(self, x):
        """Nonlinear state transition: tanh mean-reversion for RSSI."""
        rssi_next = x[0] + self.alpha * np.tanh(
            (self.rssi_mean - x[0]) / self.rssi_std)
        nbs_next  = x[1]            # nBS follows constant-state model
        return np.array([rssi_next, nbs_next])

    def _F_jacobian(self, x):
        """
        Jacobian of f at x.  F_t[0,0] changes every step — genuine EKF.
        d(f[0])/d(x[0]) = 1 - alpha / (sigma_s * cosh^2((mu-x[0])/sigma_s))
        """
        cosh_val = np.cosh((self.rssi_mean - x[0]) / self.rssi_std)
        df_drssi = 1.0 - self.alpha / (self.rssi_std * cosh_val ** 2)
        return np.array([[df_drssi, 0.0],
                         [0.0,      1.0]])

    # ------------------------------------------------------------------
    # Predict – Update cycle
    # ------------------------------------------------------------------

    def reset(self, x0):
        self.x = np.array(x0, dtype=float)
        self.P = np.eye(self.state_dim) * 10.0

    def predict(self):
        """EKF predict: nonlinear f(x) + linearized covariance update."""
        F      = self._F_jacobian(self.x)        # time-varying Jacobian
        self.x = self._f(self.x)                 # nonlinear prediction
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        EKF update (linear observation H = I).
        Returns: (state_estimate, innovation, NIS).
        """
        z = np.array(z, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        nis = float(y.T @ np.linalg.inv(S) @ y)
        return self.x.copy(), y, nis

    def step(self, z):
        """One full predict–update cycle."""
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

    def reset(self, first_obs, rssi_mean=None, rssi_std=None):
        self.ekf.reset(np.array(first_obs[:self.state_dim]))
        if rssi_mean is not None:
            self.ekf.set_rssi_stats(rssi_mean, rssi_std)

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

        rssi_vals = observations[:, 0]
        rssi_mean = float(np.mean(rssi_vals))
        rssi_std  = float(np.std(rssi_vals)) + 1e-6
        self.reset(observations[0], rssi_mean=rssi_mean, rssi_std=rssi_std)
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
        # Soft score: estimated RSSI (higher = better quality, consistent with label definition)
        est_rssi = results_df['est_rssi'].values
        try:
            auc = roc_auc_score(y_true, est_rssi)
            fpr, tpr, thresholds = roc_curve(y_true, est_rssi)
            # Youden's J: optimal threshold on estimated RSSI
            opt_idx = np.argmax(tpr - fpr)
            opt_thresh = thresholds[opt_idx]
            y_pred = (est_rssi >= opt_thresh).astype(int)
        except ValueError:
            auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])
            y_pred = results_df['quality_label'].values

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True,
                                       zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        rssi_rmse = np.sqrt(np.mean(
            (est_rssi - results_df['obs_rssi'].values) ** 2))
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
