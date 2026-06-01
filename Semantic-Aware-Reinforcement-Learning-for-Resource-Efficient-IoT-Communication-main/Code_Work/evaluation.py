# -*- coding: utf-8 -*-
"""
Evaluation module — metrics for both baseline (DQN / RF / IF)
and the proposed framework (VAE / Digital Twin EKF / MADRL+GAT).
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)


# ==========================================================================
# ── ORIGINAL BASELINE EVALUATION (kept for main.py compatibility) ──────────
# ==========================================================================

def evaluate_dqn_agent(env, policy_net, episodes=50, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nEvaluating trained DQN agent...")
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"Evaluation Episode {ep+1}: Reward = {total_reward:.2f}")
    print(f"\nAverage Evaluation Reward: {np.mean(rewards):.2f}")
    return rewards


def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = (clf.predict_proba(X_test)[:, 1]
              if hasattr(clf, "predict_proba") else None)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_test, y_pred, zero_division=0)}")
    return y_pred, y_prob, accuracy


def evaluate_anomaly_detector(iso_forest, X_scaled, anomaly_labels):
    y_true = (anomaly_labels == -1).astype(int)
    anomaly_scores = -iso_forest.decision_function(X_scaled)
    auc = roc_auc_score(y_true, anomaly_scores)
    print(f"Anomaly distribution: {np.unique(anomaly_labels, return_counts=True)}")
    print(f"AUC Score: {auc:.4f}")
    return y_true, anomaly_scores, auc


def calculate_roc_metrics(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, auc


# ==========================================================================
# ── NEW: VAE EVALUATION ───────────────────────────────────────────────────
# ==========================================================================

def evaluate_vae(vae, X_np, device=None):
    """
    Evaluate VAE reconstruction quality.
    Returns dict with MSE, MAE, and per-sample errors.
    """
    import torch
    import torch.nn.functional as F
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        x_recon, mu, log_var = vae(X_t)
        per_sample_mse = F.mse_loss(x_recon, X_t, reduction='none').mean(1)
        mse = per_sample_mse.mean().item()
        mae = (x_recon - X_t).abs().mean().item()
        kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(1).mean().item()

    return {
        'mse': mse,
        'mae': mae,
        'kl_divergence': kl,
        'per_sample_mse': per_sample_mse.cpu().numpy(),
    }


# ==========================================================================
# ── NEW: DIGITAL TWIN EVALUATION ─────────────────────────────────────────
# ==========================================================================

def evaluate_digital_twin_quality(results_df, y_true_quality):
    """
    Evaluate Digital Twin quality classification vs. ground truth.
    Returns dict with accuracy, AUC, confusion matrix, RMSE.
    """
    # Soft score: raw est_rssi (monotonic, preserves ranking correctly)
    est_rssi = results_df['est_rssi'].values
    try:
        auc = roc_auc_score(y_true_quality, est_rssi)
        fpr, tpr, thresholds = roc_curve(y_true_quality, est_rssi)
        # Youden's J: find optimal threshold on est_rssi
        opt_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[opt_idx]
        y_pred = (est_rssi >= opt_thresh).astype(int)
    except ValueError:
        auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])
        y_pred = results_df['quality_label'].values

    acc = accuracy_score(y_true_quality, y_pred)
    cm = confusion_matrix(y_true_quality, y_pred)
    report = classification_report(y_true_quality, y_pred,
                                   output_dict=True, zero_division=0)

    rssi_rmse = float(np.sqrt(np.mean(
        (est_rssi - results_df['obs_rssi'].values) ** 2)))
    bs_rmse = float(np.sqrt(np.mean(
        (results_df['est_num_bs'].values - results_df['obs_num_bs'].values) ** 2)))

    print(f"[DT Quality] Acc={acc:.4f}  AUC={auc:.4f}  "
          f"RSSI-RMSE={rssi_rmse:.4f}  BS-RMSE={bs_rmse:.4f}")
    return {
        'accuracy': acc, 'auc': auc,
        'fpr': fpr, 'tpr': tpr,
        'confusion_matrix': cm, 'report': report,
        'rssi_rmse': rssi_rmse, 'bs_rmse': bs_rmse,
    }


def evaluate_digital_twin_anomaly(results_df, y_true_anomaly):
    """Evaluate Digital Twin anomaly detection via NIS threshold."""
    nis = results_df['nis'].values
    y_pred = results_df['is_anomaly'].values
    acc = accuracy_score(y_true_anomaly, y_pred)

    try:
        auc = roc_auc_score(y_true_anomaly, nis)
        fpr, tpr, _ = roc_curve(y_true_anomaly, nis)
    except ValueError:
        auc, fpr, tpr = 0.5, np.array([0, 1]), np.array([0, 1])

    print(f"[DT Anomaly] Acc={acc:.4f}  AUC={auc:.4f}")
    return {'accuracy': acc, 'auc': auc, 'fpr': fpr, 'tpr': tpr}


# ==========================================================================
# ── NEW: MADRL EVALUATION ─────────────────────────────────────────────────
# ==========================================================================

def evaluate_madrl(policy_net, env, adj_matrix, num_episodes=20,
                   num_agents=10, device=None):
    """
    Run the trained MADRL+GAT policy for evaluation episodes.
    Returns list of total rewards per episode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj_t = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    policy_net.eval()
    rewards = []

    for ep in range(num_episodes):
        states = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(states, dtype=torch.float32).to(device)
                q = policy_net(s_t, adj_t)
                actions = q.argmax(dim=1).cpu().numpy().tolist()
            states, reward, done = env.step(actions)
            total_reward += reward
        rewards.append(total_reward)

    avg = float(np.mean(rewards))
    std = float(np.std(rewards))
    print(f"[MADRL Eval] Avg reward = {avg:.2f} ± {std:.2f}  "
          f"over {num_episodes} episodes")
    return rewards


# ==========================================================================
# ── COMPARISON TABLE ──────────────────────────────────────────────────────
# ==========================================================================

def build_comparison_table(baseline_metrics, proposed_metrics, dataset_name):
    """
    Assemble a side-by-side metric comparison dict suitable for saving to JSON.
    baseline_metrics / proposed_metrics: dicts with keys
        quality_accuracy, quality_auc, anomaly_auc, avg_reward
    """
    rows = {}
    for key in ['quality_accuracy', 'quality_auc', 'anomaly_auc', 'avg_reward']:
        base_val = baseline_metrics.get(key, float('nan'))
        prop_val = proposed_metrics.get(key, float('nan'))
        try:
            delta = prop_val - base_val
            pct = (delta / abs(base_val) * 100) if base_val != 0 else float('nan')
        except TypeError:
            delta, pct = float('nan'), float('nan')
        rows[key] = {
            'baseline': base_val,
            'proposed': prop_val,
            'delta': delta,
            'pct_change': pct,
        }
    return {dataset_name: rows}
