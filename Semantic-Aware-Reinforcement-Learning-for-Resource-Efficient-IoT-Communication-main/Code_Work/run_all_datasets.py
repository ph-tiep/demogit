# -*- coding: utf-8 -*-
"""
Master experiment runner — proposed framework vs. baseline across 3 datasets.

Usage:
    python run_all_datasets.py

Results saved to:
    results/
    ├── antwerp/
    ├── lorawan/
    └── indoor/
        ├── metrics.json
        ├── vae_loss.csv
        ├── madrl_rewards.csv
        ├── dt_results.csv        (Digital Twin per-step output)
        └── plots/
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from preprocessing import (
    load_sigfox_raw, load_lorawan_raw, load_indoor_raw,
    prepare_vae_input_sigfox, prepare_vae_input_lorawan, prepare_vae_input_indoor,
    get_obs_sequence, get_quality_labels, get_anomaly_labels_from_if,
)
from vae_model import train_vae, encode_dataset
from digital_twin import DigitalTwin
from madrl_gat import (
    train_madrl, MADRLEnvironment,
    adj_from_correlation, adj_from_distances, adj_fully_connected,
)
from evaluation import (
    evaluate_vae as vae_eval_fn,
    evaluate_digital_twin_quality, evaluate_digital_twin_anomaly,
    evaluate_madrl, build_comparison_table,
)

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / 'data'
DATA2_DIR = BASE_DIR / 'Data_2'
DATA3_DIR = BASE_DIR / 'Data_3'
RES_DIR   = BASE_DIR / 'results'

# ---------------------------------------------------------------------------
# Hyperparameters (tune per paper ablation)
# ---------------------------------------------------------------------------
CFG = {
    # num_agents  = graph nodes in GAT  (keep <= 5 for CPU speed)
    # madrl_sub   = env trajectory length per episode
    # gat_hidden/out/heads  = smaller GAT = faster per-step
    # madrl_batch = replay batch size
    'antwerp': {
        'vae_latent': 16, 'vae_hidden': 64, 'vae_epochs': 30, 'vae_batch': 256,
        'num_agents': 5,  'num_actions': 5,
        'madrl_eps': 50,  'madrl_lr': 1e-3, 'madrl_batch': 16,
        'gat_hidden': 16, 'gat_out': 8, 'num_heads': 2,
        'dt_proc_noise': 0.5, 'dt_obs_noise': 1.5,
        'madrl_subsample': 500,
    },
    'lorawan': {
        'vae_latent': 8,  'vae_hidden': 32, 'vae_epochs': 30, 'vae_batch': 512,
        'num_agents': 5,  'num_actions': 5,
        'madrl_eps': 50,  'madrl_lr': 1e-3, 'madrl_batch': 16,
        'gat_hidden': 16, 'gat_out': 8, 'num_heads': 2,
        'dt_proc_noise': 0.3, 'dt_obs_noise': 1.0,
        'madrl_subsample': 500,
    },
    'indoor': {
        'vae_latent': 8,  'vae_hidden': 32, 'vae_epochs': 30, 'vae_batch': 64,
        'num_agents': 5,  'num_actions': 5,
        'madrl_eps': 50,  'madrl_lr': 1e-3, 'madrl_batch': 16,
        'gat_hidden': 16, 'gat_out': 8, 'num_heads': 2,
        'dt_proc_noise': 0.2, 'dt_obs_noise': 0.8,
        'madrl_subsample': 500,
    },
}


# ===========================================================================
# Helpers
# ===========================================================================

# ===========================================================================
# Fault Injection — Anomaly Ground Truth
# ===========================================================================

def inject_anomalies(obs_seq, fraction=0.05, seed=42):
    """
    Inject synthetic faults into [mean_rssi, num_active_bs] sequence.
    Three fault types:
      spike : RSSI jumps far above normal range   (e.g. interference)
      drop  : RSSI collapses to near-minimum      (e.g. connection loss)
      stuck : num_active_bs drops to 0 suddenly   (e.g. sensor freeze)

    Returns
    -------
    injected : np.ndarray [N, 2]  — modified observation sequence
    y_fault  : np.ndarray [N]     — 1 = fault injected, 0 = normal
    """
    rng = np.random.RandomState(seed)
    N = len(obs_seq)
    n_faults = max(1, int(N * fraction))
    fault_idx = np.sort(rng.choice(N, n_faults, replace=False))

    injected = obs_seq.copy().astype(np.float64)
    rssi = obs_seq[:, 0]
    rssi_std = float(rssi.std()) + 1e-6
    rssi_mean = float(rssi.mean())

    fault_types = rng.choice(['spike', 'drop', 'stuck'], size=n_faults)
    for i, ft in zip(fault_idx, fault_types):
        if ft == 'spike':
            injected[i, 0] = rssi_mean + 5.0 * rssi_std   # extreme high
        elif ft == 'drop':
            injected[i, 0] = rssi_mean - 5.0 * rssi_std   # extreme low
            injected[i, 1] = 0.0                            # all BSs lost
        else:  # stuck
            injected[i, 0] = rssi_mean + 4.0 * rssi_std
            injected[i, 1] = 0.0

    y_fault = np.zeros(N, dtype=int)
    y_fault[fault_idx] = 1
    print(f"  Fault injection: {n_faults}/{N} samples "
          f"({n_faults/N*100:.1f}%) — "
          f"spike:{(fault_types=='spike').sum()} "
          f"drop:{(fault_types=='drop').sum()} "
          f"stuck:{(fault_types=='stuck').sum()}")
    return injected.astype(np.float32), y_fault


def rolling_zscore_anomaly(series, window=20):
    """
    Rolling Z-score anomaly score for a 1-D time series.
    score[t] = |x[t] - mean(x[t-W:t])| / std(x[t-W:t])
    Returns np.ndarray of anomaly scores (higher = more anomalous).
    """
    scores = np.zeros(len(series), dtype=np.float64)
    for t in range(1, len(series)):
        hist = series[max(0, t - window):t]
        mu = hist.mean()
        sigma = hist.std() + 1e-9
        scores[t] = abs(series[t] - mu) / sigma
    return scores


# ===========================================================================
# Helpers
# ===========================================================================

def save_json(data, path):
    """Recursively convert numpy types before JSON dump."""
    def _conv(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, dict):            return {k: _conv(v) for k, v in obj.items()}
        if isinstance(obj, list):            return [_conv(v) for v in obj]
        return obj
    with open(path, 'w') as f:
        json.dump(_conv(data), f, indent=2)


def baseline_unsupervised(semantic_df, quality_labels):
    """
    Unsupervised baselines (no label used during fitting):
      Quality classification : K-Means (k=2)  +  GMM (k=2)
      Anomaly detection      : LOF  (scores returned for fault-injection eval)
    All methods are fully unsupervised — fair comparison with DT-EKF.
    """
    features = [c for c in ['mean_rssi', 'num_active_bs', 'Latitude', 'Longitude', 'hour']
                if c in semantic_df.columns]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(semantic_df[features].values)
    y_true = quality_labels
    rssi_raw = semantic_df['mean_rssi'].values

    # ── K-Means quality classification ────────────────────────────────────
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    c0_rssi = rssi_raw[km_labels == 0].mean()
    c1_rssi = rssi_raw[km_labels == 1].mean()
    good_cluster = 0 if c0_rssi > c1_rssi else 1
    km_quality = (km_labels == good_cluster).astype(int)

    km_acc = accuracy_score(y_true, km_quality)
    dist_to_good = np.linalg.norm(X - km.cluster_centers_[good_cluster], axis=1)
    km_conf = 1.0 / (1.0 + dist_to_good)
    try:
        km_auc = roc_auc_score(y_true, km_conf)
        km_fpr, km_tpr, _ = roc_curve(y_true, km_conf)
    except ValueError:
        km_auc = 0.5
        km_fpr, km_tpr = np.array([0, 1]), np.array([0, 1])

    # ── GMM quality classification ─────────────────────────────────────────
    gmm = GaussianMixture(n_components=2, covariance_type='full',
                          random_state=42, max_iter=200)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)

    c0_rssi_g = rssi_raw[gmm_labels == 0].mean()
    c1_rssi_g = rssi_raw[gmm_labels == 1].mean()
    good_comp = 0 if c0_rssi_g > c1_rssi_g else 1
    gmm_quality = (gmm_labels == good_comp).astype(int)

    gmm_acc = accuracy_score(y_true, gmm_quality)
    # Posterior probability of belonging to "good" component as confidence
    gmm_proba = gmm.predict_proba(X)[:, good_comp]
    try:
        gmm_auc = roc_auc_score(y_true, gmm_proba)
        gmm_fpr, gmm_tpr, _ = roc_curve(y_true, gmm_proba)
    except ValueError:
        gmm_auc = 0.5
        gmm_fpr, gmm_tpr = np.array([0, 1]), np.array([0, 1])

    # ── LOF anomaly detection ──────────────────────────────────────────────
    # Scores are returned raw; AUC is computed in run_experiment against
    # fault-injection ground truth (y_fault), not LOF's own pseudo-labels.
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
    lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

    print(f"  [Baseline K-Means]  Acc={km_acc:.4f}  AUC={km_auc:.4f}")
    print(f"  [Baseline GMM]      Acc={gmm_acc:.4f}  AUC={gmm_auc:.4f}")
    print(f"  [Baseline LOF]      scores computed (AUC vs fault-injection GT)")

    return {
        'km_acc':  km_acc,   'km_auc':  km_auc,
        'km_fpr':  km_fpr,   'km_tpr':  km_tpr,
        'gmm_acc': gmm_acc,  'gmm_auc': gmm_auc,
        'gmm_fpr': gmm_fpr,  'gmm_tpr': gmm_tpr,
        'lof_scores': lof_scores,   # raw scores for fault-injection eval
    }


def plot_and_save(save_dir, name, fig):
    fig.savefig(save_dir / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(save_dir / f"{name}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_roc_comparison(ax, fpr_base, tpr_base, auc_base,
                        fpr_prop, tpr_prop, auc_prop, title):
    ax.plot(fpr_base, tpr_base, 'b--', lw=1.5, label=f'Baseline AUC={auc_base:.3f}')
    ax.plot(fpr_prop, tpr_prop, 'r-',  lw=2.0, label=f'Proposed AUC={auc_prop:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)


# ===========================================================================
# Per-dataset experiment
# ===========================================================================

def run_experiment(dataset_name, semantic_df, vae_X_norm, vae_input_dim,
                   obs_seq, quality_labels, adj_matrix,
                   num_agents, cfg, res_dir):
    """
    Full pipeline for one dataset:
      1. Baseline  (RF + IF)
      2. VAE       (train + evaluate)
      3. Digital Twin / EKF  (quality + anomaly)
      4. MADRL + GAT         (train + evaluate)
      5. Save all results
    """
    plot_dir = res_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  N={len(semantic_df)}")
    print(f"{'='*60}")

    # ── Adaptive threshold per dataset ────────────────────────────────────
    # If quality_labels are heavily imbalanced (< 5% or > 95% positive),
    # the -110 dBm threshold is not appropriate. Use median RSSI split.
    rssi_arr = semantic_df['mean_rssi'].values
    pos_ratio = quality_labels.mean()
    if pos_ratio < 0.05 or pos_ratio > 0.95:
        dt_rssi_thresh = float(np.median(rssi_arr))
        dt_bs_thresh   = 1
        quality_labels = (rssi_arr > dt_rssi_thresh).astype(int)
        print(f"  Adaptive threshold: {dt_rssi_thresh:.1f} dBm  "
              f"(ratio was {pos_ratio:.3f} -> now {quality_labels.mean():.3f})")
    else:
        dt_rssi_thresh = -110.0
        dt_bs_thresh   = 3 if semantic_df['num_active_bs'].max() > 1 else 1

    # ── Fault injection — ground truth for anomaly detection eval ─────────
    print("\n[Fault Injection] Injecting synthetic faults into obs_seq ...")
    obs_seq_injected, y_fault = inject_anomalies(obs_seq, fraction=0.05, seed=42)

    # ── 1. Baseline (unsupervised) ─────────────────────────────────────────
    print("\n[1/4] Baseline K-Means + GMM + LOF (unsupervised) ...")
    baseline = baseline_unsupervised(semantic_df, quality_labels)

    # Rolling Z-score on the fault-injected RSSI series
    zscore_scores = rolling_zscore_anomaly(obs_seq_injected[:, 0], window=20)

    # Evaluate LOF and Z-score against fault-injection ground truth
    try:
        lof_auc = roc_auc_score(y_fault, baseline['lof_scores'])
        lof_fpr, lof_tpr, _ = roc_curve(y_fault, baseline['lof_scores'])
    except ValueError:
        lof_auc = 0.5
        lof_fpr, lof_tpr = np.array([0, 1]), np.array([0, 1])

    try:
        zs_auc = roc_auc_score(y_fault, zscore_scores)
        zs_fpr, zs_tpr, _ = roc_curve(y_fault, zscore_scores)
    except ValueError:
        zs_auc = 0.5
        zs_fpr, zs_tpr = np.array([0, 1]), np.array([0, 1])

    print(f"  [Baseline LOF]      AUC vs fault-GT={lof_auc:.4f}")
    print(f"  [Baseline Z-Score]  AUC vs fault-GT={zs_auc:.4f}")

    y_anom_true = y_fault  # use fault-injection GT for DT evaluation

    # ── 2. VAE ────────────────────────────────────────────────────────────
    print("\n[2/4] Training VAE ...")
    vae, vae_hist = train_vae(
        vae_X_norm,
        input_dim=vae_input_dim,
        latent_dim=cfg['vae_latent'],
        hidden_dim=cfg['vae_hidden'],
        epochs=cfg['vae_epochs'],
        batch_size=cfg['vae_batch'],
        device=device,
    )
    vae_metrics = vae_eval_fn(vae, vae_X_norm, device)
    print(f"  VAE recon MSE={vae_metrics['mse']:.6f}  KL={vae_metrics['kl_divergence']:.4f}")

    # Encode full dataset
    latent_codes = encode_dataset(vae, vae_X_norm, device)
    print(f"  Latent codes shape: {latent_codes.shape}")

    # ── 3. Digital Twin ────────────────────────────────────────────────────
    print("\n[3/4] Digital Twin (EKF) ...")
    dt = DigitalTwin(
        state_dim=2, obs_dim=2,
        rssi_threshold=dt_rssi_thresh,
        bs_threshold=dt_bs_thresh,
        process_noise=cfg['dt_proc_noise'],
        obs_noise=cfg['dt_obs_noise'],
    )
    dt_results = dt.run_sequence(obs_seq_injected)

    dt_quality_m = evaluate_digital_twin_quality(dt_results, quality_labels)
    dt_anomaly_m = evaluate_digital_twin_anomaly(dt_results, y_anom_true)

    # ── 4. MADRL + GAT ─────────────────────────────────────────────────────
    print("\n[4/4] MADRL + GAT ...")
    # Subsample for very large datasets
    madrl_latent = latent_codes
    madrl_obs    = obs_seq
    sub = cfg.get('madrl_subsample', None)
    if sub and len(latent_codes) > sub:
        idx = np.random.RandomState(42).choice(len(latent_codes), sub, replace=False)
        idx.sort()
        madrl_latent = latent_codes[idx]
        madrl_obs    = obs_seq[idx]

    policy_net, target_net, ep_rewards, ep_losses = train_madrl(
        madrl_latent, madrl_obs, adj_matrix,
        num_agents=num_agents,
        num_actions=cfg['num_actions'],
        num_episodes=cfg['madrl_eps'],
        lr=cfg['madrl_lr'],
        batch_size=cfg.get('madrl_batch', 16),
        gat_hidden=cfg.get('gat_hidden', 16),
        gat_out=cfg.get('gat_out', 8),
        num_heads=cfg.get('num_heads', 2),
        device=device,
    )

    # Evaluation episodes
    eval_env = MADRLEnvironment(madrl_latent, madrl_obs, adj_matrix,
                                num_agents=num_agents,
                                num_actions=cfg['num_actions'])
    eval_rewards = evaluate_madrl(policy_net, eval_env, adj_matrix,
                                  num_episodes=20, num_agents=num_agents,
                                  device=device)

    # ── Save raw outputs ────────────────────────────────────────────────────
    pd.DataFrame(vae_hist).to_csv(res_dir / 'vae_loss.csv', index=False)
    dt_results.to_csv(res_dir / 'dt_results.csv', index=False)
    pd.DataFrame({'episode': range(len(ep_rewards)),
                  'reward': ep_rewards,
                  'loss':   ep_losses}).to_csv(res_dir / 'madrl_training.csv', index=False)
    np.save(res_dir / 'latent_codes.npy', latent_codes)

    # ── Build comparison table ─────────────────────────────────────────────
    baseline_agg = {
        'quality_accuracy': baseline['km_acc'],
        'quality_auc':      baseline['km_auc'],
        'anomaly_auc':      lof_auc,
        'avg_reward':       float(np.mean(ep_rewards[:5])),   # first-5 episodes (untrained)
    }
    proposed_agg = {
        'quality_accuracy': dt_quality_m['accuracy'],
        'quality_auc':      dt_quality_m['auc'],
        'anomaly_auc':      dt_anomaly_m['auc'],
        'avg_reward':       float(np.mean(eval_rewards)),
    }
    comp = build_comparison_table(baseline_agg, proposed_agg, dataset_name)

    all_metrics = {
        'vae':           {k: v for k, v in vae_metrics.items()
                          if k != 'per_sample_mse'},
        'dt_quality':    {k: v for k, v in dt_quality_m.items()
                          if k not in ('fpr', 'tpr', 'report', 'confusion_matrix')},
        'dt_anomaly':    {k: v for k, v in dt_anomaly_m.items()
                          if k not in ('fpr', 'tpr')},
        'madrl_train':   {'avg_reward_last20': float(np.mean(ep_rewards[-20:])),
                          'avg_reward_eval':   float(np.mean(eval_rewards)),
                          'std_reward_eval':   float(np.std(eval_rewards))},
        'baseline_kmeans':  {'accuracy': baseline['km_acc'], 'auc': baseline['km_auc']},
        'baseline_gmm':     {'accuracy': baseline['gmm_acc'], 'auc': baseline['gmm_auc']},
        'baseline_lof':     {'auc': lof_auc},
        'baseline_zscore':  {'auc': zs_auc},
        'fault_injection':  {'n_faults': int(y_fault.sum()), 'n_total': int(len(y_fault))},
        'comparison':    comp,
    }
    save_json(all_metrics, res_dir / 'metrics.json')
    print(f"\n  Results saved -> {res_dir}")

    # ── Plots ───────────────────────────────────────────────────────────────
    # VAE loss curve
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(vae_hist['recon'], label='Reconstruction', color='steelblue')
    ax.plot(vae_hist['kl'],    label='KL divergence',  color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
    ax.set_title(f'VAE Training Loss — {dataset_name}')
    plt.tight_layout()
    plot_and_save(plot_dir, 'vae_loss', fig)

    # MADRL reward curve
    fig, ax = plt.subplots(figsize=(6, 3))
    window = max(1, len(ep_rewards) // 10)
    smoothed = pd.Series(ep_rewards).rolling(window, min_periods=1).mean()
    ax.plot(ep_rewards, alpha=0.3, color='steelblue', label='Raw')
    ax.plot(smoothed,   color='steelblue', lw=2, label='Smoothed')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'MADRL+GAT Training — {dataset_name}')
    ax.legend()
    plt.tight_layout()
    plot_and_save(plot_dir, 'madrl_rewards', fig)

    # ROC curves — quality (K-Means / GMM vs DT-EKF) and anomaly (LOF / Z-Score vs DT-EKF)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(baseline['km_fpr'],  baseline['km_tpr'],  'b--', lw=1.5,
            label=f'K-Means AUC={baseline["km_auc"]:.3f}')
    ax.plot(baseline['gmm_fpr'], baseline['gmm_tpr'], 'g--', lw=1.5,
            label=f'GMM AUC={baseline["gmm_auc"]:.3f}')
    ax.plot(dt_quality_m['fpr'], dt_quality_m['tpr'], 'r-',  lw=2.0,
            label=f'DT-EKF AUC={dt_quality_m["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Quality ROC — {dataset_name}\n(K-Means / GMM vs DT-EKF)')
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    ax.plot(lof_fpr, lof_tpr, 'b--', lw=1.5,
            label=f'LOF AUC={lof_auc:.3f}')
    ax.plot(zs_fpr,  zs_tpr,  'g--', lw=1.5,
            label=f'Z-Score AUC={zs_auc:.3f}')
    ax.plot(dt_anomaly_m['fpr'], dt_anomaly_m['tpr'], 'r-', lw=2.0,
            label=f'DT-EKF AUC={dt_anomaly_m["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Anomaly ROC — {dataset_name}\n(LOF / Z-Score vs DT-EKF, fault-injection GT)')
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plot_and_save(plot_dir, 'roc_comparison', fig)

    # Digital Twin state estimation
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    n_plot = min(500, len(dt_results))
    t = np.arange(n_plot)
    axes[0].plot(t, dt_results['obs_rssi'].values[:n_plot],
                 alpha=0.5, label='Observed', color='gray')
    axes[0].plot(t, dt_results['est_rssi'].values[:n_plot],
                 label='EKF estimate', color='steelblue', lw=1.5)
    axes[0].set_ylabel('mean_rssi (dBm)'); axes[0].legend(fontsize=8)
    axes[1].plot(t, dt_results['obs_num_bs'].values[:n_plot],
                 alpha=0.5, label='Observed', color='gray')
    axes[1].plot(t, dt_results['est_num_bs'].values[:n_plot],
                 label='EKF estimate', color='tomato', lw=1.5)
    axes[1].set_ylabel('num_active_bs'); axes[1].set_xlabel('Step')
    axes[1].legend(fontsize=8)
    plt.suptitle(f'Digital Twin State Estimation — {dataset_name}')
    plt.tight_layout()
    plot_and_save(plot_dir, 'dt_estimation', fig)

    return all_metrics


# ===========================================================================
# Dataset-specific pipeline setup
# ===========================================================================

def setup_antwerp(cfg):
    print("\n>>> Loading Sigfox Antwerp dataset ...")
    rssi_matrix, semantic_df, bs_cols = load_sigfox_raw(DATA_DIR)
    vae_X, _ = prepare_vae_input_sigfox(rssi_matrix)
    obs_seq = get_obs_sequence(semantic_df)
    quality_labels = get_quality_labels(semantic_df)

    # Adjacency: top-K most active BSs, correlation-based (K = num_agents)
    adj, _ = adj_from_correlation(rssi_matrix, top_k=cfg['num_agents'], threshold=0.2)

    print(f"  Rows={len(semantic_df)}  VAE_dim={vae_X.shape[1]}  "
          f"Quality-1 ratio={quality_labels.mean():.3f}")
    return semantic_df, vae_X, vae_X.shape[1], obs_seq, quality_labels, adj


def setup_lorawan(cfg):
    print("\n>>> Loading LoRaWAN Italy dataset ...")
    feat_matrix, semantic_df, node_coords = load_lorawan_raw(DATA2_DIR)
    vae_X, _ = prepare_vae_input_lorawan(feat_matrix)

    # Use semantic_df aligned with feat_matrix
    n = min(len(semantic_df), len(feat_matrix))
    semantic_df = semantic_df.iloc[:n].copy()
    vae_X = vae_X[:n]

    obs_seq = get_obs_sequence(semantic_df)
    quality_labels = get_quality_labels(semantic_df)

    # Adjacency: 8 nodes distance-based
    coords_list = []
    node_ids = sorted(node_coords.keys())[:cfg['num_agents']]
    for nid in node_ids:
        coords_list.append(node_coords[nid])
    while len(coords_list) < cfg['num_agents']:
        coords_list.append((0.0, 0.0))

    # Convert lat/lon degrees to metres (approx)
    coords_m = np.array(coords_list)
    coords_m[:, 0] *= 111_320     # lat → metres
    coords_m[:, 1] *= 111_320 * np.cos(np.radians(coords_m[0, 0]))
    adj = adj_from_distances(coords_m, sigma=200.0, threshold=0.1)

    print(f"  Rows={len(semantic_df)}  VAE_dim={vae_X.shape[1]}  "
          f"Quality-1 ratio={quality_labels.mean():.3f}")
    return semantic_df, vae_X, vae_X.shape[1], obs_seq, quality_labels, adj


def setup_indoor(cfg):
    print("\n>>> Loading Indoor WiFi dataset ...")
    rssi_matrix, semantic_df, pos_coords = load_indoor_raw(DATA3_DIR)
    vae_X, _ = prepare_vae_input_indoor(rssi_matrix)

    n = min(len(semantic_df), len(vae_X))
    semantic_df = semantic_df.iloc[:n].copy()
    vae_X = vae_X[:n]

    obs_seq = get_obs_sequence(semantic_df)
    quality_labels = get_quality_labels(semantic_df)

    # Adjacency: fully connected (small indoor space, num_agents APs)
    adj = adj_fully_connected(cfg['num_agents'])

    print(f"  Rows={len(semantic_df)}  VAE_dim={vae_X.shape[1]}  "
          f"Quality-1 ratio={quality_labels.mean():.3f}")
    return semantic_df, vae_X, vae_X.shape[1], obs_seq, quality_labels, adj


# ===========================================================================
# Summary table
# ===========================================================================

def generate_summary(all_results):
    """Print and save a LaTeX-ready comparison table."""
    rows = []
    for ds, m in all_results.items():
        rows.append({
            'Dataset':            ds,
            'KMeans Acc':         f"{m['baseline_kmeans']['accuracy']:.4f}",
            'GMM Acc':            f"{m['baseline_gmm']['accuracy']:.4f}",
            'DT Acc':             f"{m['dt_quality']['accuracy']:.4f}",
            'KMeans AUC':         f"{m['baseline_kmeans']['auc']:.4f}",
            'GMM AUC':            f"{m['baseline_gmm']['auc']:.4f}",
            'DT AUC':             f"{m['dt_quality']['auc']:.4f}",
            'LOF AUC':            f"{m['baseline_lof']['auc']:.4f}",
            'ZScore AUC':         f"{m['baseline_zscore']['auc']:.4f}",
            'DT Anomaly AUC':     f"{m['dt_anomaly']['auc']:.4f}",
            'MADRL Avg Reward':   f"{m['madrl_train']['avg_reward_eval']:.2f}",
            'VAE MSE':            f"{m['vae']['mse']:.6f}",
        })
    df = pd.DataFrame(rows).set_index('Dataset')
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string())

    df.to_csv(RES_DIR / 'summary_table.csv')
    # Manual LaTeX export (no jinja2 dependency)
    cols = list(df.columns)
    header = ' & '.join(['Dataset'] + cols) + ' \\\\\n\\hline\n'
    rows_tex = ''
    for idx, row in df.iterrows():
        rows_tex += idx + ' & ' + ' & '.join(str(row[c]) for c in cols) + ' \\\\\n'
    latex = (
        '\\begin{table}[h]\n\\centering\n'
        '\\caption{Baseline vs. Proposed Framework Comparison}\n'
        '\\label{tab:results}\n'
        '\\begin{tabular}{l' + 'c' * len(cols) + '}\n'
        '\\hline\n' + header + rows_tex +
        '\\hline\n\\end{tabular}\n\\end{table}\n'
    )
    (RES_DIR / 'summary_table.tex').write_text(latex)
    print(f"\nSummary saved -> {RES_DIR / 'summary_table.csv'}")
    print(f"LaTeX table  -> {RES_DIR / 'summary_table.tex'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("="*70)
    print(" Proposed Framework: VAE + Digital Twin (EKF) + MADRL+GAT")
    print(" Running on 3 IoT datasets")
    print("="*70)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    all_results = {}

    # ── Antwerp (Sigfox) ──────────────────────────────────────────────────
    try:
        cfg = CFG['antwerp']
        res = RES_DIR / 'antwerp'
        res.mkdir(exist_ok=True)
        semantic_df, vae_X, vae_dim, obs_seq, q_labels, adj = setup_antwerp(cfg)
        metrics = run_experiment('antwerp', semantic_df, vae_X, vae_dim,
                                 obs_seq, q_labels, adj,
                                 num_agents=cfg['num_agents'], cfg=cfg, res_dir=res)
        all_results['antwerp'] = metrics
    except Exception as e:
        print(f"\n[ERROR] Antwerp: {e}")
        import traceback; traceback.print_exc()

    # ── LoRaWAN (Italy) ──────────────────────────────────────────────────
    try:
        cfg = CFG['lorawan']
        res = RES_DIR / 'lorawan'
        res.mkdir(exist_ok=True)
        semantic_df, vae_X, vae_dim, obs_seq, q_labels, adj = setup_lorawan(cfg)
        metrics = run_experiment('lorawan', semantic_df, vae_X, vae_dim,
                                 obs_seq, q_labels, adj,
                                 num_agents=cfg['num_agents'], cfg=cfg, res_dir=res)
        all_results['lorawan'] = metrics
    except Exception as e:
        print(f"\n[ERROR] LoRaWAN: {e}")
        import traceback; traceback.print_exc()

    # ── Indoor (WiFi) ────────────────────────────────────────────────────
    try:
        cfg = CFG['indoor']
        res = RES_DIR / 'indoor'
        res.mkdir(exist_ok=True)
        semantic_df, vae_X, vae_dim, obs_seq, q_labels, adj = setup_indoor(cfg)
        metrics = run_experiment('indoor', semantic_df, vae_X, vae_dim,
                                 obs_seq, q_labels, adj,
                                 num_agents=cfg['num_agents'], cfg=cfg, res_dir=res)
        all_results['indoor'] = metrics
    except Exception as e:
        print(f"\n[ERROR] Indoor: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────
    if all_results:
        generate_summary(all_results)

    print("\n" + "="*70)
    print(" All experiments complete.")
    print("="*70)


if __name__ == '__main__':
    main()
