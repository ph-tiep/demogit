# -*- coding: utf-8 -*-
"""Main script with VAE compression pipeline"""

import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

from preprocessing import preprocess_sigfox_data, prepare_classification_data, prepare_anomaly_data
from training import train_dqn, train_random_forest, train_isolation_forest
from evaluation import evaluate_dqn_agent, evaluate_classifier, evaluate_anomaly_detector, calculate_roc_metrics
from visualization import (plot_training_progress, plot_confusion_matrix, plot_roc_curve, 
                         plot_anomaly_map, plot_feature_importance, plot_combined_roc_curves)
from utils import load_data, normalize_features, split_data, read_training_log
from models import SimpleEnv
from vae_model import train_vae, compress_data_with_vae

def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results" / "vae"

    print("=== VAE-Enhanced Semantic-Aware RL for Resource-Efficient IoT Communication ===\n")
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    data = preprocess_sigfox_data(
        sigfox_path=base_dir / "data" / "sigfox_dataset_antwerp.csv",
        bs_mapping_path=base_dir / "data" / "sigfox_bs_mapping.csv",
        output_path=base_dir / "data" / "semantic_features_antwerp.csv"
    )
    
    # Step 1.5: VAE Compression
    print("\nStep 1.5: VAE Compression of Semantic Features...")
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    original_features = data[features].values
    
    # Train VAE to compress 5D features to 3D latent space
    print(f"Original feature dimension: {len(features)}D")
    print("Training VAE for compression...")
    vae, vae_scaler, losses, recon_losses, kl_losses = train_vae(
        data=original_features,
        input_dim=len(features),
        latent_dim=3,  # Compress from 5D to 3D
        epochs=50,
        batch_size=64,
        lr=1e-3
    )
    
    # Compress features using VAE
    compressed_features = compress_data_with_vae(original_features, vae, vae_scaler)
    print(f"Compressed feature dimension: {compressed_features.shape[1]}D")
    print(f"Compression ratio: {len(features)/compressed_features.shape[1]:.2f}x")
    
    # Step 2: DQN Training with compressed features
    print("\nStep 2: DQN Training with VAE-compressed features...")
    
    # Normalize compressed features for DQN
    from sklearn.preprocessing import StandardScaler
    dqn_scaler = StandardScaler()
    states = dqn_scaler.fit_transform(compressed_features)
    
    policy_net, target_net, losses, total_rewards = train_dqn(states, num_episodes=2)
    
    # Evaluate DQN
    env = SimpleEnv(states, lambda state, action: state[0] * 10 + state[1] * 5 - action)
    dqn_rewards = evaluate_dqn_agent(env, policy_net, episodes=50)
    
    # Step 3: Classification with Random Forest (using original features)
    print("\nStep 3: Random Forest Classification...")
    X_scaled, y, feature_names, scaler = prepare_classification_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.5)
    
    rf_clf = train_random_forest(X_train, y_train)
    y_pred, y_prob, accuracy = evaluate_classifier(rf_clf, X_test, y_test)
    
    # Step 4: Anomaly Detection with Isolation Forest
    print("\nStep 4: Anomaly Detection...")
    X_scaled_anomaly, feature_names_anomaly, scaler_anomaly = prepare_anomaly_data(data)
    iso_forest, anomaly_labels = train_isolation_forest(X_scaled_anomaly)
    
    data_with_anomalies = data.copy()
    data_with_anomalies["anomaly"] = anomaly_labels
    y_true_anomaly, anomaly_scores, auc_anomaly = evaluate_anomaly_detector(
        iso_forest, X_scaled_anomaly, anomaly_labels
    )
    
    # Step 5: Visualization
    print("\nStep 5: Generating visualizations...")
    
    # Plot training progress (if log file exists)
    try:
        episodes, rewards, epsilons = read_training_log(base_dir / 'k.txt')
        plot_training_progress(episodes, rewards, epsilons, output_dir / 'training_progress.pdf')
    except FileNotFoundError:
        print("Training log file not found, skipping training progress plot")
    
    # Classification plots
    plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix.pdf')
    plot_feature_importance(rf_clf, feature_names, output_dir / 'feature_importance.pdf')
    
    # Anomaly detection plots
    plot_anomaly_map(data_with_anomalies, output_dir / 'anomaly_map.pdf')
    
    # ROC curves
    fpr_rf, tpr_rf, auc_rf = calculate_roc_metrics(y_test, y_prob)
    plot_roc_curve(fpr_rf, tpr_rf, auc_rf, save_path=output_dir / 'RFroc_curve.pdf')
    
    fpr_if, tpr_if, auc_if = calculate_roc_metrics(y_true_anomaly, anomaly_scores)
    plot_roc_curve(fpr_if, tpr_if, auc_if, save_path=output_dir / 'IFroc_curve.pdf')
    
    # Combined ROC curves
    plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, 
                           output_dir / 'combined_ROC_curves_with_labels.pdf')
    
    # VAE-specific visualization
    print("\nStep 6: VAE-specific analysis...")
    import matplotlib.pyplot as plt
    
    # Plot VAE training loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('VAE Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(recon_losses, label='Reconstruction Loss')
    ax2.plot(kl_losses, label='KL Divergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('VAE Loss Components')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vae_training_loss.pdf', bbox_inches='tight')
    plt.close()
    
    # Plot latent space visualization (first 2 dimensions)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(compressed_features[:, 0], compressed_features[:, 1], 
                        c=data['mean_rssi'], cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('VAE Latent Space (colored by RSSI)')
    plt.colorbar(scatter, ax=ax, label='Mean RSSI')
    plt.tight_layout()
    plt.savefig(output_dir / 'vae_latent_space.pdf', bbox_inches='tight')
    plt.close()
    
    print("\n=== VAE-Enhanced Pipeline completed successfully! ===")
    print(f"\nResults saved to: {output_dir}")
    print(f"Feature compression: {len(features)}D → {compressed_features.shape[1]}D")
    print(f"Compression ratio: {len(features)/compressed_features.shape[1]:.2f}x")

if __name__ == "__main__":
    main()
