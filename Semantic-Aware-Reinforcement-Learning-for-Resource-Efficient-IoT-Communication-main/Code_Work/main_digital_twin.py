# -*- coding: utf-8 -*-
"""
Main script for Digital Twin (EKF) pipeline.
Replaces Random Forest and Isolation Forest with EKF-based state estimation
for both connection quality classification and anomaly detection.
"""

import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

from preprocessing import preprocess_sigfox_data, prepare_classification_data, prepare_anomaly_data
from training import train_dqn, train_random_forest, train_isolation_forest, train_digital_twin_on_data, predict_with_digital_twin
from digital_twin import DigitalTwinEKF
from evaluation import (
    evaluate_dqn_agent, evaluate_classifier, evaluate_anomaly_detector,
    calculate_roc_metrics,
    evaluate_digital_twin_classification, evaluate_digital_twin_anomaly,
    compute_innovation_statistics
)
from sklearn.metrics import roc_auc_score
from visualization import (
    plot_training_progress, plot_confusion_matrix, plot_roc_curve,
    plot_anomaly_map, plot_feature_importance, plot_combined_roc_curves,
    plot_digital_twin_quality_distribution, plot_innovation_history,
    plot_state_trajectory, plot_anomaly_detection_comparison,
    plot_ekf_vs_rf_roc, plot_digital_twin_summary, plot_method_comparison_3way
)
from utils import load_data, normalize_features, split_data, read_training_log
from models import SimpleEnv


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results" / "digital_twin"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Digital Twin (EKF) Pipeline for Resource-Efficient IoT Communication")
    print("=" * 70)
    print("\nReplacing: Random Forest + Isolation Forest -> EKF-based Digital Twin")

    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    data = preprocess_sigfox_data(
        sigfox_path=base_dir / "data" / "sigfox_dataset_antwerp.csv",
        bs_mapping_path=base_dir / "data" / "sigfox_bs_mapping.csv",
        output_path=base_dir / "data" / "semantic_features_antwerp.csv"
    )

    # Prepare classification data (for ground truth labels)
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    X_scaled, y, feature_names, scaler = prepare_classification_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.5, random_state=42)

    # Prepare anomaly data
    X_scaled_anomaly, feature_names_anomaly, scaler_anomaly = prepare_anomaly_data(data)

    # Step 2: DQN Training (same as baseline)
    print("\nStep 2: DQN Training...")
    states, _ = normalize_features(data, features)
    policy_net, target_net, losses, total_rewards = train_dqn(states, num_episodes=2)

    env = SimpleEnv(states, lambda state, action: state[0] * 10 + state[1] * 5 - action)
    dqn_rewards = evaluate_dqn_agent(env, policy_net, episodes=50)

    # ============================================================
    # Step 3: Digital Twin (EKF) - Classification
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: Digital Twin (EKF) for Connection Quality Classification")
    print("=" * 70)

    dt_classifier, ekf_config = train_digital_twin_on_data(X_train, feature_names)

    # Use first half for burn-in, process test set for evaluation
    burn_in_size = min(100, len(X_train) // 10)
    print(f"\nProcessing test set ({len(X_test)} samples) for classification...")

    # Reset and run on test set
    dt_classifier.reset()
    dt_classifier.initialize(X_train[0])

    qualities, anomaly_scores, is_anomalies = predict_with_digital_twin(
        dt_classifier, X_test, feature_names
    )

    y_pred_dt, dt_accuracy = evaluate_digital_twin_classification(qualities, y_test)
    dt_confusion = np.sum(y_pred_dt == y_test) / len(y_test)
    print(f"Digital Twin Classification Accuracy: {dt_accuracy:.4f}")

    # ROC for Digital Twin classification
    try:
        fpr_dt, tpr_dt, auc_dt = calculate_roc_metrics(y_test, qualities)
    except ValueError:
        fpr_dt, tpr_dt, auc_dt = np.array([0, 1]), np.array([0, 1]), 0.5

    # ============================================================
    # Step 4: Digital Twin (EKF) - Anomaly Detection
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 4: Digital Twin (EKF) for Anomaly Detection")
    print("=" * 70)

    # Use full dataset for anomaly detection
    dt_anomaly = DigitalTwinEKF(ekf_config)
    dt_anomaly.initialize(X_scaled_anomaly[0])

    qualities_full, anomaly_scores_full, is_anomalies_full = predict_with_digital_twin(
        dt_anomaly, X_scaled_anomaly, feature_names
    )

    # Create ground truth for anomaly: label=0 is "good", label=1 is "poor"
    # Poor connections are "anomalies" in this context
    y_true_anomaly = (y == 0).astype(int)

    y_true_anom, anomaly_scores_out, auc_anomaly_dt = evaluate_digital_twin_anomaly(
        anomaly_scores_full, is_anomalies_full, y_true_anomaly
    )

    # ROC for Digital Twin anomaly detection
    try:
        fpr_dt_anom, tpr_dt_anom, auc_dt_anom = calculate_roc_metrics(
            y_true_anomaly, anomaly_scores_full
        )
    except ValueError:
        fpr_dt_anom, tpr_dt_anom, auc_dt_anom = np.array([0, 1]), np.array([0, 1]), 0.5

    # ============================================================
    # Step 5: Comparison with Baseline (RF + IF)
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: Comparison with Baseline Methods")
    print("=" * 70)

    # Train baseline models
    print("\nTraining Baseline Random Forest...")
    rf_clf = train_random_forest(X_train, y_train)
    y_pred_rf, y_prob_rf, rf_accuracy = evaluate_classifier(rf_clf, X_test, y_test)

    print("\nTraining Baseline Isolation Forest...")
    iso_forest, anomaly_labels_if = train_isolation_forest(X_scaled_anomaly)
    y_true_if, anomaly_scores_if, auc_if = evaluate_anomaly_detector(
        iso_forest, X_scaled_anomaly, anomaly_labels_if
    )

    # ============================================================
    # Step 6: Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 6: Generating Visualizations")
    print("=" * 70)

    # Digital Twin-specific plots
    print("Generating Digital Twin plots...")

    plot_digital_twin_quality_distribution(
        qualities, y_test,
        save_path=output_dir / 'dt_quality_distribution.pdf'
    )

    plot_innovation_history(
        dt_classifier,
        save_path=output_dir / 'dt_innovation_history.pdf'
    )

    plot_state_trajectory(
        dt_classifier,
        save_path=output_dir / 'dt_state_trajectory.pdf'
    )

    plot_digital_twin_summary(
        dt_classifier, qualities, anomaly_scores, y_test,
        save_path=output_dir / 'dt_summary.pdf'
    )

    # ROC comparison
    plot_ekf_vs_rf_roc(
        fpr_dt, tpr_dt, auc_dt,
        fpr_rf=None, tpr_rf=None, auc_rf=None,
        save_path=output_dir / 'dt_roc_classification.pdf'
    )

    # Anomaly detection comparison
    plot_anomaly_detection_comparison(
        anomaly_scores_full[:len(X_scaled_anomaly)], is_anomalies_full.astype(int)[:len(X_scaled_anomaly)],
        if_scores=-iso_forest.decision_function(X_scaled_anomaly),
        if_labels=anomaly_labels_if,
        y_true=y_true_anomaly,
        save_path=output_dir / 'dt_vs_if_anomaly_comparison.pdf'
    )

    # Baseline plots (for comparison)
    plot_confusion_matrix(y_test, y_pred_rf, output_dir / 'baseline_confusion_matrix.pdf')
    plot_feature_importance(rf_clf, feature_names, output_dir / 'baseline_feature_importance.pdf')

    # Training progress
    try:
        episodes, rewards, epsilons = read_training_log(base_dir / 'k.txt')
        plot_training_progress(episodes, rewards, epsilons, output_dir / 'training_progress.pdf')
    except FileNotFoundError:
        print("Training log not found, skipping training progress plot")

    # ============================================================
    # Step 7: Innovation Statistics
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 7: EKF Innovation Analysis")
    print("=" * 70)

    stats = compute_innovation_statistics(dt_classifier)
    print(f"\nEKF Innovation Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    metrics = {
        'Digital Twin (EKF)': {
            'Classification Accuracy': dt_accuracy,
            'Classification AUC': auc_dt,
            'Anomaly AUC': auc_dt_anom,
            'Mean Innovation': stats.get('mean_innovation', 0),
        },
        'Random Forest (Baseline)': {
            'Classification Accuracy': rf_accuracy,
            'Classification AUC': roc_auc_score(y_test, y_prob_rf) if y_prob_rf is not None else 0,
            'Anomaly AUC': auc_if,
            'Mean Innovation': None,
        }
    }

    print("\nClassification Performance:")
    print(f"  Digital Twin (EKF):  Accuracy={dt_accuracy:.4f}, AUC={auc_dt:.4f}")
    print(f"  Random Forest:       Accuracy={rf_accuracy:.4f}, "
          f"AUC={roc_auc_score(y_test, y_prob_rf):.4f}" if y_prob_rf is not None else "N/A")

    print("\nAnomaly Detection Performance:")
    print(f"  Digital Twin (EKF):  AUC={auc_dt_anom:.4f}")
    print(f"  Isolation Forest:     AUC={auc_if:.4f}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    # Save comparison metrics
    import json
    metrics_file = output_dir / 'metrics_comparison.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
