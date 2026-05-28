import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def setup_plot_style():
    """Set up consistent plot style"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_training_progress(episodes, rewards, epsilons, save_path=None):
    """Plot DQN training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot total reward
    ax1.plot(episodes, rewards, label='Total Reward', color='tab:blue', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot epsilon
    ax2.plot(episodes, epsilons, label='Epsilon', color='tab:red', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, auc, title="ROC Curve", save_path=None):
    """Plot ROC curve"""
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_anomaly_map(df, save_path=None):
    """Plot anomaly detection results on map"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Longitude", y="Latitude",
        hue="anomaly", palette={1: "blue", -1: "red"},
        data=df, alpha=0.7
    )
    plt.legend(title="Point Type", labels=["Normal", "Anomaly"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(clf, feature_names, save_path=None):
    """Plot feature importance from Random Forest"""
    importances = clf.feature_importances_
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, save_path=None):
    """Plot combined ROC curves for comparison"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve - Random Forest
    axs[0].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'AUC = {auc_rf:.2f}')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc="lower right")
    
    # ROC Curve - Isolation Forest
    axs[1].plot(fpr_if, tpr_if, color='blue', lw=2, label=f'AUC = {auc_if:.2f}')
    axs[1].plot([0, 1], [0, 1], linestyle="--", color="navy")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend(loc="lower right")
    
    # Add subplot labels
    fig.text(0.23, 0.02, '(a) ROC Curve - Random Forest', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.77, 0.02, '(b) ROC Curve - Isolation Forest', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Digital Twin (EKF) Visualization Functions
# ============================================================

def plot_digital_twin_quality_distribution(qualities, y_true, save_path=None):
    """
    Plot distribution of predicted connection qualities vs ground truth.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    pred_labels = (qualities > 0.5).astype(int)

    ax1.hist(qualities[y_true == 0], bins=30, alpha=0.6, label='Poor (label=0)',
             color='red', edgecolor='darkred')
    ax1.hist(qualities[y_true == 1], bins=30, alpha=0.6, label='Good (label=1)',
             color='green', edgecolor='darkgreen')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Predicted Quality')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Digital Twin Quality Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    correct = pred_labels == y_true
    ax2.hist(qualities[correct], bins=30, alpha=0.6, label='Correct', color='blue')
    ax2.hist(qualities[~correct], bins=30, alpha=0.6, label='Incorrect', color='orange')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Quality')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Correctness vs Quality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_innovation_history(dt, save_path=None):
    """
    Plot EKF innovation (prediction error) history over time.
    Shows how the filter adapts and where anomalies occur.
    """
    if len(dt.chi2_history) == 0:
        print("No innovation history to plot.")
        return

    history = np.array(dt.chi2_history)
    steps = np.arange(len(history))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(steps, history, color='blue', alpha=0.5, linewidth=0.5, label='Innovation (chi2)')

    window = min(100, len(history) // 10) if len(history) > 100 else max(10, len(history) // 2)
    if window > 1:
        smoothed = np.convolve(history, np.ones(window) / window, mode='valid')
        ax1.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                color='red', linewidth=2, label=f'MA({window})')
    ax1.axhline(y=dt.config.innovation_threshold,
               color='orange', linestyle='--', linewidth=2, label='Fixed Threshold')
    ax1.axhline(y=dt._compute_adaptive_threshold(),
               color='green', linestyle='--', linewidth=2, label='Adaptive Threshold')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Innovation (Mahalanobis Distance)')
    ax1.set_title('EKF Innovation History (Digital Twin)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(history, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=np.mean(history), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(history):.2f}')
    ax2.axvline(x=np.percentile(history, 95), color='orange', linestyle='--',
               label=f'P95: {np.percentile(history, 95):.2f}')
    ax2.axvline(x=np.percentile(history, 99), color='red', linestyle='--',
               label=f'P99: {np.percentile(history, 99):.2f}')
    ax2.set_xlabel('Innovation Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Innovation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_state_trajectory(dt, save_path=None):
    """
    Plot the EKF state trajectory over time.
    Shows how connection quality estimate evolves.
    """
    if len(dt.chi2_history) == 0:
        print("No state history to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    quality_history = []
    rssi_trend_history = []
    bs_trend_history = []

    for i in range(len(dt.chi2_history)):
        quality_history.append(np.clip(dt.x[0], 0, 1))
        rssi_trend_history.append(dt.x[1])
        bs_trend_history.append(dt.x[2])

    steps = np.arange(len(quality_history))

    axes[0].plot(steps, quality_history, color='blue', linewidth=1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Quality threshold')
    axes[0].set_ylabel('Connection Quality')
    axes[0].set_title('Digital Twin State Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, rssi_trend_history, color='green', linewidth=1)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_ylabel('RSSI Trend')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, bs_trend_history, color='purple', linewidth=1)
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[2].set_ylabel('BS Trend')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_anomaly_detection_comparison(ekf_scores, ekf_labels,
                                    if_scores=None, if_labels=None,
                                    y_true=None, save_path=None):
    """
    Compare anomaly detection results between EKF-based Digital Twin
    and Isolation Forest side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if if_scores is not None:
        axes[0].scatter(np.arange(len(ekf_scores)), ekf_scores,
                       c=ekf_labels, cmap='coolwarm', alpha=0.5, s=5)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Anomaly Score (Mahalanobis)')
        axes[0].set_title('Digital Twin (EKF) Anomaly Scores')
        axes[0].axhline(y=np.mean(ekf_scores), color='green', linestyle='--',
                       label=f'Mean: {np.mean(ekf_scores):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(np.arange(len(if_scores)), if_scores,
                       c=if_labels, cmap='coolwarm', alpha=0.5, s=5)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Isolation Forest Anomaly Scores')
        axes[1].axhline(y=np.mean(if_scores), color='green', linestyle='--',
                       label=f'Mean: {np.mean(if_scores):.2f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].scatter(np.arange(len(ekf_scores)), ekf_scores,
                       c=ekf_labels, cmap='coolwarm', alpha=0.5, s=5)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Anomaly Score (Mahalanobis)')
        axes[0].set_title('Digital Twin (EKF) Anomaly Scores')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(ekf_scores[ekf_labels == 1], bins=30, alpha=0.6,
                    label='Anomaly', color='red')
        axes[1].hist(ekf_scores[ekf_labels == -1], bins=30, alpha=0.6,
                    label='Normal', color='blue')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Digital Twin Score Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_ekf_vs_rf_roc(fpr_ekf, tpr_ekf, auc_ekf,
                       fpr_rf=None, tpr_rf=None, auc_rf=None,
                       save_path=None):
    """
    Plot ROC curves comparing Digital Twin (EKF) vs Random Forest.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr_ekf, tpr_ekf, color='blue', lw=2,
           label=f'Digital Twin (EKF) AUC = {auc_ekf:.3f}')

    if fpr_rf is not None and tpr_rf is not None:
        ax.plot(fpr_rf, tpr_rf, color='orange', lw=2,
               label=f'Random Forest AUC = {auc_rf:.3f}')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison: Digital Twin (EKF) vs Random Forest')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_digital_twin_summary(dt, qualities, anomaly_scores, y_true, save_path=None):
    """
    Create a comprehensive summary plot for Digital Twin analysis.
    """
    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 3, 1)
    pred_labels = (qualities > 0.5).astype(int)
    cm = confusion_matrix(y_true, pred_labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('DT Confusion Matrix')

    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(qualities[y_true == 0], bins=25, alpha=0.6, label='Poor', color='red')
    ax2.hist(qualities[y_true == 1], bins=25, alpha=0.6, label='Good', color='green')
    ax2.axvline(x=0.5, color='black', linestyle='--')
    ax2.set_xlabel('Quality')
    ax2.set_ylabel('Freq')
    ax2.set_title('Quality Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    innovation = np.array(dt.chi2_history[-500:]) if len(dt.chi2_history) > 500 else np.array(dt.chi2_history)
    ax3.hist(innovation, bins=40, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=np.percentile(innovation, 95), color='red', linestyle='--', label='P95')
    ax3.axvline(x=np.percentile(innovation, 99), color='orange', linestyle='--', label='P99')
    ax3.set_xlabel('Innovation')
    ax3.set_ylabel('Freq')
    ax3.set_title('Innovation Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    n_plot = min(len(qualities), len(anomaly_scores))
    ax4.scatter(qualities[:n_plot], anomaly_scores[:n_plot], c=y_true[:n_plot], cmap='coolwarm', alpha=0.3, s=5)
    ax4.set_xlabel('Quality')
    ax4.set_ylabel('Anomaly Score')
    ax4.set_title('Quality vs Anomaly Score')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(np.array(dt.chi2_history[-200:]), color='blue', alpha=0.7, linewidth=0.5)
    ax5.axhline(y=dt.config.innovation_threshold, color='orange', linestyle='--',
               label=f'Threshold: {dt.config.innovation_threshold:.1f}')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Innovation')
    ax5.set_title('Innovation History (last 200)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    state_labels = ['Quality', 'RSSI Trend', 'BS Trend']
    state_values = [np.clip(dt.x[0], 0, 1), dt.x[1], dt.x[2]]
    colors = ['blue', 'green', 'purple']
    ax6.barh(state_labels, state_values, color=colors, alpha=0.7)
    ax6.set_xlabel('State Value')
    ax6.set_title('Current EKF State')
    ax6.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_method_comparison_3way(metrics_dict, save_path=None):
    """
    Bar chart comparing three approaches: Baseline (RF+IF), VAE, Digital Twin.
    metrics_dict format: { 'method': { 'metric_name': value } }
    """
    if not metrics_dict:
        return

    methods = list(metrics_dict.keys())
    metrics = list(next(iter(metrics_dict.values())).keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, metric in enumerate(metrics):
        values = [metrics_dict[m].get(metric, 0) for m in methods]
        bars = axes[idx].bar(methods, values, color=colors[:len(methods)], alpha=0.8)
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(metric)
        axes[idx].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
