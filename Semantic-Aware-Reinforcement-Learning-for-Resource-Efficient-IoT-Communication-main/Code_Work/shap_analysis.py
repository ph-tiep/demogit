# -*- coding: utf-8 -*-
"""SHAP (SHapley Additive exPlanations) Analysis for IoT Communication Models"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    def __init__(self, data_path="semantic_features_antwerp.csv"):
        """Initialize SHAP Analyzer"""
        self.base_dir = Path(__file__).resolve().parent
        self.data_path = Path(data_path)
        if not self.data_path.is_absolute():
            self.data_path = self.base_dir / self.data_path
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
        self.explainer = None
        self.shap_values = None

    def _resolve_output_path(self, save_path):
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.base_dir / save_path
        return save_path

    def _save_and_close_current_figure(self, save_path):
        save_path = self._resolve_output_path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _get_class1_shap_values(self):
        if isinstance(self.shap_values, list):
            return self.shap_values[1]
        if isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
            return self.shap_values[:, :, 1]
        return self.shap_values

    def _get_class1_expected_value(self):
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            return expected_value[1]
        return expected_value
        
    def load_and_prepare_data(self):
        """Load and prepare data for SHAP analysis"""
        print("Loading and preparing data for SHAP analysis...")
        self.data = pd.read_csv(self.data_path)
        
        # Create classification label
        self.data['label'] = ((self.data['mean_rssi'] > -110) & (self.data['num_active_bs'] >= 3)).astype(int)
        print(f"Label distribution:\n{self.data['label'].value_counts()}")
        
        # Prepare features
        X = self.data[self.feature_names]
        y = self.data['label']
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Convert back to DataFrame for better visualization
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def train_model(self):
        """Train Random Forest model for SHAP analysis"""
        print("\nTraining Random Forest model for SHAP analysis...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
    def compute_shap_values(self):
        """Compute SHAP values"""
        print("\nComputing SHAP values...")
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values for test set
        self.shap_values = self.explainer.shap_values(self.X_test_df)
        
        print("SHAP values computed successfully!")
        
    def plot_shap_summary(self, save_path="shap_summary_plot.pdf"):
        """Create SHAP summary plot"""
        print("\nCreating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        
        shap_values_class1 = self._get_class1_shap_values()
            
        shap.summary_plot(shap_values_class1, self.X_test_df, show=False)
        plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def plot_shap_summary_bar(self, save_path="shap_summary_bar.pdf"):
        """Create SHAP summary bar plot"""
        print("\nCreating SHAP summary bar plot...")
        
        plt.figure(figsize=(10, 6))
        
        shap_values_class1 = self._get_class1_shap_values()
            
        shap.summary_plot(shap_values_class1, self.X_test_df, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Mean Absolute Impact)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def plot_shap_dependence(self, feature_index=0, save_path="shap_dependence_plot.pdf"):
        """Create SHAP dependence plot for a specific feature"""
        print(f"\nCreating SHAP dependence plot for {self.feature_names[feature_index]}...")
        
        plt.figure(figsize=(10, 6))
        
        shap_values_class1 = self._get_class1_shap_values()
            
        shap.dependence_plot(
            feature_index,
            shap_values_class1,
            self.X_test_df,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {self.feature_names[feature_index]}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def plot_force_plot(self, instance_idx=0, save_path="shap_force_plot.pdf"):
        """Create SHAP force plot for a specific instance"""
        print(f"\nCreating SHAP force plot for instance {instance_idx}...")
        
        shap_values_class1 = self._get_class1_shap_values()
            
        plt.figure(figsize=(12, 4))
        
        shap.force_plot(
            self._get_class1_expected_value(),
            shap_values_class1[instance_idx, :],
            self.X_test_df.iloc[instance_idx, :],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f"SHAP Force Plot for Instance {instance_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
        # Print instance details
        print(f"\nInstance {instance_idx} details:")
        print(f"Actual class: {self.y_test.iloc[instance_idx]}")
        print(f"Predicted probability for class 1: {self.model.predict_proba([self.X_test[instance_idx]])[0][1]:.3f}")
        
    def plot_waterfall_plot(self, instance_idx=0, save_path="shap_waterfall_plot.pdf"):
        """Create SHAP waterfall plot for a specific instance"""
        print(f"\nCreating SHAP waterfall plot for instance {instance_idx}...")
        
        shap_values_class1 = self._get_class1_shap_values()
            
        plt.figure(figsize=(10, 6))
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_class1[instance_idx],
                base_values=self._get_class1_expected_value(),
                data=self.X_test_df.iloc[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        plt.title(f"SHAP Waterfall Plot for Instance {instance_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def plot_multiple_dependence_plots(self, save_path="shap_multiple_dependence.pdf"):
        """Create multiple SHAP dependence plots in a grid"""
        print("\nCreating multiple SHAP dependence plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        shap_values_class1 = self._get_class1_shap_values()
            
        for i, feature in enumerate(self.feature_names):
            if i < len(axes):
                shap.dependence_plot(
                    feature,
                    shap_values_class1,
                    self.X_test_df,
                    ax=axes[i],
                    show=False
                )
                axes[i].set_title(f'SHAP Dependence: {feature}', fontsize=12)
                
        # Remove empty subplots
        for i in range(len(self.feature_names), len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle("SHAP Dependence Plots for All Features", fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def interaction_analysis(self, feature1=0, feature2=1, save_path="shap_interaction_plot.pdf"):
        """Analyze feature interactions using SHAP"""
        print(f"\nAnalyzing interaction between {self.feature_names[feature1]} and {self.feature_names[feature2]}...")
        
        # Compute SHAP interaction values
        shap_interaction_values = self.explainer.shap_interaction_values(self.X_test_df)
        
        if isinstance(shap_interaction_values, list):
            interaction_values_class1 = shap_interaction_values[1]
        elif isinstance(shap_interaction_values, np.ndarray) and shap_interaction_values.ndim == 4:
            interaction_values_class1 = shap_interaction_values[:, :, :, 1]
        else:
            interaction_values_class1 = shap_interaction_values
            
        plt.figure(figsize=(10, 8))
        
        # Create interaction plot
        shap.dependence_plot(
            (feature1, feature2),
            interaction_values_class1,
            self.X_test_df,
            feature_names=self.feature_names,
            show=False
        )
        
        plt.title(f"SHAP Interaction: {self.feature_names[feature1]} vs {self.feature_names[feature2]}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_close_current_figure(save_path)
        
    def run_complete_shap_analysis(self):
        """Run complete SHAP analysis"""
        print("=== Starting Complete SHAP Analysis ===\n")
        
        self.load_and_prepare_data()
        self.train_model()
        self.compute_shap_values()
        
        # Generate all SHAP plots
        self.plot_shap_summary(self.base_dir / "shap_summary_plot.pdf")
        self.plot_shap_summary_bar(self.base_dir / "shap_summary_bar.pdf")
        self.plot_shap_dependence(feature_index=0, save_path=self.base_dir / "shap_dependence_plot_mean_rssi.pdf")
        self.plot_shap_dependence(feature_index=1, save_path=self.base_dir / "shap_dependence_plot_num_active_bs.pdf")
        self.plot_force_plot(instance_idx=0, save_path=self.base_dir / "shap_force_plot.pdf")
        self.plot_waterfall_plot(instance_idx=0, save_path=self.base_dir / "shap_waterfall_plot.pdf")
        self.plot_multiple_dependence_plots(self.base_dir / "shap_multiple_dependence.pdf")
        self.interaction_analysis(feature1=0, feature2=1, save_path=self.base_dir / "shap_interaction_plot.pdf")
        
        print("\n=== SHAP Analysis Completed ===")

def main():
    """Main function to run SHAP analysis"""
    analyzer = SHAPAnalyzer()
    analyzer.run_complete_shap_analysis()

if __name__ == "__main__":
    main()
