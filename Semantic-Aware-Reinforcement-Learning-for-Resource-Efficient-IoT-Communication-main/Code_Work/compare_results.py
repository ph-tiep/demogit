# -*- coding: utf-8 -*-
"""Compare results between baseline and VAE approaches"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def main():
    base_dir = Path(__file__).resolve().parent
    baseline_dir = base_dir / "results" / "baseline"
    vae_dir = base_dir / "results" / "vae"
    comparison_dir = base_dir / "results" / "comparison"
    
    # Create comparison directory
    comparison_dir.mkdir(exist_ok=True)
    
    print("=== Comparing Baseline vs VAE Results ===\n")
    
    # Get list of files in each directory
    baseline_files = set()
    vae_files = set()
    
    if baseline_dir.exists():
        baseline_files = {f.name for f in baseline_dir.iterdir() if f.is_file()}
    
    if vae_dir.exists():
        vae_files = {f.name for f in vae_dir.iterdir() if f.is_file()}
    
    common_files = baseline_files & vae_files
    baseline_only = baseline_files - vae_files
    vae_only = vae_files - baseline_files
    
    print("\n" + "="*70)
    print("FILE COMPARISON")
    print("="*70)
    
    print(f"\n📁 Common files ({len(common_files)}):")
    for f in sorted(common_files):
        print(f"   ✓ {f}")
    
    if baseline_only:
        print(f"\n📊 Baseline only ({len(baseline_only)}):")
        for f in sorted(baseline_only):
            print(f"   • {f}")
    
    if vae_only:
        print(f"\n🔬 VAE only ({len(vae_only)}):")
        for f in sorted(vae_only):
            print(f"   • {f}")
    
    # File size comparison
    print("\n" + "="*70)
    print("FILE SIZE COMPARISON")
    print("="*70)
    print(f"\n{'File':<40} {'Baseline':<12} {'VAE':<12} {'Diff'}")
    print("-" * 70)
    
    for filename in sorted(common_files):
        baseline_file = baseline_dir / filename
        vae_file = vae_dir / filename
        
        baseline_size = baseline_file.stat().st_size / 1024  # KB
        vae_size = vae_file.stat().st_size / 1024  # KB
        diff = vae_size - baseline_size
        diff_str = f"+{diff:.1f} KB" if diff > 0 else f"{diff:.1f} KB"
        
        print(f"{filename:<40} {baseline_size:>10.1f} KB {vae_size:>10.1f} KB {diff_str}")
    
    # Generate summary report
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print("\n📊 BASELINE APPROACH:")
    print("   - Uses original semantic features (5D)")
    print("   - Features: mean_rssi, num_active_bs, Latitude, Longitude, hour")
    print(f"   - Results location: {baseline_dir}")
    print(f"   - Total files: {len(baseline_files)}")
    
    print("\n🔬 VAE APPROACH:")
    print("   - Uses VAE-compressed features (3D latent space)")
    print("   - Compression ratio: 5D → 3D (1.67x compression)")
    print("   - Benefits: Reduced dimensionality, denoising, learned representations")
    print(f"   - Results location: {vae_dir}")
    print(f"   - Total files: {len(vae_files)}")
    
    # Create a text summary file
    summary_file = comparison_dir / "comparison_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE vs VAE COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("APPROACH COMPARISON:\n")
        f.write("-" * 70 + "\n")
        f.write("Baseline:\n")
        f.write("  - Feature dimension: 5D\n")
        f.write("  - Features: mean_rssi, num_active_bs, Lat, Lon, hour\n")
        f.write("  - Total files: {}\n\n".format(len(baseline_files)))
        
        f.write("VAE:\n")
        f.write("  - Feature dimension: 3D (compressed)\n")
        f.write("  - Compression ratio: 1.67x\n")
        f.write("  - Total files: {}\n\n".format(len(vae_files)))
        
        f.write("\nCOMMON FILES:\n")
        f.write("-" * 70 + "\n")
        for file in sorted(common_files):
            f.write(f"  ✓ {file}\n")
        
        if vae_only:
            f.write("\nVAE-SPECIFIC FILES:\n")
            f.write("-" * 70 + "\n")
            for file in sorted(vae_only):
                f.write(f"  • {file}\n")
    
    print(f"\n📄 Summary report saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("KEY METRICS TO COMPARE:")
    print("="*70)
    print("1. Classification Accuracy (Confusion Matrix)")
    print("2. Feature Importance (How features contribute)")
    print("3. ROC-AUC Scores (Model discrimination ability)")
    print("4. Anomaly Detection Performance")
    print("5. Training Efficiency (VAE may speed up DQN)")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Open and compare PDF files manually in:")
    print(f"   - {baseline_dir}")
    print(f"   - {vae_dir}")
    print("2. Look for differences in:")
    print("   - ROC-AUC scores (higher is better)")
    print("   - Confusion matrix accuracy")
    print("   - Feature importance patterns")
    print("3. VAE-specific files to check:")
    if vae_only:
        for f in sorted(vae_only):
            print(f"   - {f}")
    
    print("\n✅ Comparison completed successfully!")
    print(f"📊 View results in: {baseline_dir} and {vae_dir}")

if __name__ == "__main__":
    main()
