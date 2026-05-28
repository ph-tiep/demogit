# -*- coding: utf-8 -*-
"""Run baseline, VAE, and Digital Twin pipelines, then compare all results"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*70)
    print(f" {description}")
    print("="*70 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n Script {script_name} not found!")
        return False

def main():
    base_dir = Path(__file__).resolve().parent

    print("\n" + "="*70)
    print("  BASELINE vs VAE vs DIGITAL TWIN COMPARISON EXPERIMENT")
    print("="*70)
    print("\nThis script will:")
    print("1. Run baseline pipeline (original features + RF/IF)")
    print("2. Run VAE pipeline (compressed features + RF/IF)")
    print("3. Run Digital Twin pipeline (EKF-based, no RF/IF)")
    print("4. Compare and visualize all results")
    print("\n This may take several minutes...\n")

    input("Press ENTER to start or Ctrl+C to cancel...")

    results = {}

    # Run baseline
    results['baseline'] = run_script("main.py", "BASELINE PIPELINE (RF + IF)")

    # Run VAE
    if results['baseline']:
        results['vae'] = run_script("main_vae.py", "VAE-ENHANCED PIPELINE")
    else:
        print("\n Skipping VAE pipeline due to baseline failure")
        results['vae'] = False

    # Run Digital Twin
    if results['baseline']:
        results['digital_twin'] = run_script("main_digital_twin.py", "DIGITAL TWIN PIPELINE (EKF)")
    else:
        print("\n Skipping Digital Twin pipeline due to baseline failure")
        results['digital_twin'] = False

    # Compare results
    if results.get('baseline') and results.get('digital_twin'):
        results['comparison'] = run_script("compare_results.py", "RESULTS COMPARISON")
    else:
        print("\n Skipping comparison due to pipeline failures")
        results['comparison'] = False

    # Print summary
    print("\n" + "="*70)
    print(" EXECUTION SUMMARY")
    print("="*70)
    print(f"Baseline Pipeline:       {' Success' if results['baseline'] else ' Failed'}")
    print(f"VAE Pipeline:            {' Success' if results.get('vae') else ' Failed'}")
    print(f"Digital Twin Pipeline:   {' Success' if results.get('digital_twin') else ' Failed'}")
    print(f"Comparison:              {' Success' if results.get('comparison') else ' Failed'}")

    if all(v for k, v in results.items() if k != 'comparison'):
        print("\n All pipelines completed successfully!")
        print("\n Results locations:")
        print(f"   - Baseline:      {base_dir / 'results' / 'baseline'}")
        print(f"   - VAE:           {base_dir / 'results' / 'vae'}")
        print(f"   - Digital Twin:  {base_dir / 'results' / 'digital_twin'}")
        print(f"   - Comparison:    {base_dir / 'results' / 'comparison'}")
    else:
        print("\n Some pipelines failed. Check the logs above for details.")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
