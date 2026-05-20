# -*- coding: utf-8 -*-
"""Run both baseline and VAE pipelines, then compare results"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script {script_name} not found!")
        return False

def main():
    base_dir = Path(__file__).resolve().parent
    
    print("\n" + "="*70)
    print("🔬 BASELINE vs VAE COMPARISON EXPERIMENT")
    print("="*70)
    print("\nThis script will:")
    print("1. Run baseline pipeline (original features)")
    print("2. Run VAE pipeline (compressed features)")
    print("3. Compare and visualize results")
    print("\n⏱️  This may take several minutes...\n")
    
    input("Press ENTER to start or Ctrl+C to cancel...")
    
    results = {}
    
    # Run baseline
    results['baseline'] = run_script("main.py", "BASELINE PIPELINE")
    
    # Run VAE
    if results['baseline']:
        results['vae'] = run_script("main_vae.py", "VAE-ENHANCED PIPELINE")
    else:
        print("\n⚠️  Skipping VAE pipeline due to baseline failure")
        results['vae'] = False
    
    # Compare results
    if results['baseline'] and results['vae']:
        results['comparison'] = run_script("compare_results.py", "RESULTS COMPARISON")
    else:
        print("\n⚠️  Skipping comparison due to pipeline failures")
        results['comparison'] = False
    
    # Print summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    print(f"Baseline Pipeline:  {'✅ Success' if results['baseline'] else '❌ Failed'}")
    print(f"VAE Pipeline:       {'✅ Success' if results.get('vae') else '❌ Failed'}")
    print(f"Comparison:         {'✅ Success' if results.get('comparison') else '❌ Failed'}")
    
    if all(results.values()):
        print("\n🎉 All pipelines completed successfully!")
        print("\n📁 Results locations:")
        print(f"   - Baseline:   {base_dir / 'results' / 'baseline'}")
        print(f"   - VAE:        {base_dir / 'results' / 'vae'}")
        print(f"   - Comparison: {base_dir / 'results' / 'comparison'}")
    else:
        print("\n⚠️  Some pipelines failed. Check the logs above for details.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
