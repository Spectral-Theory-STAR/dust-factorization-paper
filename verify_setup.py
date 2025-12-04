#!/usr/bin/env python3
"""
Verify DUST Factorization Package Setup
Quick check that all dependencies and files are present.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verify Python >= 3.8"""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need >= 3.8)")
        return False

def check_dependencies():
    """Verify required packages"""
    required = ['numpy', 'scipy', 'matplotlib', 'mpmath']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} (missing)")
            missing.append(pkg)
    
    return len(missing) == 0, missing

def check_files():
    """Verify all required files exist"""
    base = Path(__file__).parent
    
    required_files = [
        'README.md',
        'LICENSE',
        'requirements.txt',
        'code/lattice-map.py',
        'code/theoretical_bounds.py',
        'code/quantum_chaos_connection.py',
        'code/rh_connection.py',
        'code/test_rh_universality.py',
        'paper/research-paper.tex',
        'figures/quantum_chaos_analysis.png',
        'figures/rh_connection_analysis.png',
        'figures/theoretical_convergence_bounds.png',
        'data/entropic_factorization_20251203_200407_data.csv',
        'results/entropic_factorization_20251203_200407_complete.json'
    ]
    
    missing = []
    for f in required_files:
        path = base / f
        if path.exists():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} (missing)")
            missing.append(f)
    
    return len(missing) == 0, missing

def check_data_integrity():
    """Verify data files are valid"""
    base = Path(__file__).parent
    
    # Check CSV
    try:
        import pandas as pd
        csv_path = base / 'data/entropic_factorization_20251203_200407_data.csv'
        df = pd.read_csv(csv_path)
        
        if len(df) == 100:
            print(f"✓ CSV data (100 trials)")
        else:
            print(f"⚠ CSV data ({len(df)} trials, expected 100)")
        
        required_cols = ['semiprime_N', 'factor_p', 'factor_q', 'linear_operations', 'entropic_operations', 'speedup_ratio']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if not missing_cols:
            print("✓ CSV columns")
        else:
            print(f"✗ CSV missing columns: {missing_cols}")
            return False
            
    except Exception as e:
        print(f"✗ CSV validation failed: {e}")
        return False
    
    # Check JSON
    try:
        import json
        json_path = base / 'results/entropic_factorization_20251203_200407_complete.json'
        with open(json_path) as f:
            data = json.load(f)
        
        # Check for top-level structure
        required_keys = ['metadata', 'performance_metrics', 'statistical_tests']
        missing_keys = [k for k in required_keys if k not in data]
        if not missing_keys:
            print("✓ JSON results")
        else:
            print(f"✗ JSON missing keys: {missing_keys}")
            return False
            
    except Exception as e:
        print(f"✗ JSON validation failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("DUST Factorization Package - Setup Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Check 1: Python version
    print("[1/4] Checking Python version...")
    if not check_python_version():
        all_passed = False
    print()
    
    # Check 2: Dependencies
    print("[2/4] Checking dependencies...")
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_passed = False
        print(f"\nInstall missing packages: pip install {' '.join(missing_deps)}")
    print()
    
    # Check 3: Files
    print("[3/4] Checking required files...")
    files_ok, missing_files = check_files()
    if not files_ok:
        all_passed = False
        print(f"\n✗ Missing {len(missing_files)} files")
    print()
    
    # Check 4: Data integrity
    print("[4/4] Checking data integrity...")
    if deps_ok and files_ok:
        if not check_data_integrity():
            all_passed = False
    else:
        print("⊘ Skipped (requires dependencies and files)")
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nYou're ready to reproduce the results!")
        print("Next steps:")
        print("  1. cd code")
        print("  2. python lattice-map.py")
        print("\nSee REPRODUCIBILITY.md for detailed instructions.")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("See README.md for installation instructions.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
