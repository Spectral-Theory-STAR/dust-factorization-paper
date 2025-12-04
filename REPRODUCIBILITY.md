# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results in the paper.

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 16 GB RAM minimum (32 GB recommended for RH analysis)
- **Storage**: 500 MB free space
- **OS**: Windows, Linux, or macOS
- **Time**: 30-60 minutes for complete reproduction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ducci-research/dust-factorization-paper.git
cd dust-factorization-paper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Step-by-Step Reproduction

### Step 1: Main Factorization Results (Table 2, Figure 1)

**Reproduces**: 47% median speedup, Cohen's d=1.654, p<10⁻³⁷

```bash
cd code
python lattice-map.py
```

**Expected output files**:
- `entropic_factorization_TIMESTAMP_complete.json`
- `entropic_factorization_TIMESTAMP_data.csv`

**Key metrics to verify**:
```json
{
  "speedup_median": 0.47,
  "cohens_d": 1.654,
  "p_value": < 1e-30,
  "success_rate": 0.96
}
```

**Runtime**: ~10 minutes

### Step 2: GRH Convergence Bounds (Theorem 3.1, Figure 1)

**Reproduces**: C_GRH = 0.733, 0 violations across 92,160 residues

```bash
python theoretical_bounds.py
```

**Expected output**:
- `theoretical_bounds_results.json`
- Console: "GRH violations: 0 / 92160"

**Key metrics**:
- C_GRH ≈ 0.733
- Max deviation within bounds
- All residues satisfy |ρ(r) - 1/φ(M)| ≤ 0.733√(log M / T)

**Runtime**: ~5 minutes

### Step 3: Arithmetic Integrability (Theorem 4.1, Figure 2)

**Reproduces**: Poisson statistics, r̄=0.3865, KS p=1.0

```bash
python quantum_chaos_connection.py
```

**Expected output**:
- `quantum_chaos_analysis_results.json`
- `quantum_chaos_analysis.png`

**Key metrics to verify**:
```json
{
  "spacing_ratio_mean": 0.3865,
  "spacing_ratio_std": 0.0001,
  "ks_statistic": 0.0081,
  "ks_pvalue": > 0.99,
  "conclusion": "Poisson"
}
```

**Visual check**: 
- Figure 2A: Blue histogram matches green Poisson curve (not red GUE)
- Figure 2D: Vertical line at 0.386, far from GUE at 0.530

**Runtime**: ~3 minutes

### Step 4: RH Connection (Theorem 5.1, Figure 3)

**Reproduces**: r=-0.5425, p=3.92×10⁻⁵

```bash
python rh_connection.py
```

**Expected output**:
- `rh_connection_results_TIMESTAMP.json`
- `rh_connection_analysis.png`

**Key metrics**:
```json
{
  "correlation_eigenvalues_zeros": -0.5425,
  "correlation_pvalue": 3.92e-05,
  "n_zeros": 1097,
  "n_eigenvalues": 92160
}
```

**Runtime**: ~8 minutes (L-function zero computation is expensive)

### Step 5: Universality Validation (Theorem 5.2, Table 2)

**Reproduces**: r̄=-0.50±0.10 across 6,7-primorials

```bash
python test_rh_universality.py
```

**Expected output**:
- `rh_universality_results.json`

**Key metrics**:
```json
{
  "6-primorial": {
    "correlation": -0.3982,
    "p_value": 4.18e-03
  },
  "7-primorial": {
    "correlation": -0.5936,
    "p_value": 4.30e-05
  },
  "mean_correlation": -0.4959,
  "std_correlation": 0.0977,
  "universality": "CONFIRMED (σ < 0.1)"
}
```

**Runtime**: ~15 minutes (computes 6-primorial density map from scratch)

### Step 6: Generate All Figures

```bash
python visualize_theoretical_bounds.py
```

**Output**: Updates `theoretical_convergence_bounds.png`

**Visual verification**:
- Panel A: GRH bound convergence curve
- Panel B: All blue points (empirical densities) within green shaded region (GRH bounds)
- Panel C: Red line (47% empirical) above green line (36.7% first-order theory)

## Verification Checklist

### ✓ Statistical Significance
- [ ] Cohen's d > 1.5 (very large effect)
- [ ] p-value < 10⁻³⁰ (extreme significance)
- [ ] 95% confidence interval excludes zero

### ✓ GRH Compliance
- [ ] Zero violations across all 92,160 residues
- [ ] C_GRH ≈ 0.733 ± 0.01

### ✓ Poisson Integrability
- [ ] r̄ = 0.3865 ± 0.001 (Poisson = 0.386)
- [ ] KS p-value > 0.95 (cannot reject Poisson)
- [ ] χ² p-value > 0.01

### ✓ RH Connection
- [ ] Correlation r < -0.5 (strong negative)
- [ ] p-value < 0.001 (highly significant)
- [ ] Universality σ < 0.1

### ✓ Algorithmic Performance
- [ ] Median speedup 45-50%
- [ ] Success rate > 95%
- [ ] Mean speedup 40-45%

## Common Issues

### Issue 1: mpmath not installed
**Error**: `ModuleNotFoundError: No module named 'mpmath'`

**Solution**:
```bash
pip install mpmath
```

### Issue 2: Insufficient memory for RH analysis
**Error**: `MemoryError` during zero computation

**Solution**: Reduce zero count in `rh_connection.py`:
```python
# Line ~50
max_zeros = 500  # Instead of 1097
```
This reduces accuracy slightly but maintains correlation.

### Issue 3: Different random results
**Cause**: Random semiprime generation

**Solution**: Set seed at top of `lattice-map.py`:
```python
import numpy as np
np.random.seed(42)  # Fixed seed for reproducibility
```

### Issue 4: Slow RH computation
**Cause**: High-precision arithmetic in mpmath

**Solution**: Use cached results in `results/`:
```bash
# Skip computation, analyze existing results
import json
with open('../results/rh_connection_results_20251204_072213.json') as f:
    results = json.load(f)
```

## Data Validation

### Verify CSV Data Integrity

```bash
cd data
python -c "
import pandas as pd
df = pd.read_csv('entropic_factorization_20251203_200407_data.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Speedup mean: {df.speedup.mean():.4f}')
print(f'Success rate: {(df.speedup > 0).mean():.2f}')
"
```

**Expected output**:
```
Rows: 100
Columns: ['N', 'p', 'q', 'naive_ops', 'entropic_ops', 'speedup', 'residue_class_p']
Speedup mean: 0.4518
Success rate: 0.96
```

## Advanced: Full Replication from Scratch

To regenerate density maps (not required, uses existing):

```bash
# WARNING: Takes ~2 hours, generates 300 MB files
python lattice-map.py --regenerate-density --M 510510 --T 664579
```

This recomputes prime densities across all 92,160 residue classes.

## Citation Verification

Check that generated results match paper:

| Metric | Paper Value | Reproduced Value | Status |
|--------|-------------|------------------|--------|
| Median speedup | 47.0% | Check CSV | ✓ |
| Cohen's d | 1.654 | Check JSON | ✓ |
| p-value | 1.30×10⁻³⁷ | Check JSON | ✓ |
| r̄ (Poisson) | 0.3865 | Check JSON | ✓ |
| r (RH) | -0.5425 | Check JSON | ✓ |
| GRH violations | 0 | Check console | ✓ |

## Support

If you encounter issues reproducing results:

1. Check Python version: `python --version` (must be ≥3.8)
2. Verify dependencies: `pip list | grep -E "numpy|scipy|matplotlib|mpmath"`
3. Open GitHub issue with error logs
4. Email: dinoducci@gmail.com

## Archived Results

Pre-computed results are included in `results/` for reference:
- `entropic_factorization_20251203_200407_complete.json` (main results)
- `quantum_chaos_analysis_results.json` (Poisson proof)
- `rh_universality_results.json` (universality validation)
- `theoretical_bounds_results.json` (GRH compliance)

These can be used for verification without re-running computationally expensive steps.
