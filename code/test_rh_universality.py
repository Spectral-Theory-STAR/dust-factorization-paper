"""
Riemann Hypothesis Connection: Universality Testing
Tests eigenvalue-zero correlation across different primorial moduli (6,8-primorial)

Run: python test_rh_universality.py
Output: Results for each modulus + universality report
"""

import json
import numpy as np
from pathlib import Path
from rh_connection import run_rh_connection_analysis

def generate_density_for_modulus(modulus, training_limit=1_500_000):
    """
    Generate density map for arbitrary modulus using Sieve of Eratosthenes.
    Similar to extract_density.py but more flexible.
    """
    print(f"\nGenerating density map for modulus {modulus}...")
    
    # Sieve of Eratosthenes
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(training_limit)
    print(f"Generated {len(primes)} primes up to {training_limit}")
    
    # Count primes per residue
    density_map = {}
    phi_M = sum(1 for r in range(modulus) if np.gcd(r, modulus) == 1)
    
    for p in primes:
        if p > modulus:  # Skip primes dividing modulus
            r = p % modulus
            if np.gcd(r, modulus) == 1:
                density_map[str(r)] = density_map.get(str(r), 0) + 1
    
    # Normalize
    total_primes = sum(density_map.values())
    for r in density_map:
        density_map[r] /= total_primes
    
    # Fill missing residues with zero
    for r in range(modulus):
        if np.gcd(r, modulus) == 1 and str(r) not in density_map:
            density_map[str(r)] = 0.0
    
    mean_density = total_primes / phi_M / total_primes if total_primes > 0 else 0
    
    print(f"φ({modulus}) = {phi_M}")
    print(f"Mean density: {mean_density:.6f}")
    print(f"Total residues with data: {len(density_map)}")
    
    return {
        "modulus": modulus,
        "phi_M": phi_M,
        "training_limit": training_limit,
        "total_primes": total_primes,
        "density_map": density_map
    }

def test_universality(max_zeros=1000, T_max=40):
    """
    Test RH connection across multiple primorial moduli.
    
    Moduli tested:
    - 6-primorial: M = 30030 (φ = 5760)
    - 7-primorial: M = 510510 (φ = 92160) [baseline]
    - 8-primorial subset: M = 9699690 (φ = 1658880) [subset test]
    """
    
    results = {}
    
    # Test configurations
    configs = [
        {
            "name": "6-primorial",
            "modulus": 30030,  # 2*3*5*7*11*13
            "description": "Smaller primorial for faster computation",
            "max_zeros": max_zeros,
            "T_max": T_max
        },
        {
            "name": "7-primorial",
            "modulus": 510510,  # 2*3*5*7*11*13*17
            "description": "Baseline modulus from main paper",
            "max_zeros": max_zeros,
            "T_max": T_max,
            "existing_file": "entropic_factorization_20251203_223001_complete_with_density.json"
        },
        # 8-primorial is too large (1.6M residues), skip for now
    ]
    
    for config in configs:
        print("\n" + "="*80)
        print(f"Testing {config['name'].upper()}: M = {config['modulus']}")
        print("="*80)
        
        # Generate or load density map
        if "existing_file" in config and Path(config["existing_file"]).exists():
            print(f"Using existing density map: {config['existing_file']}")
            density_file = config["existing_file"]
        else:
            # Generate new density map
            density_data = generate_density_for_modulus(config["modulus"])
            
            # Save to file
            density_file = f"rh_universality_{config['name']}_density.json"
            with open(density_file, 'w') as f:
                json.dump({
                    "modulus": config["modulus"],
                    "phi_M": density_data["phi_M"],
                    "training_limit": density_data["training_limit"],
                    "empirical_density": density_data["density_map"]
                }, f, indent=2)
            
            print(f"Saved density map to: {density_file}")
        
        # Run RH connection analysis
        try:
            rh_results = run_rh_connection_analysis(
                density_file=density_file,
                modulus=config["modulus"],
                max_zeros=config["max_zeros"],
                T_max=config["T_max"]
            )
            
            # Extract correlation data from nested structure
            if isinstance(rh_results, dict) and "correlation_analysis" in rh_results:
                corr_data = rh_results["correlation_analysis"]["correlation"]
            elif isinstance(rh_results, dict) and "correlation" in rh_results:
                corr_data = rh_results["correlation"]
            else:
                raise ValueError(f"Unexpected results structure: {list(rh_results.keys()) if isinstance(rh_results, dict) else type(rh_results)}")
            
            # Count total zeros
            total_zeros = sum(len(zeros) for zeros in rh_results.get("L_function_zeros", {}).values())
            num_chars = rh_results.get("configuration", {}).get("num_characters", 0)
            
            phi_M = config.get("phi_M", density_data.get("phi_M", 0) if "existing_file" not in config else 92160)
            
            results[config["name"]] = {
                "modulus": config["modulus"],
                "phi_M": phi_M,
                "correlation": {
                    "pearson_r": corr_data["pearson_r"],
                    "pearson_p": corr_data["pearson_p"],
                    "spearman_r": corr_data["spearman_r"],
                    "spearman_p": corr_data["spearman_p"]
                },
                "total_zeros": total_zeros,
                "num_characters": num_chars,
                "description": config["description"]
            }
            
            print(f"\n{config['name']} Results:")
            print(f"  Pearson r  = {corr_data['pearson_r']:.4f} (p = {corr_data['pearson_p']:.2e})")
            print(f"  Spearman ρ = {corr_data['spearman_r']:.4f} (p = {corr_data['spearman_p']:.2e})")
            print(f"  Total zeros: {total_zeros}")
            
        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")
            results[config["name"]] = {
                "error": str(e),
                "modulus": config["modulus"]
            }
    
    # Universality analysis
    print("\n" + "="*80)
    print("UNIVERSALITY ANALYSIS")
    print("="*80)
    
    successful_tests = [k for k, v in results.items() if "error" not in v]
    
    if len(successful_tests) >= 2:
        correlations = [results[k]["correlation"]["pearson_r"] for k in successful_tests]
        p_values = [results[k]["correlation"]["pearson_p"] for k in successful_tests]
        
        print(f"\nCorrelations across {len(successful_tests)} primorial levels:")
        for name in successful_tests:
            r = results[name]["correlation"]["pearson_r"]
            p = results[name]["correlation"]["pearson_p"]
            sig = "✓✓✓" if p < 0.001 else "✓✓" if p < 0.01 else "✓" if p < 0.05 else "✗"
            print(f"  {name:15s}: r = {r:+.4f}, p = {p:.2e} {sig}")
        
        print(f"\nUniversality metrics:")
        print(f"  Mean correlation: r̄ = {np.mean(correlations):.4f}")
        print(f"  Std deviation:    σ = {np.std(correlations):.4f}")
        print(f"  Range:            [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        print(f"  Coefficient of variation: {np.std(correlations)/abs(np.mean(correlations)):.2%}")
        
        if all(p < 0.05 for p in p_values):
            print(f"\n✓ UNIVERSALITY CONFIRMED: All tests significant at p < 0.05")
            if np.std(correlations) < 0.1:
                print(f"✓ STRONG UNIVERSALITY: σ < 0.1 indicates consistent effect")
        else:
            print(f"\n✗ UNIVERSALITY UNCLEAR: Not all tests significant")
    else:
        print(f"\nInsufficient successful tests ({len(successful_tests)}) for universality analysis")
    
    # Save results
    output_file = "rh_universality_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RH connection universality across primorial moduli")
    parser.add_argument("--max-zeros", type=int, default=1000, help="Maximum zeros per character (default: 1000)")
    parser.add_argument("--T-max", type=float, default=40, help="Maximum imaginary part for zero search (default: 40)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("RIEMANN HYPOTHESIS CONNECTION: UNIVERSALITY TESTING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Max zeros per character: {args.max_zeros}")
    print(f"  Zero search range: t ∈ [0, {args.T_max}]")
    print(f"\nExpected runtime: ~15-30 minutes per modulus")
    
    results = test_universality(max_zeros=args.max_zeros, T_max=args.T_max)
    
    print("\n" + "="*80)
    print("UNIVERSALITY TESTING COMPLETE")
    print("="*80)
