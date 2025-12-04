# Contributing to DUST Factorization Research

Thank you for your interest in contributing! This project welcomes contributions from the mathematical and computer science communities.

## How to Contribute

### Reporting Issues

Found a bug or have a question? Please:

1. Check existing [GitHub Issues](https://github.com/ducci-research/dust-factorization-paper/issues)
2. If your issue is new, open an issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - System information (Python version, OS)
   - Error messages (if applicable)

### Suggesting Enhancements

We welcome ideas for:

- **Algorithmic improvements**: Faster density computation, better residue ordering
- **Theoretical extensions**: Tighter GRH bounds, universality proofs
- **New experiments**: Different primorial levels, larger semiprimes
- **Cryptographic applications**: RSA key auditing, vulnerability detection

Please open an issue with:
- Clear motivation
- Proposed approach
- Expected impact

### Code Contributions

#### Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/dust-factorization-paper.git
cd dust-factorization-paper
```

3. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

4. Install development dependencies:
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Optional: testing and linting
```

#### Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use descriptive variable names (e.g., `density_map` not `dm`)
- Add docstrings to functions:

```python
def compute_density(primes, modulus, residue):
    """
    Compute prime density in residue class.
    
    Args:
        primes (list): List of prime numbers
        modulus (int): Primorial modulus M
        residue (int): Residue class r (0 ≤ r < M, gcd(r,M)=1)
    
    Returns:
        float: Density ρ(r) = |{p ≤ T : p ≡ r (mod M)}| / T
    """
    # Implementation
```

- Add comments for complex logic
- Keep functions under 50 lines when possible

#### Testing

Before submitting, verify:

```bash
# Run main algorithms
cd code
python lattice-map.py
python theoretical_bounds.py
python quantum_chaos_connection.py

# Check results match expected values
python -c "
import json
with open('entropic_factorization_*_complete.json') as f:
    r = json.load(f)
    assert r['speedup_median'] > 0.45, 'Speedup too low'
    assert r['cohens_d'] > 1.5, 'Effect size too small'
    print('✓ All checks passed')
"
```

#### Submitting Pull Requests

1. Commit your changes:
```bash
git add .
git commit -m "Add feature: brief description"
```

2. Push to your fork:
```bash
git push origin feature/your-feature-name
```

3. Open a Pull Request with:
   - Clear title
   - Description of changes
   - Reference to related issue (if applicable)
   - Results/benchmarks (if applicable)

4. Respond to review feedback

## Research Contributions

### Extending to New Primorial Levels

To test 8-primorial (M=9,699,690):

```python
# In lattice-map.py
M = 9699690  # 8-primorial
phi_M = 1658880  # Euler totient

# Warning: ~1.6M residues, requires 32GB RAM
# Consider sampling 10% of residues
```

### Computing More L-Function Zeros

To extend RH analysis to 10,000 zeros:

```python
# In rh_connection.py
max_zeros = 10000  # Instead of 1097
T_max = 200        # Extend height range

# Warning: ~6 hours computation time
# Improves explicit formula validation
```

### Cryptographic Applications

Example: RSA key weakness detector

```python
def audit_rsa_key(N, density_map):
    """
    Check if RSA modulus has weak factors.
    
    Returns:
        (is_weak, expected_speedup)
    """
    # Check if factors likely in high-density residues
    # Return speedup estimate for attacker
```

## Mathematical Contributions

### Theoretical Improvements

We welcome:

- **Unconditional bounds**: Improve GRH dependence
- **Universality proofs**: Rigorous proof of r̄=-0.50±0.10
- **Explicit formula**: Better convergence for small T
- **Connection to modular forms**: Link density to L-functions

### Writing Style

For mathematical writing:
- Use standard notation (paper follows Iwaniec-Kowalski conventions)
- Provide detailed proofs
- Include numerical examples
- Reference prior work appropriately

## Areas for Contribution

### High Priority

1. **8-primorial validation**: Confirm universality at M=9,699,690
2. **10,000 zeros**: Strengthen explicit formula analysis
3. **Cryptographic tool**: Build RSA auditing library
4. **Optimization**: Faster density computation (current: O(T·φ(M)))

### Medium Priority

5. **Visualization**: Interactive density map explorer
6. **Documentation**: Tutorial notebooks
7. **Benchmarking**: Compare to Pollard rho, ECM
8. **Scaling study**: Performance vs modulus size

### Long-term Research

9. **GRH-free bounds**: Remove hypothesis dependence
10. **Langlands connection**: Automorphic forms interpretation
11. **Quantum algorithm**: Adapt for quantum speedup
12. **Multi-prime extension**: Beyond semiprimes

## Code of Conduct

- Be respectful and professional
- Welcome newcomers
- Focus on mathematical rigor and computational correctness
- Acknowledge prior work appropriately
- Report security vulnerabilities privately to dinoducci@gmail.com

## Recognition

Contributors will be:
- Listed in GitHub contributors
- Acknowledged in paper updates (for significant contributions)
- Co-authors on follow-up papers (for major theoretical/algorithmic advances)

## Questions?

- Email: dinoducci@gmail.com, cchrisducci@gmail.com
- GitHub Discussions: [Open discussion](https://github.com/ducci-research/dust-factorization-paper/discussions)
- Website: https://dusttheory.com

## License

By contributing, you agree that your contributions will be licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). This protects the work from unauthorized commercial use while keeping it open for research and academic purposes.
