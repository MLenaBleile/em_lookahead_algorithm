# Lookahead EM Algorithm

An optimization algorithm which can outperform SoA on Mixture of Experts models. Includes comprehensive validation suite of simulations.

## Overview

This project implements empirical tests comparing:
- **Standard EM** (baseline)
- **Lookahead EM** (novel algorithm with adaptive gamma scheduling)
- **SQUAREM** (sequence extrapolation via R's turboEM)
- **Quasi-Newton EM** (second-order observed likelihood methods)

## Project Structure

```
lookahead_em_evaluation/
├── algorithms/          # EM algorithm implementations
│   ├── standard_em.py
│   ├── lookahead_em.py
│   ├── gamma_schedule.py
│   ├── value_estimation.py
│   └── squarem_wrapper.py
├── models/              # Statistical models (GMM, HMM, MoE)
│   ├── gmm.py
│   ├── hmm.py
│   └── mixture_of_experts.py
├── data/                # Data generation utilities
│   ├── generate_gmm.py
│   ├── generate_hmm.py
│   └── generate_moe.py
├── tests/               # Test implementations
│   ├── test_1_high_d_gmm.py
│   ├── test_2_hmm.py
│   ├── test_3_moe.py
│   ├── test_4_robustness.py
│   └── test_5_real_data.py
├── utils/               # Utilities
│   ├── timing.py
│   ├── metrics.py
│   ├── logging_utils.py
│   └── plotting.py
├── results/             # Experiment results
├── notebooks/           # Analysis notebooks
└── run_all_tests.py     # Master test runner
```

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/MLenaBleile/em_lookahead_algorithm.git
```

### From source

```bash
git clone https://github.com/MLenaBleile/em_lookahead_algorithm.git
cd em_lookahead_algorithm
pip install -e .
```

### Optional: R integration for SQUAREM

```bash
pip install "lookahead-em[r]"
```

## Quick Start

### Mixture of Experts Example

```python
from lookahead_em_evaluation import (
    LookaheadEM, MixtureOfExperts, generate_moe_data, initialize_equal_gates
)

# Generate sample data
X, y, experts, theta_true = generate_moe_data(n=500, n_experts=3, n_features=2)

# Fit with Lookahead EM
model = MixtureOfExperts(n_experts=3, n_features=2)
theta_init = initialize_equal_gates(n_experts=3, n_features=2)
em = LookaheadEM(model, gamma='adaptive')
theta_final, diagnostics = em.fit((X, y), theta_init)

print(f"Converged in {diagnostics['iterations']} iterations")
print(f"Final log-likelihood: {diagnostics['likelihood_history'][-1]:.2f}")
```

### Gaussian Mixture Model Example

```python
from lookahead_em_evaluation import (
    LookaheadEM, StandardEM, GaussianMixtureModel, generate_gmm_data
)

# Generate test data
X, z, theta_true = generate_gmm_data(n=1000, K=5, d=2, separation=4, sigma=1.0, seed=42)

# Initialize model
model = GaussianMixtureModel(n_components=5, n_features=2)
theta_init = model.initialize_kmeans(X, random_state=42)

# Compare Standard EM vs Lookahead EM
standard_em = StandardEM(model)
theta_std, diag_std = standard_em.fit(X, theta_init)

lookahead_em = LookaheadEM(model, gamma='adaptive')
theta_la, diag_la = lookahead_em.fit(X, theta_init)

print(f"Standard EM: {diag_std['iterations']} iterations")
print(f"Lookahead EM: {diag_la['iterations']} iterations")
```

### Run all tests

```bash
python run_all_tests.py --tests 1,2,3,4,5 --parallel --n_jobs 8
```

### Quick mode (fewer restarts)

```bash
python run_all_tests.py --tests 1 --quick
```

## Test Suite

| Test | Model | Description | Expected Advantage |
|------|-------|-------------|-------------------|
| 1 | High-D GMM (K=50) | Block-diagonal H_Q structure | 30-50% speedup |
| 2 | HMM (S=20, V=50) | Sparse transition matrix | Exploit sparsity |
| 3 | Mixture of Experts | Hierarchical block structure | Block-wise inversion |
| 4 | Robustness | Bad initializations | Adaptive gamma helps |
| 5 | Real Data | Practical validation | Scientific utility |

## Use case

Lookahead EM provides advantages when:
1. **H_Q is structured** (sparse, block-diagonal, low-rank)
2. **Problem is high-dimensional** (p > 100)
3. **Adaptive gamma schedule** balances exploration and exploitation

## References

- Varadhan & Roland (2008): SQUAREM method
- Jamshidian & Jennrich (1997): Quasi-Newton EM
- Dempster, Laird & Rubin (1977): Original EM algorithm

## License

MIT License
