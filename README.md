# Lookahead EM Empirical Evaluation

Systematic evaluation of the lookahead EM algorithm vs. existing acceleration methods for expectation-maximization.

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
│   └── generate_hmm.py
├── tests/               # Test implementations
│   ├── test_1_high_d_gmm.py
│   ├── test_2_hmm.py
│   ├── test_3_moe.py
│   ├── test_4_robustness.py
│   ├── test_5_stress.py
│   ├── test_moe_comprehensive.py  # Publication-ready MoE test
│   ├── test_runner.py
│   └── analyze_*.py     # Analysis scripts per test
├── analysis/            # Cross-test analysis
│   └── cross_test_comparison.py
├── utils/               # Utilities
│   ├── timing.py
│   ├── metrics.py
│   ├── logging_utils.py
│   └── plotting.py
├── results/             # Experiment results
└── notebooks/           # Analysis notebooks
```

## Setup

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate lookahead_em
```

## Quick Start

### Run a simple test

```python
from algorithms.standard_em import StandardEM
from algorithms.lookahead_em import LookaheadEM
from models.gmm import GaussianMixtureModel
from data.generate_gmm import generate_gmm_data

# Generate test data
X, z, theta_true = generate_gmm_data(n=1000, K=5, d=2, separation=4, sigma=1.0, seed=42)

# Initialize model
model = GaussianMixtureModel(n_components=5, n_features=2)

# Initialize parameters
theta_init = model.initialize_kmeans(X, random_state=42)

# Run standard EM
standard_em = StandardEM(model)
theta_std, diag_std = standard_em.fit(X, theta_init)

# Run lookahead EM with adaptive gamma
lookahead_em = LookaheadEM(model, gamma='adaptive')
theta_la, diag_la = lookahead_em.fit(X, theta_init)

print(f"Standard EM: {diag_std['iterations']} iterations, {diag_std['time_seconds']:.2f}s")
print(f"Lookahead EM: {diag_la['iterations']} iterations, {diag_la['time_seconds']:.2f}s")
```

### Run comprehensive MoE test

```bash
cd lookahead_em_evaluation
python tests/test_moe_comprehensive.py --restarts 30
```

### Quick mode (fewer restarts)

```bash
python tests/test_moe_comprehensive.py --quick
```

## Test Suite

| Test | Model | Description | Status |
|------|-------|-------------|--------|
| 1 | High-D GMM (K=50) | Block-diagonal H_Q structure | Implemented |
| 2 | HMM (S=20, V=50) | Sparse transition matrix | Implemented |
| 3 | Mixture of Experts | Hierarchical block structure | **Results Available** |
| 4 | Robustness | Bad initializations | Implemented |
| 5 | Stress Test | Edge cases and limits | Implemented |

## Latest Results: Mixture of Experts (MoE)

Comprehensive evaluation with 1,080 runs across 12 configurations:

| Algorithm | Converged | Log-Likelihood | Time (s) | Iterations |
|-----------|-----------|----------------|----------|------------|
| Standard EM | 358/360 (99.4%) | -1089.02 ± 394.90 | 1.41 ± 2.35 | 94 ± 80 |
| Lookahead EM | 345/360 (95.8%) | **-988.44 ± 360.53** | 43.50 ± 45.17 | 75 ± 64 |
| SQUAREM | 0/360 (0%)* | N/A | N/A | N/A |

*SQUAREM results not available: test uses `SQUAREMWrapper` directly instead of `get_squarem()` fallback

**Key Finding:** Lookahead EM achieves significantly better solution quality:
- **98.9% win rate** over Standard EM (177/179 paired comparisons)
- **+156.17 mean log-likelihood improvement** (p < 0.001)
- Uses **19% fewer iterations** (0.81× ratio)
- Trade-off: **26× slower** per iteration due to gradient/Hessian computation

Results saved in: `results/moe_comprehensive_20251217_234633.json`

## Core Hypothesis

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
