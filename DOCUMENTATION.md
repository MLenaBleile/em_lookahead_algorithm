# Lookahead EM Algorithm: Implementation and Evaluation

## Overview

This project implements and evaluates a novel **Lookahead EM** algorithm that augments the standard Expectation-Maximization algorithm with a value function to look ahead and accelerate convergence.

### Key Idea

Standard EM updates parameters by maximizing the Q-function:
```
θ^(t+1) = argmax_θ Q(θ | θ^(t))
```

Lookahead EM adds a value function V(θ) that estimates the benefit of being at θ for future EM steps:
```
θ^(t+1) = argmax_θ [Q(θ | θ^(t)) + γ · V(θ)]
```

where γ controls the lookahead weight (0 = standard EM, higher = more aggressive lookahead).

### Value Function Approximation

We use a second-order approximation:
```
V(θ) ≈ Q(θ|θ) + 0.5 × ∇Q^T H_Q^(-1) ∇Q
```

This requires computing:
- **Gradient**: ∇Q with respect to parameters
- **Hessian**: H_Q (block-diagonal structure exploited for efficiency)

---

## Project Structure

```
lookahead_em_evaluation/
├── algorithms/
│   ├── standard_em.py          # Baseline Standard EM
│   ├── lookahead_em.py         # Lookahead EM implementation
│   ├── squarem_wrapper.py      # SQUAREM acceleration (R/Python)
│   ├── gamma_schedule.py       # Adaptive gamma schedules
│   └── value_estimation.py     # V(θ) approximation methods
├── models/
│   ├── gmm.py                  # Gaussian Mixture Model
│   ├── hmm.py                  # Hidden Markov Model
│   └── mixture_of_experts.py   # Mixture of Experts
├── data/
│   ├── generate_gmm.py         # GMM data generation
│   └── generate_hmm.py         # HMM data generation
├── tests/
│   ├── test_1_high_d_gmm.py    # High-D GMM experiments
│   ├── test_2_hmm.py           # HMM experiments
│   ├── test_3_moe.py           # MoE experiments
│   ├── test_4_robustness.py    # Robustness tests
│   ├── test_5_stress.py        # Stress tests
│   ├── test_moe_comprehensive.py  # Publication-ready MoE experiments
│   ├── test_runner.py          # Common test infrastructure
│   └── analyze_*.py            # Analysis scripts per test
├── analysis/
│   └── cross_test_comparison.py  # Cross-test analysis
├── utils/
│   ├── timing.py               # Resource monitoring
│   ├── metrics.py              # Evaluation metrics
│   ├── logging_utils.py        # Experiment logging
│   └── plotting.py             # Visualization functions
└── results/                    # Experiment outputs
```

---

## Implemented Algorithms

### 1. Standard EM (`algorithms/standard_em.py`)
Classic EM algorithm serving as baseline.

**Features:**
- Configurable convergence tolerance
- Timeout support
- Full diagnostics (likelihood history, timing)

### 2. Lookahead EM (`algorithms/lookahead_em.py`)
Novel algorithm with value-function lookahead.

**Features:**
- Gradient-based lookahead step with Newton direction
- Line search for optimal step size
- Adaptive gamma scheduling (fixed, adaptive, exponential)
- Fallback to simple extrapolation when gradients unavailable

**Gamma Schedules:**
- `gamma=0.5` - Fixed lookahead weight
- `gamma='adaptive'` - Linearly increasing from 0.1 to 0.9
- `gamma='exponential'` - Exponentially increasing

### 3. SQUAREM (`algorithms/squarem_wrapper.py`)
Squared Iterative Methods acceleration.

**Features:**
- R implementation via rpy2 (when available)
- Pure Python fallback for Windows/no-R environments
- Automatic method selection

---

## Supported Models

### 1. Gaussian Mixture Model (GMM)

**Parameters:** `θ = {μ, σ, π}`
- `μ`: Component means (K × d)
- `σ`: Diagonal covariances (K × d)
- `π`: Mixing weights (K,)

**Gradient Coverage:** Only μ (means)
- Limitation: Gradient is zero at M-step solution
- Result: Lookahead provides no benefit for GMM

### 2. Hidden Markov Model (HMM)

**Parameters:** `θ = {A, B, π₀}`
- `A`: Transition matrix (S × S), row-stochastic
- `B`: Emission matrix (S × V), row-stochastic
- `π₀`: Initial state distribution (S,)

**Gradient Coverage:** Full (π₀, A, B)
- Challenge: Expensive O(T × S²) forward-backward algorithm
- Result: Algorithm correct but slow (timeout issues)

### 3. Mixture of Experts (MoE)

**Parameters:** `θ = {γ, β, σ}`
- `γ`: Gating weights (G × (d+1))
- `β`: Expert weights (G × (d+1))
- `σ`: Noise standard deviations (G,)

**Gradient Coverage:** Full (γ, β, σ)
- Result: **Lookahead EM finds better optima than Standard EM**

---

## Key Implementation Details

### Parameter Ordering

The gradient vector must match the order expected by `_apply_step`:

```python
# GMM: gradient only covers mu
param_keys = ['mu']  # sigma, pi copied from M-step

# HMM: full gradient in order pi_0, A, B
param_keys = ['pi_0', 'A', 'B']

# MoE: full gradient in order gamma, beta, sigma
param_keys = ['gamma', 'beta', 'sigma']
```

### Constraint Handling

Parameters with constraints are projected after each step:

| Parameter | Constraint | Projection |
|-----------|------------|------------|
| `π`, `π₀` | Simplex (sum to 1) | Clip to [ε, 1], normalize |
| `σ` | Positive | Clip to [ε, ∞) |
| `A`, `B` | Row-stochastic | Clip rows to [ε, 1], normalize rows |

### Line Search

Lookahead step uses line search to find optimal step size:
```python
for step in [0, 0.1, 0.2, ..., γ]:
    θ_candidate = θ_mstep + step * direction
    score = Q(θ_candidate) + γ * V(θ_candidate)
    # Keep best
```

---

## Experimental Results Summary

### MoE (Mixture of Experts) - **Comprehensive Results Available**

A comprehensive evaluation was conducted with 1,080 runs across 12 configurations:
- Expert counts: [3, 5]
- Feature dimensions: [2, 5, 10]
- Sample sizes: [500, 1000]
- Restarts per configuration: 30

**Overall Comparison:**

| Algorithm | N | Log-Likelihood | Time (s) | Iterations |
|-----------|---|----------------|----------|------------|
| Standard EM | 358 | -1089.02 ± 394.90 | 1.41 ± 2.35 | 94 ± 80 |
| Lookahead EM | 345 | **-988.44 ± 360.53** | 43.50 ± 45.17 | 75 ± 64 |
| SQUAREM | 0* | N/A | N/A | N/A |

*SQUAREM failed all runs (requires rpy2 and R's turboEM package)

**Statistical Analysis:**

| Metric | Value |
|--------|-------|
| Total paired comparisons | 179 |
| Lookahead EM win rate | **98.9%** |
| Mean LL improvement | **+156.17 ± 153.63** |
| Paired t-test p-value | < 0.001 |
| Time ratio (LA/Std) | 26.23× |
| Iteration ratio (LA/Std) | 0.81× |

**Per-Condition Results:**

| Experts | Dim | Samples | Pairs | Win% | ΔLL |
|---------|-----|---------|-------|------|-----|
| 3 | 2 | 500 | 30 | 100% | +25.20 |
| 3 | 2 | 1000 | 28 | 100% | +51.46 |
| 3 | 5 | 500 | 30 | 100% | +53.51 |
| 3 | 5 | 1000 | 29 | 100% | +106.69 |
| 3 | 10 | 500 | 29 | 100% | +60.36 |
| 3 | 10 | 1000 | 30 | 100% | +114.76 |
| 5 | 2 | 500 | 27 | 96% | +29.17 |
| 5 | 2 | 1000 | 27 | 96% | +60.25 |
| 5 | 5 | 500 | 28 | 100% | +116.64 |
| 5 | 5 | 1000 | 28 | 96% | +199.48 |
| 5 | 10 | 500 | 29 | 93% | +112.69 |
| 5 | 10 | 1000 | 29 | 100% | +226.41 |

**Key Findings:**
- Lookahead EM consistently achieves better log-likelihood across all configurations
- Improvement scales with problem complexity (more experts, higher dimensions)
- Trade-off: ~26× slower due to gradient/Hessian computation per iteration
- All improvements statistically significant (p < 0.001)

Results file: `results/moe_comprehensive_20251217_234633.json`

### GMM (Gaussian Mixture Model) - **Implementation Note**

GMM gradient only covers μ (means), not σ or π. Since the gradient is zero at the M-step solution, lookahead provides no benefit for the current GMM implementation. Full gradient coverage would be needed for lookahead to help.

### HMM (Hidden Markov Model) - **Computational Challenges**

The O(T × S²) forward-backward algorithm makes gradient/Hessian computation expensive. Further optimization (e.g., caching) would be needed for practical use.

---

## When to Use Lookahead EM

### Good Candidates
- Models with **full gradient coverage** (all parameters in gradient)
- Problems where **solution quality matters more than speed**
- Scenarios prone to **local minima**
- Mixture of Experts, complex latent variable models

### Poor Candidates
- Models with **partial gradients** (like current GMM implementation)
- **Time-sensitive** applications (10-50x slower)
- Models with **expensive E-steps** (like HMM)

---

## Running Experiments

### Quick Validation
```bash
cd lookahead_em_evaluation

# MoE quick test
python tests/test_3_moe.py --quick

# Comprehensive MoE (publication-ready)
python tests/test_moe_comprehensive.py --quick
```

### Full Experiments
```bash
# Full MoE experiment (4,320 runs, several hours)
python tests/test_moe_comprehensive.py --restarts 30

# Custom configuration
python tests/test_moe_comprehensive.py \
    --experts 3 5 \
    --dims 2 5 10 \
    --samples 500 1000 \
    --restarts 30
```

### Output
Results saved to `results/moe_comprehensive_TIMESTAMP.json` with:
- Raw results for all runs
- Summary statistics by algorithm
- Summary statistics by configuration
- Win rates (lookahead vs standard)

---

## Known Limitations

1. **GMM Gradient Incomplete**: Current GMM only provides gradient for μ, not σ or π. Would need extension for lookahead to help.

2. **HMM Computational Cost**: Forward-backward is O(T × S²), making gradient-based lookahead expensive.

3. **SQUAREM Dependencies**: SQUAREM requires rpy2 and R's turboEM package. Install with:
   ```bash
   pip install rpy2
   R -e "install.packages('turboEM')"
   ```
   Without these, SQUAREM tests will fail with dependency errors.

4. **Windows Compatibility**: R-based SQUAREM may not work on Windows; pure Python fallback behavior varies.

5. **Memory Usage**: Storing Hessian matrices can be expensive for large models.

---

## Future Work

1. **Extend GMM gradient** to include σ and π parameters
2. **Optimize HMM** with cached forward-backward results
3. **Adaptive line search** with fewer steps for efficiency
4. **Stochastic lookahead** for large datasets
5. **Theoretical analysis** of convergence guarantees

---

## References

- Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. (SQUAREM)
- McLachlan, G.J., & Krishnan, T. (2007). The EM Algorithm and Extensions.

---

## Authors

Implementation and evaluation conducted as part of the Lookahead EM research project.
