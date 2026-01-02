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
│   ├── generate_hmm.py         # HMM data generation
│   └── generate_moe.py         # MoE data generation
├── tests/
│   ├── test_1_gmm.py           # GMM experiments
│   ├── test_2_hmm.py           # HMM experiments
│   ├── test_3_moe.py           # MoE experiments
│   ├── test_moe_comprehensive.py  # Publication-ready MoE experiments
│   └── test_runner.py          # Common test infrastructure
├── utils/
│   └── timing.py               # Resource monitoring
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

**Example Application: Regional House Price Prediction**

MoE models excel when the input-output relationship varies across different regions of the input space. Consider predicting house prices based on features like square footage and age:

```
Input: x = [sq_ft, age, ...]
Output: y = price

Expert 1 (Urban): price = β₁ᵀx + ε   (high base, small sq_ft coefficient)
Expert 2 (Suburban): price = β₂ᵀx + ε (medium base, large sq_ft coefficient)
Expert 3 (Rural): price = β₃ᵀx + ε   (low base, land-focused)

Gating: P(expert | x) = softmax(γᵀx)  (learns which expert applies where)
```

The gating network learns to route samples to the appropriate expert based on input features. Unlike a single linear model, MoE captures that a 2000 sq ft house has very different pricing dynamics in Manhattan vs rural Montana.

**Other MoE use cases:**
- **Personalized recommendations**: Different user segments follow different preference patterns
- **Medical diagnosis**: Disease progression varies by patient subpopulations
- **Financial modeling**: Market regimes (bull/bear) require different predictive models
- **Speech recognition**: Different acoustic models for different speakers/accents

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

### MoE (Mixture of Experts) - **Positive Results**

| Algorithm | Conv% | Final LL | Time (s) | Iterations |
|-----------|-------|----------|----------|------------|
| Lookahead EM | **100%** | **-246.86** | 10.04 | 58 |
| SQUAREM | 100% | -257.96 | 0.48 | 35 |
| Standard EM | 100% | -258.02 | 0.20 | 77 |

**Key Finding:** Lookahead EM achieves ~4% better log-likelihood by escaping local minima.

### GMM (Gaussian Mixture Model) - **No Improvement**

| Algorithm | Conv% | Final LL | Time (s) | Iterations |
|-----------|-------|----------|----------|------------|
| Standard EM | 50% | -1866.43 | 0.40 | 463 |
| Lookahead EM | 50% | -1866.43 | 5.71 | 463 |

**Why:** GMM gradient only covers μ, which is zero at M-step. Lookahead correction is effectively zero.

### HMM (Hidden Markov Model) - **Timeout Issues**

| Algorithm | Conv% | Final LL | Time (s) | Iterations |
|-----------|-------|----------|----------|------------|
| Standard EM | 50% | -361.69 | 27.76 | 82 |
| Lookahead EM | 0% | -367.54 | 121.03 (timeout) | 38 |

**Why:** O(T × S²) gradient/Hessian computation makes each iteration ~3s vs 0.3s for standard EM.

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

3. **Windows Compatibility**: R-based SQUAREM may not work; pure Python fallback is used.

4. **Memory Usage**: Storing Hessian matrices can be expensive for large models.

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
