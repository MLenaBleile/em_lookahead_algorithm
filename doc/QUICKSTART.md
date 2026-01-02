# Quick Start Guide

This guide will get you up and running with the Lookahead EM package in minutes.

## Installation

### Option 1: Install from GitHub (recommended)

```bash
pip install git+https://github.com/MLenaBleile/em_lookahead_algorithm.git
```

### Option 2: Install from source

```bash
git clone https://github.com/MLenaBleile/em_lookahead_algorithm.git
cd em_lookahead_algorithm
pip install -e .
```

### Option 3: With R support (for SQUAREM comparison)

```bash
pip install "lookahead-em[r]"
```

### Verify installation

```bash
python -c "from lookahead_em_evaluation import LookaheadEM; print('OK')"
```

---

## Tutorial: Mixture of Experts

This tutorial demonstrates fitting a Mixture of Experts model using Lookahead EM.

### What is MoE?

A Mixture of Experts model combines multiple "expert" models, each specializing in different regions of the input space. A gating network learns which expert to use for each input.

**Use cases:**
- House price prediction (different pricing dynamics by region)
- Personalized recommendations (different user segments)
- Financial modeling (different market regimes)

### Step 1: Import the package

```python
from lookahead_em_evaluation import (
    LookaheadEM,
    StandardEM,
    MixtureOfExperts,
    generate_moe_data,
    initialize_equal_gates,
)
```

### Step 2: Generate sample data

```python
# Generate data from a true MoE model with 3 experts
X, y, expert_assignments, theta_true = generate_moe_data(
    n=500,           # 500 samples
    n_experts=3,     # 3 expert models
    n_features=2,    # 2 input features
    noise_level=0.5, # noise standard deviation
    seed=42          # for reproducibility
)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Expert distribution: {[sum(expert_assignments == i) for i in range(3)]}")
```

### Step 3: Initialize model and parameters

```python
# Create the model
model = MixtureOfExperts(n_experts=3, n_features=2)

# Initialize parameters (equal gating weights)
theta_init = initialize_equal_gates(n_experts=3, n_features=2)

print(f"Parameters: {list(theta_init.keys())}")
# gamma: gating weights (3 x 3)
# beta: expert regression coefficients (3 x 3)
# sigma: expert noise levels (3,)
```

### Step 4: Fit with Lookahead EM

```python
# Create the optimizer with adaptive gamma schedule
em = LookaheadEM(model, gamma='adaptive')

# Fit the model
theta_final, diagnostics = em.fit(
    data=(X, y),
    theta_init=theta_init,
    max_iter=100,
    tol=1e-6
)

print(f"Converged: {diagnostics['converged']}")
print(f"Iterations: {diagnostics['iterations']}")
print(f"Final log-likelihood: {diagnostics['likelihood_history'][-1]:.2f}")
```

### Step 5: Compare with Standard EM

```python
# Fit with standard EM for comparison
standard_em = StandardEM(model)
theta_std, diag_std = standard_em.fit(
    data=(X, y),
    theta_init=theta_init,
    max_iter=100,
    tol=1e-6
)

print(f"\nComparison:")
print(f"{'Algorithm':<15} {'Iterations':<12} {'Final LL':<12} {'Time (s)':<10}")
print("-" * 50)
print(f"{'Standard EM':<15} {diag_std['iterations']:<12} {diag_std['likelihood_history'][-1]:<12.2f} {diag_std['time_seconds']:<10.3f}")
print(f"{'Lookahead EM':<15} {diagnostics['iterations']:<12} {diagnostics['likelihood_history'][-1]:<12.2f} {diagnostics['time_seconds']:<10.3f}")
```

### Step 6: Examine the results

```python
import numpy as np

# Compare learned vs true parameters
print("\nLearned expert coefficients (beta):")
print(theta_final['beta'])

print("\nLearned noise levels (sigma):")
print(theta_final['sigma'])

# Check gating behavior
print("\nGating weights (gamma) - determines expert selection:")
print(theta_final['gamma'])
```

---

## Tutorial: Gaussian Mixture Model

### Step 1: Generate GMM data

```python
from lookahead_em_evaluation import (
    LookaheadEM,
    GaussianMixtureModel,
    generate_gmm_data,
)

# Generate clustered data
X, true_labels, theta_true = generate_gmm_data(
    n=1000,        # 1000 samples
    K=5,           # 5 clusters
    d=2,           # 2 dimensions
    separation=4,  # cluster separation
    sigma=1.0,     # cluster spread
    seed=42
)

print(f"Data shape: {X.shape}")
print(f"Cluster sizes: {[sum(true_labels == k) for k in range(5)]}")
```

### Step 2: Fit the model

```python
# Create model and initialize with k-means
model = GaussianMixtureModel(n_components=5, n_features=2)
theta_init = model.initialize_kmeans(X, random_state=42)

# Fit with Lookahead EM
em = LookaheadEM(model, gamma='adaptive')
theta_final, diagnostics = em.fit(X, theta_init)

print(f"Converged in {diagnostics['iterations']} iterations")
print(f"Final log-likelihood: {diagnostics['likelihood_history'][-1]:.2f}")
```

---

## Algorithm Options

### Gamma Schedules

The `gamma` parameter controls lookahead aggressiveness:

```python
# Fixed gamma (constant lookahead weight)
em = LookaheadEM(model, gamma=0.5)

# Adaptive gamma (recommended - increases over iterations)
em = LookaheadEM(model, gamma='adaptive')

# Exponential gamma (aggressive increase)
em = LookaheadEM(model, gamma='exponential')
```

### Convergence Settings

```python
theta_final, diagnostics = em.fit(
    data=(X, y),
    theta_init=theta_init,
    max_iter=200,     # maximum iterations
    tol=1e-8,         # convergence tolerance
    timeout=300       # timeout in seconds
)
```

### Verbose Output

```python
em = LookaheadEM(model, gamma='adaptive', verbose=True, verbose_interval=10)
```

---

## Diagnostics

The `diagnostics` dictionary contains:

| Key | Description |
|-----|-------------|
| `converged` | Whether the algorithm converged |
| `iterations` | Number of iterations run |
| `likelihood_history` | Log-likelihood at each iteration |
| `time_seconds` | Total runtime |
| `time_history` | Cumulative time at each iteration |
| `gamma_history` | Gamma values used (Lookahead EM) |
| `stopping_reason` | Why the algorithm stopped |

### Plotting convergence

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(diagnostics['likelihood_history'])
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.title('Convergence')

plt.subplot(1, 2, 2)
plt.plot(diagnostics['time_history'], diagnostics['likelihood_history'])
plt.xlabel('Time (s)')
plt.ylabel('Log-likelihood')
plt.title('Convergence vs Time')

plt.tight_layout()
plt.show()
```

---

## Complete Example Script

Save this as `example.py` and run it:

```python
"""Complete example: Compare Standard EM vs Lookahead EM on MoE"""

from lookahead_em_evaluation import (
    LookaheadEM,
    StandardEM,
    MixtureOfExperts,
    generate_moe_data,
    initialize_equal_gates,
)

# Generate data
print("Generating MoE data...")
X, y, experts, theta_true = generate_moe_data(
    n=500, n_experts=3, n_features=2, seed=42
)

# Setup
model = MixtureOfExperts(n_experts=3, n_features=2)
theta_init = initialize_equal_gates(n_experts=3, n_features=2)

# Fit Standard EM
print("Fitting Standard EM...")
std_em = StandardEM(model)
theta_std, diag_std = std_em.fit((X, y), theta_init)

# Fit Lookahead EM
print("Fitting Lookahead EM...")
la_em = LookaheadEM(model, gamma='adaptive')
theta_la, diag_la = la_em.fit((X, y), theta_init)

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"{'Algorithm':<15} {'Iters':<8} {'Final LL':<12} {'Time (s)':<10}")
print("-" * 60)
print(f"{'Standard EM':<15} {diag_std['iterations']:<8} {diag_std['likelihood_history'][-1]:<12.2f} {diag_std['time_seconds']:<10.3f}")
print(f"{'Lookahead EM':<15} {diag_la['iterations']:<8} {diag_la['likelihood_history'][-1]:<12.2f} {diag_la['time_seconds']:<10.3f}")

# Compare final likelihoods
ll_diff = diag_la['likelihood_history'][-1] - diag_std['likelihood_history'][-1]
print(f"\nLookahead EM improvement: {ll_diff:+.2f} log-likelihood units")
```

---

## Troubleshooting

### Import errors

If you get `ModuleNotFoundError`, ensure the package is installed:
```bash
pip install -e .  # from the repo root
```

### Slow convergence

Try:
- Increasing `max_iter`
- Using `gamma='adaptive'` instead of fixed gamma
- Better initialization (e.g., k-means for GMM)

### Memory issues

For large datasets, consider:
- Reducing the number of components/experts
- Using a subset of data for initialization

---

## Next Steps

- See `DOCUMENTATION.md` for algorithm details
- Run `tests/test_3_moe.py` for comprehensive experiments
- Check `tests/test_moe_comprehensive.py` for publication-ready benchmarks
