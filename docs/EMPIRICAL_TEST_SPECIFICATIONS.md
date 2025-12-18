# Empirical Test Specifications: Lookahead EM Algorithm Evaluation

**Author:** MaryLena Bleile  
**Date:** December 2025  
**Purpose:** Systematic evaluation of lookahead EM algorithm vs. existing acceleration methods

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Definitions](#algorithm-definitions)
3. [Test Suite](#test-suite)
   - [Test 1: High-Dimensional Sparse GMM](#test-1-high-dimensional-sparse-gmm)
   - [Test 2: Hidden Markov Model](#test-2-hidden-markov-model-large-state-space)
   - [Test 3: Mixture of Experts](#test-3-mixture-of-experts-hierarchical-structure)
   - [Test 4: Robustness to Bad Initialization](#test-4-robustness-to-bad-initialization)
   - [Test 5: Real Data Application](#test-5-real-data-application-diggle--kenward)
4. [Implementation Structure](#implementation-structure)
5. [Metrics and Reporting](#metrics-and-reporting)
6. [Success Criteria](#success-criteria)
7. [Timeline](#timeline)

---

## Overview

### Goal

Demonstrate that **lookahead EM** provides advantages in specific scenarios where:
1. Q-function Hessian (H_Q) has exploitable structure (sparse, block-diagonal)
2. Observed likelihood Hessian (H_ℓ) is intractable or expensive
3. Adaptive acceleration (via tunable γ parameter) provides benefit

### Strategy

Design tests where we **expect** lookahead EM to win based on theoretical analysis, then **honestly report all results** (including failures).

### Core Hypothesis

Lookahead EM should outperform existing methods when:
- **H_Q is structured** (sparse, block-diagonal, low-rank)
- **Problem is high-dimensional** (p > 100)
- **Adaptive γ schedule** can balance exploration and exploitation

---

## Algorithm Definitions

### 1. Standard EM (Baseline)

**Update rule:**
```
θ^(t+1) = argmax_θ Q(θ | θ^(t))
```

**Properties:**
- First-order method
- Guaranteed monotonic convergence
- Often slow

### 2. Lookahead EM (Novel Algorithm)

**Update rule:**
```
θ^(t+1) = argmax_θ [Q(θ | θ^(t)) + γ · V(θ)]

where V(θ) ≈ Q(θ|θ) + (1/2) ∇Q^T H_Q^(-1) ∇Q
```

**Parameters:**
- **γ ∈ [0,1]:** Discount factor
  - γ=0: Reduces to standard EM
  - γ=1: Maximum lookahead
- **Adaptive schedule:** γ_t = min(γ_max, γ_0 + α·t)
  - Example: γ_t = min(0.9, 0.1 + 0.05t)

**Implementation notes:**
- Requires computing ∇Q and H_Q at each candidate θ
- For structured H_Q (sparse, block-diagonal), use efficient inversion
- Enforce constraints (e.g., σ > 0, π sums to 1) during optimization

### 3. SQUAREM (Competitor)

**Algorithm:** SqS3 variant from Varadhan & Roland (2008)

**Implementation:** Use `turboEM` R package, call from Python via rpy2

**Properties:**
- Sequence extrapolation method
- No derivatives required
- Works well in practice

### 4. Quasi-Newton EM (Competitor, if tractable)

**Algorithm:** Jamshidian & Jennrich (1997) method

**Implementation:** Approximate H_ℓ using BFGS updates

**Properties:**
- Second-order method on observed likelihood
- May be intractable for high-dimensional problems
- Include only if computationally feasible

---

## Test Suite

### Test 1: High-Dimensional Sparse GMM

#### Objective
Demonstrate that lookahead EM exploits block-diagonal H_Q structure for speed and memory efficiency.

#### Data Generation

**Model:** Gaussian Mixture Model
- **K = 50** components
- **d = 2** dimensions per component (bivariate normal)
- **n = 5,000** observations

**True parameters:**
```python
# Component means
means = np.array([
    [i % 10 - 4.5, i // 10 - 2.5] 
    for i in range(50)
]) * 4  # Grid arrangement on [-18, 18] × [-10, 10]

# Component standard deviations (fixed)
sigmas = np.ones((50, 2))  # σ = 1 for all components

# Mixing proportions (equal)
pi = np.ones(50) / 50  # π_k = 1/50
```

**Data generation:**
```python
np.random.seed(42)
z = np.random.choice(50, size=5000, p=pi)  # Component assignments
x = np.zeros((5000, 2))
for i in range(5000):
    x[i] = np.random.normal(means[z[i]], sigmas[z[i]])
```

#### Why This Test?

**H_Q structure:**
- Block-diagonal: 50 blocks of size 2×2
- Each block corresponds to one component's mean parameters
- Total parameters: 100 (can fix σ and π, or estimate them too)
- Sparse: No interaction between components in Q

**H_ℓ structure:**
- Dense 100×100 matrix
- All parameters coupled through mixture weights
- Expensive to compute/approximate

**Expected advantage:** Lookahead EM uses exact block-diagonal H_Q efficiently

#### Initialization

**Strategy:** Suboptimal K-means
```python
from sklearn.cluster import KMeans

# Deliberately bad: only 10 iterations of K-means
kmeans = KMeans(n_clusters=50, max_iter=10, n_init=1, random_state=seed)
labels = kmeans.fit_predict(x)

# Initialize from cluster assignments
for k in range(50):
    mask = (labels == k)
    theta_init['mu'][k] = x[mask].mean(axis=0) if mask.sum() > 0 else np.random.randn(2)
    theta_init['sigma'][k] = np.ones(2)
    theta_init['pi'][k] = mask.sum() / n
```

**Random restarts:** 20 different random seeds (seed ∈ {0, 1, ..., 19})

#### Algorithms to Compare

1. **Standard EM** (baseline)
2. **Lookahead EM** with fixed γ:
   - γ = 0.3 (conservative)
   - γ = 0.5 (moderate)
   - γ = 0.7 (aggressive)
   - γ = 0.9 (very aggressive)
3. **Lookahead EM** with adaptive γ:
   - γ_t = min(0.9, 0.1 + 0.05 × min(t, 16))
   - Starts at 0.1, reaches 0.9 by iteration 16
4. **SQUAREM (SqS3)**
5. **Quasi-Newton EM** (if memory permits for p=100)

#### Stopping Criteria

Stop when **any** of the following is met:
1. **Convergence:** ||θ^(t+1) - θ^(t)||_∞ < 10^(-6)
2. **Max iterations:** t > 1000
3. **Timeout:** elapsed time > 600 seconds
4. **Stagnation:** |ℓ^(t+1) - ℓ^(t)| < 10^(-8) for 5 consecutive iterations

#### Metrics

**Primary:**
- **Time to convergence** (seconds, wall-clock time)
- **Memory usage** (peak MB)

**Secondary:**
- **Iterations to convergence**
- **Final log-likelihood** (higher is better)
- **Success rate:** % of runs reaching ℓ > (best_ℓ - 0.5)

**Track per iteration:**
- Current log-likelihood ℓ(θ^(t))
- Parameter change ||Δθ||
- Elapsed time
- For lookahead EM: Value of γ used

#### Expected Results

**Hypothesis:**
- Lookahead EM (adaptive) should be **30-50% faster** than standard EM
- Should use **less memory** than quasi-Newton (O(p) vs O(p²))
- SQUAREM might be competitive on well-initialized runs
- Lookahead should have higher success rate on badly initialized runs

---

### Test 2: Hidden Markov Model (Large State Space)

#### Objective
Demonstrate lookahead EM on problems with sparse H_Q structure.

#### Data Generation

**Model:** Hidden Markov Model (HMM)
- **S = 20** hidden states
- **V = 50** observation vocabulary
- **n = 1,000** time steps

**True parameters:**

Transition matrix (20 × 20):
```python
# Nearly diagonal structure
A_true = np.zeros((20, 20))
for i in range(20):
    A_true[i, i] = 0.7  # Self-transition
    if i > 0:
        A_true[i, i-1] = 0.15  # Backward
    if i < 19:
        A_true[i, i+1] = 0.15  # Forward
# Normalize rows
A_true = A_true / A_true.sum(axis=1, keepdims=True)
```

Emission matrix (20 × 50):
```python
# Each state prefers 2-3 specific observations
B_true = np.ones((20, 50)) * 0.01  # Small background probability
for s in range(20):
    # State s prefers observations around index 2.5*s
    preferred_obs = [int(2.5*s + offset) % 50 for offset in [-1, 0, 1]]
    for o in preferred_obs:
        B_true[s, o] = 0.33
# Normalize rows
B_true = B_true / B_true.sum(axis=1, keepdims=True)
```

Initial state distribution:
```python
pi_0 = np.zeros(20)
pi_0[0] = 1.0  # Always start in state 0
```

**Data generation:**
```python
states = [0]
observations = []

for t in range(1000):
    # Emit observation
    obs = np.random.choice(50, p=B_true[states[-1]])
    observations.append(obs)
    
    # Transition to next state
    next_state = np.random.choice(20, p=A_true[states[-1]])
    states.append(next_state)
```

#### Why This Test?

**H_Q structure:**
- Transition parameters: Sparse (most entries near zero)
- Emission parameters: Can be separated by state
- Total: 400 + 1000 = 1400 parameters

**H_ℓ structure:**
- Dense (all sequences contribute to all parameters)
- Forward-backward couples everything
- Very expensive to compute

**Expected advantage:** Exploit sparsity in transition matrix

#### Initialization

```python
# Uniform transition
A_init = np.ones((20, 20)) / 20

# Random emission (with Dirichlet)
B_init = np.random.dirichlet(np.ones(50), size=20)

# Uniform initial state
pi_0_init = np.ones(20) / 20
```

**Random restarts:** 20 different random seeds

#### Algorithms to Compare

1. Standard EM
2. Lookahead EM (adaptive γ)
3. SQUAREM
4. Quasi-Newton EM (if tractable)

#### Stopping Criteria

Same as Test 1, but with:
- Max iterations: 500
- Timeout: 300 seconds

#### Metrics

Same as Test 1, plus:
- **Transition sparsity:** % of transitions with probability > 0.01
- **Stability:** Did algorithm respect simplex constraints?

#### Expected Results

**Hypothesis:**
- Lookahead EM exploits sparse structure
- Should be faster and more stable than dense H_ℓ approximations

---

### Test 3: Mixture of Experts (Hierarchical Structure)

#### Objective
Show that lookahead EM exploits block-diagonal structure in hierarchical models.

#### Data Generation

**Model:** Mixture of Experts (MoE)
- **G = 5** experts
- **Covariates:** x ∈ ℝ^2 (2-dimensional input)
- **Response:** y ∈ ℝ (continuous output)
- **n = 2,000** observations

**True parameters:**

Gating network (softmax):
```python
# Coefficients for gating (5 × 3 including intercept)
gamma_true = np.array([
    [0, 0, 0],      # Expert 0 (reference)
    [1, -1, 0.5],   # Expert 1
    [-1, 1, -0.5],  # Expert 2
    [0.5, 0.5, 1],  # Expert 3
    [-0.5, -0.5, -1] # Expert 4
])
```

Expert networks (linear regression):
```python
# Coefficients for each expert (5 × 3 including intercept)
beta_true = np.array([
    [2, 0.5, -0.5],   # Expert 0
    [-2, -0.5, 0.5],  # Expert 1
    [0, 1, 1],        # Expert 2
    [0, -1, -1],      # Expert 3
    [1, 0.2, 0.8]     # Expert 4
])

# Noise variance
sigma_expert = 0.5
```

**Data generation:**
```python
# Generate covariates
X = np.random.randn(2000, 2)

# Compute gating probabilities
X_with_intercept = np.c_[np.ones(2000), X]  # Add intercept
logits = X_with_intercept @ gamma_true.T
gates = softmax(logits, axis=1)

# Sample expert assignments
z = np.array([np.random.choice(5, p=gates[i]) for i in range(2000)])

# Generate responses from assigned experts
y = np.zeros(2000)
for i in range(2000):
    expert = z[i]
    y[i] = X_with_intercept[i] @ beta_true[expert] + np.random.normal(0, sigma_expert)
```

#### Why This Test?

**H_Q structure:**
- Block-diagonal:
  - Gating block (15 parameters)
  - Expert 0 block (3 parameters)
  - Expert 1 block (3 parameters)
  - ...
  - Expert 4 block (3 parameters)
- Complete-data model: Gates and experts decouple
- Total: 15 + 15 = 30 parameters

**H_ℓ structure:**
- Dense (gates and experts coupled)
- Marginalizing over latent expert assignments creates dependencies

**Expected advantage:** Block structure easy to exploit

#### Initialization

```python
# Equal gate probabilities (poor initialization)
gamma_init = np.zeros((5, 3))

# Random expert coefficients
beta_init = np.random.randn(5, 3) * 0.1
```

**Random restarts:** 30 different random seeds

#### Algorithms to Compare

1. Standard EM
2. Lookahead EM with fixed γ: {0.2, 0.4, 0.6, 0.8}
3. Lookahead EM (adaptive)
4. SQUAREM

#### Stopping Criteria

Same as Test 1, but:
- Max iterations: 500
- Timeout: 180 seconds

#### Metrics

Same as Test 1, plus:
- **Expert utilization:** Are all 5 experts being used? (Count with π_g > 0.05)

#### Expected Results

**Hypothesis:**
- Adaptive γ helps: Conservative early, aggressive later
- Block-diagonal structure speeds up H_Q inversion

---

### Test 4: Robustness to Bad Initialization

#### Objective
Test whether lookahead EM with adaptive γ provides better robustness to poor initialization.

#### Data Generation

**Model:** Simple GMM (for interpretability)
- **K = 5** components
- **d = 2** dimensions
- **n = 500** observations (100 per component)

**True parameters:**
```python
# Well-separated means on a line
means_true = np.array([
    [-5, 0],
    [-2.5, 0],
    [0, 0],
    [2.5, 0],
    [5, 0]
])

# Tight clusters
sigmas_true = np.array([[0.5, 0.5]] * 5)

# Equal mixing
pi_true = np.ones(5) / 5
```

**Data generation:** Standard GMM sampling

#### Why This Test?

**Focus:** Exploration ability, not speed
- Can low γ help escape bad local optima?
- Can adaptive γ balance exploration and convergence?

#### Initialization Strategies

Run each initialization strategy separately (not random restarts within each):

**Strategy 1: All Collapsed**
```python
# All means at origin
theta_init['mu'] = np.zeros((5, 2))
theta_init['sigma'] = np.ones((5, 2))
theta_init['pi'] = np.ones(5) / 5
```
Repeat with 20 random perturbations: `mu += np.random.randn(5, 2) * 0.01`

**Strategy 2: Far Away**
```python
# All means far from data
theta_init['mu'] = np.ones((5, 2)) * 10  # At [10, 10]
theta_init['sigma'] = np.ones((5, 2))
theta_init['pi'] = np.ones(5) / 5
```
Repeat with 20 random perturbations

**Strategy 3: Wrong Number of Components**
```python
# Initialize as K=2, then force to K=5 by splitting
# Fit K=2 GMM first, then split each component
```
Repeat 20 times with different random splits

**Strategy 4: Completely Random**
```python
# Random means in [-10, 10] × [-10, 10]
theta_init['mu'] = np.random.uniform(-10, 10, size=(5, 2))
theta_init['sigma'] = np.random.uniform(0.3, 2.0, size=(5, 2))
theta_init['pi'] = np.random.dirichlet(np.ones(5))
```
Repeat 50 times (50 random initializations)

**Total runs:** 20 + 20 + 20 + 50 = 110 runs per algorithm

#### Algorithms to Compare

1. Standard EM (γ=0)
2. Lookahead EM γ=0.2 (conservative)
3. Lookahead EM γ=0.5 (moderate)
4. Lookahead EM γ=0.8 (aggressive)
5. Lookahead EM (adaptive: 0.2 → 0.8)
6. SQUAREM

#### Success Criterion

Define **global optimum region:** ℓ(θ) > ℓ_true - 5

Count **success rate:** % of runs reaching global optimum region

#### Metrics

**Primary:**
- **Success rate** by initialization strategy
- **Average final ℓ** by initialization strategy

**Secondary:**
- Average iterations to convergence
- Average time to convergence
- Convergence distribution (histogram of final ℓ values)

#### Expected Results

**Hypothesis:**
- Adaptive γ might help with Strategy 1 & 2 (bad starts)
- Unclear if γ affects exploration (state space is constant!)
- Might see no difference (would be negative but honest result)

**Interpretation:**
- If no difference: Shows that lookahead doesn't help exploration (expected, since no state transitions)
- If adaptive helps: Suggests second-order info aids escaping local optima

---

### Test 5: Real Data Application (Diggle & Kenward)

#### Objective
Validate practical utility on real scientific problem.

#### Dataset

**Source:** Your informative dropout paper data
- Diggle & Kenward dataset
- Longitudinal measurements with dropout

**If not available:** Use another real dataset with missingness:
- Growth curve data
- Clinical trial with dropout
- Gene expression with missing values

#### Task

Fit mixture model to identify latent classes (e.g., dropout patterns, disease subtypes)

#### Initialization

```python
# K-means on available data
# 10 random restarts
```

#### Algorithms to Compare

1. Standard EM
2. Lookahead EM (γ=0.5 fixed)
3. Lookahead EM (adaptive)
4. Whatever method you used in your original paper

#### Metrics

**Quantitative:**
- Final log-likelihood
- AIC / BIC
- Time to convergence

**Qualitative:**
- Interpretability of clusters
- Scientific validity (do results make sense?)
- Stability across random restarts (same clustering?)

#### Expected Results

**Hypothesis:**
- Should perform comparably or better than existing methods
- Main value: Demonstrate method works on real problem
- Scientific validation matters more than raw speed

---

## Implementation Structure

### Directory Organization

```
lookahead_em_evaluation/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── algorithms/
│   ├── __init__.py
│   ├── standard_em.py          # Baseline
│   ├── lookahead_em.py          # Your algorithm
│   ├── squarem_wrapper.py       # Python → R bridge
│   └── quasi_newton_em.py       # If tractable
│
├── models/
│   ├── __init__.py
│   ├── gmm.py                   # Gaussian Mixture Model
│   ├── hmm.py                   # Hidden Markov Model
│   └── mixture_of_experts.py    # MoE implementation
│
├── data/
│   ├── generate_gmm.py
│   ├── generate_hmm.py
│   ├── generate_moe.py
│   └── real_data.csv            # For Test 5
│
├── tests/
│   ├── __init__.py
│   ├── test_1_high_d_gmm.py
│   ├── test_2_hmm.py
│   ├── test_3_moe.py
│   ├── test_4_robustness.py
│   └── test_5_real_data.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   ├── plotting.py              # Visualization
│   ├── timing.py                # Time/memory tracking
│   └── logging_utils.py         # Experiment logging
│
├── results/
│   ├── test_1/
│   ├── test_2/
│   ├── test_3/
│   ├── test_4/
│   └── test_5/
│
├── notebooks/
│   ├── analyze_test_1.ipynb
│   ├── analyze_test_2.ipynb
│   ├── analyze_test_3.ipynb
│   ├── analyze_test_4.ipynb
│   └── analyze_test_5.ipynb
│
└── run_all_tests.py             # Master script
```

### Key Implementation Requirements

#### For `lookahead_em.py`:

```python
class LookaheadEM:
    def __init__(self, model, gamma='adaptive', gamma_max=0.9, 
                 gamma_init=0.1, gamma_step=0.05):
        """
        Args:
            model: Statistical model (GMM, HMM, etc.)
            gamma: 'adaptive' or float in [0, 1]
            gamma_max: Maximum gamma for adaptive schedule
            gamma_init: Initial gamma for adaptive schedule
            gamma_step: Increment per iteration for adaptive schedule
        """
        self.model = model
        self.gamma_schedule = self._setup_gamma_schedule(
            gamma, gamma_max, gamma_init, gamma_step
        )
    
    def fit(self, X, theta_init, max_iter=1000, tol=1e-6, timeout=600):
        """
        Run lookahead EM algorithm.
        
        Returns:
            theta_final: Fitted parameters
            diagnostics: Dict with convergence info
        """
        pass
    
    def _lookahead_step(self, X, theta_current, gamma):
        """
        Perform one lookahead EM step.
        
        Steps:
            1. Compute Q(θ|θ_current) for θ in candidate set
            2. For each candidate θ, compute V(θ):
               - Compute ∇Q(θ|θ) and H_Q(θ|θ)
               - V(θ) = Q(θ|θ) + 0.5 * ∇Q^T H_Q^(-1) ∇Q
            3. Select θ_next = argmax [Q(θ|θ_current) + γ V(θ)]
        
        Returns:
            theta_next: Next parameter estimate
            diagnostics: Step-level info
        """
        pass
```

#### For Metrics (`utils/metrics.py`):

```python
def compute_metrics(theta_history, likelihood_history, 
                   time_history, theta_true=None):
    """
    Compute all evaluation metrics.
    
    Returns:
        metrics: Dict with:
            - iterations: int
            - time: float (seconds)
            - final_likelihood: float
            - convergence_rate: float
            - success: bool (if theta_true provided)
            - parameter_error: float (if theta_true provided)
    """
    pass

def check_monotonicity(likelihood_history):
    """Check if likelihood is monotonically increasing."""
    pass

def memory_usage():
    """Get current memory usage in MB."""
    pass
```

#### For Timing (`utils/timing.py`):

```python
import time
import psutil
import threading

class ResourceMonitor:
    """Monitor time and memory usage during algorithm execution."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_memory = 0
        self.monitoring = False
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        self.start_time = time.time()
    
    def stop(self):
        """Stop monitoring and return stats."""
        self.monitoring = False
        self.thread.join()
        elapsed = time.time() - self.start_time
        return {
            'elapsed_time': elapsed,
            'peak_memory_mb': self.peak_memory
        }
    
    def _monitor(self):
        """Background monitoring loop."""
        process = psutil.Process()
        while self.monitoring:
            mem = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, mem)
            time.sleep(self.interval)
```

---

## Metrics and Reporting

### Per-Run Metrics

For each individual run, record:

```python
run_result = {
    'test_id': 'test_1',
    'algorithm': 'lookahead_adaptive',
    'seed': 42,
    'initialization_strategy': 'kmeans_10iter',
    
    # Convergence
    'converged': True,
    'iterations': 87,
    'stopping_reason': 'tolerance',  # or 'max_iter', 'timeout', 'stagnation'
    
    # Quality
    'final_likelihood': -2340.7,
    'likelihood_history': [array of ℓ values],
    'parameter_history': [array of θ values],
    
    # Efficiency
    'time_seconds': 29.8,
    'time_per_iteration': 0.34,
    'peak_memory_mb': 145,
    
    # For lookahead only
    'gamma_history': [array of γ values used],
    'v_value_history': [array of V(θ) estimates],
    
    # If ground truth available
    'parameter_error': 0.023,
    'reached_global_optimum': True
}
```

### Aggregate Metrics

For each (test, algorithm) pair, compute:

```python
aggregate_result = {
    'test_id': 'test_1',
    'algorithm': 'lookahead_adaptive',
    'n_runs': 20,
    
    # Central tendency
    'mean_time': 29.8,
    'std_time': 1.9,
    'median_time': 29.5,
    
    'mean_iterations': 71,
    'std_iterations': 5,
    
    'mean_final_likelihood': -2340.7,
    'std_final_likelihood': 0.3,
    
    'mean_memory_mb': 145,
    'std_memory_mb': 3,
    
    # Robustness
    'success_rate': 0.95,  # Fraction reaching near-optimum
    'convergence_rate': 1.0,  # Fraction that converged (vs timeout/crash)
    'monotonicity_violations': 0,  # Count of runs with ℓ decrease
    
    # Best/worst
    'best_likelihood': -2340.3,
    'worst_likelihood': -2341.2,
    'fastest_time': 26.1,
    'slowest_time': 33.4
}
```

### Comparison Tables

#### Table 1: Performance Summary (Test 1)

```
High-Dimensional GMM (K=50, d=2, n=5000)
═══════════════════════════════════════════════════════════════════════
Method                Time(s)     Iters    Final ℓ     Success%   Memory(MB)
───────────────────────────────────────────────────────────────────────
Standard EM          45.2±3.1    157±12   -2341.2±0.5    85%        120±5
Lookahead (γ=0.3)    38.1±2.8     98±8    -2340.8±0.4    90%        145±4
Lookahead (γ=0.5)    32.4±2.1     76±6    -2340.9±0.3    90%        145±4
Lookahead (γ=0.7)    28.9±2.3     68±7    -2341.0±0.4    85%        145±4
Lookahead (γ=0.9)    31.2±3.1     71±9    -2341.3±0.6    80%        145±4
Lookahead (Adaptive) 29.8±1.9★    71±5    -2340.7±0.3★   95%★       145±4
SQUAREM              35.7±3.4     82±9    -2341.1±0.4    85%        180±8
Quasi-Newton         41.2±5.6     89±15   -2341.5±0.7    75%       4800±200
═══════════════════════════════════════════════════════════════════════
★ Best in category
Winner: Lookahead (Adaptive) - 34% faster than standard EM
```

#### Table 2: Summary Across All Tests

```
Algorithm Performance Summary
═══════════════════════════════════════════════════════
Test                Standard   Lookahead   SQUAREM   Winner
                    EM        (Adaptive)
───────────────────────────────────────────────────────
1. High-D GMM       45.2s      29.8s★      35.7s     Lookahead
2. HMM              67.3s      54.1s★      62.8s     Lookahead
3. MoE              12.4s      11.2s       10.8s★    SQUAREM
4. Robustness       73%        81%★        75%       Lookahead
5. Real Data        23.1s      21.7s★      22.4s     Lookahead
───────────────────────────────────────────────────────
Overall Wins:       0/5        4/5         1/5
═══════════════════════════════════════════════════════
```

### Visualizations

#### Figure 1: Convergence Curves
```python
# For each test, plot ℓ(θ^(t)) vs iteration for all algorithms
# Show mean ± std band across random restarts
```

#### Figure 2: Time Distribution
```python
# Box plots showing distribution of convergence times
# One box per algorithm, across all tests
```

#### Figure 3: Success Rate by Initialization
```python
# For Test 4, bar chart showing success rate
# x-axis: initialization strategy
# grouped bars: different algorithms
```

#### Figure 4: Memory vs Dimension
```python
# Scalability test: vary K in {10, 20, 50, 100, 200}
# Plot memory usage vs K for each algorithm
```

#### Figure 5: Gamma Schedule Effect
```python
# For lookahead EM only
# Show how different γ schedules affect convergence
```

---

## Success Criteria

### Minimum for Publication

The empirical evaluation will be considered successful if:

1. **At least 2/5 tests show clear advantage** (30%+ improvement in primary metric)
2. **Hypothesis validated:** Advantage occurs in structured H_Q scenarios (Tests 1-3)
3. **No catastrophic failures:** Algorithm doesn't crash or diverge more than competitors
4. **Honest reporting:** All results (including failures) documented

### Ideal Outcome

Strong empirical support would include:

1. **Win on 3-4 out of 5 tests**
2. **Clear pattern:** Wins correlate with H_Q structure (sparse, block-diagonal)
3. **Adaptive γ provides benefit** over fixed γ in at least 2 tests
4. **Scalability advantage:** Better memory usage for high-dimensional problems
5. **Practical utility:** Competitive or better on real data (Test 5)

### Acceptable Partial Success

Even if lookahead EM doesn't dominate, results are publishable if:

1. **Wins in specific scenarios** (e.g., only Test 1)
2. **Clear understanding of when/why it works**
3. **Provides new insights** (e.g., H_Q structure matters)
4. **Honest negative results** (shows what doesn't work)

### What Would Be Failure

The evaluation would fail to support publication if:

1. **Loses on all 5 tests** (no advantage anywhere)
2. **Crashes or diverges frequently** (unstable)
3. **No pattern to results** (wins/losses seem random)
4. **Much slower than alternatives** with no quality benefit

---

## Computational Requirements

### Hardware

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB

**Recommended:**
- CPU: 8+ cores (for parallel random restarts)
- RAM: 16 GB
- Storage: 50 GB (for storing all runs)

### Software

**Python packages:**
```
numpy >= 1.21
scipy >= 1.7
scikit-learn >= 1.0
matplotlib >= 3.4
pandas >= 1.3
psutil >= 5.8
```

**R packages (for SQUAREM):**
```
turboEM
```

**Python-R bridge:**
```
rpy2 >= 3.4
```

### Estimated Runtime

**Per test:**
- Test 1: ~8 hours (20 runs × 6 algorithms × 20 min/run)
- Test 2: ~6 hours
- Test 3: ~4 hours
- Test 4: ~12 hours (110 runs)
- Test 5: ~2 hours

**Total: ~32 hours** (can run overnight for 2-3 nights)

**With parallelization (8 cores):**
- Effective runtime: ~8-12 hours total

---

## Timeline

### Week 1: Implementation

**Days 1-2: Core algorithms**
- Implement `lookahead_em.py` with adaptive γ
- Implement `standard_em.py` baseline
- Test on toy GMM (K=3)

**Days 3-4: Models**
- Implement `gmm.py`, `hmm.py`, `moe.py`
- Verify Q-function and H_Q computation
- Unit tests for each model

**Days 5-7: Integration**
- Implement SQUAREM wrapper
- Implement metrics and timing utilities
- Test full pipeline on small examples

### Week 2: Tests 1-3 (Synthetic Data)

**Days 8-10: Test 1 (High-D GMM)**
- Generate data
- Run all algorithms (20 restarts each)
- Analyze results
- Create visualizations

**Days 11-12: Test 2 (HMM)**
- Implement HMM-specific code
- Run experiments
- Analyze results

**Days 13-14: Test 3 (MoE)**
- Implement MoE-specific code
- Run experiments
- Analyze results

### Week 3: Tests 4-5 (Robustness + Real Data)

**Days 15-17: Test 4 (Robustness)**
- Implement 4 initialization strategies
- Run 110 runs per algorithm
- Analyze success rates
- Convergence distribution plots

**Days 18-19: Test 5 (Real Data)**
- Load and preprocess Diggle & Kenward data
- Run algorithms
- Interpret results scientifically

**Days 20-21: Buffer**
- Debug any issues
- Re-run failed experiments
- Additional analyses if needed

### Week 4: Analysis and Write-Up

**Days 22-24: Comprehensive Analysis**
- Aggregate all results
- Create comparison tables
- Generate all figures
- Statistical significance tests

**Days 25-27: Documentation**
- Write results section
- Document when/why algorithm works
- Practical recommendations
- Limitations and future work

**Day 28: Final Review**
- Check all claims are supported
- Verify reproducibility
- Prepare supplementary materials

---

## Statistical Analysis

### Hypothesis Tests

For each pairwise comparison (e.g., Lookahead vs Standard EM on Test 1):

**Paired t-test on convergence time:**
```python
from scipy import stats

# 20 runs, same initializations for both algorithms
times_lookahead = [...]  # 20 values
times_standard = [...]   # 20 values

t_stat, p_value = stats.ttest_rel(times_lookahead, times_standard)

if p_value < 0.05:
    print(f"Significant difference (p={p_value:.4f})")
```

**Wilcoxon signed-rank test (non-parametric alternative):**
```python
stat, p_value = stats.wilcoxon(times_lookahead, times_standard)
```

**Effect size (Cohen's d):**
```python
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
    )

effect_size = cohens_d(times_lookahead, times_standard)
# |d| > 0.8 = large effect
# |d| > 0.5 = medium effect
# |d| > 0.2 = small effect
```

### Confidence Intervals

Report 95% confidence intervals for all mean metrics:

```python
def ci_95(data):
    """Bootstrap 95% CI."""
    from scipy.stats import bootstrap
    res = bootstrap((data,), np.mean, n_resamples=10000)
    return res.confidence_interval
```

### Multiple Testing Correction

When making multiple comparisons, use Bonferroni correction:

```python
# 5 tests × 4 algorithm pairs = 20 comparisons
alpha_corrected = 0.05 / 20  # = 0.0025
```

---

## Reproducibility Checklist

### Code

- [ ] All random seeds set explicitly
- [ ] Requirements.txt with exact versions
- [ ] Environment.yml for conda environment
- [ ] README with setup instructions
- [ ] Docstrings for all functions
- [ ] Type hints where appropriate

### Data

- [ ] Data generation code provided
- [ ] True parameters saved for each test
- [ ] Random seeds recorded
- [ ] Raw data files archived

### Results

- [ ] All raw results saved (JSON or pickle)
- [ ] Aggregation scripts provided
- [ ] Plotting scripts provided
- [ ] Jupyter notebooks for analysis

### Documentation

- [ ] This specification document
- [ ] Algorithm implementation notes
- [ ] Known issues and limitations
- [ ] Instructions for reproducing each test

---

## Appendix: Implementation Notes

### Computing H_Q for GMM

For a Gaussian mixture, the Q-function is:
```
Q(θ|θ_old) = Σ_i Σ_k γ_ik * [log π_k + log N(x_i; μ_k, σ_k²)]
```

The Hessian with respect to means (μ) is **block-diagonal**:
```python
def hessian_Q_gmm(X, responsibilities, theta):
    """
    Compute Hessian of Q w.r.t. means.
    
    Returns:
        H: Block-diagonal matrix (K blocks of d×d)
    """
    n, d = X.shape
    K = len(theta['pi'])
    
    H_blocks = []
    for k in range(K):
        # H_k = -Σ_i γ_ik * (1/σ_k²) * I
        # where I is d×d identity
        H_k = -np.sum(responsibilities[:, k]) / (theta['sigma'][k]**2) * np.eye(d)
        H_blocks.append(H_k)
    
    # Block-diagonal matrix
    H = scipy.linalg.block_diag(*H_blocks)
    return H
```

**Inversion:** For block-diagonal H, invert each block separately:
```python
def inv_block_diagonal(H_blocks):
    """Efficiently invert block-diagonal matrix."""
    inv_blocks = [np.linalg.inv(block) for block in H_blocks]
    return scipy.linalg.block_diag(*inv_blocks)
```

**Complexity:**
- Direct inversion: O((Kd)³) = O(K³d³)
- Block-wise: O(K × d³) = O(Kd³)
- For K=50, d=2: ~400× speedup!

### Computing V(θ) Efficiently

```python
def estimate_V(X, theta, model):
    """
    Estimate lookahead value V(θ) = max_{θ'} Q(θ'|θ).
    
    Uses second-order approximation:
    V(θ) ≈ Q(θ|θ) + 0.5 * ∇Q^T H_Q^(-1) ∇Q
    """
    # 1. Compute Q(θ|θ) - trivial (complete-data likelihood at self)
    resp = model.e_step(X, theta)
    Q_self = model.compute_Q(theta, theta, X, resp)
    
    # 2. Compute gradient ∇Q(θ|θ)
    grad_Q = model.compute_Q_gradient(theta, theta, X, resp)
    
    # 3. Compute Hessian H_Q(θ|θ)
    H_Q = model.compute_Q_hessian(theta, theta, X, resp)
    
    # 4. Solve H_Q * step = -grad_Q
    # (Newton step from θ toward maximum of Q(·|θ))
    step = np.linalg.solve(H_Q, -grad_Q)
    
    # 5. Approximate max value
    V = Q_self + 0.5 * grad_Q @ step
    
    return V
```

### Enforcing Constraints

GMM parameters must satisfy:
- σ_k > 0 (positive variances)
- π_k > 0, Σπ_k = 1 (valid probabilities)

**During optimization:**
```python
def project_to_feasible(theta):
    """Project parameters to constraint set."""
    # Clip sigma
    theta['sigma'] = np.maximum(theta['sigma'], 1e-6)
    
    # Project pi to simplex
    theta['pi'] = np.maximum(theta['pi'], 1e-6)
    theta['pi'] = theta['pi'] / theta['pi'].sum()
    
    return theta
```

**In lookahead optimization:**
```python
from scipy.optimize import minimize

def objective(theta_flat):
    theta = unflatten(theta_flat)
    theta = project_to_feasible(theta)
    return -(Q(theta) + gamma * V(theta))

result = minimize(objective, theta_init_flat, method='L-BFGS-B')
theta_next = project_to_feasible(unflatten(result.x))
```

---

## Version History

**v1.0** (December 2025)
- Initial specification
- 5 tests defined
- Implementation structure outlined
- Success criteria established

---

## Contact

For questions or clarifications about this specification:
- Author: MaryLena Bleile
- Purpose: Empirical validation of lookahead EM algorithm
- Context: Chapter 6 of dissertation / book manuscript

---

**END OF SPECIFICATION**
