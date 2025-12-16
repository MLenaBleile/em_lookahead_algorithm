# Implementation Prompt Suite: Lookahead EM Empirical Evaluation

**Purpose:** Step-by-step prompts for implementing the empirical evaluation described in `EMPIRICAL_TEST_SPECIFICATIONS.md`

**How to Use:**
1. Follow prompts in order (they build on each other)
2. Each prompt is self-contained - copy/paste to your LLM
3. Verify each step works before moving to the next
4. Mark completed prompts with ✓

**Timeline:** ~4 weeks, following the specification

---

## Table of Contents

**Phase 1: Setup & Infrastructure (Prompts 1-5)**
**Phase 2: Core Algorithms (Prompts 6-10)**
**Phase 3: Model Implementations (Prompts 11-15)**
**Phase 4: Test Implementation (Prompts 16-25)**
**Phase 5: Analysis & Visualization (Prompts 26-30)**

---

# Phase 1: Setup & Infrastructure

## Prompt 1: Project Setup

```
I need to set up a Python project for empirical evaluation of EM algorithms.

Please create:

1. Project directory structure following this layout:
   - lookahead_em_evaluation/
   - algorithms/, models/, data/, tests/, utils/, results/, notebooks/
   - See EMPIRICAL_TEST_SPECIFICATIONS.md section "Implementation Structure"

2. Create requirements.txt with:
   - numpy >= 1.21
   - scipy >= 1.7
   - scikit-learn >= 1.0
   - matplotlib >= 3.4
   - pandas >= 1.3
   - psutil >= 5.8
   - rpy2 >= 3.4

3. Create environment.yml for conda

4. Create README.md with:
   - Project description
   - Setup instructions
   - Quick start guide

5. Create .gitignore for Python projects

6. Add __init__.py files to all Python package directories

Provide all the file contents.
```

**Success Criteria:**
- [ ] All directories created
- [ ] Can run `pip install -r requirements.txt` successfully
- [ ] All __init__.py files in place

---

## Prompt 2: Utilities - Timing and Monitoring

```
Implement the resource monitoring utilities described in EMPIRICAL_TEST_SPECIFICATIONS.md.

Create utils/timing.py with:

1. ResourceMonitor class that:
   - Tracks elapsed time
   - Monitors peak memory usage in background thread
   - Has start() and stop() methods
   - Returns dict with 'elapsed_time' and 'peak_memory_mb'

2. Simple timing decorator: @timed

Example usage:
```python
monitor = ResourceMonitor()
monitor.start()
# ... run algorithm ...
stats = monitor.stop()
print(f"Time: {stats['elapsed_time']:.2f}s, Memory: {stats['peak_memory_mb']:.1f}MB")
```

Include comprehensive docstrings and a test at the bottom.
```

**Success Criteria:**
- [ ] Can monitor a dummy function
- [ ] Memory tracking works (peak > current)
- [ ] No thread leaks (thread stops properly)

---

## Prompt 3: Utilities - Metrics

```
Implement the metrics utilities described in EMPIRICAL_TEST_SPECIFICATIONS.md.

Create utils/metrics.py with these functions:

1. compute_metrics(theta_history, likelihood_history, time_history, theta_true=None)
   - Returns dict with: iterations, time, final_likelihood, convergence_rate
   - If theta_true provided: add parameter_error, success (boolean)

2. check_monotonicity(likelihood_history, tol=1e-8)
   - Returns: (is_monotonic: bool, violations: list of indices)

3. parameter_error(theta_estimated, theta_true)
   - Returns: Frobenius norm of difference (for dict of arrays)

4. convergence_rate(likelihood_history)
   - Returns: Average log-improvement per iteration

5. cohens_d(x, y)
   - Returns: Effect size for comparing two samples

6. bootstrap_ci(data, statistic=np.mean, n_resamples=10000, confidence=0.95)
   - Returns: (lower, upper) confidence interval

Include docstrings and examples for each function.
```

**Success Criteria:**
- [ ] All functions have unit tests
- [ ] check_monotonicity correctly identifies violations
- [ ] CI function returns reasonable intervals

---

## Prompt 4: Utilities - Logging

```
Create utils/logging_utils.py for experiment tracking.

Implement:

1. ExperimentLogger class that:
   - Saves run results to JSON
   - Organizes by test_id and algorithm
   - Creates results/{test_id}/ directories
   - Handles appending to existing logs
   
2. Methods:
   - log_run(run_result: dict) -> saves to file
   - load_results(test_id, algorithm) -> loads all runs
   - summary_stats(test_id) -> aggregate across all algorithms
   
3. Result format (see EMPIRICAL_TEST_SPECIFICATIONS.md "Per-Run Metrics")

Example:
```python
logger = ExperimentLogger(base_dir='results')
logger.log_run({
    'test_id': 'test_1',
    'algorithm': 'standard_em',
    'seed': 42,
    'final_likelihood': -2341.2,
    # ... more fields
})
```

Include automatic timestamping and unique run IDs.
```

**Success Criteria:**
- [ ] Can save and load results
- [ ] Directory structure auto-created
- [ ] JSON files are human-readable
- [ ] No data loss on multiple writes

---

## Prompt 5: Utilities - Plotting

```
Create utils/plotting.py with visualization functions.

Implement:

1. plot_convergence(likelihood_histories: dict, title, save_path=None)
   - Input: {'algorithm_name': [ℓ values], ...}
   - Plots ℓ vs iteration for each algorithm
   - Shows mean ± std band if multiple runs provided
   
2. plot_time_comparison(results_dict: dict, save_path=None)
   - Box plots of convergence times
   - x-axis: algorithms, y-axis: time (seconds)
   
3. plot_success_rates(results_dict: dict, save_path=None)
   - Bar chart of success rates by algorithm
   
4. plot_memory_usage(results_dict: dict, save_path=None)
   - Memory comparison across algorithms

5. plot_gamma_schedule(gamma_history, likelihood_history, save_path=None)
   - Two subplots: γ vs iteration, ℓ vs iteration
   - For analyzing adaptive γ effect

Use consistent style (seaborn or matplotlib with nice defaults).
Save figures as high-res PNG (dpi=300).
```

**Success Criteria:**
- [ ] All plots render correctly
- [ ] Saves to file work
- [ ] Plots are publication-quality
- [ ] Legend and labels clear

---

# Phase 2: Core Algorithms

## Prompt 6: Standard EM Implementation

```
Implement the standard EM algorithm baseline.

Create algorithms/standard_em.py with:

1. StandardEM class:
   - __init__(model): Takes a model object (GMM, HMM, etc.)
   - fit(X, theta_init, max_iter=1000, tol=1e-6, timeout=600)
   - Returns: (theta_final, diagnostics)
   
2. Diagnostics dict should include:
   - 'converged': bool
   - 'iterations': int
   - 'stopping_reason': str
   - 'theta_history': list of theta dicts
   - 'likelihood_history': list of floats
   - 'time_history': list of floats (cumulative)
   - 'time_seconds': float (total)
   - 'peak_memory_mb': float

3. Stopping conditions (see spec):
   - Convergence: ||θ^(t+1) - θ^(t)||_∞ < tol
   - Max iterations reached
   - Timeout exceeded
   - Stagnation: |ℓ^(t+1) - ℓ^(t)| < 1e-8 for 5 iterations

4. Use ResourceMonitor for timing/memory

Include verbose mode that prints progress every 10 iterations.

The model object should have methods:
- model.e_step(X, theta) -> responsibilities
- model.m_step(X, responsibilities) -> theta_new
- model.log_likelihood(X, theta) -> float
```

**Success Criteria:**
- [ ] Runs on toy GMM (K=2, n=100)
- [ ] Monotonically increases likelihood
- [ ] Stops at convergence
- [ ] Returns proper diagnostics

---

## Prompt 7: Lookahead EM - Gamma Schedules

```
Implement gamma scheduling for lookahead EM.

Create algorithms/gamma_schedule.py with:

1. GammaSchedule abstract base class

2. Implementations:
   - FixedGamma(gamma): Always returns gamma
   - AdaptiveGamma(gamma_init, gamma_max, gamma_step)
     * γ_t = min(gamma_max, gamma_init + gamma_step × t)
   - ExponentialGamma(gamma_init, gamma_max, rate)
     * γ_t = gamma_max - (gamma_max - gamma_init) × exp(-rate × t)

3. Interface:
   - schedule.get_gamma(iteration: int) -> float
   - schedule.reset()  # For new run

Example:
```python
schedule = AdaptiveGamma(gamma_init=0.1, gamma_max=0.9, gamma_step=0.05)
for t in range(100):
    gamma = schedule.get_gamma(t)
    # use gamma...
```

Include tests showing that schedules behave correctly.
```

**Success Criteria:**
- [ ] FixedGamma returns constant
- [ ] AdaptiveGamma increases linearly then plateaus
- [ ] Can plot schedule over 100 iterations

---

## Prompt 8: Lookahead EM - V(θ) Estimation

```
Implement the lookahead value function estimation.

Create algorithms/value_estimation.py with:

1. estimate_V_second_order(model, X, theta, responsibilities=None)
   - Computes V(θ) ≈ Q(θ|θ) + 0.5 × ∇Q^T H_Q^(-1) ∇Q
   - Uses model.compute_Q_gradient() and model.compute_Q_hessian()
   - Returns: (V_value, diagnostics_dict)
   
2. Diagnostics should include:
   - 'Q_self': Q(θ|θ)
   - 'grad_norm': ||∇Q||
   - 'newton_step_norm': ||H^(-1)∇Q||
   - 'hessian_min_eigenvalue': min(eig(H))
   - 'hessian_condition_number': cond(H)

3. Handle numerical issues:
   - If H not positive definite, regularize: H + λI
   - If gradient very small, return Q_self only
   - If inversion fails, catch and return Q_self

Example usage:
```python
V, diag = estimate_V_second_order(model, X, theta)
print(f"V(θ) = {V:.4f}, Q(θ|θ) = {diag['Q_self']:.4f}")
```

The model object must provide:
- model.compute_Q(theta, theta_old, X, responsibilities) -> float
- model.compute_Q_gradient(theta, theta_old, X, responsibilities) -> array
- model.compute_Q_hessian(theta, theta_old, X, responsibilities) -> matrix
```

**Success Criteria:**
- [ ] V ≥ Q for quadratic test case
- [ ] Handles singular Hessian gracefully
- [ ] Diagnostics reveal numerical issues

---

## Prompt 9: Lookahead EM - Main Algorithm

```
Implement the main lookahead EM algorithm.

Create algorithms/lookahead_em.py with:

1. LookaheadEM class following the structure in EMPIRICAL_TEST_SPECIFICATIONS.md

2. Key components:
   - __init__(model, gamma, n_candidates=20, **kwargs)
   - fit(X, theta_init, max_iter, tol, timeout)
   - _lookahead_step(X, theta_current, gamma)

3. The lookahead step should:
   a) Compute responsibilities at current theta
   b) Generate n_candidates parameter candidates (sample near current)
   c) For each candidate θ:
      - Compute Q(θ|θ_current)
      - Compute V(θ) using estimate_V_second_order
      - Score = Q + gamma × V
   d) Select best candidate
   e) Apply constraints (project to feasible set)
   
4. Candidate generation:
   - Sample in log/logit space for positive/simplex constraints
   - Always include current theta (for monotonicity)
   - Sampling width: adaptive based on current parameter scale

5. Return same diagnostics format as StandardEM, plus:
   - 'gamma_history': list of gamma values used
   - 'v_value_history': list of V(θ) estimates
   - 'n_candidates_evaluated': int

If gamma=0, should exactly match StandardEM (use closed-form M-step).

Include detailed docstrings and usage example.
```

**Success Criteria:**
- [ ] gamma=0 matches StandardEM output
- [ ] gamma>0 uses V(θ) in selection
- [ ] Monotonically increases likelihood
- [ ] Respects constraints (σ>0, π simplex)

---

## Prompt 10: SQUAREM Wrapper

```
Create a Python wrapper for R's turboEM package.

Create algorithms/squarem_wrapper.py:

1. Check if rpy2 and R's turboEM are available
   - Graceful fallback if not installed
   
2. SQUAREMWrapper class:
   - fit(X, theta_init, max_iter, tol, model)
   - Converts Python model to R-compatible format
   - Calls turboEM::turboem with method='squarem'
   - Converts results back to Python
   
3. Handle conversions:
   - numpy arrays ↔ R vectors/matrices
   - Python dicts ↔ R lists
   - Python functions ↔ R functions

4. For GMM specifically, implement R-side functions:
   - fixptfn = one EM step
   - objfn = log-likelihood

Example:
```python
try:
    squarem = SQUAREMWrapper()
    theta_final, diagnostics = squarem.fit(X, theta_init, max_iter=1000, model=gmm)
except ImportError:
    print("SQUAREM not available, skipping")
```

If rpy2/turboEM not available, document installation instructions.
```

**Success Criteria:**
- [ ] Runs on toy GMM if rpy2 installed
- [ ] Graceful error if not installed
- [ ] Returns standardized diagnostics format
- [ ] Handles R errors appropriately

---

# Phase 3: Model Implementations

## Prompt 11: GMM - Core Implementation

```
Implement Gaussian Mixture Model with all required methods.

Create models/gmm.py:

1. GaussianMixtureModel class with:
   - __init__(n_components, n_features)
   - Implements all methods needed by EM algorithms
   
2. Required methods:
   a) e_step(X, theta) -> responsibilities (n × K)
   b) m_step(X, responsibilities) -> theta
   c) log_likelihood(X, theta) -> float
   d) compute_Q(theta, theta_old, X, responsibilities) -> float
   e) compute_Q_gradient(theta, theta_old, X, responsibilities) -> array
   f) compute_Q_hessian(theta, theta_old, X, responsibilities) -> matrix

3. Theta format (dict):
   - 'mu': array of shape (K, d)
   - 'sigma': array of shape (K, d) [diagonal covariances]
   - 'pi': array of shape (K,) [mixing weights]

4. For compute_Q_hessian:
   - Exploit block-diagonal structure (see spec appendix)
   - Return sparse or block format for efficiency
   - Provide inv_hessian() method that inverts block-wise

5. Include:
   - Numerical stability (log-space, check for zeros)
   - Input validation
   - Constraint projection: project_to_feasible(theta)

6. Helper functions:
   - initialize_kmeans(X, n_components, random_state)
   - initialize_random(X, n_components, random_state)

Include comprehensive docstrings and examples.
```

**Success Criteria:**
- [ ] Runs standard EM successfully
- [ ] Hessian is block-diagonal
- [ ] Gradient matches finite differences
- [ ] Handles K=2, d=2 correctly

---

## Prompt 12: GMM - Data Generation

```
Implement GMM data generation for testing.

Create data/generate_gmm.py:

1. generate_gmm_data(n, K, d, separation, sigma, seed):
   - n: number of samples
   - K: number of components  
   - d: dimensions
   - separation: how far apart means are
   - sigma: component std devs
   - Returns: (X, z, theta_true)
     * X: (n, d) array of observations
     * z: (n,) array of true component assignments
     * theta_true: dict with true parameters

2. Presets for tests:
   - test_1_high_d(): K=50, d=2, n=5000 (Test 1 from spec)
   - test_4_simple(): K=5, d=2, n=500 (Test 4 from spec)
   
3. generate_initializations(X, K, strategy, n_inits, random_state):
   - strategy: 'kmeans', 'random', 'collapsed', 'far_away'
   - Returns: list of n_inits theta_init dicts

For "collapsed" initialization, all means at centroid.
For "far_away", all means at distance 10 from data centroid.

Include visualization function to plot 2D GMM data with true clusters.
```

**Success Criteria:**
- [ ] Can generate Test 1 data (K=50)
- [ ] Can generate Test 4 data (K=5)
- [ ] All initialization strategies work
- [ ] Visualization shows clear clusters

---

## Prompt 13: HMM Implementation

```
Implement Hidden Markov Model.

Create models/hmm.py:

1. HiddenMarkovModel class with:
   - __init__(n_states, n_observations)
   - All methods needed by EM algorithms
   
2. Theta format (dict):
   - 'A': transition matrix (S × S)
   - 'B': emission matrix (S × V)
   - 'pi_0': initial state distribution (S,)

3. Required methods:
   a) e_step(X, theta) -> responsibilities
      - Use forward-backward algorithm
      - Returns: (gamma, xi) where
        * gamma[t, s] = P(z_t = s | x)
        * xi[t, s, s'] = P(z_t = s, z_{t+1} = s' | x)
   
   b) m_step(X, responsibilities) -> theta
      - Closed-form updates from Baum-Welch
   
   c) log_likelihood(X, theta) -> float
      - Use forward algorithm
   
   d) compute_Q, compute_Q_gradient, compute_Q_hessian
      - Exploit structure (transitions separate from emissions)

4. For sparse transition matrices:
   - Store as sparse matrix if >50% zeros
   - Efficient Hessian for sparse A

Include forward/backward helper functions.
```

**Success Criteria:**
- [ ] Forward-backward numerically stable
- [ ] Baum-Welch EM converges on toy example
- [ ] Handles S=20, V=50, n=1000
- [ ] Respects stochastic constraints

---

## Prompt 14: HMM Data Generation

```
Implement HMM data generation.

Create data/generate_hmm.py:

1. generate_hmm_data(n_steps, n_states, n_obs, structure, seed):
   - structure: 'nearly_diagonal', 'cyclic', 'random'
   - Returns: (observations, states, theta_true)
   
2. test_2_hmm(): Generate data for Test 2 (S=20, V=50, n=1000)
   - Nearly diagonal transition (see spec)
   - Each state prefers 2-3 observations
   
3. generate_hmm_initializations(observations, S, V, strategy, n_inits, seed):
   - strategy: 'uniform', 'random', 'perturbed'
   - Returns: list of theta_init dicts

Include visualization:
- plot_transition_matrix(A): Heatmap of transitions
- plot_emission_matrix(B): Heatmap of emissions
- plot_sequence(observations[:100]): First 100 observations
```

**Success Criteria:**
- [ ] Generated HMM data looks reasonable
- [ ] Transition matrix is sparse (nearly diagonal)
- [ ] Can generate Test 2 data exactly
- [ ] Visualizations are informative

---

## Prompt 15: Mixture of Experts

```
Implement Mixture of Experts model.

Create models/mixture_of_experts.py:

1. MixtureOfExperts class:
   - __init__(n_experts, n_features)
   - Experts are linear regressors
   - Gating network is softmax classifier
   
2. Theta format:
   - 'gamma': gating weights (G × (d+1)) [includes intercept]
   - 'beta': expert weights (G × (d+1))
   - 'sigma': noise std (scalar or G-array)

3. Required EM methods:
   - e_step: Compute expert responsibilities given (x, y)
   - m_step: Update gates and experts
   - log_likelihood: P(y|x, theta)
   
4. For compute_Q_hessian:
   - Block-diagonal: [Gate block, Expert 1, ..., Expert G]
   - Each expert block is independent
   
5. Generate data function in data/generate_moe.py:
   - test_3_moe(): G=5, d=2, n=2000 (from spec)

Include example fitting MoE to synthetic data.
```

**Success Criteria:**
- [ ] Fits MoE on toy regression problem
- [ ] Hessian is block-diagonal
- [ ] Experts learn different regions
- [ ] Gating network assigns appropriately

---

# Phase 4: Test Implementation

## Prompt 16: Test Runner Infrastructure

```
Create a unified test runner.

Create tests/test_runner.py:

1. TestRunner class that:
   - Loads test configuration from spec
   - Runs all algorithms on one test
   - Handles multiple random restarts in parallel
   - Saves results via ExperimentLogger
   - Catches and logs exceptions
   
2. Configuration format (from spec):
```python
test_config = {
    'test_id': 'test_1',
    'data_generator': generate_gmm_data_test1,
    'initialization': 'kmeans_suboptimal',
    'n_restarts': 20,
    'algorithms': ['standard_em', 'lookahead_adaptive', 'squarem'],
    'max_iter': 1000,
    'timeout': 600,
    'tol': 1e-6
}
```

3. run_single_trial(test_config, algorithm, seed):
   - Generates data with seed
   - Initializes algorithm
   - Runs algorithm.fit()
   - Returns run_result dict (see spec "Per-Run Metrics")
   
4. run_test(test_config, parallel=True, n_jobs=4):
   - Runs all algorithms × all seeds
   - Uses joblib for parallelization
   - Progress bar (tqdm)
   - Returns list of all run_results

Exception handling: If an algorithm crashes, log error and continue.
```

**Success Criteria:**
- [ ] Can run toy test (K=2, 5 restarts)
- [ ] Parallel execution works
- [ ] All results saved correctly
- [ ] Handles crashes gracefully

---

## Prompt 17: Test 1 - High-Dimensional GMM

```
Implement Test 1 from EMPIRICAL_TEST_SPECIFICATIONS.md.

Create tests/test_1_high_d_gmm.py:

1. Load configuration from spec (see "Test 1: High-Dimensional Sparse GMM")

2. Define all algorithms to test:
   - StandardEM()
   - LookaheadEM(gamma=0.3)
   - LookaheadEM(gamma=0.5)
   - LookaheadEM(gamma=0.7)
   - LookaheadEM(gamma=0.9)
   - LookaheadEM(gamma='adaptive')
   - SQUAREMWrapper() [if available]
   
3. Data generation: K=50, d=2, n=5000 (exactly as in spec)

4. Initialization: K-means with max_iter=10 (deliberately suboptimal)

5. Run 20 random restarts for each algorithm

6. Save results to results/test_1/

7. Main execution:
```python
if __name__ == '__main__':
    config = load_test_1_config()
    results = run_test(config, parallel=True, n_jobs=8)
    print(f"Completed {len(results)} runs")
    # Save summary statistics
```

Expected runtime: ~8 hours on 8 cores.
```

**Success Criteria:**
- [ ] Runs without errors
- [ ] All 20 restarts complete per algorithm
- [ ] Results saved in proper format
- [ ] Can load and inspect results

---

## Prompt 18: Test 1 - Analysis Script

```
Create analysis script for Test 1 results.

Create tests/analyze_test_1.py:

1. Load all results from results/test_1/

2. Compute aggregate statistics (see spec "Aggregate Metrics"):
   - Mean/std time, iterations, likelihood
   - Success rate (reaching ℓ > best - 0.5)
   - Convergence rate
   - Memory usage
   
3. Statistical tests:
   - Paired t-tests comparing each algorithm to StandardEM
   - Bonferroni correction for multiple comparisons
   - Effect sizes (Cohen's d)
   
4. Generate Table 1 from spec:
   - ASCII table format for terminal
   - LaTeX table format for paper
   - CSV for further analysis
   
5. Generate visualizations:
   - Convergence curves (mean ± std)
   - Time comparison (box plots)
   - Success rate (bar chart)
   - Memory usage (bar chart)
   
6. Save figures to results/test_1/figures/

Output summary statistics to terminal and save to results/test_1/summary.txt
```

**Success Criteria:**
- [ ] Correctly computes all statistics
- [ ] Identifies best performing algorithm
- [ ] Generates publication-quality figures
- [ ] LaTeX table ready for paper

---

## Prompt 19: Test 2 - HMM Implementation

```
Implement Test 2 from EMPIRICAL_TEST_SPECIFICATIONS.md.

Create tests/test_2_hmm.py following the same structure as Test 1:

1. Configuration from spec "Test 2: Hidden Markov Model"
   - S=20 states, V=50 observations, n=1000 steps
   - Nearly diagonal transition matrix (0.7 self-transition)
   
2. Algorithms:
   - StandardEM()
   - LookaheadEM(gamma='adaptive')
   - SQUAREMWrapper()
   
3. Initialization: Uniform transition, random emission

4. Run 20 random restarts

5. Track additional metrics:
   - Transition sparsity (% > 0.01)
   - Constraint violations (any non-stochastic matrices?)

Expected runtime: ~6 hours on 8 cores.

Include test_2_hmm.py and analyze_test_2.py (similar to Test 1).
```

**Success Criteria:**
- [ ] HMM EM converges correctly
- [ ] Sparse transition matrix preserved
- [ ] All algorithms respect constraints
- [ ] Results comparable to Test 1 format

---

## Prompt 20: Test 3 - Mixture of Experts

```
Implement Test 3 from EMPIRICAL_TEST_SPECIFICATIONS.md.

Create tests/test_3_moe.py:

1. Configuration: G=5 experts, d=2 features, n=2000

2. Algorithms:
   - StandardEM()
   - LookaheadEM(gamma=0.2, 0.4, 0.6, 0.8)
   - LookaheadEM(gamma='adaptive')
   - SQUAREMWrapper()

3. Initialization: Equal gates, random expert coefficients

4. Run 30 random restarts

5. Track expert utilization:
   - Count experts with π_g > 0.05
   - Check if all experts used

Expected runtime: ~4 hours on 8 cores.

Create corresponding analyze_test_3.py.
```

**Success Criteria:**
- [ ] MoE EM converges
- [ ] Experts learn different regions
- [ ] Block-diagonal Hessian exploited
- [ ] Analysis script works

---

## Prompt 21: Test 4 - Robustness Study

```
Implement Test 4 from EMPIRICAL_TEST_SPECIFICATIONS.md.

Create tests/test_4_robustness.py:

1. Simple GMM: K=5, d=2, n=500, well-separated

2. Four initialization strategies (see spec):
   - Strategy 1: All collapsed (20 runs)
   - Strategy 2: Far away (20 runs)
   - Strategy 3: Wrong K (20 runs)
   - Strategy 4: Random (50 runs)
   - Total: 110 runs per algorithm

3. Algorithms:
   - StandardEM (γ=0)
   - LookaheadEM(γ=0.2, 0.5, 0.8)
   - LookaheadEM(gamma='adaptive')
   - SQUAREMWrapper()

4. Success criterion: ℓ(θ) > ℓ_true - 5

5. Main metric: Success rate by initialization strategy

Create analyze_test_4.py that:
- Computes success rate per strategy
- Creates grouped bar chart
- Tests if adaptive γ helps with bad initializations

Expected runtime: ~12 hours on 8 cores.
```

**Success Criteria:**
- [ ] All 110 runs complete per algorithm
- [ ] Success rates vary by initialization
- [ ] Can identify which methods are more robust
- [ ] Clear visualization of results

---

## Prompt 22: Test 4 - Convergence Distribution

```
Create additional analysis for Test 4: convergence distribution.

Add to analyze_test_4.py:

1. For each initialization strategy, plot histogram of final likelihoods
   - One subplot per strategy
   - Overlay distributions for all algorithms
   - Mark true optimum with vertical line
   
2. Identify modes in distribution:
   - How many local optima exist?
   - Which algorithms find which optima?
   
3. Convergence trajectory analysis:
   - For runs that failed, plot their trajectories
   - Compare to successful runs
   - Identify where they got stuck

4. Create "success heatmap":
   - Rows: initialization strategies
   - Columns: algorithms
   - Color: success rate
   
This helps understand: Do certain methods escape bad initializations?
```

**Success Criteria:**
- [ ] Histograms show clear modes
- [ ] Can identify local vs global optima
- [ ] Heatmap reveals patterns
- [ ] Failed trajectories are informative

---

## Prompt 23: Test 5 - Real Data Preparation

```
Prepare Test 5 for real data analysis.

Create tests/test_5_real_data.py:

1. Load Diggle & Kenward data (or substitute real dataset)
   - If not available, use UCI dataset with missingness
   - Suggested: Breast Cancer Wisconsin, with induced dropout
   
2. Preprocessing:
   - Handle missing values (for EM, this is latent data)
   - Standardize features
   - Split into train/test if needed
   
3. Model selection:
   - Try K ∈ {2, 3, 4, 5} components
   - Use BIC to select K
   
4. Algorithms:
   - StandardEM()
   - LookaheadEM(γ=0.5)
   - LookaheadEM(gamma='adaptive')
   - Your original paper's method (if applicable)

5. Run 10 random restarts per K

6. Metrics:
   - Log-likelihood
   - BIC
   - Clustering stability (adjusted Rand index across restarts)
   - Time to convergence

Create analyze_test_5.py that includes scientific interpretation.
```

**Success Criteria:**
- [ ] Data loads correctly
- [ ] BIC selects reasonable K
- [ ] Results are stable across restarts
- [ ] Scientific interpretation makes sense

---

## Prompt 24: Test 5 - Scientific Validation

```
Add scientific validation to Test 5 analysis.

Extend analyze_test_5.py:

1. Cluster interpretation:
   - What distinguishes each cluster?
   - Are clusters scientifically meaningful?
   - Compare to known groups (if labels available)
   
2. Visualization:
   - PCA plot colored by cluster
   - Feature importance per cluster
   - Cluster sizes
   
3. Stability analysis:
   - Do different initializations find same clustering?
   - Adjusted Rand Index matrix across runs
   - Consensus clustering
   
4. Comparison to literature:
   - If using Diggle & Kenward: Compare to published results
   - Are results consistent?
   
5. Clinical/scientific relevance:
   - Do results make domain sense?
   - Any surprising findings?

Output: Scientific report (markdown or HTML) with figures.
```

**Success Criteria:**
- [ ] Clusters are interpretable
- [ ] Results stable across runs
- [ ] Consistent with domain knowledge
- [ ] Report is publication-ready

---

## Prompt 25: Master Test Runner

```
Create master script to run all tests.

Create run_all_tests.py:

1. Command-line interface:
```bash
python run_all_tests.py --tests 1,2,3,4,5 --parallel --n_jobs 8
python run_all_tests.py --test 1 --quick  # Quick mode: 5 restarts only
python run_all_tests.py --analyze-only  # Just run analysis scripts
```

2. Options:
   - --tests: Which tests to run (default: all)
   - --quick: Reduce n_restarts for testing (5 instead of 20)
   - --n_jobs: Number of parallel jobs
   - --analyze-only: Skip computation, just analyze existing results
   - --force: Overwrite existing results

3. Progress reporting:
   - Overall progress bar
   - Current test + algorithm
   - Estimated time remaining
   - Log file of all output
   
4. Email notification when complete (optional):
   - Send summary statistics
   - Attach key figures
   
5. Generate master summary:
   - Table comparing all tests
   - Overall winner
   - Publication-ready summary

Save log to results/run_log_{timestamp}.txt
```

**Success Criteria:**
- [ ] Can run all tests sequentially
- [ ] Parallel execution works
- [ ] Can resume interrupted runs
- [ ] Summary is comprehensive

---

# Phase 5: Analysis & Visualization

## Prompt 26: Cross-Test Comparison

```
Create comprehensive cross-test analysis.

Create analysis/cross_test_comparison.py:

1. Load results from all 5 tests

2. Generate "Table 2: Summary Across All Tests" from spec:
   - Rows: Tests
   - Columns: Algorithms
   - Cells: Primary metric (time or success rate)
   - Mark winners
   
3. Compute overall statistics:
   - How many tests did each algorithm win?
   - Average speedup over standard EM
   - Consistency (low variance across tests)
   
4. Generate figure: "Algorithm Performance Heatmap"
   - Rows: Tests
   - Columns: Algorithms
   - Color: Normalized performance (0=worst, 1=best)
   
5. Statistical meta-analysis:
   - Aggregate effect sizes across tests
   - Overall confidence intervals
   
6. Identify patterns:
   - When does lookahead win? (structured H_Q?)
   - When does it lose? (simple problems?)
   - Is adaptive γ consistently better than fixed?

Output: Cross-test summary report with recommendations.
```

**Success Criteria:**
- [ ] Table 2 matches spec format
- [ ] Heatmap reveals clear patterns
- [ ] Can identify when to use each method
- [ ] Recommendations are data-driven

---

## Prompt 27: Publication Figures

```
Create publication-quality figures.

Create analysis/publication_figures.py:

1. Figure 1: Convergence curves for all tests (5 subplots)
   - Clean, consistent styling
   - Same axis ranges where comparable
   - Legend only in first subplot
   
2. Figure 2: Time comparison across tests (grouped bar chart)
   - x-axis: tests
   - Grouped bars: algorithms
   - Error bars: ±1 SEM
   
3. Figure 3: Success rate analysis (Test 4 focus)
   - Grouped bars by initialization strategy
   - Clearly shows robustness differences
   
4. Figure 4: Memory usage scaling
   - New experiment: Vary K from 10 to 200
   - Plot memory vs K for each algorithm
   - Log scale if needed
   
5. Figure 5: Gamma schedule effect (for lookahead only)
   - Compare fixed vs adaptive γ
   - Show example convergence trajectories
   
Save all as:
- High-res PNG (300 dpi)
- PDF (vector graphics)
- Dimensions: 6" wide for single column, 12" for double

Use consistent color scheme across all figures.
```

**Success Criteria:**
- [ ] All figures publication-ready
- [ ] Fonts readable at journal size
- [ ] Color scheme is colorblind-friendly
- [ ] Vector formats for line plots

---

## Prompt 28: LaTeX Tables Generator

```
Generate LaTeX tables for paper.

Create analysis/latex_tables.py:

1. Generate all tables from spec in LaTeX format:
   - Table 1: Test 1 results
   - Table 2: Cross-test summary
   - Tables for Tests 2-5
   
2. Features:
   - Proper LaTeX formatting (booktabs package)
   - Bold for best values
   - ± symbols for std
   - Alignment (numbers right-aligned)
   - Captions
   
3. Example output:
```latex
\begin{table}[ht]
\centering
\caption{Performance on High-Dimensional GMM (Test 1)}
\label{tab:test1}
\begin{tabular}{lrrrrr}
\toprule
Method & Time (s) & Iters & Final $\ell$ & Success & Memory (MB) \\
\midrule
Standard EM & 45.2 $\pm$ 3.1 & 157 $\pm$ 12 & -2341.2 & 85\% & 120 \\
Lookahead (Adaptive) & \textbf{29.8} $\pm$ \textbf{1.9} & \textbf{71} $\pm$ \textbf{5} & \textbf{-2340.7} & \textbf{95\%} & 145 \\
\bottomrule
\end{tabular}
\end{table}
```

4. Save all tables to results/tables/ as .tex files

Also generate markdown versions for README.
```

**Success Criteria:**
- [ ] LaTeX compiles without errors
- [ ] Tables look professional
- [ ] Bold formatting highlights winners
- [ ] Can be directly inserted in paper

---

## Prompt 29: Results Webpage

```
Create interactive HTML results webpage.

Create analysis/generate_webpage.py:

1. Generate self-contained HTML page with:
   - Executive summary
   - All tables (sortable)
   - All figures (zoomable)
   - Navigation menu
   - Download links for data
   
2. Structure:
   - index.html (overview)
   - test_1.html, test_2.html, ... (detailed results)
   - comparison.html (cross-test analysis)
   
3. Use simple template (no heavy frameworks):
   - Bootstrap for styling
   - DataTables for interactive tables
   - Plotly for interactive plots (optional)
   
4. Include:
   - Filtering options (show/hide algorithms)
   - Download buttons (CSV, PNG)
   - Links to original data/code
   
5. Deploy option:
   - Can be hosted on GitHub Pages
   - Or included as supplementary material

Save to results/webpage/
```

**Success Criteria:**
- [ ] Page loads correctly in browser
- [ ] Tables are sortable/filterable
- [ ] Figures display properly
- [ ] Navigation works
- [ ] Mobile-responsive

---

## Prompt 30: Final Report Generator

```
Create comprehensive final report.

Create analysis/generate_report.py:

1. Generate report.md with:
   - Executive Summary
   - Methods (brief)
   - Results (all 5 tests)
   - Discussion
   - Conclusions
   - Recommendations

2. For each test include:
   - Objective
   - Setup
   - Results (inline tables/figures)
   - Interpretation
   
3. Discussion should address:
   - When does lookahead EM work? (structured H_Q)
   - When doesn't it work? (simple problems, overhead)
   - Is adaptive γ beneficial?
   - Practical recommendations
   - Limitations
   - Future work
   
4. Conclusion format:
   - "Lookahead EM wins on X/5 tests"
   - "Provides Y% average speedup on problems with structured H_Q"
   - "Best used when: [specific conditions]"
   - "Not recommended when: [specific conditions]"
   
5. Convert to PDF using pandoc

Output saved to results/FINAL_REPORT.{md,pdf}

Should be suitable as supplementary material for paper.
```

**Success Criteria:**
- [ ] Report is comprehensive (20-30 pages)
- [ ] All claims backed by data
- [ ] Honestly reports failures
- [ ] Provides actionable recommendations
- [ ] PDF looks professional

---

# Verification Checklist

After completing all prompts, verify:

## Functionality
- [ ] All 5 tests run without errors
- [ ] Results are reproducible (same seed → same result)
- [ ] All algorithms respect constraints
- [ ] No memory leaks
- [ ] Proper error handling

## Quality
- [ ] Code is documented (docstrings)
- [ ] Tests have unit tests
- [ ] Results match expectations (sanity checks)
- [ ] Figures are publication-quality
- [ ] Tables are formatted correctly

## Completeness
- [ ] All algorithms implemented
- [ ] All models implemented
- [ ] All tests implemented
- [ ] All analyses implemented
- [ ] All visualizations created

## Reproducibility
- [ ] requirements.txt is complete
- [ ] Random seeds set everywhere
- [ ] Data generation is deterministic
- [ ] README has clear instructions
- [ ] Can reproduce from scratch

## Scientific Rigor
- [ ] Statistical tests appropriate
- [ ] Multiple comparison correction applied
- [ ] Effect sizes reported
- [ ] Confidence intervals included
- [ ] Honest reporting (including failures)

---

# Quick Start Guide

**Minimum path to results (if time is limited):**

1. Prompts 1-5: Setup (1 day)
2. Prompts 6-9: Core algorithms (2 days)
3. Prompt 11-12: GMM only (1 day)
4. Prompts 16-18: Test 1 only (2 days)
5. Prompt 26: Quick analysis (1 day)

**Total: 1 week for proof-of-concept on Test 1 only**

Then expand to other tests if results are promising.

---

# Troubleshooting Guide

**Common issues and solutions:**

1. **SQUAREM wrapper fails**
   - Skip SQUAREM if rpy2/R installation problematic
   - Focus on lookahead vs standard EM comparison

2. **Memory issues with large K**
   - Reduce K for initial tests
   - Use sparse matrices for Hessian
   - Process restarts sequentially

3. **Convergence issues**
   - Check constraint enforcement
   - Verify gradient computation
   - Add regularization to Hessian

4. **Tests take too long**
   - Use --quick mode (5 restarts)
   - Run overnight
   - Use more parallel jobs

5. **Results inconsistent**
   - Check random seed setting
   - Verify data generation is deterministic
   - Check for race conditions in parallel code

---

# Resources

**Key files to reference:**
- EMPIRICAL_TEST_SPECIFICATIONS.md (the spec)
- This document (the prompts)
- Your corrected lookahead_em_corrected.py (as reference)

**External resources:**
- turboEM R package documentation
- scikit-learn GMM implementation (for reference)
- scipy.optimize docs (for constraint handling)

---

**Good luck with implementation!**

Track your progress by checking off completed prompts.
Estimated total time: 4 weeks following this guide.
