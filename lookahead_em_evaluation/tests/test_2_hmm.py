"""
Test 2: Hidden Markov Model

Implements Test 2 from EMPIRICAL_TEST_SPECIFICATIONS.md.
Tests EM algorithms on HMM with S=20 states, V=50 observations, T=1000 steps.

This test evaluates:
1. Convergence speed with structured latent space
2. Preservation of transition sparsity
3. Algorithm stability with constraint enforcement
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_hmm import test_2_hmm, generate_hmm_initializations
from models.hmm import HiddenMarkovModel
from tests.test_runner import TestConfig, run_test


class HMMTestAdapter:
    """
    Adapter to make HMM compatible with test_runner infrastructure.

    The test_runner expects a model that takes (n_components, n_features) in __init__
    and works with theta_init having 'mu' key for shape detection.
    """

    def __init__(self, n_components: int, n_features: int):
        """
        Initialize HMM adapter.

        For HMM: n_components = n_states, n_features = n_observations
        """
        self.n_states = n_components
        self.n_obs = n_features
        self.hmm = HiddenMarkovModel(n_states=n_components, n_observations=n_features)

    def e_step(self, X: np.ndarray, theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """E-step wrapper."""
        return self.hmm.e_step(X, theta)

    def m_step(self, X: np.ndarray, responsibilities: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """M-step wrapper."""
        return self.hmm.m_step(X, responsibilities)

    def log_likelihood(self, X: np.ndarray, theta: Dict[str, np.ndarray]) -> float:
        """Log-likelihood wrapper."""
        return self.hmm.log_likelihood(X, theta)

    def compute_Q(self, theta, theta_old, X, responsibilities) -> float:
        """Q-function wrapper."""
        return self.hmm.compute_Q(theta, theta_old, X, responsibilities)

    def compute_Q_gradient(self, theta, theta_old, X, responsibilities) -> np.ndarray:
        """Gradient wrapper."""
        return self.hmm.compute_Q_gradient(theta, theta_old, X, responsibilities)

    def compute_Q_hessian(self, theta, theta_old, X, responsibilities) -> np.ndarray:
        """Hessian wrapper."""
        return self.hmm.compute_Q_hessian(theta, theta_old, X, responsibilities)


def generate_hmm_init_wrapper(
    X: np.ndarray,
    K: int,
    strategy: str,
    n_inits: int,
    random_state: int
) -> List[Dict[str, np.ndarray]]:
    """
    Wrapper for HMM initialization compatible with test_runner.

    Args:
        X: Observation sequence
        K: Number of states (used as n_states)
        strategy: Initialization strategy
        n_inits: Number of initializations
        random_state: Random seed

    Returns:
        List of theta_init dicts with 'mu' key added for compatibility
    """
    # Infer n_obs from the observation sequence
    n_obs = int(X.max()) + 1

    # Map strategy names
    if strategy == 'kmeans':
        strategy = 'empirical'  # Use empirical for HMM
    elif strategy == 'random':
        strategy = 'random'

    inits = generate_hmm_initializations(
        observations=X,
        n_states=K,
        n_obs=n_obs,
        strategy=strategy,
        n_inits=n_inits,
        random_state=random_state
    )

    return inits


# Monkey-patch the initialization function for HMM tests
import data.generate_gmm as gmm_module
_original_generate_inits = gmm_module.generate_initializations


def load_test_2_config(
    n_restarts: int = 20,
    max_iter: int = 1000,
    timeout: float = 600.0,
    tol: float = 1e-6,
    results_dir: str = None
) -> TestConfig:
    """
    Load Test 2 configuration.

    Configuration from EMPIRICAL_TEST_SPECIFICATIONS.md:
    - S = 20 states
    - V = 50 observations
    - T = 1000 steps
    - Nearly diagonal transition (0.7 self-transition)

    Args:
        n_restarts: Number of random restarts per algorithm
        max_iter: Maximum EM iterations
        timeout: Timeout in seconds per run
        tol: Convergence tolerance
        results_dir: Directory for saving results

    Returns:
        TestConfig object
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    # Define algorithms to test
    algorithms = [
        # Standard EM baseline
        {'name': 'standard_em', 'params': {}},

        # Lookahead EM with adaptive gamma
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},

        # SQUAREM (uses pure Python fallback)
        {'name': 'squarem', 'params': {}},
    ]

    # Custom data generator that returns proper format
    def hmm_data_generator():
        X, Z, theta_true = test_2_hmm(seed=42)
        # Return format: (X, z, theta_true)
        return X, Z, theta_true

    config = TestConfig(
        test_id='test_2_hmm',
        data_generator=hmm_data_generator,
        model_class=HMMTestAdapter,
        initialization_strategy='random',  # Will be mapped to HMM strategy
        n_restarts=n_restarts,
        algorithms=algorithms,
        max_iter=max_iter,
        timeout=timeout,
        tol=tol,
        results_dir=results_dir
    )

    return config


def run_test_2(
    n_restarts: int = 20,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Test 2: HMM experiment.

    Args:
        n_restarts: Number of random restarts per algorithm
        parallel: Whether to use parallel execution
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress
        save_results: Whether to save results to file

    Returns:
        List of run result dictionaries
    """
    config = load_test_2_config(n_restarts=n_restarts)

    if verbose:
        print("=" * 60)
        print("Test 2: Hidden Markov Model")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  S = 20 states")
        print(f"  V = 50 observations")
        print(f"  T = 1000 steps")
        print(f"  n_restarts = {n_restarts}")
        print(f"  max_iter = {config.max_iter}")
        print(f"  timeout = {config.timeout}s")
        print(f"  tol = {config.tol}")
        print(f"  Algorithms: {len(config.algorithms)}")
        for algo in config.algorithms:
            if algo.get('params'):
                print(f"    - {algo['name']} ({algo['params']})")
            else:
                print(f"    - {algo['name']}")
        print("=" * 60)

    # Temporarily replace initialization function
    gmm_module.generate_initializations = generate_hmm_init_wrapper

    try:
        results = run_test(
            config,
            parallel=parallel,
            n_jobs=n_jobs,
            verbose=verbose,
            save_results=save_results
        )
    finally:
        # Restore original function
        gmm_module.generate_initializations = _original_generate_inits

    return results


def run_quick_test(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run a quick version of Test 2 for validation.

    Uses fewer restarts and smaller HMM.

    Args:
        verbose: Whether to print progress

    Returns:
        List of run result dictionaries
    """
    from data.generate_hmm import test_2_hmm_small, generate_hmm_initializations

    # Use smaller HMM for quick test
    def quick_data_generator():
        X, Z, theta_true = test_2_hmm_small(seed=42)
        return X, Z, theta_true

    # Test algorithms including lookahead
    quick_algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},
        {'name': 'squarem', 'params': {}},
    ]

    config = TestConfig(
        test_id='test_2_quick',
        data_generator=quick_data_generator,
        model_class=HMMTestAdapter,
        initialization_strategy='random',
        n_restarts=2,
        algorithms=quick_algorithms,
        max_iter=100,
        timeout=120.0,
        tol=1e-4,
        results_dir='/tmp/test_results'
    )

    if verbose:
        print("=" * 60)
        print("Test 2: Quick Validation Run")
        print("  (Using S=5, V=10, T=200 for faster execution)")
        print("=" * 60)

    # Replace initialization function
    gmm_module.generate_initializations = generate_hmm_init_wrapper

    try:
        results = run_test(
            config,
            parallel=False,
            verbose=verbose,
            save_results=False
        )
    finally:
        gmm_module.generate_initializations = _original_generate_inits

    return results


def summarize_test_2_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for Test 2 results.

    Args:
        results: List of run result dictionaries

    Returns:
        Dictionary with summary statistics per algorithm
    """
    summary = {}

    # Group by algorithm
    by_algo = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        params = r.get('algorithm_params', {})

        # Create unique key
        if params:
            gamma = params.get('gamma', '')
            key = f"{algo}_gamma_{gamma}"
        else:
            key = algo

        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    # Compute statistics for each algorithm
    for key, algo_results in by_algo.items():
        n_total = len(algo_results)
        n_converged = sum(1 for r in algo_results if r.get('converged', False))
        n_errors = sum(1 for r in algo_results if r.get('error') is not None)

        # Filter successful runs
        successful = [r for r in algo_results if r.get('error') is None]

        # Extract metrics
        final_lls = [r['final_ll'] for r in successful if r.get('final_ll', float('-inf')) > float('-inf')]
        times = [r['elapsed_time'] for r in successful if r.get('elapsed_time', 0) > 0]
        iters = [r['n_iterations'] for r in successful if r.get('n_iterations', 0) > 0]

        summary[key] = {
            'n_total': n_total,
            'n_converged': n_converged,
            'n_errors': n_errors,
            'convergence_rate': n_converged / n_total if n_total > 0 else 0,
            'final_ll_mean': np.mean(final_lls) if final_lls else float('nan'),
            'final_ll_std': np.std(final_lls) if final_lls else float('nan'),
            'final_ll_best': max(final_lls) if final_lls else float('nan'),
            'time_mean': np.mean(times) if times else float('nan'),
            'time_std': np.std(times) if times else float('nan'),
            'iterations_mean': np.mean(iters) if iters else float('nan'),
            'iterations_std': np.std(iters) if iters else float('nan'),
        }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary statistics."""
    print("\n" + "=" * 80)
    print("Test 2 Results Summary")
    print("=" * 80)

    # Sort by final_ll_mean descending
    sorted_keys = sorted(
        summary.keys(),
        key=lambda k: summary[k].get('final_ll_mean', float('-inf')),
        reverse=True
    )

    print(f"\n{'Algorithm':<30} {'Conv%':>8} {'Final LL':>15} {'Time (s)':>12} {'Iters':>10}")
    print("-" * 80)

    for key in sorted_keys:
        s = summary[key]
        conv_pct = f"{100*s['convergence_rate']:.1f}%"
        ll_str = f"{s['final_ll_mean']:.2f}±{s['final_ll_std']:.2f}" if not np.isnan(s['final_ll_mean']) else "N/A"
        time_str = f"{s['time_mean']:.2f}±{s['time_std']:.2f}" if not np.isnan(s['time_mean']) else "N/A"
        iter_str = f"{s['iterations_mean']:.0f}±{s['iterations_std']:.0f}" if not np.isnan(s['iterations_mean']) else "N/A"

        print(f"{key:<30} {conv_pct:>8} {ll_str:>15} {time_str:>12} {iter_str:>10}")

    print("=" * 80)


# ============================================================
# Unit Tests
# ============================================================

def test_config_loading():
    """Test that configuration loads correctly."""
    config = load_test_2_config(n_restarts=5)

    assert config.test_id == 'test_2_hmm', "Test ID mismatch"
    assert config.n_restarts == 5, "n_restarts mismatch"
    assert len(config.algorithms) > 0, "No algorithms defined"

    # Test data generator
    X, z, theta_true = config.data_generator()
    assert len(X) == 1000, f"X length: {len(X)}"
    assert theta_true['A'].shape == (20, 20), f"A shape: {theta_true['A'].shape}"
    assert theta_true['B'].shape == (20, 50), f"B shape: {theta_true['B'].shape}"

    print("  PASSED")


def test_quick_run():
    """Test that quick test runs without errors."""
    results = run_quick_test(verbose=False)

    assert len(results) > 0, "No results returned"

    # Check all results have required fields
    for r in results:
        assert 'algorithm' in r, "Missing algorithm field"
        assert 'seed' in r, "Missing seed field"

    print("  PASSED")


def test_summary_generation():
    """Test summary statistics generation."""
    # Create mock results
    results = [
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -800.0, 'elapsed_time': 5.0, 'n_iterations': 50, 'error': None},
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -790.0, 'elapsed_time': 4.5, 'n_iterations': 45, 'error': None},
        {'algorithm': 'squarem', 'algorithm_params': {}, 'converged': True,
         'final_ll': -785.0, 'elapsed_time': 3.0, 'n_iterations': 25, 'error': None},
    ]

    summary = summarize_test_2_results(results)

    assert 'standard_em' in summary, "Missing standard_em in summary"
    assert 'squarem' in summary, "Missing squarem in summary"

    assert summary['standard_em']['n_total'] == 2, "Wrong count for standard_em"
    assert summary['standard_em']['convergence_rate'] == 1.0, "Wrong convergence rate"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_2_hmm.py unit tests")
    print("=" * 60)

    print("Testing config loading...")
    test_config_loading()

    print("Testing quick run...")
    test_quick_run()

    print("Testing summary generation...")
    test_summary_generation()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 2: HMM")
    parser.add_argument('--quick', action='store_true', help="Run quick validation test")
    parser.add_argument('--test', action='store_true', help="Run unit tests only")
    parser.add_argument('--restarts', type=int, default=20, help="Number of restarts")
    parser.add_argument('--jobs', type=int, default=8, help="Number of parallel jobs")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallelization")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    elif args.quick:
        results = run_quick_test()
        summary = summarize_test_2_results(results)
        print_summary(summary)
    else:
        results = run_test_2(
            n_restarts=args.restarts,
            parallel=not args.no_parallel,
            n_jobs=args.jobs
        )
        summary = summarize_test_2_results(results)
        print_summary(summary)
        print(f"\nCompleted {len(results)} runs")
