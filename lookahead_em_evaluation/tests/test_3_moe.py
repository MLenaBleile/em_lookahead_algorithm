"""
Test 3: Mixture of Experts

Implements Test 3 from EMPIRICAL_TEST_SPECIFICATIONS.md.
Tests EM algorithms on MoE with G=5 experts, d=2 features, n=2000 samples.

This test evaluates:
1. Convergence in regression setting
2. Expert utilization (all experts used)
3. Block-diagonal Hessian exploitation
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mixture_of_experts import (
    MixtureOfExperts, test_3_moe, generate_moe_data,
    initialize_random, initialize_equal_gates
)
from tests.test_runner import TestConfig, run_test


class MoETestAdapter:
    """
    Adapter to make MoE compatible with test_runner infrastructure.

    The test_runner expects (n_components, n_features) constructor and theta with 'mu' key.
    """

    def __init__(self, n_components: int, n_features: int):
        """
        Initialize MoE adapter.

        For MoE: n_components = n_experts, n_features = n_features
        """
        self.n_experts = n_components
        self.n_features = n_features
        self.moe = MixtureOfExperts(n_experts=n_components, n_features=n_features)
        self._X = None
        self._y = None

    def e_step(self, data: Tuple[np.ndarray, np.ndarray], theta: Dict[str, np.ndarray]) -> np.ndarray:
        """E-step wrapper."""
        return self.moe.e_step(data, theta)

    def m_step(self, data: Tuple[np.ndarray, np.ndarray], responsibilities: np.ndarray) -> Dict[str, np.ndarray]:
        """M-step wrapper."""
        return self.moe.m_step(data, responsibilities)

    def log_likelihood(self, data: Tuple[np.ndarray, np.ndarray], theta: Dict[str, np.ndarray]) -> float:
        """Log-likelihood wrapper."""
        return self.moe.log_likelihood(data, theta)

    def compute_Q(self, theta, theta_old, data, responsibilities) -> float:
        """Q-function wrapper."""
        return self.moe.compute_Q(theta, theta_old, data, responsibilities)

    def compute_Q_gradient(self, theta, theta_old, data, responsibilities) -> np.ndarray:
        """Gradient wrapper."""
        return self.moe.compute_Q_gradient(theta, theta_old, data, responsibilities)

    def compute_Q_hessian(self, theta, theta_old, data, responsibilities) -> np.ndarray:
        """Hessian wrapper."""
        return self.moe.compute_Q_hessian(theta, theta_old, data, responsibilities)


def generate_moe_init_wrapper(
    X: Any,  # Can be tuple (X_features, y) or just X_features
    K: int,
    strategy: str,
    n_inits: int,
    random_state: int
) -> List[Dict[str, np.ndarray]]:
    """
    Generate MoE initializations.

    Args:
        X: Data - can be tuple (X_features, y) or just X_features
        K: Number of experts
        strategy: Initialization strategy
        n_inits: Number of initializations
        random_state: Random seed

    Returns:
        List of theta_init dicts with 'mu' key added for compatibility
    """
    # Handle tuple input from test_runner
    if isinstance(X, tuple):
        X_features, y = X
        d = X_features.shape[1]
    else:
        d = X.shape[1] if X.ndim == 2 else 1

    inits = []
    for i in range(n_inits):
        seed = random_state + i

        if strategy in ['kmeans', 'equal_gates']:
            theta = initialize_equal_gates(K, d, random_state=seed)
        else:
            theta = initialize_random(K, d, random_state=seed)

        # Add 'mu' key for compatibility with test_runner shape detection
        theta['mu'] = theta['gamma']  # Use gamma shape as proxy

        inits.append(theta)

    return inits


# Monkey-patch support
import data.generate_gmm as gmm_module
_original_generate_inits = gmm_module.generate_initializations


def load_test_3_config(
    n_restarts: int = 30,
    max_iter: int = 500,
    timeout: float = 300.0,
    tol: float = 1e-6,
    results_dir: str = None
) -> TestConfig:
    """
    Load Test 3 configuration.

    Configuration from EMPIRICAL_TEST_SPECIFICATIONS.md:
    - G = 5 experts
    - d = 2 features
    - n = 2000 samples

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

        # Lookahead EM with various gamma values
        {'name': 'lookahead_em', 'params': {'gamma': 0.2}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.4}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.6}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.8}},

        # Lookahead EM with adaptive gamma
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},

        # SQUAREM
        {'name': 'squarem', 'params': {}},
    ]

    # Custom data generator
    def moe_data_generator():
        X, y, experts, theta_true = test_3_moe(seed=42)
        # Return as tuple (X_combined, experts, theta_true)
        # where X_combined has y appended for passing through test runner
        return (X, y), experts, theta_true

    config = TestConfig(
        test_id='test_3_moe',
        data_generator=moe_data_generator,
        model_class=MoETestAdapter,
        initialization_strategy='equal_gates',
        n_restarts=n_restarts,
        algorithms=algorithms,
        max_iter=max_iter,
        timeout=timeout,
        tol=tol,
        results_dir=results_dir
    )

    return config


def run_test_3(
    n_restarts: int = 30,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Test 3: MoE experiment.

    Args:
        n_restarts: Number of random restarts per algorithm
        parallel: Whether to use parallel execution
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress
        save_results: Whether to save results to file

    Returns:
        List of run result dictionaries
    """
    config = load_test_3_config(n_restarts=n_restarts)

    if verbose:
        print("=" * 60)
        print("Test 3: Mixture of Experts")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  G = 5 experts")
        print(f"  d = 2 features")
        print(f"  n = 2000 samples")
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

    # Custom initialization function for MoE
    gmm_module.generate_initializations = generate_moe_init_wrapper

    try:
        results = run_test(
            config,
            parallel=parallel,
            n_jobs=n_jobs,
            verbose=verbose,
            save_results=save_results
        )
    finally:
        gmm_module.generate_initializations = _original_generate_inits

    return results


def run_quick_test(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run a quick version of Test 3 for validation.

    Args:
        verbose: Whether to print progress

    Returns:
        List of run result dictionaries
    """
    # Smaller MoE for quick test
    def quick_data_generator():
        X, y, experts, theta_true = generate_moe_data(
            n=200, n_experts=3, n_features=2, seed=42
        )
        return (X, y), experts, theta_true

    # Only test 2 algorithms for speed
    quick_algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'squarem', 'params': {}},
    ]

    config = TestConfig(
        test_id='test_3_quick',
        data_generator=quick_data_generator,
        model_class=MoETestAdapter,
        initialization_strategy='equal_gates',
        n_restarts=2,
        algorithms=quick_algorithms,
        max_iter=100,
        timeout=60.0,
        tol=1e-4,
        results_dir='/tmp/test_results'
    )

    if verbose:
        print("=" * 60)
        print("Test 3: Quick Validation Run")
        print("  (Using G=3, d=2, n=200 for faster execution)")
        print("=" * 60)

    gmm_module.generate_initializations = generate_moe_init_wrapper

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


def summarize_test_3_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for Test 3 results.

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

        if params:
            gamma = params.get('gamma', '')
            key = f"{algo}_gamma_{gamma}" if gamma else algo
        else:
            key = algo

        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    for key, algo_results in by_algo.items():
        n_total = len(algo_results)
        n_converged = sum(1 for r in algo_results if r.get('converged', False))
        n_errors = sum(1 for r in algo_results if r.get('error') is not None)

        successful = [r for r in algo_results if r.get('error') is None]

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
    print("Test 3 Results Summary")
    print("=" * 80)

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
    config = load_test_3_config(n_restarts=5)

    assert config.test_id == 'test_3_moe', "Test ID mismatch"
    assert config.n_restarts == 5, "n_restarts mismatch"
    assert len(config.algorithms) > 0, "No algorithms defined"

    # Test data generator
    data, experts, theta_true = config.data_generator()
    X, y = data
    assert X.shape == (2000, 2), f"X shape: {X.shape}"
    assert y.shape == (2000,), f"y shape: {y.shape}"
    assert theta_true['gamma'].shape == (5, 3), f"gamma shape: {theta_true['gamma'].shape}"

    print("  PASSED")


def test_quick_run():
    """Test that quick test runs without errors."""
    results = run_quick_test(verbose=False)

    assert len(results) > 0, "No results returned"

    for r in results:
        assert 'algorithm' in r, "Missing algorithm field"
        assert 'seed' in r, "Missing seed field"

    print("  PASSED")


def test_summary_generation():
    """Test summary statistics generation."""
    results = [
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -200.0, 'elapsed_time': 2.0, 'n_iterations': 30, 'error': None},
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -195.0, 'elapsed_time': 1.8, 'n_iterations': 28, 'error': None},
        {'algorithm': 'squarem', 'algorithm_params': {}, 'converged': True,
         'final_ll': -190.0, 'elapsed_time': 1.2, 'n_iterations': 15, 'error': None},
    ]

    summary = summarize_test_3_results(results)

    assert 'standard_em' in summary, "Missing standard_em"
    assert 'squarem' in summary, "Missing squarem"
    assert summary['standard_em']['n_total'] == 2, "Wrong count"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_3_moe.py unit tests")
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

    parser = argparse.ArgumentParser(description="Test 3: Mixture of Experts")
    parser.add_argument('--quick', action='store_true', help="Run quick validation test")
    parser.add_argument('--test', action='store_true', help="Run unit tests only")
    parser.add_argument('--restarts', type=int, default=30, help="Number of restarts")
    parser.add_argument('--jobs', type=int, default=8, help="Number of parallel jobs")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallelization")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    elif args.quick:
        results = run_quick_test()
        summary = summarize_test_3_results(results)
        print_summary(summary)
    else:
        results = run_test_3(
            n_restarts=args.restarts,
            parallel=not args.no_parallel,
            n_jobs=args.jobs
        )
        summary = summarize_test_3_results(results)
        print_summary(summary)
        print(f"\nCompleted {len(results)} runs")
