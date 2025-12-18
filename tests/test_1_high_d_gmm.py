"""
Test 1: High-Dimensional Sparse GMM

Implements Test 1 from EMPIRICAL_TEST_SPECIFICATIONS.md.
Tests EM algorithms on a GMM with K=50 components, d=2 dimensions, n=5000 samples.

This test evaluates:
1. Convergence speed (iterations to tolerance)
2. Final log-likelihood achieved
3. Computational efficiency
4. Stability across random restarts
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookahead_em_evaluation.data.generate_gmm import test_1_high_d, generate_initializations
from lookahead_em_evaluation.models.gmm import GaussianMixtureModel
from tests.test_runner import TestConfig, run_test, load_results


def load_test_1_config(
    n_restarts: int = 20,
    max_iter: int = 1000,
    timeout: float = 600.0,
    tol: float = 1e-6,
    results_dir: str = None
) -> TestConfig:
    """
    Load Test 1 configuration.

    Configuration from EMPIRICAL_TEST_SPECIFICATIONS.md:
    - K = 50 components
    - d = 2 dimensions
    - n = 5000 samples
    - Suboptimal K-means initialization (max_iter=10)

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

        # Lookahead EM with fixed gamma values
        {'name': 'lookahead_em', 'params': {'gamma': 0.3}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.5}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.7}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.9}},

        # Lookahead EM with adaptive gamma
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},
    ]

    config = TestConfig(
        test_id='test_1_high_d_gmm',
        data_generator=lambda: test_1_high_d(seed=42),
        model_class=GaussianMixtureModel,
        initialization_strategy='kmeans',  # Using K-means with limited iterations
        n_restarts=n_restarts,
        algorithms=algorithms,
        max_iter=max_iter,
        timeout=timeout,
        tol=tol,
        results_dir=results_dir
    )

    return config


def run_test_1(
    n_restarts: int = 20,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run Test 1: High-Dimensional GMM experiment.

    Args:
        n_restarts: Number of random restarts per algorithm
        parallel: Whether to use parallel execution
        n_jobs: Number of parallel jobs
        verbose: Whether to print progress
        save_results: Whether to save results to file

    Returns:
        List of run result dictionaries
    """
    config = load_test_1_config(n_restarts=n_restarts)

    if verbose:
        print("=" * 60)
        print("Test 1: High-Dimensional Sparse GMM")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  K = 50 components")
        print(f"  d = 2 dimensions")
        print(f"  n = 5000 samples")
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

    results = run_test(
        config,
        parallel=parallel,
        n_jobs=n_jobs,
        verbose=verbose,
        save_results=save_results
    )

    return results


def run_quick_test(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run a quick version of Test 1 for validation.

    Uses fewer restarts, iterations, and a smaller K for faster execution.

    Args:
        verbose: Whether to print progress

    Returns:
        List of run result dictionaries
    """
    from lookahead_em_evaluation.data.generate_gmm import generate_gmm_data

    # Use smaller GMM for quick test (K=5 instead of K=50)
    def quick_data_generator():
        return generate_gmm_data(n=500, K=5, d=2, seed=42)

    # Only test 2 algorithms for speed
    quick_algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.9}},
    ]

    config = TestConfig(
        test_id='test_1_quick',
        data_generator=quick_data_generator,
        model_class=GaussianMixtureModel,
        initialization_strategy='random',  # Harder initialization to show lookahead benefit
        n_restarts=2,
        algorithms=quick_algorithms,
        max_iter=500,  # More iterations
        timeout=120.0,
        tol=1e-4,  # Practical tolerance for GMM
        results_dir='/tmp/test_results'
    )

    if verbose:
        print("=" * 60)
        print("Test 1: Quick Validation Run")
        print("  (Using K=5, n=500 for faster execution)")
        print("=" * 60)

    results = run_test(
        config,
        parallel=False,  # Sequential for quick test
        verbose=verbose,
        save_results=False
    )

    return results


def summarize_test_1_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for Test 1 results.

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
    print("Test 1 Results Summary")
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
    config = load_test_1_config(n_restarts=5)

    assert config.test_id == 'test_1_high_d_gmm', "Test ID mismatch"
    assert config.n_restarts == 5, "n_restarts mismatch"
    assert len(config.algorithms) > 0, "No algorithms defined"

    # Test data generator
    X, z, theta_true = config.data_generator()
    assert X.shape == (5000, 2), f"X shape: {X.shape}"
    assert theta_true['mu'].shape == (50, 2), f"mu shape: {theta_true['mu'].shape}"

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
         'final_ll': -1000.0, 'elapsed_time': 10.0, 'n_iterations': 100, 'error': None},
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -990.0, 'elapsed_time': 9.0, 'n_iterations': 95, 'error': None},
        {'algorithm': 'lookahead_em', 'algorithm_params': {'gamma': 0.5}, 'converged': True,
         'final_ll': -985.0, 'elapsed_time': 12.0, 'n_iterations': 50, 'error': None},
    ]

    summary = summarize_test_1_results(results)

    assert 'standard_em' in summary, "Missing standard_em in summary"
    assert 'lookahead_em_gamma_0.5' in summary, "Missing lookahead_em in summary"

    assert summary['standard_em']['n_total'] == 2, "Wrong count for standard_em"
    assert summary['standard_em']['convergence_rate'] == 1.0, "Wrong convergence rate"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_1_high_d_gmm.py unit tests")
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

    parser = argparse.ArgumentParser(description="Test 1: High-Dimensional GMM")
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
        summary = summarize_test_1_results(results)
        print_summary(summary)
    else:
        results = run_test_1(
            n_restarts=args.restarts,
            parallel=not args.no_parallel,
            n_jobs=args.jobs
        )
        summary = summarize_test_1_results(results)
        print_summary(summary)
        print(f"\nCompleted {len(results)} runs")
