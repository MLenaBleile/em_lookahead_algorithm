"""
Test 5: Stress Test / Scaling Study

Implements Test 5 from EMPIRICAL_TEST_SPECIFICATIONS.md.
Tests EM algorithms on increasingly challenging scenarios.

This test evaluates:
1. Scaling behavior with K (number of components)
2. Memory usage scaling
3. Time scaling
4. Numerical stability under stress
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookahead_em_evaluation.data.generate_gmm import generate_gmm_data, generate_initializations
from lookahead_em_evaluation.models.gmm import GaussianMixtureModel
from tests.test_runner import TestConfig, run_test


def generate_scaling_data(K: int, d: int = 2, n_per_k: int = 100, seed: int = 42):
    """Generate GMM data that scales with K."""
    n = n_per_k * K
    return generate_gmm_data(n=n, K=K, d=d, seed=seed)


def load_test_5_config(
    K_values: List[int] = None,
    n_restarts: int = 10,
    max_iter: int = 500,
    timeout: float = 300.0,
    tol: float = 1e-4,
    results_dir: str = None
) -> Dict[int, TestConfig]:
    """
    Load Test 5 configurations for different K values.

    Returns dict of TestConfig objects, one per K value.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    if K_values is None:
        K_values = [10, 25, 50, 100]

    algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},
        {'name': 'squarem', 'params': {}},
    ]

    configs = {}
    for K in K_values:
        def data_gen_factory(K_val):
            return lambda: generate_scaling_data(K=K_val, d=2, seed=42)

        configs[K] = TestConfig(
            test_id=f'test_5_stress_K{K}',
            data_generator=data_gen_factory(K),
            model_class=GaussianMixtureModel,
            initialization_strategy='kmeans',
            n_restarts=n_restarts,
            algorithms=algorithms,
            max_iter=max_iter,
            timeout=timeout,
            tol=tol,
            results_dir=results_dir
        )

    return configs


def run_test_5(
    K_values: List[int] = None,
    n_restarts: int = 10,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[int, List[Dict[str, Any]]]:
    """Run Test 5: Scaling/Stress test with various K values."""
    if K_values is None:
        K_values = [10, 25, 50, 100]

    configs = load_test_5_config(K_values=K_values, n_restarts=n_restarts)

    if verbose:
        print("=" * 60)
        print("Test 5: Stress Test / Scaling Study")
        print("=" * 60)
        print(f"  K values: {K_values}")
        print(f"  n_restarts per K: {n_restarts}")
        print("=" * 60)

    all_results = {}
    for K in K_values:
        config = configs[K]
        if verbose:
            print(f"\nRunning K={K} (n={K*100})...")

        results = run_test(
            config,
            parallel=parallel,
            n_jobs=n_jobs,
            verbose=verbose,
            save_results=save_results
        )

        for r in results:
            r['K'] = K

        all_results[K] = results

    return all_results


def run_quick_test(verbose: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """Quick version of Test 5."""
    K_values = [5, 10]

    quick_algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},
        {'name': 'squarem', 'params': {}},
    ]

    if verbose:
        print("=" * 60)
        print("Test 5: Quick Validation Run")
        print(f"  K values: {K_values}")
        print("=" * 60)

    all_results = {}
    for K in K_values:
        def data_gen():
            return generate_scaling_data(K=K, d=2, n_per_k=50, seed=42)

        config = TestConfig(
            test_id=f'test_5_quick_K{K}',
            data_generator=data_gen,
            model_class=GaussianMixtureModel,
            initialization_strategy='kmeans',
            n_restarts=2,
            algorithms=quick_algorithms,
            max_iter=100,
            timeout=60.0,
            tol=1e-4,
            results_dir='/tmp/test_results'
        )

        if verbose:
            print(f"\nRunning K={K}...")

        results = run_test(config, parallel=False, verbose=verbose, save_results=False)
        for r in results:
            r['K'] = K
        all_results[K] = results

    return all_results


def summarize_test_5_results(all_results: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate summary showing scaling behavior."""
    summary = {'by_K': {}, 'by_algo': {}}

    # Organize by K
    for K, results in all_results.items():
        by_algo = {}
        for r in results:
            algo = r.get('algorithm', 'unknown')
            params = r.get('algorithm_params', {})
            key = f"{algo}_gamma_{params.get('gamma', '')}" if params.get('gamma') else algo
            if key not in by_algo:
                by_algo[key] = []
            by_algo[key].append(r)

        K_summary = {}
        for algo, algo_results in by_algo.items():
            successful = [r for r in algo_results if r.get('error') is None]
            times = [r['elapsed_time'] for r in successful if r.get('elapsed_time', 0) > 0]
            memory = [r['peak_memory_mb'] for r in successful if r.get('peak_memory_mb', 0) > 0]
            iters = [r['n_iterations'] for r in successful if r.get('n_iterations', 0) > 0]

            K_summary[algo] = {
                'n_successful': len(successful),
                'time_mean': np.mean(times) if times else float('nan'),
                'time_std': np.std(times) if times else float('nan'),
                'memory_mean': np.mean(memory) if memory else float('nan'),
                'iterations_mean': np.mean(iters) if iters else float('nan'),
            }

        summary['by_K'][K] = K_summary

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print scaling summary."""
    print("\n" + "=" * 100)
    print("Test 5: Scaling Study Results")
    print("=" * 100)

    by_K = summary['by_K']
    K_values = sorted(by_K.keys())

    # Get all algorithms
    all_algos = set()
    for K_data in by_K.values():
        all_algos.update(K_data.keys())
    algos = sorted(all_algos)

    # Time scaling
    print("\nTime (seconds) by K:")
    print(f"{'K':>6}", end="")
    for algo in algos:
        print(f" {algo[:15]:>15}", end="")
    print()

    for K in K_values:
        print(f"{K:>6}", end="")
        for algo in algos:
            t = by_K[K].get(algo, {}).get('time_mean', float('nan'))
            print(f" {t:>15.2f}", end="")
        print()

    # Memory scaling
    print("\nMemory (MB) by K:")
    print(f"{'K':>6}", end="")
    for algo in algos:
        print(f" {algo[:15]:>15}", end="")
    print()

    for K in K_values:
        print(f"{K:>6}", end="")
        for algo in algos:
            m = by_K[K].get(algo, {}).get('memory_mean', float('nan'))
            print(f" {m:>15.1f}", end="")
        print()

    print("=" * 100)


# Unit Tests
def test_config_loading():
    """Test configuration loading."""
    configs = load_test_5_config(K_values=[5, 10])
    assert len(configs) == 2
    assert 5 in configs and 10 in configs
    print("  PASSED")


def test_quick_run():
    """Test quick test runs."""
    results = run_quick_test(verbose=False)
    assert len(results) >= 2
    for K, r_list in results.items():
        assert len(r_list) > 0
    print("  PASSED")


def test_summary_generation():
    """Test summary generation."""
    mock_results = {
        10: [
            {'algorithm': 'standard_em', 'algorithm_params': {},
             'final_ll': -1000.0, 'elapsed_time': 5.0, 'peak_memory_mb': 100.0,
             'n_iterations': 50, 'error': None, 'K': 10},
        ],
        20: [
            {'algorithm': 'standard_em', 'algorithm_params': {},
             'final_ll': -2000.0, 'elapsed_time': 12.0, 'peak_memory_mb': 200.0,
             'n_iterations': 60, 'error': None, 'K': 20},
        ]
    }
    summary = summarize_test_5_results(mock_results)
    assert 'by_K' in summary
    assert 10 in summary['by_K'] and 20 in summary['by_K']
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_5_stress.py unit tests")
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

    parser = argparse.ArgumentParser(description="Test 5: Stress/Scaling Test")
    parser.add_argument('--quick', action='store_true', help="Run quick test")
    parser.add_argument('--test', action='store_true', help="Run unit tests")
    parser.add_argument('--K', type=int, nargs='+', default=None, help="K values to test")
    parser.add_argument('--restarts', type=int, default=10, help="Restarts per K")
    parser.add_argument('--jobs', type=int, default=8, help="Parallel jobs")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallel")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    elif args.quick:
        results = run_quick_test()
        summary = summarize_test_5_results(results)
        print_summary(summary)
    else:
        results = run_test_5(
            K_values=args.K,
            n_restarts=args.restarts,
            parallel=not args.no_parallel,
            n_jobs=args.jobs
        )
        summary = summarize_test_5_results(results)
        print_summary(summary)
