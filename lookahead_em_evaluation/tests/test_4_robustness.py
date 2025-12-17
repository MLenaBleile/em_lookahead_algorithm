"""
Test 4: Robustness/Initialization Study

Implements Test 4 from EMPIRICAL_TEST_SPECIFICATIONS.md.
Tests EM algorithms on simple GMM with various initialization strategies.

This test evaluates:
1. Success rate with challenging initializations
2. Robustness to poor starting points
3. Ability to escape local optima
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_gmm import generate_gmm_data, generate_initializations
from models.gmm import GaussianMixtureModel
from tests.test_runner import TestConfig, run_test, run_single_trial


def test_4_simple_gmm(seed: int = 42):
    """Generate simple, well-separated GMM for Test 4."""
    return generate_gmm_data(n=500, K=5, d=2, separation=4.0, seed=seed)


def load_test_4_config(
    n_restarts_per_strategy: int = 20,
    max_iter: int = 500,
    timeout: float = 120.0,
    tol: float = 1e-4,
    results_dir: str = None
) -> Dict[str, TestConfig]:
    """
    Load Test 4 configurations for each initialization strategy.

    Returns dict of TestConfig objects, one per initialization strategy.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.2}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.5}},
        {'name': 'lookahead_em', 'params': {'gamma': 0.8}},
        {'name': 'lookahead_em', 'params': {'gamma': 'adaptive'}},
        {'name': 'squarem', 'params': {}},
    ]

    strategies = {
        'collapsed': ('collapsed', 20),
        'far_away': ('far_away', 20),
        'random': ('random', 50),
        'kmeans': ('kmeans', 20),
    }

    configs = {}
    for name, (strategy, n_restarts) in strategies.items():
        configs[name] = TestConfig(
            test_id=f'test_4_robustness_{name}',
            data_generator=lambda: test_4_simple_gmm(seed=42),
            model_class=GaussianMixtureModel,
            initialization_strategy=strategy,
            n_restarts=n_restarts,
            algorithms=algorithms,
            max_iter=max_iter,
            timeout=timeout,
            tol=tol,
            results_dir=results_dir
        )

    return configs


def run_test_4(
    n_restarts_per_strategy: int = 20,
    parallel: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Test 4: Robustness study with all initialization strategies."""
    configs = load_test_4_config(n_restarts_per_strategy=n_restarts_per_strategy)

    if verbose:
        print("=" * 60)
        print("Test 4: Robustness/Initialization Study")
        print("=" * 60)
        print("  K = 5 components, d = 2 dimensions, n = 500 samples")
        print(f"  Initialization strategies: {list(configs.keys())}")
        print("=" * 60)

    all_results = {}
    for strategy_name, config in configs.items():
        if verbose:
            print(f"\nRunning strategy: {strategy_name}")

        results = run_test(
            config,
            parallel=parallel,
            n_jobs=n_jobs,
            verbose=verbose,
            save_results=save_results
        )

        # Tag results with strategy
        for r in results:
            r['initialization_strategy'] = strategy_name

        all_results[strategy_name] = results

    return all_results


def run_quick_test(verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """Quick version of Test 4."""
    def quick_data_generator():
        return generate_gmm_data(n=200, K=3, d=2, seed=42)

    quick_algorithms = [
        {'name': 'standard_em', 'params': {}},
        {'name': 'squarem', 'params': {}},
    ]

    configs = {
        'random': TestConfig(
            test_id='test_4_quick_random',
            data_generator=quick_data_generator,
            model_class=GaussianMixtureModel,
            initialization_strategy='random',
            n_restarts=2,
            algorithms=quick_algorithms,
            max_iter=100,
            timeout=60.0,
            tol=1e-4,
            results_dir='/tmp/test_results'
        ),
        'collapsed': TestConfig(
            test_id='test_4_quick_collapsed',
            data_generator=quick_data_generator,
            model_class=GaussianMixtureModel,
            initialization_strategy='collapsed',
            n_restarts=2,
            algorithms=quick_algorithms,
            max_iter=100,
            timeout=60.0,
            tol=1e-4,
            results_dir='/tmp/test_results'
        ),
    }

    if verbose:
        print("=" * 60)
        print("Test 4: Quick Validation Run")
        print("=" * 60)

    all_results = {}
    for strategy_name, config in configs.items():
        if verbose:
            print(f"\nRunning strategy: {strategy_name}")
        results = run_test(config, parallel=False, verbose=verbose, save_results=False)
        for r in results:
            r['initialization_strategy'] = strategy_name
        all_results[strategy_name] = results

    return all_results


def summarize_test_4_results(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate summary by strategy and algorithm."""
    summary = {}

    # Collect all results for computing success threshold
    all_lls = []
    for results in all_results.values():
        for r in results:
            if r.get('final_ll', float('-inf')) > float('-inf') and r.get('error') is None:
                all_lls.append(r['final_ll'])

    best_ll = max(all_lls) if all_lls else float('-inf')
    success_threshold = best_ll - 5.0

    for strategy, results in all_results.items():
        # Group by algorithm
        by_algo = {}
        for r in results:
            algo = r.get('algorithm', 'unknown')
            params = r.get('algorithm_params', {})
            key = f"{algo}_gamma_{params.get('gamma', '')}" if params.get('gamma') else algo
            if key not in by_algo:
                by_algo[key] = []
            by_algo[key].append(r)

        strategy_summary = {}
        for algo, algo_results in by_algo.items():
            successful = [r for r in algo_results if r.get('error') is None]
            final_lls = [r['final_ll'] for r in successful if r.get('final_ll', float('-inf')) > float('-inf')]

            n_success = sum(1 for ll in final_lls if ll >= success_threshold)
            success_rate = n_success / len(final_lls) if final_lls else 0

            strategy_summary[algo] = {
                'n_total': len(algo_results),
                'n_success': n_success,
                'success_rate': success_rate,
                'final_ll_mean': np.mean(final_lls) if final_lls else float('nan'),
                'final_ll_std': np.std(final_lls) if final_lls else float('nan'),
            }

        summary[strategy] = strategy_summary

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary."""
    print("\n" + "=" * 100)
    print("Test 4: Robustness Study - Success Rates by Initialization Strategy")
    print("=" * 100)

    # Get all algorithms
    all_algos = set()
    for strategy_data in summary.values():
        all_algos.update(strategy_data.keys())
    algos = sorted(all_algos)

    # Header
    header = f"{'Strategy':<15}"
    for algo in algos:
        header += f" {algo[:12]:>12}"
    print(header)
    print("-" * 100)

    # Data rows
    for strategy in summary:
        row = f"{strategy:<15}"
        for algo in algos:
            data = summary[strategy].get(algo, {})
            rate = data.get('success_rate', 0)
            row += f" {100*rate:>11.1f}%"
        print(row)

    print("=" * 100)


# Unit Tests
def test_config_loading():
    """Test configuration loading."""
    configs = load_test_4_config(n_restarts_per_strategy=5)
    assert len(configs) >= 4, "Should have multiple strategies"
    for name, config in configs.items():
        assert config.test_id.startswith('test_4'), f"Wrong test_id for {name}"
    print("  PASSED")


def test_quick_run():
    """Test quick test runs."""
    results = run_quick_test(verbose=False)
    assert len(results) > 0, "No results"
    for strategy, r_list in results.items():
        assert len(r_list) > 0, f"No results for {strategy}"
    print("  PASSED")


def test_summary_generation():
    """Test summary generation."""
    mock_results = {
        'random': [
            {'algorithm': 'standard_em', 'algorithm_params': {}, 'final_ll': -100.0, 'error': None},
            {'algorithm': 'standard_em', 'algorithm_params': {}, 'final_ll': -95.0, 'error': None},
        ],
        'collapsed': [
            {'algorithm': 'standard_em', 'algorithm_params': {}, 'final_ll': -110.0, 'error': None},
        ],
    }
    summary = summarize_test_4_results(mock_results)
    assert 'random' in summary and 'collapsed' in summary
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_4_robustness.py unit tests")
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

    parser = argparse.ArgumentParser(description="Test 4: Robustness Study")
    parser.add_argument('--quick', action='store_true', help="Run quick test")
    parser.add_argument('--test', action='store_true', help="Run unit tests")
    parser.add_argument('--restarts', type=int, default=20, help="Restarts per strategy")
    parser.add_argument('--jobs', type=int, default=8, help="Parallel jobs")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallel")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    elif args.quick:
        results = run_quick_test()
        summary = summarize_test_4_results(results)
        print_summary(summary)
    else:
        results = run_test_4(
            n_restarts_per_strategy=args.restarts,
            parallel=not args.no_parallel,
            n_jobs=args.jobs
        )
        summary = summarize_test_4_results(results)
        print_summary(summary)
