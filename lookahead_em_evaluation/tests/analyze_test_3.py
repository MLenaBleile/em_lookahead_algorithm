"""
Test 3 Analysis: Mixture of Experts Results

Analyzes results from Test 3: MoE experiment.
Computes aggregate statistics, performs statistical tests,
and generates visualizations.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import ExperimentLogger


def load_test_3_results(results_dir: str = None) -> List[Dict[str, Any]]:
    """Load all Test 3 results."""
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )
    test_dir = os.path.join(results_dir, 'test_3_moe')
    logger = ExperimentLogger(test_dir)
    return logger.load_all()


def compute_aggregate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate statistics for each algorithm."""
    by_algo = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        params = r.get('algorithm_params', {})
        key = f"{algo}_gamma_{params.get('gamma', '')}" if params.get('gamma') else algo
        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    stats_dict = {}
    for key, algo_results in by_algo.items():
        n_total = len(algo_results)
        n_converged = sum(1 for r in algo_results if r.get('converged', False))
        n_errors = sum(1 for r in algo_results if r.get('error') is not None)
        successful = [r for r in algo_results if r.get('error') is None]

        final_lls = [r['final_ll'] for r in successful if r.get('final_ll', float('-inf')) > float('-inf')]
        times = [r['elapsed_time'] for r in successful if r.get('elapsed_time', 0) > 0]
        iters = [r['n_iterations'] for r in successful if r.get('n_iterations', 0) > 0]
        memory = [r['peak_memory_mb'] for r in successful if r.get('peak_memory_mb', 0) > 0]

        if final_lls:
            best_ll = max(final_lls)
            success_rate = sum(1 for ll in final_lls if ll >= best_ll - 5.0) / len(final_lls)
        else:
            success_rate = 0.0

        stats_dict[key] = {
            'n_total': n_total,
            'n_converged': n_converged,
            'n_errors': n_errors,
            'convergence_rate': n_converged / n_total if n_total > 0 else 0,
            'success_rate': success_rate,
            'final_ll': {
                'mean': np.mean(final_lls) if final_lls else float('nan'),
                'std': np.std(final_lls) if final_lls else float('nan'),
                'values': final_lls
            },
            'time': {
                'mean': np.mean(times) if times else float('nan'),
                'std': np.std(times) if times else float('nan'),
                'values': times
            },
            'iterations': {
                'mean': np.mean(iters) if iters else float('nan'),
                'std': np.std(iters) if iters else float('nan'),
                'values': iters
            },
            'memory': {
                'mean': np.mean(memory) if memory else float('nan'),
                'std': np.std(memory) if memory else float('nan'),
                'values': memory
            }
        }
    return stats_dict


def perform_statistical_tests(
    stats_dict: Dict[str, Dict[str, Any]],
    baseline: str = 'standard_em'
) -> Dict[str, Dict[str, Any]]:
    """Perform paired statistical tests vs baseline."""
    test_results = {}
    if baseline not in stats_dict:
        return test_results

    baseline_stats = stats_dict[baseline]
    n_comparisons = len(stats_dict) - 1

    for algo, algo_stats in stats_dict.items():
        if algo == baseline:
            continue

        tests = {}
        baseline_lls = baseline_stats['final_ll']['values']
        algo_lls = algo_stats['final_ll']['values']

        if len(baseline_lls) > 1 and len(algo_lls) > 1:
            n = min(len(baseline_lls), len(algo_lls))
            t_stat, p_value = stats.ttest_ind(algo_lls[:n], baseline_lls[:n])
            tests['ll_ttest'] = {
                'p_value': p_value,
                'p_value_corrected': min(p_value * n_comparisons, 1.0),
                'significant': p_value * n_comparisons < 0.05,
                'direction': 'better' if np.mean(algo_lls) > np.mean(baseline_lls) else 'worse'
            }

        baseline_times = baseline_stats['time']['values']
        algo_times = algo_stats['time']['values']
        if len(baseline_times) > 1 and len(algo_times) > 1:
            tests['speedup'] = np.mean(baseline_times) / np.mean(algo_times) if np.mean(algo_times) > 0 else float('nan')

        test_results[algo] = tests
    return test_results


def generate_ascii_table(stats_dict: Dict[str, Dict[str, Any]]) -> str:
    """Generate ASCII table for terminal display."""
    lines = ["=" * 90, "Test 3 Results: Mixture of Experts", "=" * 90]
    lines.append(f"{'Algorithm':<30} {'Conv%':>8} {'Final LL':>15} {'Time (s)':>12} {'Iters':>10}")
    lines.append("-" * 90)

    sorted_algos = sorted(stats_dict.keys(),
        key=lambda k: stats_dict[k]['final_ll'].get('mean', float('-inf')), reverse=True)

    for algo in sorted_algos:
        s = stats_dict[algo]
        conv_pct = f"{100*s['convergence_rate']:.1f}%"
        ll_str = f"{s['final_ll']['mean']:.1f}±{s['final_ll']['std']:.1f}" if not np.isnan(s['final_ll']['mean']) else "N/A"
        time_str = f"{s['time']['mean']:.2f}±{s['time']['std']:.2f}" if not np.isnan(s['time']['mean']) else "N/A"
        iter_str = f"{s['iterations']['mean']:.0f}±{s['iterations']['std']:.0f}" if not np.isnan(s['iterations']['mean']) else "N/A"
        lines.append(f"{algo:<30} {conv_pct:>8} {ll_str:>15} {time_str:>12} {iter_str:>10}")

    lines.append("=" * 90)
    return "\n".join(lines)


def run_analysis(results_dir: str = None, verbose: bool = True) -> Dict[str, Any]:
    """Run full analysis of Test 3 results."""
    results = load_test_3_results(results_dir)
    if not results:
        print("No results found for Test 3")
        return {}

    if verbose:
        print(f"Loaded {len(results)} results")

    stats_dict = compute_aggregate_statistics(results)
    test_results = perform_statistical_tests(stats_dict)

    if verbose:
        print(generate_ascii_table(stats_dict))

    return {'stats_dict': stats_dict, 'test_results': test_results, 'n_results': len(results)}


# Unit Tests
def test_statistics_computation():
    """Test aggregate statistics computation."""
    results = [
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -200.0, 'elapsed_time': 2.0, 'n_iterations': 30, 'peak_memory_mb': 50.0, 'error': None},
    ]
    stats = compute_aggregate_statistics(results)
    assert 'standard_em' in stats
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running analyze_test_3.py unit tests")
    print("=" * 60)
    print("Testing statistics computation...")
    test_statistics_computation()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Test 3 Results")
    parser.add_argument('--test', action='store_true', help="Run unit tests")
    args = parser.parse_args()

    if args.test:
        run_all_tests()
    else:
        run_analysis()
