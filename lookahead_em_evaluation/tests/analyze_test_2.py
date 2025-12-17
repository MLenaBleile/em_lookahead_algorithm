"""
Test 2 Analysis: HMM Results

Analyzes results from Test 2: Hidden Markov Model experiment.
Computes aggregate statistics, performs statistical tests,
and generates visualizations.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import ExperimentLogger


def load_test_2_results(results_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load all Test 2 results.

    Args:
        results_dir: Base results directory

    Returns:
        List of run result dictionaries
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    test_dir = os.path.join(results_dir, 'test_2_hmm')
    logger = ExperimentLogger(test_dir)
    return logger.load_all()


def compute_aggregate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute aggregate statistics for each algorithm.

    Args:
        results: List of run result dictionaries

    Returns:
        Dict mapping algorithm name to statistics dict
    """
    # Group by algorithm
    by_algo = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        params = r.get('algorithm_params', {})

        # Create unique key
        if params:
            gamma = params.get('gamma', '')
            key = f"{algo}_gamma_{gamma}" if gamma else algo
        else:
            key = algo

        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    # Compute statistics
    stats_dict = {}

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
        memory = [r['peak_memory_mb'] for r in successful if r.get('peak_memory_mb', 0) > 0]

        # Compute success rate (reaching good likelihood)
        if final_lls:
            best_ll = max(final_lls)
            success_threshold = best_ll - 5.0  # Within 5 of best
            n_success = sum(1 for ll in final_lls if ll >= success_threshold)
            success_rate = n_success / len(final_lls)
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
                'median': np.median(final_lls) if final_lls else float('nan'),
                'best': max(final_lls) if final_lls else float('nan'),
                'worst': min(final_lls) if final_lls else float('nan'),
                'values': final_lls
            },
            'time': {
                'mean': np.mean(times) if times else float('nan'),
                'std': np.std(times) if times else float('nan'),
                'median': np.median(times) if times else float('nan'),
                'values': times
            },
            'iterations': {
                'mean': np.mean(iters) if iters else float('nan'),
                'std': np.std(iters) if iters else float('nan'),
                'median': np.median(iters) if iters else float('nan'),
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
    """
    Perform paired statistical tests comparing each algorithm to baseline.

    Args:
        stats_dict: Statistics dictionary from compute_aggregate_statistics
        baseline: Name of baseline algorithm

    Returns:
        Dict mapping algorithm name to test results
    """
    test_results = {}

    if baseline not in stats_dict:
        print(f"Warning: Baseline '{baseline}' not found in results")
        return test_results

    baseline_stats = stats_dict[baseline]
    baseline_lls = baseline_stats['final_ll']['values']
    baseline_times = baseline_stats['time']['values']
    baseline_iters = baseline_stats['iterations']['values']

    n_comparisons = len(stats_dict) - 1  # Bonferroni correction

    for algo, algo_stats in stats_dict.items():
        if algo == baseline:
            continue

        algo_lls = algo_stats['final_ll']['values']
        algo_times = algo_stats['time']['values']
        algo_iters = algo_stats['iterations']['values']

        tests = {}

        # Log-likelihood comparison (higher is better)
        if len(baseline_lls) > 1 and len(algo_lls) > 1:
            # Use min of lengths for paired comparison
            n = min(len(baseline_lls), len(algo_lls))
            t_stat, p_value = stats.ttest_ind(algo_lls[:n], baseline_lls[:n])
            tests['ll_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_value_corrected': min(p_value * n_comparisons, 1.0),
                'significant': p_value * n_comparisons < 0.05,
                'direction': 'better' if np.mean(algo_lls) > np.mean(baseline_lls) else 'worse'
            }

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline_lls) + np.var(algo_lls)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(algo_lls) - np.mean(baseline_lls)) / pooled_std
                tests['ll_effect_size'] = cohens_d

        # Time comparison (lower is better)
        if len(baseline_times) > 1 and len(algo_times) > 1:
            n = min(len(baseline_times), len(algo_times))
            t_stat, p_value = stats.ttest_ind(algo_times[:n], baseline_times[:n])
            tests['time_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_value_corrected': min(p_value * n_comparisons, 1.0),
                'significant': p_value * n_comparisons < 0.05,
                'direction': 'faster' if np.mean(algo_times) < np.mean(baseline_times) else 'slower'
            }

            # Speedup ratio
            tests['speedup'] = np.mean(baseline_times) / np.mean(algo_times) if np.mean(algo_times) > 0 else float('nan')

        # Iteration comparison
        if len(baseline_iters) > 1 and len(algo_iters) > 1:
            n = min(len(baseline_iters), len(algo_iters))
            t_stat, p_value = stats.ttest_ind(algo_iters[:n], baseline_iters[:n])
            tests['iter_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_value_corrected': min(p_value * n_comparisons, 1.0),
                'significant': p_value * n_comparisons < 0.05,
                'direction': 'fewer' if np.mean(algo_iters) < np.mean(baseline_iters) else 'more'
            }

        test_results[algo] = tests

    return test_results


def generate_ascii_table(stats_dict: Dict[str, Dict[str, Any]]) -> str:
    """Generate ASCII table for terminal display."""
    lines = []
    lines.append("=" * 90)
    lines.append("Test 2 Results: HMM")
    lines.append("=" * 90)

    header = f"{'Algorithm':<25} {'Conv%':>8} {'Success%':>10} {'Final LL':>15} {'Time (s)':>12} {'Iters':>10}"
    lines.append(header)
    lines.append("-" * 90)

    # Sort by final_ll mean descending
    sorted_algos = sorted(
        stats_dict.keys(),
        key=lambda k: stats_dict[k]['final_ll'].get('mean', float('-inf')),
        reverse=True
    )

    for algo in sorted_algos:
        s = stats_dict[algo]
        conv_pct = f"{100*s['convergence_rate']:.1f}%"
        success_pct = f"{100*s['success_rate']:.1f}%"

        ll_mean = s['final_ll']['mean']
        ll_std = s['final_ll']['std']
        ll_str = f"{ll_mean:.1f}±{ll_std:.1f}" if not np.isnan(ll_mean) else "N/A"

        time_mean = s['time']['mean']
        time_std = s['time']['std']
        time_str = f"{time_mean:.2f}±{time_std:.2f}" if not np.isnan(time_mean) else "N/A"

        iter_mean = s['iterations']['mean']
        iter_std = s['iterations']['std']
        iter_str = f"{iter_mean:.0f}±{iter_std:.0f}" if not np.isnan(iter_mean) else "N/A"

        lines.append(f"{algo:<25} {conv_pct:>8} {success_pct:>10} {ll_str:>15} {time_str:>12} {iter_str:>10}")

    lines.append("=" * 90)
    return "\n".join(lines)


def generate_latex_table(stats_dict: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX table for publication."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance on HMM (Test 2)}")
    lines.append(r"\label{tab:test2}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Conv\% & Success\% & Final $\ell$ & Time (s) & Iters \\")
    lines.append(r"\midrule")

    # Find best values for bold formatting
    best_ll = max(s['final_ll']['mean'] for s in stats_dict.values() if not np.isnan(s['final_ll']['mean']))
    best_time = min(s['time']['mean'] for s in stats_dict.values() if not np.isnan(s['time']['mean']))
    best_iter = min(s['iterations']['mean'] for s in stats_dict.values() if not np.isnan(s['iterations']['mean']))

    sorted_algos = sorted(
        stats_dict.keys(),
        key=lambda k: stats_dict[k]['final_ll'].get('mean', float('-inf')),
        reverse=True
    )

    for algo in sorted_algos:
        s = stats_dict[algo]

        # Format algorithm name for LaTeX
        name = algo.replace('_', ' ').replace('gamma', r'$\gamma$=')

        conv_pct = f"{100*s['convergence_rate']:.0f}\\%"
        success_pct = f"{100*s['success_rate']:.0f}\\%"

        ll_mean = s['final_ll']['mean']
        ll_std = s['final_ll']['std']
        if not np.isnan(ll_mean):
            ll_str = f"{ll_mean:.1f} $\\pm$ {ll_std:.1f}"
            if abs(ll_mean - best_ll) < 0.5:
                ll_str = r"\textbf{" + ll_str + "}"
        else:
            ll_str = "N/A"

        time_mean = s['time']['mean']
        time_std = s['time']['std']
        if not np.isnan(time_mean):
            time_str = f"{time_mean:.2f} $\\pm$ {time_std:.2f}"
            if abs(time_mean - best_time) < 0.1:
                time_str = r"\textbf{" + time_str + "}"
        else:
            time_str = "N/A"

        iter_mean = s['iterations']['mean']
        iter_std = s['iterations']['std']
        if not np.isnan(iter_mean):
            iter_str = f"{iter_mean:.0f} $\\pm$ {iter_std:.0f}"
            if abs(iter_mean - best_iter) < 1:
                iter_str = r"\textbf{" + iter_str + "}"
        else:
            iter_str = "N/A"

        lines.append(f"{name} & {conv_pct} & {success_pct} & {ll_str} & {time_str} & {iter_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_csv(stats_dict: Dict[str, Dict[str, Any]]) -> str:
    """Generate CSV for further analysis."""
    lines = []
    lines.append("algorithm,n_total,n_converged,convergence_rate,success_rate,ll_mean,ll_std,time_mean,time_std,iter_mean,iter_std,memory_mean,memory_std")

    for algo, s in stats_dict.items():
        row = [
            algo,
            str(s['n_total']),
            str(s['n_converged']),
            f"{s['convergence_rate']:.4f}",
            f"{s['success_rate']:.4f}",
            f"{s['final_ll']['mean']:.4f}" if not np.isnan(s['final_ll']['mean']) else "",
            f"{s['final_ll']['std']:.4f}" if not np.isnan(s['final_ll']['std']) else "",
            f"{s['time']['mean']:.4f}" if not np.isnan(s['time']['mean']) else "",
            f"{s['time']['std']:.4f}" if not np.isnan(s['time']['std']) else "",
            f"{s['iterations']['mean']:.4f}" if not np.isnan(s['iterations']['mean']) else "",
            f"{s['iterations']['std']:.4f}" if not np.isnan(s['iterations']['std']) else "",
            f"{s['memory']['mean']:.4f}" if not np.isnan(s['memory']['mean']) else "",
            f"{s['memory']['std']:.4f}" if not np.isnan(s['memory']['std']) else "",
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def save_analysis(
    stats_dict: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Dict[str, Any]],
    results_dir: str
) -> None:
    """Save all analysis outputs."""
    output_dir = os.path.join(results_dir, 'test_2_hmm', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # ASCII summary
    ascii_table = generate_ascii_table(stats_dict)
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(ascii_table)
        f.write("\n\n")
        f.write("Statistical Tests (vs standard_em):\n")
        f.write("-" * 50 + "\n")
        for algo, tests in test_results.items():
            f.write(f"\n{algo}:\n")
            for test_name, test_data in tests.items():
                if isinstance(test_data, dict):
                    f.write(f"  {test_name}: p={test_data.get('p_value', 'N/A'):.4f} ({test_data.get('direction', '')})\n")
                else:
                    f.write(f"  {test_name}: {test_data:.4f}\n")

    # LaTeX table
    latex_table = generate_latex_table(stats_dict)
    with open(os.path.join(output_dir, 'table.tex'), 'w') as f:
        f.write(latex_table)

    # CSV
    csv_data = generate_csv(stats_dict)
    with open(os.path.join(output_dir, 'results.csv'), 'w') as f:
        f.write(csv_data)

    print(f"Analysis saved to {output_dir}/")


def run_analysis(results_dir: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run full analysis of Test 2 results.

    Args:
        results_dir: Base results directory
        verbose: Whether to print results

    Returns:
        Dict with stats_dict and test_results
    """
    # Load results
    results = load_test_2_results(results_dir)

    if not results:
        print("No results found for Test 2")
        return {}

    if verbose:
        print(f"Loaded {len(results)} results")

    # Compute statistics
    stats_dict = compute_aggregate_statistics(results)

    # Perform statistical tests
    test_results = perform_statistical_tests(stats_dict)

    if verbose:
        print(generate_ascii_table(stats_dict))
        print("\nStatistical Tests (vs standard_em):")
        print("-" * 50)
        for algo, tests in test_results.items():
            print(f"\n{algo}:")
            for test_name, test_data in tests.items():
                if isinstance(test_data, dict):
                    sig = "*" if test_data.get('significant', False) else ""
                    print(f"  {test_name}: p={test_data.get('p_value', 'N/A'):.4f}{sig} ({test_data.get('direction', '')})")
                else:
                    print(f"  {test_name}: {test_data:.4f}")

    return {
        'stats_dict': stats_dict,
        'test_results': test_results,
        'n_results': len(results)
    }


# ============================================================
# Unit Tests
# ============================================================

def test_statistics_computation():
    """Test aggregate statistics computation."""
    mock_results = [
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -800.0, 'elapsed_time': 5.0, 'n_iterations': 50, 'peak_memory_mb': 100.0, 'error': None},
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -795.0, 'elapsed_time': 4.8, 'n_iterations': 48, 'peak_memory_mb': 102.0, 'error': None},
        {'algorithm': 'squarem', 'algorithm_params': {}, 'converged': True,
         'final_ll': -790.0, 'elapsed_time': 3.0, 'n_iterations': 25, 'peak_memory_mb': 110.0, 'error': None},
    ]

    stats = compute_aggregate_statistics(mock_results)

    assert 'standard_em' in stats, "Missing standard_em"
    assert 'squarem' in stats, "Missing squarem"
    assert stats['standard_em']['n_total'] == 2, "Wrong count"
    assert stats['standard_em']['convergence_rate'] == 1.0, "Wrong convergence rate"

    print("  PASSED")


def test_statistical_tests():
    """Test statistical test computation."""
    mock_stats = {
        'standard_em': {
            'final_ll': {'values': [-800, -795, -802, -798]},
            'time': {'values': [5.0, 4.8, 5.2, 4.9]},
            'iterations': {'values': [50, 48, 52, 49]}
        },
        'squarem': {
            'final_ll': {'values': [-790, -792, -788, -791]},
            'time': {'values': [3.0, 3.1, 2.9, 3.0]},
            'iterations': {'values': [25, 26, 24, 25]}
        }
    }

    test_results = perform_statistical_tests(mock_stats)

    assert 'squarem' in test_results, "Missing squarem in test results"
    assert 'll_ttest' in test_results['squarem'], "Missing ll_ttest"
    assert 'speedup' in test_results['squarem'], "Missing speedup"

    print("  PASSED")


def test_table_generation():
    """Test table generation."""
    mock_stats = {
        'standard_em': {
            'n_total': 10, 'n_converged': 9, 'n_errors': 0,
            'convergence_rate': 0.9, 'success_rate': 0.8,
            'final_ll': {'mean': -800.0, 'std': 5.0},
            'time': {'mean': 5.0, 'std': 0.5},
            'iterations': {'mean': 50.0, 'std': 5.0},
            'memory': {'mean': 100.0, 'std': 10.0}
        }
    }

    ascii_table = generate_ascii_table(mock_stats)
    assert "standard_em" in ascii_table, "Missing algorithm in ASCII table"

    latex_table = generate_latex_table(mock_stats)
    assert r"\begin{table}" in latex_table, "Missing LaTeX table start"

    csv_data = generate_csv(mock_stats)
    assert "standard_em" in csv_data, "Missing algorithm in CSV"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running analyze_test_2.py unit tests")
    print("=" * 60)

    print("Testing statistics computation...")
    test_statistics_computation()

    print("Testing statistical tests...")
    test_statistical_tests()

    print("Testing table generation...")
    test_table_generation()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Test 2 Results")
    parser.add_argument('--test', action='store_true', help="Run unit tests only")
    parser.add_argument('--results-dir', type=str, default=None, help="Results directory")
    parser.add_argument('--save', action='store_true', help="Save analysis to files")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    else:
        analysis = run_analysis(args.results_dir)
        if args.save and analysis:
            results_dir = args.results_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results'
            )
            save_analysis(analysis['stats_dict'], analysis['test_results'], results_dir)
