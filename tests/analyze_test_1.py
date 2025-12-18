"""
Test 1 Analysis Script

Analyzes results from Test 1 (High-Dimensional GMM) experiment.
Computes aggregate statistics, performs statistical tests,
generates tables and visualizations.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookahead_em_evaluation.utils.logging_utils import ExperimentLogger
from lookahead_em_evaluation.utils.metrics import cohens_d, bootstrap_ci
from lookahead_em_evaluation.utils.plotting import (
    plot_convergence, plot_time_comparison, plot_success_rates,
    plot_memory_usage, save_figure
)

import matplotlib.pyplot as plt


def load_test_1_results(results_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load all results from Test 1.

    Args:
        results_dir: Path to results directory

    Returns:
        List of run result dictionaries
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', 'test_1_high_d_gmm'
        )

    logger = ExperimentLogger(results_dir)
    return logger.load_all()


def compute_aggregate_statistics(
    results: List[Dict[str, Any]],
    baseline_ll: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute aggregate statistics for each algorithm.

    Args:
        results: List of run result dictionaries
        baseline_ll: Best log-likelihood for success rate calculation
                    (if None, uses max across all runs)

    Returns:
        Dictionary mapping algorithm names to their statistics
    """
    # Group by algorithm
    by_algo = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        params = r.get('algorithm_params', {})

        # Create unique key
        if params and params.get('gamma'):
            gamma = params.get('gamma')
            key = f"{algo}_gamma_{gamma}"
        else:
            key = algo

        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    # Find best LL if not provided
    if baseline_ll is None:
        all_lls = [r['final_ll'] for r in results
                  if r.get('final_ll', float('-inf')) > float('-inf')]
        baseline_ll = max(all_lls) if all_lls else 0

    # Compute statistics
    stats_dict = {}

    for key, algo_results in by_algo.items():
        n_total = len(algo_results)
        successful = [r for r in algo_results if r.get('error') is None]
        n_successful = len(successful)
        n_errors = n_total - n_successful

        # Extract metrics from successful runs
        final_lls = np.array([r['final_ll'] for r in successful
                            if r.get('final_ll', float('-inf')) > float('-inf')])
        times = np.array([r['elapsed_time'] for r in successful
                         if r.get('elapsed_time', 0) > 0])
        iters = np.array([r['n_iterations'] for r in successful
                         if r.get('n_iterations', 0) > 0])
        memories = np.array([r['peak_memory_mb'] for r in successful
                            if r.get('peak_memory_mb', 0) > 0])

        # Convergence count
        n_converged = sum(1 for r in successful if r.get('converged', False))

        # Success rate (reaching within 0.5 of best)
        n_success = sum(1 for ll in final_lls if ll >= baseline_ll - 0.5)

        stats_dict[key] = {
            # Counts
            'n_total': n_total,
            'n_successful': n_successful,
            'n_errors': n_errors,
            'n_converged': n_converged,
            'n_success': n_success,

            # Rates
            'convergence_rate': n_converged / n_total if n_total > 0 else 0,
            'success_rate': n_success / n_successful if n_successful > 0 else 0,

            # Log-likelihood statistics
            'final_ll_mean': np.mean(final_lls) if len(final_lls) > 0 else np.nan,
            'final_ll_std': np.std(final_lls) if len(final_lls) > 0 else np.nan,
            'final_ll_min': np.min(final_lls) if len(final_lls) > 0 else np.nan,
            'final_ll_max': np.max(final_lls) if len(final_lls) > 0 else np.nan,
            'final_ll_values': final_lls,

            # Time statistics
            'time_mean': np.mean(times) if len(times) > 0 else np.nan,
            'time_std': np.std(times) if len(times) > 0 else np.nan,
            'time_median': np.median(times) if len(times) > 0 else np.nan,
            'time_values': times,

            # Iteration statistics
            'iterations_mean': np.mean(iters) if len(iters) > 0 else np.nan,
            'iterations_std': np.std(iters) if len(iters) > 0 else np.nan,
            'iterations_median': np.median(iters) if len(iters) > 0 else np.nan,
            'iterations_values': iters,

            # Memory statistics
            'memory_mean': np.mean(memories) if len(memories) > 0 else np.nan,
            'memory_std': np.std(memories) if len(memories) > 0 else np.nan,
            'memory_max': np.max(memories) if len(memories) > 0 else np.nan,
            'memory_values': memories,

            # Raw results for further analysis
            'results': successful,
        }

    return stats_dict


def perform_statistical_tests(
    stats_dict: Dict[str, Dict[str, Any]],
    baseline_algo: str = 'standard_em'
) -> Dict[str, Dict[str, Any]]:
    """
    Perform statistical tests comparing each algorithm to baseline.

    Uses paired t-tests with Bonferroni correction for multiple comparisons.

    Args:
        stats_dict: Statistics dictionary from compute_aggregate_statistics
        baseline_algo: Name of baseline algorithm for comparison

    Returns:
        Dictionary with test results for each algorithm
    """
    if baseline_algo not in stats_dict:
        return {}

    baseline = stats_dict[baseline_algo]
    n_comparisons = len(stats_dict) - 1  # Exclude baseline

    test_results = {}

    for algo, algo_stats in stats_dict.items():
        if algo == baseline_algo:
            continue

        test_results[algo] = {}

        # Compare log-likelihoods
        baseline_lls = baseline['final_ll_values']
        algo_lls = algo_stats['final_ll_values']

        if len(baseline_lls) > 1 and len(algo_lls) > 1:
            # Use independent t-test if sample sizes differ
            min_len = min(len(baseline_lls), len(algo_lls))
            if len(baseline_lls) == len(algo_lls):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(algo_lls, baseline_lls)
            else:
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(algo_lls, baseline_lls)

            # Bonferroni correction
            p_corrected = min(p_value * n_comparisons, 1.0)

            # Effect size
            d = cohens_d(algo_lls, baseline_lls)

            test_results[algo]['ll_comparison'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_corrected': p_corrected,
                'cohens_d': d,
                'significant': p_corrected < 0.05,
                'better': np.mean(algo_lls) > np.mean(baseline_lls)
            }

        # Compare iteration counts
        baseline_iters = baseline['iterations_values']
        algo_iters = algo_stats['iterations_values']

        if len(baseline_iters) > 1 and len(algo_iters) > 1:
            if len(baseline_iters) == len(algo_iters):
                t_stat, p_value = stats.ttest_rel(algo_iters, baseline_iters)
            else:
                t_stat, p_value = stats.ttest_ind(algo_iters, baseline_iters)

            p_corrected = min(p_value * n_comparisons, 1.0)
            d = cohens_d(algo_iters, baseline_iters)

            test_results[algo]['iterations_comparison'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_corrected': p_corrected,
                'cohens_d': d,
                'significant': p_corrected < 0.05,
                'better': np.mean(algo_iters) < np.mean(baseline_iters)  # Fewer is better
            }

        # Compare times
        baseline_times = baseline['time_values']
        algo_times = algo_stats['time_values']

        if len(baseline_times) > 1 and len(algo_times) > 1:
            if len(baseline_times) == len(algo_times):
                t_stat, p_value = stats.ttest_rel(algo_times, baseline_times)
            else:
                t_stat, p_value = stats.ttest_ind(algo_times, baseline_times)

            p_corrected = min(p_value * n_comparisons, 1.0)
            d = cohens_d(algo_times, baseline_times)

            test_results[algo]['time_comparison'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_corrected': p_corrected,
                'cohens_d': d,
                'significant': p_corrected < 0.05,
                'better': np.mean(algo_times) < np.mean(baseline_times)  # Faster is better
            }

    return test_results


def generate_ascii_table(
    stats_dict: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    Generate ASCII table of results.

    Args:
        stats_dict: Statistics dictionary
        test_results: Optional statistical test results

    Returns:
        Formatted ASCII table string
    """
    # Sort algorithms by final_ll_mean descending
    sorted_algos = sorted(
        stats_dict.keys(),
        key=lambda k: stats_dict[k].get('final_ll_mean', float('-inf')),
        reverse=True
    )

    # Build table
    lines = []
    lines.append("=" * 100)
    lines.append("Test 1: High-Dimensional GMM Results")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Algorithm':<30} {'Conv%':>8} {'Success%':>9} {'Final LL':>15} {'Time (s)':>12} {'Iters':>10}"
    lines.append(header)
    lines.append("-" * 100)

    for algo in sorted_algos:
        s = stats_dict[algo]

        conv_pct = f"{100*s['convergence_rate']:.1f}%"
        succ_pct = f"{100*s['success_rate']:.1f}%"

        if not np.isnan(s['final_ll_mean']):
            ll_str = f"{s['final_ll_mean']:.2f}±{s['final_ll_std']:.2f}"
        else:
            ll_str = "N/A"

        if not np.isnan(s['time_mean']):
            time_str = f"{s['time_mean']:.2f}±{s['time_std']:.2f}"
        else:
            time_str = "N/A"

        if not np.isnan(s['iterations_mean']):
            iter_str = f"{s['iterations_mean']:.0f}±{s['iterations_std']:.0f}"
        else:
            iter_str = "N/A"

        # Add significance markers
        sig_marker = ""
        if test_results and algo in test_results:
            ll_test = test_results[algo].get('ll_comparison', {})
            if ll_test.get('significant') and ll_test.get('better'):
                sig_marker = " *"

        lines.append(f"{algo:<30} {conv_pct:>8} {succ_pct:>9} {ll_str:>15} {time_str:>12} {iter_str:>10}{sig_marker}")

    lines.append("-" * 100)
    lines.append("* indicates significantly better than StandardEM (p < 0.05, Bonferroni corrected)")
    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_table(
    stats_dict: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Dict[str, Any]] = None
) -> str:
    """
    Generate LaTeX table of results.

    Args:
        stats_dict: Statistics dictionary
        test_results: Optional statistical test results

    Returns:
        LaTeX table string
    """
    # Sort algorithms by final_ll_mean descending
    sorted_algos = sorted(
        stats_dict.keys(),
        key=lambda k: stats_dict[k].get('final_ll_mean', float('-inf')),
        reverse=True
    )

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Test 1: High-Dimensional GMM Results}")
    lines.append("\\label{tab:test1_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Algorithm & Conv.\\% & Success\\% & Final LL & Time (s) & Iterations \\\\")
    lines.append("\\midrule")

    for algo in sorted_algos:
        s = stats_dict[algo]

        # Format algorithm name for LaTeX
        algo_latex = algo.replace('_', '\\_')

        conv_pct = f"{100*s['convergence_rate']:.1f}"
        succ_pct = f"{100*s['success_rate']:.1f}"

        if not np.isnan(s['final_ll_mean']):
            ll_str = f"${s['final_ll_mean']:.2f} \\pm {s['final_ll_std']:.2f}$"
        else:
            ll_str = "N/A"

        if not np.isnan(s['time_mean']):
            time_str = f"${s['time_mean']:.2f} \\pm {s['time_std']:.2f}$"
        else:
            time_str = "N/A"

        if not np.isnan(s['iterations_mean']):
            iter_str = f"${s['iterations_mean']:.0f} \\pm {s['iterations_std']:.0f}$"
        else:
            iter_str = "N/A"

        # Bold if best
        if algo == sorted_algos[0]:
            ll_str = f"\\textbf{{{ll_str}}}"

        lines.append(f"{algo_latex} & {conv_pct} & {succ_pct} & {ll_str} & {time_str} & {iter_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_csv_table(stats_dict: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate CSV table of results.

    Args:
        stats_dict: Statistics dictionary

    Returns:
        CSV string
    """
    lines = []
    lines.append("algorithm,n_total,n_converged,convergence_rate,success_rate,final_ll_mean,final_ll_std,time_mean,time_std,iterations_mean,iterations_std,memory_mean,memory_std")

    for algo, s in stats_dict.items():
        row = [
            algo,
            str(s['n_total']),
            str(s['n_converged']),
            f"{s['convergence_rate']:.4f}",
            f"{s['success_rate']:.4f}",
            f"{s['final_ll_mean']:.4f}" if not np.isnan(s['final_ll_mean']) else "",
            f"{s['final_ll_std']:.4f}" if not np.isnan(s['final_ll_std']) else "",
            f"{s['time_mean']:.4f}" if not np.isnan(s['time_mean']) else "",
            f"{s['time_std']:.4f}" if not np.isnan(s['time_std']) else "",
            f"{s['iterations_mean']:.4f}" if not np.isnan(s['iterations_mean']) else "",
            f"{s['iterations_std']:.4f}" if not np.isnan(s['iterations_std']) else "",
            f"{s['memory_mean']:.4f}" if not np.isnan(s['memory_mean']) else "",
            f"{s['memory_std']:.4f}" if not np.isnan(s['memory_std']) else "",
        ]
        lines.append(",".join(row))

    return "\n".join(lines)


def generate_visualizations(
    stats_dict: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Generate and save all visualizations.

    Args:
        stats_dict: Statistics dictionary
        results: Raw results list
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get algorithm order by performance
    sorted_algos = sorted(
        stats_dict.keys(),
        key=lambda k: stats_dict[k].get('final_ll_mean', float('-inf')),
        reverse=True
    )

    # 1. Convergence curves (if we have ll_history)
    # Group results by algorithm
    by_algo = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        params = r.get('algorithm_params', {})
        if params and params.get('gamma'):
            key = f"{algo}_gamma_{params.get('gamma')}"
        else:
            key = algo

        if key not in by_algo:
            by_algo[key] = []
        by_algo[key].append(r)

    # Find max iterations for padding
    max_iters = 0
    for algo_results in by_algo.values():
        for r in algo_results:
            ll_hist = r.get('ll_history', [])
            if len(ll_hist) > max_iters:
                max_iters = len(ll_hist)

    if max_iters > 0:
        # Build convergence data
        convergence_data = {}
        for algo, algo_results in by_algo.items():
            histories = []
            for r in algo_results:
                ll_hist = r.get('ll_history', [])
                if len(ll_hist) > 0:
                    # Pad to max length
                    padded = np.full(max_iters, ll_hist[-1])
                    padded[:len(ll_hist)] = ll_hist
                    histories.append(padded)

            if histories:
                histories = np.array(histories)
                convergence_data[algo] = {
                    'mean': np.mean(histories, axis=0),
                    'std': np.std(histories, axis=0)
                }

        if convergence_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            for algo in sorted_algos:
                if algo in convergence_data:
                    mean = convergence_data[algo]['mean']
                    std = convergence_data[algo]['std']
                    iters = np.arange(len(mean))
                    ax.plot(iters, mean, label=algo)
                    ax.fill_between(iters, mean - std, mean + std, alpha=0.2)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Log-Likelihood')
            ax.set_title('Test 1: Convergence Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_figure(fig, os.path.join(output_dir, 'convergence_curves.pdf'))
            save_figure(fig, os.path.join(output_dir, 'convergence_curves.png'))
            plt.close(fig)

    # 2. Time comparison (box plots)
    time_data = {algo: stats_dict[algo]['time_values'] for algo in sorted_algos
                 if len(stats_dict[algo]['time_values']) > 0}

    if time_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(time_data.keys())
        data = [time_data[k] for k in labels]
        ax.boxplot(data, labels=[l.replace('_', '\n') for l in labels])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Test 1: Algorithm Runtime Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'time_comparison.pdf'))
        save_figure(fig, os.path.join(output_dir, 'time_comparison.png'))
        plt.close(fig)

    # 3. Success rate (bar chart)
    success_data = {algo: stats_dict[algo]['success_rate'] for algo in sorted_algos}

    if success_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(success_data.keys())
        rates = [success_data[k] * 100 for k in labels]
        bars = ax.bar(range(len(labels)), rates, color='steelblue')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Test 1: Success Rates (% reaching within 0.5 of best LL)')
        ax.set_ylim(0, 105)

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'success_rates.pdf'))
        save_figure(fig, os.path.join(output_dir, 'success_rates.png'))
        plt.close(fig)

    # 4. Memory usage (bar chart)
    memory_data = {algo: stats_dict[algo]['memory_mean'] for algo in sorted_algos
                   if not np.isnan(stats_dict[algo]['memory_mean'])}

    if memory_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(memory_data.keys())
        memories = [memory_data[k] for k in labels]
        errors = [stats_dict[k]['memory_std'] for k in labels]

        bars = ax.bar(range(len(labels)), memories, yerr=errors, capsize=5, color='forestgreen')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=45, ha='right')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Test 1: Memory Usage Comparison')

        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'memory_usage.pdf'))
        save_figure(fig, os.path.join(output_dir, 'memory_usage.png'))
        plt.close(fig)

    print(f"Figures saved to {output_dir}/")


def analyze_test_1(
    results_dir: str = None,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Run complete analysis of Test 1 results.

    Args:
        results_dir: Path to results directory
        output_dir: Path to output directory (defaults to results_dir/figures)
        verbose: Whether to print results

    Returns:
        Tuple of (statistics_dict, test_results_dict)
    """
    # Load results
    if verbose:
        print("Loading Test 1 results...")

    results = load_test_1_results(results_dir)

    if not results:
        print("No results found!")
        return {}, {}

    if verbose:
        print(f"Loaded {len(results)} run results")

    # Compute statistics
    if verbose:
        print("Computing aggregate statistics...")

    stats_dict = compute_aggregate_statistics(results)

    # Perform statistical tests
    if verbose:
        print("Performing statistical tests...")

    test_results = perform_statistical_tests(stats_dict)

    # Generate tables
    if verbose:
        ascii_table = generate_ascii_table(stats_dict, test_results)
        print("\n" + ascii_table)

    # Set output directory
    if output_dir is None:
        if results_dir:
            output_dir = os.path.join(results_dir, 'figures')
        else:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results', 'test_1_high_d_gmm', 'figures'
            )

    # Generate visualizations
    if verbose:
        print(f"\nGenerating visualizations...")

    generate_visualizations(stats_dict, results, output_dir)

    # Save tables
    results_base = os.path.dirname(output_dir) if output_dir.endswith('figures') else output_dir

    # Save ASCII summary
    with open(os.path.join(results_base, 'summary.txt'), 'w') as f:
        f.write(generate_ascii_table(stats_dict, test_results))

    # Save LaTeX table
    with open(os.path.join(results_base, 'table.tex'), 'w') as f:
        f.write(generate_latex_table(stats_dict, test_results))

    # Save CSV
    with open(os.path.join(results_base, 'results.csv'), 'w') as f:
        f.write(generate_csv_table(stats_dict))

    if verbose:
        print(f"\nSummary saved to {results_base}/summary.txt")
        print(f"LaTeX table saved to {results_base}/table.tex")
        print(f"CSV saved to {results_base}/results.csv")

    return stats_dict, test_results


# ============================================================
# Unit Tests
# ============================================================

def test_compute_statistics():
    """Test aggregate statistics computation."""
    # Create mock results
    results = [
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -1000.0, 'elapsed_time': 10.0, 'n_iterations': 100,
         'peak_memory_mb': 100.0, 'error': None},
        {'algorithm': 'standard_em', 'algorithm_params': {}, 'converged': True,
         'final_ll': -990.0, 'elapsed_time': 9.0, 'n_iterations': 95,
         'peak_memory_mb': 105.0, 'error': None},
        {'algorithm': 'lookahead_em', 'algorithm_params': {'gamma': 0.5}, 'converged': True,
         'final_ll': -985.0, 'elapsed_time': 12.0, 'n_iterations': 50,
         'peak_memory_mb': 120.0, 'error': None},
    ]

    stats = compute_aggregate_statistics(results)

    assert 'standard_em' in stats, "Missing standard_em"
    assert 'lookahead_em_gamma_0.5' in stats, "Missing lookahead_em"

    assert stats['standard_em']['n_total'] == 2
    assert stats['standard_em']['convergence_rate'] == 1.0
    assert abs(stats['standard_em']['final_ll_mean'] - (-995.0)) < 0.01

    print("  PASSED")


def test_statistical_tests():
    """Test statistical test computation."""
    # Create mock results with different performance
    np.random.seed(42)
    results = []

    # StandardEM: worse performance
    for i in range(10):
        results.append({
            'algorithm': 'standard_em', 'algorithm_params': {},
            'converged': True, 'final_ll': -1000 + np.random.randn() * 5,
            'elapsed_time': 10 + np.random.randn(), 'n_iterations': 100 + int(np.random.randn() * 10),
            'peak_memory_mb': 100, 'error': None
        })

    # LookaheadEM: better performance
    for i in range(10):
        results.append({
            'algorithm': 'lookahead_em', 'algorithm_params': {'gamma': 0.5},
            'converged': True, 'final_ll': -980 + np.random.randn() * 5,
            'elapsed_time': 12 + np.random.randn(), 'n_iterations': 50 + int(np.random.randn() * 10),
            'peak_memory_mb': 120, 'error': None
        })

    stats = compute_aggregate_statistics(results)
    test_results = perform_statistical_tests(stats)

    assert 'lookahead_em_gamma_0.5' in test_results, "Missing test results"
    assert 'll_comparison' in test_results['lookahead_em_gamma_0.5'], "Missing LL comparison"

    print("  PASSED")


def test_table_generation():
    """Test table generation functions."""
    stats = {
        'standard_em': {
            'n_total': 10, 'n_converged': 10, 'n_errors': 0, 'n_success': 8,
            'convergence_rate': 1.0, 'success_rate': 0.8,
            'final_ll_mean': -1000.0, 'final_ll_std': 5.0,
            'final_ll_min': -1010, 'final_ll_max': -990,
            'time_mean': 10.0, 'time_std': 1.0,
            'iterations_mean': 100, 'iterations_std': 10,
            'memory_mean': 100, 'memory_std': 5,
        }
    }

    ascii_table = generate_ascii_table(stats)
    assert 'standard_em' in ascii_table, "Missing algorithm in ASCII table"

    latex_table = generate_latex_table(stats)
    assert '\\begin{table}' in latex_table, "Missing LaTeX table start"

    csv_table = generate_csv_table(stats)
    assert 'standard_em' in csv_table, "Missing algorithm in CSV"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running analyze_test_1.py unit tests")
    print("=" * 60)

    print("Testing compute_statistics...")
    test_compute_statistics()

    print("Testing statistical_tests...")
    test_statistical_tests()

    print("Testing table_generation...")
    test_table_generation()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 1 Analysis")
    parser.add_argument('--test', action='store_true', help="Run unit tests only")
    parser.add_argument('--results-dir', type=str, help="Path to results directory")
    parser.add_argument('--output-dir', type=str, help="Path to output directory")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    else:
        analyze_test_1(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            verbose=True
        )
