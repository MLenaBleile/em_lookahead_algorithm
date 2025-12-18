"""
Cross-Test Comparison Analysis

Comprehensive analysis comparing algorithm performance across all tests.
Generates summary tables, heatmaps, and statistical meta-analysis.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings

from lookahead_em_evaluation.utils.logging_utils import ExperimentLogger
from lookahead_em_evaluation.utils.metrics import cohens_d

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Test configuration
TESTS = {
    'test_1_high_d_gmm': {
        'name': 'Test 1: High-D GMM',
        'short_name': 'High-D GMM',
        'model': 'GMM',
        'primary_metric': 'iterations',  # Iterations to converge
    },
    'test_2_hmm': {
        'name': 'Test 2: HMM',
        'short_name': 'HMM',
        'model': 'HMM',
        'primary_metric': 'success_rate',
    },
    'test_3_fa': {
        'name': 'Test 3: Factor Analysis',
        'short_name': 'FA',
        'model': 'FA',
        'primary_metric': 'iterations',
    },
    'test_4_gmm_init': {
        'name': 'Test 4: GMM Init Study',
        'short_name': 'GMM Init',
        'model': 'GMM',
        'primary_metric': 'success_rate',
    },
    'test_5_stress': {
        'name': 'Test 5: Stress Test',
        'short_name': 'Stress',
        'model': 'GMM',
        'primary_metric': 'time_mean',
    },
}


def load_all_test_results(results_base_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load results from all available tests.

    Args:
        results_base_dir: Base directory containing test result subdirectories

    Returns:
        Dictionary mapping test_id to list of run results
    """
    if results_base_dir is None:
        results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    all_results = {}

    for test_id in TESTS.keys():
        test_dir = os.path.join(results_base_dir, test_id)
        if os.path.exists(test_dir):
            try:
                logger = ExperimentLogger(test_dir)
                results = logger.load_all()
                if results:
                    all_results[test_id] = results
                    print(f"Loaded {len(results)} results from {test_id}")
            except Exception as e:
                print(f"Error loading {test_id}: {e}")

    return all_results


def compute_test_statistics(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics for a single test's results.

    Args:
        results: List of run results

    Returns:
        Dictionary mapping algorithm name to statistics
    """
    # Group by algorithm
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

    # Compute statistics
    stats_dict = {}

    for key, algo_results in by_algo.items():
        successful = [r for r in algo_results if r.get('error') is None]

        final_lls = np.array([r['final_ll'] for r in successful
                            if r.get('final_ll', float('-inf')) > float('-inf')])
        times = np.array([r['elapsed_time'] for r in successful
                         if r.get('elapsed_time', 0) > 0])
        iters = np.array([r['n_iterations'] for r in successful
                         if r.get('n_iterations', 0) > 0])

        n_converged = sum(1 for r in successful if r.get('converged', False))
        n_total = len(algo_results)

        # Find best LL for success calculation
        best_ll = max(final_lls) if len(final_lls) > 0 else float('-inf')
        n_success = sum(1 for ll in final_lls if ll >= best_ll - 0.5)

        stats_dict[key] = {
            'n_total': n_total,
            'n_converged': n_converged,
            'convergence_rate': n_converged / n_total if n_total > 0 else 0,
            'success_rate': n_success / len(successful) if len(successful) > 0 else 0,
            'final_ll_mean': np.mean(final_lls) if len(final_lls) > 0 else np.nan,
            'final_ll_std': np.std(final_lls) if len(final_lls) > 0 else np.nan,
            'time_mean': np.mean(times) if len(times) > 0 else np.nan,
            'time_std': np.std(times) if len(times) > 0 else np.nan,
            'iterations_mean': np.mean(iters) if len(iters) > 0 else np.nan,
            'iterations_std': np.std(iters) if len(iters) > 0 else np.nan,
            'time_values': times,
            'iterations_values': iters,
        }

    return stats_dict


def generate_summary_table(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str = 'iterations_mean'
) -> Tuple[str, str]:
    """
    Generate Table 2: Summary Across All Tests.

    Args:
        all_stats: Dictionary mapping test_id -> algorithm -> statistics
        metric: Which metric to display ('iterations_mean', 'time_mean', 'success_rate')

    Returns:
        Tuple of (ASCII table, LaTeX table)
    """
    # Collect all algorithms across all tests
    all_algos = set()
    for test_stats in all_stats.values():
        all_algos.update(test_stats.keys())
    all_algos = sorted(all_algos)

    # Collect available tests
    test_ids = sorted(all_stats.keys())

    # Build ASCII table
    ascii_lines = []
    ascii_lines.append("=" * (30 + 15 * len(all_algos)))
    ascii_lines.append("Table 2: Summary Across All Tests")
    ascii_lines.append(f"Metric: {metric}")
    ascii_lines.append("=" * (30 + 15 * len(all_algos)))

    # Header
    header = f"{'Test':<30}"
    for algo in all_algos:
        header += f"{algo[:14]:>15}"
    ascii_lines.append(header)
    ascii_lines.append("-" * (30 + 15 * len(all_algos)))

    # Data rows
    for test_id in test_ids:
        test_stats = all_stats[test_id]
        row = f"{TESTS[test_id]['short_name']:<30}"

        # Find best value for this test (lower is better for time/iters)
        values = {}
        for algo in all_algos:
            if algo in test_stats:
                val = test_stats[algo].get(metric, np.nan)
                if not np.isnan(val):
                    values[algo] = val

        if values:
            if metric in ['iterations_mean', 'time_mean']:
                best_val = min(values.values())
            else:  # success_rate, convergence_rate
                best_val = max(values.values())
        else:
            best_val = None

        for algo in all_algos:
            if algo in test_stats:
                val = test_stats[algo].get(metric, np.nan)
                if not np.isnan(val):
                    if metric == 'success_rate':
                        val_str = f"{100*val:.1f}%"
                    elif metric == 'time_mean':
                        val_str = f"{val:.2f}s"
                    else:
                        val_str = f"{val:.1f}"

                    # Mark winner
                    if best_val is not None and val == best_val:
                        val_str = f"*{val_str}"

                    row += f"{val_str:>15}"
                else:
                    row += f"{'N/A':>15}"
            else:
                row += f"{'--':>15}"

        ascii_lines.append(row)

    ascii_lines.append("-" * (30 + 15 * len(all_algos)))
    ascii_lines.append("* indicates best performer for that test")
    ascii_lines.append("=" * (30 + 15 * len(all_algos)))

    ascii_table = "\n".join(ascii_lines)

    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{Summary Across All Tests (Metric: {metric.replace('_', ' ')})}}")
    latex_lines.append("\\label{tab:cross_test_summary}")

    col_spec = "l" + "c" * len(all_algos)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    # Header
    header = "Test"
    for algo in all_algos:
        algo_latex = algo.replace('_', '\\_')[:14]
        header += f" & {algo_latex}"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for test_id in test_ids:
        test_stats = all_stats[test_id]
        test_name = TESTS[test_id]['short_name'].replace('_', '\\_')
        row = test_name

        values = {}
        for algo in all_algos:
            if algo in test_stats:
                val = test_stats[algo].get(metric, np.nan)
                if not np.isnan(val):
                    values[algo] = val

        if values:
            if metric in ['iterations_mean', 'time_mean']:
                best_val = min(values.values())
            else:
                best_val = max(values.values())
        else:
            best_val = None

        for algo in all_algos:
            if algo in test_stats:
                val = test_stats[algo].get(metric, np.nan)
                if not np.isnan(val):
                    if metric == 'success_rate':
                        val_str = f"{100*val:.1f}\\%"
                    elif metric == 'time_mean':
                        val_str = f"{val:.2f}"
                    else:
                        val_str = f"{val:.1f}"

                    if best_val is not None and val == best_val:
                        val_str = f"\\textbf{{{val_str}}}"

                    row += f" & {val_str}"
                else:
                    row += " & N/A"
            else:
                row += " & --"

        row += " \\\\"
        latex_lines.append(row)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    return ascii_table, latex_table


def compute_overall_statistics(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute overall statistics across all tests.

    Args:
        all_stats: Dictionary mapping test_id -> algorithm -> statistics

    Returns:
        Dictionary with overall statistics per algorithm
    """
    # Collect all algorithms
    all_algos = set()
    for test_stats in all_stats.values():
        all_algos.update(test_stats.keys())

    overall = {}

    for algo in all_algos:
        # Collect metrics across tests
        n_tests = 0
        n_wins_iter = 0
        n_wins_time = 0
        n_wins_success = 0
        speedups = []
        iter_reductions = []

        for test_id, test_stats in all_stats.items():
            if algo not in test_stats:
                continue

            n_tests += 1
            algo_stats = test_stats[algo]

            # Check for wins
            if 'standard_em' in test_stats:
                baseline = test_stats['standard_em']

                # Iteration comparison
                if not np.isnan(algo_stats['iterations_mean']) and not np.isnan(baseline['iterations_mean']):
                    if algo_stats['iterations_mean'] < baseline['iterations_mean']:
                        iter_reduction = 1 - algo_stats['iterations_mean'] / baseline['iterations_mean']
                        iter_reductions.append(iter_reduction)

                    # Check if best
                    all_iters = [s['iterations_mean'] for s in test_stats.values()
                                if not np.isnan(s.get('iterations_mean', np.nan))]
                    if all_iters and algo_stats['iterations_mean'] == min(all_iters):
                        n_wins_iter += 1

                # Time comparison
                if not np.isnan(algo_stats['time_mean']) and not np.isnan(baseline['time_mean']):
                    if algo_stats['time_mean'] < baseline['time_mean']:
                        speedup = baseline['time_mean'] / algo_stats['time_mean']
                        speedups.append(speedup)

                    all_times = [s['time_mean'] for s in test_stats.values()
                                if not np.isnan(s.get('time_mean', np.nan))]
                    if all_times and algo_stats['time_mean'] == min(all_times):
                        n_wins_time += 1

                # Success rate comparison
                if not np.isnan(algo_stats['success_rate']):
                    all_rates = [s['success_rate'] for s in test_stats.values()
                                if not np.isnan(s.get('success_rate', np.nan))]
                    if all_rates and algo_stats['success_rate'] == max(all_rates):
                        n_wins_success += 1

        overall[algo] = {
            'n_tests': n_tests,
            'n_wins_iterations': n_wins_iter,
            'n_wins_time': n_wins_time,
            'n_wins_success': n_wins_success,
            'avg_speedup': np.mean(speedups) if speedups else np.nan,
            'avg_iter_reduction': np.mean(iter_reductions) if iter_reductions else np.nan,
        }

    return overall


def generate_heatmap(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str = 'iterations_mean',
    output_path: str = None
) -> plt.Figure:
    """
    Generate Algorithm Performance Heatmap.

    Rows: Tests, Columns: Algorithms, Color: Normalized performance

    Args:
        all_stats: Statistics dictionary
        metric: Which metric to visualize
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Collect algorithms and tests
    all_algos = sorted(set(
        algo for test_stats in all_stats.values()
        for algo in test_stats.keys()
    ))
    test_ids = sorted(all_stats.keys())

    # Build data matrix
    n_tests = len(test_ids)
    n_algos = len(all_algos)
    data = np.full((n_tests, n_algos), np.nan)

    for i, test_id in enumerate(test_ids):
        test_stats = all_stats[test_id]

        for j, algo in enumerate(all_algos):
            if algo in test_stats:
                val = test_stats[algo].get(metric, np.nan)
                data[i, j] = val

    # Normalize each row to [0, 1] (1 = best)
    normalized = np.full_like(data, np.nan)
    for i in range(n_tests):
        row = data[i]
        valid = ~np.isnan(row)
        if valid.sum() > 0:
            row_min = np.min(row[valid])
            row_max = np.max(row[valid])
            if row_max > row_min:
                # For time/iterations, lower is better (invert scale)
                if metric in ['iterations_mean', 'time_mean']:
                    normalized[i, valid] = 1 - (row[valid] - row_min) / (row_max - row_min)
                else:
                    normalized[i, valid] = (row[valid] - row_min) / (row_max - row_min)
            else:
                normalized[i, valid] = 0.5

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, n_algos * 1.5), max(6, n_tests * 1.2)))

    # Create custom colormap (red=worst, green=best)
    cmap = plt.cm.RdYlGn

    # Plot heatmap
    im = ax.imshow(normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(n_algos))
    ax.set_yticks(np.arange(n_tests))
    ax.set_xticklabels([a.replace('_', '\n') for a in all_algos], rotation=45, ha='right')
    ax.set_yticklabels([TESTS[t]['short_name'] for t in test_ids])

    # Add text annotations with actual values
    for i in range(n_tests):
        for j in range(n_algos):
            if not np.isnan(data[i, j]):
                if metric == 'success_rate':
                    text = f"{100*data[i, j]:.0f}%"
                elif metric == 'time_mean':
                    text = f"{data[i, j]:.1f}s"
                else:
                    text = f"{data[i, j]:.0f}"

                # Choose text color based on background
                color = 'white' if normalized[i, j] < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Performance (1=best)', rotation=-90, va="bottom")

    ax.set_title(f'Algorithm Performance Heatmap\n(Metric: {metric.replace("_", " ")})')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")

    return fig


def identify_patterns(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Identify patterns in algorithm performance.

    Analyzes when lookahead EM wins/loses and gamma schedule effectiveness.

    Args:
        all_stats: Statistics dictionary

    Returns:
        Dictionary with pattern analysis
    """
    patterns = {
        'lookahead_wins': [],
        'lookahead_loses': [],
        'adaptive_vs_fixed': {},
    }

    for test_id, test_stats in all_stats.items():
        if 'standard_em' not in test_stats:
            continue

        baseline = test_stats['standard_em']

        # Check lookahead algorithms
        for algo, stats in test_stats.items():
            if 'lookahead' not in algo:
                continue

            # Compare iterations
            if not np.isnan(stats['iterations_mean']) and not np.isnan(baseline['iterations_mean']):
                reduction = 1 - stats['iterations_mean'] / baseline['iterations_mean']

                if reduction > 0.1:  # >10% reduction
                    patterns['lookahead_wins'].append({
                        'test': test_id,
                        'algorithm': algo,
                        'reduction': reduction,
                        'model': TESTS[test_id]['model']
                    })
                elif reduction < -0.1:  # >10% increase
                    patterns['lookahead_loses'].append({
                        'test': test_id,
                        'algorithm': algo,
                        'increase': -reduction,
                        'model': TESTS[test_id]['model']
                    })

        # Compare adaptive vs fixed gamma
        adaptive_algo = None
        fixed_algos = []

        for algo in test_stats.keys():
            if 'lookahead' in algo:
                if 'adaptive' in algo:
                    adaptive_algo = algo
                elif 'gamma' in algo:
                    fixed_algos.append(algo)

        if adaptive_algo and fixed_algos:
            adaptive_stats = test_stats[adaptive_algo]
            best_fixed = min(fixed_algos,
                           key=lambda a: test_stats[a].get('iterations_mean', float('inf')))
            best_fixed_stats = test_stats[best_fixed]

            if not np.isnan(adaptive_stats['iterations_mean']) and not np.isnan(best_fixed_stats['iterations_mean']):
                patterns['adaptive_vs_fixed'][test_id] = {
                    'adaptive_iters': adaptive_stats['iterations_mean'],
                    'best_fixed_iters': best_fixed_stats['iterations_mean'],
                    'best_fixed_algo': best_fixed,
                    'adaptive_better': adaptive_stats['iterations_mean'] < best_fixed_stats['iterations_mean']
                }

    return patterns


def generate_report(
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    overall_stats: Dict[str, Dict[str, Any]],
    patterns: Dict[str, Any],
    output_path: str = None
) -> str:
    """
    Generate comprehensive cross-test analysis report.

    Args:
        all_stats: Statistics per test per algorithm
        overall_stats: Overall statistics per algorithm
        patterns: Pattern analysis results
        output_path: Path to save report

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-TEST COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("1. OVERALL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Tests analyzed: {len(all_stats)}")
    lines.append(f"Algorithms compared: {len(overall_stats)}")
    lines.append("")

    # Algorithm rankings
    lines.append("2. ALGORITHM RANKINGS")
    lines.append("-" * 40)

    # Sort by wins
    by_wins = sorted(
        overall_stats.items(),
        key=lambda x: x[1]['n_wins_iterations'] + x[1]['n_wins_time'],
        reverse=True
    )

    for algo, stats in by_wins:
        total_wins = stats['n_wins_iterations'] + stats['n_wins_time'] + stats['n_wins_success']
        lines.append(f"  {algo}:")
        lines.append(f"    Tests participated: {stats['n_tests']}")
        lines.append(f"    Iteration wins: {stats['n_wins_iterations']}")
        lines.append(f"    Time wins: {stats['n_wins_time']}")
        lines.append(f"    Success wins: {stats['n_wins_success']}")
        if not np.isnan(stats['avg_speedup']):
            lines.append(f"    Avg speedup vs StandardEM: {stats['avg_speedup']:.2f}x")
        if not np.isnan(stats['avg_iter_reduction']):
            lines.append(f"    Avg iteration reduction: {100*stats['avg_iter_reduction']:.1f}%")
        lines.append("")

    # Pattern analysis
    lines.append("3. PATTERN ANALYSIS")
    lines.append("-" * 40)

    lines.append("\n  When Lookahead EM Wins:")
    if patterns['lookahead_wins']:
        for win in patterns['lookahead_wins']:
            lines.append(f"    - {win['test']} ({win['algorithm']}): "
                        f"{100*win['reduction']:.1f}% fewer iterations")
    else:
        lines.append("    - No clear wins identified")

    lines.append("\n  When Lookahead EM Loses:")
    if patterns['lookahead_loses']:
        for loss in patterns['lookahead_loses']:
            lines.append(f"    - {loss['test']} ({loss['algorithm']}): "
                        f"{100*loss['increase']:.1f}% more iterations")
    else:
        lines.append("    - No clear losses identified")

    lines.append("\n  Adaptive vs Fixed Gamma:")
    if patterns['adaptive_vs_fixed']:
        adaptive_wins = sum(1 for x in patterns['adaptive_vs_fixed'].values() if x['adaptive_better'])
        total = len(patterns['adaptive_vs_fixed'])
        lines.append(f"    Adaptive wins: {adaptive_wins}/{total} tests")

        for test_id, comp in patterns['adaptive_vs_fixed'].items():
            winner = "Adaptive" if comp['adaptive_better'] else comp['best_fixed_algo']
            lines.append(f"    - {test_id}: {winner} wins "
                        f"({comp['adaptive_iters']:.0f} vs {comp['best_fixed_iters']:.0f} iters)")
    else:
        lines.append("    - No comparison data available")

    # Recommendations
    lines.append("\n4. RECOMMENDATIONS")
    lines.append("-" * 40)

    # Find best overall algorithm
    if overall_stats:
        best_algo = max(
            overall_stats.items(),
            key=lambda x: x[1]['n_wins_iterations'] + x[1]['n_wins_time']
        )[0]
        lines.append(f"  Best overall algorithm: {best_algo}")

    # Adaptive gamma recommendation
    if patterns['adaptive_vs_fixed']:
        adaptive_wins = sum(1 for x in patterns['adaptive_vs_fixed'].values() if x['adaptive_better'])
        total = len(patterns['adaptive_vs_fixed'])
        if adaptive_wins > total / 2:
            lines.append("  Recommendation: Use adaptive gamma for general use")
        else:
            lines.append("  Recommendation: Consider fixed gamma for specific applications")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


def run_cross_test_comparison(
    results_base_dir: str = None,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """
    Run complete cross-test comparison analysis.

    Args:
        results_base_dir: Base directory containing test results
        output_dir: Directory to save outputs
        verbose: Whether to print results

    Returns:
        Tuple of (all_stats, overall_stats)
    """
    if results_base_dir is None:
        results_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    if output_dir is None:
        output_dir = os.path.join(results_base_dir, 'cross_test_analysis')

    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    if verbose:
        print("Loading results from all tests...")

    all_results = load_all_test_results(results_base_dir)

    if not all_results:
        print("No results found!")
        return {}, {}

    # Compute statistics for each test
    if verbose:
        print("\nComputing statistics...")

    all_stats = {}
    for test_id, results in all_results.items():
        all_stats[test_id] = compute_test_statistics(results)

    # Generate summary table
    if verbose:
        print("\nGenerating summary tables...")

    ascii_table, latex_table = generate_summary_table(all_stats)

    if verbose:
        print("\n" + ascii_table)

    # Save tables
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w') as f:
        f.write(ascii_table)

    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write(latex_table)

    # Compute overall statistics
    overall_stats = compute_overall_statistics(all_stats)

    # Generate heatmap
    if verbose:
        print("\nGenerating heatmap...")

    fig = generate_heatmap(
        all_stats,
        metric='iterations_mean',
        output_path=os.path.join(output_dir, 'performance_heatmap.png')
    )
    plt.close(fig)

    # Identify patterns
    if verbose:
        print("\nAnalyzing patterns...")

    patterns = identify_patterns(all_stats)

    # Generate report
    if verbose:
        print("\nGenerating report...")

    report = generate_report(
        all_stats,
        overall_stats,
        patterns,
        output_path=os.path.join(output_dir, 'report.txt')
    )

    if verbose:
        print("\n" + report)

    return all_stats, overall_stats


# ============================================================
# Unit Tests
# ============================================================

def test_compute_test_statistics():
    """Test statistics computation."""
    results = [
        {'algorithm': 'standard_em', 'algorithm_params': {},
         'converged': True, 'final_ll': -100, 'elapsed_time': 1.0,
         'n_iterations': 50, 'error': None},
        {'algorithm': 'lookahead_em', 'algorithm_params': {'gamma': 0.5},
         'converged': True, 'final_ll': -95, 'elapsed_time': 1.2,
         'n_iterations': 30, 'error': None},
    ]

    stats = compute_test_statistics(results)

    assert 'standard_em' in stats
    assert 'lookahead_em_gamma_0.5' in stats
    assert stats['standard_em']['iterations_mean'] == 50

    print("  PASSED")


def test_generate_summary_table():
    """Test summary table generation."""
    all_stats = {
        'test_1_high_d_gmm': {
            'standard_em': {
                'iterations_mean': 100, 'time_mean': 10.0, 'success_rate': 0.8
            },
            'lookahead_em_gamma_0.5': {
                'iterations_mean': 60, 'time_mean': 12.0, 'success_rate': 0.9
            }
        }
    }

    ascii_table, latex_table = generate_summary_table(all_stats)

    assert 'standard_em' in ascii_table
    assert 'lookahead' in ascii_table
    assert '\\begin{table}' in latex_table

    print("  PASSED")


def test_compute_overall_statistics():
    """Test overall statistics computation."""
    all_stats = {
        'test_1_high_d_gmm': {
            'standard_em': {
                'iterations_mean': 100, 'time_mean': 10.0, 'success_rate': 0.8,
                'iterations_values': np.array([100]), 'time_values': np.array([10.0])
            },
            'lookahead_em_gamma_0.5': {
                'iterations_mean': 60, 'time_mean': 12.0, 'success_rate': 0.9,
                'iterations_values': np.array([60]), 'time_values': np.array([12.0])
            }
        }
    }

    overall = compute_overall_statistics(all_stats)

    assert 'standard_em' in overall
    assert 'lookahead_em_gamma_0.5' in overall
    assert overall['lookahead_em_gamma_0.5']['n_wins_iterations'] == 1

    print("  PASSED")


def test_identify_patterns():
    """Test pattern identification."""
    all_stats = {
        'test_1_high_d_gmm': {
            'standard_em': {'iterations_mean': 100, 'time_mean': 10.0},
            'lookahead_em_gamma_0.5': {'iterations_mean': 60, 'time_mean': 12.0},
            'lookahead_em_gamma_adaptive': {'iterations_mean': 55, 'time_mean': 11.0},
        }
    }

    patterns = identify_patterns(all_stats)

    assert len(patterns['lookahead_wins']) > 0
    assert 'test_1_high_d_gmm' in patterns['adaptive_vs_fixed']

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running cross_test_comparison.py unit tests")
    print("=" * 60)

    print("Testing compute_test_statistics...")
    test_compute_test_statistics()

    print("Testing generate_summary_table...")
    test_generate_summary_table()

    print("Testing compute_overall_statistics...")
    test_compute_overall_statistics()

    print("Testing identify_patterns...")
    test_identify_patterns()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Test Comparison Analysis")
    parser.add_argument('--test', action='store_true', help="Run unit tests only")
    parser.add_argument('--results-dir', type=str, help="Base results directory")
    parser.add_argument('--output-dir', type=str, help="Output directory")

    args = parser.parse_args()

    if args.test:
        run_all_tests()
    else:
        run_cross_test_comparison(
            results_base_dir=args.results_dir,
            output_dir=args.output_dir,
            verbose=True
        )
