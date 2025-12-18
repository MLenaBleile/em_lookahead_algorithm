#!/usr/bin/env python3
"""
MoE Comprehensive Test Results Analysis

Analyzes results from the comprehensive MoE experiment.
Produces publication-quality statistics and tables.

Usage:
    python analyze_moe_comprehensive.py [--input JSON_FILE] [--latex] [--csv]
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('config', {}), data.get('results', [])


def filter_valid_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out failed runs (errors, non-converged, invalid log-likelihoods)."""
    valid = []
    for r in results:
        if r.get('error') is not None:
            continue
        if not r.get('converged', False):
            continue
        ll = r.get('final_ll', float('-inf'))
        if ll == float('-inf') or math.isinf(ll) or math.isnan(ll):
            continue
        valid.append(r)
    return valid


def group_by_condition(results: List[Dict[str, Any]]) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Group results by experimental condition (n_experts, n_features, n_samples)."""
    grouped = defaultdict(list)
    for r in results:
        key = (r['n_experts'], r['n_features'], r['n_samples'])
        grouped[key].append(r)
    return dict(grouped)


def group_by_algorithm(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by algorithm."""
    grouped = defaultdict(list)
    for r in results:
        grouped[r['algorithm']].append(r)
    return dict(grouped)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'q25': float('nan'),
            'q75': float('nan'),
            'n': 0
        }
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'n': len(arr)
    }


def compute_paired_comparison(
    std_results: List[Dict[str, Any]],
    la_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute paired comparison statistics between standard and lookahead EM."""
    # Match by seed
    std_by_seed = {r['seed']: r for r in std_results}
    la_by_seed = {r['seed']: r for r in la_results}

    common_seeds = set(std_by_seed.keys()) & set(la_by_seed.keys())

    if not common_seeds:
        return {'n_pairs': 0}

    std_lls = []
    la_lls = []
    std_times = []
    la_times = []
    std_iters = []
    la_iters = []
    ll_improvements = []
    la_wins = 0

    for seed in sorted(common_seeds):
        std_r = std_by_seed[seed]
        la_r = la_by_seed[seed]

        std_lls.append(std_r['final_ll'])
        la_lls.append(la_r['final_ll'])
        std_times.append(std_r['elapsed_time'])
        la_times.append(la_r['elapsed_time'])
        std_iters.append(std_r['n_iterations'])
        la_iters.append(la_r['n_iterations'])

        # LL improvement (higher is better)
        ll_diff = la_r['final_ll'] - std_r['final_ll']
        ll_improvements.append(ll_diff)
        if ll_diff > 0:
            la_wins += 1

    n_pairs = len(common_seeds)

    # Paired t-test on log-likelihoods
    if n_pairs >= 2:
        t_stat, p_value = stats.ttest_rel(la_lls, std_lls)
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_p_value = stats.wilcoxon(la_lls, std_lls)
        except ValueError:
            w_stat, w_p_value = float('nan'), float('nan')
    else:
        t_stat, p_value = float('nan'), float('nan')
        w_stat, w_p_value = float('nan'), float('nan')

    return {
        'n_pairs': n_pairs,
        'standard_em': {
            'll': compute_statistics(std_lls),
            'time': compute_statistics(std_times),
            'iterations': compute_statistics(std_iters)
        },
        'lookahead_em': {
            'll': compute_statistics(la_lls),
            'time': compute_statistics(la_times),
            'iterations': compute_statistics(la_iters)
        },
        'll_improvement': compute_statistics(ll_improvements),
        'la_win_rate': la_wins / n_pairs if n_pairs > 0 else 0,
        'paired_ttest': {'t_stat': t_stat, 'p_value': p_value},
        'wilcoxon': {'w_stat': w_stat, 'p_value': w_p_value},
        'time_ratio': np.mean(la_times) / np.mean(std_times) if np.mean(std_times) > 0 else float('nan'),
        'iter_ratio': np.mean(la_iters) / np.mean(std_iters) if np.mean(std_iters) > 0 else float('nan')
    }


def analyze_by_condition(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results grouped by experimental condition."""
    valid = filter_valid_results(results)
    grouped = group_by_condition(valid)

    analysis = {}
    for condition, cond_results in sorted(grouped.items()):
        n_experts, n_features, n_samples = condition
        key = f"K{n_experts}_D{n_features}_N{n_samples}"

        by_algo = group_by_algorithm(cond_results)
        std_results = by_algo.get('standard_em', [])
        la_results = by_algo.get('lookahead_em', [])

        analysis[key] = {
            'n_experts': n_experts,
            'n_features': n_features,
            'n_samples': n_samples,
            'comparison': compute_paired_comparison(std_results, la_results)
        }

    return analysis


def analyze_overall(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall statistics across all conditions."""
    valid = filter_valid_results(results)
    by_algo = group_by_algorithm(valid)

    overall = {}
    for algo, algo_results in by_algo.items():
        lls = [r['final_ll'] for r in algo_results]
        times = [r['elapsed_time'] for r in algo_results]
        iters = [r['n_iterations'] for r in algo_results]

        overall[algo] = {
            'n_runs': len(algo_results),
            'convergence_rate': len(algo_results) / len([r for r in results if r['algorithm'] == algo]) if results else 0,
            'll': compute_statistics(lls),
            'time': compute_statistics(times),
            'iterations': compute_statistics(iters)
        }

    # Overall paired comparison
    std_results = by_algo.get('standard_em', [])
    la_results = by_algo.get('lookahead_em', [])
    overall['paired_comparison'] = compute_paired_comparison(std_results, la_results)

    return overall


def format_stat(stat: Dict[str, float], fmt: str = '.2f') -> str:
    """Format mean±std string."""
    mean = stat.get('mean', float('nan'))
    std = stat.get('std', float('nan'))
    if math.isnan(mean):
        return "N/A"
    return f"{mean:{fmt}}±{std:{fmt}}"


def format_p_value(p: float) -> str:
    """Format p-value with significance stars."""
    if math.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def generate_console_tables(analysis: Dict[str, Any], overall: Dict[str, Any]) -> str:
    """Generate console-formatted tables."""
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("MoE COMPREHENSIVE TEST RESULTS - PUBLICATION STATISTICS")
    lines.append("=" * 100)
    lines.append("")

    # Overall Summary
    lines.append("-" * 100)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'Algorithm':<20} {'N Runs':>10} {'Conv%':>10} {'Log-Lik (mean±std)':>25} {'Time (s)':>20} {'Iterations':>15}")
    lines.append("-" * 100)

    for algo in ['standard_em', 'lookahead_em']:
        if algo not in overall:
            continue
        s = overall[algo]
        conv_pct = f"{100*s['convergence_rate']:.1f}%"
        ll_str = format_stat(s['ll'], '.2f')
        time_str = format_stat(s['time'], '.2f')
        iter_str = format_stat(s['iterations'], '.0f')
        lines.append(f"{algo:<20} {s['n_runs']:>10} {conv_pct:>10} {ll_str:>25} {time_str:>20} {iter_str:>15}")

    lines.append("")

    # Paired comparison overall
    pc = overall.get('paired_comparison', {})
    if pc.get('n_pairs', 0) > 0:
        lines.append("-" * 100)
        lines.append("OVERALL PAIRED COMPARISON (Lookahead vs Standard)")
        lines.append("-" * 100)
        lines.append(f"Number of paired comparisons: {pc['n_pairs']}")
        lines.append(f"Lookahead EM win rate: {100*pc['la_win_rate']:.1f}%")
        lines.append(f"Mean LL improvement: {format_stat(pc['ll_improvement'], '.2f')}")
        lines.append(f"Paired t-test p-value: {format_p_value(pc['paired_ttest']['p_value'])}")
        lines.append(f"Wilcoxon test p-value: {format_p_value(pc['wilcoxon']['p_value'])}")
        lines.append(f"Time ratio (LA/Std): {pc['time_ratio']:.2f}x")
        lines.append(f"Iteration ratio (LA/Std): {pc['iter_ratio']:.2f}x")

    lines.append("")

    # Per-condition results table
    lines.append("-" * 100)
    lines.append("PER-CONDITION COMPARISON")
    lines.append("-" * 100)
    lines.append(f"{'Condition':<15} {'K':>3} {'D':>3} {'N':>6} {'N':>5} {'LA Win%':>10} {'LL Improve':>15} {'p-value':>15}")
    lines.append("-" * 100)

    for key, cond in sorted(analysis.items()):
        pc = cond['comparison']
        if pc['n_pairs'] == 0:
            continue
        win_pct = f"{100*pc['la_win_rate']:.0f}%"
        ll_imp = format_stat(pc['ll_improvement'], '.2f')
        p_val = format_p_value(pc['paired_ttest']['p_value'])
        lines.append(f"{key:<15} {cond['n_experts']:>3} {cond['n_features']:>3} {cond['n_samples']:>6} "
                    f"{pc['n_pairs']:>5} {win_pct:>10} {ll_imp:>15} {p_val:>15}")

    lines.append("=" * 100)
    lines.append("")

    # Detailed per-condition statistics
    lines.append("-" * 100)
    lines.append("DETAILED PER-CONDITION STATISTICS")
    lines.append("-" * 100)

    for key, cond in sorted(analysis.items()):
        pc = cond['comparison']
        if pc['n_pairs'] == 0:
            continue

        lines.append(f"\n{key}: K={cond['n_experts']} experts, D={cond['n_features']} features, N={cond['n_samples']} samples")
        lines.append("-" * 60)

        std = pc['standard_em']
        la = pc['lookahead_em']

        lines.append(f"  {'Metric':<20} {'Standard EM':>20} {'Lookahead EM':>20}")
        lines.append(f"  {'-'*60}")
        lines.append(f"  {'Log-Likelihood':<20} {format_stat(std['ll'], '.2f'):>20} {format_stat(la['ll'], '.2f'):>20}")
        lines.append(f"  {'Time (s)':<20} {format_stat(std['time'], '.2f'):>20} {format_stat(la['time'], '.2f'):>20}")
        lines.append(f"  {'Iterations':<20} {format_stat(std['iterations'], '.0f'):>20} {format_stat(la['iterations'], '.0f'):>20}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("Note: * p<0.05, ** p<0.01, *** p<0.001 (paired t-test)")
    lines.append("=" * 100)

    return "\n".join(lines)


def generate_latex_tables(analysis: Dict[str, Any], overall: Dict[str, Any]) -> str:
    """Generate LaTeX-formatted tables for publication."""
    lines = []

    # Overall comparison table
    lines.append("% Overall Algorithm Comparison")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Overall comparison of Standard EM and Lookahead EM on Mixture of Experts}")
    lines.append("\\label{tab:moe_overall}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Algorithm & N & Log-Likelihood & Time (s) & Iterations \\\\")
    lines.append("\\midrule")

    for algo in ['standard_em', 'lookahead_em']:
        if algo not in overall:
            continue
        s = overall[algo]
        algo_name = "Standard EM" if algo == "standard_em" else "Lookahead EM"
        ll_str = f"${s['ll']['mean']:.2f} \\pm {s['ll']['std']:.2f}$"
        time_str = f"${s['time']['mean']:.2f} \\pm {s['time']['std']:.2f}$"
        iter_str = f"${s['iterations']['mean']:.0f} \\pm {s['iterations']['std']:.0f}$"
        lines.append(f"{algo_name} & {s['n_runs']} & {ll_str} & {time_str} & {iter_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Per-condition table
    lines.append("% Per-Condition Comparison")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Per-condition comparison: Lookahead EM vs Standard EM}")
    lines.append("\\label{tab:moe_conditions}")
    lines.append("\\begin{tabular}{cccrccr}")
    lines.append("\\toprule")
    lines.append("$K$ & $D$ & $N$ & Pairs & Win\\% & $\\Delta$LL & $p$-value \\\\")
    lines.append("\\midrule")

    for key, cond in sorted(analysis.items()):
        pc = cond['comparison']
        if pc['n_pairs'] == 0:
            continue

        win_pct = f"{100*pc['la_win_rate']:.0f}\\%"
        ll_imp = pc['ll_improvement']['mean']
        ll_imp_str = f"${ll_imp:+.2f}$"

        p = pc['paired_ttest']['p_value']
        if math.isnan(p):
            p_str = "---"
        elif p < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{p:.3f}"

        lines.append(f"{cond['n_experts']} & {cond['n_features']} & {cond['n_samples']} & "
                    f"{pc['n_pairs']} & {win_pct} & {ll_imp_str} & {p_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item $K$: number of experts, $D$: feature dimensionality, $N$: sample size.")
    lines.append("\\item $\\Delta$LL: mean log-likelihood improvement (Lookahead $-$ Standard).")
    lines.append("\\item $p$-values from paired $t$-test.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    lines.append("")

    # Summary statistics table
    lines.append("% Summary Statistics")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Summary of Lookahead EM improvements over Standard EM}")
    lines.append("\\label{tab:moe_summary}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")

    pc = overall.get('paired_comparison', {})
    if pc.get('n_pairs', 0) > 0:
        lines.append(f"Total paired comparisons & {pc['n_pairs']} \\\\")
        lines.append(f"Lookahead EM win rate & {100*pc['la_win_rate']:.1f}\\% \\\\")
        lines.append(f"Mean LL improvement & ${pc['ll_improvement']['mean']:+.2f} \\pm {pc['ll_improvement']['std']:.2f}$ \\\\")

        p = pc['paired_ttest']['p_value']
        if not math.isnan(p):
            if p < 0.001:
                p_str = "$<$0.001"
            else:
                p_str = f"{p:.4f}"
            lines.append(f"Paired $t$-test $p$-value & {p_str} \\\\")

        lines.append(f"Time ratio (LA/Std) & {pc['time_ratio']:.2f}$\\times$ \\\\")
        lines.append(f"Iteration ratio (LA/Std) & {pc['iter_ratio']:.2f}$\\times$ \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_csv_tables(analysis: Dict[str, Any], overall: Dict[str, Any]) -> Tuple[str, str]:
    """Generate CSV-formatted tables."""
    # Overall results CSV
    overall_lines = ["algorithm,n_runs,convergence_rate,ll_mean,ll_std,time_mean,time_std,iter_mean,iter_std"]
    for algo in ['standard_em', 'lookahead_em']:
        if algo not in overall:
            continue
        s = overall[algo]
        overall_lines.append(f"{algo},{s['n_runs']},{s['convergence_rate']:.4f},"
                            f"{s['ll']['mean']:.4f},{s['ll']['std']:.4f},"
                            f"{s['time']['mean']:.4f},{s['time']['std']:.4f},"
                            f"{s['iterations']['mean']:.4f},{s['iterations']['std']:.4f}")

    # Per-condition CSV
    cond_lines = ["n_experts,n_features,n_samples,n_pairs,la_win_rate,"
                  "std_ll_mean,std_ll_std,la_ll_mean,la_ll_std,ll_improve_mean,ll_improve_std,"
                  "std_time_mean,la_time_mean,time_ratio,"
                  "ttest_p,wilcoxon_p"]

    for key, cond in sorted(analysis.items()):
        pc = cond['comparison']
        if pc['n_pairs'] == 0:
            continue

        std = pc['standard_em']
        la = pc['lookahead_em']

        cond_lines.append(f"{cond['n_experts']},{cond['n_features']},{cond['n_samples']},"
                         f"{pc['n_pairs']},{pc['la_win_rate']:.4f},"
                         f"{std['ll']['mean']:.4f},{std['ll']['std']:.4f},"
                         f"{la['ll']['mean']:.4f},{la['ll']['std']:.4f},"
                         f"{pc['ll_improvement']['mean']:.4f},{pc['ll_improvement']['std']:.4f},"
                         f"{std['time']['mean']:.4f},{la['time']['mean']:.4f},{pc['time_ratio']:.4f},"
                         f"{pc['paired_ttest']['p_value']:.6f},{pc['wilcoxon']['p_value']:.6f}")

    return "\n".join(overall_lines), "\n".join(cond_lines)


def find_default_results_file() -> Optional[str]:
    """Find the most recent MoE comprehensive results file."""
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )

    import glob
    pattern = os.path.join(results_dir, 'moe_comprehensive_*.json')
    files = glob.glob(pattern)

    if not files:
        return None

    # Return most recent
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MoE comprehensive test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_moe_comprehensive.py
    python analyze_moe_comprehensive.py --input results/moe_comprehensive_20251217_234633.json
    python analyze_moe_comprehensive.py --latex --csv
        """
    )
    parser.add_argument('--input', '-i', type=str, help="Input JSON results file")
    parser.add_argument('--latex', action='store_true', help="Output LaTeX tables")
    parser.add_argument('--csv', action='store_true', help="Output CSV tables")
    parser.add_argument('--output-dir', '-o', type=str, help="Output directory for files")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress console output")

    args = parser.parse_args()

    # Find input file
    if args.input:
        json_path = args.input
    else:
        json_path = find_default_results_file()
        if json_path is None:
            print("Error: No MoE comprehensive results file found. Specify with --input")
            sys.exit(1)

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"Loading results from: {json_path}")

    # Load and analyze
    config, results = load_results(json_path)

    print(f"Loaded {len(results)} total results")
    print(f"Configuration: {config}")

    # Run analysis
    analysis = analyze_by_condition(results)
    overall = analyze_overall(results)

    # Console output
    if not args.quiet:
        print(generate_console_tables(analysis, overall))

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(json_path)

    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # LaTeX output
    if args.latex:
        latex_content = generate_latex_tables(analysis, overall)
        latex_path = os.path.join(out_dir, f"{base_name}_tables.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"LaTeX tables written to: {latex_path}")

    # CSV output
    if args.csv:
        overall_csv, cond_csv = generate_csv_tables(analysis, overall)

        overall_path = os.path.join(out_dir, f"{base_name}_overall.csv")
        with open(overall_path, 'w') as f:
            f.write(overall_csv)
        print(f"Overall CSV written to: {overall_path}")

        cond_path = os.path.join(out_dir, f"{base_name}_conditions.csv")
        with open(cond_path, 'w') as f:
            f.write(cond_csv)
        print(f"Conditions CSV written to: {cond_path}")

    # Print summary
    pc = overall.get('paired_comparison', {})
    if pc.get('n_pairs', 0) > 0:
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)
        print(f"  - Lookahead EM wins in {100*pc['la_win_rate']:.1f}% of paired comparisons")
        print(f"  - Mean log-likelihood improvement: {pc['ll_improvement']['mean']:+.2f}")
        p = pc['paired_ttest']['p_value']
        sig = "YES" if p < 0.05 else "NO"
        print(f"  - Statistically significant (p<0.05): {sig} (p={p:.4f})")
        print(f"  - Lookahead EM takes {pc['time_ratio']:.1f}x longer per run")
        print("=" * 60)

    return analysis, overall


if __name__ == "__main__":
    main()
