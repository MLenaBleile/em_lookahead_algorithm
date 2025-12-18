"""
Test 4 Analysis: Robustness/Initialization Study Results

Analyzes results from Test 4: Initialization robustness experiment.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lookahead_em_evaluation.utils.logging_utils import ExperimentLogger


def load_test_4_results(results_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load all Test 4 results by initialization strategy."""
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    strategies = ['collapsed', 'far_away', 'random', 'kmeans']
    all_results = {}

    for strategy in strategies:
        test_dir = os.path.join(results_dir, f'test_4_robustness_{strategy}')
        if os.path.exists(test_dir):
            logger = ExperimentLogger(test_dir)
            all_results[strategy] = logger.load_all()

    return all_results


def compute_success_heatmap(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Compute success rate heatmap: strategies × algorithms."""
    # Find global best likelihood
    all_lls = []
    for results in all_results.values():
        for r in results:
            if r.get('final_ll', float('-inf')) > float('-inf') and r.get('error') is None:
                all_lls.append(r['final_ll'])

    best_ll = max(all_lls) if all_lls else float('-inf')
    success_threshold = best_ll - 5.0

    heatmap = {}
    for strategy, results in all_results.items():
        by_algo = {}
        for r in results:
            algo = r.get('algorithm', 'unknown')
            params = r.get('algorithm_params', {})
            key = f"{algo}_gamma_{params.get('gamma', '')}" if params.get('gamma') else algo
            if key not in by_algo:
                by_algo[key] = []
            by_algo[key].append(r)

        strategy_rates = {}
        for algo, algo_results in by_algo.items():
            final_lls = [r['final_ll'] for r in algo_results
                        if r.get('error') is None and r.get('final_ll', float('-inf')) > float('-inf')]
            if final_lls:
                n_success = sum(1 for ll in final_lls if ll >= success_threshold)
                strategy_rates[algo] = n_success / len(final_lls)
            else:
                strategy_rates[algo] = 0.0

        heatmap[strategy] = strategy_rates

    return heatmap


def analyze_convergence_modes(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Identify modes (local optima) in final likelihood distribution."""
    modes = {}

    for strategy, results in all_results.items():
        final_lls = [r['final_ll'] for r in results
                    if r.get('error') is None and r.get('final_ll', float('-inf')) > float('-inf')]

        if len(final_lls) > 5:
            # Simple mode detection: cluster by rounding
            rounded = [round(ll, 0) for ll in final_lls]
            unique_modes = sorted(set(rounded), reverse=True)
            mode_counts = {m: rounded.count(m) for m in unique_modes}

            modes[strategy] = {
                'n_modes': len(unique_modes),
                'mode_counts': mode_counts,
                'best_mode': max(unique_modes) if unique_modes else None,
                'll_range': (min(final_lls), max(final_lls)) if final_lls else (None, None)
            }

    return modes


def generate_ascii_heatmap(heatmap: Dict[str, Dict[str, float]]) -> str:
    """Generate ASCII representation of success rate heatmap."""
    lines = ["=" * 100, "Success Rate Heatmap: Strategies × Algorithms", "=" * 100]

    # Get all algorithms
    all_algos = set()
    for strategy_data in heatmap.values():
        all_algos.update(strategy_data.keys())
    algos = sorted(all_algos)

    # Header
    header = f"{'Strategy':<15}"
    for algo in algos:
        header += f" {algo[:12]:>12}"
    lines.append(header)
    lines.append("-" * 100)

    # Data
    for strategy in sorted(heatmap.keys()):
        row = f"{strategy:<15}"
        for algo in algos:
            rate = heatmap[strategy].get(algo, 0)
            row += f" {100*rate:>11.0f}%"
        print(row)
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def run_analysis(results_dir: str = None, verbose: bool = True) -> Dict[str, Any]:
    """Run full Test 4 analysis."""
    all_results = load_test_4_results(results_dir)

    if not all_results:
        print("No results found for Test 4")
        return {}

    if verbose:
        total = sum(len(r) for r in all_results.values())
        print(f"Loaded {total} results across {len(all_results)} strategies")

    heatmap = compute_success_heatmap(all_results)
    modes = analyze_convergence_modes(all_results)

    if verbose:
        print(generate_ascii_heatmap(heatmap))
        print("\nConvergence Modes by Strategy:")
        for strategy, mode_info in modes.items():
            print(f"  {strategy}: {mode_info['n_modes']} modes, range: {mode_info['ll_range']}")

    return {
        'heatmap': heatmap,
        'modes': modes,
        'n_results': sum(len(r) for r in all_results.values())
    }


# Unit Tests
def test_heatmap_computation():
    """Test success rate heatmap computation."""
    mock_results = {
        'random': [
            {'algorithm': 'standard_em', 'algorithm_params': {}, 'final_ll': -100.0, 'error': None},
            {'algorithm': 'standard_em', 'algorithm_params': {}, 'final_ll': -95.0, 'error': None},
        ]
    }
    heatmap = compute_success_heatmap(mock_results)
    assert 'random' in heatmap
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running analyze_test_4.py unit tests")
    print("=" * 60)
    print("Testing heatmap computation...")
    test_heatmap_computation()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Test 4 Results")
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        run_all_tests()
    else:
        run_analysis()
