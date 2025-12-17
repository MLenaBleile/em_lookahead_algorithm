"""
Comprehensive MoE Experiment for Publication

Tests Lookahead EM vs Standard EM and SQUAREM on Mixture of Experts
with varying configurations:
- Expert counts: K = 2, 3, 5, 10
- Dimensions: d = 2, 5, 10, 20
- Sample sizes: n = 100, 500, 1000

Each configuration runs 30 restarts for statistical significance.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from itertools import product
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mixture_of_experts import (
    MixtureOfExperts, generate_moe_data,
    initialize_random, initialize_equal_gates
)
from algorithms.standard_em import StandardEM
from algorithms.lookahead_em import LookaheadEM
from algorithms.squarem_wrapper import SQUAREM


class MoEModelWrapper:
    """Wrapper to make MoE compatible with EM algorithms."""

    def __init__(self, n_experts: int, n_features: int):
        self.n_experts = n_experts
        self.n_features = n_features
        self.moe = MixtureOfExperts(n_experts=n_experts, n_features=n_features)

    def e_step(self, data, theta):
        return self.moe.e_step(data, theta)

    def m_step(self, data, responsibilities):
        return self.moe.m_step(data, responsibilities)

    def log_likelihood(self, data, theta):
        return self.moe.log_likelihood(data, theta)

    def compute_Q(self, theta, theta_old, data, responsibilities):
        return self.moe.compute_Q(theta, theta_old, data, responsibilities)

    def compute_Q_gradient(self, theta, theta_old, data, responsibilities):
        return self.moe.compute_Q_gradient(theta, theta_old, data, responsibilities)

    def compute_Q_hessian(self, theta, theta_old, data, responsibilities):
        return self.moe.compute_Q_hessian(theta, theta_old, data, responsibilities)


def run_single_experiment(
    n_experts: int,
    n_features: int,
    n_samples: int,
    algorithm: str,
    seed: int,
    max_iter: int = 500,
    timeout: float = 300.0,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Run a single MoE experiment.

    Args:
        n_experts: Number of experts (K)
        n_features: Number of features (d)
        n_samples: Number of samples (n)
        algorithm: 'standard_em', 'lookahead_em', or 'squarem'
        seed: Random seed
        max_iter: Maximum iterations
        timeout: Timeout in seconds
        tol: Convergence tolerance

    Returns:
        Result dictionary
    """
    result = {
        'n_experts': n_experts,
        'n_features': n_features,
        'n_samples': n_samples,
        'algorithm': algorithm,
        'seed': seed,
        'converged': False,
        'final_ll': float('-inf'),
        'n_iterations': 0,
        'elapsed_time': 0.0,
        'error': None
    }

    try:
        # Generate data
        X, y, true_experts, theta_true = generate_moe_data(
            n=n_samples,
            n_experts=n_experts,
            n_features=n_features,
            seed=seed
        )
        data = (X, y)

        # Initialize model
        model = MoEModelWrapper(n_experts=n_experts, n_features=n_features)

        # Initialize parameters
        theta_init = initialize_equal_gates(n_experts, n_features, random_state=seed)

        # Create algorithm
        if algorithm == 'standard_em':
            em = StandardEM(model, verbose=False)
        elif algorithm == 'lookahead_em':
            em = LookaheadEM(model, gamma='adaptive', verbose=False)
        elif algorithm == 'squarem':
            em = SQUAREM(model, verbose=False)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Run algorithm
        theta_final, diagnostics = em.fit(
            data, theta_init,
            max_iter=max_iter,
            timeout=timeout,
            tol=tol
        )

        # Record results
        result['converged'] = diagnostics.get('converged', False)
        result['final_ll'] = diagnostics.get('final_likelihood', float('-inf'))
        result['n_iterations'] = diagnostics.get('iterations', 0)
        result['elapsed_time'] = diagnostics.get('time_seconds', 0.0)
        result['stopping_reason'] = diagnostics.get('stopping_reason', 'unknown')

    except Exception as e:
        result['error'] = str(e)

    return result


def run_configuration(
    n_experts: int,
    n_features: int,
    n_samples: int,
    n_restarts: int = 30,
    algorithms: List[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run all algorithms on a single configuration with multiple restarts.

    Args:
        n_experts: Number of experts (K)
        n_features: Number of features (d)
        n_samples: Number of samples (n)
        n_restarts: Number of random restarts
        algorithms: List of algorithms to test
        verbose: Whether to show progress

    Returns:
        List of result dictionaries
    """
    if algorithms is None:
        algorithms = ['standard_em', 'lookahead_em', 'squarem']

    results = []
    total_runs = len(algorithms) * n_restarts

    desc = f"K={n_experts}, d={n_features}, n={n_samples}"
    iterator = range(n_restarts)
    if verbose:
        iterator = tqdm(iterator, desc=desc, leave=False)

    for restart in iterator:
        for algo in algorithms:
            seed = 1000 * n_experts + 100 * n_features + restart
            result = run_single_experiment(
                n_experts=n_experts,
                n_features=n_features,
                n_samples=n_samples,
                algorithm=algo,
                seed=seed
            )
            results.append(result)

    return results


def run_comprehensive_experiment(
    expert_counts: List[int] = None,
    dimensions: List[int] = None,
    sample_sizes: List[int] = None,
    n_restarts: int = 30,
    algorithms: List[str] = None,
    save_results: bool = True,
    results_dir: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive MoE experiment across all configurations.

    Args:
        expert_counts: List of K values (default: [2, 3, 5, 10])
        dimensions: List of d values (default: [2, 5, 10, 20])
        sample_sizes: List of n values (default: [100, 500, 1000])
        n_restarts: Number of restarts per configuration
        algorithms: List of algorithms to test
        save_results: Whether to save results to file
        results_dir: Directory for saving results
        verbose: Whether to print progress

    Returns:
        Dictionary with all results and summary
    """
    if expert_counts is None:
        expert_counts = [2, 3, 5, 10]
    if dimensions is None:
        dimensions = [2, 5, 10, 20]
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000]
    if algorithms is None:
        algorithms = ['standard_em', 'lookahead_em', 'squarem']
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )

    # Calculate total runs
    n_configs = len(expert_counts) * len(dimensions) * len(sample_sizes)
    total_runs = n_configs * len(algorithms) * n_restarts

    if verbose:
        print("=" * 70)
        print("Comprehensive MoE Experiment")
        print("=" * 70)
        print(f"Expert counts (K): {expert_counts}")
        print(f"Dimensions (d): {dimensions}")
        print(f"Sample sizes (n): {sample_sizes}")
        print(f"Algorithms: {algorithms}")
        print(f"Restarts per config: {n_restarts}")
        print(f"Total configurations: {n_configs}")
        print(f"Total runs: {total_runs}")
        print("=" * 70)

    all_results = []
    configs = list(product(expert_counts, dimensions, sample_sizes))

    for K, d, n in tqdm(configs, desc="Configurations", disable=not verbose):
        config_results = run_configuration(
            n_experts=K,
            n_features=d,
            n_samples=n,
            n_restarts=n_restarts,
            algorithms=algorithms,
            verbose=verbose
        )
        all_results.extend(config_results)

    # Generate summary
    summary = generate_summary(all_results)

    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = os.path.join(results_dir, f"moe_comprehensive_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'expert_counts': expert_counts,
                    'dimensions': dimensions,
                    'sample_sizes': sample_sizes,
                    'n_restarts': n_restarts,
                    'algorithms': algorithms
                },
                'results': all_results,
                'summary': summary
            }, f, indent=2, default=str)

        if verbose:
            print(f"\nResults saved to: {results_file}")

    return {'results': all_results, 'summary': summary}


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Summary dictionary with statistics per algorithm and configuration
    """
    summary = {
        'by_algorithm': {},
        'by_config': {},
        'overall': {}
    }

    # Group by algorithm
    by_algo = {}
    for r in results:
        algo = r['algorithm']
        if algo not in by_algo:
            by_algo[algo] = []
        by_algo[algo].append(r)

    for algo, algo_results in by_algo.items():
        successful = [r for r in algo_results if r['error'] is None]
        converged = [r for r in successful if r['converged']]

        final_lls = [r['final_ll'] for r in successful if r['final_ll'] > float('-inf')]
        times = [r['elapsed_time'] for r in successful]
        iters = [r['n_iterations'] for r in successful]

        summary['by_algorithm'][algo] = {
            'n_total': len(algo_results),
            'n_successful': len(successful),
            'n_converged': len(converged),
            'convergence_rate': len(converged) / len(algo_results) if algo_results else 0,
            'final_ll_mean': float(np.mean(final_lls)) if final_lls else float('nan'),
            'final_ll_std': float(np.std(final_lls)) if final_lls else float('nan'),
            'final_ll_best': float(max(final_lls)) if final_lls else float('nan'),
            'time_mean': float(np.mean(times)) if times else float('nan'),
            'time_std': float(np.std(times)) if times else float('nan'),
            'iterations_mean': float(np.mean(iters)) if iters else float('nan'),
            'iterations_std': float(np.std(iters)) if iters else float('nan'),
        }

    # Group by configuration
    by_config = {}
    for r in results:
        config_key = f"K{r['n_experts']}_d{r['n_features']}_n{r['n_samples']}"
        algo = r['algorithm']

        if config_key not in by_config:
            by_config[config_key] = {}
        if algo not in by_config[config_key]:
            by_config[config_key][algo] = []
        by_config[config_key][algo].append(r)

    for config_key, config_data in by_config.items():
        summary['by_config'][config_key] = {}
        for algo, algo_results in config_data.items():
            successful = [r for r in algo_results if r['error'] is None]
            final_lls = [r['final_ll'] for r in successful if r['final_ll'] > float('-inf')]

            summary['by_config'][config_key][algo] = {
                'n_total': len(algo_results),
                'n_converged': sum(1 for r in successful if r['converged']),
                'final_ll_mean': float(np.mean(final_lls)) if final_lls else float('nan'),
                'final_ll_std': float(np.std(final_lls)) if final_lls else float('nan'),
            }

    # Calculate win rates (how often lookahead beats standard EM)
    wins = {'lookahead_em': 0, 'standard_em': 0, 'tie': 0}
    for config_key, config_data in by_config.items():
        if 'lookahead_em' in config_data and 'standard_em' in config_data:
            la_ll = summary['by_config'][config_key]['lookahead_em']['final_ll_mean']
            std_ll = summary['by_config'][config_key]['standard_em']['final_ll_mean']

            if not np.isnan(la_ll) and not np.isnan(std_ll):
                if la_ll > std_ll + 0.1:  # Lookahead wins by margin
                    wins['lookahead_em'] += 1
                elif std_ll > la_ll + 0.1:  # Standard wins by margin
                    wins['standard_em'] += 1
                else:
                    wins['tie'] += 1

    summary['overall']['win_rates'] = wins
    summary['overall']['n_configurations'] = len(by_config)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("OVERALL RESULTS BY ALGORITHM")
    print("=" * 80)

    print(f"\n{'Algorithm':<20} {'Conv%':>8} {'Final LL':>18} {'Time (s)':>14} {'Iters':>12}")
    print("-" * 80)

    for algo in ['lookahead_em', 'standard_em', 'squarem']:
        if algo in summary['by_algorithm']:
            s = summary['by_algorithm'][algo]
            conv_pct = f"{100*s['convergence_rate']:.1f}%"
            ll_str = f"{s['final_ll_mean']:.2f} +/- {s['final_ll_std']:.2f}"
            time_str = f"{s['time_mean']:.2f} +/- {s['time_std']:.2f}"
            iter_str = f"{s['iterations_mean']:.0f} +/- {s['iterations_std']:.0f}"
            print(f"{algo:<20} {conv_pct:>8} {ll_str:>18} {time_str:>14} {iter_str:>12}")

    print("\n" + "=" * 80)
    print("WIN RATES (Lookahead EM vs Standard EM by configuration)")
    print("=" * 80)
    wins = summary['overall']['win_rates']
    total = wins['lookahead_em'] + wins['standard_em'] + wins['tie']
    if total > 0:
        print(f"  Lookahead EM wins: {wins['lookahead_em']}/{total} ({100*wins['lookahead_em']/total:.1f}%)")
        print(f"  Standard EM wins:  {wins['standard_em']}/{total} ({100*wins['standard_em']/total:.1f}%)")
        print(f"  Ties:              {wins['tie']}/{total} ({100*wins['tie']/total:.1f}%)")

    print("\n" + "=" * 80)
    print("RESULTS BY CONFIGURATION")
    print("=" * 80)

    print(f"\n{'Config':<20} {'Lookahead LL':>18} {'Standard LL':>18} {'Winner':>12}")
    print("-" * 80)

    for config_key in sorted(summary['by_config'].keys()):
        config_data = summary['by_config'][config_key]

        la_ll = config_data.get('lookahead_em', {}).get('final_ll_mean', float('nan'))
        std_ll = config_data.get('standard_em', {}).get('final_ll_mean', float('nan'))

        la_str = f"{la_ll:.2f}" if not np.isnan(la_ll) else "N/A"
        std_str = f"{std_ll:.2f}" if not np.isnan(std_ll) else "N/A"

        if not np.isnan(la_ll) and not np.isnan(std_ll):
            if la_ll > std_ll + 0.1:
                winner = "Lookahead"
            elif std_ll > la_ll + 0.1:
                winner = "Standard"
            else:
                winner = "Tie"
        else:
            winner = "N/A"

        print(f"{config_key:<20} {la_str:>18} {std_str:>18} {winner:>12}")

    print("=" * 80)


def run_quick_test(verbose: bool = True) -> Dict[str, Any]:
    """Run a quick test with reduced configurations."""
    return run_comprehensive_experiment(
        expert_counts=[3],
        dimensions=[2, 5],
        sample_sizes=[200],
        n_restarts=5,
        algorithms=['standard_em', 'lookahead_em'],
        save_results=False,
        verbose=verbose
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive MoE Experiment")
    parser.add_argument('--quick', action='store_true', help="Run quick test")
    parser.add_argument('--restarts', type=int, default=30, help="Number of restarts")
    parser.add_argument('--experts', type=int, nargs='+', default=[2, 3, 5, 10],
                        help="Expert counts to test")
    parser.add_argument('--dims', type=int, nargs='+', default=[2, 5, 10, 20],
                        help="Dimensions to test")
    parser.add_argument('--samples', type=int, nargs='+', default=[100, 500, 1000],
                        help="Sample sizes to test")
    parser.add_argument('--no-save', action='store_true', help="Don't save results")

    args = parser.parse_args()

    if args.quick:
        output = run_quick_test()
    else:
        output = run_comprehensive_experiment(
            expert_counts=args.experts,
            dimensions=args.dims,
            sample_sizes=args.samples,
            n_restarts=args.restarts,
            save_results=not args.no_save
        )

    print_summary(output['summary'])
