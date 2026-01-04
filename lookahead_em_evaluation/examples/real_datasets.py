"""
Real Dataset Demonstrations for MoE Lookahead EM

This module demonstrates the practical advantages of the lookahead EM algorithm
on real-world datasets. The key insight is that better likelihood optimization
can lead to fundamentally different conclusions about the data:

1. Model Selection: Different optimal number of experts
2. Expert Assignment: Different data partitioning
3. Prediction Quality: Better generalization from better optimization

Datasets:
- California Housing: Geographic market segments
- Wine Quality: Different quality tier patterns
- Auto MPG: Vehicle type regimes (economy vs performance)
- Diabetes: Patient subgroup patterns
"""

import os
import sys
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add package root to path
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, os.path.dirname(pkg_root))

from lookahead_em_evaluation.models.mixture_of_experts import (
    MixtureOfExperts, initialize_random, initialize_equal_gates
)
from lookahead_em_evaluation.algorithms.standard_em import StandardEM
from lookahead_em_evaluation.algorithms.lookahead_em import LookaheadEM


@dataclass
class DatasetResult:
    """Results from running both algorithms on a dataset."""
    dataset_name: str
    n_experts: int
    n_samples: int
    n_features: int

    # Algorithm results
    standard_ll: float
    lookahead_ll: float
    ll_improvement: float
    ll_improvement_pct: float

    # Expert utilization
    standard_expert_counts: np.ndarray
    lookahead_expert_counts: np.ndarray
    standard_n_active: int
    lookahead_n_active: int

    # Prediction quality (if test set provided)
    standard_rmse: Optional[float] = None
    lookahead_rmse: Optional[float] = None

    # Parameters
    standard_theta: Optional[Dict] = None
    lookahead_theta: Optional[Dict] = None

    # Timing
    standard_time: float = 0.0
    lookahead_time: float = 0.0


def load_california_housing() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load California Housing dataset.

    This dataset contains median house values for California districts.
    Different geographic regions (coastal, inland, urban, rural) may exhibit
    different pricing dynamics - a natural fit for MoE.

    Returns:
        X: Features (n, 8)
        y: Target (median house value)
        name: Dataset name
    """
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data, data.target

        # Standardize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        # Standardize target
        y = (y - y.mean()) / (y.std() + 1e-8)

        return X, y, "California Housing"
    except Exception as e:
        print(f"Could not load California Housing: {e}")
        return None, None, None


def load_wine_quality() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Wine Quality dataset.

    Different quality tiers (low, medium, high) may have different
    feature relationships - e.g., what makes a mediocre wine might
    differ from what distinguishes excellent wines.

    Returns:
        X: Features
        y: Quality score
        name: Dataset name
    """
    try:
        from sklearn.datasets import load_wine
        data = load_wine()
        X, y = data.data, data.target.astype(float)

        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        return X, y, "Wine"
    except Exception as e:
        print(f"Could not load Wine: {e}")
        return None, None, None


def load_diabetes() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Diabetes dataset.

    Disease progression may follow different patterns for different
    patient subgroups based on their characteristics.

    Returns:
        X: Features (10 baseline variables)
        y: Disease progression measure
        name: Dataset name
    """
    try:
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X, y = data.data, data.target

        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        return X, y, "Diabetes"
    except Exception as e:
        print(f"Could not load Diabetes: {e}")
        return None, None, None


def load_auto_mpg() -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Auto MPG dataset.

    Different vehicle types (economy, performance, trucks) have
    fundamentally different fuel efficiency characteristics.

    Returns:
        X: Features (displacement, horsepower, weight, acceleration)
        y: MPG
        name: Dataset name
    """
    try:
        # Try OpenML
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name='autoMpg', version=1, as_frame=False, parser='auto')
        X, y = data.data, data.target.astype(float)

        # Remove NaN rows
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[valid], y[valid]

        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        return X, y, "Auto MPG"
    except Exception as e:
        print(f"Could not load Auto MPG: {e}")
        return None, None, None


def create_synthetic_multimodal(
    n_samples: int = 1000,
    n_features: int = 4,
    n_true_experts: int = 4,
    noise_level: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Create synthetic dataset with clear multimodal structure.

    This provides ground truth for validating that lookahead EM
    finds the correct structure more often.

    Returns:
        X: Features
        y: Target
        name: Dataset name
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n_samples, n_features)
    y = np.zeros(n_samples)

    # Create distinct expert regions
    samples_per_expert = n_samples // n_true_experts

    for g in range(n_true_experts):
        start = g * samples_per_expert
        end = start + samples_per_expert if g < n_true_experts - 1 else n_samples

        # Each expert has different coefficients
        beta = rng.randn(n_features) * (g + 1)  # Varying magnitudes
        intercept = g * 2.0  # Different intercepts

        y[start:end] = X[start:end] @ beta + intercept + noise_level * rng.randn(end - start)

    # Shuffle to mix experts
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    return X, y, f"Synthetic (k={n_true_experts})"


def compute_expert_assignments(
    X: np.ndarray,
    y: np.ndarray,
    theta: Dict[str, np.ndarray],
    model: MixtureOfExperts
) -> np.ndarray:
    """Compute hard expert assignments from responsibilities."""
    resp = model.e_step((X, y), theta)
    return np.argmax(resp, axis=1)


def count_active_experts(assignments: np.ndarray, threshold: float = 0.01) -> int:
    """Count experts with more than threshold fraction of data."""
    n = len(assignments)
    counts = np.bincount(assignments)
    return np.sum(counts > threshold * n)


def run_single_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_experts: int,
    n_restarts: int = 10,
    max_iter: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> DatasetResult:
    """
    Run standard EM and lookahead EM on a dataset.

    Args:
        X: Features (n, d)
        y: Target (n,)
        dataset_name: Name for reporting
        n_experts: Number of experts to fit
        n_restarts: Number of random restarts
        max_iter: Maximum iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        DatasetResult with comparison
    """
    import time
    n_samples, n_features = X.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples: {n_samples}, Features: {n_features}, Experts: {n_experts}")
        print(f"{'='*60}")

    # Track best results
    best_standard_ll = float('-inf')
    best_lookahead_ll = float('-inf')
    best_standard_theta = None
    best_lookahead_theta = None
    standard_time = 0.0
    lookahead_time = 0.0

    model = MixtureOfExperts(n_experts=n_experts, n_features=n_features)
    data = (X, y)

    for restart in range(n_restarts):
        # Initialize
        theta_init = initialize_random(n_experts, n_features, random_state=seed + restart)

        # Standard EM
        t0 = time.time()
        theta_std = theta_init.copy()
        for _ in range(max_iter):
            resp = model.e_step(data, theta_std)
            theta_std = model.m_step(data, resp, theta_std)
        standard_time += time.time() - t0
        ll_std = model.log_likelihood(data, theta_std)

        if ll_std > best_standard_ll:
            best_standard_ll = ll_std
            best_standard_theta = theta_std

        # Lookahead EM
        t0 = time.time()
        try:
            em = LookaheadEM(
                model=model,
                gamma='adaptive',
                verbose=False
            )
            theta_la, diagnostics = em.fit(
                data,
                theta_init.copy(),
                max_iter=max_iter,
                tol=1e-6
            )
            ll_la = model.log_likelihood(data, theta_la)
        except Exception as e:
            # Fallback if lookahead fails
            if verbose:
                print(f"  Lookahead failed on restart {restart}: {e}")
            ll_la = ll_std
            theta_la = theta_std
        lookahead_time += time.time() - t0

        if ll_la > best_lookahead_ll:
            best_lookahead_ll = ll_la
            best_lookahead_theta = theta_la

        if verbose and (restart + 1) % 5 == 0:
            print(f"  Restart {restart + 1}/{n_restarts}: "
                  f"Std LL={ll_std:.2f}, LA LL={ll_la:.2f}")

    # Compute expert assignments
    std_assignments = compute_expert_assignments(X, y, best_standard_theta, model)
    la_assignments = compute_expert_assignments(X, y, best_lookahead_theta, model)

    std_counts = np.bincount(std_assignments, minlength=n_experts)
    la_counts = np.bincount(la_assignments, minlength=n_experts)

    std_n_active = count_active_experts(std_assignments)
    la_n_active = count_active_experts(la_assignments)

    # Compute improvement
    ll_improvement = best_lookahead_ll - best_standard_ll
    ll_improvement_pct = 100 * ll_improvement / abs(best_standard_ll) if best_standard_ll != 0 else 0

    result = DatasetResult(
        dataset_name=dataset_name,
        n_experts=n_experts,
        n_samples=n_samples,
        n_features=n_features,
        standard_ll=best_standard_ll,
        lookahead_ll=best_lookahead_ll,
        ll_improvement=ll_improvement,
        ll_improvement_pct=ll_improvement_pct,
        standard_expert_counts=std_counts,
        lookahead_expert_counts=la_counts,
        standard_n_active=std_n_active,
        lookahead_n_active=la_n_active,
        standard_theta=best_standard_theta,
        lookahead_theta=best_lookahead_theta,
        standard_time=standard_time,
        lookahead_time=lookahead_time
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Standard EM:  LL = {best_standard_ll:.2f} ({standard_time:.1f}s)")
        print(f"  Lookahead EM: LL = {best_lookahead_ll:.2f} ({lookahead_time:.1f}s)")
        print(f"  Improvement:  {ll_improvement:.2f} ({ll_improvement_pct:.2f}%)")
        print(f"\nExpert Utilization:")
        print(f"  Standard:  {std_counts} ({std_n_active} active)")
        print(f"  Lookahead: {la_counts} ({la_n_active} active)")

    return result


def run_model_selection_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    expert_range: range = range(2, 8),
    n_restarts: int = 5,
    max_iter: int = 150,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Model selection experiment: Find optimal number of experts.

    This demonstrates how better likelihood optimization can lead
    to different conclusions about model complexity.

    Uses BIC for model selection: BIC = -2*LL + k*log(n)
    where k = number of parameters

    Returns:
        Dictionary with model selection results
    """
    n_samples, n_features = X.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL SELECTION: {dataset_name}")
        print(f"Testing experts: {list(expert_range)}")
        print(f"{'='*60}")

    results = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'expert_range': list(expert_range),
        'standard_lls': [],
        'lookahead_lls': [],
        'standard_bics': [],
        'lookahead_bics': [],
    }

    for n_experts in expert_range:
        if verbose:
            print(f"\nFitting {n_experts} experts...")

        result = run_single_experiment(
            X, y, dataset_name, n_experts,
            n_restarts=n_restarts,
            max_iter=max_iter,
            seed=seed,
            verbose=False
        )

        # Compute BIC
        # Parameters: G*(d+1) for gamma + G*(d+1) for beta + G for sigma
        n_params = n_experts * (n_features + 1) * 2 + n_experts
        std_bic = -2 * result.standard_ll + n_params * np.log(n_samples)
        la_bic = -2 * result.lookahead_ll + n_params * np.log(n_samples)

        results['standard_lls'].append(result.standard_ll)
        results['lookahead_lls'].append(result.lookahead_ll)
        results['standard_bics'].append(std_bic)
        results['lookahead_bics'].append(la_bic)

        if verbose:
            print(f"  k={n_experts}: Std BIC={std_bic:.1f}, LA BIC={la_bic:.1f}")

    # Find optimal
    std_optimal = expert_range[np.argmin(results['standard_bics'])]
    la_optimal = expert_range[np.argmin(results['lookahead_bics'])]

    results['standard_optimal_k'] = std_optimal
    results['lookahead_optimal_k'] = la_optimal
    results['different_conclusion'] = std_optimal != la_optimal

    if verbose:
        print(f"\n{'='*60}")
        print(f"CONCLUSION:")
        print(f"  Standard EM selects: {std_optimal} experts")
        print(f"  Lookahead EM selects: {la_optimal} experts")
        if results['different_conclusion']:
            print(f"  *** DIFFERENT CONCLUSIONS! ***")
        print(f"{'='*60}")

    return results


def run_holdout_prediction_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_experts: int = 4,
    test_size: float = 0.2,
    n_restarts: int = 10,
    max_iter: int = 200,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Holdout prediction experiment.

    Demonstrates that better likelihood on training data can lead
    to better generalization on unseen test data.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"HOLDOUT PREDICTION: {dataset_name}")
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        print(f"{'='*60}")

    # Train both methods
    result = run_single_experiment(
        X_train, y_train, dataset_name, n_experts,
        n_restarts=n_restarts,
        max_iter=max_iter,
        seed=seed,
        verbose=verbose
    )

    # Predict on test set
    n_features = X.shape[1]
    model = MixtureOfExperts(n_experts=n_experts, n_features=n_features)

    y_pred_std = model.predict(X_test, result.standard_theta)
    y_pred_la = model.predict(X_test, result.lookahead_theta)

    rmse_std = np.sqrt(np.mean((y_test - y_pred_std) ** 2))
    rmse_la = np.sqrt(np.mean((y_test - y_pred_la) ** 2))

    if verbose:
        print(f"\nTest Set Performance:")
        print(f"  Standard EM RMSE:  {rmse_std:.4f}")
        print(f"  Lookahead EM RMSE: {rmse_la:.4f}")
        print(f"  Improvement: {100*(rmse_std - rmse_la)/rmse_std:.2f}%")

    return {
        'dataset': dataset_name,
        'train_result': result,
        'standard_rmse': rmse_std,
        'lookahead_rmse': rmse_la,
        'rmse_improvement_pct': 100 * (rmse_std - rmse_la) / rmse_std
    }


def run_all_real_datasets(
    n_experts: int = 4,
    n_restarts: int = 10,
    verbose: bool = True
) -> List[DatasetResult]:
    """
    Run experiments on all available real datasets.
    """
    results = []

    # Try loading each dataset
    datasets = [
        load_california_housing,
        load_wine_quality,
        load_diabetes,
    ]

    for loader in datasets:
        X, y, name = loader()
        if X is not None:
            # Subsample if too large
            if len(y) > 3000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(y), 3000, replace=False)
                X, y = X[idx], y[idx]

            result = run_single_experiment(
                X, y, name, n_experts,
                n_restarts=n_restarts,
                verbose=verbose
            )
            results.append(result)

    return results


def print_summary_table(results: List[DatasetResult]) -> None:
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("SUMMARY OF REAL DATASET EXPERIMENTS")
    print("=" * 90)

    print(f"\n{'Dataset':<20} {'n':<8} {'d':<4} {'k':<4} "
          f"{'Std LL':>12} {'LA LL':>12} {'Δ LL':>10} {'Δ %':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r.dataset_name:<20} {r.n_samples:<8} {r.n_features:<4} {r.n_experts:<4} "
              f"{r.standard_ll:>12.2f} {r.lookahead_ll:>12.2f} "
              f"{r.ll_improvement:>10.2f} {r.ll_improvement_pct:>7.2f}%")

    print("=" * 90)

    # Expert utilization comparison
    print("\nEXPERT UTILIZATION (counts per expert):")
    print("-" * 90)
    for r in results:
        print(f"{r.dataset_name:<20}")
        print(f"  Standard:  {r.standard_expert_counts} ({r.standard_n_active} active)")
        print(f"  Lookahead: {r.lookahead_expert_counts} ({r.lookahead_n_active} active)")

    print("=" * 90)


def demonstrate_different_conclusions(
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main demonstration showing cases where lookahead EM leads to
    different conclusions than standard EM.

    Focuses on:
    1. Model selection (choosing different number of experts)
    2. Expert structure (different data partitioning)
    3. Prediction quality
    """
    warnings.filterwarnings('ignore')

    results = {
        'single_experiments': [],
        'model_selection': [],
        'prediction': [],
        'different_conclusions_found': False
    }

    # 1. Basic experiments on real datasets
    if verbose:
        print("\n" + "#" * 70)
        print("# PART 1: Real Dataset Comparisons")
        print("#" * 70)

    real_results = run_all_real_datasets(n_experts=4, n_restarts=10, verbose=verbose)
    results['single_experiments'] = real_results

    # 2. Model selection experiment
    if verbose:
        print("\n" + "#" * 70)
        print("# PART 2: Model Selection (Finding Optimal # of Experts)")
        print("#" * 70)

    # Use diabetes for model selection (moderate size, clear structure)
    X, y, name = load_diabetes()
    if X is not None:
        ms_result = run_model_selection_experiment(
            X, y, name,
            expert_range=range(2, 7),
            n_restarts=5,
            verbose=verbose
        )
        results['model_selection'].append(ms_result)
        if ms_result['different_conclusion']:
            results['different_conclusions_found'] = True

    # Synthetic with known structure
    X, y, name = create_synthetic_multimodal(n_samples=800, n_true_experts=4, seed=seed)
    ms_result = run_model_selection_experiment(
        X, y, name,
        expert_range=range(2, 7),
        n_restarts=5,
        verbose=verbose
    )
    results['model_selection'].append(ms_result)
    if ms_result['different_conclusion']:
        results['different_conclusions_found'] = True

    # 3. Holdout prediction
    if verbose:
        print("\n" + "#" * 70)
        print("# PART 3: Holdout Prediction (Generalization)")
        print("#" * 70)

    X, y, name = load_california_housing()
    if X is not None:
        # Subsample for speed
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), 2000, replace=False)
        X, y = X[idx], y[idx]

        pred_result = run_holdout_prediction_experiment(
            X, y, name,
            n_experts=5,
            n_restarts=8,
            verbose=verbose
        )
        results['prediction'].append(pred_result)

    # Summary
    if verbose:
        print("\n" + "#" * 70)
        print("# FINAL SUMMARY")
        print("#" * 70)

        print_summary_table(real_results)

        print("\nMODEL SELECTION RESULTS:")
        for ms in results['model_selection']:
            print(f"\n  {ms['dataset']}:")
            print(f"    Standard EM optimal k: {ms['standard_optimal_k']}")
            print(f"    Lookahead EM optimal k: {ms['lookahead_optimal_k']}")
            if ms['different_conclusion']:
                print(f"    >>> DIFFERENT CONCLUSION <<<")

        print("\nPREDICTION RESULTS:")
        for pr in results['prediction']:
            print(f"\n  {pr['dataset']}:")
            print(f"    Standard EM RMSE: {pr['standard_rmse']:.4f}")
            print(f"    Lookahead EM RMSE: {pr['lookahead_rmse']:.4f}")
            print(f"    Improvement: {pr['rmse_improvement_pct']:.2f}%")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Real Dataset Demonstrations for MoE Lookahead EM"
    )
    parser.add_argument('--demo', action='store_true',
                        help="Run full demonstration")
    parser.add_argument('--single', type=str, default=None,
                        help="Run single dataset (california, wine, diabetes, synthetic)")
    parser.add_argument('--model-selection', action='store_true',
                        help="Run model selection experiment")
    parser.add_argument('--prediction', action='store_true',
                        help="Run holdout prediction experiment")
    parser.add_argument('--experts', type=int, default=4,
                        help="Number of experts (default: 4)")
    parser.add_argument('--restarts', type=int, default=10,
                        help="Number of restarts (default: 10)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument('--quiet', action='store_true',
                        help="Reduce output")

    args = parser.parse_args()

    verbose = not args.quiet

    if args.demo:
        results = demonstrate_different_conclusions(seed=args.seed, verbose=verbose)
    elif args.single:
        loaders = {
            'california': load_california_housing,
            'wine': load_wine_quality,
            'diabetes': load_diabetes,
            'synthetic': lambda: create_synthetic_multimodal(seed=args.seed)
        }
        if args.single in loaders:
            X, y, name = loaders[args.single]()
            if X is not None:
                run_single_experiment(
                    X, y, name, args.experts,
                    n_restarts=args.restarts,
                    verbose=verbose
                )
        else:
            print(f"Unknown dataset: {args.single}")
            print(f"Available: {list(loaders.keys())}")
    elif args.model_selection:
        X, y, name = load_diabetes()
        if X is not None:
            run_model_selection_experiment(
                X, y, name,
                expert_range=range(2, 7),
                n_restarts=args.restarts,
                verbose=verbose
            )
    elif args.prediction:
        X, y, name = load_california_housing()
        if X is not None:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(y), 2000, replace=False)
            run_holdout_prediction_experiment(
                X[idx], y[idx], name,
                n_experts=args.experts,
                n_restarts=args.restarts,
                verbose=verbose
            )
    else:
        print("Use --demo, --single, --model-selection, or --prediction")
        print("Run with --help for more options")
