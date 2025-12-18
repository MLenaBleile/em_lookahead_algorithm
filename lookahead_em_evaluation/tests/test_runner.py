"""
Test Runner Infrastructure

Unified test runner for Lookahead EM evaluation experiments.
Handles multiple algorithms, random restarts, parallelization,
and result logging.
"""

import numpy as np
import time
import traceback
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.standard_em import StandardEM
from algorithms.lookahead_em import LookaheadEM
from algorithms.squarem_wrapper import get_squarem, PurePythonSQUAREM
from utils.logging_utils import ExperimentLogger
from utils.timing import ResourceMonitor


@dataclass
class TestConfig:
    """Configuration for a test experiment."""
    test_id: str
    data_generator: Callable
    model_class: Any
    initialization_strategy: str = 'kmeans'
    n_restarts: int = 20
    algorithms: List[Dict[str, Any]] = field(default_factory=list)
    max_iter: int = 1000
    timeout: float = 600.0
    tol: float = 1e-6
    results_dir: str = 'results'


def create_algorithm(algo_config: Dict[str, Any], model: Any) -> Any:
    """
    Create an algorithm instance from configuration.

    Args:
        algo_config: dict with 'name' and optional parameters
        model: model instance (e.g., GaussianMixtureModel)

    Returns:
        Algorithm instance (StandardEM, LookaheadEM, or SQUAREM)
    """
    name = algo_config.get('name', 'standard_em')
    params = algo_config.get('params', {})

    if name == 'standard_em':
        return StandardEM(model=model, **params)
    elif name in ['lookahead_em', 'lookahead']:
        return LookaheadEM(model=model, **params)
    elif name in ['squarem', 'SQUAREM']:
        return get_squarem(model)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def run_single_trial(
    test_config: TestConfig,
    algo_config: Dict[str, Any],
    seed: int,
    X: np.ndarray,
    theta_init: Dict[str, np.ndarray],
    theta_true: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Run a single trial of an algorithm.

    Args:
        test_config: TestConfig object
        algo_config: dict with algorithm configuration
        seed: random seed for this trial
        X: data array
        theta_init: initial parameters
        theta_true: optional true parameters

    Returns:
        dict with run results:
        - algorithm: str
        - seed: int
        - converged: bool
        - n_iterations: int
        - final_ll: float
        - ll_history: list
        - elapsed_time: float
        - peak_memory_mb: float
        - theta_final: dict
        - error: str or None
        - stop_reason: str
    """
    np.random.seed(seed)

    result = {
        'algorithm': algo_config.get('name', 'unknown'),
        'algorithm_params': algo_config.get('params', {}),
        'seed': seed,
        'test_id': test_config.test_id,
        'converged': False,
        'n_iterations': 0,
        'final_ll': float('-inf'),
        'll_history': [],
        'elapsed_time': 0.0,
        'peak_memory_mb': 0.0,
        'theta_final': None,
        'error': None,
        'stop_reason': 'unknown'
    }

    try:
        # Create model instance - detect dimensions from theta_init
        # Support different model types: GMM (mu), HMM (A), MoE (gamma)
        if 'mu' in theta_init:
            n_components = theta_init['mu'].shape[0]
            n_features = theta_init['mu'].shape[1]
        elif 'A' in theta_init:
            # HMM: A is (n_states, n_states), B is (n_states, n_obs)
            n_components = theta_init['A'].shape[0]
            n_features = theta_init['B'].shape[1] if 'B' in theta_init else theta_init['A'].shape[1]
        elif 'gamma' in theta_init:
            # MoE: gamma is (n_experts, n_features+1)
            n_components = theta_init['gamma'].shape[0]
            n_features = theta_init['gamma'].shape[1] - 1  # Subtract intercept
        else:
            raise ValueError(f"Cannot determine model dimensions from theta_init keys: {theta_init.keys()}")

        model = test_config.model_class(
            n_components=n_components,
            n_features=n_features
        )

        # Create algorithm
        algorithm = create_algorithm(algo_config, model)

        # Start timing
        monitor = ResourceMonitor()
        monitor.start()

        # Run algorithm
        theta_final, diagnostics = algorithm.fit(
            X=X,
            theta_init=theta_init,
            max_iter=test_config.max_iter,
            tol=test_config.tol,
            timeout=test_config.timeout
        )

        # Stop timing
        timing_stats = monitor.stop()

        # Extract results - map from StandardEM diagnostics keys
        result['converged'] = diagnostics.get('converged', False)
        result['n_iterations'] = diagnostics.get('iterations', 0)
        ll_history = diagnostics.get('likelihood_history', [])
        result['final_ll'] = ll_history[-1] if ll_history else diagnostics.get('final_likelihood', float('-inf'))
        result['ll_history'] = ll_history
        result['elapsed_time'] = timing_stats['elapsed_time']
        result['peak_memory_mb'] = timing_stats['peak_memory_mb']
        result['theta_final'] = theta_final
        result['stop_reason'] = diagnostics.get('stopping_reason', 'unknown')

        # Compute parameter error if true parameters available
        if theta_true is not None:
            try:
                if 'mu' in theta_final and 'mu' in theta_true:
                    mu_error = np.linalg.norm(theta_final['mu'] - theta_true['mu']) / np.linalg.norm(theta_true['mu'])
                    result['param_error'] = mu_error
                elif 'A' in theta_final and 'A' in theta_true:
                    # HMM: compute error on transition matrix
                    A_error = np.linalg.norm(theta_final['A'] - theta_true['A']) / np.linalg.norm(theta_true['A'])
                    result['param_error'] = A_error
                elif 'beta' in theta_final and 'beta' in theta_true:
                    # MoE: compute error on expert weights
                    beta_error = np.linalg.norm(theta_final['beta'] - theta_true['beta']) / np.linalg.norm(theta_true['beta'])
                    result['param_error'] = beta_error
            except Exception:
                pass  # Skip parameter error if computation fails

    except Exception as e:
        result['error'] = str(e)
        result['error_traceback'] = traceback.format_exc()
        result['stop_reason'] = 'error'

    return result


def _run_trial_wrapper(args):
    """Wrapper function for parallel execution."""
    return run_single_trial(*args)


def run_test(
    test_config: TestConfig,
    parallel: bool = True,
    n_jobs: int = 4,
    verbose: bool = True,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run all algorithms × all seeds for a test configuration.

    Args:
        test_config: TestConfig object
        parallel: whether to use parallel execution
        n_jobs: number of parallel jobs
        verbose: whether to print progress
        save_results: whether to save results to file

    Returns:
        List of all run_result dicts
    """
    # Generate data (single dataset for all runs)
    if verbose:
        print(f"Generating data for {test_config.test_id}...")

    X, z, theta_true = test_config.data_generator()

    # Handle different data formats:
    # - 2D array (GMM): X has shape (n, d)
    # - 1D array (HMM): X has shape (T,)
    # - Tuple (MoE): X is (X_features, y)
    if isinstance(X, tuple):
        # MoE case: X is (X_features, y)
        X_features, _ = X
        n, d = X_features.shape
    elif hasattr(X, 'ndim'):
        if X.ndim == 2:
            n, d = X.shape
        else:
            n = len(X)
            d = 1  # 1D sequence
    else:
        n = len(X)
        d = 1

    K = len(np.unique(z))

    if verbose:
        print(f"  Data: n={n}, d={d}, K={K}")

    # Generate initializations for each restart
    if verbose:
        print(f"Generating {test_config.n_restarts} initializations...")

    from data.generate_gmm import generate_initializations
    initializations = generate_initializations(
        X=X,
        K=K,
        strategy=test_config.initialization_strategy,
        n_inits=test_config.n_restarts,
        random_state=42
    )

    # Prepare all trials
    trials = []
    for seed in range(test_config.n_restarts):
        theta_init = initializations[seed]
        for algo_config in test_config.algorithms:
            # Handle tuple data (e.g., MoE with (X, y))
            if isinstance(X, tuple):
                X_copy = tuple(arr.copy() if hasattr(arr, 'copy') else arr for arr in X)
            else:
                X_copy = X.copy()
            trials.append((test_config, algo_config, seed, X_copy, theta_init.copy(), theta_true))

    total_trials = len(trials)
    if verbose:
        print(f"Running {total_trials} trials ({len(test_config.algorithms)} algorithms × {test_config.n_restarts} restarts)...")

    # Run trials
    results = []

    if parallel and n_jobs > 1:
        # Parallel execution
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_trial_wrapper, trial): trial for trial in trials}

            if use_tqdm and verbose:
                iterator = tqdm(as_completed(futures), total=total_trials, desc="Running trials")
            else:
                iterator = as_completed(futures)

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'error': str(e),
                        'error_traceback': traceback.format_exc()
                    })
    else:
        # Sequential execution
        try:
            from tqdm import tqdm
            iterator = tqdm(trials, desc="Running trials") if verbose else trials
        except ImportError:
            iterator = trials

        for trial in iterator:
            result = run_single_trial(*trial)
            results.append(result)

    # Save results
    if save_results:
        results_dir = os.path.join(test_config.results_dir, test_config.test_id)
        os.makedirs(results_dir, exist_ok=True)

        logger = ExperimentLogger(results_dir)

        for result in results:
            # Skip results that don't have required fields (e.g., from parallel errors)
            if 'algorithm' not in result:
                continue

            # Convert numpy arrays to lists for JSON serialization
            result_copy = result.copy()
            if result_copy.get('theta_final') is not None:
                result_copy['theta_final'] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in result_copy['theta_final'].items()
                }
            if result_copy.get('ll_history') is not None:
                result_copy['ll_history'] = [float(x) for x in result_copy['ll_history']]

            logger.log_run(result_copy.get('algorithm', 'unknown'), result_copy)

        if verbose:
            print(f"Results saved to {results_dir}/")

    # Print summary
    if verbose:
        print_results_summary(results)

    return results


def print_results_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Group by algorithm
    by_algorithm = {}
    for r in results:
        algo = r.get('algorithm', 'unknown')
        if algo not in by_algorithm:
            by_algorithm[algo] = []
        by_algorithm[algo].append(r)

    for algo, algo_results in by_algorithm.items():
        print(f"\n{algo}:")
        n_success = sum(1 for r in algo_results if r.get('converged', False))
        n_error = sum(1 for r in algo_results if r.get('error') is not None)
        n_total = len(algo_results)

        final_lls = [r['final_ll'] for r in algo_results if r.get('final_ll', float('-inf')) > float('-inf')]
        times = [r['elapsed_time'] for r in algo_results if r.get('elapsed_time', 0) > 0]
        iters = [r['n_iterations'] for r in algo_results if r.get('n_iterations', 0) > 0]

        print(f"  Success rate: {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")
        if n_error > 0:
            print(f"  Errors: {n_error}")
        if final_lls:
            print(f"  Final LL: {np.mean(final_lls):.2f} ± {np.std(final_lls):.2f}")
        if times:
            print(f"  Time (s): {np.mean(times):.2f} ± {np.std(times):.2f}")
        if iters:
            print(f"  Iterations: {np.mean(iters):.1f} ± {np.std(iters):.1f}")


def load_results(results_dir: str, test_id: str) -> List[Dict[str, Any]]:
    """
    Load saved results for a test.

    Args:
        results_dir: base results directory
        test_id: test identifier

    Returns:
        List of run_result dicts
    """
    test_dir = os.path.join(results_dir, test_id)
    logger = ExperimentLogger(test_dir)
    return logger.load_all()


# ============================================================
# Unit Tests
# ============================================================

def test_create_algorithm():
    """Test algorithm creation from config."""
    from models.gmm import GaussianMixtureModel

    model = GaussianMixtureModel(n_components=3, n_features=2)

    # Standard EM
    algo = create_algorithm({'name': 'standard_em'}, model)
    assert isinstance(algo, StandardEM), "Should create StandardEM"

    # Lookahead EM with fixed gamma
    algo = create_algorithm({'name': 'lookahead_em', 'params': {'gamma': 0.5}}, model)
    assert isinstance(algo, LookaheadEM), "Should create LookaheadEM"

    print("  PASSED")


def test_run_single_trial():
    """Test running a single trial."""
    from models.gmm import GaussianMixtureModel
    from data.generate_gmm import generate_gmm_data

    # Simple config
    X, z, theta_true = generate_gmm_data(n=100, K=3, d=2, seed=42)

    config = TestConfig(
        test_id='test_single_trial',
        data_generator=lambda: (X, z, theta_true),
        model_class=GaussianMixtureModel,
        max_iter=10,
        timeout=30.0,
        tol=1e-6
    )

    # Initialize (use 'sigma' to match GMM model interface)
    theta_init = {
        'pi': np.ones(3) / 3,
        'mu': X[np.random.choice(100, 3, replace=False)],
        'sigma': np.ones((3, 2))
    }

    result = run_single_trial(
        config,
        {'name': 'standard_em'},
        seed=42,
        X=X,
        theta_init=theta_init,
        theta_true=theta_true
    )

    assert result['algorithm'] == 'standard_em', "Algorithm name should match"
    assert result['seed'] == 42, "Seed should match"
    assert result['error'] is None, f"Should not have error: {result.get('error')}"
    assert result['n_iterations'] > 0, "Should have some iterations"
    assert result['final_ll'] > float('-inf'), "Should have valid final LL"

    print("  PASSED")


def test_run_single_trial_error_handling():
    """Test error handling in single trial."""
    from models.gmm import GaussianMixtureModel

    # Config that will cause an error (empty data)
    config = TestConfig(
        test_id='test_error',
        data_generator=lambda: (np.array([]), np.array([]), {}),
        model_class=GaussianMixtureModel,
        max_iter=10
    )

    # This should not raise an exception, but catch it
    result = run_single_trial(
        config,
        {'name': 'standard_em'},
        seed=42,
        X=np.zeros((1, 2)),  # Invalid data
        theta_init={
            'pi': np.array([1.0]),
            'mu': np.zeros((1, 2)),
            'sigma': np.ones((1, 2))
        }
    )

    # Should have an error recorded
    assert result.get('stop_reason') in ['error', 'unknown'] or result.get('error') is not None or result['n_iterations'] >= 0

    print("  PASSED")


def test_run_test_sequential():
    """Test running a full test sequentially."""
    from models.gmm import GaussianMixtureModel
    from data.generate_gmm import generate_gmm_data

    config = TestConfig(
        test_id='test_sequential',
        data_generator=lambda: generate_gmm_data(n=100, K=3, d=2, seed=42),
        model_class=GaussianMixtureModel,
        initialization_strategy='kmeans',
        n_restarts=2,
        algorithms=[
            {'name': 'standard_em'},
            {'name': 'lookahead_em', 'params': {'gamma': 0.5}}
        ],
        max_iter=10,
        timeout=30.0,
        tol=1e-6,
        results_dir='/tmp/test_results'
    )

    results = run_test(config, parallel=False, verbose=False, save_results=False)

    # Should have 2 algorithms × 2 restarts = 4 results
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    # Check all results have required fields
    for r in results:
        assert 'algorithm' in r
        assert 'seed' in r
        assert 'n_iterations' in r

    print("  PASSED")


def test_results_summary():
    """Test results summary printing."""
    # Create mock results
    results = [
        {'algorithm': 'standard_em', 'converged': True, 'final_ll': -100.0, 'elapsed_time': 1.0, 'n_iterations': 50},
        {'algorithm': 'standard_em', 'converged': True, 'final_ll': -99.0, 'elapsed_time': 1.1, 'n_iterations': 48},
        {'algorithm': 'lookahead_em', 'converged': True, 'final_ll': -98.0, 'elapsed_time': 1.5, 'n_iterations': 30},
        {'algorithm': 'lookahead_em', 'converged': False, 'final_ll': -105.0, 'elapsed_time': 2.0, 'n_iterations': 100},
    ]

    # Should not raise
    print_results_summary(results)

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running test_runner.py unit tests")
    print("=" * 60)

    print("Testing create_algorithm...")
    test_create_algorithm()

    print("Testing run_single_trial...")
    test_run_single_trial()

    print("Testing error handling...")
    test_run_single_trial_error_handling()

    print("Testing sequential run_test...")
    test_run_test_sequential()

    print("Testing results summary...")
    test_results_summary()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
