"""
Metrics Utilities for EM Algorithm Evaluation

This module provides functions for computing evaluation metrics,
checking convergence properties, and performing statistical analysis.

Example usage:
    >>> metrics = compute_metrics(theta_history, likelihood_history, time_history)
    >>> is_mono, violations = check_monotonicity(likelihood_history)
    >>> effect = cohens_d(times_algorithm1, times_algorithm2)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any


def compute_metrics(
    theta_history: List[Dict[str, np.ndarray]],
    likelihood_history: List[float],
    time_history: List[float],
    theta_true: Optional[Dict[str, np.ndarray]] = None,
    success_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for an EM run.

    Args:
        theta_history: List of parameter dictionaries at each iteration.
        likelihood_history: List of log-likelihood values at each iteration.
        time_history: List of cumulative elapsed times at each iteration.
        theta_true: Optional true parameters for computing error metrics.
        success_threshold: Maximum likelihood difference from best to count
                          as success (default: 0.5).

    Returns:
        Dictionary containing:
            - 'iterations': Number of iterations run
            - 'time': Total time in seconds
            - 'final_likelihood': Final log-likelihood value
            - 'convergence_rate': Average improvement per iteration
            - 'likelihood_history': Copy of likelihood history
            If theta_true provided:
            - 'parameter_error': Frobenius norm of (theta_final - theta_true)
            - 'success': Whether final likelihood is within threshold of best

    Example:
        >>> theta_history = [{'mu': np.array([[0, 0]])}, {'mu': np.array([[1, 0]])}]
        >>> likelihood_history = [-100.0, -95.0]
        >>> time_history = [0.1, 0.2]
        >>> metrics = compute_metrics(theta_history, likelihood_history, time_history)
        >>> print(f"Iterations: {metrics['iterations']}")
        Iterations: 2
    """
    if len(likelihood_history) == 0:
        raise ValueError("likelihood_history cannot be empty")

    metrics = {
        'iterations': len(likelihood_history),
        'time': time_history[-1] if time_history else 0.0,
        'final_likelihood': likelihood_history[-1],
        'convergence_rate': convergence_rate(likelihood_history),
        'likelihood_history': list(likelihood_history)
    }

    if theta_true is not None and len(theta_history) > 0:
        theta_final = theta_history[-1]
        metrics['parameter_error'] = parameter_error(theta_final, theta_true)

        # Success = close to best observed likelihood
        best_likelihood = max(likelihood_history)
        metrics['success'] = (best_likelihood - likelihood_history[-1]) <= success_threshold

    return metrics


def check_monotonicity(
    likelihood_history: List[float],
    tol: float = 1e-8
) -> Tuple[bool, List[int]]:
    """
    Check if likelihood is monotonically increasing.

    EM should monotonically increase likelihood. This function checks for
    violations and returns their locations.

    Args:
        likelihood_history: List of log-likelihood values.
        tol: Tolerance for detecting decreases (default: 1e-8).
             Small numerical decreases within tol are ignored.

    Returns:
        Tuple of:
            - is_monotonic: True if no violations found
            - violations: List of indices where likelihood decreased

    Example:
        >>> history = [-100.0, -95.0, -96.0, -90.0]  # Decrease at index 2
        >>> is_mono, violations = check_monotonicity(history)
        >>> print(f"Monotonic: {is_mono}, Violations at: {violations}")
        Monotonic: False, Violations at: [2]
    """
    if len(likelihood_history) < 2:
        return True, []

    violations = []
    history = np.array(likelihood_history)

    for i in range(1, len(history)):
        if history[i] < history[i - 1] - tol:
            violations.append(i)

    is_monotonic = len(violations) == 0
    return is_monotonic, violations


def parameter_error(
    theta_estimated: Dict[str, np.ndarray],
    theta_true: Dict[str, np.ndarray]
) -> float:
    """
    Compute Frobenius norm of parameter error.

    Computes the combined Frobenius norm across all parameter arrays
    in the theta dictionaries.

    Args:
        theta_estimated: Estimated parameters (dict of arrays).
        theta_true: True parameters (dict of arrays).

    Returns:
        Total Frobenius norm: sqrt(sum of squared differences)

    Example:
        >>> theta_est = {'mu': np.array([[1.0, 0.0]]), 'sigma': np.array([[1.0, 1.0]])}
        >>> theta_true = {'mu': np.array([[0.0, 0.0]]), 'sigma': np.array([[1.0, 1.0]])}
        >>> error = parameter_error(theta_est, theta_true)
        >>> print(f"Error: {error:.4f}")
        Error: 1.0000
    """
    total_squared_error = 0.0

    for key in theta_true:
        if key in theta_estimated:
            est = np.asarray(theta_estimated[key])
            true = np.asarray(theta_true[key])

            # Handle potential shape mismatches
            if est.shape == true.shape:
                diff = est - true
                total_squared_error += np.sum(diff ** 2)
            else:
                # If shapes don't match, try to compute error anyway
                # This handles cases like permuted components
                min_size = min(est.size, true.size)
                diff = est.flatten()[:min_size] - true.flatten()[:min_size]
                total_squared_error += np.sum(diff ** 2)

    return np.sqrt(total_squared_error)


def convergence_rate(likelihood_history: List[float]) -> float:
    """
    Compute average log-improvement per iteration.

    Measures how quickly the algorithm improves likelihood on average.
    Higher values indicate faster convergence.

    Args:
        likelihood_history: List of log-likelihood values.

    Returns:
        Average improvement per iteration. Returns 0.0 if fewer than 2 values.

    Example:
        >>> history = [-100.0, -90.0, -85.0, -83.0]
        >>> rate = convergence_rate(history)
        >>> print(f"Rate: {rate:.2f}")  # Total improvement = 17, over 3 steps
        Rate: 5.67
    """
    if len(likelihood_history) < 2:
        return 0.0

    history = np.array(likelihood_history)
    improvements = np.diff(history)  # ℓ[i+1] - ℓ[i]

    # Only count positive improvements
    positive_improvements = improvements[improvements > 0]

    if len(positive_improvements) == 0:
        return 0.0

    return float(np.mean(positive_improvements))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for comparing two samples.

    Cohen's d measures the standardized difference between two means.
    Interpretation guidelines:
        - |d| > 0.8: Large effect
        - |d| > 0.5: Medium effect
        - |d| > 0.2: Small effect

    Args:
        x: First sample (array-like).
        y: Second sample (array-like).

    Returns:
        Cohen's d effect size.

    Example:
        >>> times_fast = np.array([10, 12, 11, 13, 10])
        >>> times_slow = np.array([20, 22, 21, 23, 20])
        >>> d = cohens_d(times_fast, times_slow)
        >>> print(f"Effect size: {d:.2f}")  # Large negative effect
    """
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Pooled standard deviation
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    # Pooled variance (weighted by degrees of freedom)
    pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-10:
        return 0.0

    d = (mean_x - mean_y) / pooled_std
    return float(d)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Uses the percentile method to compute a confidence interval by
    resampling with replacement.

    Args:
        data: Input data array.
        statistic: Function to compute on each resample (default: np.mean).
        n_resamples: Number of bootstrap resamples (default: 10000).
        confidence: Confidence level (default: 0.95 for 95% CI).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound).

    Example:
        >>> data = np.array([10, 12, 11, 13, 10, 14, 12, 11])
        >>> lower, upper = bootstrap_ci(data, statistic=np.mean)
        >>> print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    data = np.asarray(data)
    n = len(data)

    if n == 0:
        return (np.nan, np.nan)

    rng = np.random.RandomState(random_state)

    # Generate bootstrap statistics
    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)

    return (float(lower), float(upper))


def compute_success_rate(
    final_likelihoods: List[float],
    best_likelihood: Optional[float] = None,
    threshold: float = 0.5
) -> float:
    """
    Compute success rate across multiple runs.

    A run is successful if its final likelihood is within threshold
    of the best observed likelihood.

    Args:
        final_likelihoods: List of final likelihood values from multiple runs.
        best_likelihood: Best likelihood to compare against (default: max of inputs).
        threshold: Maximum difference from best to count as success.

    Returns:
        Success rate in [0, 1].

    Example:
        >>> likelihoods = [-100.0, -100.2, -105.0, -100.1]
        >>> rate = compute_success_rate(likelihoods)
        >>> print(f"Success rate: {rate:.0%}")
        Success rate: 75%
    """
    if len(final_likelihoods) == 0:
        return 0.0

    if best_likelihood is None:
        best_likelihood = max(final_likelihoods)

    successes = sum(1 for ll in final_likelihoods if (best_likelihood - ll) <= threshold)
    return successes / len(final_likelihoods)


def compute_aggregate_statistics(
    run_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute aggregate statistics across multiple runs.

    Args:
        run_results: List of run result dictionaries, each containing
                    at minimum 'time', 'iterations', 'final_likelihood'.

    Returns:
        Dictionary with mean, std, min, max for each metric.

    Example:
        >>> results = [
        ...     {'time': 10.0, 'iterations': 50, 'final_likelihood': -100.0},
        ...     {'time': 12.0, 'iterations': 60, 'final_likelihood': -99.5},
        ... ]
        >>> stats = compute_aggregate_statistics(results)
        >>> print(f"Mean time: {stats['mean_time']:.1f}s")
    """
    if len(run_results) == 0:
        return {}

    # Extract metrics from runs
    times = [r.get('time', r.get('time_seconds', 0)) for r in run_results]
    iterations = [r.get('iterations', 0) for r in run_results]
    likelihoods = [r.get('final_likelihood', 0) for r in run_results]

    times = np.array(times)
    iterations = np.array(iterations)
    likelihoods = np.array(likelihoods)

    stats = {
        'n_runs': len(run_results),

        # Time statistics
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),

        # Iteration statistics
        'mean_iterations': float(np.mean(iterations)),
        'std_iterations': float(np.std(iterations)),
        'min_iterations': int(np.min(iterations)),
        'max_iterations': int(np.max(iterations)),

        # Likelihood statistics
        'mean_final_likelihood': float(np.mean(likelihoods)),
        'std_final_likelihood': float(np.std(likelihoods)),
        'best_likelihood': float(np.max(likelihoods)),
        'worst_likelihood': float(np.min(likelihoods)),

        # Success rate
        'success_rate': compute_success_rate(likelihoods.tolist())
    }

    # Add memory stats if available
    if 'peak_memory_mb' in run_results[0]:
        memories = [r.get('peak_memory_mb', 0) for r in run_results]
        stats['mean_memory_mb'] = float(np.mean(memories))
        stats['std_memory_mb'] = float(np.std(memories))

    return stats


# =============================================================================
# Unit Tests
# =============================================================================

def test_compute_metrics():
    """Test compute_metrics function."""
    print("Testing compute_metrics...")

    theta_history = [
        {'mu': np.array([[0.0, 0.0]])},
        {'mu': np.array([[0.5, 0.0]])},
        {'mu': np.array([[1.0, 0.0]])}
    ]
    likelihood_history = [-100.0, -90.0, -85.0]
    time_history = [0.1, 0.2, 0.3]

    metrics = compute_metrics(theta_history, likelihood_history, time_history)

    assert metrics['iterations'] == 3, f"Expected 3 iterations, got {metrics['iterations']}"
    assert metrics['time'] == 0.3, f"Expected time 0.3, got {metrics['time']}"
    assert metrics['final_likelihood'] == -85.0, "Wrong final likelihood"
    assert metrics['convergence_rate'] > 0, "Convergence rate should be positive"

    # Test with theta_true
    theta_true = {'mu': np.array([[1.0, 0.0]])}
    metrics = compute_metrics(theta_history, likelihood_history, time_history, theta_true)

    assert 'parameter_error' in metrics, "Should have parameter_error"
    assert 'success' in metrics, "Should have success flag"
    assert metrics['parameter_error'] < 0.01, "Parameter error should be near 0"
    assert metrics['success'] is True, "Should be successful"

    print("  PASSED")


def test_check_monotonicity():
    """Test check_monotonicity function."""
    print("Testing check_monotonicity...")

    # Monotonic sequence
    mono_history = [-100.0, -95.0, -90.0, -85.0]
    is_mono, violations = check_monotonicity(mono_history)
    assert is_mono is True, "Should be monotonic"
    assert len(violations) == 0, "Should have no violations"

    # Non-monotonic sequence
    non_mono_history = [-100.0, -95.0, -96.0, -85.0]
    is_mono, violations = check_monotonicity(non_mono_history)
    assert is_mono is False, "Should not be monotonic"
    assert violations == [2], f"Violation at index 2, got {violations}"

    # Multiple violations
    multi_violations = [-100.0, -95.0, -96.0, -90.0, -92.0]
    is_mono, violations = check_monotonicity(multi_violations)
    assert violations == [2, 4], f"Expected [2, 4], got {violations}"

    # Within tolerance
    within_tol = [-100.0, -100.0 + 1e-9, -100.0]
    is_mono, violations = check_monotonicity(within_tol, tol=1e-8)
    assert is_mono is True, "Small fluctuations within tolerance"

    print("  PASSED")


def test_parameter_error():
    """Test parameter_error function."""
    print("Testing parameter_error...")

    theta_est = {
        'mu': np.array([[1.0, 0.0], [0.0, 1.0]]),
        'sigma': np.array([[1.0, 1.0], [1.0, 1.0]])
    }
    theta_true = {
        'mu': np.array([[0.0, 0.0], [0.0, 0.0]]),
        'sigma': np.array([[1.0, 1.0], [1.0, 1.0]])
    }

    error = parameter_error(theta_est, theta_true)
    expected = np.sqrt(1**2 + 1**2)  # sqrt(2) for the mu differences
    assert abs(error - expected) < 1e-10, f"Expected {expected}, got {error}"

    # Zero error
    theta_same = {'mu': np.array([[1.0, 2.0]])}
    error = parameter_error(theta_same, theta_same)
    assert error == 0.0, "Same parameters should have zero error"

    print("  PASSED")


def test_convergence_rate():
    """Test convergence_rate function."""
    print("Testing convergence_rate...")

    # Steady improvement
    history = [-100.0, -90.0, -80.0, -70.0]
    rate = convergence_rate(history)
    assert rate == 10.0, f"Expected 10.0, got {rate}"

    # Mixed improvements
    history = [-100.0, -90.0, -90.0, -80.0]  # Only 2 positive improvements
    rate = convergence_rate(history)
    assert rate == 10.0, f"Expected 10.0, got {rate}"

    # Single value
    rate = convergence_rate([-100.0])
    assert rate == 0.0, "Single value should return 0"

    print("  PASSED")


def test_cohens_d():
    """Test cohens_d function."""
    print("Testing cohens_d...")

    # Large effect (d > 0.8)
    x = np.array([10, 11, 12, 10, 11])
    y = np.array([20, 21, 22, 20, 21])
    d = cohens_d(x, y)
    assert abs(d) > 0.8, f"Should be large effect, got {d}"
    assert d < 0, "x < y means negative d"

    # No effect (same means)
    x = np.array([10, 11, 12, 10, 11])
    y = np.array([10, 11, 12, 10, 11])
    d = cohens_d(x, y)
    assert abs(d) < 0.01, f"Same data should have d ≈ 0, got {d}"

    print("  PASSED")


def test_bootstrap_ci():
    """Test bootstrap_ci function."""
    print("Testing bootstrap_ci...")

    np.random.seed(42)
    data = np.random.normal(10, 2, size=100)

    lower, upper = bootstrap_ci(data, n_resamples=5000, random_state=42)

    # True mean is 10, CI should contain it
    assert lower < 10 < upper, f"CI [{lower}, {upper}] should contain true mean 10"

    # CI width should be reasonable
    ci_width = upper - lower
    assert 0.5 < ci_width < 2.0, f"CI width {ci_width} seems unreasonable"

    print(f"  95% CI: [{lower:.3f}, {upper:.3f}]")
    print("  PASSED")


def test_success_rate():
    """Test compute_success_rate function."""
    print("Testing compute_success_rate...")

    # 3 out of 4 within threshold
    likelihoods = [-100.0, -100.2, -105.0, -100.1]
    rate = compute_success_rate(likelihoods, threshold=0.5)
    assert rate == 0.75, f"Expected 0.75, got {rate}"

    # All successful
    likelihoods = [-100.0, -100.1, -99.9, -100.2]
    rate = compute_success_rate(likelihoods, threshold=0.5)
    assert rate == 1.0, f"Expected 1.0, got {rate}"

    # None successful (all far from best)
    likelihoods = [-100.0, -110.0, -120.0]
    rate = compute_success_rate(likelihoods, threshold=0.5)
    assert rate == 1/3, f"Expected {1/3}, got {rate}"

    print("  PASSED")


def test_aggregate_statistics():
    """Test compute_aggregate_statistics function."""
    print("Testing compute_aggregate_statistics...")

    run_results = [
        {'time': 10.0, 'iterations': 50, 'final_likelihood': -100.0, 'peak_memory_mb': 100},
        {'time': 12.0, 'iterations': 60, 'final_likelihood': -99.5, 'peak_memory_mb': 110},
        {'time': 11.0, 'iterations': 55, 'final_likelihood': -99.8, 'peak_memory_mb': 105},
    ]

    stats = compute_aggregate_statistics(run_results)

    assert stats['n_runs'] == 3, f"Expected 3 runs, got {stats['n_runs']}"
    assert stats['mean_time'] == 11.0, f"Expected mean time 11.0, got {stats['mean_time']}"
    assert stats['best_likelihood'] == -99.5, "Wrong best likelihood"
    assert 'mean_memory_mb' in stats, "Should have memory stats"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running metrics.py unit tests")
    print("=" * 60)

    test_compute_metrics()
    test_check_monotonicity()
    test_parameter_error()
    test_convergence_rate()
    test_cohens_d()
    test_bootstrap_ci()
    test_success_rate()
    test_aggregate_statistics()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
