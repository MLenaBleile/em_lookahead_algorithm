"""
Standard EM Algorithm Implementation

This module provides the baseline EM algorithm for comparison with
accelerated methods like Lookahead EM and SQUAREM.

Example usage:
    >>> from models.gmm import GaussianMixtureModel
    >>> model = GaussianMixtureModel(n_components=5, n_features=2)
    >>> em = StandardEM(model)
    >>> theta_final, diagnostics = em.fit(X, theta_init)
"""

import time
import copy
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from lookahead_em_evaluation.utils.timing import ResourceMonitor


class StandardEM:
    """
    Standard Expectation-Maximization algorithm.

    Implements the classic EM algorithm with configurable stopping conditions
    and comprehensive diagnostics tracking.

    Attributes:
        model: Statistical model providing e_step, m_step, and log_likelihood.
        verbose: Whether to print progress during fitting.
        verbose_interval: Iterations between progress prints.

    Example:
        >>> model = GaussianMixtureModel(n_components=3, n_features=2)
        >>> em = StandardEM(model, verbose=True)
        >>> theta, diag = em.fit(X, theta_init, max_iter=100, tol=1e-6)
        >>> print(f"Converged in {diag['iterations']} iterations")
    """

    def __init__(
        self,
        model: Any,
        verbose: bool = False,
        verbose_interval: int = 10
    ):
        """
        Initialize StandardEM.

        Args:
            model: Statistical model object with methods:
                   - e_step(X, theta) -> responsibilities
                   - m_step(X, responsibilities) -> theta_new
                   - log_likelihood(X, theta) -> float
            verbose: If True, print progress during fitting.
            verbose_interval: Print progress every N iterations.
        """
        self.model = model
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(
        self,
        X: np.ndarray,
        theta_init: Dict[str, np.ndarray],
        max_iter: int = 1000,
        tol: float = 1e-6,
        timeout: float = 600,
        track_history: bool = True,
        stagnation_iters: int = 5,
        stagnation_tol: float = 1e-8
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Fit the model using EM algorithm.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            theta_init: Initial parameter dictionary.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance for parameter change (L-infinity norm).
            timeout: Maximum time in seconds.
            track_history: Whether to store full history (uses more memory).
            stagnation_iters: Number of iterations to check for stagnation.
            stagnation_tol: Minimum likelihood improvement to avoid stagnation.

        Returns:
            Tuple of:
                - theta_final: Final parameter estimates
                - diagnostics: Dictionary with convergence information

        Diagnostics dictionary contains:
            - 'converged': bool - Whether algorithm converged
            - 'iterations': int - Number of iterations run
            - 'stopping_reason': str - Why algorithm stopped
            - 'theta_history': list - Parameter history (if track_history)
            - 'likelihood_history': list - Log-likelihood at each iteration
            - 'time_history': list - Cumulative time at each iteration
            - 'time_seconds': float - Total wall-clock time
            - 'peak_memory_mb': float - Peak memory usage
        """
        # Initialize monitoring
        monitor = ResourceMonitor(interval=0.1)
        monitor.start()
        start_time = time.perf_counter()

        # Initialize state
        theta = self._deep_copy_theta(theta_init)
        converged = False
        stopping_reason = "max_iter"

        # History tracking
        theta_history = [self._deep_copy_theta(theta)] if track_history else []
        likelihood_history = []
        time_history = []
        stagnation_count = 0

        # Compute initial likelihood
        try:
            current_ll = self.model.log_likelihood(X, theta)
        except Exception as e:
            monitor.stop()
            return theta, {
                'converged': False,
                'iterations': 0,
                'stopping_reason': f'initial_error: {str(e)}',
                'theta_history': theta_history,
                'likelihood_history': [],
                'time_history': [],
                'time_seconds': time.perf_counter() - start_time,
                'peak_memory_mb': 0
            }

        likelihood_history.append(current_ll)
        time_history.append(0.0)

        if self.verbose:
            print(f"Iteration 0: log-likelihood = {current_ll:.6f}")

        # Main EM loop
        for iteration in range(1, max_iter + 1):
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                stopping_reason = "timeout"
                break

            try:
                # E-step
                responsibilities = self.model.e_step(X, theta)

                # M-step
                theta_new = self.model.m_step(X, responsibilities)

                # Compute new likelihood
                new_ll = self.model.log_likelihood(X, theta_new)

            except Exception as e:
                stopping_reason = f"error: {str(e)}"
                break

            # Check convergence (parameter change)
            param_change = self._compute_param_change(theta, theta_new)
            if param_change < tol:
                converged = True
                stopping_reason = "tolerance"
                theta = theta_new
                likelihood_history.append(new_ll)
                time_history.append(time.perf_counter() - start_time)
                if track_history:
                    theta_history.append(self._deep_copy_theta(theta))
                break

            # Check stagnation
            ll_improvement = new_ll - current_ll
            if abs(ll_improvement) < stagnation_tol:
                stagnation_count += 1
                if stagnation_count >= stagnation_iters:
                    stopping_reason = "stagnation"
                    theta = theta_new
                    likelihood_history.append(new_ll)
                    time_history.append(time.perf_counter() - start_time)
                    if track_history:
                        theta_history.append(self._deep_copy_theta(theta))
                    break
            else:
                stagnation_count = 0

            # Update state
            theta = theta_new
            current_ll = new_ll

            # Track history
            likelihood_history.append(new_ll)
            time_history.append(time.perf_counter() - start_time)
            if track_history:
                theta_history.append(self._deep_copy_theta(theta))

            # Verbose output
            if self.verbose and iteration % self.verbose_interval == 0:
                print(f"Iteration {iteration}: log-likelihood = {new_ll:.6f}, "
                      f"change = {param_change:.2e}")

        # Stop monitoring
        resource_stats = monitor.stop()

        # Build diagnostics
        diagnostics = {
            'converged': converged,
            'iterations': len(likelihood_history),
            'stopping_reason': stopping_reason,
            'theta_history': theta_history,
            'likelihood_history': likelihood_history,
            'time_history': time_history,
            'time_seconds': resource_stats['elapsed_time'],
            'peak_memory_mb': resource_stats['peak_memory_mb'],
            'final_likelihood': likelihood_history[-1] if likelihood_history else None
        }

        if self.verbose:
            print(f"\nEM completed: {stopping_reason}")
            print(f"  Iterations: {diagnostics['iterations']}")
            print(f"  Final log-likelihood: {diagnostics['final_likelihood']:.6f}")
            print(f"  Time: {diagnostics['time_seconds']:.2f}s")

        return theta, diagnostics

    def _compute_param_change(
        self,
        theta_old: Dict[str, np.ndarray],
        theta_new: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute L-infinity norm of parameter change.

        Args:
            theta_old: Previous parameters.
            theta_new: New parameters.

        Returns:
            Maximum absolute change across all parameters.
        """
        max_change = 0.0

        for key in theta_old:
            if key in theta_new:
                old_val = np.asarray(theta_old[key])
                new_val = np.asarray(theta_new[key])

                if old_val.shape == new_val.shape:
                    change = np.max(np.abs(new_val - old_val))
                    max_change = max(max_change, change)

        return max_change

    def _deep_copy_theta(
        self,
        theta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Create deep copy of parameter dictionary.

        Args:
            theta: Parameter dictionary to copy.

        Returns:
            Deep copy with new numpy arrays.
        """
        return {k: np.array(v, copy=True) for k, v in theta.items()}


# =============================================================================
# Simple Model Interface for Testing
# =============================================================================

class SimpleGMMInterface:
    """
    Simple GMM interface for testing StandardEM.

    This provides a minimal implementation to test the EM algorithm
    without requiring the full GMM model.
    """

    def __init__(self, n_components: int, n_features: int):
        self.n_components = n_components
        self.n_features = n_features

    def e_step(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute responsibilities."""
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        # Compute log responsibilities
        log_resp = np.zeros((n, K))

        for k in range(K):
            diff = X - mu[k]
            # Diagonal covariance
            log_det = np.sum(np.log(sigma[k]))
            mahal = np.sum(diff ** 2 / sigma[k], axis=1)
            log_resp[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (
                self.n_features * np.log(2 * np.pi) + log_det + mahal
            )

        # Normalize (log-sum-exp trick)
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        log_resp_normalized = log_resp - log_resp_max
        resp = np.exp(log_resp_normalized)
        resp /= resp.sum(axis=1, keepdims=True)

        return resp

    def m_step(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Update parameters."""
        n = X.shape[0]
        K = self.n_components

        # Soft counts
        Nk = responsibilities.sum(axis=0) + 1e-10

        # Update mixing proportions
        pi = Nk / n

        # Update means
        mu = np.zeros((K, self.n_features))
        for k in range(K):
            mu[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]

        # Update variances (diagonal)
        sigma = np.zeros((K, self.n_features))
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = (responsibilities[:, k:k+1] * diff ** 2).sum(axis=0) / Nk[k]
            sigma[k] = np.maximum(sigma[k], 1e-6)  # Prevent collapse

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    def log_likelihood(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> float:
        """Compute log-likelihood."""
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        # Compute log probabilities for each component
        log_probs = np.zeros((n, K))

        for k in range(K):
            diff = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal = np.sum(diff ** 2 / sigma[k], axis=1)
            log_probs[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (
                self.n_features * np.log(2 * np.pi) + log_det + mahal
            )

        # Log-sum-exp for mixture
        max_log = np.max(log_probs, axis=1)
        log_sum = max_log + np.log(np.sum(np.exp(log_probs - max_log[:, None]), axis=1))

        return float(np.sum(log_sum))


# =============================================================================
# Unit Tests
# =============================================================================

def test_standard_em_basic():
    """Test basic EM functionality."""
    print("Testing StandardEM basic functionality...")

    np.random.seed(42)

    # Generate simple 2-component data
    n = 200
    X1 = np.random.randn(n // 2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n // 2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])

    # Initialize
    model = SimpleGMMInterface(n_components=2, n_features=2)
    theta_init = {
        'mu': np.array([[0.0, 0.0], [1.0, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    # Fit
    em = StandardEM(model, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=100, tol=1e-6)

    # Check results
    assert diag['converged'] or diag['stopping_reason'] in ['tolerance', 'stagnation'], \
        f"Should converge, got: {diag['stopping_reason']}"
    assert len(diag['likelihood_history']) > 1, "Should have multiple iterations"
    assert diag['likelihood_history'][-1] > diag['likelihood_history'][0], \
        "Likelihood should increase"

    print(f"  Iterations: {diag['iterations']}")
    print(f"  Final likelihood: {diag['final_likelihood']:.2f}")
    print("  PASSED")


def test_standard_em_monotonicity():
    """Test that likelihood is monotonically increasing."""
    print("Testing StandardEM monotonicity...")

    np.random.seed(123)

    # Generate data
    n = 300
    K = 3
    X = np.vstack([
        np.random.randn(n // 3, 2) + np.array([-3, 0]),
        np.random.randn(n // 3, 2) + np.array([0, 3]),
        np.random.randn(n // 3, 2) + np.array([3, 0])
    ])

    model = SimpleGMMInterface(n_components=K, n_features=2)
    theta_init = {
        'mu': np.random.randn(K, 2),
        'sigma': np.ones((K, 2)),
        'pi': np.ones(K) / K
    }

    em = StandardEM(model, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=50, tol=1e-8)

    # Check monotonicity
    ll_history = diag['likelihood_history']
    for i in range(1, len(ll_history)):
        assert ll_history[i] >= ll_history[i-1] - 1e-10, \
            f"Likelihood decreased at iteration {i}: {ll_history[i-1]:.6f} -> {ll_history[i]:.6f}"

    print(f"  All {len(ll_history)} iterations monotonic")
    print("  PASSED")


def test_standard_em_stopping_conditions():
    """Test different stopping conditions."""
    print("Testing StandardEM stopping conditions...")

    np.random.seed(42)

    n = 100
    X = np.random.randn(n, 2)

    model = SimpleGMMInterface(n_components=2, n_features=2)
    theta_init = {
        'mu': np.array([[0.0, 0.0], [0.1, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    # Test max_iter
    em = StandardEM(model, verbose=False)
    _, diag = em.fit(X, theta_init, max_iter=5, tol=1e-12)
    assert diag['iterations'] <= 6, f"Should stop at max_iter, got {diag['iterations']}"

    # Test timeout
    _, diag = em.fit(X, theta_init, max_iter=10000, timeout=0.01)
    assert diag['stopping_reason'] == 'timeout', \
        f"Should timeout, got {diag['stopping_reason']}"

    print("  PASSED")


def test_standard_em_diagnostics():
    """Test that diagnostics are complete."""
    print("Testing StandardEM diagnostics...")

    np.random.seed(42)

    n = 100
    X = np.random.randn(n, 2)

    model = SimpleGMMInterface(n_components=2, n_features=2)
    theta_init = {
        'mu': np.random.randn(2, 2),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    em = StandardEM(model, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=10)

    # Check all required keys
    required_keys = [
        'converged', 'iterations', 'stopping_reason', 'theta_history',
        'likelihood_history', 'time_history', 'time_seconds', 'peak_memory_mb'
    ]
    for key in required_keys:
        assert key in diag, f"Missing diagnostic key: {key}"

    # Check types
    assert isinstance(diag['converged'], bool), "converged should be bool"
    assert isinstance(diag['iterations'], int), "iterations should be int"
    assert isinstance(diag['likelihood_history'], list), "likelihood_history should be list"
    assert diag['time_seconds'] > 0, "time_seconds should be positive"
    assert diag['peak_memory_mb'] > 0, "peak_memory_mb should be positive"

    print("  PASSED")


def test_standard_em_verbose():
    """Test verbose mode."""
    print("Testing StandardEM verbose mode...")

    np.random.seed(42)

    n = 100
    X = np.random.randn(n, 2)

    model = SimpleGMMInterface(n_components=2, n_features=2)
    theta_init = {
        'mu': np.random.randn(2, 2),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    # Just check it runs without error
    em = StandardEM(model, verbose=True, verbose_interval=5)
    theta, diag = em.fit(X, theta_init, max_iter=15)

    assert diag['iterations'] > 0, "Should have some iterations"
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running standard_em.py unit tests")
    print("=" * 60)

    test_standard_em_basic()
    test_standard_em_monotonicity()
    test_standard_em_stopping_conditions()
    test_standard_em_diagnostics()
    test_standard_em_verbose()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
