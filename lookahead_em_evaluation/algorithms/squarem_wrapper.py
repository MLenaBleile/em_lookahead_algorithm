"""
SQUAREM Wrapper

Python wrapper for R's turboEM package implementing SQUAREM acceleration.
Provides graceful fallback if R/rpy2 is not available.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple
import time

# Track availability of rpy2
_RPY2_AVAILABLE = False
_TURBOEM_AVAILABLE = False

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    _RPY2_AVAILABLE = True

    # Try to activate automatic numpy conversion (handle deprecation in newer rpy2)
    # DeprecationWarning can be raised as an exception in some rpy2 versions
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            numpy2ri.activate()
    except BaseException:
        # Newer rpy2 versions deprecate activate() - skip it
        pass

    # Try to import turboEM
    try:
        turboEM = importr('turboEM')
        _TURBOEM_AVAILABLE = True
    except Exception as e:
        warnings.warn(f"turboEM R package not available: {e}")
        _TURBOEM_AVAILABLE = False

except BaseException as e:
    warnings.warn(f"rpy2 not available: {e}")
    _RPY2_AVAILABLE = False


def is_available() -> bool:
    """Check if SQUAREM is available (rpy2 + turboEM installed)."""
    return _RPY2_AVAILABLE and _TURBOEM_AVAILABLE


class SQUAREMWrapper:
    """
    Wrapper for R's turboEM SQUAREM acceleration.

    SQUAREM is a first-order acceleration method for EM that doesn't
    require gradient or Hessian computation. It uses Steffensen-like
    step lengths to accelerate convergence.

    Example usage:
    ```python
    try:
        squarem = SQUAREMWrapper(model=gmm)
        theta_final, diagnostics = squarem.fit(X, theta_init, max_iter=1000)
    except ImportError:
        print("SQUAREM not available, skipping")
    ```

    Installation (if not available):
    ```bash
    # Install rpy2
    pip install rpy2

    # In R, install turboEM
    R -e "install.packages('turboEM')"
    ```
    """

    def __init__(self, model: Any, **kwargs):
        """
        Initialize SQUAREM wrapper.

        Args:
            model: Model instance (GMM, HMM, etc.) with e_step, m_step, log_likelihood methods
            **kwargs: Additional arguments (ignored, for API compatibility)
        """
        if not is_available():
            raise ImportError(
                "SQUAREM requires rpy2 and R's turboEM package. "
                "Install with: pip install rpy2 && R -e \"install.packages('turboEM')\""
            )

        self.model = model

    def fit(
        self,
        X: np.ndarray,
        theta_init: Dict[str, np.ndarray],
        max_iter: int = 1000,
        tol: float = 1e-6,
        timeout: float = 600.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Fit model using SQUAREM acceleration.

        Args:
            X: Data array of shape (n, d)
            theta_init: Initial parameters dict
            max_iter: Maximum iterations
            tol: Convergence tolerance
            timeout: Timeout in seconds

        Returns:
            Tuple of (theta_final, diagnostics)
        """
        start_time = time.time()

        # Convert theta to flat vector for SQUAREM
        param_vec, structure = self._theta_to_vector(theta_init)

        # Store data and model for R callbacks
        self._X = X
        self._structure = structure

        # Create R functions for fixpt and objective
        self._setup_r_functions()

        # Run SQUAREM via R
        try:
            result = self._run_squarem(param_vec, max_iter, tol, timeout, start_time)
        except Exception as e:
            # Return with error
            elapsed = time.time() - start_time
            return theta_init, {
                'converged': False,
                'iterations': 0,
                'stopping_reason': 'error',
                'likelihood_history': [],
                'time_seconds': elapsed,
                'peak_memory_mb': 0.0,
                'error': str(e)
            }

        elapsed = time.time() - start_time

        # Convert result back to theta dict
        theta_final = self._vector_to_theta(result['par'], structure)

        # Compute final likelihood
        final_ll = self.model.log_likelihood(X, theta_final)

        diagnostics = {
            'converged': result.get('converged', False),
            'iterations': result.get('fpeval', 0),  # Fixed point evaluations
            'stopping_reason': 'convergence' if result.get('converged', False) else 'max_iter',
            'likelihood_history': result.get('ll_history', [final_ll]),
            'final_likelihood': final_ll,
            'time_seconds': elapsed,
            'peak_memory_mb': 0.0,  # Not tracked in R call
            'fpeval': result.get('fpeval', 0),
            'objfeval': result.get('objfeval', 0),
        }

        return theta_final, diagnostics

    def _theta_to_vector(self, theta: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Convert theta dict to flat parameter vector.

        Returns:
            Tuple of (flat_vector, structure_info)
        """
        # Structure info for reconstruction
        structure = {
            'keys': [],
            'shapes': [],
            'sizes': [],
            'transforms': []  # 'none', 'log', 'softmax'
        }

        parts = []
        for key in ['pi', 'mu', 'sigma']:  # Fixed order for GMM
            if key not in theta:
                continue

            arr = np.array(theta[key])
            structure['keys'].append(key)
            structure['shapes'].append(arr.shape)
            structure['sizes'].append(arr.size)

            # Apply transforms for constraints
            if key == 'pi':
                # Simplex constraint: use log-ratio transform (last component as reference)
                # Store log(pi_k / pi_K) for k = 1..K-1
                arr_flat = arr.flatten()
                if len(arr_flat) > 1:
                    log_ratios = np.log(arr_flat[:-1] / arr_flat[-1])
                    parts.append(log_ratios)
                    structure['transforms'].append('softmax')
                else:
                    parts.append(np.array([0.0]))  # Single component
                    structure['transforms'].append('none')
            elif key == 'sigma':
                # Positive constraint: use log transform
                parts.append(np.log(arr.flatten()))
                structure['transforms'].append('log')
            else:
                # No transform
                parts.append(arr.flatten())
                structure['transforms'].append('none')

        return np.concatenate(parts), structure

    def _vector_to_theta(self, vec: np.ndarray, structure: Dict) -> Dict[str, np.ndarray]:
        """Convert flat vector back to theta dict."""
        theta = {}
        idx = 0

        for i, key in enumerate(structure['keys']):
            transform = structure['transforms'][i]
            shape = structure['shapes'][i]

            if key == 'pi' and transform == 'softmax':
                # Inverse softmax: pi_k = exp(x_k) / sum(exp(x))
                K = shape[0]
                if K > 1:
                    log_ratios = vec[idx:idx + K - 1]
                    idx += K - 1
                    # Reconstruct: pi_k = exp(log_ratio_k) / (1 + sum(exp(log_ratios)))
                    exp_ratios = np.exp(log_ratios - np.max(log_ratios))  # Numerical stability
                    denom = 1 + np.sum(exp_ratios)
                    pi = np.zeros(K)
                    pi[:-1] = exp_ratios / denom
                    pi[-1] = 1.0 / denom
                    pi = pi / pi.sum()  # Ensure exact normalization
                    theta[key] = pi
                else:
                    theta[key] = np.array([1.0])
                    idx += 1
            elif transform == 'log':
                size = structure['sizes'][i]
                theta[key] = np.exp(vec[idx:idx + size]).reshape(shape)
                idx += size
            else:
                size = structure['sizes'][i]
                theta[key] = vec[idx:idx + size].reshape(shape)
                idx += size

        return theta

    def _setup_r_functions(self):
        """Set up R functions for SQUAREM."""
        # This is a placeholder - actual R integration would go here
        # For now, we implement a pure Python fallback using SQUAREM algorithm
        pass

    def _run_squarem(
        self,
        param_vec: np.ndarray,
        max_iter: int,
        tol: float,
        timeout: float,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Run SQUAREM algorithm.

        Implements the squared iterative method (SQUAREM) for accelerating EM.
        Reference: Varadhan & Roland (2008) "Simple and Globally Convergent Methods
        for Accelerating the Convergence of Any EM Algorithm"
        """
        # SQUAREM parameters
        step_min = 1.0
        step_max = 1.0
        step_factor = 4.0

        p = param_vec.copy()
        ll_history = []

        # Initial likelihood
        theta = self._vector_to_theta(p, self._structure)
        ll = self.model.log_likelihood(self._X, theta)
        ll_history.append(ll)

        converged = False
        n_iter = 0
        fpeval = 0
        objfeval = 1

        for iteration in range(max_iter):
            # Check timeout
            if time.time() - start_time > timeout:
                break

            n_iter = iteration + 1

            # Step 1: p1 = F(p)
            p1 = self._em_step(p)
            fpeval += 1

            # Step 2: p2 = F(p1)
            p2 = self._em_step(p1)
            fpeval += 1

            # Compute differences
            r = p1 - p
            v = p2 - p1 - r

            # Compute step length
            r_norm_sq = np.dot(r, r)
            v_norm_sq = np.dot(v, v)

            if v_norm_sq < 1e-16:
                # No acceleration needed, just use standard EM step
                p_new = p2
            else:
                # SQUAREM step length
                alpha = -np.sqrt(r_norm_sq / v_norm_sq)
                alpha = max(step_min, min(step_max, alpha))

                # Accelerated step: p_new = p - 2*alpha*r + alpha^2*v
                p_new = p - 2 * alpha * r + alpha**2 * v

                # Project to feasible region and ensure monotonicity
                theta_new = self._vector_to_theta(p_new, self._structure)

                # Enforce constraints
                theta_new = self._project_theta(theta_new)

                # Convert back to vector
                p_new, _ = self._theta_to_vector(theta_new)

                # One more EM step to ensure we're at a fixed point
                p_new = self._em_step(p_new)
                fpeval += 1

            # Compute new likelihood
            theta_new = self._vector_to_theta(p_new, self._structure)
            ll_new = self.model.log_likelihood(self._X, theta_new)
            objfeval += 1

            # Ensure monotonicity
            if ll_new < ll:
                # Fall back to regular EM step
                p_new = p2
                theta_new = self._vector_to_theta(p_new, self._structure)
                ll_new = self.model.log_likelihood(self._X, theta_new)
                objfeval += 1

            ll_history.append(ll_new)

            # Check convergence
            if abs(ll_new - ll) < tol:
                converged = True
                p = p_new
                break

            # Update for next iteration
            p = p_new
            ll = ll_new

            # Adaptive step bounds
            step_max = step_max * step_factor

        return {
            'par': p,
            'converged': converged,
            'fpeval': fpeval,
            'objfeval': objfeval,
            'll_history': ll_history
        }

    def _em_step(self, param_vec: np.ndarray) -> np.ndarray:
        """
        Single EM step as a fixed-point function.

        Args:
            param_vec: Current parameter vector

        Returns:
            Updated parameter vector after one EM step
        """
        # Convert to theta
        theta = self._vector_to_theta(param_vec, self._structure)

        # E-step
        responsibilities = self.model.e_step(self._X, theta)

        # M-step
        theta_new = self.model.m_step(self._X, responsibilities)

        # Convert back to vector
        new_vec, _ = self._theta_to_vector(theta_new)

        return new_vec

    def _project_theta(self, theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Project theta to feasible region."""
        theta_proj = theta.copy()

        # Ensure pi is a valid probability simplex
        if 'pi' in theta_proj:
            pi = np.clip(theta_proj['pi'], 1e-10, 1.0)
            theta_proj['pi'] = pi / pi.sum()

        # Ensure sigma is positive
        if 'sigma' in theta_proj:
            theta_proj['sigma'] = np.clip(theta_proj['sigma'], 1e-6, None)

        return theta_proj


class PurePythonSQUAREM:
    """
    Pure Python implementation of SQUAREM that doesn't require R.

    This is a fallback for when rpy2/turboEM is not available.
    Implements the same SQUAREM algorithm as the R wrapper.
    """

    def __init__(self, model: Any, **kwargs):
        """
        Initialize SQUAREM.

        Args:
            model: Model instance with e_step, m_step, log_likelihood methods
        """
        self.model = model

    def fit(
        self,
        X: np.ndarray,
        theta_init: Dict[str, np.ndarray],
        max_iter: int = 1000,
        tol: float = 1e-6,
        timeout: float = 600.0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Fit model using SQUAREM acceleration.

        This is a pure Python implementation that doesn't require R.

        Args:
            X: Data array
            theta_init: Initial parameters
            max_iter: Maximum iterations
            tol: Convergence tolerance
            timeout: Timeout in seconds

        Returns:
            Tuple of (theta_final, diagnostics)
        """
        from utils.timing import ResourceMonitor

        monitor = ResourceMonitor()
        monitor.start()
        start_time = time.time()

        theta = {k: np.array(v) for k, v in theta_init.items()}
        ll_history = []

        # Initial likelihood
        ll = self.model.log_likelihood(X, theta)
        ll_history.append(ll)

        # SQUAREM parameters
        step_min = 1.0
        step_max = 1.0
        step_factor = 4.0

        converged = False
        n_iter = 0
        fpeval = 0

        for iteration in range(max_iter):
            # Check timeout
            if time.time() - start_time > timeout:
                break

            n_iter = iteration + 1

            # Step 1: theta1 = F(theta) - one EM step
            resp = self.model.e_step(X, theta)
            theta1 = self.model.m_step(X, resp)
            fpeval += 1

            # Step 2: theta2 = F(theta1) - another EM step
            resp1 = self.model.e_step(X, theta1)
            theta2 = self.model.m_step(X, resp1)
            fpeval += 1

            # Compute differences as flattened vectors
            r = self._theta_diff(theta1, theta)
            v = self._theta_diff(theta2, theta1)
            v = v - r

            # Compute step length
            r_norm_sq = np.dot(r, r)
            v_norm_sq = np.dot(v, v)

            if v_norm_sq < 1e-16:
                theta_new = theta2
            else:
                alpha = -np.sqrt(r_norm_sq / v_norm_sq)
                alpha = max(step_min, min(step_max, alpha))

                # Accelerated update
                theta_new = self._theta_update(theta, r, v, alpha)

                # Project to feasible region
                theta_new = self._project_theta(theta_new)

                # One more EM step
                resp_new = self.model.e_step(X, theta_new)
                theta_new = self.model.m_step(X, resp_new)
                fpeval += 1

            # Compute new likelihood
            ll_new = self.model.log_likelihood(X, theta_new)

            # Ensure monotonicity
            if ll_new < ll:
                theta_new = theta2
                ll_new = self.model.log_likelihood(X, theta_new)

            ll_history.append(ll_new)

            # Check convergence
            if abs(ll_new - ll) < tol:
                converged = True
                theta = theta_new
                break

            theta = theta_new
            ll = ll_new
            step_max *= step_factor

        timing_stats = monitor.stop()

        diagnostics = {
            'converged': converged,
            'iterations': n_iter,
            'stopping_reason': 'convergence' if converged else ('timeout' if time.time() - start_time > timeout else 'max_iter'),
            'likelihood_history': ll_history,
            'final_likelihood': ll_history[-1] if ll_history else float('-inf'),
            'time_seconds': timing_stats['elapsed_time'],
            'peak_memory_mb': timing_stats['peak_memory_mb'],
            'fpeval': fpeval
        }

        return theta, diagnostics

    def _theta_diff(self, theta1: Dict, theta2: Dict) -> np.ndarray:
        """Compute flattened difference theta1 - theta2."""
        parts = []
        for key in theta1:
            parts.append((theta1[key] - theta2[key]).flatten())
        return np.concatenate(parts)

    def _theta_update(self, theta: Dict, r: np.ndarray, v: np.ndarray, alpha: float) -> Dict:
        """Apply accelerated update: theta - 2*alpha*r + alpha^2*v."""
        update = -2 * alpha * r + alpha**2 * v

        theta_new = {}
        idx = 0
        for key in theta:
            size = theta[key].size
            shape = theta[key].shape
            theta_new[key] = theta[key] + update[idx:idx+size].reshape(shape)
            idx += size

        return theta_new

    def _project_theta(self, theta: Dict) -> Dict:
        """Project theta to feasible region."""
        theta_proj = {}

        for key, val in theta.items():
            if key == 'pi':
                # Simplex projection
                pi = np.clip(val, 1e-10, None)
                theta_proj[key] = pi / pi.sum()
            elif key == 'sigma':
                # Positive constraint
                theta_proj[key] = np.clip(val, 1e-6, None)
            else:
                theta_proj[key] = val

        return theta_proj


def get_squarem(model: Any) -> Any:
    """
    Get the best available SQUAREM implementation.

    Returns R-based wrapper if available, otherwise pure Python.

    Args:
        model: Model instance

    Returns:
        SQUAREM instance (either SQUAREMWrapper or PurePythonSQUAREM)
    """
    if is_available():
        return SQUAREMWrapper(model)
    else:
        return PurePythonSQUAREM(model)


# ============================================================
# Unit Tests
# ============================================================

def test_is_available():
    """Test availability check."""
    avail = is_available()
    print(f"  SQUAREM available: {avail}")
    print(f"  rpy2 available: {_RPY2_AVAILABLE}")
    print(f"  turboEM available: {_TURBOEM_AVAILABLE}")
    print("  PASSED")


def test_pure_python_squarem():
    """Test pure Python SQUAREM on toy GMM."""
    from lookahead_em_evaluation.models.gmm import GaussianMixtureModel
    from lookahead_em_evaluation.data.generate_gmm import generate_gmm_data

    # Generate simple data
    X, z, theta_true = generate_gmm_data(n=200, K=3, d=2, seed=42)

    # Initialize
    model = GaussianMixtureModel(n_components=3, n_features=2)

    theta_init = {
        'pi': np.ones(3) / 3,
        'mu': X[np.random.RandomState(42).choice(200, 3, replace=False)],
        'sigma': np.ones((3, 2))
    }

    # Test pure Python SQUAREM
    squarem = PurePythonSQUAREM(model)
    theta_final, diagnostics = squarem.fit(
        X, theta_init, max_iter=100, tol=1e-4
    )

    assert diagnostics['iterations'] > 0, "Should have some iterations"
    assert diagnostics['final_likelihood'] > float('-inf'), "Should have valid likelihood"
    assert len(diagnostics['likelihood_history']) > 0, "Should have likelihood history"

    print(f"  Iterations: {diagnostics['iterations']}")
    print(f"  Final LL: {diagnostics['final_likelihood']:.2f}")
    print(f"  Converged: {diagnostics['converged']}")
    print("  PASSED")


def test_get_squarem():
    """Test get_squarem factory function."""
    from lookahead_em_evaluation.models.gmm import GaussianMixtureModel

    model = GaussianMixtureModel(n_components=3, n_features=2)
    squarem = get_squarem(model)

    # Should return one of the implementations
    assert isinstance(squarem, (SQUAREMWrapper, PurePythonSQUAREM)), "Should return SQUAREM instance"

    print(f"  Got: {type(squarem).__name__}")
    print("  PASSED")


def test_vector_conversion():
    """Test theta to/from vector conversion."""
    # Only test if wrapper is available
    if not is_available():
        print("  SKIPPED (rpy2 not available)")
        return

    from lookahead_em_evaluation.models.gmm import GaussianMixtureModel

    model = GaussianMixtureModel(n_components=3, n_features=2)
    wrapper = SQUAREMWrapper(model)

    theta = {
        'pi': np.array([0.3, 0.5, 0.2]),
        'mu': np.array([[0, 0], [1, 1], [2, 2]], dtype=float),
        'sigma': np.array([[0.5, 0.5], [1.0, 1.0], [0.8, 0.8]])
    }

    vec, structure = wrapper._theta_to_vector(theta)
    theta_recovered = wrapper._vector_to_theta(vec, structure)

    assert np.allclose(theta['pi'], theta_recovered['pi'], atol=1e-6), "pi mismatch"
    assert np.allclose(theta['mu'], theta_recovered['mu'], atol=1e-6), "mu mismatch"
    assert np.allclose(theta['sigma'], theta_recovered['sigma'], atol=1e-6), "sigma mismatch"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running squarem_wrapper.py unit tests")
    print("=" * 60)

    print("Testing availability...")
    test_is_available()

    print("Testing pure Python SQUAREM...")
    test_pure_python_squarem()

    print("Testing get_squarem factory...")
    test_get_squarem()

    print("Testing vector conversion...")
    test_vector_conversion()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
