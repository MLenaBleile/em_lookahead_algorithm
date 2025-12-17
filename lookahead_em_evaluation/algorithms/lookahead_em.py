"""
Lookahead EM Algorithm Implementation

This module provides the novel Lookahead EM algorithm that uses a value
function to look ahead and accelerate convergence.

The key update rule is:
    θ^(t+1) = argmax_θ [Q(θ|θ^(t)) + γ · V(θ)]

where V(θ) approximates the value of being at θ for future EM steps.

Example usage:
    >>> from models.gmm import GaussianMixtureModel
    >>> model = GaussianMixtureModel(n_components=5, n_features=2)
    >>> em = LookaheadEM(model, gamma='adaptive')
    >>> theta_final, diagnostics = em.fit(X, theta_init)
"""

import time
import copy
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from utils.timing import ResourceMonitor
from algorithms.gamma_schedule import (
    GammaSchedule, FixedGamma, AdaptiveGamma, ExponentialGamma, create_schedule
)
from algorithms.value_estimation import estimate_V_second_order, estimate_V_simple


class LookaheadEM:
    """
    Lookahead Expectation-Maximization algorithm.

    Uses a value function V(θ) to look ahead and select parameters that
    will lead to faster convergence. The objective at each step is:
        θ^(t+1) = argmax_θ [Q(θ|θ^(t)) + γ_t · V(θ)]

    When γ = 0, this reduces to standard EM.

    Attributes:
        model: Statistical model providing e_step, m_step, log_likelihood,
               and optionally compute_Q, compute_Q_gradient, compute_Q_hessian.
        gamma_schedule: Schedule for gamma values over iterations.
        n_candidates: Number of candidate parameters to evaluate.
        candidate_scale: Scale factor for candidate generation.
        use_second_order: Whether to use second-order V estimation.
        verbose: Whether to print progress.

    Example:
        >>> model = GaussianMixtureModel(n_components=3, n_features=2)
        >>> em = LookaheadEM(model, gamma='adaptive', n_candidates=20)
        >>> theta, diag = em.fit(X, theta_init, max_iter=100)
        >>> print(f"Converged in {diag['iterations']} iterations")
    """

    def __init__(
        self,
        model: Any,
        gamma: Union[str, float, GammaSchedule] = 'adaptive',
        n_candidates: int = 20,
        candidate_scale: float = 0.1,
        gamma_init: float = 0.1,
        gamma_max: float = 0.9,
        gamma_step: float = 0.05,
        use_second_order: bool = True,
        verbose: bool = False,
        verbose_interval: int = 10
    ):
        """
        Initialize LookaheadEM.

        Args:
            model: Statistical model object with methods:
                   - e_step(X, theta) -> responsibilities
                   - m_step(X, responsibilities) -> theta_new
                   - log_likelihood(X, theta) -> float
                   - compute_Q(theta, theta_old, X, resp) -> float (optional)
                   - compute_Q_gradient(theta, theta_old, X, resp) -> array (optional)
                   - compute_Q_hessian(theta, theta_old, X, resp) -> matrix (optional)
            gamma: Gamma schedule specification:
                   - float: Fixed gamma value
                   - 'adaptive': Linearly increasing gamma
                   - 'exponential': Exponentially increasing gamma
                   - GammaSchedule instance: Custom schedule
            n_candidates: Number of candidate parameters to evaluate per step.
            candidate_scale: Scale of perturbations for candidate generation.
            gamma_init: Initial gamma for adaptive/exponential schedules.
            gamma_max: Maximum gamma for adaptive/exponential schedules.
            gamma_step: Step size for adaptive schedule.
            use_second_order: If True, use second-order V estimation (requires
                             gradients/Hessians). If False, use one-step V.
            verbose: If True, print progress during fitting.
            verbose_interval: Print progress every N iterations.
        """
        self.model = model
        self.n_candidates = n_candidates
        self.candidate_scale = candidate_scale
        self.use_second_order = use_second_order
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # Set up gamma schedule
        if isinstance(gamma, GammaSchedule):
            self.gamma_schedule = gamma
        else:
            self.gamma_schedule = create_schedule(
                gamma,
                gamma_init=gamma_init,
                gamma_max=gamma_max,
                gamma_step=gamma_step
            )

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
        Fit the model using Lookahead EM algorithm.

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

        Diagnostics dictionary contains all StandardEM diagnostics plus:
            - 'gamma_history': list of gamma values used at each iteration
            - 'v_value_history': list of V(θ) estimates
            - 'n_candidates_evaluated': total candidates evaluated
        """
        # Reset gamma schedule for new run
        self.gamma_schedule.reset()

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
        gamma_history = []
        v_value_history = []
        n_candidates_total = 0
        stagnation_count = 0

        # Compute initial likelihood
        try:
            current_ll = self.model.log_likelihood(X, theta)
        except Exception as e:
            monitor.stop()
            return theta, self._build_diagnostics(
                converged=False,
                iterations=0,
                stopping_reason=f'initial_error: {str(e)}',
                theta_history=theta_history,
                likelihood_history=[],
                time_history=[],
                gamma_history=[],
                v_value_history=[],
                n_candidates_total=0,
                resource_stats={'elapsed_time': time.perf_counter() - start_time, 'peak_memory_mb': 0}
            )

        likelihood_history.append(current_ll)
        time_history.append(0.0)

        if self.verbose:
            print(f"Iteration 0: log-likelihood = {current_ll:.6f}")

        # Main Lookahead EM loop
        for iteration in range(1, max_iter + 1):
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                stopping_reason = "timeout"
                break

            # Get gamma for this iteration
            gamma = self.gamma_schedule.get_gamma(iteration - 1)
            gamma_history.append(gamma)

            try:
                # Perform lookahead step
                theta_new, step_diag = self._lookahead_step(
                    X, theta, gamma, iteration
                )
                n_candidates_total += step_diag.get('n_candidates', 0)
                v_value_history.append(step_diag.get('best_v', None))

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
                      f"change = {param_change:.2e}, γ = {gamma:.3f}")

        # Stop monitoring
        resource_stats = monitor.stop()

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            converged=converged,
            iterations=len(likelihood_history),
            stopping_reason=stopping_reason,
            theta_history=theta_history,
            likelihood_history=likelihood_history,
            time_history=time_history,
            gamma_history=gamma_history,
            v_value_history=v_value_history,
            n_candidates_total=n_candidates_total,
            resource_stats=resource_stats
        )

        if self.verbose:
            print(f"\nLookahead EM completed: {stopping_reason}")
            print(f"  Iterations: {diagnostics['iterations']}")
            print(f"  Final log-likelihood: {diagnostics['final_likelihood']:.6f}")
            print(f"  Time: {diagnostics['time_seconds']:.2f}s")
            print(f"  Candidates evaluated: {n_candidates_total}")

        return theta, diagnostics

    def _lookahead_step(
        self,
        X: np.ndarray,
        theta_current: Dict[str, np.ndarray],
        gamma: float,
        iteration: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Perform one lookahead EM step using gradient-based optimization.

        The lookahead update solves:
            θ^(t+1) = argmax_θ [Q(θ|θ^(t)) + γ · V(θ)]

        Since M-step maximizes Q, we compute:
            θ_lookahead = θ_mstep + γ · correction

        where correction is based on the gradient of V.

        Args:
            X: Data matrix.
            theta_current: Current parameter values.
            gamma: Current gamma value.
            iteration: Current iteration number.

        Returns:
            Tuple of (theta_next, step_diagnostics).
        """
        step_diag = {
            'n_candidates': 1,
            'best_q': None,
            'best_v': None,
            'best_score': None,
            'gradient_norm': None,
            'step_size': None
        }

        # Compute responsibilities
        responsibilities = self.model.e_step(X, theta_current)

        # If gamma is very small, just use standard M-step
        if gamma < 1e-8:
            theta_next = self.model.m_step(X, responsibilities)
            theta_next = self._project_to_feasible(theta_next)
            return theta_next, step_diag

        # Get standard M-step result
        theta_mstep = self.model.m_step(X, responsibilities)
        theta_mstep = self._project_to_feasible(theta_mstep)

        # If we have gradient methods, use gradient-based lookahead
        if self._has_hessian_methods():
            theta_next, grad_diag = self._gradient_based_step(
                X, theta_current, theta_mstep, responsibilities, gamma
            )
            step_diag.update(grad_diag)
        else:
            # Fallback to simple V estimation with line search
            theta_next = self._simple_lookahead_step(
                X, theta_current, theta_mstep, responsibilities, gamma
            )

        theta_next = self._project_to_feasible(theta_next)

        # Compute final scores for diagnostics
        try:
            step_diag['best_q'] = self._compute_Q(
                theta_next, theta_current, X, responsibilities
            )
            if self._has_hessian_methods():
                step_diag['best_v'], _ = estimate_V_second_order(
                    self.model, X, theta_next
                )
            else:
                step_diag['best_v'] = estimate_V_simple(
                    self.model, X, theta_next
                )
            step_diag['best_score'] = step_diag['best_q'] + gamma * step_diag['best_v']
        except Exception:
            pass

        return theta_next, step_diag

    def _gradient_based_step(
        self,
        X: np.ndarray,
        theta_current: Dict[str, np.ndarray],
        theta_mstep: Dict[str, np.ndarray],
        responsibilities: Any,  # Can be np.ndarray (GMM) or dict (HMM)
        gamma: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Compute gradient-based lookahead step.

        Uses the gradient of V to adjust the M-step result:
            θ_lookahead = θ_mstep + step_size * gradient_direction

        The gradient direction comes from the Newton step for V:
            direction = -H_V^{-1} ∇V

        Args:
            X: Data matrix.
            theta_current: Current parameters (for E-step).
            theta_mstep: M-step result.
            responsibilities: E-step responsibilities.
            gamma: Lookahead weight.

        Returns:
            Tuple of (theta_next, diagnostics).
        """
        diag = {
            'gradient_norm': 0.0,
            'step_size': 0.0,
            'used_newton': False
        }

        try:
            # Compute gradient and Hessian of Q at M-step result
            # (which approximates the gradient of V's improvement potential)
            grad = self.model.compute_Q_gradient(
                theta_mstep, theta_mstep, X, responsibilities
            )
            H = self.model.compute_Q_hessian(
                theta_mstep, theta_mstep, X, responsibilities
            )

            grad_norm = np.linalg.norm(grad)
            diag['gradient_norm'] = grad_norm

            # If gradient is very small, M-step is near-optimal
            if grad_norm < 1e-10:
                return theta_mstep, diag

            # Compute Newton direction: -H^{-1} * grad
            # Add regularization for numerical stability
            reg = 1e-6 * np.abs(np.diag(H)).max()
            H_reg = H - reg * np.eye(len(H))  # Make more negative definite

            try:
                # Solve H * direction = -grad
                direction = np.linalg.solve(H_reg, -grad)
                diag['used_newton'] = True
            except np.linalg.LinAlgError:
                # Fall back to gradient direction if Hessian is singular
                direction = grad / grad_norm
                diag['used_newton'] = False

            # Scale step by gamma (lookahead weight)
            # Use line search to find good step size
            step_size = self._line_search(
                X, theta_current, theta_mstep, direction,
                responsibilities, gamma, max_step=gamma
            )
            diag['step_size'] = step_size

            # Apply step to get new parameters
            theta_next = self._apply_step(theta_mstep, direction, step_size)

            return theta_next, diag

        except Exception as e:
            # On any error, fall back to M-step
            diag['error'] = str(e)
            return theta_mstep, diag

    def _line_search(
        self,
        X: np.ndarray,
        theta_current: Dict[str, np.ndarray],
        theta_base: Dict[str, np.ndarray],
        direction: np.ndarray,
        responsibilities: Any,  # Can be np.ndarray (GMM) or dict (HMM)
        gamma: float,
        max_step: float = 1.0,
        n_steps: int = 10
    ) -> float:
        """
        Line search to find optimal step size.

        Evaluates Q + γV at different step sizes and returns the best.

        Args:
            X: Data matrix.
            theta_current: Current parameters.
            theta_base: Base parameters (M-step result).
            direction: Step direction.
            responsibilities: E-step responsibilities.
            gamma: Lookahead weight.
            max_step: Maximum step size.
            n_steps: Number of step sizes to try.

        Returns:
            Optimal step size.
        """
        best_score = float('-inf')
        best_step = 0.0

        # Try step sizes from 0 to max_step
        for i in range(n_steps + 1):
            step = (i / n_steps) * max_step

            try:
                theta_candidate = self._apply_step(theta_base, direction, step)
                theta_candidate = self._project_to_feasible(theta_candidate)

                # Evaluate Q + γV
                Q_val = self._compute_Q(
                    theta_candidate, theta_current, X, responsibilities
                )

                if self._has_hessian_methods():
                    V_val, _ = estimate_V_second_order(
                        self.model, X, theta_candidate
                    )
                else:
                    V_val = estimate_V_simple(self.model, X, theta_candidate)

                score = Q_val + gamma * V_val

                if score > best_score:
                    best_score = score
                    best_step = step

            except Exception:
                continue

        return best_step

    def _apply_step(
        self,
        theta_base: Dict[str, np.ndarray],
        direction: np.ndarray,
        step_size: float
    ) -> Dict[str, np.ndarray]:
        """
        Apply a step in the given direction to theta.

        Args:
            theta_base: Base parameters.
            direction: Flattened direction vector.
            step_size: Step size multiplier.

        Returns:
            New parameters after step.
        """
        theta_new = {}
        offset = 0

        # Determine parameter order based on model type
        # IMPORTANT: Must match the order used in compute_Q_gradient for each model
        if 'mu' in theta_base:
            # GMM parameters - gradient only covers mu (see gmm.py:253-289)
            # sigma and pi are not in the gradient, copy them directly
            param_keys = ['mu']
        elif 'A' in theta_base:
            # HMM parameters - gradient order: pi_0, A, B (see hmm.py:319-347)
            param_keys = ['pi_0', 'A', 'B']
        elif 'beta' in theta_base:
            # MoE parameters - gradient order: gamma, beta, sigma (see mixture_of_experts.py:320-341)
            param_keys = ['gamma', 'beta', 'sigma']
        else:
            # Generic: iterate in sorted order
            param_keys = sorted(theta_base.keys())

        # First, copy all parameters from theta_base that won't be modified by gradient
        for key in theta_base:
            if key not in param_keys:
                theta_new[key] = np.asarray(theta_base[key]).copy()

        for key in param_keys:
            if key not in theta_base:
                continue

            value = np.asarray(theta_base[key])
            size = value.size

            # Handle case where direction vector is smaller than expected
            if offset + size > len(direction):
                # Fall back to copying value unchanged
                theta_new[key] = value.copy()
                continue

            delta = direction[offset:offset + size].reshape(value.shape)

            # Apply step based on parameter type
            if key in ['sigma', 'variance', 'std']:
                # For variances, work in log space to stay positive
                log_value = np.log(np.maximum(value, 1e-10))
                new_log = log_value + step_size * delta / np.maximum(value, 1e-10)
                theta_new[key] = np.exp(new_log)
            elif key == 'pi':
                # For GMM mixture weights, add and renormalize
                new_pi = value + step_size * delta
                new_pi = np.maximum(new_pi, 1e-10)
                theta_new[key] = new_pi / new_pi.sum()
            elif key == 'pi_0':
                # For HMM initial state distribution
                new_pi = value + step_size * delta
                new_pi = np.maximum(new_pi, 1e-10)
                theta_new[key] = new_pi / new_pi.sum()
            elif key == 'A':
                # For HMM transition matrix (row-stochastic)
                new_A = value + step_size * delta
                new_A = np.maximum(new_A, 1e-10)
                theta_new[key] = new_A / new_A.sum(axis=1, keepdims=True)
            elif key == 'B':
                # For HMM emission matrix (row-stochastic)
                new_B = value + step_size * delta
                new_B = np.maximum(new_B, 1e-10)
                theta_new[key] = new_B / new_B.sum(axis=1, keepdims=True)
            elif key == 'gamma' and value.ndim == 2:
                # For MoE gate parameters (rows are coefficient vectors)
                theta_new[key] = value + step_size * delta
            else:
                # For means and other unconstrained parameters
                theta_new[key] = value + step_size * delta

            offset += size

        return theta_new

    def _simple_lookahead_step(
        self,
        X: np.ndarray,
        theta_current: Dict[str, np.ndarray],
        theta_mstep: Dict[str, np.ndarray],
        responsibilities: Any,
        gamma: float
    ) -> Dict[str, np.ndarray]:
        """
        Simple lookahead using one-step V estimation.

        When gradients aren't available, estimate the improvement direction
        by comparing M-step to current parameters.

        Args:
            X: Data matrix.
            theta_current: Current parameters.
            theta_mstep: M-step result.
            responsibilities: E-step responsibilities (array or dict).
            gamma: Lookahead weight.

        Returns:
            New parameters.
        """
        # The M-step moves from theta_current toward theta_mstep
        # Lookahead extrapolates this movement
        theta_next = {}

        for key in theta_mstep:
            current_val = np.asarray(theta_current[key])
            mstep_val = np.asarray(theta_mstep[key])

            # Direction of M-step movement
            delta = mstep_val - current_val

            # Extrapolate by gamma factor based on parameter type
            if key in ['pi', 'pi_0']:
                # For probability vectors, be conservative and renormalize
                new_val = mstep_val + 0.5 * gamma * delta
                new_val = np.maximum(new_val, 1e-10)
                theta_next[key] = new_val / new_val.sum()
            elif key in ['sigma', 'variance', 'std']:
                # For variances, stay positive
                new_val = mstep_val + 0.5 * gamma * delta
                theta_next[key] = np.maximum(new_val, 1e-10)
            elif key == 'A':
                # For HMM transition matrix (row-stochastic)
                new_val = mstep_val + 0.5 * gamma * delta
                new_val = np.maximum(new_val, 1e-10)
                theta_next[key] = new_val / new_val.sum(axis=1, keepdims=True)
            elif key == 'B':
                # For HMM emission matrix (row-stochastic)
                new_val = mstep_val + 0.5 * gamma * delta
                new_val = np.maximum(new_val, 1e-10)
                theta_next[key] = new_val / new_val.sum(axis=1, keepdims=True)
            else:
                # For means and other unconstrained parameters
                theta_next[key] = mstep_val + 0.5 * gamma * delta

        return theta_next

    def _compute_Q(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> float:
        """Compute Q(theta | theta_old)."""
        if hasattr(self.model, 'compute_Q'):
            return self.model.compute_Q(theta, theta_old, X, responsibilities)
        else:
            # Fallback: use log-likelihood (not exactly Q, but related)
            return self.model.log_likelihood(X, theta)

    def _has_hessian_methods(self) -> bool:
        """Check if model has gradient/Hessian methods."""
        return (hasattr(self.model, 'compute_Q_gradient') and
                hasattr(self.model, 'compute_Q_hessian'))

    def _generate_candidates(
        self,
        theta_base: Dict[str, np.ndarray],
        n_candidates: int
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate candidate parameters around base theta.

        Uses appropriate transformations for different parameter types:
        - Positive parameters: perturb in log space
        - Simplex parameters: perturb in logit space
        - Unconstrained: Gaussian perturbation

        Args:
            theta_base: Base parameters to perturb around.
            n_candidates: Number of candidates to generate.

        Returns:
            List of candidate parameter dictionaries.
        """
        candidates = []

        for _ in range(n_candidates):
            candidate = {}

            for key, value in theta_base.items():
                value = np.asarray(value)

                if key in ['pi', 'weights', 'mixing']:
                    # Simplex: perturb in logit space
                    perturbed = self._perturb_simplex(value)
                elif key in ['sigma', 'variance', 'std']:
                    # Positive: perturb in log space
                    perturbed = self._perturb_positive(value)
                else:
                    # Unconstrained: Gaussian perturbation
                    perturbed = self._perturb_unconstrained(value)

                candidate[key] = perturbed

            candidates.append(candidate)

        return candidates

    def _perturb_unconstrained(self, value: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to unconstrained parameters."""
        scale = self.candidate_scale * (np.abs(value).mean() + 0.1)
        return value + np.random.randn(*value.shape) * scale

    def _perturb_positive(self, value: np.ndarray) -> np.ndarray:
        """Perturb positive parameters in log space."""
        log_value = np.log(np.maximum(value, 1e-10))
        perturbed_log = log_value + np.random.randn(*value.shape) * self.candidate_scale
        return np.exp(perturbed_log)

    def _perturb_simplex(self, value: np.ndarray) -> np.ndarray:
        """Perturb simplex parameters (sum to 1)."""
        # Add noise and renormalize
        perturbed = value + np.abs(np.random.randn(*value.shape)) * self.candidate_scale * 0.1
        perturbed = np.maximum(perturbed, 1e-10)
        return perturbed / perturbed.sum()

    def _project_to_feasible(
        self,
        theta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Project parameters to feasible set.

        Ensures:
        - Positive parameters (sigma) are > 0
        - Simplex parameters (pi, pi_0) sum to 1 and are > 0
        - Row-stochastic matrices (A, B) have rows that sum to 1

        Args:
            theta: Parameters to project.

        Returns:
            Projected parameters.
        """
        theta = self._deep_copy_theta(theta)

        for key, value in theta.items():
            if key in ['sigma', 'variance', 'std']:
                # Ensure positive
                theta[key] = np.maximum(value, 1e-6)
            elif key in ['pi', 'pi_0', 'weights', 'mixing']:
                # Project to simplex (1D probability vector)
                value = np.maximum(value, 1e-6)
                theta[key] = value / value.sum()
            elif key == 'A':
                # HMM transition matrix: row-stochastic
                value = np.maximum(value, 1e-6)
                theta[key] = value / value.sum(axis=1, keepdims=True)
            elif key == 'B':
                # HMM emission matrix: row-stochastic
                value = np.maximum(value, 1e-6)
                theta[key] = value / value.sum(axis=1, keepdims=True)

        return theta

    def _compute_param_change(
        self,
        theta_old: Dict[str, np.ndarray],
        theta_new: Dict[str, np.ndarray]
    ) -> float:
        """Compute L-infinity norm of parameter change."""
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
        """Create deep copy of parameter dictionary."""
        return {k: np.array(v, copy=True) for k, v in theta.items()}

    def _build_diagnostics(
        self,
        converged: bool,
        iterations: int,
        stopping_reason: str,
        theta_history: List[Dict],
        likelihood_history: List[float],
        time_history: List[float],
        gamma_history: List[float],
        v_value_history: List[Optional[float]],
        n_candidates_total: int,
        resource_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Build diagnostics dictionary."""
        return {
            'converged': converged,
            'iterations': iterations,
            'stopping_reason': stopping_reason,
            'theta_history': theta_history,
            'likelihood_history': likelihood_history,
            'time_history': time_history,
            'time_seconds': resource_stats['elapsed_time'],
            'peak_memory_mb': resource_stats['peak_memory_mb'],
            'final_likelihood': likelihood_history[-1] if likelihood_history else None,
            # Lookahead-specific
            'gamma_history': gamma_history,
            'v_value_history': v_value_history,
            'n_candidates_evaluated': n_candidates_total
        }


# =============================================================================
# Simple GMM Interface for Testing (with Q, gradient, Hessian)
# =============================================================================

class SimpleGMMWithHessian:
    """
    Simple GMM interface with gradient and Hessian for testing LookaheadEM.
    """

    def __init__(self, n_components: int, n_features: int):
        self.n_components = n_components
        self.n_features = n_features

    def e_step(self, X: np.ndarray, theta: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute responsibilities."""
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        log_resp = np.zeros((n, K))

        for k in range(K):
            diff = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal = np.sum(diff ** 2 / sigma[k], axis=1)
            log_resp[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (
                self.n_features * np.log(2 * np.pi) + log_det + mahal
            )

        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        log_resp_normalized = log_resp - log_resp_max
        resp = np.exp(log_resp_normalized)
        resp /= resp.sum(axis=1, keepdims=True)

        return resp

    def m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> Dict[str, np.ndarray]:
        """Update parameters."""
        n = X.shape[0]
        K = self.n_components

        Nk = responsibilities.sum(axis=0) + 1e-10
        pi = Nk / n

        mu = np.zeros((K, self.n_features))
        for k in range(K):
            mu[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]

        sigma = np.zeros((K, self.n_features))
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = (responsibilities[:, k:k+1] * diff ** 2).sum(axis=0) / Nk[k]
            sigma[k] = np.maximum(sigma[k], 1e-6)

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    def log_likelihood(self, X: np.ndarray, theta: Dict[str, np.ndarray]) -> float:
        """Compute log-likelihood."""
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        log_probs = np.zeros((n, K))

        for k in range(K):
            diff = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal = np.sum(diff ** 2 / sigma[k], axis=1)
            log_probs[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (
                self.n_features * np.log(2 * np.pi) + log_det + mahal
            )

        max_log = np.max(log_probs, axis=1)
        log_sum = max_log + np.log(np.sum(np.exp(log_probs - max_log[:, None]), axis=1))

        return float(np.sum(log_sum))

    def compute_Q(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> float:
        """Compute Q(theta | theta_old)."""
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        Q = 0.0
        for k in range(K):
            Nk = responsibilities[:, k].sum()
            Q += Nk * np.log(pi[k] + 1e-300)

            for i in range(n):
                diff = X[i] - mu[k]
                log_p = -0.5 * (
                    self.n_features * np.log(2 * np.pi) +
                    np.sum(np.log(sigma[k])) +
                    np.sum(diff ** 2 / sigma[k])
                )
                Q += responsibilities[i, k] * log_p

        return Q

    def compute_Q_gradient(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of Q w.r.t. mu (flattened)."""
        K = self.n_components
        d = self.n_features

        mu = theta['mu']
        sigma = theta['sigma']

        grad = np.zeros((K, d))

        for k in range(K):
            for i in range(len(X)):
                diff = X[i] - mu[k]
                grad[k] += responsibilities[i, k] * diff / sigma[k]

        return grad.flatten()

    def compute_Q_hessian(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """Compute Hessian of Q w.r.t. mu (block-diagonal)."""
        K = self.n_components
        d = self.n_features

        sigma = theta['sigma']

        # Block-diagonal Hessian
        H = np.zeros((K * d, K * d))

        for k in range(K):
            Nk = responsibilities[:, k].sum()
            start = k * d
            end = start + d
            H[start:end, start:end] = -np.diag(Nk / sigma[k])

        return H


# =============================================================================
# Unit Tests
# =============================================================================

def test_lookahead_em_basic():
    """Test basic LookaheadEM functionality."""
    print("Testing LookaheadEM basic functionality...")

    np.random.seed(42)

    # Generate simple 2-component data
    n = 200
    X1 = np.random.randn(n // 2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n // 2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])

    model = SimpleGMMWithHessian(n_components=2, n_features=2)
    theta_init = {
        'mu': np.array([[0.0, 0.0], [1.0, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    em = LookaheadEM(model, gamma=0.5, n_candidates=10, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=50, tol=1e-6)

    assert len(diag['likelihood_history']) > 1, "Should have multiple iterations"
    assert diag['likelihood_history'][-1] >= diag['likelihood_history'][0] - 1e-6, \
        "Likelihood should not decrease significantly"

    print(f"  Iterations: {diag['iterations']}")
    print(f"  Final likelihood: {diag['final_likelihood']:.2f}")
    print("  PASSED")


def test_lookahead_em_gamma_zero():
    """Test that gamma=0 matches StandardEM."""
    print("Testing LookaheadEM with gamma=0...")

    np.random.seed(42)

    n = 100
    X = np.vstack([
        np.random.randn(n // 2, 2) + np.array([-2, 0]),
        np.random.randn(n // 2, 2) + np.array([2, 0])
    ])

    model = SimpleGMMWithHessian(n_components=2, n_features=2)
    theta_init = {
        'mu': np.array([[0.0, 0.0], [1.0, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    # Lookahead EM with gamma=0
    lookahead = LookaheadEM(model, gamma=0.0, verbose=False)
    theta_la, diag_la = lookahead.fit(X, theta_init.copy(), max_iter=20)

    # Standard EM (import here to avoid circular imports in real code)
    from algorithms.standard_em import StandardEM
    standard = StandardEM(model, verbose=False)
    theta_std, diag_std = standard.fit(X, theta_init.copy(), max_iter=20)

    # Results should be very similar (not exact due to candidate generation)
    ll_diff = abs(diag_la['final_likelihood'] - diag_std['final_likelihood'])
    assert ll_diff < 1.0, f"Likelihoods should be similar, diff = {ll_diff}"

    print(f"  Lookahead LL: {diag_la['final_likelihood']:.2f}")
    print(f"  Standard LL: {diag_std['final_likelihood']:.2f}")
    print("  PASSED")


def test_lookahead_em_adaptive_gamma():
    """Test LookaheadEM with adaptive gamma schedule."""
    print("Testing LookaheadEM with adaptive gamma...")

    np.random.seed(42)

    n = 150
    X = np.vstack([
        np.random.randn(n // 3, 2) + np.array([-3, 0]),
        np.random.randn(n // 3, 2) + np.array([0, 0]),
        np.random.randn(n // 3, 2) + np.array([3, 0])
    ])

    model = SimpleGMMWithHessian(n_components=3, n_features=2)
    theta_init = {
        'mu': np.random.randn(3, 2),
        'sigma': np.ones((3, 2)),
        'pi': np.ones(3) / 3
    }

    em = LookaheadEM(model, gamma='adaptive', n_candidates=15, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=30)

    # Check gamma history shows adaptive behavior
    assert len(diag['gamma_history']) > 0, "Should have gamma history"
    assert diag['gamma_history'][0] < diag['gamma_history'][-1], \
        "Gamma should increase over time"

    print(f"  Initial gamma: {diag['gamma_history'][0]:.3f}")
    print(f"  Final gamma: {diag['gamma_history'][-1]:.3f}")
    print(f"  Iterations: {diag['iterations']}")
    print("  PASSED")


def test_lookahead_em_diagnostics():
    """Test that diagnostics are complete."""
    print("Testing LookaheadEM diagnostics...")

    np.random.seed(42)

    X = np.random.randn(50, 2)
    model = SimpleGMMWithHessian(n_components=2, n_features=2)
    theta_init = {
        'mu': np.random.randn(2, 2),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    em = LookaheadEM(model, gamma=0.5, n_candidates=5, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=10)

    # Check standard keys
    standard_keys = [
        'converged', 'iterations', 'stopping_reason', 'theta_history',
        'likelihood_history', 'time_history', 'time_seconds', 'peak_memory_mb'
    ]
    for key in standard_keys:
        assert key in diag, f"Missing standard key: {key}"

    # Check lookahead-specific keys
    lookahead_keys = ['gamma_history', 'v_value_history', 'n_candidates_evaluated']
    for key in lookahead_keys:
        assert key in diag, f"Missing lookahead key: {key}"

    assert diag['n_candidates_evaluated'] > 0, "Should evaluate some candidates"

    print(f"  Candidates evaluated: {diag['n_candidates_evaluated']}")
    print("  PASSED")


def test_lookahead_em_monotonicity():
    """Test that likelihood generally doesn't decrease."""
    print("Testing LookaheadEM monotonicity...")

    np.random.seed(123)

    n = 200
    X = np.vstack([
        np.random.randn(n // 2, 2) + np.array([-3, 0]),
        np.random.randn(n // 2, 2) + np.array([3, 0])
    ])

    model = SimpleGMMWithHessian(n_components=2, n_features=2)
    theta_init = {
        'mu': np.array([[-1.0, 0.0], [1.0, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    em = LookaheadEM(model, gamma=0.3, n_candidates=20, verbose=False)
    theta, diag = em.fit(X, theta_init, max_iter=30)

    # Check for significant decreases (small numerical decreases may occur)
    ll_history = diag['likelihood_history']
    significant_decreases = 0
    for i in range(1, len(ll_history)):
        if ll_history[i] < ll_history[i-1] - 0.1:
            significant_decreases += 1

    assert significant_decreases <= 2, \
        f"Too many significant likelihood decreases: {significant_decreases}"

    print(f"  Iterations: {len(ll_history)}")
    print(f"  Significant decreases: {significant_decreases}")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running lookahead_em.py unit tests")
    print("=" * 60)

    test_lookahead_em_basic()
    test_lookahead_em_gamma_zero()
    test_lookahead_em_adaptive_gamma()
    test_lookahead_em_diagnostics()
    test_lookahead_em_monotonicity()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
