"""
Value Function Estimation for Lookahead EM

This module provides functions for estimating the lookahead value V(θ),
which represents the expected future improvement from a parameter setting.

The key approximation is:
    V(θ) ≈ Q(θ|θ) + 0.5 × ∇Q^T H_Q^(-1) ∇Q

where:
- Q(θ|θ) is the complete-data log-likelihood at θ
- ∇Q is the gradient of Q w.r.t. θ
- H_Q is the Hessian of Q w.r.t. θ (negative definite)

Methods available:
- Second-order: Exact Hessian computation (expensive for large p)
- L-BFGS: Approximate H^(-1) using gradient history (O(mp) instead of O(p²))

Example usage:
    >>> V, diagnostics = estimate_V_second_order(model, X, theta)
    >>> print(f"V(θ) = {V:.4f}, Q(θ|θ) = {diagnostics['Q_self']:.4f}")

    >>> # Using L-BFGS approximation (faster for high-dimensional problems)
    >>> lbfgs = LBFGSApproximation(memory_size=10)
    >>> direction = lbfgs.compute_direction(gradient)
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import warnings
from collections import deque


def estimate_V_second_order(
    model: Any,
    X: np.ndarray,
    theta: Dict[str, np.ndarray],
    responsibilities: Optional[np.ndarray] = None,
    regularization: float = 1e-6,
    gradient_tol: float = 1e-10
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate lookahead value V(θ) using second-order approximation.

    Uses the Taylor expansion to approximate the maximum of Q(·|θ):
        V(θ) ≈ Q(θ|θ) + 0.5 × ∇Q^T H_Q^(-1) ∇Q

    This represents an approximation to the value of being at θ,
    accounting for the improvement available from the next EM step.

    Args:
        model: Statistical model with methods:
               - compute_Q(theta, theta_old, X, responsibilities)
               - compute_Q_gradient(theta, theta_old, X, responsibilities)
               - compute_Q_hessian(theta, theta_old, X, responsibilities)
        X: Data matrix of shape (n_samples, n_features).
        theta: Current parameter dictionary.
        responsibilities: Pre-computed responsibilities (optional).
                         If None, will compute via model.e_step(X, theta).
        regularization: Regularization parameter for Hessian (λ in H + λI).
        gradient_tol: If gradient norm < gradient_tol, skip Newton step.

    Returns:
        Tuple of:
            - V: Estimated value V(θ)
            - diagnostics: Dictionary with:
                - 'Q_self': Q(θ|θ)
                - 'grad_norm': ||∇Q||
                - 'newton_step_norm': ||H^(-1)∇Q|| (or 0 if skipped)
                - 'hessian_min_eigenvalue': min(eig(H))
                - 'hessian_condition_number': cond(H)
                - 'regularization_used': float (actual λ used)
                - 'newton_step_computed': bool

    Example:
        >>> V, diag = estimate_V_second_order(model, X, theta)
        >>> print(f"V(θ) = {V:.4f}")
        >>> print(f"Q(θ|θ) = {diag['Q_self']:.4f}")
        >>> print(f"Improvement = {V - diag['Q_self']:.4f}")
    """
    diagnostics = {
        'Q_self': None,
        'grad_norm': 0.0,
        'newton_step_norm': 0.0,
        'hessian_min_eigenvalue': None,
        'hessian_condition_number': None,
        'regularization_used': 0.0,
        'newton_step_computed': False
    }

    # Compute responsibilities if not provided
    if responsibilities is None:
        try:
            responsibilities = model.e_step(X, theta)
        except Exception as e:
            warnings.warn(f"E-step failed: {e}. Returning 0 for V.")
            return 0.0, diagnostics

    # Compute Q(θ|θ) - the complete-data log-likelihood at self
    try:
        Q_self = model.compute_Q(theta, theta, X, responsibilities)
        diagnostics['Q_self'] = Q_self
    except Exception as e:
        warnings.warn(f"Q computation failed: {e}. Returning 0 for V.")
        return 0.0, diagnostics

    # Compute gradient ∇Q(θ|θ)
    try:
        grad_Q = model.compute_Q_gradient(theta, theta, X, responsibilities)
        grad_Q = np.asarray(grad_Q).flatten()
        grad_norm = np.linalg.norm(grad_Q)
        diagnostics['grad_norm'] = float(grad_norm)
    except Exception as e:
        warnings.warn(f"Gradient computation failed: {e}. Returning Q_self.")
        return Q_self, diagnostics

    # If gradient is very small, we're at/near optimum - just return Q_self
    if grad_norm < gradient_tol:
        return Q_self, diagnostics

    # Compute Hessian H_Q(θ|θ)
    try:
        H_Q = model.compute_Q_hessian(theta, theta, X, responsibilities)
        H_Q = np.asarray(H_Q)

        # Ensure Hessian is square
        if H_Q.ndim != 2 or H_Q.shape[0] != H_Q.shape[1]:
            warnings.warn(f"Invalid Hessian shape: {H_Q.shape}. Returning Q_self.")
            return Q_self, diagnostics

    except Exception as e:
        warnings.warn(f"Hessian computation failed: {e}. Returning Q_self.")
        return Q_self, diagnostics

    # Analyze Hessian eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(H_Q)
        min_eig = float(np.min(eigenvalues))
        max_eig = float(np.max(np.abs(eigenvalues)))
        diagnostics['hessian_min_eigenvalue'] = min_eig

        if max_eig > 0:
            diagnostics['hessian_condition_number'] = float(
                np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-15]))
            )
    except Exception:
        pass  # Non-critical diagnostic

    # Regularize Hessian if needed (should be negative definite)
    # For Q-function Hessian, we expect it to be negative semi-definite
    # We need H to be invertible, so we regularize if min eigenvalue is too close to 0
    reg_used = 0.0
    H_reg = H_Q.copy()

    # Check if Hessian is negative definite (as expected for Q-function)
    # If min eigenvalue > -regularization, we need to regularize
    if min_eig is not None and min_eig > -regularization:
        # Make Hessian more negative definite
        reg_used = -min_eig + regularization
        H_reg = H_Q - reg_used * np.eye(H_Q.shape[0])
        diagnostics['regularization_used'] = float(reg_used)

    # Solve H_Q * step = -grad_Q for Newton step
    # Since H_Q should be negative definite, the step should point toward maximum
    try:
        # Use solve for numerical stability
        newton_step = np.linalg.solve(H_reg, -grad_Q)
        newton_step_norm = np.linalg.norm(newton_step)
        diagnostics['newton_step_norm'] = float(newton_step_norm)
        diagnostics['newton_step_computed'] = True
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Linear solve failed: {e}. Returning Q_self.")
        return Q_self, diagnostics

    # Compute V(θ) = Q(θ|θ) + 0.5 * grad_Q^T * step
    # This is the second-order approximation to max_θ' Q(θ'|θ)
    improvement = 0.5 * np.dot(grad_Q, newton_step)

    # The improvement should be positive (moving toward maximum)
    # If negative, something is wrong - clamp to 0
    if improvement < 0:
        warnings.warn(f"Negative improvement ({improvement:.4f}). Clamping to 0.")
        improvement = 0.0

    V = Q_self + improvement

    return V, diagnostics


def estimate_V_simple(
    model: Any,
    X: np.ndarray,
    theta: Dict[str, np.ndarray],
    responsibilities: Optional[np.ndarray] = None
) -> float:
    """
    Simple V estimation - just returns Q(θ|θ).

    This is a fallback when second-order estimation is not available
    or too expensive. It's equivalent to setting γ = 0 in lookahead EM.

    Args:
        model: Statistical model with compute_Q method.
        X: Data matrix.
        theta: Current parameters.
        responsibilities: Pre-computed responsibilities (optional).

    Returns:
        Q(θ|θ) value.
    """
    if responsibilities is None:
        responsibilities = model.e_step(X, theta)

    return model.compute_Q(theta, theta, X, responsibilities)


def estimate_V_one_step(
    model: Any,
    X: np.ndarray,
    theta: Dict[str, np.ndarray],
    responsibilities: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate V by computing one EM step and evaluating Q there.

    This is an exact computation of V for one step, but doesn't account
    for further steps. More expensive than second-order approximation
    but doesn't require gradients/Hessians.

    Args:
        model: Statistical model with e_step, m_step, and compute_Q.
        X: Data matrix.
        theta: Current parameters.
        responsibilities: Pre-computed responsibilities (optional).

    Returns:
        Tuple of (V_value, diagnostics).
    """
    diagnostics = {
        'Q_self': None,
        'Q_next': None,
        'improvement': None
    }

    if responsibilities is None:
        responsibilities = model.e_step(X, theta)

    # Q at current theta
    Q_self = model.compute_Q(theta, theta, X, responsibilities)
    diagnostics['Q_self'] = Q_self

    # One EM step
    theta_next = model.m_step(X, responsibilities)

    # Q at next theta (evaluated at next theta)
    resp_next = model.e_step(X, theta_next)
    Q_next = model.compute_Q(theta_next, theta_next, X, resp_next)
    diagnostics['Q_next'] = Q_next

    diagnostics['improvement'] = Q_next - Q_self

    return Q_next, diagnostics


# =============================================================================
# L-BFGS Approximation for Inverse Hessian
# =============================================================================

class LBFGSApproximation:
    """
    L-BFGS (Limited-memory BFGS) approximation for the inverse Hessian.

    Instead of computing and storing the full Hessian matrix H (O(p²) storage,
    O(p³) inversion), L-BFGS maintains a history of gradient and parameter
    differences to approximate H⁻¹ directly.

    Complexity:
    - Storage: O(mp) where m is memory size (typically 5-20)
    - Direction computation: O(mp) using two-loop recursion
    - Compare to exact: O(p²) storage, O(p³) solve

    For Lookahead EM, this approximates the Newton direction -H⁻¹∇Q without
    computing or storing the full Hessian.

    Example:
        >>> lbfgs = LBFGSApproximation(memory_size=10)
        >>> for iteration in range(max_iter):
        ...     grad = compute_gradient(theta)
        ...     direction = lbfgs.compute_direction(grad)
        ...     theta_new = theta + step_size * direction
        ...     lbfgs.update(theta_new - theta, grad_new - grad)
        ...     theta = theta_new

    Reference:
        Nocedal & Wright (2006), "Numerical Optimization", Chapter 7
    """

    def __init__(
        self,
        memory_size: int = 10,
        initial_scale: float = 1.0,
        min_curvature: float = 1e-10
    ):
        """
        Initialize L-BFGS approximation.

        Args:
            memory_size: Number of (s, y) pairs to store (m). Typical values: 5-20.
                        Larger m = better approximation but more memory/computation.
            initial_scale: Initial scaling for H₀ (identity scaled by this factor).
                          Will be auto-updated based on curvature.
            min_curvature: Minimum curvature (y^T s) to accept an update.
                          Prevents numerical issues from near-zero curvature.
        """
        self.memory_size = memory_size
        self.initial_scale = initial_scale
        self.min_curvature = min_curvature

        # Storage for (s, y) pairs: s = θ_{k+1} - θ_k, y = g_{k+1} - g_k
        self.s_history: deque = deque(maxlen=memory_size)
        self.y_history: deque = deque(maxlen=memory_size)
        self.rho_history: deque = deque(maxlen=memory_size)  # rho = 1 / (y^T s)

        # Current scale for H₀
        self.gamma = initial_scale

    def reset(self):
        """Clear the history (call when starting a new optimization)."""
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
        self.gamma = self.initial_scale

    def update(
        self,
        s: np.ndarray,
        y: np.ndarray
    ) -> bool:
        """
        Update L-BFGS history with a new (s, y) pair.

        Args:
            s: Parameter difference s = θ_{k+1} - θ_k
            y: Gradient difference y = ∇f_{k+1} - ∇f_k
               For Q-function: y = ∇Q(θ_{k+1}) - ∇Q(θ_k)

        Returns:
            True if update was accepted, False if skipped due to curvature.

        Note:
            For maximization (like Q-function), gradients point uphill.
            The curvature condition y^T s > 0 should hold for valid steps.
        """
        s = np.asarray(s).flatten()
        y = np.asarray(y).flatten()

        # Compute curvature
        ys = np.dot(y, s)

        # Skip update if curvature is too small (would cause numerical issues)
        if abs(ys) < self.min_curvature:
            return False

        # For maximization problems, we expect ys < 0 (Hessian is negative definite)
        # L-BFGS formula works with |ys|
        rho = 1.0 / ys

        # Store the pair
        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.rho_history.append(rho)

        # Update scaling factor: gamma = s^T y / y^T y
        yy = np.dot(y, y)
        if yy > 1e-15:
            self.gamma = ys / yy

        return True

    def compute_direction(
        self,
        gradient: np.ndarray,
        negate: bool = True
    ) -> np.ndarray:
        """
        Compute search direction using L-BFGS two-loop recursion.

        Computes: direction = -H⁻¹ @ gradient (if negate=True)
                  direction = H⁻¹ @ gradient  (if negate=False)

        For Lookahead EM optimization, use negate=True to get descent direction.

        Args:
            gradient: Current gradient vector ∇Q(θ)
            negate: If True, returns -H⁻¹g (descent direction for minimization,
                   ascent for maximization with negative definite H)

        Returns:
            Search direction vector (same shape as gradient)

        Algorithm:
            Two-loop recursion (Nocedal & Wright Algorithm 7.4):
            1. First loop: compute intermediate q using history
            2. Apply initial Hessian approximation H₀
            3. Second loop: compute final direction r
        """
        g = np.asarray(gradient).flatten()
        n_stored = len(self.s_history)

        # If no history yet, return scaled gradient
        if n_stored == 0:
            direction = self.gamma * g
            return -direction if negate else direction

        # Two-loop recursion
        q = g.copy()
        alpha = np.zeros(n_stored)

        # First loop (backward through history)
        for i in range(n_stored - 1, -1, -1):
            alpha[i] = self.rho_history[i] * np.dot(self.s_history[i], q)
            q = q - alpha[i] * self.y_history[i]

        # Apply initial Hessian approximation: r = H₀ @ q = gamma * q
        r = self.gamma * q

        # Second loop (forward through history)
        for i in range(n_stored):
            beta = self.rho_history[i] * np.dot(self.y_history[i], r)
            r = r + (alpha[i] - beta) * self.s_history[i]

        return -r if negate else r

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the L-BFGS state."""
        return {
            'memory_size': self.memory_size,
            'pairs_stored': len(self.s_history),
            'gamma': self.gamma,
            'curvatures': [1.0 / rho if abs(rho) > 1e-15 else 0.0
                          for rho in self.rho_history]
        }


def estimate_V_lbfgs(
    model: Any,
    X: np.ndarray,
    theta: Dict[str, np.ndarray],
    lbfgs: LBFGSApproximation,
    responsibilities: Optional[np.ndarray] = None,
    gradient_tol: float = 1e-10
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate lookahead value V(θ) using L-BFGS approximation.

    Similar to estimate_V_second_order but uses L-BFGS to approximate
    the inverse Hessian instead of computing it explicitly.

    V(θ) ≈ Q(θ|θ) + 0.5 × ∇Q^T H_approx^(-1) ∇Q

    This is much faster for high-dimensional problems:
    - O(mp) instead of O(p²) storage
    - O(mp) instead of O(p³) computation

    Args:
        model: Statistical model with compute_Q and compute_Q_gradient methods.
               Does NOT need compute_Q_hessian.
        X: Data matrix of shape (n_samples, n_features).
        theta: Current parameter dictionary.
        lbfgs: LBFGSApproximation instance (maintains history across calls).
        responsibilities: Pre-computed responsibilities (optional).
        gradient_tol: If gradient norm < this, skip improvement computation.

    Returns:
        Tuple of (V, diagnostics)

    Example:
        >>> lbfgs = LBFGSApproximation(memory_size=10)
        >>> V, diag = estimate_V_lbfgs(model, X, theta, lbfgs)
    """
    diagnostics = {
        'Q_self': None,
        'grad_norm': 0.0,
        'direction_norm': 0.0,
        'lbfgs_pairs_used': len(lbfgs.s_history),
        'lbfgs_gamma': lbfgs.gamma
    }

    # Compute responsibilities if not provided
    if responsibilities is None:
        try:
            responsibilities = model.e_step(X, theta)
        except Exception as e:
            warnings.warn(f"E-step failed: {e}. Returning 0 for V.")
            return 0.0, diagnostics

    # Compute Q(θ|θ)
    try:
        Q_self = model.compute_Q(theta, theta, X, responsibilities)
        diagnostics['Q_self'] = Q_self
    except Exception as e:
        warnings.warn(f"Q computation failed: {e}. Returning 0 for V.")
        return 0.0, diagnostics

    # Compute gradient
    try:
        grad_Q = model.compute_Q_gradient(theta, theta, X, responsibilities)
        grad_Q = np.asarray(grad_Q).flatten()
        grad_norm = np.linalg.norm(grad_Q)
        diagnostics['grad_norm'] = float(grad_norm)
    except Exception as e:
        warnings.warn(f"Gradient computation failed: {e}. Returning Q_self.")
        return Q_self, diagnostics

    # If gradient is very small, we're at/near optimum
    if grad_norm < gradient_tol:
        return Q_self, diagnostics

    # Use L-BFGS to compute direction = -H⁻¹ @ grad
    direction = lbfgs.compute_direction(grad_Q, negate=True)
    diagnostics['direction_norm'] = float(np.linalg.norm(direction))

    # Compute V(θ) = Q(θ|θ) + 0.5 * grad^T * direction
    # Note: direction = -H⁻¹ @ grad, so grad^T @ direction = -grad^T @ H⁻¹ @ grad
    # For negative definite H, this should be positive (improvement)
    improvement = -0.5 * np.dot(grad_Q, direction)

    # Clamp negative improvements
    if improvement < 0:
        improvement = 0.0

    V = Q_self + improvement
    return V, diagnostics


# =============================================================================
# Test Model with Gradient/Hessian Support
# =============================================================================

class QuadraticTestModel:
    """
    Simple quadratic model for testing V estimation.

    The Q-function is: Q(θ|θ₀) = -0.5 * ||θ - θ*||² + const
    where θ* is the true optimum.

    This has known analytical solutions for V(θ).
    """

    def __init__(self, n_params: int = 2, theta_star: Optional[np.ndarray] = None):
        self.n_params = n_params
        self.theta_star = theta_star if theta_star is not None else np.zeros(n_params)

    def e_step(self, X, theta):
        """Return dummy responsibilities."""
        return np.ones((len(X), 1)) / len(X)

    def m_step(self, X, responsibilities):
        """Return optimal theta."""
        return {'params': self.theta_star.copy()}

    def compute_Q(self, theta, theta_old, X, responsibilities):
        """Quadratic Q-function centered at theta_star."""
        params = theta['params']
        diff = params - self.theta_star
        return -0.5 * np.sum(diff ** 2)

    def compute_Q_gradient(self, theta, theta_old, X, responsibilities):
        """Gradient: -(θ - θ*)."""
        params = theta['params']
        return -(params - self.theta_star)

    def compute_Q_hessian(self, theta, theta_old, X, responsibilities):
        """Hessian: -I."""
        return -np.eye(self.n_params)

    def log_likelihood(self, X, theta):
        """For interface compatibility."""
        return self.compute_Q(theta, theta, X, None)


class BlockDiagonalTestModel:
    """
    Model with block-diagonal Hessian for testing.

    Tests that V estimation works correctly with structured Hessians.
    """

    def __init__(self, n_blocks: int = 3, block_size: int = 2):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_params = n_blocks * block_size

    def e_step(self, X, theta):
        return np.ones((len(X), 1)) / len(X)

    def m_step(self, X, responsibilities):
        return {'params': np.zeros(self.n_params)}

    def compute_Q(self, theta, theta_old, X, responsibilities):
        params = theta['params']
        return -0.5 * np.sum(params ** 2)

    def compute_Q_gradient(self, theta, theta_old, X, responsibilities):
        return -theta['params']

    def compute_Q_hessian(self, theta, theta_old, X, responsibilities):
        """Block-diagonal Hessian with varying eigenvalues."""
        H = np.zeros((self.n_params, self.n_params))
        for i in range(self.n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            # Each block has eigenvalues -(i+1)
            H[start:end, start:end] = -(i + 1) * np.eye(self.block_size)
        return H

    def log_likelihood(self, X, theta):
        return self.compute_Q(theta, theta, X, None)


# =============================================================================
# Unit Tests
# =============================================================================

def test_estimate_V_basic():
    """Test basic V estimation functionality."""
    print("Testing estimate_V_second_order basic...")

    model = QuadraticTestModel(n_params=2)
    X = np.random.randn(10, 2)

    # Theta away from optimum
    theta = {'params': np.array([1.0, 1.0])}

    V, diag = estimate_V_second_order(model, X, theta)

    # Check diagnostics
    assert diag['Q_self'] is not None, "Should compute Q_self"
    assert diag['grad_norm'] > 0, "Should have non-zero gradient"
    assert diag['newton_step_computed'], "Should compute Newton step"

    # V should be >= Q_self (lookahead should improve)
    assert V >= diag['Q_self'] - 1e-10, f"V ({V}) should be >= Q_self ({diag['Q_self']})"

    print(f"  Q_self = {diag['Q_self']:.4f}")
    print(f"  V = {V:.4f}")
    print(f"  Improvement = {V - diag['Q_self']:.4f}")
    print("  PASSED")


def test_estimate_V_at_optimum():
    """Test V estimation at the optimum."""
    print("Testing estimate_V_second_order at optimum...")

    model = QuadraticTestModel(n_params=2)
    X = np.random.randn(10, 2)

    # Theta at optimum
    theta = {'params': np.array([0.0, 0.0])}

    V, diag = estimate_V_second_order(model, X, theta)

    # At optimum, gradient should be near zero
    assert diag['grad_norm'] < 1e-10, "Gradient should be near zero at optimum"

    # V should equal Q_self at optimum
    assert abs(V - diag['Q_self']) < 1e-10, "V should equal Q_self at optimum"

    print(f"  V = Q_self = {V:.4f}")
    print("  PASSED")


def test_estimate_V_quadratic_exact():
    """Test that V estimation is exact for quadratic Q."""
    print("Testing estimate_V_second_order exactness for quadratic...")

    model = QuadraticTestModel(n_params=2)
    X = np.random.randn(10, 2)

    # Theta away from optimum
    theta = {'params': np.array([2.0, 1.0])}

    V, diag = estimate_V_second_order(model, X, theta)

    # For quadratic Q, V should equal Q at the optimum
    # Q* = Q(θ*|θ*) = 0 (since θ* = 0)
    Q_optimal = 0.0

    # V should be very close to Q_optimal
    assert abs(V - Q_optimal) < 1e-8, f"V ({V}) should equal Q_optimal ({Q_optimal})"

    print(f"  V = {V:.6f}")
    print(f"  Q_optimal = {Q_optimal:.6f}")
    print("  PASSED")


def test_estimate_V_block_diagonal():
    """Test V estimation with block-diagonal Hessian."""
    print("Testing estimate_V_second_order with block-diagonal Hessian...")

    model = BlockDiagonalTestModel(n_blocks=3, block_size=2)
    X = np.random.randn(10, 2)

    theta = {'params': np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])}

    V, diag = estimate_V_second_order(model, X, theta)

    assert diag['Q_self'] is not None, "Should compute Q_self"
    assert diag['newton_step_computed'], "Should compute Newton step"
    assert V >= diag['Q_self'] - 1e-10, "V should be >= Q_self"

    print(f"  Q_self = {diag['Q_self']:.4f}")
    print(f"  V = {V:.4f}")
    print("  PASSED")


def test_estimate_V_regularization():
    """Test that regularization handles ill-conditioned Hessians."""
    print("Testing regularization for ill-conditioned Hessian...")

    class IllConditionedModel:
        def e_step(self, X, theta):
            return np.ones((len(X), 1))

        def compute_Q(self, theta, theta_old, X, resp):
            return -0.5 * np.sum(theta['params'] ** 2)

        def compute_Q_gradient(self, theta, theta_old, X, resp):
            return -theta['params']

        def compute_Q_hessian(self, theta, theta_old, X, resp):
            # Nearly singular Hessian (all eigenvalues very small)
            H = -1e-10 * np.eye(2)
            return H

    model = IllConditionedModel()
    X = np.random.randn(10, 2)
    theta = {'params': np.array([1.0, 1.0])}

    # Should not raise an error
    V, diag = estimate_V_second_order(model, X, theta, regularization=1e-6)

    assert V is not None, "Should return a value"
    assert diag['regularization_used'] > 0, "Should use regularization"

    print(f"  Regularization used: {diag['regularization_used']:.2e}")
    print("  PASSED")


def test_estimate_V_simple():
    """Test simple V estimation."""
    print("Testing estimate_V_simple...")

    model = QuadraticTestModel(n_params=2)
    X = np.random.randn(10, 2)
    theta = {'params': np.array([1.0, 1.0])}

    V_simple = estimate_V_simple(model, X, theta)
    V_full, diag = estimate_V_second_order(model, X, theta)

    # Simple should equal Q_self
    assert abs(V_simple - diag['Q_self']) < 1e-10, "Simple V should equal Q_self"

    print(f"  V_simple = {V_simple:.4f}")
    print(f"  V_full = {V_full:.4f}")
    print("  PASSED")


def test_estimate_V_one_step():
    """Test one-step V estimation."""
    print("Testing estimate_V_one_step...")

    model = QuadraticTestModel(n_params=2)
    X = np.random.randn(10, 2)
    theta = {'params': np.array([1.0, 1.0])}

    V, diag = estimate_V_one_step(model, X, theta)

    assert diag['Q_self'] is not None, "Should compute Q_self"
    assert diag['Q_next'] is not None, "Should compute Q_next"
    assert diag['improvement'] >= -1e-10, "Improvement should be non-negative"

    print(f"  Q_self = {diag['Q_self']:.4f}")
    print(f"  Q_next = {diag['Q_next']:.4f}")
    print(f"  Improvement = {diag['improvement']:.4f}")
    print("  PASSED")


def test_diagnostics_complete():
    """Test that all diagnostics are provided."""
    print("Testing diagnostics completeness...")

    model = QuadraticTestModel(n_params=3)
    X = np.random.randn(10, 3)
    theta = {'params': np.array([1.0, 2.0, 3.0])}

    V, diag = estimate_V_second_order(model, X, theta)

    required_keys = [
        'Q_self', 'grad_norm', 'newton_step_norm',
        'hessian_min_eigenvalue', 'hessian_condition_number',
        'regularization_used', 'newton_step_computed'
    ]

    for key in required_keys:
        assert key in diag, f"Missing diagnostic key: {key}"

    print(f"  All {len(required_keys)} diagnostic keys present")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running value_estimation.py unit tests")
    print("=" * 60)

    test_estimate_V_basic()
    test_estimate_V_at_optimum()
    test_estimate_V_quadratic_exact()
    test_estimate_V_block_diagonal()
    test_estimate_V_regularization()
    test_estimate_V_simple()
    test_estimate_V_one_step()
    test_diagnostics_complete()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
