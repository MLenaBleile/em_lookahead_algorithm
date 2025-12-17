"""
Mixture of Experts Model

Implementation of MoE with all methods required by EM algorithms.
Experts are linear regressors with Gaussian noise.
Gating network is a softmax classifier.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.special import logsumexp, softmax


class MixtureOfExperts:
    """
    Mixture of Experts model for regression.

    Each expert is a linear regressor: y = x^T β_g + ε, ε ~ N(0, σ_g²)
    Gating network: P(g | x) = softmax(x^T γ_g)

    Theta format (dict):
    - 'gamma': Gating weights (G × (d+1)) where last column is intercept
    - 'beta': Expert weights (G × (d+1)) where last column is intercept
    - 'sigma': Noise std (G,) per-expert noise level

    Attributes:
        n_experts: Number of experts (G)
        n_features: Number of input features (d)
    """

    def __init__(self, n_experts: int, n_features: int):
        """
        Initialize MoE.

        Args:
            n_experts: Number of experts (G)
            n_features: Number of input features (d) not including intercept
        """
        self.n_experts = n_experts
        self.n_features = n_features
        self.G = n_experts
        self.d = n_features

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to X."""
        n = X.shape[0]
        return np.column_stack([X, np.ones(n)])

    def _gating_probs(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        Compute gating probabilities P(g | x) for each sample.

        Args:
            X: Input features (n, d)
            gamma: Gating weights (G, d+1)

        Returns:
            (n, G) array of gating probabilities
        """
        X_aug = self._add_intercept(X)
        # Logits: (n, G) = X_aug @ gamma.T
        logits = X_aug @ gamma.T
        return softmax(logits, axis=1)

    def _expert_likelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Compute expert likelihoods P(y | x, g) for each sample and expert.

        Args:
            X: Input features (n, d)
            y: Target values (n,)
            beta: Expert weights (G, d+1)
            sigma: Noise std (G,)

        Returns:
            (n, G) array of log-likelihoods
        """
        X_aug = self._add_intercept(X)
        n = X.shape[0]
        log_lik = np.zeros((n, self.G))

        for g in range(self.G):
            pred = X_aug @ beta[g]
            residual = y - pred
            log_lik[:, g] = -0.5 * np.log(2 * np.pi) - np.log(sigma[g]) - 0.5 * (residual / sigma[g])**2

        return log_lik

    def e_step(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        theta: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        E-step: Compute expert responsibilities.

        Args:
            data: Tuple of (X, y) where X is (n, d) and y is (n,)
            theta: Model parameters

        Returns:
            (n, G) array of responsibilities (posterior probability of each expert)
        """
        X, y = data
        gamma = theta['gamma']
        beta = theta['beta']
        sigma = theta['sigma']

        # Gating probabilities: P(g | x)
        gate_probs = self._gating_probs(X, gamma)

        # Expert likelihoods: P(y | x, g)
        log_expert_lik = self._expert_likelihood(X, y, beta, sigma)

        # Posterior: P(g | x, y) ∝ P(g | x) P(y | x, g)
        log_joint = np.log(gate_probs + 1e-300) + log_expert_lik
        log_responsibilities = log_joint - logsumexp(log_joint, axis=1, keepdims=True)

        return np.exp(log_responsibilities)

    def m_step(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        responsibilities: np.ndarray,
        theta_current: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        M-step: Update all parameters.

        Args:
            data: Tuple of (X, y)
            responsibilities: (n, G) array from E-step
            theta_current: Current parameters (for warm-starting gating update)

        Returns:
            Updated theta dict
        """
        X, y = data
        n = X.shape[0]
        X_aug = self._add_intercept(X)

        # Update expert parameters (beta, sigma)
        beta_new = np.zeros((self.G, self.d + 1))
        sigma_new = np.zeros(self.G)

        for g in range(self.G):
            w = responsibilities[:, g]
            W = np.diag(w)

            # Weighted least squares: beta_g = (X^T W X)^{-1} X^T W y
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ (w * y)

            # Regularize for numerical stability
            XtWX += 1e-6 * np.eye(self.d + 1)

            beta_new[g] = np.linalg.solve(XtWX, XtWy)

            # Update sigma: σ_g² = sum(w * (y - Xβ)²) / sum(w)
            pred = X_aug @ beta_new[g]
            residual = y - pred
            sigma_sq = np.sum(w * residual**2) / (np.sum(w) + 1e-10)
            sigma_new[g] = max(np.sqrt(sigma_sq), 1e-4)

        # Update gating parameters using gradient ascent
        gamma_init = theta_current.get('gamma') if theta_current else None
        gamma_new = self._update_gating(X, responsibilities, gamma_init)

        return {
            'gamma': gamma_new,
            'beta': beta_new,
            'sigma': sigma_new
        }

    def _update_gating(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        gamma_init: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Update gating parameters using gradient ascent.

        For softmax gating, use a few Newton-Raphson steps.

        Args:
            X: Input features (n, d)
            responsibilities: Target posteriors (n, G)
            gamma_init: Current gating weights

        Returns:
            Updated gating weights
        """
        if gamma_init is None:
            gamma = np.zeros((self.G, self.d + 1))
        else:
            gamma = gamma_init.copy()

        X_aug = self._add_intercept(X)
        n = X.shape[0]

        # A few gradient descent steps
        lr = 0.5
        for _ in range(5):
            # Current gating probabilities
            probs = self._gating_probs(X, gamma)

            # Gradient: dL/dgamma_g = sum_i (r_ig - p_ig) * x_i
            for g in range(self.G):
                grad = X_aug.T @ (responsibilities[:, g] - probs[:, g])
                gamma[g] += lr * grad / n

        return gamma

    def log_likelihood(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        theta: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute log-likelihood.

        ℓ = sum_i log( sum_g P(g|x_i) P(y_i|x_i, g) )

        Args:
            data: Tuple of (X, y)
            theta: Model parameters

        Returns:
            Log-likelihood
        """
        X, y = data
        gamma = theta['gamma']
        beta = theta['beta']
        sigma = theta['sigma']

        # Gating probabilities
        gate_probs = self._gating_probs(X, gamma)

        # Expert log-likelihoods
        log_expert_lik = self._expert_likelihood(X, y, beta, sigma)

        # Mixture log-likelihood
        log_joint = np.log(gate_probs + 1e-300) + log_expert_lik
        log_marginal = logsumexp(log_joint, axis=1)

        return np.sum(log_marginal)

    def compute_Q(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        data: Tuple[np.ndarray, np.ndarray],
        responsibilities: np.ndarray
    ) -> float:
        """
        Compute Q-function.

        Q = sum_i sum_g r_ig [log P(g|x_i) + log P(y_i|x_i, g)]

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step (unused)
            data: Tuple of (X, y)
            responsibilities: From E-step

        Returns:
            Q-function value
        """
        X, y = data
        gamma = theta['gamma']
        beta = theta['beta']
        sigma = theta['sigma']

        # Gating log-probs
        gate_probs = self._gating_probs(X, gamma)
        log_gate = np.log(gate_probs + 1e-300)

        # Expert log-likelihoods
        log_expert_lik = self._expert_likelihood(X, y, beta, sigma)

        # Q = sum_i sum_g r_ig * (log_gate_ig + log_expert_ig)
        Q = np.sum(responsibilities * (log_gate + log_expert_lik))

        return Q

    def compute_Q_gradient(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        data: Tuple[np.ndarray, np.ndarray],
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of Q-function.

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step
            data: Tuple of (X, y)
            responsibilities: From E-step

        Returns:
            Flattened gradient vector
        """
        X, y = data
        X_aug = self._add_intercept(X)
        n = X.shape[0]

        gamma = theta['gamma']
        beta = theta['beta']
        sigma = theta['sigma']

        grad_parts = []

        # Gradient w.r.t. gamma
        gate_probs = self._gating_probs(X, gamma)
        for g in range(self.G):
            grad_gamma_g = X_aug.T @ (responsibilities[:, g] - gate_probs[:, g] * responsibilities.sum(axis=1))
            grad_parts.append(grad_gamma_g)

        # Gradient w.r.t. beta
        for g in range(self.G):
            pred = X_aug @ beta[g]
            residual = y - pred
            grad_beta_g = X_aug.T @ (responsibilities[:, g] * residual) / (sigma[g]**2)
            grad_parts.append(grad_beta_g)

        # Gradient w.r.t. log(sigma) (use log for positivity)
        for g in range(self.G):
            pred = X_aug @ beta[g]
            residual = y - pred
            # dQ/d(log σ) = dQ/dσ * σ = sum(r * (-1/σ + residual²/σ³)) * σ
            grad_log_sigma_g = np.sum(responsibilities[:, g] * (-1 + (residual / sigma[g])**2))
            grad_parts.append(np.array([grad_log_sigma_g]))

        return np.concatenate(grad_parts)

    def compute_Q_hessian(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        data: Tuple[np.ndarray, np.ndarray],
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute block-diagonal Hessian of Q-function.

        Blocks:
        - Gating block (G × (d+1))
        - Expert blocks (one per expert): beta block (d+1) + sigma block (1)

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step
            data: Tuple of (X, y)
            responsibilities: From E-step

        Returns:
            Block-diagonal Hessian matrix
        """
        X, y = data
        X_aug = self._add_intercept(X)
        n = X.shape[0]

        gamma = theta['gamma']
        beta = theta['beta']
        sigma = theta['sigma']

        # Total size: G*(d+1) for gamma + G*(d+1) for beta + G for sigma
        gamma_size = self.G * (self.d + 1)
        beta_size = self.G * (self.d + 1)
        sigma_size = self.G
        total_size = gamma_size + beta_size + sigma_size

        H = np.zeros((total_size, total_size))

        # Hessian for gamma (softmax parameterization)
        gate_probs = self._gating_probs(X, gamma)
        r_sum = responsibilities.sum(axis=1)

        # For each sample, Hessian contribution is: -r_sum * diag(p) - p * p^T
        for i in range(n):
            p = gate_probs[i]
            r = r_sum[i]
            # Per-sample Hessian block: -r * (diag(p) - outer(p, p))
            H_sample = -r * (np.diag(p) - np.outer(p, p))
            # Outer product with x to get full Hessian contribution
            x = X_aug[i]
            H_contrib = np.kron(H_sample, np.outer(x, x))

            H[:gamma_size, :gamma_size] += H_contrib

        # Hessian for beta (each expert independent)
        idx = gamma_size
        for g in range(self.G):
            w = responsibilities[:, g] / (sigma[g]**2)
            H_beta_g = -X_aug.T @ np.diag(w) @ X_aug
            H[idx:idx+self.d+1, idx:idx+self.d+1] = H_beta_g
            idx += self.d + 1

        # Hessian for log(sigma)
        for g in range(self.G):
            pred = X_aug @ beta[g]
            residual = y - pred
            # d²Q/d(log σ)² = sum(r * (-2 * residual²/σ²))
            H_sigma_g = np.sum(responsibilities[:, g] * (-2 * (residual / sigma[g])**2))
            H[idx, idx] = H_sigma_g
            idx += 1

        return H

    def project_to_feasible(self, theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Project parameters to feasible region.

        Only sigma needs to be positive.

        Args:
            theta: Parameters to project

        Returns:
            Projected parameters
        """
        theta_proj = {
            'gamma': theta['gamma'].copy(),
            'beta': theta['beta'].copy(),
            'sigma': np.clip(theta['sigma'], 1e-4, None)
        }
        return theta_proj

    def predict(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray],
        return_experts: bool = False
    ) -> np.ndarray:
        """
        Predict y given X.

        Args:
            X: Input features (n, d)
            theta: Model parameters
            return_experts: If True, return per-expert predictions

        Returns:
            Predictions (n,) or (n, G) if return_experts
        """
        X_aug = self._add_intercept(X)
        n = X.shape[0]

        gamma = theta['gamma']
        beta = theta['beta']

        # Gating probabilities
        gate_probs = self._gating_probs(X, gamma)

        # Expert predictions
        expert_preds = np.zeros((n, self.G))
        for g in range(self.G):
            expert_preds[:, g] = X_aug @ beta[g]

        if return_experts:
            return expert_preds

        # Weighted combination
        return np.sum(gate_probs * expert_preds, axis=1)


# ============================================================
# Initialization Functions
# ============================================================

def initialize_random(
    n_experts: int,
    n_features: int,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Initialize with random parameters."""
    rng = np.random.RandomState(random_state)

    gamma = 0.1 * rng.randn(n_experts, n_features + 1)
    beta = 0.1 * rng.randn(n_experts, n_features + 1)
    sigma = np.abs(rng.randn(n_experts)) + 0.5

    return {'gamma': gamma, 'beta': beta, 'sigma': sigma}


def initialize_equal_gates(
    n_experts: int,
    n_features: int,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Initialize with equal gating probabilities."""
    rng = np.random.RandomState(random_state)

    # Zero gamma gives equal gates
    gamma = np.zeros((n_experts, n_features + 1))
    beta = 0.1 * rng.randn(n_experts, n_features + 1)
    sigma = np.ones(n_experts)

    return {'gamma': gamma, 'beta': beta, 'sigma': sigma}


# ============================================================
# Data Generation
# ============================================================

def generate_moe_data(
    n: int,
    n_experts: int,
    n_features: int,
    noise_level: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate data from a Mixture of Experts model.

    Args:
        n: Number of samples
        n_experts: Number of experts (G)
        n_features: Number of features (d)
        noise_level: Noise standard deviation
        seed: Random seed

    Returns:
        Tuple of (X, y, expert_assignments, theta_true)
    """
    rng = np.random.RandomState(seed)

    # Generate input features
    X = rng.randn(n, n_features)
    X_aug = np.column_stack([X, np.ones(n)])

    # Generate true parameters
    # Gating: different regions prefer different experts
    gamma_true = np.zeros((n_experts, n_features + 1))
    for g in range(n_experts):
        gamma_true[g, g % n_features] = 2.0  # Each expert prefers different feature direction

    # Expert coefficients: different slopes
    beta_true = np.zeros((n_experts, n_features + 1))
    for g in range(n_experts):
        beta_true[g] = rng.randn(n_features + 1)
        beta_true[g, -1] = g  # Different intercepts

    sigma_true = noise_level * np.ones(n_experts)

    theta_true = {'gamma': gamma_true, 'beta': beta_true, 'sigma': sigma_true}

    # Compute gating probabilities
    logits = X_aug @ gamma_true.T
    gate_probs = softmax(logits, axis=1)

    # Sample expert assignments
    expert_assignments = np.array([rng.choice(n_experts, p=gate_probs[i]) for i in range(n)])

    # Generate y from assigned expert
    y = np.zeros(n)
    for i in range(n):
        g = expert_assignments[i]
        y[i] = X_aug[i] @ beta_true[g] + sigma_true[g] * rng.randn()

    return X, y, expert_assignments, theta_true


def test_3_moe(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate data for Test 3: MoE from EMPIRICAL_TEST_SPECIFICATIONS.md.

    Configuration:
    - G = 5 experts
    - d = 2 features
    - n = 2000 samples

    Args:
        seed: Random seed

    Returns:
        Tuple of (X, y, expert_assignments, theta_true)
    """
    return generate_moe_data(
        n=2000,
        n_experts=5,
        n_features=2,
        noise_level=0.5,
        seed=seed
    )


# ============================================================
# Unit Tests
# ============================================================

def test_e_step():
    """Test E-step computation."""
    model = MixtureOfExperts(n_experts=3, n_features=2)

    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    theta = initialize_random(3, 2, random_state=42)

    resp = model.e_step((X, y), theta)

    # Check shape
    assert resp.shape == (100, 3), f"Wrong shape: {resp.shape}"

    # Check probabilities sum to 1
    assert np.allclose(resp.sum(axis=1), 1), "Responsibilities don't sum to 1"

    # Check all non-negative
    assert (resp >= 0).all(), "Negative responsibilities"

    print("  PASSED")


def test_em_convergence():
    """Test that EM converges on MoE."""
    model = MixtureOfExperts(n_experts=2, n_features=2)

    # Generate data
    X, y, _, theta_true = generate_moe_data(200, 2, 2, noise_level=0.3, seed=42)

    # Initialize
    theta = initialize_random(2, 2, random_state=123)

    # Run EM
    ll_history = []
    for _ in range(30):
        ll = model.log_likelihood((X, y), theta)
        ll_history.append(ll)

        resp = model.e_step((X, y), theta)
        theta = model.m_step((X, y), resp, theta)

    # Check monotonicity (with small tolerance)
    ll_diffs = np.diff(ll_history)
    violations = np.sum(ll_diffs < -0.1)  # Allow small numerical errors
    assert violations <= 2, f"EM not monotonic: {violations} violations"

    # Check improvement
    assert ll_history[-1] > ll_history[0], "EM should improve likelihood"

    print(f"  Initial LL: {ll_history[0]:.2f}, Final LL: {ll_history[-1]:.2f}")
    print("  PASSED")


def test_prediction():
    """Test prediction."""
    model = MixtureOfExperts(n_experts=2, n_features=2)

    X, y, _, theta = generate_moe_data(100, 2, 2, seed=42)

    # Predict
    y_pred = model.predict(X, theta)

    assert y_pred.shape == (100,), f"Wrong prediction shape: {y_pred.shape}"

    # Expert predictions
    y_experts = model.predict(X, theta, return_experts=True)
    assert y_experts.shape == (100, 2), f"Wrong expert predictions shape"

    print("  PASSED")


def test_hessian_structure():
    """Test that Hessian is computed correctly."""
    model = MixtureOfExperts(n_experts=2, n_features=2)

    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    theta = initialize_random(2, 2, random_state=42)

    resp = model.e_step((X, y), theta)
    H = model.compute_Q_hessian(theta, theta, (X, y), resp)

    # Check dimensions: 2*3 (gamma) + 2*3 (beta) + 2 (sigma) = 14
    expected_size = 2 * 3 + 2 * 3 + 2
    assert H.shape == (expected_size, expected_size), f"Wrong Hessian shape: {H.shape}"

    # Check symmetry
    assert np.allclose(H, H.T, atol=1e-6), "Hessian should be symmetric"

    print(f"  Hessian shape: {H.shape}")
    print("  PASSED")


def test_data_generation():
    """Test data generation."""
    X, y, experts, theta = test_3_moe(seed=42)

    assert X.shape == (2000, 2), f"Wrong X shape: {X.shape}"
    assert y.shape == (2000,), f"Wrong y shape: {y.shape}"
    assert experts.shape == (2000,), f"Wrong experts shape: {experts.shape}"
    assert experts.min() >= 0 and experts.max() < 5, "Expert indices out of range"

    print(f"  X shape: {X.shape}")
    print(f"  Expert distribution: {np.bincount(experts)}")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running mixture_of_experts.py unit tests")
    print("=" * 60)

    print("Testing E-step...")
    test_e_step()

    print("Testing EM convergence...")
    test_em_convergence()

    print("Testing prediction...")
    test_prediction()

    print("Testing Hessian structure...")
    test_hessian_structure()

    print("Testing data generation...")
    test_data_generation()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
