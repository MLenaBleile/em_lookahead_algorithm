"""
Gaussian Mixture Model Implementation

This module provides a complete GMM implementation with all methods
required by the EM algorithms, including gradient and Hessian computation
for the lookahead EM algorithm.

The GMM uses diagonal covariances for efficiency, resulting in a
block-diagonal Hessian structure that can be inverted efficiently.

Example usage:
    >>> model = GaussianMixtureModel(n_components=5, n_features=2)
    >>> theta_init = model.initialize_kmeans(X, random_state=42)
    >>> resp = model.e_step(X, theta_init)
    >>> theta_new = model.m_step(X, resp)
"""

import numpy as np
from scipy import linalg
from typing import Dict, Tuple, Optional, List, Any
from sklearn.cluster import KMeans


class GaussianMixtureModel:
    """
    Gaussian Mixture Model with diagonal covariances.

    Provides all methods needed for standard EM, lookahead EM, and
    other EM acceleration methods.

    The model uses diagonal covariances, which results in:
    - O(Kd) parameters for covariances instead of O(Kd²)
    - Block-diagonal Hessian structure (K blocks of d×d)
    - Efficient O(Kd³) Hessian inversion instead of O((Kd)³)

    Attributes:
        n_components: Number of mixture components (K).
        n_features: Dimensionality of data (d).
        min_covar: Minimum covariance value to prevent collapse.

    Example:
        >>> model = GaussianMixtureModel(n_components=3, n_features=2)
        >>> X = np.random.randn(100, 2)
        >>> theta = model.initialize_random(X, random_state=42)
        >>> ll = model.log_likelihood(X, theta)
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        min_covar: float = 1e-6
    ):
        """
        Initialize Gaussian Mixture Model.

        Args:
            n_components: Number of mixture components (K).
            n_features: Dimensionality of data (d).
            min_covar: Minimum variance value to prevent collapse.
        """
        self.n_components = n_components
        self.n_features = n_features
        self.min_covar = min_covar

    # =========================================================================
    # Core EM Methods
    # =========================================================================

    def e_step(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        E-step: Compute responsibilities (posterior probabilities).

        Computes γ_ik = P(z_i = k | x_i, θ) for all observations and components.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            theta: Parameter dictionary with 'mu', 'sigma', 'pi'.

        Returns:
            Responsibilities of shape (n_samples, n_components).
            Each row sums to 1.

        Example:
            >>> resp = model.e_step(X, theta)
            >>> assert resp.shape == (len(X), model.n_components)
            >>> assert np.allclose(resp.sum(axis=1), 1.0)
        """
        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        # Compute log responsibilities (log-sum-exp for numerical stability)
        log_resp = np.zeros((n, K))

        for k in range(K):
            log_resp[:, k] = self._log_component_density(X, mu[k], sigma[k])
            log_resp[:, k] += np.log(pi[k] + 1e-300)

        # Normalize using log-sum-exp trick
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        log_resp_shifted = log_resp - log_resp_max
        resp = np.exp(log_resp_shifted)
        resp /= resp.sum(axis=1, keepdims=True)

        # Handle numerical issues
        resp = np.clip(resp, 1e-300, 1.0)
        resp /= resp.sum(axis=1, keepdims=True)

        return resp

    def m_step(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        M-step: Update parameters given responsibilities.

        Computes maximum likelihood estimates of θ given the responsibilities.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            responsibilities: Responsibility matrix of shape (n_samples, n_components).

        Returns:
            Updated parameter dictionary with 'mu', 'sigma', 'pi'.

        Example:
            >>> theta_new = model.m_step(X, resp)
            >>> assert theta_new['mu'].shape == (model.n_components, model.n_features)
        """
        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components
        d = self.n_features

        # Soft counts
        Nk = responsibilities.sum(axis=0) + 1e-10

        # Update mixing proportions
        pi = Nk / n

        # Update means
        mu = np.zeros((K, d))
        for k in range(K):
            mu[k] = np.dot(responsibilities[:, k], X) / Nk[k]

        # Update variances (diagonal)
        sigma = np.zeros((K, d))
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = np.dot(responsibilities[:, k], diff ** 2) / Nk[k]
            sigma[k] = np.maximum(sigma[k], self.min_covar)

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    def log_likelihood(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute log-likelihood of data under the model.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            theta: Parameter dictionary with 'mu', 'sigma', 'pi'.

        Returns:
            Log-likelihood value (sum over all observations).

        Example:
            >>> ll = model.log_likelihood(X, theta)
            >>> print(f"Log-likelihood: {ll:.2f}")
        """
        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        # Compute log probability for each component
        log_probs = np.zeros((n, K))
        for k in range(K):
            log_probs[:, k] = self._log_component_density(X, mu[k], sigma[k])
            log_probs[:, k] += np.log(pi[k] + 1e-300)

        # Log-sum-exp for mixture
        max_log = np.max(log_probs, axis=1)
        log_sum = max_log + np.log(np.sum(np.exp(log_probs - max_log[:, None]), axis=1))

        return float(np.sum(log_sum))

    # =========================================================================
    # Q-function and Derivatives
    # =========================================================================

    def compute_Q(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> float:
        """
        Compute Q(θ|θ_old) - the expected complete-data log-likelihood.

        Q(θ|θ') = Σ_i Σ_k γ_ik(θ') × [log π_k + log N(x_i; μ_k, σ_k²)]

        Args:
            theta: Parameters to evaluate.
            theta_old: Parameters used to compute responsibilities.
            X: Data matrix.
            responsibilities: Responsibilities computed at theta_old.

        Returns:
            Q-function value.
        """
        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        Q = 0.0

        for k in range(K):
            # Sum of responsibilities for component k
            Nk = responsibilities[:, k].sum()

            # Log mixing proportion term
            Q += Nk * np.log(pi[k] + 1e-300)

            # Log Gaussian density terms
            log_density = self._log_component_density(X, mu[k], sigma[k])
            Q += np.dot(responsibilities[:, k], log_density)

        return float(Q)

    def compute_Q_gradient(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of Q with respect to means (flattened).

        ∂Q/∂μ_k = Σ_i γ_ik × (x_i - μ_k) / σ_k²

        Args:
            theta: Parameters to evaluate.
            theta_old: Parameters used to compute responsibilities.
            X: Data matrix.
            responsibilities: Responsibilities computed at theta_old.

        Returns:
            Gradient vector of shape (K * d,).
        """
        X = np.asarray(X)
        K = self.n_components
        d = self.n_features

        mu = theta['mu']
        sigma = theta['sigma']

        grad = np.zeros((K, d))

        for k in range(K):
            diff = X - mu[k]  # (n, d)
            # Gradient: Σ_i γ_ik × (x_i - μ_k) / σ_k²
            weighted_diff = responsibilities[:, k:k+1] * diff / sigma[k]
            grad[k] = weighted_diff.sum(axis=0)

        return grad.flatten()

    def compute_Q_hessian(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute Hessian of Q with respect to means.

        The Hessian is block-diagonal with K blocks of size d×d:
        H_k = -Σ_i γ_ik × diag(1/σ_k²)

        Args:
            theta: Parameters to evaluate.
            theta_old: Parameters used to compute responsibilities.
            X: Data matrix.
            responsibilities: Responsibilities computed at theta_old.

        Returns:
            Hessian matrix of shape (K*d, K*d).
        """
        K = self.n_components
        d = self.n_features

        sigma = theta['sigma']

        # Build block-diagonal Hessian
        H = np.zeros((K * d, K * d))

        for k in range(K):
            # Soft count for component k
            Nk = responsibilities[:, k].sum()

            # Block for component k: -Nk × diag(1/σ_k²)
            start = k * d
            end = start + d
            H[start:end, start:end] = -np.diag(Nk / sigma[k])

        return H

    def compute_Q_hessian_blocks(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> List[np.ndarray]:
        """
        Compute Hessian blocks for efficient inversion.

        Returns list of K diagonal matrices (d×d each).

        Args:
            theta: Parameters to evaluate.
            theta_old: Parameters used to compute responsibilities.
            X: Data matrix.
            responsibilities: Responsibilities computed at theta_old.

        Returns:
            List of K Hessian blocks.
        """
        K = self.n_components
        d = self.n_features
        sigma = theta['sigma']

        blocks = []
        for k in range(K):
            Nk = responsibilities[:, k].sum()
            block = -np.diag(Nk / sigma[k])
            blocks.append(block)

        return blocks

    def inv_hessian_blocks(
        self,
        hessian_blocks: List[np.ndarray],
        regularization: float = 1e-6
    ) -> List[np.ndarray]:
        """
        Efficiently invert block-diagonal Hessian.

        Since each block is diagonal, inversion is O(d) per block
        instead of O(d³).

        Args:
            hessian_blocks: List of K Hessian blocks.
            regularization: Regularization for numerical stability.

        Returns:
            List of K inverse blocks.
        """
        inv_blocks = []

        for H_k in hessian_blocks:
            # Since H_k is diagonal, just invert the diagonal
            diag = np.diag(H_k)
            # Regularize to ensure invertibility
            diag = np.minimum(diag, -regularization)
            inv_diag = 1.0 / diag
            inv_blocks.append(np.diag(inv_diag))

        return inv_blocks

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def initialize_kmeans(
        self,
        X: np.ndarray,
        random_state: Optional[int] = None,
        max_iter: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Initialize parameters using K-means clustering.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            random_state: Random seed for reproducibility.
            max_iter: Maximum K-means iterations.

        Returns:
            Initial parameter dictionary.

        Example:
            >>> theta_init = model.initialize_kmeans(X, random_state=42)
        """
        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components
        d = self.n_features

        # Run K-means
        kmeans = KMeans(
            n_clusters=K,
            max_iter=max_iter,
            n_init=1,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X)

        # Initialize from clusters
        mu = np.zeros((K, d))
        sigma = np.zeros((K, d))
        pi = np.zeros(K)

        for k in range(K):
            mask = (labels == k)
            count = mask.sum()

            if count > 0:
                mu[k] = X[mask].mean(axis=0)
                sigma[k] = X[mask].var(axis=0) + self.min_covar
                pi[k] = count / n
            else:
                # Empty cluster - initialize randomly
                mu[k] = X[np.random.randint(n)]
                sigma[k] = np.ones(d)
                pi[k] = 1.0 / K

        # Ensure pi sums to 1
        pi = pi / pi.sum()

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    def initialize_random(
        self,
        X: np.ndarray,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Initialize parameters randomly.

        Means are sampled from the data, variances from data variance,
        and mixing proportions uniformly.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            random_state: Random seed for reproducibility.

        Returns:
            Initial parameter dictionary.
        """
        if random_state is not None:
            np.random.seed(random_state)

        X = np.asarray(X)
        n = X.shape[0]
        K = self.n_components
        d = self.n_features

        # Random means from data points
        indices = np.random.choice(n, size=K, replace=False)
        mu = X[indices].copy()

        # Variance based on data
        data_var = X.var(axis=0)
        sigma = np.tile(data_var, (K, 1)) * np.random.uniform(0.5, 1.5, (K, d))
        sigma = np.maximum(sigma, self.min_covar)

        # Uniform mixing proportions
        pi = np.ones(K) / K

        return {'mu': mu, 'sigma': sigma, 'pi': pi}

    # =========================================================================
    # Constraint Projection
    # =========================================================================

    def project_to_feasible(
        self,
        theta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Project parameters to feasible set.

        Ensures:
        - sigma > min_covar (positive variances)
        - pi > 0 and sum(pi) = 1 (valid probabilities)

        Args:
            theta: Parameters to project.

        Returns:
            Projected parameters.
        """
        theta = {k: np.array(v, copy=True) for k, v in theta.items()}

        # Project sigma to positive
        theta['sigma'] = np.maximum(theta['sigma'], self.min_covar)

        # Project pi to probability simplex
        pi = theta['pi']
        pi = np.maximum(pi, 1e-10)
        theta['pi'] = pi / pi.sum()

        return theta

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _log_component_density(
        self,
        X: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Compute log density of Gaussian component (diagonal covariance).

        log N(x; μ, Σ) = -0.5 × [d×log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ)]

        For diagonal Σ with variances σ²:
        = -0.5 × [d×log(2π) + Σ_j log(σ_j²) + Σ_j (x_j - μ_j)² / σ_j²]

        Args:
            X: Data matrix of shape (n, d).
            mu: Mean vector of shape (d,).
            sigma: Variance vector of shape (d,).

        Returns:
            Log densities of shape (n,).
        """
        d = self.n_features
        diff = X - mu

        log_det = np.sum(np.log(sigma + 1e-300))
        mahal = np.sum(diff ** 2 / (sigma + 1e-300), axis=1)

        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)

    def get_n_parameters(self) -> int:
        """Get total number of parameters."""
        K = self.n_components
        d = self.n_features
        return K * d + K * d + (K - 1)  # means + variances + (K-1) for pi

    def sample(
        self,
        theta: Dict[str, np.ndarray],
        n_samples: int,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the GMM.

        Args:
            theta: Parameter dictionary.
            n_samples: Number of samples to generate.
            random_state: Random seed.

        Returns:
            Tuple of (X, z) where X is samples and z is component assignments.
        """
        if random_state is not None:
            np.random.seed(random_state)

        K = self.n_components
        d = self.n_features

        mu = theta['mu']
        sigma = theta['sigma']
        pi = theta['pi']

        # Sample component assignments
        z = np.random.choice(K, size=n_samples, p=pi)

        # Sample from components
        X = np.zeros((n_samples, d))
        for i in range(n_samples):
            k = z[i]
            X[i] = np.random.randn(d) * np.sqrt(sigma[k]) + mu[k]

        return X, z


# =============================================================================
# Unit Tests
# =============================================================================

def test_gmm_e_step():
    """Test E-step computation."""
    print("Testing GMM E-step...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    # Generate simple data
    X = np.vstack([
        np.random.randn(50, 2) + np.array([-3, 0]),
        np.random.randn(50, 2) + np.array([3, 0])
    ])

    theta = {
        'mu': np.array([[-3, 0], [3, 0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    resp = model.e_step(X, theta)

    # Check shape
    assert resp.shape == (100, 2), f"Wrong shape: {resp.shape}"

    # Check normalization
    assert np.allclose(resp.sum(axis=1), 1.0), "Responsibilities should sum to 1"

    # Check that data from cluster 1 has high responsibility for component 1
    assert resp[:50, 0].mean() > 0.9, "First half should be assigned to first component"
    assert resp[50:, 1].mean() > 0.9, "Second half should be assigned to second component"

    print("  PASSED")


def test_gmm_m_step():
    """Test M-step computation."""
    print("Testing GMM M-step...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    # Generate simple data
    X = np.vstack([
        np.random.randn(50, 2) + np.array([-3, 0]),
        np.random.randn(50, 2) + np.array([3, 0])
    ])

    # Perfect responsibilities
    resp = np.zeros((100, 2))
    resp[:50, 0] = 1.0
    resp[50:, 1] = 1.0

    theta = model.m_step(X, resp)

    # Check shapes
    assert theta['mu'].shape == (2, 2), f"Wrong mu shape: {theta['mu'].shape}"
    assert theta['sigma'].shape == (2, 2), f"Wrong sigma shape: {theta['sigma'].shape}"
    assert theta['pi'].shape == (2,), f"Wrong pi shape: {theta['pi'].shape}"

    # Check means are close to true values
    assert np.abs(theta['mu'][0, 0] - (-3)) < 0.5, "First mean should be near -3"
    assert np.abs(theta['mu'][1, 0] - 3) < 0.5, "Second mean should be near 3"

    # Check mixing proportions
    assert np.allclose(theta['pi'], [0.5, 0.5]), "Pi should be [0.5, 0.5]"

    print("  PASSED")


def test_gmm_log_likelihood():
    """Test log-likelihood computation."""
    print("Testing GMM log-likelihood...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    X = np.random.randn(100, 2)

    theta = {
        'mu': np.array([[0, 0], [2, 2]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    ll = model.log_likelihood(X, theta)

    # Log-likelihood should be negative for reasonable data
    assert ll < 0, "Log-likelihood should be negative"
    assert np.isfinite(ll), "Log-likelihood should be finite"

    print(f"  Log-likelihood: {ll:.2f}")
    print("  PASSED")


def test_gmm_em_convergence():
    """Test that EM converges on GMM."""
    print("Testing GMM EM convergence...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    # Generate well-separated data
    X = np.vstack([
        np.random.randn(100, 2) + np.array([-3, 0]),
        np.random.randn(100, 2) + np.array([3, 0])
    ])

    theta = model.initialize_kmeans(X, random_state=42)
    ll_history = [model.log_likelihood(X, theta)]

    # Run EM
    for _ in range(20):
        resp = model.e_step(X, theta)
        theta = model.m_step(X, resp)
        ll_history.append(model.log_likelihood(X, theta))

    # Check monotonicity
    for i in range(1, len(ll_history)):
        assert ll_history[i] >= ll_history[i-1] - 1e-8, \
            f"Likelihood decreased: {ll_history[i-1]:.4f} -> {ll_history[i]:.4f}"

    print(f"  Initial LL: {ll_history[0]:.2f}")
    print(f"  Final LL: {ll_history[-1]:.2f}")
    print("  PASSED")


def test_gmm_q_function():
    """Test Q-function computation."""
    print("Testing GMM Q-function...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    X = np.random.randn(50, 2)
    theta = model.initialize_random(X, random_state=42)
    resp = model.e_step(X, theta)

    Q = model.compute_Q(theta, theta, X, resp)

    assert np.isfinite(Q), "Q should be finite"

    print(f"  Q(θ|θ) = {Q:.2f}")
    print("  PASSED")


def test_gmm_gradient():
    """Test gradient computation via finite differences."""
    print("Testing GMM gradient...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    X = np.random.randn(50, 2)
    theta = model.initialize_random(X, random_state=42)
    resp = model.e_step(X, theta)

    grad = model.compute_Q_gradient(theta, theta, X, resp)

    # Check gradient via finite differences
    eps = 1e-5
    grad_fd = np.zeros_like(grad)

    mu_flat = theta['mu'].flatten()
    for i in range(len(mu_flat)):
        mu_plus = mu_flat.copy()
        mu_plus[i] += eps
        theta_plus = {**theta, 'mu': mu_plus.reshape(theta['mu'].shape)}

        mu_minus = mu_flat.copy()
        mu_minus[i] -= eps
        theta_minus = {**theta, 'mu': mu_minus.reshape(theta['mu'].shape)}

        Q_plus = model.compute_Q(theta_plus, theta, X, resp)
        Q_minus = model.compute_Q(theta_minus, theta, X, resp)

        grad_fd[i] = (Q_plus - Q_minus) / (2 * eps)

    # Check gradient matches finite differences
    rel_error = np.abs(grad - grad_fd) / (np.abs(grad_fd) + 1e-8)
    assert rel_error.max() < 0.01, f"Gradient error too large: {rel_error.max():.4f}"

    print(f"  Max relative error: {rel_error.max():.6f}")
    print("  PASSED")


def test_gmm_hessian():
    """Test Hessian is block-diagonal and negative semi-definite."""
    print("Testing GMM Hessian...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=3, n_features=2)

    X = np.random.randn(50, 2)
    theta = model.initialize_random(X, random_state=42)
    resp = model.e_step(X, theta)

    H = model.compute_Q_hessian(theta, theta, X, resp)

    K = model.n_components
    d = model.n_features

    # Check shape
    assert H.shape == (K * d, K * d), f"Wrong Hessian shape: {H.shape}"

    # Check block-diagonal structure
    for k1 in range(K):
        for k2 in range(K):
            if k1 != k2:
                start1, end1 = k1 * d, (k1 + 1) * d
                start2, end2 = k2 * d, (k2 + 1) * d
                block = H[start1:end1, start2:end2]
                assert np.allclose(block, 0), f"Off-diagonal block ({k1},{k2}) not zero"

    # Check negative semi-definite
    eigenvalues = np.linalg.eigvalsh(H)
    assert np.all(eigenvalues <= 1e-8), "Hessian should be negative semi-definite"

    print(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    print("  PASSED")


def test_gmm_initialization():
    """Test initialization methods."""
    print("Testing GMM initialization...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=3, n_features=2)
    X = np.random.randn(100, 2)

    # Test K-means init
    theta_km = model.initialize_kmeans(X, random_state=42)
    assert theta_km['mu'].shape == (3, 2), "Wrong mu shape"
    assert np.allclose(theta_km['pi'].sum(), 1.0), "Pi should sum to 1"
    assert np.all(theta_km['sigma'] > 0), "Sigma should be positive"

    # Test random init
    theta_rand = model.initialize_random(X, random_state=42)
    assert theta_rand['mu'].shape == (3, 2), "Wrong mu shape"
    assert np.allclose(theta_rand['pi'].sum(), 1.0), "Pi should sum to 1"

    print("  PASSED")


def test_gmm_project_to_feasible():
    """Test constraint projection."""
    print("Testing GMM project_to_feasible...")

    model = GaussianMixtureModel(n_components=2, n_features=2)

    # Invalid theta
    theta_bad = {
        'mu': np.array([[0, 0], [1, 1]]),
        'sigma': np.array([[-1, 0.5], [0, -2]]),  # Negative variances
        'pi': np.array([0.3, 0.5])  # Doesn't sum to 1
    }

    theta_proj = model.project_to_feasible(theta_bad)

    # Check sigma is positive
    assert np.all(theta_proj['sigma'] > 0), "Sigma should be positive"

    # Check pi sums to 1
    assert np.allclose(theta_proj['pi'].sum(), 1.0), "Pi should sum to 1"

    print("  PASSED")


def test_gmm_sample():
    """Test sampling from GMM."""
    print("Testing GMM sampling...")

    np.random.seed(42)

    model = GaussianMixtureModel(n_components=2, n_features=2)

    theta = {
        'mu': np.array([[-3, 0], [3, 0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.3, 0.7])
    }

    X, z = model.sample(theta, n_samples=1000, random_state=42)

    assert X.shape == (1000, 2), f"Wrong X shape: {X.shape}"
    assert z.shape == (1000,), f"Wrong z shape: {z.shape}"

    # Check component proportions roughly match
    prop_0 = (z == 0).mean()
    assert 0.2 < prop_0 < 0.4, f"Component 0 proportion should be near 0.3, got {prop_0}"

    print(f"  Component proportions: {(z == 0).mean():.2f}, {(z == 1).mean():.2f}")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running gmm.py unit tests")
    print("=" * 60)

    test_gmm_e_step()
    test_gmm_m_step()
    test_gmm_log_likelihood()
    test_gmm_em_convergence()
    test_gmm_q_function()
    test_gmm_gradient()
    test_gmm_hessian()
    test_gmm_initialization()
    test_gmm_project_to_feasible()
    test_gmm_sample()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
