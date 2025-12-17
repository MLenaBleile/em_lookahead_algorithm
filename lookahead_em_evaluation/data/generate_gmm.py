"""
GMM Data Generation Module

Generates synthetic Gaussian Mixture Model data for testing.
Includes preset configurations for standard tests and various
initialization strategies.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt


def generate_gmm_data(
    n: int,
    K: int,
    d: int,
    separation: float = 2.0,
    sigma: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic GMM data.

    Args:
        n: Number of samples
        K: Number of mixture components
        d: Number of dimensions
        separation: Distance between component means (multiplier)
        sigma: Standard deviation for all components
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - X: (n, d) array of observations
        - z: (n,) array of true component assignments
        - theta_true: dict with keys 'pi', 'mu', 'sigma'
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate mixing proportions (uniform with small perturbation)
    pi = np.ones(K) / K
    pi = pi + 0.1 * np.random.randn(K)
    pi = np.abs(pi)
    pi = pi / pi.sum()

    # Generate means on a grid or randomly spaced
    if d == 2 and K <= 100:
        # For 2D, arrange means in a grid-like pattern
        grid_size = int(np.ceil(np.sqrt(K)))
        mu = np.zeros((K, d))
        for k in range(K):
            row = k // grid_size
            col = k % grid_size
            mu[k, 0] = col * separation
            mu[k, 1] = row * separation
    else:
        # For higher dimensions, use random means with separation
        mu = np.zeros((K, d))
        mu[0] = np.zeros(d)
        for k in range(1, K):
            # Try to find a point that is far from existing means
            for _ in range(100):
                candidate = np.random.randn(d) * separation * np.sqrt(K)
                min_dist = min(np.linalg.norm(candidate - mu[j])
                              for j in range(k))
                if min_dist > separation:
                    break
            mu[k] = candidate

    # Center the means
    mu = mu - mu.mean(axis=0)

    # Generate variances (all equal) - stored as 'sigma' to match GMM model interface
    sigma_arr = np.full((K, d), sigma ** 2)

    # Generate samples
    z = np.random.choice(K, size=n, p=pi)
    X = np.zeros((n, d))
    for k in range(K):
        mask = (z == k)
        n_k = mask.sum()
        if n_k > 0:
            X[mask] = mu[k] + np.random.randn(n_k, d) * np.sqrt(sigma_arr[k])

    theta_true = {
        'pi': pi,
        'mu': mu,
        'sigma': sigma_arr
    }

    return X, z, theta_true


def test_1_high_d(seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate data for Test 1: High-dimensional GMM.

    Configuration from spec:
    - K = 50 components
    - d = 2 dimensions
    - n = 5000 samples

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, z, theta_true)
    """
    return generate_gmm_data(
        n=5000,
        K=50,
        d=2,
        separation=3.0,  # Good separation for visualization
        sigma=0.5,
        seed=seed
    )


def test_4_simple(seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate data for Test 4: Simple GMM.

    Configuration from spec:
    - K = 5 components
    - d = 2 dimensions
    - n = 500 samples

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, z, theta_true)
    """
    return generate_gmm_data(
        n=500,
        K=5,
        d=2,
        separation=4.0,  # Good separation for simple case
        sigma=0.5,
        seed=seed
    )


def generate_initializations(
    X: np.ndarray,
    K: int,
    strategy: str,
    n_inits: int = 1,
    random_state: Optional[int] = None
) -> List[Dict[str, np.ndarray]]:
    """
    Generate initial parameter values for GMM EM.

    Args:
        X: (n, d) array of observations
        K: Number of mixture components
        strategy: One of 'kmeans', 'random', 'collapsed', 'far_away'
        n_inits: Number of initializations to generate
        random_state: Random seed

    Returns:
        List of n_inits theta_init dicts, each with keys 'pi', 'mu', 'sigma'
    """
    n, d = X.shape

    if random_state is not None:
        np.random.seed(random_state)

    initializations = []

    for init_idx in range(n_inits):
        seed = random_state + init_idx if random_state is not None else None

        if strategy == 'kmeans':
            theta_init = _init_kmeans(X, K, seed)
        elif strategy == 'random':
            theta_init = _init_random(X, K, seed)
        elif strategy == 'collapsed':
            theta_init = _init_collapsed(X, K)
        elif strategy == 'far_away':
            theta_init = _init_far_away(X, K, seed)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

        initializations.append(theta_init)

    return initializations


def _init_kmeans(X: np.ndarray, K: int, seed: Optional[int]) -> Dict[str, np.ndarray]:
    """Initialize using K-means clustering."""
    n, d = X.shape

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=1)
    labels = kmeans.fit_predict(X)

    mu = kmeans.cluster_centers_.copy()

    # Compute mixing proportions and variances from clusters
    pi = np.zeros(K)
    sigma_sq = np.zeros((K, d))

    for k in range(K):
        mask = (labels == k)
        n_k = mask.sum()
        pi[k] = max(n_k / n, 1e-6)

        if n_k > 1:
            sigma_sq[k] = np.var(X[mask], axis=0) + 1e-6
        else:
            sigma_sq[k] = np.var(X, axis=0) + 1e-6

    pi = pi / pi.sum()

    return {'pi': pi, 'mu': mu, 'sigma': sigma_sq}


def _init_random(X: np.ndarray, K: int, seed: Optional[int]) -> Dict[str, np.ndarray]:
    """Initialize randomly from data range."""
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape

    # Uniform mixing proportions
    pi = np.ones(K) / K

    # Random means within data range
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    mu = np.random.uniform(X_min, X_max, size=(K, d))

    # Use overall variance
    sigma_sq = np.tile(np.var(X, axis=0) + 1e-6, (K, 1))

    return {'pi': pi, 'mu': mu, 'sigma': sigma_sq}


def _init_collapsed(X: np.ndarray, K: int) -> Dict[str, np.ndarray]:
    """Initialize with all means at the data centroid."""
    n, d = X.shape

    # Uniform mixing proportions
    pi = np.ones(K) / K

    # All means at centroid
    centroid = X.mean(axis=0)
    mu = np.tile(centroid, (K, 1))

    # Use overall variance
    sigma_sq = np.tile(np.var(X, axis=0) + 1e-6, (K, 1))

    return {'pi': pi, 'mu': mu, 'sigma': sigma_sq}


def _init_far_away(X: np.ndarray, K: int, seed: Optional[int]) -> Dict[str, np.ndarray]:
    """Initialize with means far from data centroid."""
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape

    # Uniform mixing proportions
    pi = np.ones(K) / K

    # Means at distance 10 from centroid in random directions
    centroid = X.mean(axis=0)
    directions = np.random.randn(K, d)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    mu = centroid + 10 * directions

    # Use overall variance
    sigma_sq = np.tile(np.var(X, axis=0) + 1e-6, (K, 1))

    return {'pi': pi, 'mu': mu, 'sigma': sigma_sq}


def plot_gmm_data(
    X: np.ndarray,
    z: np.ndarray,
    theta_true: Optional[Dict[str, Any]] = None,
    title: str = "GMM Data",
    ax: Optional[plt.Axes] = None,
    show_centers: bool = True,
    alpha: float = 0.5
) -> plt.Axes:
    """
    Visualize 2D GMM data with true cluster assignments.

    Args:
        X: (n, d) array of observations (must be 2D)
        z: (n,) array of true component assignments
        theta_true: Optional dict with true parameters
        title: Plot title
        ax: Optional matplotlib axes
        show_centers: Whether to show cluster centers
        alpha: Point transparency

    Returns:
        matplotlib Axes object
    """
    if X.shape[1] != 2:
        raise ValueError("Visualization only supports 2D data")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique components
    K = len(np.unique(z))

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, min(K, 20)))
    if K > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, K))

    # Plot each cluster
    for k in range(K):
        mask = (z == k)
        color_idx = k % len(colors)
        ax.scatter(X[mask, 0], X[mask, 1],
                  c=[colors[color_idx]],
                  alpha=alpha,
                  s=10,
                  label=f'Component {k}' if K <= 10 else None)

    # Plot true centers
    if show_centers and theta_true is not None:
        mu = theta_true['mu']
        ax.scatter(mu[:, 0], mu[:, 1],
                  c='black',
                  marker='x',
                  s=100,
                  linewidths=2,
                  label='True centers')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)

    if K <= 10:
        ax.legend(loc='best', fontsize=8)

    ax.grid(True, alpha=0.3)

    return ax


def plot_gmm_with_ellipses(
    X: np.ndarray,
    theta: Dict[str, Any],
    z: Optional[np.ndarray] = None,
    title: str = "GMM with Fitted Components",
    ax: Optional[plt.Axes] = None,
    n_std: float = 2.0
) -> plt.Axes:
    """
    Visualize 2D GMM data with fitted component ellipses.

    Args:
        X: (n, d) array of observations (must be 2D)
        theta: dict with 'pi', 'mu', 'sigma'
        z: Optional (n,) array of component assignments
        title: Plot title
        ax: Optional matplotlib axes
        n_std: Number of standard deviations for ellipse radius

    Returns:
        matplotlib Axes object
    """
    from matplotlib.patches import Ellipse

    if X.shape[1] != 2:
        raise ValueError("Visualization only supports 2D data")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    mu = theta['mu']
    sigma_sq = theta['sigma']
    pi = theta['pi']
    K = len(pi)

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, min(K, 20)))
    if K > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, K))

    # Plot data points
    if z is not None:
        for k in range(K):
            mask = (z == k)
            if mask.sum() > 0:
                color_idx = k % len(colors)
                ax.scatter(X[mask, 0], X[mask, 1],
                          c=[colors[color_idx]],
                          alpha=0.3,
                          s=10)
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, c='gray')

    # Plot ellipses for each component
    for k in range(K):
        color_idx = k % len(colors)

        # Ellipse dimensions (diagonal covariance)
        width = 2 * n_std * np.sqrt(sigma_sq[k, 0])
        height = 2 * n_std * np.sqrt(sigma_sq[k, 1])

        ellipse = Ellipse(
            xy=(mu[k, 0], mu[k, 1]),
            width=width,
            height=height,
            facecolor='none',
            edgecolor=colors[color_idx],
            linewidth=2,
            linestyle='--',
            alpha=min(max(pi[k] * 3, 0.3), 1.0)  # More visible for larger weights, clamp to [0, 1]
        )
        ax.add_patch(ellipse)

        # Plot center
        ax.scatter(mu[k, 0], mu[k, 1],
                  c=[colors[color_idx]],
                  marker='x',
                  s=100,
                  linewidths=2)

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return ax


# ============================================================
# Unit Tests
# ============================================================

def test_generate_gmm_data():
    """Test basic GMM data generation."""
    X, z, theta_true = generate_gmm_data(n=100, K=3, d=2, seed=42)

    assert X.shape == (100, 2), "X shape mismatch"
    assert z.shape == (100,), "z shape mismatch"
    assert len(np.unique(z)) <= 3, "Too many components"
    assert theta_true['pi'].shape == (3,), "pi shape mismatch"
    assert theta_true['mu'].shape == (3, 2), "mu shape mismatch"
    assert theta_true['sigma'].shape == (3, 2), "sigma shape mismatch"
    assert np.abs(theta_true['pi'].sum() - 1.0) < 1e-10, "pi doesn't sum to 1"

    print("  PASSED")


def test_test_1_high_d():
    """Test Test 1 preset generation."""
    X, z, theta_true = test_1_high_d(seed=42)

    assert X.shape == (5000, 2), f"X shape: {X.shape}"
    assert z.shape == (5000,), f"z shape: {z.shape}"
    assert theta_true['mu'].shape == (50, 2), f"mu shape: {theta_true['mu'].shape}"

    print("  PASSED")


def test_test_4_simple():
    """Test Test 4 preset generation."""
    X, z, theta_true = test_4_simple(seed=42)

    assert X.shape == (500, 2), f"X shape: {X.shape}"
    assert z.shape == (500,), f"z shape: {z.shape}"
    assert theta_true['mu'].shape == (5, 2), f"mu shape: {theta_true['mu'].shape}"

    print("  PASSED")


def test_init_kmeans():
    """Test K-means initialization."""
    X, z, _ = generate_gmm_data(n=200, K=4, d=2, seed=42)
    inits = generate_initializations(X, K=4, strategy='kmeans', n_inits=1, random_state=42)

    assert len(inits) == 1, "Should return 1 initialization"
    theta_init = inits[0]
    assert theta_init['pi'].shape == (4,), "pi shape mismatch"
    assert theta_init['mu'].shape == (4, 2), "mu shape mismatch"
    assert theta_init['sigma'].shape == (4, 2), "sigma shape mismatch"
    assert np.abs(theta_init['pi'].sum() - 1.0) < 1e-10, "pi doesn't sum to 1"

    print("  PASSED")


def test_init_random():
    """Test random initialization."""
    X, z, _ = generate_gmm_data(n=200, K=4, d=2, seed=42)
    inits = generate_initializations(X, K=4, strategy='random', n_inits=3, random_state=42)

    assert len(inits) == 3, "Should return 3 initializations"

    # Check that different initializations are different
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(inits[i]['mu'], inits[j]['mu']), \
                "Initializations should be different"

    print("  PASSED")


def test_init_collapsed():
    """Test collapsed initialization."""
    X, z, _ = generate_gmm_data(n=200, K=4, d=2, seed=42)
    inits = generate_initializations(X, K=4, strategy='collapsed', n_inits=1, random_state=42)

    theta_init = inits[0]
    centroid = X.mean(axis=0)

    # All means should be at centroid
    for k in range(4):
        assert np.allclose(theta_init['mu'][k], centroid), \
            f"Mean {k} not at centroid"

    print("  PASSED")


def test_init_far_away():
    """Test far-away initialization."""
    X, z, _ = generate_gmm_data(n=200, K=4, d=2, seed=42)
    inits = generate_initializations(X, K=4, strategy='far_away', n_inits=1, random_state=42)

    theta_init = inits[0]
    centroid = X.mean(axis=0)

    # All means should be at distance ~10 from centroid
    for k in range(4):
        dist = np.linalg.norm(theta_init['mu'][k] - centroid)
        assert np.abs(dist - 10.0) < 1e-10, \
            f"Mean {k} distance: {dist}, expected ~10"

    print("  PASSED")


def test_multiple_inits():
    """Test generating multiple initializations."""
    X, z, _ = generate_gmm_data(n=200, K=4, d=2, seed=42)

    for strategy in ['kmeans', 'random', 'collapsed', 'far_away']:
        inits = generate_initializations(X, K=4, strategy=strategy, n_inits=5, random_state=42)
        assert len(inits) == 5, f"Strategy {strategy}: expected 5 inits"

    print("  PASSED")


def test_visualization():
    """Test visualization functions don't error."""
    X, z, theta_true = generate_gmm_data(n=100, K=3, d=2, seed=42)

    # Test plot_gmm_data
    try:
        ax = plot_gmm_data(X, z, theta_true, title="Test")
        plt.close('all')
    except Exception as e:
        raise AssertionError(f"plot_gmm_data failed: {e}")

    # Test plot_gmm_with_ellipses
    try:
        ax = plot_gmm_with_ellipses(X, theta_true, z, title="Test")
        plt.close('all')
    except Exception as e:
        raise AssertionError(f"plot_gmm_with_ellipses failed: {e}")

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running generate_gmm.py unit tests")
    print("=" * 60)

    print("Testing generate_gmm_data...")
    test_generate_gmm_data()

    print("Testing test_1_high_d preset...")
    test_test_1_high_d()

    print("Testing test_4_simple preset...")
    test_test_4_simple()

    print("Testing K-means initialization...")
    test_init_kmeans()

    print("Testing random initialization...")
    test_init_random()

    print("Testing collapsed initialization...")
    test_init_collapsed()

    print("Testing far-away initialization...")
    test_init_far_away()

    print("Testing multiple initializations...")
    test_multiple_inits()

    print("Testing visualization functions...")
    test_visualization()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
