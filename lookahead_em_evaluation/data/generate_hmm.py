"""
HMM Data Generation

Generate HMM data for testing EM algorithms.
Includes various transition structures and initialization strategies.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import matplotlib.pyplot as plt


def generate_hmm_data(
    n_steps: int,
    n_states: int,
    n_obs: int,
    structure: str = 'nearly_diagonal',
    seed: Optional[int] = None,
    self_transition_prob: float = 0.7,
    obs_concentration: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate HMM observation sequence with specified structure.

    Args:
        n_steps: Number of time steps (T)
        n_states: Number of hidden states (S)
        n_obs: Number of possible observations (V)
        structure: Transition structure - 'nearly_diagonal', 'cyclic', 'random', 'block'
        seed: Random seed
        self_transition_prob: Probability of staying in same state (for nearly_diagonal)
        obs_concentration: Dirichlet concentration for emission matrix

    Returns:
        Tuple of (observations, states, theta_true) where:
        - observations: (T,) array of observation indices
        - states: (T,) array of true state indices
        - theta_true: dict with 'A', 'B', 'pi_0'
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate transition matrix based on structure
    A = _generate_transition_matrix(n_states, structure, self_transition_prob, seed)

    # Generate emission matrix
    B = _generate_emission_matrix(n_states, n_obs, obs_concentration, seed)

    # Generate initial distribution
    pi_0 = _generate_initial_distribution(n_states, seed)

    theta_true = {'A': A, 'B': B, 'pi_0': pi_0}

    # Sample sequence
    observations, states = _sample_hmm(theta_true, n_steps)

    return observations, states, theta_true


def _generate_transition_matrix(
    n_states: int,
    structure: str,
    self_transition_prob: float,
    seed: Optional[int]
) -> np.ndarray:
    """Generate transition matrix with specified structure."""
    rng = np.random.RandomState(seed)

    if structure == 'nearly_diagonal':
        # High self-transition probability, uniform elsewhere
        off_diag_prob = (1 - self_transition_prob) / (n_states - 1)
        A = np.full((n_states, n_states), off_diag_prob)
        np.fill_diagonal(A, self_transition_prob)

    elif structure == 'cyclic':
        # Prefer transitioning to next state (with wrap-around)
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            A[i, i] = 0.3  # Some self-transition
            A[i, (i + 1) % n_states] = 0.5  # Forward transition
            # Small probability for other states
            remaining = 1.0 - 0.3 - 0.5
            for j in range(n_states):
                if j != i and j != (i + 1) % n_states:
                    A[i, j] = remaining / (n_states - 2) if n_states > 2 else 0

    elif structure == 'block':
        # States grouped into blocks with high intra-block transition
        n_blocks = max(2, n_states // 5)
        block_size = n_states // n_blocks
        A = np.zeros((n_states, n_states))

        for i in range(n_states):
            block_i = i // block_size
            for j in range(n_states):
                block_j = j // block_size
                if block_i == block_j:
                    A[i, j] = 0.8 / block_size
                else:
                    A[i, j] = 0.2 / (n_states - block_size)

        # Normalize rows
        A = A / A.sum(axis=1, keepdims=True)

    elif structure == 'random':
        # Random row-stochastic matrix
        A = rng.dirichlet(np.ones(n_states), size=n_states)

    else:
        raise ValueError(f"Unknown structure: {structure}")

    return A


def _generate_emission_matrix(
    n_states: int,
    n_obs: int,
    concentration: float,
    seed: Optional[int]
) -> np.ndarray:
    """Generate emission matrix with concentrated distributions."""
    rng = np.random.RandomState(seed)

    # Each state has a few preferred observations
    B = np.zeros((n_states, n_obs))

    # Determine how many observations each state prefers
    n_preferred = max(2, n_obs // n_states)

    for s in range(n_states):
        # Each state prefers a subset of observations
        preferred_start = (s * n_preferred) % n_obs
        preferred_obs = [(preferred_start + i) % n_obs for i in range(n_preferred)]

        # Higher concentration on preferred observations
        alpha = np.ones(n_obs) * 0.1
        for obs in preferred_obs:
            alpha[obs] = concentration

        B[s] = rng.dirichlet(alpha)

    return B


def _generate_initial_distribution(n_states: int, seed: Optional[int]) -> np.ndarray:
    """Generate initial state distribution."""
    rng = np.random.RandomState(seed)
    return rng.dirichlet(np.ones(n_states))


def _sample_hmm(
    theta: Dict[str, np.ndarray],
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an observation sequence from HMM."""
    A = theta['A']
    B = theta['B']
    pi_0 = theta['pi_0']
    n_states = A.shape[0]
    n_obs = B.shape[1]

    states = np.zeros(n_steps, dtype=int)
    observations = np.zeros(n_steps, dtype=int)

    # Sample initial state
    states[0] = np.random.choice(n_states, p=pi_0)
    observations[0] = np.random.choice(n_obs, p=B[states[0]])

    # Sample sequence
    for t in range(1, n_steps):
        states[t] = np.random.choice(n_states, p=A[states[t-1]])
        observations[t] = np.random.choice(n_obs, p=B[states[t]])

    return observations, states


# ============================================================
# Test Presets
# ============================================================

def test_2_hmm(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate data for Test 2: HMM from EMPIRICAL_TEST_SPECIFICATIONS.md.

    Configuration:
    - S = 20 states
    - V = 50 observations
    - T = 1000 steps
    - Nearly diagonal transition (0.7 self-transition)
    - Each state prefers 2-3 observations

    Args:
        seed: Random seed

    Returns:
        Tuple of (observations, states, theta_true)
    """
    return generate_hmm_data(
        n_steps=1000,
        n_states=20,
        n_obs=50,
        structure='nearly_diagonal',
        seed=seed,
        self_transition_prob=0.7,
        obs_concentration=3.0
    )


def test_2_hmm_small(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Smaller version of Test 2 for quick testing.

    Configuration:
    - S = 5 states
    - V = 10 observations
    - T = 200 steps
    """
    return generate_hmm_data(
        n_steps=200,
        n_states=5,
        n_obs=10,
        structure='nearly_diagonal',
        seed=seed,
        self_transition_prob=0.7,
        obs_concentration=3.0
    )


# ============================================================
# Initialization Strategies
# ============================================================

def generate_hmm_initializations(
    observations: np.ndarray,
    n_states: int,
    n_obs: int,
    strategy: str,
    n_inits: int,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Generate multiple initializations for HMM.

    Args:
        observations: Observation sequence (not used for most strategies)
        n_states: Number of states (S)
        n_obs: Number of observations (V)
        strategy: 'uniform', 'random', 'perturbed', 'sparse'
        n_inits: Number of initializations
        random_state: Base random seed

    Returns:
        List of theta_init dicts
    """
    inits = []

    for i in range(n_inits):
        seed = random_state + i

        if strategy == 'uniform':
            theta = {
                'A': np.ones((n_states, n_states)) / n_states,
                'B': np.ones((n_states, n_obs)) / n_obs,
                'pi_0': np.ones(n_states) / n_states
            }

        elif strategy == 'random':
            rng = np.random.RandomState(seed)
            theta = {
                'A': rng.dirichlet(np.ones(n_states), size=n_states),
                'B': rng.dirichlet(np.ones(n_obs), size=n_states),
                'pi_0': rng.dirichlet(np.ones(n_states))
            }

        elif strategy == 'sparse':
            # Initialize with sparse transition matrix
            rng = np.random.RandomState(seed)

            # Sparse A: high self-transition, random elsewhere
            A = np.zeros((n_states, n_states))
            for s in range(n_states):
                A[s, s] = 0.5 + 0.3 * rng.rand()
                remaining = 1 - A[s, s]
                other_probs = rng.rand(n_states - 1)
                other_probs = other_probs / other_probs.sum() * remaining
                idx = 0
                for j in range(n_states):
                    if j != s:
                        A[s, j] = other_probs[idx]
                        idx += 1

            # Random emission
            B = rng.dirichlet(np.ones(n_obs), size=n_states)

            # Random initial
            pi_0 = rng.dirichlet(np.ones(n_states))

            theta = {'A': A, 'B': B, 'pi_0': pi_0}

        elif strategy == 'empirical':
            # Initialize emission matrix from observation frequencies
            rng = np.random.RandomState(seed)

            # Count observation frequencies
            obs_counts = np.bincount(observations, minlength=n_obs)
            obs_freq = obs_counts / obs_counts.sum()

            # Initialize B with perturbed observation frequencies
            B = np.zeros((n_states, n_obs))
            for s in range(n_states):
                noise = rng.rand(n_obs) * 0.1
                B[s] = obs_freq + noise
                B[s] = B[s] / B[s].sum()

            # Random A and pi_0
            A = rng.dirichlet(np.ones(n_states), size=n_states)
            pi_0 = rng.dirichlet(np.ones(n_states))

            theta = {'A': A, 'B': B, 'pi_0': pi_0}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        inits.append(theta)

    return inits


# ============================================================
# Visualization
# ============================================================

def plot_transition_matrix(A: np.ndarray, title: str = "Transition Matrix", save_path: str = None):
    """Plot heatmap of transition matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(A, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Probability')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_emission_matrix(B: np.ndarray, title: str = "Emission Matrix", save_path: str = None):
    """Plot heatmap of emission matrix."""
    plt.figure(figsize=(12, 6))
    plt.imshow(B, cmap='Greens', aspect='auto', vmin=0, vmax=B.max())
    plt.colorbar(label='Probability')
    plt.xlabel('Observation')
    plt.ylabel('State')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sequence(
    observations: np.ndarray,
    states: np.ndarray = None,
    n_show: int = 100,
    title: str = "HMM Sequence",
    save_path: str = None
):
    """Plot first n observations (and optionally states)."""
    n_show = min(n_show, len(observations))

    if states is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        ax1.step(range(n_show), states[:n_show], where='mid', color='blue')
        ax1.set_ylabel('Hidden State')
        ax1.set_title(title)

        ax2.step(range(n_show), observations[:n_show], where='mid', color='green')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Observation')
    else:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.step(range(n_show), observations[:n_show], where='mid', color='green')
        ax.set_xlabel('Time')
        ax.set_ylabel('Observation')
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hmm_summary(theta: Dict[str, np.ndarray], save_path: str = None):
    """Plot summary of HMM parameters."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Transition matrix
    axes[0].imshow(theta['A'], cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Transition Matrix A')
    axes[0].set_xlabel('To State')
    axes[0].set_ylabel('From State')

    # Emission matrix
    im = axes[1].imshow(theta['B'], cmap='Greens', aspect='auto')
    axes[1].set_title('Emission Matrix B')
    axes[1].set_xlabel('Observation')
    axes[1].set_ylabel('State')

    # Initial distribution
    axes[2].bar(range(len(theta['pi_0'])), theta['pi_0'], color='purple', alpha=0.7)
    axes[2].set_title('Initial Distribution π₀')
    axes[2].set_xlabel('State')
    axes[2].set_ylabel('Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================
# Unit Tests
# ============================================================

def test_generate_hmm_data():
    """Test HMM data generation."""
    X, Z, theta = generate_hmm_data(
        n_steps=100,
        n_states=5,
        n_obs=10,
        structure='nearly_diagonal',
        seed=42
    )

    assert len(X) == 100, f"Wrong sequence length: {len(X)}"
    assert len(Z) == 100, f"Wrong state length: {len(Z)}"
    assert X.min() >= 0 and X.max() < 10, "Observations out of range"
    assert Z.min() >= 0 and Z.max() < 5, "States out of range"

    # Check theta structure
    assert theta['A'].shape == (5, 5), f"Wrong A shape: {theta['A'].shape}"
    assert theta['B'].shape == (5, 10), f"Wrong B shape: {theta['B'].shape}"
    assert len(theta['pi_0']) == 5, f"Wrong pi_0 length: {len(theta['pi_0'])}"

    # Check row-stochastic
    assert np.allclose(theta['A'].sum(axis=1), 1), "A not row-stochastic"
    assert np.allclose(theta['B'].sum(axis=1), 1), "B not row-stochastic"
    assert np.allclose(theta['pi_0'].sum(), 1), "pi_0 not normalized"

    print("  PASSED")


def test_transition_structures():
    """Test different transition matrix structures."""
    for structure in ['nearly_diagonal', 'cyclic', 'block', 'random']:
        A = _generate_transition_matrix(10, structure, 0.7, 42)

        assert A.shape == (10, 10), f"Wrong shape for {structure}"
        assert np.allclose(A.sum(axis=1), 1), f"{structure} not row-stochastic"
        assert (A >= 0).all(), f"{structure} has negative entries"

        # Check specific properties
        if structure == 'nearly_diagonal':
            # Diagonal should be high
            assert np.diag(A).mean() > 0.5, "Nearly diagonal should have high self-transition"

    print("  PASSED")


def test_test_2_preset():
    """Test Test 2 preset."""
    X, Z, theta = test_2_hmm(seed=42)

    assert len(X) == 1000, f"Wrong sequence length: {len(X)}"
    assert theta['A'].shape == (20, 20), f"Wrong A shape"
    assert theta['B'].shape == (20, 50), f"Wrong B shape"

    # Check nearly diagonal structure
    diag_mean = np.diag(theta['A']).mean()
    assert diag_mean >= 0.6, f"Not nearly diagonal: diag mean = {diag_mean}"

    print(f"  Sequence length: {len(X)}")
    print(f"  Diagonal mean: {diag_mean:.3f}")
    print("  PASSED")


def test_initialization_strategies():
    """Test initialization strategies."""
    X, _, _ = generate_hmm_data(100, 5, 10, seed=42)

    for strategy in ['uniform', 'random', 'sparse', 'empirical']:
        inits = generate_hmm_initializations(X, 5, 10, strategy, 3, random_state=42)

        assert len(inits) == 3, f"Wrong number of inits for {strategy}"

        for theta in inits:
            assert 'A' in theta and 'B' in theta and 'pi_0' in theta
            assert np.allclose(theta['A'].sum(axis=1), 1), f"{strategy}: A not row-stochastic"
            assert np.allclose(theta['B'].sum(axis=1), 1), f"{strategy}: B not row-stochastic"
            assert np.allclose(theta['pi_0'].sum(), 1), f"{strategy}: pi_0 not normalized"

    print("  PASSED")


def test_reproducibility():
    """Test that data generation is reproducible."""
    X1, Z1, theta1 = generate_hmm_data(100, 5, 10, seed=42)
    X2, Z2, theta2 = generate_hmm_data(100, 5, 10, seed=42)

    assert np.array_equal(X1, X2), "Observations not reproducible"
    assert np.array_equal(Z1, Z2), "States not reproducible"
    assert np.allclose(theta1['A'], theta2['A']), "A not reproducible"

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running generate_hmm.py unit tests")
    print("=" * 60)

    print("Testing HMM data generation...")
    test_generate_hmm_data()

    print("Testing transition structures...")
    test_transition_structures()

    print("Testing Test 2 preset...")
    test_test_2_preset()

    print("Testing initialization strategies...")
    test_initialization_strategies()

    print("Testing reproducibility...")
    test_reproducibility()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
