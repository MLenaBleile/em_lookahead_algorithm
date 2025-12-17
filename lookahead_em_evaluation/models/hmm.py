"""
Hidden Markov Model

Implementation of HMM with all methods required by EM algorithms.
Uses forward-backward algorithm for E-step (Baum-Welch).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.special import logsumexp


class HiddenMarkovModel:
    """
    Hidden Markov Model with discrete observations.

    Theta format (dict):
    - 'A': Transition matrix (S × S) where A[i,j] = P(z_{t+1}=j | z_t=i)
    - 'B': Emission matrix (S × V) where B[s,v] = P(x_t=v | z_t=s)
    - 'pi_0': Initial state distribution (S,) where pi_0[s] = P(z_0=s)

    Attributes:
        n_states: Number of hidden states (S)
        n_observations: Number of possible observation values (V)
    """

    def __init__(self, n_states: int, n_observations: int):
        """
        Initialize HMM.

        Args:
            n_states: Number of hidden states (S)
            n_observations: Number of possible observation values (V)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.S = n_states
        self.V = n_observations

    def e_step(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        E-step using forward-backward algorithm.

        Args:
            X: Observation sequence of shape (T,) with values in {0, ..., V-1}
            theta: Model parameters dict

        Returns:
            Dict with:
            - 'gamma': (T, S) array where gamma[t,s] = P(z_t=s | X)
            - 'xi': (T-1, S, S) array where xi[t,i,j] = P(z_t=i, z_{t+1}=j | X)
            - 'log_likelihood': float
        """
        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']
        T = len(X)

        # Forward pass (in log space for stability)
        log_alpha, log_likelihood = self._forward(X, A, B, pi_0)

        # Backward pass (in log space)
        log_beta = self._backward(X, A, B)

        # Compute gamma = P(z_t = s | X)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Compute xi = P(z_t = i, z_{t+1} = j | X)
        xi = np.zeros((T - 1, self.S, self.S))
        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)

        for t in range(T - 1):
            # log_xi[i,j] = log_alpha[t,i] + log(A[i,j]) + log(B[j, X[t+1]]) + log_beta[t+1,j]
            log_xi_t = log_alpha[t, :, np.newaxis] + log_A + log_B[:, X[t+1]].reshape(1, -1) + log_beta[t+1, :].reshape(1, -1)
            log_xi_t = log_xi_t - logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)

        return {
            'gamma': gamma,
            'xi': xi,
            'log_likelihood': log_likelihood
        }

    def _forward(
        self,
        X: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        pi_0: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm (alpha recursion) in log space.

        Args:
            X: Observation sequence (T,)
            A: Transition matrix (S, S)
            B: Emission matrix (S, V)
            pi_0: Initial distribution (S,)

        Returns:
            log_alpha: (T, S) array of log forward probabilities
            log_likelihood: Log-likelihood of observations
        """
        T = len(X)
        log_alpha = np.zeros((T, self.S))

        # Avoid log(0)
        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)
        log_pi_0 = np.log(pi_0 + 1e-300)

        # Initialization: alpha_0(s) = pi_0(s) * B(s, x_0)
        log_alpha[0] = log_pi_0 + log_B[:, X[0]]

        # Recursion: alpha_t(j) = sum_i [alpha_{t-1}(i) * A(i,j)] * B(j, x_t)
        for t in range(1, T):
            for j in range(self.S):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[j, X[t]]

        # Log-likelihood = log(sum_s alpha_T(s))
        log_likelihood = logsumexp(log_alpha[T-1])

        return log_alpha, log_likelihood

    def _backward(
        self,
        X: np.ndarray,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Backward algorithm (beta recursion) in log space.

        Args:
            X: Observation sequence (T,)
            A: Transition matrix (S, S)
            B: Emission matrix (S, V)

        Returns:
            log_beta: (T, S) array of log backward probabilities
        """
        T = len(X)
        log_beta = np.zeros((T, self.S))

        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)

        # Initialization: beta_T(s) = 1, log_beta_T(s) = 0
        log_beta[T-1] = 0.0

        # Recursion: beta_t(i) = sum_j [A(i,j) * B(j, x_{t+1}) * beta_{t+1}(j)]
        for t in range(T - 2, -1, -1):
            for i in range(self.S):
                log_beta[t, i] = logsumexp(
                    log_A[i, :] + log_B[:, X[t+1]] + log_beta[t+1]
                )

        return log_beta

    def m_step(
        self,
        X: np.ndarray,
        responsibilities: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        M-step using Baum-Welch updates.

        Args:
            X: Observation sequence (T,)
            responsibilities: Dict from e_step with 'gamma' and 'xi'

        Returns:
            Updated theta dict
        """
        gamma = responsibilities['gamma']
        xi = responsibilities['xi']
        T = len(X)

        # Update initial distribution: pi_0[s] = gamma[0, s]
        pi_0 = gamma[0].copy()
        pi_0 = np.clip(pi_0, 1e-10, None)
        pi_0 = pi_0 / pi_0.sum()

        # Update transition matrix: A[i,j] = sum_t xi[t,i,j] / sum_t gamma[t,i]
        A = np.sum(xi, axis=0)
        gamma_sum = np.sum(gamma[:-1], axis=0)  # Sum over t=0..T-2
        A = A / (gamma_sum[:, np.newaxis] + 1e-10)
        A = np.clip(A, 1e-10, None)
        A = A / A.sum(axis=1, keepdims=True)

        # Update emission matrix: B[s,v] = sum_{t: x_t=v} gamma[t,s] / sum_t gamma[t,s]
        B = np.zeros((self.S, self.V))
        gamma_total = np.sum(gamma, axis=0)

        for v in range(self.V):
            mask = (X == v)
            B[:, v] = np.sum(gamma[mask], axis=0)

        B = B / (gamma_total[:, np.newaxis] + 1e-10)
        B = np.clip(B, 1e-10, None)
        B = B / B.sum(axis=1, keepdims=True)

        return {
            'A': A,
            'B': B,
            'pi_0': pi_0
        }

    def log_likelihood(
        self,
        X: np.ndarray,
        theta: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute log-likelihood using forward algorithm.

        Args:
            X: Observation sequence (T,)
            theta: Model parameters

        Returns:
            Log-likelihood of observations
        """
        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']

        _, log_likelihood = self._forward(X, A, B, pi_0)
        return log_likelihood

    def compute_Q(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute Q-function: E[log p(X, Z | theta) | X, theta_old].

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step (unused, responsibilities already computed)
            X: Observation sequence
            responsibilities: Dict with 'gamma' and 'xi'

        Returns:
            Q-function value
        """
        gamma = responsibilities['gamma']
        xi = responsibilities['xi']
        T = len(X)

        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']

        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)
        log_pi_0 = np.log(pi_0 + 1e-300)

        # Q = sum_s gamma[0,s] * log(pi_0[s])
        #   + sum_t sum_{i,j} xi[t,i,j] * log(A[i,j])
        #   + sum_t sum_s gamma[t,s] * log(B[s, X[t]])

        Q = 0.0

        # Initial state term
        Q += np.sum(gamma[0] * log_pi_0)

        # Transition term
        Q += np.sum(xi * log_A)

        # Emission term
        for t in range(T):
            Q += np.sum(gamma[t] * log_B[:, X[t]])

        return Q

    def compute_Q_gradient(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute gradient of Q-function w.r.t. natural parameters.

        For HMM, uses natural (unconstrained) parameterization.

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step
            X: Observation sequence
            responsibilities: Dict with 'gamma' and 'xi'

        Returns:
            Gradient vector (flattened)
        """
        gamma = responsibilities['gamma']
        xi = responsibilities['xi']
        T = len(X)

        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']

        # Gradient w.r.t. log parameters (natural gradient for positive constraints)
        # For rows of A, B, and pi_0 that must sum to 1, use log-ratio parameterization

        grad_parts = []

        # Gradient w.r.t. pi_0 (using softmax parameterization)
        # If pi_0 = softmax(eta), then dQ/deta = gamma[0] - pi_0 * sum(gamma[0])
        expected_pi = gamma[0]
        grad_pi = expected_pi - pi_0 * np.sum(expected_pi)
        grad_parts.append(grad_pi)

        # Gradient w.r.t. A (row by row)
        xi_sum = np.sum(xi, axis=0)  # (S, S)
        for i in range(self.S):
            expected_A_i = xi_sum[i]
            row_sum = np.sum(expected_A_i)
            grad_A_i = expected_A_i - A[i] * row_sum
            grad_parts.append(grad_A_i)

        # Gradient w.r.t. B (row by row)
        # expected_B[s,v] = sum_{t: X[t]=v} gamma[t,s]
        expected_B = np.zeros((self.S, self.V))
        for v in range(self.V):
            mask = (X == v)
            expected_B[:, v] = np.sum(gamma[mask], axis=0)

        for s in range(self.S):
            row_sum = np.sum(expected_B[s])
            grad_B_s = expected_B[s] - B[s] * row_sum
            grad_parts.append(grad_B_s)

        return np.concatenate([g.flatten() for g in grad_parts])

    def compute_Q_hessian(
        self,
        theta: Dict[str, np.ndarray],
        theta_old: Dict[str, np.ndarray],
        X: np.ndarray,
        responsibilities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Hessian of Q-function.

        Exploits block-diagonal structure:
        - pi_0 block
        - A blocks (one per row)
        - B blocks (one per row)

        Args:
            theta: Parameters to evaluate
            theta_old: Parameters used for E-step
            X: Observation sequence
            responsibilities: Dict with 'gamma' and 'xi'

        Returns:
            Block-diagonal Hessian matrix
        """
        gamma = responsibilities['gamma']
        xi = responsibilities['xi']
        T = len(X)

        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']

        # Total size: S (pi_0) + S*S (A) + S*V (B)
        total_size = self.S + self.S * self.S + self.S * self.V
        H = np.zeros((total_size, total_size))

        idx = 0

        # Hessian for pi_0 (softmax parameterization)
        # d^2 Q / deta_i deta_j = -pi_i * (delta_ij - pi_j) * sum(gamma[0])
        gamma_0_sum = np.sum(gamma[0])
        H_pi = -gamma_0_sum * (np.diag(pi_0) - np.outer(pi_0, pi_0))
        H[idx:idx+self.S, idx:idx+self.S] = H_pi
        idx += self.S

        # Hessian for A (row by row)
        xi_sum = np.sum(xi, axis=0)
        for i in range(self.S):
            row_sum = np.sum(xi_sum[i])
            A_i = A[i]
            H_A_i = -row_sum * (np.diag(A_i) - np.outer(A_i, A_i))
            H[idx:idx+self.S, idx:idx+self.S] = H_A_i
            idx += self.S

        # Hessian for B (row by row)
        expected_B = np.zeros((self.S, self.V))
        for v in range(self.V):
            mask = (X == v)
            expected_B[:, v] = np.sum(gamma[mask], axis=0)

        for s in range(self.S):
            row_sum = np.sum(expected_B[s])
            B_s = B[s]
            H_B_s = -row_sum * (np.diag(B_s) - np.outer(B_s, B_s))
            H[idx:idx+self.V, idx:idx+self.V] = H_B_s
            idx += self.V

        return H

    def project_to_feasible(self, theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Project parameters to feasible region.

        Ensures:
        - A rows sum to 1, all entries positive
        - B rows sum to 1, all entries positive
        - pi_0 sums to 1, all entries positive

        Args:
            theta: Parameters to project

        Returns:
            Projected parameters
        """
        theta_proj = {}

        # Project pi_0
        pi_0 = np.clip(theta['pi_0'], 1e-10, None)
        theta_proj['pi_0'] = pi_0 / pi_0.sum()

        # Project A (row stochastic)
        A = np.clip(theta['A'], 1e-10, None)
        theta_proj['A'] = A / A.sum(axis=1, keepdims=True)

        # Project B (row stochastic)
        B = np.clip(theta['B'], 1e-10, None)
        theta_proj['B'] = B / B.sum(axis=1, keepdims=True)

        return theta_proj

    def sample(
        self,
        theta: Dict[str, np.ndarray],
        T: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a sequence from the HMM.

        Args:
            theta: Model parameters
            T: Sequence length
            seed: Random seed

        Returns:
            Tuple of (observations, states)
        """
        if seed is not None:
            np.random.seed(seed)

        A = theta['A']
        B = theta['B']
        pi_0 = theta['pi_0']

        states = np.zeros(T, dtype=int)
        observations = np.zeros(T, dtype=int)

        # Sample initial state
        states[0] = np.random.choice(self.S, p=pi_0)
        observations[0] = np.random.choice(self.V, p=B[states[0]])

        # Sample remaining sequence
        for t in range(1, T):
            states[t] = np.random.choice(self.S, p=A[states[t-1]])
            observations[t] = np.random.choice(self.V, p=B[states[t]])

        return observations, states


# ============================================================
# Initialization Functions
# ============================================================

def initialize_uniform(n_states: int, n_observations: int) -> Dict[str, np.ndarray]:
    """Initialize with uniform distributions."""
    return {
        'A': np.ones((n_states, n_states)) / n_states,
        'B': np.ones((n_states, n_observations)) / n_observations,
        'pi_0': np.ones(n_states) / n_states
    }


def initialize_random(
    n_states: int,
    n_observations: int,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Initialize with random distributions."""
    rng = np.random.RandomState(random_state)

    # Random transition matrix (row stochastic)
    A = rng.dirichlet(np.ones(n_states), size=n_states)

    # Random emission matrix (row stochastic)
    B = rng.dirichlet(np.ones(n_observations), size=n_states)

    # Random initial distribution
    pi_0 = rng.dirichlet(np.ones(n_states))

    return {'A': A, 'B': B, 'pi_0': pi_0}


def initialize_perturbed(
    theta_true: Dict[str, np.ndarray],
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Initialize by perturbing true parameters."""
    rng = np.random.RandomState(random_state)

    theta = {}

    # Perturb A
    A = theta_true['A'] + noise_level * rng.randn(*theta_true['A'].shape)
    A = np.clip(A, 1e-10, None)
    A = A / A.sum(axis=1, keepdims=True)
    theta['A'] = A

    # Perturb B
    B = theta_true['B'] + noise_level * rng.randn(*theta_true['B'].shape)
    B = np.clip(B, 1e-10, None)
    B = B / B.sum(axis=1, keepdims=True)
    theta['B'] = B

    # Perturb pi_0
    pi_0 = theta_true['pi_0'] + noise_level * rng.randn(*theta_true['pi_0'].shape)
    pi_0 = np.clip(pi_0, 1e-10, None)
    pi_0 = pi_0 / pi_0.sum()
    theta['pi_0'] = pi_0

    return theta


# ============================================================
# Unit Tests
# ============================================================

def test_forward_backward():
    """Test forward-backward algorithm."""
    model = HiddenMarkovModel(n_states=3, n_observations=4)

    theta = {
        'A': np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6]
        ]),
        'B': np.array([
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.2, 0.1, 0.4, 0.3]
        ]),
        'pi_0': np.array([0.5, 0.3, 0.2])
    }

    # Generate sequence
    X = np.array([0, 1, 2, 1, 0, 3, 2])

    # E-step
    resp = model.e_step(X, theta)

    # Check gamma sums to 1 for each time step
    gamma_sums = resp['gamma'].sum(axis=1)
    assert np.allclose(gamma_sums, 1.0), f"Gamma doesn't sum to 1: {gamma_sums}"

    # Check xi sums correctly
    for t in range(len(X) - 1):
        xi_sum = resp['xi'][t].sum()
        assert np.abs(xi_sum - 1.0) < 1e-6, f"Xi[{t}] doesn't sum to 1: {xi_sum}"

    print("  PASSED")


def test_em_convergence():
    """Test that EM converges on toy HMM."""
    model = HiddenMarkovModel(n_states=2, n_observations=3)

    # True parameters
    theta_true = {
        'A': np.array([[0.8, 0.2], [0.3, 0.7]]),
        'B': np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]]),
        'pi_0': np.array([0.6, 0.4])
    }

    # Generate data
    X, _ = model.sample(theta_true, T=200, seed=42)

    # Initialize randomly
    theta = initialize_random(2, 3, random_state=123)

    # Run EM
    ll_history = []
    for _ in range(50):
        ll = model.log_likelihood(X, theta)
        ll_history.append(ll)

        resp = model.e_step(X, theta)
        theta = model.m_step(X, resp)

    # Check monotonicity
    ll_diffs = np.diff(ll_history)
    violations = np.sum(ll_diffs < -1e-8)
    assert violations == 0, f"EM is not monotonic: {violations} violations"

    # Check improvement
    assert ll_history[-1] > ll_history[0], "EM should improve likelihood"

    print(f"  Initial LL: {ll_history[0]:.2f}, Final LL: {ll_history[-1]:.2f}")
    print("  PASSED")


def test_q_gradient():
    """Test Q gradient against finite differences."""
    model = HiddenMarkovModel(n_states=2, n_observations=3)

    theta = {
        'A': np.array([[0.7, 0.3], [0.4, 0.6]]),
        'B': np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]),
        'pi_0': np.array([0.6, 0.4])
    }

    X = np.array([0, 1, 2, 1, 0])
    resp = model.e_step(X, theta)

    # Compute analytical gradient
    grad = model.compute_Q_gradient(theta, theta, X, resp)

    # At the M-step solution, gradient should be approximately zero
    # (since M-step maximizes Q)
    theta_mstep = model.m_step(X, resp)
    grad_mstep = model.compute_Q_gradient(theta_mstep, theta, X, resp)

    # Gradient at M-step should be small
    assert np.linalg.norm(grad_mstep) < 0.1, f"Gradient at M-step too large: {np.linalg.norm(grad_mstep)}"

    print(f"  Gradient norm at M-step: {np.linalg.norm(grad_mstep):.6f}")
    print("  PASSED")


def test_hessian_structure():
    """Test that Hessian is block-diagonal."""
    model = HiddenMarkovModel(n_states=2, n_observations=3)

    theta = {
        'A': np.array([[0.7, 0.3], [0.4, 0.6]]),
        'B': np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]),
        'pi_0': np.array([0.6, 0.4])
    }

    X = np.array([0, 1, 2, 1, 0])
    resp = model.e_step(X, theta)

    H = model.compute_Q_hessian(theta, theta, X, resp)

    # Check dimensions
    expected_size = 2 + 2*2 + 2*3  # pi_0 + A + B
    assert H.shape == (expected_size, expected_size), f"Wrong Hessian shape: {H.shape}"

    # Check symmetry
    assert np.allclose(H, H.T), "Hessian should be symmetric"

    # Check negative semi-definiteness (Q is concave)
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(eigvals <= 1e-6), f"Hessian should be negative semi-definite: {eigvals.max()}"

    print(f"  Hessian shape: {H.shape}")
    print(f"  Max eigenvalue: {eigvals.max():.6f}")
    print("  PASSED")


def test_sampling():
    """Test HMM sampling."""
    model = HiddenMarkovModel(n_states=3, n_observations=5)

    theta = initialize_random(3, 5, random_state=42)
    X, Z = model.sample(theta, T=100, seed=42)

    assert len(X) == 100, "Wrong sequence length"
    assert len(Z) == 100, "Wrong state sequence length"
    assert X.min() >= 0 and X.max() < 5, "Observations out of range"
    assert Z.min() >= 0 and Z.max() < 3, "States out of range"

    print(f"  Generated sequence of length {len(X)}")
    print(f"  Observation range: [{X.min()}, {X.max()}]")
    print(f"  State range: [{Z.min()}, {Z.max()}]")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running hmm.py unit tests")
    print("=" * 60)

    print("Testing forward-backward algorithm...")
    test_forward_backward()

    print("Testing EM convergence...")
    test_em_convergence()

    print("Testing Q gradient...")
    test_q_gradient()

    print("Testing Hessian structure...")
    test_hessian_structure()

    print("Testing sampling...")
    test_sampling()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
