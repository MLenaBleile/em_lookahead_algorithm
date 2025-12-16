"""
Unit Tests for Algorithm Modules

This module provides pytest-compatible tests for all algorithm modules.
Run with: pytest tests/test_algorithms.py -v
"""

import pytest
import numpy as np
import time


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_gmm_data():
    """Generate simple 2-component GMM data."""
    np.random.seed(42)
    n = 200
    X1 = np.random.randn(n // 2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n // 2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])

    theta_init = {
        'mu': np.array([[0.0, 0.0], [1.0, 0.0]]),
        'sigma': np.ones((2, 2)),
        'pi': np.array([0.5, 0.5])
    }

    return X, theta_init


@pytest.fixture
def simple_gmm_model():
    """Create simple GMM model."""
    from algorithms.standard_em import SimpleGMMInterface
    return SimpleGMMInterface(n_components=2, n_features=2)


@pytest.fixture
def gmm_with_hessian():
    """Create GMM model with gradient/Hessian support."""
    from algorithms.lookahead_em import SimpleGMMWithHessian
    return SimpleGMMWithHessian(n_components=2, n_features=2)


# =============================================================================
# Standard EM Tests
# =============================================================================

class TestStandardEM:
    """Tests for StandardEM algorithm."""

    def test_basic_convergence(self, simple_gmm_model, simple_gmm_data):
        """Test that EM converges on simple data."""
        from algorithms.standard_em import StandardEM

        X, theta_init = simple_gmm_data
        em = StandardEM(simple_gmm_model, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=100, tol=1e-6)

        assert diag['stopping_reason'] in ['tolerance', 'stagnation', 'max_iter']
        assert len(diag['likelihood_history']) > 1

    def test_monotonicity(self, simple_gmm_model, simple_gmm_data):
        """Test that likelihood is monotonically increasing."""
        from algorithms.standard_em import StandardEM

        X, theta_init = simple_gmm_data
        em = StandardEM(simple_gmm_model, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=50)

        ll_history = diag['likelihood_history']
        for i in range(1, len(ll_history)):
            assert ll_history[i] >= ll_history[i-1] - 1e-8

    def test_diagnostics_complete(self, simple_gmm_model, simple_gmm_data):
        """Test that all required diagnostics are present."""
        from algorithms.standard_em import StandardEM

        X, theta_init = simple_gmm_data
        em = StandardEM(simple_gmm_model, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=10)

        required = ['converged', 'iterations', 'stopping_reason',
                   'likelihood_history', 'time_seconds', 'peak_memory_mb']
        for key in required:
            assert key in diag


# =============================================================================
# Gamma Schedule Tests
# =============================================================================

class TestGammaSchedules:
    """Tests for gamma schedule classes."""

    def test_fixed_gamma(self):
        """Test FixedGamma returns constant."""
        from algorithms.gamma_schedule import FixedGamma

        schedule = FixedGamma(gamma=0.5)
        assert schedule.get_gamma(0) == 0.5
        assert schedule.get_gamma(100) == 0.5

    def test_adaptive_gamma(self):
        """Test AdaptiveGamma increases linearly."""
        from algorithms.gamma_schedule import AdaptiveGamma

        schedule = AdaptiveGamma(gamma_init=0.1, gamma_max=0.9, gamma_step=0.1)
        assert schedule.get_gamma(0) == 0.1
        assert abs(schedule.get_gamma(4) - 0.5) < 1e-10
        assert schedule.get_gamma(10) == 0.9  # Should plateau

    def test_exponential_gamma(self):
        """Test ExponentialGamma approaches max."""
        from algorithms.gamma_schedule import ExponentialGamma

        schedule = ExponentialGamma(gamma_init=0.1, gamma_max=0.9, rate=0.1)
        assert abs(schedule.get_gamma(0) - 0.1) < 1e-10
        assert schedule.get_gamma(100) > 0.89  # Should be close to max

    def test_create_schedule(self):
        """Test schedule factory function."""
        from algorithms.gamma_schedule import create_schedule, FixedGamma, AdaptiveGamma

        schedule = create_schedule(0.5)
        assert isinstance(schedule, FixedGamma)

        schedule = create_schedule('adaptive')
        assert isinstance(schedule, AdaptiveGamma)


# =============================================================================
# Value Estimation Tests
# =============================================================================

class TestValueEstimation:
    """Tests for V(θ) estimation."""

    def test_basic_estimation(self):
        """Test basic V estimation."""
        from algorithms.value_estimation import (
            estimate_V_second_order, QuadraticTestModel
        )

        model = QuadraticTestModel(n_params=2)
        X = np.random.randn(10, 2)
        theta = {'params': np.array([1.0, 1.0])}

        V, diag = estimate_V_second_order(model, X, theta)

        assert diag['Q_self'] is not None
        assert V >= diag['Q_self'] - 1e-10  # V should be >= Q

    def test_at_optimum(self):
        """Test V at optimum."""
        from algorithms.value_estimation import (
            estimate_V_second_order, QuadraticTestModel
        )

        model = QuadraticTestModel(n_params=2)
        X = np.random.randn(10, 2)
        theta = {'params': np.array([0.0, 0.0])}  # At optimum

        V, diag = estimate_V_second_order(model, X, theta)

        # At optimum, V should equal Q_self
        assert abs(V - diag['Q_self']) < 1e-10


# =============================================================================
# Lookahead EM Tests
# =============================================================================

class TestLookaheadEM:
    """Tests for LookaheadEM algorithm."""

    def test_basic_functionality(self, gmm_with_hessian, simple_gmm_data):
        """Test basic lookahead EM."""
        from algorithms.lookahead_em import LookaheadEM

        X, theta_init = simple_gmm_data
        em = LookaheadEM(gmm_with_hessian, gamma=0.5, n_candidates=10, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=30)

        assert len(diag['likelihood_history']) > 1
        assert diag['n_candidates_evaluated'] > 0

    def test_gamma_zero_matches_standard(self, gmm_with_hessian, simple_gmm_data):
        """Test that gamma=0 behaves like standard EM."""
        from algorithms.lookahead_em import LookaheadEM
        from algorithms.standard_em import StandardEM

        X, theta_init = simple_gmm_data

        # Create a simple model for standard EM
        from algorithms.standard_em import SimpleGMMInterface
        std_model = SimpleGMMInterface(n_components=2, n_features=2)

        lookahead = LookaheadEM(gmm_with_hessian, gamma=0.0, verbose=False)
        standard = StandardEM(std_model, verbose=False)

        theta_la, diag_la = lookahead.fit(X, theta_init.copy(), max_iter=15)
        theta_std, diag_std = standard.fit(X, theta_init.copy(), max_iter=15)

        # Likelihoods should be similar
        ll_diff = abs(diag_la['final_likelihood'] - diag_std['final_likelihood'])
        assert ll_diff < 1.0

    def test_adaptive_gamma(self, gmm_with_hessian, simple_gmm_data):
        """Test adaptive gamma schedule."""
        from algorithms.lookahead_em import LookaheadEM

        X, theta_init = simple_gmm_data
        em = LookaheadEM(gmm_with_hessian, gamma='adaptive', n_candidates=10, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=20)

        # Gamma should increase
        if len(diag['gamma_history']) > 1:
            assert diag['gamma_history'][-1] >= diag['gamma_history'][0]

    def test_lookahead_diagnostics(self, gmm_with_hessian, simple_gmm_data):
        """Test lookahead-specific diagnostics."""
        from algorithms.lookahead_em import LookaheadEM

        X, theta_init = simple_gmm_data
        em = LookaheadEM(gmm_with_hessian, gamma=0.5, n_candidates=5, verbose=False)
        theta, diag = em.fit(X, theta_init, max_iter=10)

        lookahead_keys = ['gamma_history', 'v_value_history', 'n_candidates_evaluated']
        for key in lookahead_keys:
            assert key in diag


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests across modules."""

    def test_full_pipeline(self):
        """Test full EM pipeline."""
        from algorithms.standard_em import StandardEM, SimpleGMMInterface
        from algorithms.lookahead_em import LookaheadEM, SimpleGMMWithHessian
        from utils.metrics import compute_metrics, check_monotonicity

        np.random.seed(42)

        # Generate data
        X = np.vstack([
            np.random.randn(100, 2) + np.array([-3, 0]),
            np.random.randn(100, 2) + np.array([3, 0])
        ])

        theta_init = {
            'mu': np.array([[-1.0, 0.0], [1.0, 0.0]]),
            'sigma': np.ones((2, 2)),
            'pi': np.array([0.5, 0.5])
        }

        # Standard EM
        std_model = SimpleGMMInterface(n_components=2, n_features=2)
        std_em = StandardEM(std_model, verbose=False)
        theta_std, diag_std = std_em.fit(X, theta_init.copy(), max_iter=50)

        # Lookahead EM
        la_model = SimpleGMMWithHessian(n_components=2, n_features=2)
        la_em = LookaheadEM(la_model, gamma='adaptive', n_candidates=10, verbose=False)
        theta_la, diag_la = la_em.fit(X, theta_init.copy(), max_iter=50)

        # Check monotonicity
        is_mono_std, _ = check_monotonicity(diag_std['likelihood_history'])
        is_mono_la, violations = check_monotonicity(diag_la['likelihood_history'], tol=0.1)

        assert is_mono_std, "Standard EM should be monotonic"
        # Lookahead may have small violations due to candidate selection
        assert len(violations) <= 2, "Lookahead should have few violations"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
