"""
Unit Tests for Utility Modules

This module provides pytest-compatible tests for all utility modules.
Run with: pytest tests/test_utils.py -v
"""

import pytest
import numpy as np
import tempfile
import shutil
import time
import os


# =============================================================================
# Timing Tests
# =============================================================================

class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_basic_functionality(self):
        """Test basic monitoring."""
        from utils.timing import ResourceMonitor

        monitor = ResourceMonitor(interval=0.05)
        monitor.start()

        # Do some work
        data = [list(range(10000)) for _ in range(10)]
        time.sleep(0.1)

        stats = monitor.stop()

        assert 'elapsed_time' in stats
        assert 'peak_memory_mb' in stats
        assert stats['elapsed_time'] > 0.05
        assert stats['peak_memory_mb'] > 0

        # Cleanup
        del data

    def test_context_manager(self):
        """Test as context manager."""
        from utils.timing import ResourceMonitor

        with ResourceMonitor(interval=0.05) as monitor:
            time.sleep(0.1)

        assert not monitor.monitoring

    def test_error_on_double_start(self):
        """Test error when starting twice."""
        from utils.timing import ResourceMonitor

        monitor = ResourceMonitor()
        monitor.start()

        with pytest.raises(RuntimeError):
            monitor.start()

        monitor.stop()

    def test_error_on_stop_without_start(self):
        """Test error when stopping without starting."""
        from utils.timing import ResourceMonitor

        monitor = ResourceMonitor()

        with pytest.raises(RuntimeError):
            monitor.stop()


class TestTimedDecorator:
    """Tests for @timed decorator."""

    def test_timing_recorded(self):
        """Test that timing is recorded."""
        from utils.timing import timed

        @timed
        def test_func(n):
            return sum(range(n))

        result = test_func(10000)

        assert result == sum(range(10000))
        assert test_func.last_timing is not None
        assert test_func.last_timing > 0


# =============================================================================
# Metrics Tests
# =============================================================================

class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_basic_metrics(self):
        """Test basic metrics computation."""
        from utils.metrics import compute_metrics

        theta_history = [
            {'mu': np.array([[0.0, 0.0]])},
            {'mu': np.array([[0.5, 0.0]])},
            {'mu': np.array([[1.0, 0.0]])}
        ]
        likelihood_history = [-100.0, -90.0, -85.0]
        time_history = [0.1, 0.2, 0.3]

        metrics = compute_metrics(theta_history, likelihood_history, time_history)

        assert metrics['iterations'] == 3
        assert metrics['time'] == 0.3
        assert metrics['final_likelihood'] == -85.0
        assert metrics['convergence_rate'] > 0

    def test_with_theta_true(self):
        """Test with ground truth parameters."""
        from utils.metrics import compute_metrics

        theta_history = [
            {'mu': np.array([[1.0, 0.0]])}
        ]
        likelihood_history = [-85.0]
        time_history = [0.1]
        theta_true = {'mu': np.array([[1.0, 0.0]])}

        metrics = compute_metrics(
            theta_history, likelihood_history, time_history, theta_true
        )

        assert 'parameter_error' in metrics
        assert 'success' in metrics
        assert metrics['parameter_error'] < 0.01


class TestCheckMonotonicity:
    """Tests for check_monotonicity function."""

    def test_monotonic_sequence(self):
        """Test with monotonic sequence."""
        from utils.metrics import check_monotonicity

        history = [-100.0, -95.0, -90.0, -85.0]
        is_mono, violations = check_monotonicity(history)

        assert is_mono is True
        assert len(violations) == 0

    def test_non_monotonic_sequence(self):
        """Test with non-monotonic sequence."""
        from utils.metrics import check_monotonicity

        history = [-100.0, -95.0, -96.0, -85.0]
        is_mono, violations = check_monotonicity(history)

        assert is_mono is False
        assert violations == [2]

    def test_within_tolerance(self):
        """Test fluctuations within tolerance."""
        from utils.metrics import check_monotonicity

        history = [-100.0, -100.0 + 1e-9, -100.0]
        is_mono, violations = check_monotonicity(history, tol=1e-8)

        assert is_mono is True


class TestParameterError:
    """Tests for parameter_error function."""

    def test_zero_error(self):
        """Test with identical parameters."""
        from utils.metrics import parameter_error

        theta = {'mu': np.array([[1.0, 2.0]])}
        error = parameter_error(theta, theta)

        assert error == 0.0

    def test_nonzero_error(self):
        """Test with different parameters."""
        from utils.metrics import parameter_error

        theta_est = {'mu': np.array([[1.0, 0.0]])}
        theta_true = {'mu': np.array([[0.0, 0.0]])}

        error = parameter_error(theta_est, theta_true)
        assert abs(error - 1.0) < 1e-10


class TestCohensD:
    """Tests for cohens_d function."""

    def test_large_effect(self):
        """Test large effect size."""
        from utils.metrics import cohens_d

        x = np.array([10, 11, 12, 10, 11])
        y = np.array([20, 21, 22, 20, 21])

        d = cohens_d(x, y)
        assert abs(d) > 0.8  # Large effect

    def test_no_effect(self):
        """Test no effect (same data)."""
        from utils.metrics import cohens_d

        x = np.array([10, 11, 12, 10, 11])
        d = cohens_d(x, x)

        assert abs(d) < 0.01


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_contains_true_mean(self):
        """Test that CI contains true mean."""
        from utils.metrics import bootstrap_ci

        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)

        lower, upper = bootstrap_ci(data, n_resamples=5000, random_state=42)

        assert lower < 10 < upper


# =============================================================================
# Logging Tests
# =============================================================================

class TestExperimentLogger:
    """Tests for ExperimentLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_basic_log_and_load(self, temp_dir):
        """Test basic logging and loading."""
        from utils.logging_utils import ExperimentLogger

        logger = ExperimentLogger(base_dir=temp_dir)

        run_id = logger.log_run({
            'test_id': 'test_1',
            'algorithm': 'standard_em',
            'seed': 42,
            'final_likelihood': -2341.2
        })

        assert len(run_id) == 8

        results = logger.load_results('test_1', 'standard_em')
        assert len(results) == 1
        assert results[0]['seed'] == 42

    def test_multiple_algorithms(self, temp_dir):
        """Test multiple algorithms."""
        from utils.logging_utils import ExperimentLogger

        logger = ExperimentLogger(base_dir=temp_dir)

        for algo in ['standard_em', 'lookahead', 'squarem']:
            logger.log_run({
                'test_id': 'test_1',
                'algorithm': algo,
                'seed': 42
            })

        algos = logger.list_algorithms('test_1')
        assert len(algos) == 3

    def test_numpy_serialization(self, temp_dir):
        """Test numpy types serialize correctly."""
        from utils.logging_utils import ExperimentLogger

        logger = ExperimentLogger(base_dir=temp_dir)

        logger.log_run({
            'test_id': 'test_1',
            'algorithm': 'standard_em',
            'seed': np.int64(42),
            'likelihood': np.float64(-100.5),
            'array': np.array([1.0, 2.0, 3.0])
        })

        results = logger.load_results('test_1', 'standard_em')
        assert results[0]['seed'] == 42
        assert results[0]['array'] == [1.0, 2.0, 3.0]


# =============================================================================
# Plotting Tests
# =============================================================================

class TestPlotting:
    """Tests for plotting functions."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Set up matplotlib for testing."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        yield
        plt.close('all')

    def test_plot_convergence(self):
        """Test convergence plot."""
        from utils.plotting import plot_convergence
        import matplotlib.pyplot as plt

        histories = {
            'Standard EM': list(np.cumsum(-np.random.rand(50)) - 100),
            'Lookahead': list(np.cumsum(-np.random.rand(40)) - 100)
        }

        fig = plot_convergence(histories)
        assert fig is not None
        plt.close(fig)

    def test_plot_time_comparison(self):
        """Test time comparison plot."""
        from utils.plotting import plot_time_comparison
        import matplotlib.pyplot as plt

        times = {
            'Standard EM': list(np.random.normal(45, 5, 20)),
            'Lookahead': list(np.random.normal(30, 3, 20))
        }

        fig = plot_time_comparison(times)
        assert fig is not None
        plt.close(fig)

    def test_plot_success_rates(self):
        """Test success rates plot."""
        from utils.plotting import plot_success_rates
        import matplotlib.pyplot as plt

        rates = {'Standard EM': 0.85, 'Lookahead': 0.95}

        fig = plot_success_rates(rates)
        assert fig is not None
        plt.close(fig)

    def test_plot_memory_usage(self):
        """Test memory usage plot."""
        from utils.plotting import plot_memory_usage
        import matplotlib.pyplot as plt

        memory = {
            'Standard EM': list(np.random.normal(120, 10, 20)),
            'Lookahead': list(np.random.normal(145, 8, 20))
        }

        fig = plot_memory_usage(memory)
        assert fig is not None
        plt.close(fig)

    def test_plot_gamma_schedule(self):
        """Test gamma schedule plot."""
        from utils.plotting import plot_gamma_schedule
        import matplotlib.pyplot as plt

        gammas = [min(0.9, 0.1 + 0.05 * t) for t in range(50)]
        likelihoods = list(np.cumsum(-np.random.rand(50)) - 100)

        fig = plot_gamma_schedule(gammas, likelihoods)
        assert fig is not None
        plt.close(fig)

    def test_save_figure(self):
        """Test figure saving."""
        from utils.plotting import plot_success_rates
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test.png')

            fig = plot_success_rates({'A': 0.9}, save_path=save_path)
            plt.close(fig)

            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 1000


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
