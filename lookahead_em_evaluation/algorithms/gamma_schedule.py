"""
Gamma Scheduling for Lookahead EM

This module provides gamma scheduling strategies for the lookahead EM algorithm.
The gamma parameter controls the lookahead weight in the objective:
    θ^(t+1) = argmax [Q(θ|θ^(t)) + γ · V(θ)]

Different schedules allow for adaptive exploration-exploitation tradeoffs:
- FixedGamma: Constant gamma throughout
- AdaptiveGamma: Linear increase from initial to max
- ExponentialGamma: Exponential approach to max

Example usage:
    >>> schedule = AdaptiveGamma(gamma_init=0.1, gamma_max=0.9, gamma_step=0.05)
    >>> for t in range(100):
    ...     gamma = schedule.get_gamma(t)
    ...     # Use gamma in lookahead step
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class GammaSchedule(ABC):
    """
    Abstract base class for gamma schedules.

    Gamma schedules determine how the lookahead weight γ changes
    over iterations. The value should be in [0, 1] where:
    - γ = 0: Standard EM (no lookahead)
    - γ = 1: Maximum lookahead weight

    Subclasses must implement get_gamma(iteration).
    """

    @abstractmethod
    def get_gamma(self, iteration: int) -> float:
        """
        Get gamma value for the given iteration.

        Args:
            iteration: Current iteration number (0-indexed).

        Returns:
            Gamma value in [0, 1].
        """
        pass

    def reset(self) -> None:
        """
        Reset schedule for a new run.

        Override in subclasses if schedule maintains internal state.
        """
        pass

    def get_schedule(self, n_iterations: int) -> List[float]:
        """
        Get gamma values for multiple iterations.

        Args:
            n_iterations: Number of iterations.

        Returns:
            List of gamma values for iterations 0 to n_iterations-1.
        """
        return [self.get_gamma(t) for t in range(n_iterations)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FixedGamma(GammaSchedule):
    """
    Fixed gamma schedule - returns constant value.

    This is equivalent to using a fixed lookahead weight throughout
    the algorithm. Use gamma=0 to recover standard EM.

    Attributes:
        gamma: The constant gamma value.

    Example:
        >>> schedule = FixedGamma(gamma=0.5)
        >>> schedule.get_gamma(0)
        0.5
        >>> schedule.get_gamma(100)
        0.5
    """

    def __init__(self, gamma: float = 0.5):
        """
        Initialize fixed gamma schedule.

        Args:
            gamma: Constant gamma value in [0, 1].

        Raises:
            ValueError: If gamma is outside [0, 1].
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self.gamma = gamma

    def get_gamma(self, iteration: int) -> float:
        """Return the constant gamma value."""
        return self.gamma

    def __repr__(self) -> str:
        return f"FixedGamma(gamma={self.gamma})"


class AdaptiveGamma(GammaSchedule):
    """
    Adaptive gamma schedule - linear increase then plateau.

    Gamma increases linearly from gamma_init by gamma_step per iteration
    until it reaches gamma_max:
        γ_t = min(gamma_max, gamma_init + gamma_step × t)

    This allows conservative exploration early, then more aggressive
    lookahead as the algorithm approaches convergence.

    Attributes:
        gamma_init: Initial gamma value.
        gamma_max: Maximum gamma value (plateau).
        gamma_step: Increase per iteration.

    Example:
        >>> schedule = AdaptiveGamma(gamma_init=0.1, gamma_max=0.9, gamma_step=0.05)
        >>> schedule.get_gamma(0)
        0.1
        >>> schedule.get_gamma(10)
        0.6
        >>> schedule.get_gamma(20)
        0.9  # Capped at gamma_max
    """

    def __init__(
        self,
        gamma_init: float = 0.1,
        gamma_max: float = 0.9,
        gamma_step: float = 0.05
    ):
        """
        Initialize adaptive gamma schedule.

        Args:
            gamma_init: Initial gamma value (at iteration 0).
            gamma_max: Maximum gamma value (plateau).
            gamma_step: Gamma increase per iteration.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 <= gamma_init <= 1:
            raise ValueError(f"gamma_init must be in [0, 1], got {gamma_init}")
        if not 0 <= gamma_max <= 1:
            raise ValueError(f"gamma_max must be in [0, 1], got {gamma_max}")
        if gamma_init > gamma_max:
            raise ValueError(f"gamma_init ({gamma_init}) must be <= gamma_max ({gamma_max})")
        if gamma_step < 0:
            raise ValueError(f"gamma_step must be >= 0, got {gamma_step}")

        self.gamma_init = gamma_init
        self.gamma_max = gamma_max
        self.gamma_step = gamma_step

    def get_gamma(self, iteration: int) -> float:
        """
        Get gamma for the given iteration.

        Returns min(gamma_max, gamma_init + gamma_step * iteration).
        """
        return min(self.gamma_max, self.gamma_init + self.gamma_step * iteration)

    def iterations_to_max(self) -> int:
        """
        Calculate iterations needed to reach gamma_max.

        Returns:
            Number of iterations to reach gamma_max.
            Returns 0 if gamma_init >= gamma_max.
        """
        if self.gamma_step <= 0:
            return float('inf') if self.gamma_init < self.gamma_max else 0

        return int(np.ceil((self.gamma_max - self.gamma_init) / self.gamma_step))

    def __repr__(self) -> str:
        return (f"AdaptiveGamma(gamma_init={self.gamma_init}, "
                f"gamma_max={self.gamma_max}, gamma_step={self.gamma_step})")


class ExponentialGamma(GammaSchedule):
    """
    Exponential gamma schedule - exponential approach to maximum.

    Gamma approaches gamma_max exponentially from gamma_init:
        γ_t = gamma_max - (gamma_max - gamma_init) × exp(-rate × t)

    This provides smooth acceleration with diminishing rate of change.

    Attributes:
        gamma_init: Initial gamma value.
        gamma_max: Maximum gamma value (asymptote).
        rate: Exponential decay rate.

    Example:
        >>> schedule = ExponentialGamma(gamma_init=0.1, gamma_max=0.9, rate=0.1)
        >>> schedule.get_gamma(0)
        0.1
        >>> schedule.get_gamma(10)  # About 0.63 of the way
        0.5965...
        >>> schedule.get_gamma(50)  # Very close to gamma_max
        0.8993...
    """

    def __init__(
        self,
        gamma_init: float = 0.1,
        gamma_max: float = 0.9,
        rate: float = 0.1
    ):
        """
        Initialize exponential gamma schedule.

        Args:
            gamma_init: Initial gamma value (at iteration 0).
            gamma_max: Maximum gamma value (asymptote).
            rate: Exponential approach rate. Higher = faster approach.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 <= gamma_init <= 1:
            raise ValueError(f"gamma_init must be in [0, 1], got {gamma_init}")
        if not 0 <= gamma_max <= 1:
            raise ValueError(f"gamma_max must be in [0, 1], got {gamma_max}")
        if gamma_init > gamma_max:
            raise ValueError(f"gamma_init ({gamma_init}) must be <= gamma_max ({gamma_max})")
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")

        self.gamma_init = gamma_init
        self.gamma_max = gamma_max
        self.rate = rate

    def get_gamma(self, iteration: int) -> float:
        """
        Get gamma for the given iteration.

        Returns gamma_max - (gamma_max - gamma_init) * exp(-rate * t).
        """
        decay = np.exp(-self.rate * iteration)
        return self.gamma_max - (self.gamma_max - self.gamma_init) * decay

    def half_life(self) -> float:
        """
        Calculate iterations for gamma to reach halfway to gamma_max.

        Returns:
            Number of iterations for gamma to reach (gamma_init + gamma_max) / 2.
        """
        return np.log(2) / self.rate

    def __repr__(self) -> str:
        return (f"ExponentialGamma(gamma_init={self.gamma_init}, "
                f"gamma_max={self.gamma_max}, rate={self.rate})")


class CosineAnnealingGamma(GammaSchedule):
    """
    Cosine annealing gamma schedule.

    Gamma follows a cosine curve from gamma_init to gamma_max over
    a specified period, optionally with restarts.

    This can help with escaping local optima through periodic
    exploration phases.

    Attributes:
        gamma_init: Initial/minimum gamma value.
        gamma_max: Maximum gamma value.
        period: Iterations per cosine cycle.
    """

    def __init__(
        self,
        gamma_init: float = 0.1,
        gamma_max: float = 0.9,
        period: int = 50
    ):
        """
        Initialize cosine annealing schedule.

        Args:
            gamma_init: Minimum gamma (start and restart value).
            gamma_max: Maximum gamma (peak of cycle).
            period: Iterations per full cycle.
        """
        if not 0 <= gamma_init <= 1:
            raise ValueError(f"gamma_init must be in [0, 1], got {gamma_init}")
        if not 0 <= gamma_max <= 1:
            raise ValueError(f"gamma_max must be in [0, 1], got {gamma_max}")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")

        self.gamma_init = gamma_init
        self.gamma_max = gamma_max
        self.period = period

    def get_gamma(self, iteration: int) -> float:
        """Get gamma using cosine annealing."""
        progress = (iteration % self.period) / self.period
        cosine_factor = 0.5 * (1 - np.cos(np.pi * progress))
        return self.gamma_init + (self.gamma_max - self.gamma_init) * cosine_factor

    def __repr__(self) -> str:
        return (f"CosineAnnealingGamma(gamma_init={self.gamma_init}, "
                f"gamma_max={self.gamma_max}, period={self.period})")


def create_schedule(
    gamma: str | float,
    gamma_init: float = 0.1,
    gamma_max: float = 0.9,
    gamma_step: float = 0.05,
    rate: float = 0.1
) -> GammaSchedule:
    """
    Factory function to create gamma schedules.

    Args:
        gamma: Either a float (creates FixedGamma) or string:
               - 'adaptive': Creates AdaptiveGamma
               - 'exponential': Creates ExponentialGamma
               - 'fixed': Creates FixedGamma with gamma_init
        gamma_init: Initial gamma for adaptive/exponential schedules.
        gamma_max: Maximum gamma for adaptive/exponential schedules.
        gamma_step: Step size for adaptive schedule.
        rate: Rate for exponential schedule.

    Returns:
        GammaSchedule instance.

    Example:
        >>> schedule = create_schedule('adaptive')
        >>> schedule = create_schedule(0.5)  # Fixed at 0.5
    """
    if isinstance(gamma, (int, float)):
        return FixedGamma(gamma=float(gamma))

    gamma_lower = gamma.lower()

    if gamma_lower == 'fixed':
        return FixedGamma(gamma=gamma_init)
    elif gamma_lower == 'adaptive':
        return AdaptiveGamma(
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            gamma_step=gamma_step
        )
    elif gamma_lower == 'exponential':
        return ExponentialGamma(
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            rate=rate
        )
    else:
        raise ValueError(f"Unknown gamma schedule: {gamma}")


# =============================================================================
# Unit Tests
# =============================================================================

def test_fixed_gamma():
    """Test FixedGamma schedule."""
    print("Testing FixedGamma...")

    schedule = FixedGamma(gamma=0.5)

    # Should always return same value
    assert schedule.get_gamma(0) == 0.5, "Should return 0.5 at iteration 0"
    assert schedule.get_gamma(100) == 0.5, "Should return 0.5 at iteration 100"
    assert schedule.get_gamma(1000) == 0.5, "Should return 0.5 at iteration 1000"

    # Test get_schedule
    schedule_list = schedule.get_schedule(10)
    assert all(g == 0.5 for g in schedule_list), "All values should be 0.5"

    # Test invalid gamma
    try:
        FixedGamma(gamma=1.5)
        assert False, "Should raise ValueError for gamma > 1"
    except ValueError:
        pass

    print("  PASSED")


def test_adaptive_gamma():
    """Test AdaptiveGamma schedule."""
    print("Testing AdaptiveGamma...")

    schedule = AdaptiveGamma(gamma_init=0.1, gamma_max=0.9, gamma_step=0.05)

    # Test initial value
    assert schedule.get_gamma(0) == 0.1, "Should start at gamma_init"

    # Test linear increase
    assert abs(schedule.get_gamma(5) - 0.35) < 1e-10, "Should be 0.35 at iteration 5"
    assert abs(schedule.get_gamma(10) - 0.6) < 1e-10, "Should be 0.6 at iteration 10"

    # Test plateau
    assert schedule.get_gamma(20) == 0.9, "Should plateau at gamma_max"
    assert schedule.get_gamma(100) == 0.9, "Should stay at gamma_max"

    # Test iterations_to_max
    assert schedule.iterations_to_max() == 16, "Should take 16 iterations to reach max"

    # Test get_schedule
    schedule_list = schedule.get_schedule(25)
    assert len(schedule_list) == 25, "Should have 25 values"
    assert schedule_list[0] == 0.1, "First should be gamma_init"
    assert schedule_list[-1] == 0.9, "Last should be gamma_max"

    # Verify monotonicity
    for i in range(1, len(schedule_list)):
        assert schedule_list[i] >= schedule_list[i-1], "Should be monotonically increasing"

    print("  PASSED")


def test_exponential_gamma():
    """Test ExponentialGamma schedule."""
    print("Testing ExponentialGamma...")

    schedule = ExponentialGamma(gamma_init=0.1, gamma_max=0.9, rate=0.1)

    # Test initial value (use approximate comparison for floating point)
    assert abs(schedule.get_gamma(0) - 0.1) < 1e-10, "Should start at gamma_init"

    # Test exponential approach
    gamma_10 = schedule.get_gamma(10)
    expected_10 = 0.9 - 0.8 * np.exp(-0.1 * 10)
    assert abs(gamma_10 - expected_10) < 1e-10, "Should follow exponential formula"

    # Test approaches but never exceeds max
    gamma_100 = schedule.get_gamma(100)
    assert gamma_100 < 0.9, "Should not exceed gamma_max"
    assert gamma_100 > 0.89, "Should be close to gamma_max"

    # Test half-life
    half_life = schedule.half_life()
    expected_half_life = np.log(2) / 0.1
    assert abs(half_life - expected_half_life) < 1e-10, "Half-life formula should match"

    # Verify monotonicity
    schedule_list = schedule.get_schedule(50)
    for i in range(1, len(schedule_list)):
        assert schedule_list[i] > schedule_list[i-1], \
            "Should be strictly monotonically increasing"

    print("  PASSED")


def test_cosine_annealing_gamma():
    """Test CosineAnnealingGamma schedule."""
    print("Testing CosineAnnealingGamma...")

    schedule = CosineAnnealingGamma(gamma_init=0.1, gamma_max=0.9, period=20)

    # Test initial value (use approximate comparison for floating point)
    assert abs(schedule.get_gamma(0) - 0.1) < 1e-10, "Should start at gamma_init"

    # Test midpoint value (half of range)
    gamma_half = schedule.get_gamma(10)
    assert abs(gamma_half - 0.5) < 1e-10, "Should be at midpoint at half period"

    # Test near peak (just before period ends)
    gamma_near_peak = schedule.get_gamma(19)
    assert gamma_near_peak > 0.85, "Should be near gamma_max just before period ends"

    # Test restart at period boundary
    gamma_restart = schedule.get_gamma(20)
    assert abs(gamma_restart - 0.1) < 1e-10, "Should restart at gamma_init at period boundary"

    # Test periodicity
    assert abs(schedule.get_gamma(5) - schedule.get_gamma(25)) < 1e-10, \
        "Should be periodic"

    print("  PASSED")


def test_create_schedule():
    """Test create_schedule factory function."""
    print("Testing create_schedule...")

    # Test with float
    schedule = create_schedule(0.5)
    assert isinstance(schedule, FixedGamma), "Float should create FixedGamma"
    assert schedule.gamma == 0.5, "Gamma should be 0.5"

    # Test with string
    schedule = create_schedule('adaptive')
    assert isinstance(schedule, AdaptiveGamma), "Should create AdaptiveGamma"

    schedule = create_schedule('exponential')
    assert isinstance(schedule, ExponentialGamma), "Should create ExponentialGamma"

    # Test invalid
    try:
        create_schedule('invalid')
        assert False, "Should raise ValueError for invalid schedule"
    except ValueError:
        pass

    print("  PASSED")


def test_schedule_visualization():
    """Test that schedules can be visualized (generates plot data)."""
    print("Testing schedule visualization data...")

    schedules = {
        'Fixed (0.5)': FixedGamma(0.5),
        'Adaptive': AdaptiveGamma(0.1, 0.9, 0.05),
        'Exponential': ExponentialGamma(0.1, 0.9, 0.1),
        'Cosine': CosineAnnealingGamma(0.1, 0.9, 50)
    }

    n_iter = 100
    data = {}

    for name, schedule in schedules.items():
        data[name] = schedule.get_schedule(n_iter)
        assert len(data[name]) == n_iter, f"{name} should have {n_iter} values"
        assert all(0 <= g <= 1 for g in data[name]), f"{name} values should be in [0, 1]"

    print(f"  Generated data for {len(schedules)} schedules over {n_iter} iterations")
    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running gamma_schedule.py unit tests")
    print("=" * 60)

    test_fixed_gamma()
    test_adaptive_gamma()
    test_exponential_gamma()
    test_cosine_annealing_gamma()
    test_create_schedule()
    test_schedule_visualization()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
