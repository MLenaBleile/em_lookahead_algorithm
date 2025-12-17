"""
Utility Functions

This module contains utilities for:
- Timing and resource monitoring
- Metrics computation
- Experiment logging
- Visualization/plotting
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .timing import ResourceMonitor, timed
    from .metrics import compute_metrics, check_monotonicity, parameter_error, convergence_rate
    from .logging_utils import ExperimentLogger
    from .plotting import plot_convergence, plot_time_comparison
