"""
EM Algorithm Implementations

This module contains implementations of:
- StandardEM: Baseline EM algorithm
- LookaheadEM: Novel lookahead EM with adaptive gamma
- SQUAREMWrapper: Python wrapper for R's turboEM SQUAREM
- Gamma schedules for lookahead EM
- Value function estimation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .standard_em import StandardEM
    from .lookahead_em import LookaheadEM
    from .gamma_schedule import GammaSchedule, FixedGamma, AdaptiveGamma, ExponentialGamma
    from .value_estimation import estimate_V_second_order
