"""
Lookahead EM: Accelerated Expectation-Maximization with Value Function Lookahead

A Python package implementing the Lookahead EM algorithm, which augments standard EM
with a value function to look ahead and accelerate convergence.

Quick Start
-----------
>>> from lookahead_em_evaluation import (
...     LookaheadEM, MixtureOfExperts, generate_moe_data, initialize_equal_gates
... )
>>>
>>> # Generate sample data
>>> X, y, experts, theta_true = generate_moe_data(n=500, n_experts=3, n_features=2)
>>>
>>> # Fit with Lookahead EM
>>> model = MixtureOfExperts(n_experts=3, n_features=2)
>>> theta_init = initialize_equal_gates(n_experts=3, n_features=2)
>>> em = LookaheadEM(model, gamma='adaptive')
>>> theta_final, diagnostics = em.fit((X, y), theta_init)
>>>
>>> print(f"Converged in {diagnostics['iterations']} iterations")

Algorithms
----------
- StandardEM: Baseline EM algorithm
- LookaheadEM: Novel EM with value-function lookahead (recommended)

Models
------
- GaussianMixtureModel: Classic GMM with diagonal covariance
- HiddenMarkovModel: Discrete HMM with forward-backward algorithm
- MixtureOfExperts: Gated linear regression with soft expert assignment

Installation
------------
pip install lookahead-em

Or from source:
pip install git+https://github.com/MLenaBleile/em_lookahead_algorithm.git
"""

__version__ = "1.0.0"
__author__ = "MaryLena Bleile"

# Algorithms
from .algorithms import StandardEM, LookaheadEM

# Models
from .models import GaussianMixtureModel, HiddenMarkovModel, MixtureOfExperts

# Data generation
from .data import generate_gmm_data, generate_hmm_data
from .models.mixture_of_experts import generate_moe_data, initialize_equal_gates

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Algorithms
    "StandardEM",
    "LookaheadEM",
    # Models
    "GaussianMixtureModel",
    "HiddenMarkovModel",
    "MixtureOfExperts",
    # Data generation
    "generate_gmm_data",
    "generate_hmm_data",
    "generate_moe_data",
    "initialize_equal_gates",
]
