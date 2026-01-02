"""
Statistical Models for EM

This module contains model implementations:
- GaussianMixtureModel: GMM with block-diagonal Hessian
- HiddenMarkovModel: HMM with sparse transition structure
- MixtureOfExperts: MoE with hierarchical block structure
"""

from .gmm import GaussianMixtureModel
from .hmm import HiddenMarkovModel
from .mixture_of_experts import MixtureOfExperts

__all__ = [
    "GaussianMixtureModel",
    "HiddenMarkovModel",
    "MixtureOfExperts",
]
