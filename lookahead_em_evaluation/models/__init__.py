"""
Statistical Models for EM

This module contains model implementations:
- GaussianMixtureModel: GMM with block-diagonal Hessian
- HiddenMarkovModel: HMM with sparse transition structure
- MixtureOfExperts: MoE with hierarchical block structure
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gmm import GaussianMixtureModel
    from .hmm import HiddenMarkovModel
    from .mixture_of_experts import MixtureOfExperts
