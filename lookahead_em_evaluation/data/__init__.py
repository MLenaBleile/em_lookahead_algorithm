"""
Data Generation Utilities

This module contains data generators for:
- GMM: Gaussian Mixture Model data
- HMM: Hidden Markov Model sequences
- MoE: Mixture of Experts regression data (in models.mixture_of_experts)
"""

from .generate_gmm import generate_gmm_data, generate_initializations
from .generate_hmm import generate_hmm_data

__all__ = [
    "generate_gmm_data",
    "generate_initializations",
    "generate_hmm_data",
]
