"""
Data Generation Utilities

This module contains data generators for:
- GMM: Gaussian Mixture Model data
- HMM: Hidden Markov Model sequences
- MoE: Mixture of Experts regression data
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate_gmm import generate_gmm_data, generate_initializations
    from .generate_hmm import generate_hmm_data
    from .generate_moe import generate_moe_data
