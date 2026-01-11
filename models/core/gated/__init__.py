"""Gated recurrence modules."""

from models.core.gated.QuantumGatedRecurrence import (
    QuantumFeatureExtractor,
    QuantumStateProcessor,
    QuantumSuperpositionBranches,
    ChunkedGatedSuperposition,
    QuantumMambaGated,
    QuantumHydraGated,
)

__all__ = [
    'QuantumFeatureExtractor',
    'QuantumStateProcessor',
    'QuantumSuperpositionBranches',
    'ChunkedGatedSuperposition',
    'QuantumMambaGated',
    'QuantumHydraGated',
]
