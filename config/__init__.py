"""
Configuration module for Quantum Hydra Mamba project.
"""

from .data_paths import (
    DATA_ROOT,
    DATA_DIRS,
    ABSOLUTE_DATA_PATHS,
    get_data_path,
    get_synthetic_path,
    get_dna_path,
    get_physionet_path,
    get_genomic_path,
)

__all__ = [
    'DATA_ROOT',
    'DATA_DIRS',
    'ABSOLUTE_DATA_PATHS',
    'get_data_path',
    'get_synthetic_path',
    'get_dna_path',
    'get_physionet_path',
    'get_genomic_path',
]
