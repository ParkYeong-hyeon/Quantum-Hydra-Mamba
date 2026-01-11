"""
Forrelation Dataset Loader and Generator

BQP-complete Forrelation task for testing quantum advantage.
"""

from .forrelation_dataloader import (
    get_forrelation_dataloader,
)

from .generate_forrelation_dataset import (
    generate_dataset as generate_forrelation_dataset,
    generate_forrelation_pair,
    get_forrelation,
    fourier_transform,
    fwht,
)

from .generate_forrelation_dataset_v2 import (
    generate_dataset as generate_forrelation_dataset_v2,
    generate_forrelated_pair,
    generate_unforrelated_pair,
    compute_forrelation,
    verify_no_data_leakage,
    test_forrelation_computation,
    walsh_hadamard_transform,
    fwht_inplace,
)

__all__ = [
    # Dataloader
    'get_forrelation_dataloader',
    # Generator v1
    'generate_forrelation_dataset',
    'generate_forrelation_pair',
    'get_forrelation',
    'fourier_transform',
    'fwht',
    # Generator v2
    'generate_forrelation_dataset_v2',
    'generate_forrelated_pair',
    'generate_unforrelated_pair',
    'compute_forrelation',
    'verify_no_data_leakage',
    'test_forrelation_computation',
    'walsh_hadamard_transform',
    'fwht_inplace',
]
