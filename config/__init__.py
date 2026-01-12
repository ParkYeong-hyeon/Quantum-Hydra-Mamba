"""
Configuration module for Quantum Hydra Mamba project.
"""

from .data_paths import (
    DATA_ROOT,
    DATA_DIRS,
    ABSOLUTE_DATA_PATHS,
    PROJECT_ROOT,
    get_data_path,
    get_synthetic_path,
    get_dna_path,
    get_physionet_path,
    get_genomic_path,
)

from .job_config import (
    PY,
    CONDA_ENV,
    CONDA_ENV_PATH,
    MICROMAMBA_PATH,
    SLURM_DEFAULTS,
    get_output_dir,
    get_jobs_dir,
    get_script_path,
    get_slurm_config,
)

__all__ = [
    # Data paths
    'DATA_ROOT',
    'DATA_DIRS',
    'ABSOLUTE_DATA_PATHS',
    'PROJECT_ROOT',
    'get_data_path',
    'get_synthetic_path',
    'get_dna_path',
    'get_physionet_path',
    'get_genomic_path',
    # Job config
    'PY',
    'CONDA_ENV',
    'CONDA_ENV_PATH',
    'MICROMAMBA_PATH',
    'SLURM_DEFAULTS',
    'get_output_dir',
    'get_jobs_dir',
    'get_script_path',
    'get_slurm_config',
]
