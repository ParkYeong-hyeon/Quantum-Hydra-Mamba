"""
Scripts for Quantum Hydra/Mamba Project

Organized by functionality:
- training/: Model training execution scripts
- job_generation/: SLURM job script generators
- aggregation/: Result aggregation and statistics
- analysis/: Experiment analysis and diagnostics
- evaluation/: Model evaluation and testing
- specialized/: Task-specific wrappers

Backward compatibility: Scripts can be executed directly from their new locations,
or accessed via module imports. For direct execution, use the new paths:
  python scripts/training/run_ablation_eeg.py
  python scripts/job_generation/generate_ablation_eeg_jobs.py

Or use the symbolic links in the scripts root for backward compatibility:
  python scripts/run_ablation_eeg.py (still works via symlink)
"""

import sys
import importlib.util
from pathlib import Path

# Backward compatibility: Setup module aliases for script imports
# This allows: from scripts.run_ablation_eeg import ... (if needed)
def _setup_backward_compatibility():
    """Setup module aliases for backward compatibility."""
    _base_path = Path(__file__).parent
    
    # Training modules
    training_scripts = [
        'run_ablation_eeg', 'run_synthetic_benchmark',
        'run_single_model_eeg', 'run_single_model_dna',
        'run_single_model_mnist', 'run_single_model_forrelation',
        'run_full_comparison', 'run_gated_models',
        'run_gated_genomic', 'run_glue', 'run_ssm_genomic_comparison'
    ]
    
    # Job generation modules
    job_scripts = [
        'generate_ablation_eeg_jobs', 'generate_synthetic_jobs',
        'generate_dna_job_scripts', 'generate_eeg_job_scripts',
        'generate_forrelation_job_scripts', 'generate_mnist_job_scripts',
    ]
    
    # Data generation modules (in scripts root)
    data_generation_scripts = [
        'generate_all_synthetic_datasets',
    ]
    
    # Aggregation modules
    aggregation_scripts = [
        'aggregate_ablation_results', 'aggregate_synthetic_results',
        'aggregate_dna_results', 'aggregate_eeg_results',
        'aggregate_mnist_results', 'aggregate_forrelation_results'
    ]
    
    # Evaluation modules
    evaluation_scripts = [
        'evaluate_checkpoint', 'test_eeg_with_quantum_models',
        'test_faced_with_quantum_models'
    ]
    
    # Register all modules for backward compatibility
    for script_name in training_scripts:
        _register_module(script_name, _base_path / 'training' / f'{script_name}.py')
    
    for script_name in job_scripts:
        _register_module(script_name, _base_path / 'job_generation' / f'{script_name}.py')
    
    for script_name in data_generation_scripts:
        _register_module(script_name, _base_path / f'{script_name}.py')
    
    for script_name in aggregation_scripts:
        _register_module(script_name, _base_path / 'aggregation' / f'{script_name}.py')
    
    _register_module('analyze_all_experiments', _base_path / 'analysis' / 'analyze_all_experiments.py')
    
    for script_name in evaluation_scripts:
        _register_module(script_name, _base_path / 'evaluation' / f'{script_name}.py')
    
    _register_module('QuantumHydraGLUE', _base_path / 'specialized' / 'QuantumHydraGLUE.py')

def _register_module(module_name, file_path):
    """Register a module in sys.modules for backward compatibility."""
    if not file_path.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(
            f'scripts.{module_name}',
            file_path
        )
        if spec and spec.loader:
            # Only register if not already registered
            if f'scripts.{module_name}' not in sys.modules:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f'scripts.{module_name}'] = module
    except Exception:
        # Silently fail - module will be loaded on first access
        pass

_setup_backward_compatibility()

__all__ = [
    # Submodules
    'training',
    'job_generation',
    'aggregation',
    'analysis',
    'evaluation',
    'specialized',
]
