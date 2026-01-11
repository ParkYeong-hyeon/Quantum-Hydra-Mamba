"""
Data Loaders for Quantum Hydra/Mamba Models

Unified interface for loading various datasets:
- EEG: PhysioNet, SEED, FACED
- Genomic: DNA sequences, Genomic benchmarks
- Image: MNIST, CIFAR, CelebA, COCO
- NLP: GLUE benchmark
- Synthetic: Forrelation, Adding Problem, Selective Copy

Backward compatibility: All original imports still work.
"""

# EEG datasets
from .eeg import (
    # FACED
    load_faced_eeg,
    load_faced_eeg_simple,
    load_faced_valence,
    load_faced_emotion,
    FACED_EMOTION_LABELS,
    FACED_VALENCE_LABELS,
    # SEED
    load_seed_eeg,
    load_seed_eeg_simple,
    load_eeg_ts_seed,
    # PhysioNet
    load_eeg_ts_revised,
    # Unified
    load_eeg_dataset,
    list_datasets as list_eeg_datasets,
    get_dataset_config as get_eeg_dataset_config,
    print_comparison_table,
    DATASET_INFO as EEG_DATASET_INFO,
)

# Genomic datasets
from .genomic import (
    # DNA Sequences
    load_dna_promoter,
    encode_dna_onehot,
    encode_dna_integer,
    download_promoter_dataset,
    parse_promoter_data,
    create_synthetic_dna_data,
    # Genomic Benchmarks
    load_genomic_benchmark,
    list_available_datasets,
    get_dataset_info,
    load_human_nontata_promoters,
    load_human_enhancers_cohn,
    load_demo_coding_vs_intergenomic,
    one_hot_encode,
)

# Image datasets
from .image import (
    load_mnist,
    load_fashion,
    load_cifar,
    load_celeba,
    load_coco,
    load_eeg,
    load_eeg_ts,
    load_binary_classification,
    load_data,
)

# NLP datasets
from .nlp import (
    load_glue_task,
    get_glue_task_info,
    list_glue_tasks,
    GLUEDataset,
    SimpleTokenizer,
    GLUE_TASKS,
)

# Synthetic datasets
from .synthetic import (
    # Forrelation
    get_forrelation_dataloader,
    generate_forrelation_dataset,
    generate_forrelation_dataset_v2,
    generate_forrelation_pair,
    generate_forrelated_pair,
    generate_unforrelated_pair,
    get_forrelation,
    compute_forrelation,
    fourier_transform,
    fwht,
    walsh_hadamard_transform,
    fwht_inplace,
    verify_no_data_leakage,
    test_forrelation_computation,
    # Adding Problem
    get_adding_problem_dataloader,
    load_adding_problem_for_training,
    AddingProblemDataModule,
    generate_adding_problem_dataset,
    generate_adding_problem_sample,
    # Selective Copy
    get_selective_copy_dataloader,
    load_selective_copy_for_training,
    SelectiveCopyDataModule,
    compute_selective_copy_metrics,
    generate_selective_copy_dataset,
    generate_selective_copy_sample,
    generate_multiple_sequence_lengths,
)

# Backward compatibility: Direct imports from old module names
# These allow imports like: from data_loaders.Load_FACED_EEG import load_faced_eeg
import sys
import importlib

def _setup_backward_compatibility():
    """Setup module aliases for backward compatibility."""
    try:
        # EEG modules
        from . import eeg
        sys.modules['data_loaders.Load_FACED_EEG'] = eeg.Load_FACED_EEG
        sys.modules['data_loaders.Load_SEED_EEG'] = eeg.Load_SEED_EEG
        sys.modules['data_loaders.Load_PhysioNet_EEG_NoPrompt'] = eeg.Load_PhysioNet_EEG_NoPrompt
        sys.modules['data_loaders.eeg_datasets'] = eeg.eeg_datasets
        
        # Genomic modules
        from . import genomic
        sys.modules['data_loaders.Load_DNA_Sequences'] = genomic.Load_DNA_Sequences
        sys.modules['data_loaders.Load_Genomic_Benchmarks'] = genomic.Load_Genomic_Benchmarks
        
        # Image modules
        from . import image
        sys.modules['data_loaders.Load_Image_Datasets'] = image.Load_Image_Datasets
        
        # NLP modules
        from . import nlp
        sys.modules['data_loaders.Load_GLUE'] = nlp.Load_GLUE
        
        # Synthetic modules
        from .synthetic import forrelation, adding_problem, selective_copy
        sys.modules['data_loaders.forrelation_dataloader'] = forrelation.forrelation_dataloader
        sys.modules['data_loaders.generate_forrelation_dataset'] = forrelation.generate_forrelation_dataset
        sys.modules['data_loaders.generate_forrelation_dataset_v2'] = forrelation.generate_forrelation_dataset_v2
        sys.modules['data_loaders.adding_problem_dataloader'] = adding_problem.adding_problem_dataloader
        sys.modules['data_loaders.generate_adding_problem'] = adding_problem.generate_adding_problem
        sys.modules['data_loaders.selective_copy_dataloader'] = selective_copy.selective_copy_dataloader
        sys.modules['data_loaders.generate_selective_copy'] = selective_copy.generate_selective_copy
    except Exception:
        # If modules aren't loaded yet, that's okay - they'll be set up on first access
        pass

_setup_backward_compatibility()

__all__ = [
    # EEG
    'load_faced_eeg',
    'load_faced_eeg_simple',
    'load_faced_valence',
    'load_faced_emotion',
    'FACED_EMOTION_LABELS',
    'FACED_VALENCE_LABELS',
    'load_seed_eeg',
    'load_seed_eeg_simple',
    'load_eeg_ts_seed',
    'load_eeg_ts_revised',
    'load_eeg_dataset',
    'list_eeg_datasets',
    'get_eeg_dataset_config',
    'print_comparison_table',
    'EEG_DATASET_INFO',
    # Genomic
    'load_dna_promoter',
    'encode_dna_onehot',
    'encode_dna_integer',
    'download_promoter_dataset',
    'parse_promoter_data',
    'create_synthetic_dna_data',
    'load_genomic_benchmark',
    'list_available_datasets',
    'get_dataset_info',
    'load_human_nontata_promoters',
    'load_human_enhancers_cohn',
    'load_demo_coding_vs_intergenomic',
    'one_hot_encode',
    # Image
    'load_mnist',
    'load_fashion',
    'load_cifar',
    'load_celeba',
    'load_coco',
    'load_eeg',
    'load_eeg_ts',
    'load_binary_classification',
    'load_data',
    # NLP
    'load_glue_task',
    'get_glue_task_info',
    'list_glue_tasks',
    'GLUEDataset',
    'SimpleTokenizer',
    'GLUE_TASKS',
    # Synthetic
    'get_forrelation_dataloader',
    'generate_forrelation_dataset',
    'generate_forrelation_dataset_v2',
    'generate_forrelation_pair',
    'generate_forrelated_pair',
    'generate_unforrelated_pair',
    'get_forrelation',
    'compute_forrelation',
    'fourier_transform',
    'fwht',
    'walsh_hadamard_transform',
    'fwht_inplace',
    'verify_no_data_leakage',
    'test_forrelation_computation',
    'get_adding_problem_dataloader',
    'load_adding_problem_for_training',
    'AddingProblemDataModule',
    'generate_adding_problem_dataset',
    'generate_adding_problem_sample',
    'get_selective_copy_dataloader',
    'load_selective_copy_for_training',
    'SelectiveCopyDataModule',
    'compute_selective_copy_metrics',
    'generate_selective_copy_dataset',
    'generate_selective_copy_sample',
    'generate_multiple_sequence_lengths',
]
