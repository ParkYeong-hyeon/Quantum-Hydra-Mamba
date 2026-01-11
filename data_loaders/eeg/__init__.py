"""
EEG Dataset Loaders

Unified interface for loading EEG datasets:
- PhysioNet Motor Imagery
- SEED Emotion Recognition
- FACED Emotion Recognition
"""

from .Load_FACED_EEG import (
    load_faced_eeg,
    load_faced_eeg_simple,
    load_faced_valence,
    load_faced_emotion,
    FACED_EMOTION_LABELS,
    FACED_VALENCE_LABELS,
)

from .Load_SEED_EEG import (
    load_seed_eeg,
    load_seed_eeg_simple,
    load_eeg_ts_seed,
)

from .Load_PhysioNet_EEG_NoPrompt import (
    load_eeg_ts_revised,
)

from .eeg_datasets import (
    load_eeg_dataset,
    list_datasets,
    get_dataset_config,
    print_comparison_table,
    DATASET_INFO,
)

__all__ = [
    # FACED
    'load_faced_eeg',
    'load_faced_eeg_simple',
    'load_faced_valence',
    'load_faced_emotion',
    'FACED_EMOTION_LABELS',
    'FACED_VALENCE_LABELS',
    # SEED
    'load_seed_eeg',
    'load_seed_eeg_simple',
    'load_eeg_ts_seed',
    # PhysioNet
    'load_eeg_ts_revised',
    # Unified
    'load_eeg_dataset',
    'list_datasets',
    'get_dataset_config',
    'print_comparison_table',
    'DATASET_INFO',
]
