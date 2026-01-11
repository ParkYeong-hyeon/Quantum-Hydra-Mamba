"""
Image Dataset Loaders

Loaders for image classification datasets.
"""

from .Load_Image_Datasets import (
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

__all__ = [
    'load_mnist',
    'load_fashion',
    'load_cifar',
    'load_celeba',
    'load_coco',
    'load_eeg',
    'load_eeg_ts',
    'load_binary_classification',
    'load_data',
]
