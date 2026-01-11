"""
Genomic Dataset Loaders

Loaders for DNA sequence and genomic benchmark datasets.
"""

from .Load_DNA_Sequences import (
    load_dna_promoter,
    encode_dna_onehot,
    encode_dna_integer,
    download_promoter_dataset,
    parse_promoter_data,
    create_synthetic_dna_data,
)

from .Load_Genomic_Benchmarks import (
    load_genomic_benchmark,
    list_available_datasets,
    get_dataset_info,
    load_human_nontata_promoters,
    load_human_enhancers_cohn,
    load_demo_coding_vs_intergenomic,
    one_hot_encode,
)

__all__ = [
    # DNA Sequences
    'load_dna_promoter',
    'encode_dna_onehot',
    'encode_dna_integer',
    'download_promoter_dataset',
    'parse_promoter_data',
    'create_synthetic_dna_data',
    # Genomic Benchmarks
    'load_genomic_benchmark',
    'list_available_datasets',
    'get_dataset_info',
    'load_human_nontata_promoters',
    'load_human_enhancers_cohn',
    'load_demo_coding_vs_intergenomic',
    'one_hot_encode',
]
