#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --account=m4727_g
#SBATCH --nodelist=node4
#SBATCH --qos=shared
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
# Logs are useful for tracking download progress and debugging failures
#SBATCH --output=jobs/download_datasets/logs/download_datasets_%j.out
#SBATCH --error=jobs/download_datasets/logs/download_datasets_%j.err

# ============================================
# Dataset Download/Generation Job
# ============================================
# This job downloads/generates all datasets for Quantum-Hydra-Mamba project:
#   - Synthetic Benchmarks (Forrelation, Adding Problem, Selective Copy)
#   - DNA Promoter dataset
#   - Genomic Benchmarks datasets
#   - PhysioNet EEG dataset
# ============================================

echo "=========================================="
echo "Starting dataset download job"
echo "=========================================="
echo "Job name: download_datasets"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Navigate to project directory first (needed to import config)
cd /scratch/connectome/mandy/projects/quantum_hydra_mamba/Quantum-Hydra-Mamba

# Get Python path from config
PYTHON_CMD=$(/scratch/connectome/mandy/miniforge/bin/python -c "import sys; sys.path.insert(0, '.'); from config import PY; print(PY)")

# Verify Python path exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found at $PYTHON_CMD"
    exit 1
fi

echo "Python path (from config): $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version 2>&1)"

# Verify data directory structure
echo ""
echo "Checking data directory structure..."
$PYTHON_CMD -c "
from config.data_paths import DATA_DIRS, DATA_ROOT
from pathlib import Path
print(f'Data root: {DATA_ROOT}')
print(f'Data directories:')
for key, path in DATA_DIRS.items():
    actual_path = path.resolve()
    print(f'  {key}: {actual_path}')
    if actual_path.exists():
        print(f'    ✓ Directory exists')
    else:
        print(f'    ✗ Directory does not exist')
"

# Run dataset download script
echo ""
echo "Running dataset download script..."
echo "Command: python scripts/mandy_download_all_datasets.py --skip-synthetic"
echo ""

$PYTHON_CMD scripts/mandy_download_all_datasets.py --skip-synthetic

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Dataset download job completed successfully!"
else
    echo "❌ Dataset download job failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Verify downloaded datasets
echo ""
echo "Verifying downloaded datasets..."
$PYTHON_CMD -c "
from config.data_paths import get_synthetic_path, get_dna_path, get_genomic_path, get_physionet_path
from pathlib import Path

print('Dataset verification:')
print('-' * 50)

# Synthetic benchmarks
syn_path = get_synthetic_path()
print(f'Synthetic benchmarks: {syn_path.resolve()}')
if syn_path.exists():
    forrelation = list((syn_path / 'forrelation').glob('*.pt'))
    adding = list((syn_path / 'adding_problem').glob('*.pt'))
    selective = list((syn_path / 'selective_copy').glob('*.pt'))
    print(f'  Forrelation: {len(forrelation)} files')
    print(f'  Adding Problem: {len(adding)} files')
    print(f'  Selective Copy: {len(selective)} files')
else:
    print('  ✗ Directory does not exist')

# DNA dataset
dna_path = get_dna_path()
dna_file = dna_path / 'promoters.data'
print(f'DNA dataset: {dna_file.resolve()}')
if dna_file.exists():
    size = dna_file.stat().st_size / 1024  # KB
    print(f'  ✓ File exists ({size:.1f} KB)')
else:
    print('  ✗ File does not exist')

# Genomic benchmarks
genomic_path = get_genomic_path()
print(f'Genomic benchmarks: {genomic_path.resolve()}')
if genomic_path.exists():
    datasets = [d for d in genomic_path.iterdir() if d.is_dir()]
    print(f'  {len(datasets)} datasets found')
    for ds in datasets:
        train_dir = ds / 'train'
        if train_dir.exists():
            print(f'    ✓ {ds.name}')
        else:
            print(f'    ✗ {ds.name} (incomplete)')
else:
    print('  ✗ Directory does not exist')

# PhysioNet EEG
physionet_path = get_physionet_path()
print(f'PhysioNet EEG: {physionet_path.resolve()}')
if physionet_path.exists():
    eegbci = physionet_path / 'files' / 'eegmmidb'
    if eegbci.exists():
        subjects = list(eegbci.glob('*'))
        print(f'  ✓ {len(subjects)} subjects found')
    else:
        print('  ✗ Data not downloaded yet')
else:
    print('  ✗ Directory does not exist')
"

exit $EXIT_CODE
