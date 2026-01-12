#!/usr/bin/env python3
"""
Generate SLURM job script for downloading/generating all datasets.

This script creates a SLURM job script that uses job_config.py settings
to properly download all datasets for the Quantum-Hydra-Mamba project.

Usage:
    python scripts/job_generation/generate_download_datasets_job.py
    bash jobs/download_datasets/download_datasets.sh
"""

from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    PROJECT_ROOT,
    PY,
    get_jobs_dir,
    get_slurm_config,
)

# Job configuration
JOB_NAME = "download_datasets"
JOBS_DIR = get_jobs_dir('download_datasets')

# SLURM Configuration (from config, can be overridden)
# Data download doesn't need GPU, but may need time and CPUs
slurm_config = get_slurm_config(
    time='12:00:00',  # Data download may take time
    gpus=0,  # No GPU needed for data download
    cpus_per_task=16,  # Some parallel processing for data generation
    constraint=None,  # No GPU constraint needed for CPU-only data download
    nodelist='node4',  # Use node4 (CPU-only node)
)
SLURM_ACCOUNT = slurm_config['account']
SLURM_CONSTRAINT = slurm_config.get('constraint')  # May be None
SLURM_NODELIST = slurm_config.get('nodelist')  # Specific node to use
SLURM_QOS = slurm_config['qos']
SLURM_TIME = slurm_config['time']
SLURM_NODES = slurm_config['nodes']
SLURM_GPUS = slurm_config['gpus']
SLURM_CPUS = slurm_config['cpus_per_task']

# Use CONDA_ENV_PATH from config (already imported)

# Download script path
DOWNLOAD_SCRIPT = PROJECT_ROOT / 'scripts' / 'mandy_download_all_datasets.py'

# Log directory
LOGS_DIR = PROJECT_ROOT / 'jobs' / 'download_datasets' / 'logs'


def create_job_script(**kwargs):
    """
    Create SLURM job script for downloading datasets.
    
    Args:
        **kwargs: Optional arguments to pass to download script:
            - skip_synthetic: Skip synthetic dataset generation
            - skip_dna: Skip DNA dataset download
            - skip_genomic: Skip Genomic Benchmarks download
            - skip_physionet: Skip PhysioNet EEG download
            - genomic_datasets: List of genomic datasets to download
            - synthetic_seed: Random seed for synthetic dataset generation
            - physionet_subjects: Number of PhysioNet subjects to download
    """
    
    # Build command arguments
    cmd_args = []
    if kwargs.get('skip_synthetic'):
        cmd_args.append('--skip-synthetic')
    if kwargs.get('skip_dna'):
        cmd_args.append('--skip-dna')
    if kwargs.get('skip_genomic'):
        cmd_args.append('--skip-genomic')
    if kwargs.get('skip_physionet'):
        cmd_args.append('--skip-physionet')
    if kwargs.get('genomic_datasets'):
        cmd_args.extend(['--genomic-datasets'] + kwargs['genomic_datasets'])
    if kwargs.get('synthetic_seed'):
        cmd_args.extend(['--synthetic-seed', str(kwargs['synthetic_seed'])])
    if kwargs.get('physionet_subjects'):
        cmd_args.extend(['--physionet-subjects', str(kwargs['physionet_subjects'])])
    
    args_str = ' '.join(cmd_args) if cmd_args else ''
    
    # Build SLURM header
    constraint_line = f"#SBATCH --constraint={SLURM_CONSTRAINT}\n" if SLURM_CONSTRAINT else ""
    nodelist_line = f"#SBATCH --nodelist={SLURM_NODELIST}\n" if SLURM_NODELIST else ""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --account={SLURM_ACCOUNT}
{constraint_line}{nodelist_line}#SBATCH --qos={SLURM_QOS}
#SBATCH -t {SLURM_TIME}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --cpus-per-task={SLURM_CPUS}
# Logs are useful for tracking download progress and debugging failures
#SBATCH --output={str(LOGS_DIR)}/{JOB_NAME}_%j.out
#SBATCH --error={str(LOGS_DIR)}/{JOB_NAME}_%j.err

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
echo "Job name: {JOB_NAME}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Navigate to project directory first (needed to import config)
cd {PROJECT_ROOT.resolve()}

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
print(f'Data root: {{DATA_ROOT}}')
print(f'Data directories:')
for key, path in DATA_DIRS.items():
    actual_path = path.resolve()
    print(f'  {{key}}: {{actual_path}}')
    if actual_path.exists():
        print(f'    ✓ Directory exists')
    else:
        print(f'    ✗ Directory does not exist')
"

# Run dataset download script
echo ""
echo "Running dataset download script..."
echo "Command: $PYTHON_CMD {str(DOWNLOAD_SCRIPT)} {args_str}"
echo ""

$PYTHON_CMD {str(DOWNLOAD_SCRIPT)} {args_str}

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
print(f'Synthetic benchmarks: {{syn_path.resolve()}}')
if syn_path.exists():
    forrelation = list((syn_path / 'forrelation').glob('*.pt'))
    adding = list((syn_path / 'adding_problem').glob('*.pt'))
    selective = list((syn_path / 'selective_copy').glob('*.pt'))
    print(f'  Forrelation: {{len(forrelation)}} files')
    print(f'  Adding Problem: {{len(adding)}} files')
    print(f'  Selective Copy: {{len(selective)}} files')
else:
    print('  ✗ Directory does not exist')

# DNA dataset
dna_path = get_dna_path()
dna_file = dna_path / 'promoters.data'
print(f'DNA dataset: {{dna_file.resolve()}}')
if dna_file.exists():
    size = dna_file.stat().st_size / 1024  # KB
    print(f'  ✓ File exists ({{size:.1f}} KB)')
else:
    print('  ✗ File does not exist')

# Genomic benchmarks
genomic_path = get_genomic_path()
print(f'Genomic benchmarks: {{genomic_path.resolve()}}')
if genomic_path.exists():
    datasets = [d for d in genomic_path.iterdir() if d.is_dir()]
    print(f'  {{len(datasets)}} datasets found')
    for ds in datasets:
        train_dir = ds / 'train'
        if train_dir.exists():
            print(f'    ✓ {{ds.name}}')
        else:
            print(f'    ✗ {{ds.name}} (incomplete)')
else:
    print('  ✗ Directory does not exist')

# PhysioNet EEG
physionet_path = get_physionet_path()
print(f'PhysioNet EEG: {{physionet_path.resolve()}}')
if physionet_path.exists():
    eegbci = physionet_path / 'files' / 'eegmmidb'
    if eegbci.exists():
        subjects = list(eegbci.glob('*'))
        print(f'  ✓ {{len(subjects)}} subjects found')
    else:
        print('  ✗ Data not downloaded yet')
else:
    print('  ✗ Directory does not exist')
"

exit $EXIT_CODE
"""

    return script_content


def main():
    """Generate SLURM job script for downloading datasets."""
    
    # Create directories
    jobs_dir = JOBS_DIR
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Dataset Download SLURM Job Script")
    print("=" * 80)
    print(f"Job name: {JOB_NAME}")
    print(f"Jobs directory: {jobs_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Python: {PY}")
    print(f"SLURM time limit: {SLURM_TIME}")
    print(f"SLURM CPUs: {SLURM_CPUS}")
    print(f"SLURM GPUs: {SLURM_GPUS}")
    print("=" * 80)
    
    # Create job script (default: download all datasets)
    script_content = create_job_script()
    
    # Save to file
    job_path = jobs_dir / f"{JOB_NAME}.sh"
    
    with open(job_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    job_path.chmod(0o755)
    
    print(f"\n✅ Job script created: {job_path}")
    
    # Create alternative scripts for partial downloads
    print("\nCreating alternative job scripts for partial downloads...")
    
    # Only synthetic datasets
    syn_script = create_job_script(skip_dna=True, skip_genomic=True, skip_physionet=True)
    syn_path = jobs_dir / f"{JOB_NAME}_synthetic_only.sh"
    with open(syn_path, 'w') as f:
        f.write(syn_script)
    syn_path.chmod(0o755)
    print(f"  ✓ {syn_path.name}")
    
    # Only real datasets (skip synthetic)
    real_script = create_job_script(skip_synthetic=True)
    real_path = jobs_dir / f"{JOB_NAME}_real_datasets_only.sh"
    with open(real_path, 'w') as f:
        f.write(real_script)
    real_path.chmod(0o755)
    print(f"  ✓ {real_path.name}")
    
    print("\n" + "=" * 80)
    print("Job Scripts Generation Complete!")
    print("=" * 80)
    print("\nUsage Instructions:")
    print("=" * 80)
    print("1. Submit the main job (download all datasets):")
    print(f"   sbatch {job_path}")
    print()
    print("2. Submit job for synthetic datasets only:")
    print(f"   sbatch {syn_path}")
    print()
    print("3. Submit job for real datasets only (skip synthetic):")
    print(f"   sbatch {real_path}")
    print()
    print("4. Monitor progress:")
    print(f"   tail -f {logs_dir}/{JOB_NAME}_*.out")
    print(f"   tail -f {logs_dir}/{JOB_NAME}_*.err")
    print()
    print("5. Check job status:")
    print("   squeue -u $USER")
    print("=" * 80)


if __name__ == "__main__":
    main()
