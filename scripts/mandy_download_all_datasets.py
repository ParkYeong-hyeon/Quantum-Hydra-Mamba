#!/usr/bin/env python3
"""
Download/Generate all datasets for Quantum-Hydra-Mamba project.

This script downloads or generates:
  - Synthetic Benchmarks (Forrelation, Adding Problem, Selective Copy)
  - DNA Promoter dataset
  - Genomic Benchmarks datasets
  - PhysioNet EEG dataset (via MNE)

Usage:
    python scripts/mandy_download_all_datasets.py
    python scripts/mandy_download_all_datasets.py --skip-synthetic
    python scripts/mandy_download_all_datasets.py --genomic-datasets human_nontata_promoters demo_human_or_worm
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_and_create_dirs():
    """Create necessary data directories."""
    from config.data_paths import DATA_DIRS
    
    print_section("Creating Data Directories")
    for key, path in DATA_DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {key}: {path}")
    print()


def download_synthetic_datasets(data_dir=None, seed=2024):
    """Generate synthetic benchmark datasets.
    
    This function follows the exact logic from generate_all_synthetic_datasets.py:
    - Forrelation: seq_len = [50, 100, 200]
    - Adding Problem: seq_len = [100, 200, 500, 1000]
    - Selective Copy: seq_len = [100, 200, 500, 1000], num_markers=8
    
    Each function checks for existing files individually and skips if they exist.
    """
    print_section("Synthetic Benchmarks")
    
    try:
        # Import from the generate_all_synthetic_datasets module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_all_synthetic_datasets",
            Path(__file__).parent / "generate_all_synthetic_datasets.py"
        )
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        
        from config.data_paths import get_synthetic_path
        
        if data_dir is None:
            data_dir = get_synthetic_path()
        else:
            data_dir = Path(data_dir)
        
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data directory: {data_dir}")
        print(f"Seed: {seed}")
        
        # Follow generate_all_synthetic_datasets.py logic exactly
        # Each function handles its own existence checks and directory creation
        print("\n  📦 Generating Forrelation datasets...")
        gen_module.generate_forrelation_datasets(data_dir, seed)
        
        print("\n  📦 Generating Adding Problem datasets...")
        gen_module.generate_adding_problem_datasets(data_dir, seed)
        
        print("\n  📦 Generating Selective Copy datasets...")
        gen_module.generate_selective_copy_datasets(data_dir, num_markers=8, seed=seed)
        
        print("\n  ✅ Synthetic datasets ready!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error generating synthetic datasets: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_dna_dataset():
    """Download DNA Promoter dataset.
    
    Uses the existing download logic from data_loaders.genomic.Load_DNA_Sequences.download_promoter_dataset()
    which handles:
    - Downloading from UCI repository
    - Creating synthetic data as fallback if download fails
    - File existence checking
    """
    print_section("DNA Promoter Dataset")
    
    try:
        # Use the existing download function from Load_DNA_Sequences
        from data_loaders.genomic.Load_DNA_Sequences import download_promoter_dataset
        from config.data_paths import get_dna_path
        
        # Check if file already exists at expected location
        dna_dir = get_dna_path()
        dna_file = dna_dir / "promoters.data"
        
        if dna_file.exists():
            print(f"  ⏭️  DNA dataset already exists: {dna_file}")
            print("  ✅ DNA dataset ready!")
            return True
        
        # Call the existing download function
        # Note: download_promoter_dataset() uses ./data/dna by default,
        # but we'll let it handle the download logic and then verify
        print("  📥 Downloading DNA Promoter dataset...")
        print("     Using existing download logic from Load_DNA_Sequences...")
        filepath = download_promoter_dataset()
        
        # Verify the file was downloaded
        if filepath.exists():
            print(f"  ✅ Downloaded to: {filepath}")
            return True
        else:
            print(f"  ⚠️  Warning: Download completed but file not found at {filepath}")
            return False
        
    except Exception as e:
        print(f"  ❌ Error downloading DNA dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_genomic_benchmarks(dataset_names=None):
    """Download Genomic Benchmarks datasets.
    
    Uses the same download logic as Load_Genomic_Benchmarks.load_genomic_benchmark():
    - Checks if dataset exists before downloading
    - Uses genomic_benchmarks.loc2seq.download_dataset()
    - Follows the same existence check pattern (dataset_path/train must exist)
    """
    print_section("Genomic Benchmarks")
    
    if dataset_names is None:
        # Default datasets
        dataset_names = [
            'human_nontata_promoters',
            'demo_human_or_worm',
            'demo_coding_vs_intergenomic_seqs'
        ]
    
    try:
        # Use the same import and logic as Load_Genomic_Benchmarks
        from genomic_benchmarks.loc2seq import download_dataset
        from config.data_paths import get_genomic_path
        
        cache_dir = get_genomic_path()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cache directory: {cache_dir}")
        print(f"Datasets to download: {dataset_names}")
        
        # Follow the same logic as Load_Genomic_Benchmarks.load_genomic_benchmark()
        for dataset_name in dataset_names:
            dataset_path = cache_dir / dataset_name
            
            # Same existence check as in Load_Genomic_Benchmarks (line 106)
            if dataset_path.exists() and (dataset_path / 'train').exists():
                print(f"  ⏭️  {dataset_name} already exists, skipping...")
                continue
            
            print(f"  📥 Downloading {dataset_name}...")
            try:
                # Same download call as in Load_Genomic_Benchmarks (line 110)
                download_dataset(dataset_name, dest_path=cache_dir)
                print(f"  ✅ {dataset_name} downloaded successfully!")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not download {dataset_name}: {e}")
                print(f"     This dataset will be downloaded on first use.")
        
        print("  ✅ Genomic Benchmarks ready!")
        return True
        
    except ImportError:
        print("  ⚠️  Warning: genomic_benchmarks package not installed.")
        print("     Install with: pip install genomic-benchmarks")
        print("     Datasets will be downloaded on first use.")
        return False
    except Exception as e:
        print(f"  ❌ Error downloading Genomic Benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_physionet_eeg(num_subjects=5):
    """Download PhysioNet EEG dataset (small subset for testing).
    
    Uses the same download logic as Load_PhysioNet_EEG_NoPrompt.load_eeg_ts_revised():
    - Uses mne.datasets.eegbci.load_data() with update_path=True
    - Uses IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12] runs
    - Uses config.data_paths.get_physionet_path() for data path
    """
    print_section("PhysioNet EEG Dataset")
    
    try:
        import mne
        from config.data_paths import get_physionet_path
        
        data_path = get_physionet_path()
        data_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Data path: {data_path}")
        print(f"MNE version: {mne.__version__}")
        
        # Check if data already exists
        eegbci_path = data_path / "files" / "eegmmidb"
        if eegbci_path.exists() and list(eegbci_path.glob("*")):
            print("  ⏭️  PhysioNet EEG data already exists, skipping download...")
            print("  ✅ PhysioNet EEG ready!")
            return True
        
        print(f"  📥 Downloading PhysioNet EEG data (first {num_subjects} subjects for testing)...")
        print("     Note: This may take a while. Full dataset will be downloaded on first use.")
        
        # Use the same download logic as Load_PhysioNet_EEG_NoPrompt.load_eeg_ts_revised()
        # Same runs and parameters (lines 65-76)
        try:
            IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]  # Same as Load_PhysioNet_EEG_NoPrompt
            test_subjects = list(range(1, min(num_subjects + 1, 6)))  # Max 5 subjects for initial download
            
            for subj_id in test_subjects:
                print(f"     Downloading subject {subj_id}...")
                # Same call as in Load_PhysioNet_EEG_NoPrompt (lines 71-76)
                paths = mne.datasets.eegbci.load_data(
                    subjects=subj_id,
                    runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
                    path=str(data_path),
                    update_path=True  # Same as Load_PhysioNet_EEG_NoPrompt
                )
                print(f"     ✓ Subject {subj_id} downloaded")
            
            print("  ✅ PhysioNet EEG initial download complete!")
            print("     Note: Remaining subjects will be downloaded automatically on first use.")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not download PhysioNet EEG: {e}")
            print("     The dataset will be downloaded automatically on first use.")
            return False
        
    except ImportError:
        print("  ⚠️  Warning: MNE package not installed.")
        print("     Install with: pip install mne")
        print("     Dataset will be downloaded on first use.")
        return False
    except Exception as e:
        print(f"  ❌ Error downloading PhysioNet EEG: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download/Generate all datasets for Quantum-Hydra-Mamba",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/mandy_download_all_datasets.py
  
  # Skip synthetic dataset generation
  python scripts/mandy_download_all_datasets.py --skip-synthetic
  
  # Specify genomic datasets to download
  python scripts/mandy_download_all_datasets.py --genomic-datasets human_nontata_promoters demo_human_or_worm
  
  # Only download specific datasets
  python scripts/mandy_download_all_datasets.py --only synthetic dna
        """
    )
    
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic dataset generation")
    parser.add_argument("--skip-dna", action="store_true",
                        help="Skip DNA dataset download")
    parser.add_argument("--skip-genomic", action="store_true",
                        help="Skip Genomic Benchmarks download")
    parser.add_argument("--skip-physionet", action="store_true",
                        help="Skip PhysioNet EEG download")
    parser.add_argument("--genomic-datasets", nargs="+",
                        default=None,
                        help="List of Genomic Benchmarks datasets to download")
    parser.add_argument("--synthetic-seed", type=int, default=2024,
                        help="Random seed for synthetic dataset generation")
    parser.add_argument("--physionet-subjects", type=int, default=5,
                        help="Number of PhysioNet subjects to download initially")
    parser.add_argument("--only", nargs="+",
                        choices=['synthetic', 'dna', 'genomic', 'physionet'],
                        help="Only download specified datasets")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Quantum-Hydra-Mamba: Dataset Downloader")
    print("=" * 70)
    print()
    
    # Create data directories
    check_and_create_dirs()
    
    results = {}
    start_time = time.time()
    
    # Determine which datasets to download
    download_synthetic = not args.skip_synthetic and (args.only is None or 'synthetic' in args.only)
    download_dna = not args.skip_dna and (args.only is None or 'dna' in args.only)
    download_genomic = not args.skip_genomic and (args.only is None or 'genomic' in args.only)
    download_physionet = not args.skip_physionet and (args.only is None or 'physionet' in args.only)
    
    # Download datasets
    if download_synthetic:
        results['synthetic'] = download_synthetic_datasets(seed=args.synthetic_seed)
    
    if download_dna:
        results['dna'] = download_dna_dataset()
    
    if download_genomic:
        results['genomic'] = download_genomic_benchmarks(args.genomic_datasets)
    
    if download_physionet:
        results['physionet'] = download_physionet_eeg(args.physionet_subjects)
    
    # Summary
    elapsed_time = time.time() - start_time
    print_section("Download Summary")
    
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {dataset:15s}: {status}")
    
    print()
    print(f"Total time: {elapsed_time:.1f} seconds")
    print()
    
    if all(results.values()):
        print("🎉 All datasets are ready!")
    else:
        print("⚠️  Some datasets failed to download.")
        print("   They will be downloaded automatically on first use.")
    
    print()


if __name__ == "__main__":
    main()
