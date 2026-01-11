#!/usr/bin/env python
"""
Test EEG Datasets with Quantum Hydra, Mamba, and Transformer Models

This script tests that all three quantum models work correctly with the EEG datasets.
It performs a quick forward pass and training step to verify compatibility.

Author: Junghoon Park
Date: December 2024
"""

import sys
import os
sys.path.insert(0, './') # quantum_hydra_mamba directory

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Set random seeds
def set_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2024)


def test_model_with_data(model, train_loader, model_name, device, num_batches=2):
    """
    Test a model with a few batches of data.

    Args:
        model: The quantum model to test
        train_loader: DataLoader with training data
        model_name: Name for display
        device: torch device
        num_batches: Number of batches to test

    Returns:
        dict: Test results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = {
        'model': model_name,
        'forward_pass': False,
        'backward_pass': False,
        'output_shape': None,
        'loss': None,
        'error': None
    }

    try:
        batch_count = 0
        for batch_x, batch_y in train_loader:
            if batch_count >= num_batches:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Ensure labels are long for CrossEntropyLoss
            if batch_y.dtype == torch.float32:
                batch_y = batch_y.long()

            print(f"\n  Batch {batch_count + 1}:")
            print(f"    Input shape: {batch_x.shape}")
            print(f"    Label shape: {batch_y.shape}, dtype: {batch_y.dtype}")
            print(f"    Label values: {batch_y[:5].tolist()}...")

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)

            print(f"    Output shape: {outputs.shape}")
            results['output_shape'] = tuple(outputs.shape)
            results['forward_pass'] = True

            # Compute loss
            loss = criterion(outputs, batch_y)
            print(f"    Loss: {loss.item():.4f}")
            results['loss'] = loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            results['backward_pass'] = True

            # Check predictions
            preds = outputs.argmax(dim=1)
            acc = (preds == batch_y).float().mean().item()
            print(f"    Accuracy: {acc*100:.1f}%")

            batch_count += 1

        print(f"\n  ✓ {model_name} - SUCCESS")

    except Exception as e:
        results['error'] = str(e)
        print(f"\n  ✗ {model_name} - FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def get_model_param_count(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def main():
    print("="*70)
    print("EEG Dataset + Quantum Model Compatibility Test")
    print("="*70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ========================================================================
    # Load PhysioNet EEG Data
    # ========================================================================
    print("\n" + "="*70)
    print("Loading PhysioNet EEG Dataset")
    print("="*70)

    from data_loaders.Load_PhysioNet_EEG_NoPrompt import load_eeg_ts_revised

    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=2024,
        device=device,
        batch_size=8,  # Small batch for testing
        sampling_freq=80,  # 80 Hz
        sample_size=10  # Only 10 subjects for quick test
    )

    print(f"\nInput dimension: {input_dim}")
    n_samples, n_channels, n_timesteps = input_dim
    print(f"  Channels: {n_channels}")
    print(f"  Timesteps: {n_timesteps}")

    # Model parameters
    feature_dim = n_channels  # 64 channels
    num_classes = 2  # Left/Right
    n_qubits = 4
    n_layers = 2

    print(f"\nModel configuration:")
    print(f"  feature_dim: {feature_dim}")
    print(f"  n_timesteps: {n_timesteps}")
    print(f"  num_classes: {num_classes}")
    print(f"  n_qubits: {n_qubits}")
    print(f"  n_layers: {n_layers}")

    # ========================================================================
    # Test Quantum Models
    # ========================================================================
    all_results = []

    # --- Test 1: QTSQuantumHydraSSMAdvanced ---
    print("\n" + "-"*70)
    print("Loading QTSQuantumHydraSSMAdvanced...")
    try:
        from models.QTSQuantumHydraSSMAdvanced import QTSQuantumHydraSSMAdvanced

        hydra_model = QTSQuantumHydraSSMAdvanced(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=3,
            projection_type='Conv2d_GLU',
            dropout=0.1,
            device=str(device)
        )
        print(f"Parameters: {get_model_param_count(hydra_model):,}")

        result = test_model_with_data(hydra_model, train_loader, "QTSQuantumHydraSSMAdvanced", device)
        all_results.append(result)

    except Exception as e:
        print(f"Failed to load QTSQuantumHydraSSMAdvanced: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({'model': 'QTSQuantumHydraSSMAdvanced', 'error': str(e)})

    # --- Test 2: QTSQuantumMambaSSMAdvanced ---
    print("\n" + "-"*70)
    print("Loading QTSQuantumMambaSSMAdvanced...")
    try:
        from models.QTSQuantumMambaSSMAdvanced import QTSQuantumMambaSSMAdvanced

        mamba_model = QTSQuantumMambaSSMAdvanced(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=3,
            dt_rank="auto",
            projection_type='Conv2d_GLU',
            dropout=0.1,
            device=str(device)
        )
        print(f"Parameters: {get_model_param_count(mamba_model):,}")

        result = test_model_with_data(mamba_model, train_loader, "QTSQuantumMambaSSMAdvanced", device)
        all_results.append(result)

    except Exception as e:
        print(f"Failed to load QTSQuantumMambaSSMAdvanced: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({'model': 'QTSQuantumMambaSSMAdvanced', 'error': str(e)})

    # --- Test 3: QTSQuantumTransformer ---
    print("\n" + "-"*70)
    print("Loading QTSQuantumTransformer...")
    try:
        from models.QTSQuantumTransformer import QTSQuantumTransformer

        transformer_model = QTSQuantumTransformer(
            feature_dim=feature_dim,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            n_qubits=n_qubits,
            n_layers=n_layers,
            qsvt_degree=3,
            projection_type='Conv2d_GLU',
            dropout=0.1,
            device=str(device)
        )
        print(f"Parameters: {get_model_param_count(transformer_model):,}")

        result = test_model_with_data(transformer_model, train_loader, "QTSQuantumTransformer", device)
        all_results.append(result)

    except Exception as e:
        print(f"Failed to load QTSQuantumTransformer: {e}")
        import traceback
        traceback.print_exc()
        all_results.append({'model': 'QTSQuantumTransformer', 'error': str(e)})

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for result in all_results:
        model_name = result['model']
        if result.get('error'):
            print(f"  ✗ {model_name}: FAILED - {result['error'][:50]}...")
        else:
            fwd = "✓" if result['forward_pass'] else "✗"
            bwd = "✓" if result['backward_pass'] else "✗"
            print(f"  {fwd} {model_name}: Forward={fwd}, Backward={bwd}, Loss={result.get('loss', 'N/A'):.4f}")

    # Check overall success
    all_passed = all(r.get('forward_pass') and r.get('backward_pass') for r in all_results if not r.get('error'))
    print("\n" + "="*70)
    if all_passed and len(all_results) == 3:
        print("ALL TESTS PASSED! Models are compatible with PhysioNet EEG data.")
    else:
        print("SOME TESTS FAILED. Please check the errors above.")
    print("="*70)

    return all_results


if __name__ == "__main__":
    results = main()
