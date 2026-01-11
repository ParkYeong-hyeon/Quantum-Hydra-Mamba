#!/usr/bin/env python3
"""
Genomic SSM Comparison Experiment

Compares QuantumMambaSSM, QuantumHydraSSM, ClassicalMamba, and ClassicalHydra
on Genomic Benchmarks datasets (e.g., human_nontata_promoters).

This script runs all four models sequentially with matched configurations
to ensure fair comparison.
"""

import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Import SSM models
from models.QuantumSSM import QuantumMambaSSM, QuantumHydraSSM
from models.TrueClassicalHydra import TrueClassicalHydra
from models.TrueClassicalMamba import TrueClassicalMamba

# Import Genomic Benchmarks loader
from data_loaders.Load_Genomic_Benchmarks import load_genomic_benchmark


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, verbose=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    import sys
    n_batches = len(train_loader)

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Loading data...", flush=True)

        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Forward pass...", flush=True)
        outputs = model(batch_x)

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Computing loss...", flush=True)
        loss = criterion(outputs, batch_y)

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Backward pass...", flush=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Done. Loss={loss.item():.4f}", flush=True)
        elif verbose and batch_idx == 3:
            print(f"    (Suppressing further batch output...)", flush=True)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate additional metrics
    try:
        all_probs = np.array(all_probs)
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def create_model(model_name, n_channels, seq_len, d_model, d_state, n_layers,
                 n_qubits, qlcu_layers, output_dim, dropout, device):
    """Create model based on model_name."""

    # For Genomic Benchmarks: input shape is (batch, 4, seq_len)
    # where 4 is the one-hot encoding dimension (A, C, G, T)

    if model_name == 'quantum_mamba_ssm':
        model = QuantumMambaSSM(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,  # 4 for one-hot
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'quantum_hydra_ssm':
        model = QuantumHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'classical_mamba':
        model = TrueClassicalMamba(
            n_channels=n_channels,
            n_timesteps=seq_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout
        )

    elif model_name == 'classical_hydra':
        model = TrueClassicalHydra(
            n_channels=n_channels,
            n_timesteps=seq_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def train_single_model(
    model_name,
    train_loader,
    val_loader,
    test_loader,
    n_channels,
    seq_len,
    d_model,
    d_state,
    n_layers,
    n_qubits,
    qlcu_layers,
    output_dim,
    n_epochs,
    learning_rate,
    seed,
    device,
    early_stopping_patience=15
):
    """Train a single model and return results."""

    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    # Create model
    model = create_model(
        model_name=model_name,
        n_channels=n_channels,
        seq_len=seq_len,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        n_qubits=n_qubits,
        qlcu_layers=qlcu_layers,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Verbose for first epoch only
        verbose = (epoch == 0)
        if verbose:
            print(f"  Starting epoch {epoch+1}/{n_epochs}...", flush=True)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, verbose=verbose)

        if verbose:
            print(f"  Epoch {epoch+1} training done. Validating...", flush=True)

        # Validate
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Print every epoch for better visibility
        print(f"  Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | AUC: {val_auc:.4f}", flush=True)

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(model, test_loader, criterion, device)

    print(f"\n  Results for {model_name}:")
    print(f"    Parameters: {n_params:,}")
    print(f"    Training time: {training_time:.1f}s ({training_time/60:.2f} min)")
    print(f"    Best Val Acc: {best_val_acc:.4f}")
    print(f"    Test Acc: {test_acc:.4f}")
    print(f"    Test AUC: {test_auc:.4f}")
    print(f"    Test F1: {test_f1:.4f}")

    return {
        'model_name': model_name,
        'n_params': n_params,
        'training_time': training_time,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description="Genomic SSM Comparison Experiment")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="human_nontata_promoters",
                       choices=['human_nontata_promoters', 'human_enhancers_cohn',
                               'human_enhancers_ensembl', 'human_ocr_ensembl',
                               'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
                               'drosophila_enhancers_stark', 'dummy_mouse_enhancers_ensembl'],
                       help="Genomic Benchmarks dataset name")
    parser.add_argument("--max-samples", type=int, default=2000,
                       help="Maximum samples to use (None for all)")

    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--d-state", type=int, default=16, help="SSM state dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--qlcu-layers", type=int, default=2, help="Quantum circuit layers")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--early-stopping", type=int, default=15, help="Early stopping patience")

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./results/genomic_ssm_comparison",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--models", type=str, nargs='+',
                       default=['quantum_mamba_ssm', 'quantum_hydra_ssm',
                               'classical_mamba', 'classical_hydra'],
                       help="Models to run")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("="*80)
    print("GENOMIC SSM COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Seed: {args.seed}")
    print(f"Max samples: {args.max_samples}")
    print(f"d_model: {args.d_model}, d_state: {args.d_state}, n_layers: {args.n_layers}")
    print(f"n_qubits: {args.n_qubits}, qlcu_layers: {args.qlcu_layers}")
    print("="*80)

    # Load data once
    set_seed(args.seed)
    print(f"\nLoading {args.dataset} from Genomic Benchmarks...")
    train_loader, val_loader, test_loader, metadata = load_genomic_benchmark(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        device=str(device),
        max_samples=args.max_samples
    )

    n_channels = metadata['n_channels']  # 4 (one-hot encoding)
    seq_len = metadata['seq_len']
    output_dim = metadata['n_classes']

    print(f"\nData configuration:")
    print(f"  Channels (one-hot): {n_channels}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Output classes: {output_dim}")

    # Run experiments for all models
    all_results = []

    for model_name in args.models:
        try:
            results = train_single_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                n_channels=n_channels,
                seq_len=seq_len,
                d_model=args.d_model,
                d_state=args.d_state,
                n_layers=args.n_layers,
                n_qubits=args.n_qubits,
                qlcu_layers=args.qlcu_layers,
                output_dim=output_dim,
                n_epochs=args.n_epochs,
                learning_rate=args.lr,
                seed=args.seed,
                device=device,
                early_stopping_patience=args.early_stopping
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Params':>10} {'Time (s)':>10} {'Val Acc':>10} {'Test Acc':>10} {'Test AUC':>10}")
    print("-"*80)

    for r in all_results:
        print(f"{r['model_name']:<25} {r['n_params']:>10,} {r['training_time']:>10.1f} "
              f"{r['best_val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['test_auc']:>10.4f}")

    print("="*80)

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"comparison_{args.dataset}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        'config': vars(args),
        'metadata': metadata,
        'results': all_results,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    main()
