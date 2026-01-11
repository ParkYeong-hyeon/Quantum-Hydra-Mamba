#!/usr/bin/env python3
"""
Full Comparison: SSM vs Gated vs Attention vs Classical Models on Genomic Benchmarks

Compares 8 models with fair configurations:
1. QuantumMambaSSM - State-space model for selective information
2. QuantumHydraSSM - Bidirectional SSM with quantum superposition
3. QuantumMambaGated - LSTM-style gates for selective forgetting
4. QuantumHydraGated - Bidirectional gated with quantum superposition
5. QuantumMambaAttention - Self-attention for global mixing
6. QuantumHydraAttention - Bidirectional attention with quantum superposition
7. ClassicalMamba - Classical SSM baseline
8. ClassicalHydra - Classical bidirectional SSM baseline

All quantum models use same n_qubits and qlcu_layers for fair comparison.

Features:
- Checkpoint saving/loading for resumable training
- Experiment-level checkpoints to skip completed models
- Model-level checkpoints to resume mid-training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import argparse
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Import all models
from models.QuantumSSM import QuantumMambaSSM, QuantumHydraSSM
from models.QuantumGatedRecurrence import QuantumMambaGated, QuantumHydraGated
from models.QuantumAttention import QuantumMambaAttention, QuantumHydraAttention
from models.TrueClassicalMamba import TrueClassicalMamba
from models.TrueClassicalHydra import TrueClassicalHydra

# Import Genomic Benchmarks loader
from data_loaders.Load_Genomic_Benchmarks import load_genomic_benchmark


def get_checkpoint_dir(output_dir, dataset, seed):
    """Get checkpoint directory for this experiment."""
    checkpoint_dir = Path(output_dir) / 'checkpoints' / f'{dataset}_seed{seed}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_model_checkpoint_path(checkpoint_dir, model_name):
    """Get checkpoint path for a specific model."""
    return checkpoint_dir / f'{model_name}_checkpoint.pt'


def get_experiment_state_path(checkpoint_dir):
    """Get path for experiment state file."""
    return checkpoint_dir / 'experiment_state.json'


def save_model_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch,
                          best_val_acc, best_model_state, history, epochs_without_improvement):
    """Save model checkpoint for resumable training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'best_model_state': best_model_state,
        'history': history,
        'epochs_without_improvement': epochs_without_improvement
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"    [Checkpoint saved: epoch {epoch+1}]", flush=True)


def load_model_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model checkpoint for resuming training."""
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'best_val_acc': checkpoint['best_val_acc'],
        'best_model_state': checkpoint['best_model_state'],
        'history': checkpoint['history'],
        'epochs_without_improvement': checkpoint['epochs_without_improvement']
    }


def save_experiment_state(state_path, completed_models, results):
    """Save experiment-level state (completed models and their results)."""
    state = {
        'completed_models': completed_models,
        'results': results
    }
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  [Experiment state saved: {len(completed_models)} models completed]", flush=True)


def load_experiment_state(state_path):
    """Load experiment-level state."""
    if not state_path.exists():
        return {'completed_models': [], 'results': []}

    with open(state_path, 'r') as f:
        state = json.load(f)
    return state


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

    n_batches = len(train_loader)
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Forward pass...", flush=True)

        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Backward pass...", flush=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

        if verbose and batch_idx < 3:
            print(f"    Batch {batch_idx+1}/{n_batches}: Done. Loss={loss.item():.4f}", flush=True)
        elif verbose and batch_idx == 3:
            print(f"    (Suppressing further batch output...)", flush=True)

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
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

    try:
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def create_model(model_name, config, device):
    """Create model based on name and config."""

    n_qubits = config['n_qubits']
    qlcu_layers = config['qlcu_layers']
    d_model = config['d_model']
    d_state = config['d_state']
    n_layers = config['n_layers']
    n_channels = config['n_channels']
    seq_len = config['seq_len']
    n_classes = config['n_classes']
    chunk_size = config['chunk_size']
    dropout = config['dropout']

    if model_name == 'quantum_mamba_ssm':
        model = QuantumMambaSSM(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=n_classes,
            dropout=dropout,
            chunk_size=chunk_size,
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
            output_dim=n_classes,
            dropout=dropout,
            chunk_size=chunk_size,
            device=device
        )
    elif model_name == 'quantum_mamba_gated':
        model = QuantumMambaGated(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=d_model,
            output_dim=n_classes,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )
    elif model_name == 'quantum_hydra_gated':
        model = QuantumHydraGated(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=d_model,
            output_dim=n_classes,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )
    elif model_name == 'quantum_mamba_attention':
        model = QuantumMambaAttention(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=d_model,
            output_dim=n_classes,
            n_heads=4,
            n_layers=n_layers,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )
    elif model_name == 'quantum_hydra_attention':
        model = QuantumHydraAttention(
            n_qubits=n_qubits,
            n_timesteps=seq_len,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=d_model,
            output_dim=n_classes,
            n_heads=4,
            n_layers=n_layers,
            chunk_size=chunk_size,
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
            output_dim=n_classes,
            dropout=dropout,
            device=device
        )
    elif model_name == 'classical_hydra':
        model = TrueClassicalHydra(
            n_channels=n_channels,
            n_timesteps=seq_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=n_classes,
            dropout=dropout,
            device=device
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def train_model(model_name, config, train_loader, val_loader, test_loader,
                n_epochs, lr, device, early_stopping=10, verbose=True,
                checkpoint_path=None):
    """Train a single model and return results. Supports checkpointing for resume."""

    print(f"\n{'='*60}", flush=True)
    print(f"Training: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Create model
    model = create_model(model_name, config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop state
    start_epoch = 0
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    accumulated_time = 0  # Track time from previous runs

    # Try to load checkpoint for resuming
    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"  Loading checkpoint from {checkpoint_path}...", flush=True)
        checkpoint_data = load_model_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        if checkpoint_data is not None:
            start_epoch = checkpoint_data['epoch'] + 1  # Resume from next epoch
            best_val_acc = checkpoint_data['best_val_acc']
            best_model_state = checkpoint_data['best_model_state']
            history = checkpoint_data['history']
            epochs_without_improvement = checkpoint_data['epochs_without_improvement']
            # Estimate accumulated time from history
            accumulated_time = len(history['train_loss']) * 500  # rough estimate
            print(f"  Resuming from epoch {start_epoch + 1}/{n_epochs} (best val acc: {best_val_acc:.4f})", flush=True)

    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Show verbose output for first epoch of this run
        is_first_epoch_of_run = (epoch == start_epoch)
        if is_first_epoch_of_run:
            print(f"  Starting epoch {epoch+1}/{n_epochs}...", flush=True)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            verbose=is_first_epoch_of_run
        )

        if is_first_epoch_of_run:
            print(f"  Epoch {epoch+1} training done. Validating...", flush=True)

        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        # Print every epoch
        print(f"  Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | AUC: {val_auc:.4f}", flush=True)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save checkpoint after each epoch
        if checkpoint_path is not None:
            save_model_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch,
                best_val_acc, best_model_state, history, epochs_without_improvement
            )

        # Check early stopping after saving checkpoint
        if epochs_without_improvement >= early_stopping:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    training_time = time.time() - start_time + accumulated_time

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  Results for {model_name}:", flush=True)
    print(f"    Parameters: {n_params:,}", flush=True)
    print(f"    Training time: {training_time:.1f}s ({training_time/60:.2f} min)", flush=True)
    print(f"    Best Val Acc: {best_val_acc:.4f}", flush=True)
    print(f"    Test Acc: {test_acc:.4f}", flush=True)
    print(f"    Test AUC: {test_auc:.4f}", flush=True)
    print(f"    Test F1: {test_f1:.4f}", flush=True)

    # Clean up checkpoint after successful completion
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  [Checkpoint removed after successful completion]", flush=True)

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


def run_comparison(args):
    """Run full comparison experiment with checkpoint support for resuming."""

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # All 8 models (6 quantum + 2 classical)
    all_models = [
        'quantum_mamba_ssm',
        'quantum_hydra_ssm',
        'quantum_mamba_gated',
        'quantum_hydra_gated',
        'quantum_mamba_attention',
        'quantum_hydra_attention',
        'classical_mamba',
        'classical_hydra'
    ]

    # Filter models if specified
    if args.models:
        models_to_run = [m for m in args.models if m in all_models]
    else:
        models_to_run = all_models

    # Setup checkpoint directory
    checkpoint_dir = get_checkpoint_dir(args.output_dir, args.dataset, args.seed)
    experiment_state_path = get_experiment_state_path(checkpoint_dir)

    # Load experiment state (completed models and their results)
    experiment_state = load_experiment_state(experiment_state_path)
    completed_models = experiment_state['completed_models']
    results = experiment_state['results']

    print("=" * 80, flush=True)
    print("FULL SSM vs GATED vs ATTENTION vs CLASSICAL COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Models: {models_to_run}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Max samples: {args.max_samples}", flush=True)
    print(f"Epochs: {args.n_epochs}", flush=True)
    print(f"n_qubits: {args.n_qubits}, qlcu_layers: {args.qlcu_layers}", flush=True)
    print(f"d_model: {args.d_model}, d_state: {args.d_state}, n_layers: {args.n_layers}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    if completed_models:
        print(f"Resuming: {len(completed_models)} models already completed: {completed_models}", flush=True)
    print("=" * 80, flush=True)

    # Load data
    print(f"\nLoading {args.dataset} from Genomic Benchmarks...", flush=True)
    train_loader, val_loader, test_loader, metadata = load_genomic_benchmark(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples
    )

    print(f"\nData configuration:", flush=True)
    print(f"  Channels (one-hot): {metadata['n_channels']}", flush=True)
    print(f"  Sequence length: {metadata['seq_len']}", flush=True)
    print(f"  Output classes: {metadata['n_classes']}", flush=True)

    # Model config (same for all models for fair comparison)
    config = {
        'n_qubits': args.n_qubits,
        'qlcu_layers': args.qlcu_layers,
        'd_model': args.d_model,
        'd_state': args.d_state,
        'n_layers': args.n_layers,
        'n_channels': metadata['n_channels'],
        'seq_len': metadata['seq_len'],
        'n_classes': metadata['n_classes'],
        'chunk_size': args.chunk_size,
        'dropout': args.dropout
    }

    # Run comparison
    for model_name in models_to_run:
        # Skip already completed models
        if model_name in completed_models:
            print(f"\n[Skipping {model_name} - already completed]", flush=True)
            continue

        try:
            # Get checkpoint path for this model
            model_checkpoint_path = get_model_checkpoint_path(checkpoint_dir, model_name)

            result = train_model(
                model_name=model_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                n_epochs=args.n_epochs,
                lr=args.lr,
                device=device,
                early_stopping=args.early_stopping,
                verbose=True,
                checkpoint_path=model_checkpoint_path
            )
            results.append(result)

            # Mark model as completed and save experiment state
            completed_models.append(model_name)
            save_experiment_state(experiment_state_path, completed_models, results)

        except Exception as e:
            print(f"\nError training {model_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Model':<30} {'Params':>10} {'Time (s)':>10} {'Val Acc':>10} {'Test Acc':>10} {'Test AUC':>10}", flush=True)
    print("-" * 80, flush=True)

    for r in results:
        print(f"{r['model_name']:<30} {r['n_params']:>10,} {r['training_time']:>10.1f} "
              f"{r['best_val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['test_auc']:>10.4f}", flush=True)

    print("=" * 80, flush=True)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_comparison_{args.dataset}_seed{args.seed}_{timestamp}.json"

    output_data = {
        'config': {
            'dataset': args.dataset,
            'max_samples': args.max_samples,
            'd_model': args.d_model,
            'd_state': args.d_state,
            'n_layers': args.n_layers,
            'n_qubits': args.n_qubits,
            'qlcu_layers': args.qlcu_layers,
            'chunk_size': args.chunk_size,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'early_stopping': args.early_stopping,
            'seed': args.seed,
            'output_dir': args.output_dir,
            'device': args.device,
            'models': models_to_run
        },
        'metadata': metadata,
        'results': results,
        'timestamp': timestamp
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description='Full SSM vs Gated vs Classical Comparison')

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="demo_human_or_worm",
                       choices=['human_nontata_promoters', 'human_enhancers_cohn',
                               'human_enhancers_ensembl', 'human_ocr_ensembl',
                               'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
                               'drosophila_enhancers_stark', 'dummy_mouse_enhancers_ensembl'],
                       help="Genomic Benchmark dataset name")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Maximum samples (for faster testing)")

    # Model architecture (same for all models)
    parser.add_argument("--n-qubits", type=int, default=4,
                       help="Number of qubits (same for all quantum models)")
    parser.add_argument("--qlcu-layers", type=int, default=2,
                       help="QLCU circuit layers (same for all quantum models)")
    parser.add_argument("--d-model", type=int, default=64,
                       help="Model dimension / hidden dimension")
    parser.add_argument("--d-state", type=int, default=16,
                       help="SSM state dimension")
    parser.add_argument("--n-layers", type=int, default=1,
                       help="Number of layers")
    parser.add_argument("--chunk-size", type=int, default=16,
                       help="Chunk size for processing")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")

    # Training arguments
    parser.add_argument("--n-epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--early-stopping", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results/full_comparison",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")

    # Model selection
    parser.add_argument("--models", type=str, nargs='+', default=None,
                       help="Specific models to run (default: all 6)")

    args = parser.parse_args()

    run_comparison(args)


if __name__ == "__main__":
    main()
