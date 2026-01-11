#!/usr/bin/env python3
"""
Training Script for Quantum Gated Recurrence Models on Genomic Benchmarks
Supports QuantumMambaGated and QuantumHydraGated on various genomic datasets.
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
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Import gated models
from models.QuantumGatedRecurrence import QuantumMambaGated, QuantumHydraGated

# Import Genomic Benchmarks loader
from data_loaders.Load_Genomic_Benchmarks import load_genomic_benchmark


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

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


def create_model(model_name, n_qubits, qlcu_layers, n_channels, n_timesteps,
                 hidden_dim, output_dim, chunk_size, dropout, device):
    """Create gated model."""

    if model_name == 'quantum_mamba_gated':
        model = QuantumMambaGated(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )
    elif model_name == 'quantum_hydra_gated':
        model = QuantumHydraGated(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=qlcu_layers,
            feature_dim=n_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Quantum Gated Models on Genomic Benchmarks')

    # Model arguments
    parser.add_argument('--model-name', type=str, required=True,
                        choices=['quantum_mamba_gated', 'quantum_hydra_gated'])
    parser.add_argument('--n-qubits', type=int, default=6)
    parser.add_argument('--qlcu-layers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=16)

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['human_nontata_promoters', 'human_enhancers_cohn',
                                 'human_enhancers_ensembl', 'human_ocr_ensembl',
                                 'demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm',
                                 'drosophila_enhancers_stark', 'dummy_mouse_enhancers_ensembl'])
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to use (None for all)')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)

    # Training arguments
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output-dir', type=str, default='./results/genomic')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    # Load data
    print(f"Loading {args.dataset} from Genomic Benchmarks...")
    train_loader, val_loader, test_loader, metadata = load_genomic_benchmark(
        dataset_name=args.dataset,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples
    )

    # Get dimensions from metadata
    # Data shape: (batch, 4, seq_len) - need to treat as (batch, channels=4, timesteps=seq_len)
    n_channels = metadata['n_channels']  # 4 (one-hot)
    n_timesteps = metadata['seq_len']
    output_dim = metadata['n_classes']

    print(f"\nData configuration:")
    print(f"  Channels (one-hot): {n_channels}")
    print(f"  Sequence length: {n_timesteps}")
    print(f"  Output classes: {output_dim}")

    # Create model
    print(f"\nCreating {args.model_name}...")
    model = create_model(
        model_name=args.model_name,
        n_qubits=args.n_qubits,
        qlcu_layers=args.qlcu_layers,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        chunk_size=args.chunk_size,
        dropout=0.1,
        device=device
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'epochs': []
    }

    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()

    print(f"\nStarting training for {args.n_epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.n_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_acc)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch)

        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(),
                      output_dir / f'{args.model_name}_{args.dataset}_seed{args.seed}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {training_time:.2f}s")

    # Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(output_dir / f'{args.model_name}_{args.dataset}_seed{args.seed}_best.pt')
    )

    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{test_cm}")

    # Save results
    results = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'seed': args.seed,
        'n_params': n_params,
        'data_info': metadata,
        'hyperparameters': {
            'n_qubits': args.n_qubits,
            'qlcu_layers': args.qlcu_layers,
            'hidden_dim': args.hidden_dim,
            'chunk_size': args.chunk_size,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
        },
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'training_time': training_time,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    results_file = output_dir / f'{args.model_name}_{args.dataset}_seed{args.seed}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
