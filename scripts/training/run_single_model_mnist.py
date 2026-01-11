#!/usr/bin/env python3
"""
Single Model Training Script for MNIST Image Classification
Supports all 6 models from EXPERIMENTAL_PLAN_README.md
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

# Import all 6 models
from models.QuantumHydra import QuantumHydraLayer
from models.QuantumHydraHybrid import QuantumHydraHybridLayer
from models.QuantumMamba import QuantumMambaLayer
from models.QuantumMambaHybrid import QuantumMambaHybridLayer
from models.TrueClassicalHydra import TrueClassicalHydra
from models.TrueClassicalMamba import TrueClassicalMamba

# Import data loader (LoadData_MultiChip is in parent directory)
import sys
sys.path.insert(0, '/pscratch/sd/j/junghoon')
from LoadData_MultiChip import load_mnist


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
        optimizer.step()

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
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def create_model(model_name, n_qubits, qlcu_layers, feature_dim, d_model, d_state,
                 output_dim, dropout, device):
    """
    Create model based on model_name.

    Args:
        model_name: One of ['quantum_hydra', 'quantum_hydra_hybrid', 'quantum_mamba',
                           'quantum_mamba_hybrid', 'classical_hydra', 'classical_mamba']
    """
    if model_name == 'quantum_hydra':
        model = QuantumHydraLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            shift_amount=1,
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'quantum_hydra_hybrid':
        model = QuantumHydraHybridLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            shift_amount=1,
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'quantum_mamba':
        model = QuantumMambaLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'quantum_mamba_hybrid':
        model = QuantumMambaHybridLayer(
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

    elif model_name == 'classical_hydra':
        # Treat flattened image as 1-channel time series
        model = TrueClassicalHydra(
            n_channels=1,
            n_timesteps=feature_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=2,
            output_dim=output_dim,
            dropout=dropout
        )

    elif model_name == 'classical_mamba':
        # Treat flattened image as 1-channel time series
        model = TrueClassicalMamba(
            n_channels=1,
            n_timesteps=feature_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=2,
            output_dim=output_dim,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def train_single_model(
    model_name,
    n_qubits,
    qlcu_layers,
    d_model,
    d_state,
    n_epochs,
    batch_size,
    learning_rate,
    n_train,
    n_valtest,
    seed,
    output_dir,
    device,
    early_stopping_patience=10,
    resume_from=None,
    save_checkpoints=True
):
    """
    Train a single model and save results.

    Args:
        early_stopping_patience: Stop training if no improvement for N epochs (0 to disable)
        resume_from: Path to checkpoint file to resume from
        save_checkpoints: Whether to save per-epoch checkpoints

    Returns:
        dict with training history and final test metrics
    """
    set_seed(seed)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    print("="*80)
    print(f"Training Configuration")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device_obj}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train samples: {n_train}")
    print(f"Val+Test samples: {n_valtest}")
    print(f"Early stopping patience: {early_stopping_patience} (0 = disabled)")
    print(f"Resume from: {resume_from if resume_from else 'None'}")
    print("="*80)

    # Load data
    print("\nLoading MNIST Data...")
    train_loader, val_loader, test_loader, feature_dim = load_mnist(
        seed=seed,
        n_train=n_train,
        n_valtest=n_valtest,
        device=device_obj,
        batch_size=batch_size
    )

    # MNIST: 10 classes
    output_dim = 10

    print(f"Data loaded successfully!")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Output classes: {output_dim}")

    # Create model
    print(f"\nCreating model: {model_name}...")
    model = create_model(
        model_name=model_name,
        n_qubits=n_qubits,
        qlcu_layers=qlcu_layers,
        feature_dim=feature_dim,
        d_model=d_model,
        d_state=d_state,
        output_dim=output_dim,
        dropout=0.1,
        device=device_obj
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'epochs': []
    }

    best_val_acc = 0
    best_model_state = None
    start_epoch = 0
    epochs_without_improvement = 0

    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        print(f"\n{'='*40}")
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        best_model_state = checkpoint['best_model_state']
        history = checkpoint['history']
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best val acc so far: {best_val_acc:.4f}")
        print(f"{'='*40}\n")

    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device_obj)

        # Validate
        val_loss, val_acc, val_auc, val_f1, _ = evaluate(model, val_loader, criterion, device_obj)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Check for improvement
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch + 1)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0:
            status = "✓ BEST" if improved else f"({epochs_without_improvement} no improve)"
            print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.2f}s) | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f} | {status}")

        # Save checkpoint if requested
        if save_checkpoints and (epoch + 1) % 10 == 0:
            checkpoint_path = Path(output_dir) / f"{model_name}_seed{seed}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_model_state': best_model_state,
                'history': history,
                'epochs_without_improvement': epochs_without_improvement
            }, checkpoint_path)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"{'='*80}\n")
            break

    total_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(model, test_loader, criterion, device_obj)

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Save results
    results = {
        'model_name': model_name,
        'seed': seed,
        'n_params': n_params,
        'hyperparameters': {
            'n_qubits': n_qubits,
            'qlcu_layers': qlcu_layers,
            'd_model': d_model,
            'd_state': d_state,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_train': n_train,
            'n_valtest': n_valtest
        },
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'training_time': total_time,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results_file = output_path / f"{model_name}_seed{seed}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save model checkpoint
    checkpoint_file = output_path / f"{model_name}_seed{seed}_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'n_params': n_params,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'hyperparameters': results['hyperparameters']
    }, checkpoint_file)
    print(f"Model checkpoint saved to: {checkpoint_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single model on MNIST image classification"
    )

    # Model selection
    parser.add_argument("--model-name", type=str, required=True,
                       choices=['quantum_hydra', 'quantum_hydra_hybrid',
                               'quantum_mamba', 'quantum_mamba_hybrid',
                               'classical_hydra', 'classical_mamba'],
                       help="Model to train")

    # Model hyperparameters
    parser.add_argument("--n-qubits", type=int, default=6,
                       help="Number of qubits (for quantum models)")
    parser.add_argument("--qlcu-layers", type=int, default=2,
                       help="QLCU circuit depth (for quantum models)")
    parser.add_argument("--d-model", type=int, default=128,
                       help="Model dimension (for classical models)")
    parser.add_argument("--d-state", type=int, default=16,
                       help="State dimension (for classical Mamba)")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32, Option 1 modified)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")

    # Data parameters (Option 1 modified - reduced for faster training)
    parser.add_argument("--n-train", type=int, default=500,
                       help="Number of training samples (default: 500, Option 1)")
    parser.add_argument("--n-valtest", type=int, default=250,
                       help="Number of val+test samples (default: 250, Option 1)")

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="./results/mnist_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    # Training control parameters
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Stop if no improvement for N epochs (0 to disable)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--save-checkpoints", action="store_true", default=False,
                       help="Save per-epoch checkpoints every 10 epochs")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MNIST IMAGE CLASSIFICATION - SINGLE MODEL TRAINING")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    results = train_single_model(
        model_name=args.model_name,
        n_qubits=args.n_qubits,
        qlcu_layers=args.qlcu_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_train=args.n_train,
        n_valtest=args.n_valtest,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        resume_from=args.resume_from,
        save_checkpoints=args.save_checkpoints
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final Test Accuracy: {results['test_acc']:.4f}")
    print(f"Final Test AUC: {results['test_auc']:.4f}")
    print("="*80 + "\n")
