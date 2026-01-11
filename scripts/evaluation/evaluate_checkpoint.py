#!/usr/bin/env python3
"""
Quick evaluation script for saved checkpoints
Loads a saved model checkpoint and evaluates on test set
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.QuantumGatedRecurrence import QuantumHydraGated, QuantumMambaGated
from data_loaders.Load_Genomic_Benchmarks import load_genomic_benchmark


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def main():
    # Configuration
    model_name = "quantum_hydra_gated"
    dataset_name = "dummy_mouse_enhancers_ensembl"
    seed = 2025

    # Paths (relative to quantum_hydra_mamba directory)
    base_dir = Path(__file__).parent.parent
    checkpoint_path = base_dir / "results" / "genomic" / \
                      f"{model_name}_{dataset_name}_seed{seed}_best.pt"
    output_dir = base_dir / "results" / "genomic"

    print("=" * 80)
    print("EVALUATING SAVED CHECKPOINT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Seed: {seed}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset (same as training script)
    print(f"Loading {dataset_name} from Genomic Benchmarks...")
    train_loader, val_loader, test_loader, metadata = load_genomic_benchmark(
        dataset_name=dataset_name,
        batch_size=16,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )

    num_classes = metadata['n_classes']
    print(f"Test set loader created")
    print(f"Number of classes: {num_classes}\n")

    # Create model
    print(f"Creating {model_name}...")
    feature_dim = 4  # One-hot encoded DNA (A, C, G, T)

    print(f"Model parameters:")
    print(f"  n_qubits: 6")
    print(f"  n_timesteps: 4776")
    print(f"  qlcu_layers: 2")
    print(f"  feature_dim: {feature_dim}")
    print(f"  hidden_dim: 64")
    print(f"  output_dim: {num_classes}")
    print(f"  chunk_size: 64")
    print(f"  dropout: 0.1")
    print(f"  device: {device_str}")

    if model_name == "quantum_hydra_gated":
        model = QuantumHydraGated(
            n_qubits=6,
            n_timesteps=4776,  # Max sequence length
            qlcu_layers=2,
            feature_dim=feature_dim,
            hidden_dim=64,
            output_dim=num_classes,
            chunk_size=64,
            dropout=0.1,
            device=device_str
        )
    else:
        model = QuantumMambaGated(
            n_qubits=6,
            n_timesteps=4776,  # Max sequence length
            qlcu_layers=2,
            feature_dim=feature_dim,
            hidden_dim=64,
            output_dim=num_classes,
            chunk_size=64,
            dropout=0.1,
            device=device_str
        )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print("✓ Checkpoint loaded successfully")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(
        model, test_loader, criterion, device
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_cm)

    # Save results
    results = {
        'model_name': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'n_params': n_params,
        'hyperparameters': {
            'n_qubits': 6,
            'qlcu_layers': 2,
            'hidden_dim': 64,
            'chunk_size': 64,
            'n_epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001
        },
        'history': {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'epochs': []
        },
        'best_val_acc': None,  # Not available from checkpoint
        'test_acc': float(test_acc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'test_cm': test_cm.tolist(),
        'training_time': None,  # Not available
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'note': 'Evaluated from saved checkpoint (training was cancelled at epoch 39 due to time limit)'
    }

    output_file = output_dir / f"{model_name}_{dataset_name}_seed{seed}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
