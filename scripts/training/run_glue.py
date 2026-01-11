#!/usr/bin/env python3
"""
GLUE Benchmark Training Script for Quantum Models

Trains and evaluates QuantumHydraGLUE, QuantumMambaGLUE, and LSTMBaseline
on GLUE benchmark tasks.

Usage:
    python run_glue.py --task=sst2 --model=quantum_hydra --n-epochs=50

    # Quick test run
    python run_glue.py --task=cola --model=quantum_hydra --mini --n-epochs=5

    # Full experiment with all models
    python run_glue.py --task=sst2 --model=all --n-epochs=100
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr

# Local imports
from Load_GLUE import load_glue_task, GLUE_TASKS
from QuantumHydraGLUE import create_glue_model


# ================================================================================
# Reproducibility
# ================================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================================
# Metrics Computation
# ================================================================================

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    is_regression: bool = False
) -> Dict[str, float]:
    """
    Compute task-specific metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_name: GLUE task name
        is_regression: Whether task is regression

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    if is_regression:
        # STS-B: Pearson and Spearman correlation
        pearson_corr, _ = pearsonr(predictions, labels)
        spearman_corr, _ = spearmanr(predictions, labels)
        metrics['pearson'] = pearson_corr
        metrics['spearman'] = spearman_corr
        metrics['combined'] = (pearson_corr + spearman_corr) / 2
    else:
        # Classification metrics
        pred_labels = predictions.argmax(axis=1) if predictions.ndim > 1 else predictions

        metrics['accuracy'] = accuracy_score(labels, pred_labels)

        if task_name == 'cola':
            # Matthews correlation for CoLA
            metrics['matthews_correlation'] = matthews_corrcoef(labels, pred_labels)
            metrics['primary'] = metrics['matthews_correlation']
        elif task_name in ['mrpc', 'qqp']:
            # F1 for MRPC and QQP
            metrics['f1'] = f1_score(labels, pred_labels, average='binary')
            metrics['precision'] = precision_score(labels, pred_labels, average='binary')
            metrics['recall'] = recall_score(labels, pred_labels, average='binary')
            metrics['primary'] = metrics['f1']
        else:
            # Accuracy for others
            metrics['primary'] = metrics['accuracy']

    return metrics


# ================================================================================
# Training and Evaluation
# ================================================================================

def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        logits = outputs['logits'].detach()

        if logits.shape[-1] == 1:
            # Regression
            preds = logits.squeeze(-1).cpu().numpy()
        else:
            # Classification
            preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)

    # Simple accuracy for progress tracking
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if all_preds.ndim > 1 or np.issubdtype(all_preds.dtype, np.floating):
        # Regression or probabilities
        accuracy = 0.0
    else:
        accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    task_name: str,
    is_regression: bool = False
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.

    Returns:
        avg_loss, metrics_dict, all_predictions, all_labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']

            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)

    if is_regression:
        all_preds = all_logits.squeeze(-1)
    else:
        all_preds = all_logits

    metrics = compute_metrics(all_preds, all_labels, task_name, is_regression)
    metrics['loss'] = avg_loss

    return avg_loss, metrics, all_preds, all_labels


# ================================================================================
# Main Training Loop
# ================================================================================

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    task_name: str,
    is_regression: bool,
    n_epochs: int,
    patience: int,
    output_dir: Path,
    model_name: str,
    seed: int
) -> Dict:
    """
    Full training loop with early stopping.

    Returns:
        Results dictionary
    """
    best_val_metric = -float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_metrics': [],
        'epochs': []
    }

    checkpoint_path = output_dir / f'{model_name}_{task_name}_seed{seed}_best.pt'

    print(f"\nStarting training for {n_epochs} epochs...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, device, task_name, is_regression
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get('primary', val_metrics.get('accuracy', -val_loss)))
            else:
                scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['epochs'].append(epoch)

        # Get primary metric for early stopping
        primary_metric = val_metrics.get('primary', val_metrics.get('accuracy', -val_loss))

        # Print progress
        metric_str = ' '.join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if k != 'loss'])
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {metric_str}")

        # Early stopping check
        if primary_metric > best_val_metric:
            best_val_metric = primary_metric
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_val_metric,
            }, checkpoint_path)
            print(f"  -> New best! Saved to {checkpoint_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {training_time:.2f}s ({training_time/60:.1f} min)")

    # Load best model and evaluate on test set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, device, task_name, is_regression
    )

    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Compile results
    results = {
        'model_name': model_name,
        'task_name': task_name,
        'seed': seed,
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_epoch': checkpoint['epoch'],
        'best_val_metric': best_val_metric,
        'test_metrics': test_metrics,
        'history': history,
        'training_time': training_time,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    return results


# ================================================================================
# Main Function
# ================================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Quantum Models on GLUE Benchmark')

    # Task arguments
    parser.add_argument('--task', type=str, required=True,
                        choices=list(GLUE_TASKS.keys()),
                        help='GLUE task to run')

    # Model arguments
    parser.add_argument('--model', type=str, default='quantum_hydra',
                        choices=['quantum_hydra', 'quantum_mamba', 'lstm_baseline', 'all'],
                        help='Model to train')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--n-qubits', type=int, default=6,
                        help='Number of qubits for quantum models')
    parser.add_argument('--qlcu-layers', type=int, default=2,
                        help='Number of quantum circuit layers')
    parser.add_argument('--chunk-size', type=int, default=16,
                        help='Chunk size for quantum processing')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')

    # Training arguments
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup ratio for learning rate scheduler')

    # Other arguments
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./glue_results',
                        help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--mini', action='store_true',
                        help='Use mini dataset for quick testing')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get task config
    task_config = GLUE_TASKS[args.task]
    num_labels = task_config['num_labels']
    is_regression = task_config.get('is_regression', False)

    print(f"\n{'='*70}")
    print(f"GLUE Task: {task_config['name']}")
    print(f"Type: {'Regression' if is_regression else 'Classification'}")
    print(f"Num Labels: {num_labels}")
    print(f"Metric: {task_config['metric']}")
    print(f"{'='*70}")

    # Load data
    print("\nLoading data...")
    max_train = 1000 if args.mini else None
    max_val = 200 if args.mini else None

    train_loader, val_loader, test_loader, metadata = load_glue_task(
        task_name=args.task,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=max_train,
        max_val_samples=max_val
    )

    vocab_size = metadata['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train samples: {metadata['train_size']}")
    print(f"Val samples: {metadata['val_size']}")
    print(f"Test samples: {metadata['test_size']}")

    # Determine which models to train
    if args.model == 'all':
        models_to_train = ['quantum_hydra', 'quantum_mamba', 'lstm_baseline']
    else:
        models_to_train = [args.model]

    all_results = []

    for model_name in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*70}")

        # Create model
        model = create_glue_model(
            model_name=model_name,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_labels=num_labels,
            max_length=args.max_length,
            n_qubits=args.n_qubits,
            qlcu_layers=args.qlcu_layers,
            chunk_size=args.chunk_size,
            dropout=0.1,
            is_regression=is_regression,
            device=str(device)
        )

        n_params = model.get_num_params()
        print(f"Model parameters: {n_params:,}")

        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        total_steps = len(train_loader) * args.n_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Train
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            task_name=args.task,
            is_regression=is_regression,
            n_epochs=args.n_epochs,
            patience=args.patience,
            output_dir=output_dir,
            model_name=model_name,
            seed=args.seed
        )

        # Add hyperparameters
        results['hyperparameters'] = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'n_qubits': args.n_qubits,
            'qlcu_layers': args.qlcu_layers,
            'chunk_size': args.chunk_size,
            'max_length': args.max_length,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
        }

        all_results.append(results)

        # Save individual results
        results_file = output_dir / f'{model_name}_{args.task}_seed{args.seed}_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = json.loads(
                json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            )
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {results_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Params':>12} {'Test Metric':>15}")
    print("-" * 50)

    for r in all_results:
        primary = r['test_metrics'].get('primary', r['test_metrics'].get('accuracy', 0))
        print(f"{r['model_name']:<20} {r['n_params']:>12,} {primary:>15.4f}")

    # Save combined results
    combined_file = output_dir / f'all_models_{args.task}_seed{args.seed}_summary.json'
    with open(combined_file, 'w') as f:
        summary = {
            'task': args.task,
            'seed': args.seed,
            'results': [
                {
                    'model': r['model_name'],
                    'params': r['n_params'],
                    'test_metrics': r['test_metrics'],
                    'training_time': r['training_time']
                }
                for r in all_results
            ]
        }
        json.dump(summary, f, indent=2)

    print(f"\nCombined summary saved to {combined_file}")


if __name__ == '__main__':
    main()
