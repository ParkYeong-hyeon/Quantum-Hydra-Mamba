#!/usr/bin/env python3
"""
Synthetic Benchmark Runner for Long-Range Sequence Learning
Supports Forrelation, Adding Problem, and Selective Copy tasks

Models tested (12 selected models):
  Group 1 (Quantum Features → Classical Mixing):
    1a: QuantumTransformer, 1b: QuantumMambaSSM, 1c: QuantumHydraSSM

  Group 2 (Classical Features → Quantum Mixing):
    2a: ClassicalQuantumAttention
    2d: QuantumMambaHydraSSM (True Superposition), 2e: QuantumHydraHydraSSM (True Superposition)

  Group 3 (Classical Features → Classical Mixing - Baseline):
    3a: ClassicalTransformer, 3b: TrueClassicalMamba, 3c: TrueClassicalHydra

  Group 4 (Quantum Features → Quantum Mixing - E2E):
    4a: QuantumTransformerE2E
    4d: QuantumMambaE2E_Superposition, 4e: QuantumHydraE2E_Superposition

Usage:
    python run_synthetic_benchmark.py --model-id 1c --task forrelation --seq-len 200 --seed 2024
    python run_synthetic_benchmark.py --model-id 2e --task adding_problem --seq-len 500 --seed 2025
    python run_synthetic_benchmark.py --model-id 3b --task selective_copy --seq-len 1000 --seed 2026
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force Pennylane to use default.qubit (set before importing pennylane)
os.environ["PENNYLANE_DEVICE"] = "default.qubit"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix


# ============================================
# JSON Serialization Helper
# ============================================
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# ============================================
# Model ID to Class Mapping
# ============================================
MODEL_REGISTRY = {
    # Group 1: Quantum Features → Classical Mixing
    '1a': {'name': 'QuantumTransformer', 'group': 1, 'feat': 'quantum', 'mix': 'classical', 'type': 'transformer'},
    '1b': {'name': 'QuantumMambaSSM', 'group': 1, 'feat': 'quantum', 'mix': 'classical', 'type': 'mamba'},
    '1c': {'name': 'QuantumHydraSSM', 'group': 1, 'feat': 'quantum', 'mix': 'classical', 'type': 'hydra'},

    # Group 2: Classical Features → Quantum Mixing
    '2a': {'name': 'ClassicalQuantumAttention', 'group': 2, 'feat': 'classical', 'mix': 'quantum', 'type': 'transformer'},
    '2d': {'name': 'QuantumMambaHydraSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_superposition', 'type': 'mamba'},
    '2e': {'name': 'QuantumHydraHydraSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_superposition', 'type': 'hydra'},

    # Group 3: Classical Features → Classical Mixing (Baseline)
    '3a': {'name': 'ClassicalTransformer', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'transformer'},
    '3b': {'name': 'TrueClassicalMamba', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'mamba'},
    '3c': {'name': 'TrueClassicalHydra', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'hydra'},

    # Group 4: Quantum Features → Quantum Mixing (E2E)
    '4a': {'name': 'QuantumTransformerE2E', 'group': 4, 'feat': 'quantum', 'mix': 'quantum', 'type': 'transformer'},
    '4d': {'name': 'QuantumMambaE2ESuperposition', 'group': 4, 'feat': 'quantum_superposition', 'mix': 'quantum_superposition', 'type': 'mamba'},
    '4e': {'name': 'QuantumHydraE2ESuperposition', 'group': 4, 'feat': 'quantum_superposition', 'mix': 'quantum_superposition', 'type': 'hydra'},
}

# Task configurations
TASK_CONFIGS = {
    'forrelation': {
        'type': 'classification',
        'n_classes': 2,
        'metric': 'accuracy',
        'baseline': 0.5,
    },
    'adding_problem': {
        'type': 'regression',
        'output_dim': 1,
        'metric': 'mse',
        'baseline': 0.167,
    },
    'selective_copy': {
        'type': 'regression',
        'output_dim': 8,  # num_markers
        'metric': 'mse',
        'baseline': 0.083,
    }
}


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import pennylane as qml
        qml.numpy.random.seed(seed)
    except:
        pass


def get_checkpoint_path(output_dir, model_id, task, seq_len, seed):
    """Get path for checkpoint file."""
    output_path = Path(output_dir) / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"checkpoint_{model_id}_{task}_L{seq_len}_seed{seed}.pt"


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, history,
                    best_metric, best_model_state, epochs_without_improvement):
    """Save training checkpoint for resume capability."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_metric': best_metric,
        'best_model_state': best_model_state,
        'epochs_without_improvement': epochs_without_improvement,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training checkpoint for resume."""
    if not checkpoint_path.exists():
        return None

    print(f"\n{'=' * 60}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'=' * 60}")
    print(f"Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Resumed from epoch {checkpoint['epoch'] + 1}")
    print(f"Best metric so far: {checkpoint['best_metric']:.4f}")
    print(f"{'=' * 60}\n")

    return checkpoint


def load_data(task, seq_len, batch_size, num_markers=8, seed=2024, data_dir="./data/synthetic_benchmarks"):
    """Load dataset for the specified task."""
    data_dir = Path(data_dir)

    if task == 'forrelation':
        from data_loaders.forrelation_dataloader import get_forrelation_dataloader

        dataset_path = data_dir / "forrelation" / f"forrelation_L{seq_len}_seed{seed}.pt"

        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}. Generating...")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            # Use V2 generator: BQP-complete Forrelation (Gaussian rounding method)
            # This fixes the data leakage bug in V1 where delta functions were used
            from data_loaders.generate_forrelation_dataset_v2 import generate_dataset as generate_forrelation_dataset
            generate_forrelation_dataset(
                num_pairs=5000,
                n_bits=6,
                seq_len=seq_len,
                seed=seed,
                filename=str(dataset_path)
            )

        train_loader, test_loader, params = get_forrelation_dataloader(
            dataset_path=str(dataset_path),
            batch_size=batch_size,
            shuffle=True
        )
        # Create val loader from test (split 50/50)
        val_loader = test_loader
        n_channels = params['num_channels']
        output_dim = 2  # binary classification

    elif task == 'adding_problem':
        from data_loaders.adding_problem_dataloader import get_adding_problem_dataloader

        dataset_path = data_dir / "adding_problem" / f"adding_L{seq_len}_seed{seed}.pt"

        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}. Generating...")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            from data_loaders.generate_adding_problem import generate_adding_problem_dataset
            generate_adding_problem_dataset(
                num_samples=5000,
                seq_len=seq_len,
                marker_strategy="extremes",
                filename=str(dataset_path),
                seed=seed
            )

        train_loader, val_loader, test_loader, params = get_adding_problem_dataloader(
            dataset_path=str(dataset_path),
            batch_size=batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed
        )
        n_channels = params['num_channels']
        output_dim = 1  # regression

    elif task == 'selective_copy':
        from data_loaders.selective_copy_dataloader import get_selective_copy_dataloader

        dataset_path = data_dir / "selective_copy" / f"selective_copy_L{seq_len}_M{num_markers}_seed{seed}.pt"

        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}. Generating...")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            from data_loaders.generate_selective_copy import generate_selective_copy_dataset
            generate_selective_copy_dataset(
                num_samples=5000,
                seq_len=seq_len,
                num_markers=num_markers,
                marker_strategy="uniform",
                filename=str(dataset_path),
                seed=seed
            )

        train_loader, val_loader, test_loader, params = get_selective_copy_dataloader(
            dataset_path=str(dataset_path),
            batch_size=batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed
        )
        n_channels = params['num_channels']
        output_dim = num_markers  # multi-output regression

    else:
        raise ValueError(f"Unknown task: {task}")

    return train_loader, val_loader, test_loader, n_channels, seq_len, output_dim


def create_model(model_id, n_channels, n_timesteps, n_qubits, n_layers,
                 d_model, d_state, output_dim, dropout, device, task_type='classification'):
    """Create model based on model_id."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}. Valid options: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_id]
    model_name = model_info['name']

    print(f"Creating model: {model_name} (ID: {model_id})")
    print(f"  Group {model_info['group']}: {model_info['feat']} features → {model_info['mix']} mixing ({model_info['type']})")

    device_str = str(device) if hasattr(device, '__str__') else device

    # ========================================
    # Group 1: Quantum Features → Classical Mixing
    # ========================================
    if model_id == '1a':
        from models.QuantumTransformer import QuantumTransformer
        model = QuantumTransformer(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '1b':
        from models.QuantumSSM import QuantumMambaSSM
        model = QuantumMambaSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '1c':
        from models.QuantumSSM import QuantumHydraSSM
        model = QuantumHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # ========================================
    # Group 2: Classical Features → Quantum Mixing
    # ========================================
    elif model_id == '2a':
        from models.QuantumMixingSSM import ClassicalQuantumAttention
        model = ClassicalQuantumAttention(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '2d':
        from models.QuantumHydraSSM import QuantumMambaHydraSSM
        model = QuantumMambaHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '2e':
        from models.QuantumHydraSSM import QuantumHydraHydraSSM
        model = QuantumHydraHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # ========================================
    # Group 3: Classical Features → Classical Mixing (Baselines)
    # ========================================
    elif model_id == '3a':
        from models.ClassicalTransformer import ClassicalTransformer
        model = ClassicalTransformer(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_heads=4,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '3b':
        from models.TrueClassicalMamba import TrueClassicalMamba
        model = TrueClassicalMamba(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '3c':
        from models.TrueClassicalHydra import TrueClassicalHydra
        model = TrueClassicalHydra(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # ========================================
    # Group 4: Quantum Features → Quantum Mixing (E2E)
    # ========================================
    elif model_id == '4a':
        from models.QuantumE2E import QuantumTransformerE2E
        model = QuantumTransformerE2E(
            n_qubits=n_qubits,
            n_feature_layers=n_layers,
            n_mixing_layers=n_layers,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '4d':
        from models.QuantumE2E_Superposition import QuantumMambaE2E_Superposition
        model = QuantumMambaE2E_Superposition(
            n_qubits=n_qubits,
            n_feature_layers=n_layers,
            n_mixing_layers=n_layers,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '4e':
        from models.QuantumE2E_Superposition import QuantumHydraE2E_Superposition
        model = QuantumHydraE2E_Superposition(
            n_qubits=n_qubits,
            n_feature_layers=n_layers,
            n_mixing_layers=n_layers,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    else:
        raise ValueError(f"Model {model_id} not implemented")

    return model.to(device)


def train_epoch_classification(model, train_loader, criterion, optimizer, device):
    """Train for one epoch (classification task)."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        if outputs.device != batch_y.device:
            outputs = outputs.to(batch_y.device)

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


def train_epoch_regression(model, train_loader, criterion, optimizer, device):
    """Train for one epoch (regression task)."""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        if outputs.device != batch_y.device:
            outputs = outputs.to(batch_y.device)

        # Ensure output shape matches target
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        if batch_y.dim() == 1:
            batch_y = batch_y.unsqueeze(1)

        loss = criterion(outputs, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss, avg_loss  # Return loss as both metrics for regression


def evaluate_classification(model, data_loader, criterion, device):
    """Evaluate model (classification)."""
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

            if outputs.device != batch_y.device:
                outputs = outputs.to(batch_y.device)

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

    return avg_loss, accuracy, auc, f1


def evaluate_regression(model, data_loader, criterion, device):
    """Evaluate model (regression)."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)

            if outputs.device != batch_y.device:
                outputs = outputs.to(batch_y.device)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)

            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_samples  # MSE

    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = np.abs(all_preds - all_targets).mean()

    # R² score
    ss_res = ((all_targets - all_preds) ** 2).sum()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return avg_loss, mae, r2


def train_model(
    model_id,
    task,
    seq_len,
    n_qubits,
    n_layers,
    d_model,
    d_state,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    num_markers,
    seed,
    output_dir,
    data_dir,
    device,
    early_stopping_patience=20,
    resume=False,
):
    """Train a single model on synthetic benchmark."""
    set_seed(seed)

    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device_obj = torch.device("cpu")
        print("Using CPU")

    model_info = MODEL_REGISTRY[model_id]
    model_name = model_info['name']
    task_config = TASK_CONFIGS[task]

    print("=" * 80)
    print(f"SYNTHETIC BENCHMARK - {task.upper()}")
    print("=" * 80)
    print(f"Model ID: {model_id} ({model_name})")
    print(f"  Group {model_info['group']}: {model_info['feat']} feat → {model_info['mix']} mix ({model_info['type']})")
    print(f"Task: {task} ({task_config['type']})")
    print(f"Sequence Length: {seq_len}")
    print(f"Seed: {seed}")
    print(f"Device: {device_obj}")
    print("-" * 80)
    print(f"Hyperparameters:")
    print(f"  n_qubits={n_qubits}, n_layers={n_layers}")
    print(f"  d_model={d_model}, d_state={d_state}")
    print(f"  epochs={n_epochs}, batch_size={batch_size}")
    print(f"  lr={learning_rate}, weight_decay={weight_decay}")
    print(f"  early_stopping={early_stopping_patience}")
    print("=" * 80)

    # Load data
    print("\nLoading Data...", flush=True)
    train_loader, val_loader, test_loader, n_channels, n_timesteps, output_dim = load_data(
        task=task,
        seq_len=seq_len,
        batch_size=batch_size,
        num_markers=num_markers,
        seed=seed,
        data_dir=data_dir
    )

    print(f"Data loaded!", flush=True)
    print(f"  Input: ({n_channels} channels, {n_timesteps} timesteps)", flush=True)
    print(f"  Output dim: {output_dim}", flush=True)

    # Create model
    print(f"\nCreating model...", flush=True)
    model = create_model(
        model_id=model_id,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        n_qubits=n_qubits,
        n_layers=n_layers,
        d_model=d_model,
        d_state=d_state,
        output_dim=output_dim,
        dropout=0.1,
        device=device_obj,
        task_type=task_config['type']
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # Verify all model parameters are on the correct device
    print(f"Verifying device placement...", flush=True)
    device_mismatch = False
    for name, param in model.named_parameters():
        if param.device != device_obj:
            print(f"  WARNING: {name} is on {param.device}, moving to {device_obj}", flush=True)
            param.data = param.data.to(device_obj)
            device_mismatch = True

    # Also check buffers (e.g., batch norm running stats)
    for name, buffer in model.named_buffers():
        if buffer.device != device_obj:
            print(f"  WARNING: Buffer {name} is on {buffer.device}", flush=True)

    if not device_mismatch:
        print(f"  All parameters are on {device_obj}", flush=True)

    # Ensure model is on device
    model = model.to(device_obj)
    print(f"Model moved to {device_obj}", flush=True)

    # Training setup
    if task_config['type'] == 'classification':
        criterion = nn.CrossEntropyLoss()
        train_epoch_fn = train_epoch_classification
        eval_fn = evaluate_classification
        better_fn = lambda new, old: new > old  # Higher is better for accuracy
    else:
        criterion = nn.MSELoss()
        train_epoch_fn = train_epoch_regression
        eval_fn = evaluate_regression
        better_fn = lambda new, old: new < old  # Lower is better for MSE

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        'train_loss': [], 'train_metric': [],
        'val_loss': [], 'val_metric': [],
        'epochs': [], 'lr': []
    }

    if task_config['type'] == 'classification':
        best_metric = 0  # Best accuracy
    else:
        best_metric = float('inf')  # Best MSE (lower is better)

    best_model_state = None
    epochs_without_improvement = 0
    start_epoch = 0

    # Resume from checkpoint if requested
    checkpoint_path = get_checkpoint_path(output_dir, model_id, task, seq_len, seed)
    if resume:
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device_obj)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_metric = checkpoint['best_metric']
            best_model_state = checkpoint['best_model_state']
            epochs_without_improvement = checkpoint['epochs_without_improvement']

    start_time = time.time()

    print(f"\n{'=' * 80}", flush=True)
    print(f"Training Started (epochs {start_epoch + 1} to {n_epochs})", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_metric = train_epoch_fn(model, train_loader, criterion, optimizer, device_obj)

        # Validate
        if task_config['type'] == 'classification':
            val_loss, val_acc, val_auc, val_f1 = eval_fn(model, val_loader, criterion, device_obj)
            val_metric = val_acc
        else:
            val_mse, val_mae, val_r2 = eval_fn(model, val_loader, criterion, device_obj)
            val_loss = val_mse
            val_metric = val_mse

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Check for improvement
        improved = False
        if better_fn(val_metric, best_metric):
            best_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1

        # Record history
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)
        history['epochs'].append(epoch + 1)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress
        if (epoch + 1) % 5 == 0 or improved:
            improve_str = " *" if improved else ""
            if task_config['type'] == 'classification':
                print(f"Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s){improve_str} | "
                      f"Train: {train_loss:.4f}/{train_metric:.4f} | "
                      f"Val: {val_loss:.4f}/{val_metric:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s){improve_str} | "
                      f"Train MSE: {train_loss:.6f} | "
                      f"Val MSE: {val_metric:.6f}", flush=True)

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch,
                history, best_metric, best_model_state, epochs_without_improvement
            )

        # Early stopping
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
            break

    total_time = time.time() - start_time

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device_obj)

    if task_config['type'] == 'classification':
        test_loss, test_acc, test_auc, test_f1 = eval_fn(model, test_loader, criterion, device_obj)
        print(f"\n{'=' * 80}")
        print(f"Training Complete!")
        print(f"{'=' * 80}")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Best Val Acc: {best_metric:.4f}")
        print(f"Test Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  F1: {test_f1:.4f}")
        print(f"  Baseline: {task_config['baseline']:.4f}")

        test_metrics = {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_loss': test_loss
        }
    else:
        test_mse, test_mae, test_r2 = eval_fn(model, test_loader, criterion, device_obj)
        print(f"\n{'=' * 80}")
        print(f"Training Complete!")
        print(f"{'=' * 80}")
        print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Best Val MSE: {best_metric:.6f}")
        print(f"Test Results:")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  Baseline MSE: {task_config['baseline']:.4f}")
        print(f"  Improvement over baseline: {(1 - test_mse/task_config['baseline'])*100:.1f}%")

        test_metrics = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }

    # Prepare results
    results = {
        'model_id': model_id,
        'model_name': model_name,
        'model_info': model_info,
        'task': task,
        'task_type': task_config['type'],
        'seq_len': seq_len,
        'seed': seed,
        'n_params': n_params,
        'hyperparameters': {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'd_model': d_model,
            'd_state': d_state,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_markers': num_markers
        },
        'data_info': {
            'n_channels': n_channels,
            'n_timesteps': n_timesteps,
            'output_dim': output_dim
        },
        'history': history,
        'best_val_metric': best_metric,
        **test_metrics,
        'baseline': task_config['baseline'],
        'training_time': total_time,
        'epochs_trained': len(history['epochs']),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results JSON (convert numpy types to Python types for JSON serialization)
    results_file = output_path / f"synthetic_{model_id}_{task}_L{seq_len}_seed{seed}_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save final model checkpoint
    final_checkpoint_file = output_path / f"synthetic_{model_id}_{task}_L{seq_len}_seed{seed}_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'model_id': model_id,
        'model_name': model_name,
        'n_params': n_params,
        **test_metrics,
        'hyperparameters': results['hyperparameters'],
        'data_info': results['data_info']
    }, final_checkpoint_file)
    print(f"Model saved to: {final_checkpoint_file}")

    # Clean up training checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Cleaned up training checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic Benchmark Runner for Long-Range Sequence Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (12 selected):
  Group 1: 1a (QuantumTransformer), 1b (QuantumMambaSSM), 1c (QuantumHydraSSM)
  Group 2: 2a (ClassicalQuantumAttention), 2d (QuantumMambaHydraSSM), 2e (QuantumHydraHydraSSM)
  Group 3: 3a (ClassicalTransformer), 3b (TrueClassicalMamba), 3c (TrueClassicalHydra)
  Group 4: 4a (QuantumTransformerE2E), 4d (QuantumMambaE2E_Super), 4e (QuantumHydraE2E_Super)

Tasks:
  forrelation: Classification - detect quantum correlations
  adding_problem: Regression - sum of two marked values
  selective_copy: Multi-output regression - output marked tokens

Examples:
  python run_synthetic_benchmark.py --model-id 1c --task forrelation --seq-len 200 --seed 2024
  python run_synthetic_benchmark.py --model-id 2e --task adding_problem --seq-len 500 --seed 2025
  python run_synthetic_benchmark.py --model-id 3b --task selective_copy --seq-len 1000 --seed 2026
        """
    )

    # Model selection
    parser.add_argument("--model-id", type=str, required=True,
                        choices=['1a', '1b', '1c', '2a', '2d', '2e', '3a', '3b', '3c', '4a', '4d', '4e'],
                        help="Model ID")

    # Task selection
    parser.add_argument("--task", type=str, required=True,
                        choices=['forrelation', 'adding_problem', 'selective_copy'],
                        help="Synthetic benchmark task")

    # Sequence length
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Sequence length (e.g., 100, 200, 500, 1000)")

    # Quantum hyperparameters
    parser.add_argument("--n-qubits", type=int, default=6,
                        help="Number of qubits (for quantum models)")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of quantum layers")

    # Classical hyperparameters
    parser.add_argument("--d-model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--d-state", type=int, default=16,
                        help="State dimension (for SSM models)")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping", type=int, default=20,
                        help="Early stopping patience (0 to disable)")

    # Task-specific
    parser.add_argument("--num-markers", type=int, default=8,
                        help="Number of markers for selective copy task")

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./results/synthetic_benchmarks",
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default="./data/synthetic_benchmarks",
                        help="Data directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint if available")

    args = parser.parse_args()

    results = train_model(
        model_id=args.model_id,
        task=args.task,
        seq_len=args.seq_len,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_markers=args.num_markers,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        device=args.device,
        early_stopping_patience=args.early_stopping,
        resume=args.resume,
    )

    print("\n" + "=" * 80)
    print("SYNTHETIC BENCHMARK RUN COMPLETE")
    print("=" * 80)
    print(f"Model: {args.model_id} ({results['model_name']})")
    print(f"Task: {args.task}")
    print(f"Seq Length: {args.seq_len}")
    if results['task_type'] == 'classification':
        print(f"Test Accuracy: {results['test_acc']:.4f}")
        print(f"Test AUC: {results['test_auc']:.4f}")
    else:
        print(f"Test MSE: {results['test_mse']:.6f}")
        print(f"Test R²: {results['test_r2']:.4f}")
    print("=" * 80 + "\n")
