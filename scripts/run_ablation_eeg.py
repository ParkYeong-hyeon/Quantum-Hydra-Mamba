#!/usr/bin/env python3
"""
Ablation Study Runner for EEG Classification
Supports all 12 models from the 2×2×3 factorial design (ABLATION_STUDY_PLAN_V3.md)

Features:
  - Resume training from checkpoint if interrupted
  - Early stopping when validation performance plateaus
  - Multiple random seeds for statistical rigor
  - Proper device handling (GPU for classical, CPU for quantum circuits)
  - Uses Pennylane's default.qubit for quantum simulation

Models:
  Group 1 (Quantum Features → Classical Mixing):
    1a: QuantumTransformer, 1b: QuantumMambaSSM, 1c: QuantumHydraSSM

  Group 2 (Classical Features → Quantum Mixing):
    2a: ClassicalQuantumAttention, 2b: ClassicalMambaQuantumSSM, 2c: ClassicalHydraQuantumSSM
    NEW - True Superposition + Delta Recurrence:
    2d: QuantumMambaHydraSSM (unidirectional), 2e: QuantumHydraHydraSSM (bidirectional)

  Group 3 (Classical Features → Classical Mixing - Baseline):
    3a: ClassicalTransformer, 3b: TrueClassicalMamba, 3c: TrueClassicalHydra

  Group 4 (Quantum Features → Quantum Mixing - E2E):
    4a: QuantumTransformerE2E, 4b: QuantumMambaE2E, 4c: QuantumHydraE2E

Usage:
    python run_ablation_eeg.py --model-id 1a --sampling-freq 80 --seed 2024
    python run_ablation_eeg.py --model-id 2b --sampling-freq 160 --seed 2025 --resume
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Import data loader
from data_loaders.Load_PhysioNet_EEG_NoPrompt import load_eeg_ts_revised


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
    '2b': {'name': 'ClassicalMambaQuantumSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum', 'type': 'mamba'},
    '2c': {'name': 'ClassicalHydraQuantumSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum', 'type': 'hydra'},
    # NEW: True Superposition + Delta Recurrence (proposed hybrid)
    '2d': {'name': 'QuantumMambaHydraSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_superposition', 'type': 'mamba'},
    '2e': {'name': 'QuantumHydraHydraSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_superposition', 'type': 'hydra'},
    # NEW: QSVT+LCU SSM (true quantum cross-timestep mixing)
    '2h': {'name': 'QuantumQSVTMambaSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_qsvt_lcu', 'type': 'mamba'},
    '2i': {'name': 'QuantumQSVTHydraSSM', 'group': 2, 'feat': 'classical', 'mix': 'quantum_qsvt_lcu', 'type': 'hydra'},

    # Group 3: Classical Features → Classical Mixing (Baseline)
    '3a': {'name': 'ClassicalTransformer', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'transformer'},
    '3b': {'name': 'TrueClassicalMamba', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'mamba'},
    '3c': {'name': 'TrueClassicalHydra', 'group': 3, 'feat': 'classical', 'mix': 'classical', 'type': 'hydra'},

    # Group 4: Quantum Features → Quantum Mixing (E2E)
    '4a': {'name': 'QuantumTransformerE2E', 'group': 4, 'feat': 'quantum', 'mix': 'quantum', 'type': 'transformer'},
    '4b': {'name': 'QuantumMambaE2E', 'group': 4, 'feat': 'quantum', 'mix': 'quantum', 'type': 'mamba'},
    '4c': {'name': 'QuantumHydraE2E', 'group': 4, 'feat': 'quantum', 'mix': 'quantum', 'type': 'hydra'},
    # NEW: E2E + True Superposition (full quantum pipeline with superposition)
    '4d': {'name': 'QuantumMambaE2ESuperposition', 'group': 4, 'feat': 'quantum_superposition', 'mix': 'quantum_superposition', 'type': 'mamba'},
    '4e': {'name': 'QuantumHydraE2ESuperposition', 'group': 4, 'feat': 'quantum_superposition', 'mix': 'quantum_superposition', 'type': 'hydra'},
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
    # Set PennyLane seed
    try:
        import pennylane as qml
        qml.numpy.random.seed(seed)
    except:
        pass


def get_checkpoint_path(output_dir, model_id, sampling_freq, seed):
    """Get path for checkpoint file."""
    output_path = Path(output_dir) / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"checkpoint_{model_id}_{sampling_freq}Hz_seed{seed}.pt"


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, history,
                    best_val_acc, best_model_state, epochs_without_improvement):
    """Save training checkpoint for resume capability."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_val_acc': best_val_acc,
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
    print(f"Best val acc so far: {checkpoint['best_val_acc']:.4f}")
    print(f"{'=' * 60}\n")

    return checkpoint


def create_model(model_id, n_channels, n_timesteps, n_qubits, n_layers,
                 d_model, d_state, output_dim, dropout, device):
    """
    Create model based on model_id (1a, 1b, ..., 4c).
    All models use default.qubit for quantum simulation.

    Parameter mapping:
    - Groups 1 & 2 (quantum models): use 'feature_dim' instead of 'n_channels'
    - Group 3 (classical): use 'n_channels'
    - Group 4 (E2E): use 'n_channels'
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}. Valid options: {list(MODEL_REGISTRY.keys())}")

    model_info = MODEL_REGISTRY[model_id]
    model_name = model_info['name']

    print(f"Creating model: {model_name} (ID: {model_id})")
    print(f"  Group {model_info['group']}: {model_info['feat']} features → {model_info['mix']} mixing ({model_info['type']})")

    # Device string for models that expect it
    device_str = str(device) if hasattr(device, '__str__') else device

    # ========================================
    # Group 1: Quantum Features → Classical Mixing
    # These models use 'feature_dim' instead of 'n_channels'
    # ========================================
    if model_id == '1a':
        from models.QuantumTransformer import QuantumTransformer
        model = QuantumTransformer(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,  # Map n_channels -> feature_dim
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
            feature_dim=n_channels,  # Map n_channels -> feature_dim
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
            feature_dim=n_channels,  # Map n_channels -> feature_dim
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # ========================================
    # Group 2: Classical Features → Quantum Mixing
    # These models also use 'feature_dim' instead of 'n_channels'
    # ========================================
    elif model_id == '2a':
        from models.QuantumMixingSSM import ClassicalQuantumAttention
        model = ClassicalQuantumAttention(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,  # Map n_channels -> feature_dim
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '2b':
        from models.QuantumMixingSSM import ClassicalMambaQuantumSSM
        model = ClassicalMambaQuantumSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,  # Map n_channels -> feature_dim
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    elif model_id == '2c':
        from models.QuantumMixingSSM import ClassicalHydraQuantumSSM
        model = ClassicalHydraQuantumSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,  # Map n_channels -> feature_dim
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # NEW: 2d and 2e - True Superposition + Delta Recurrence
    elif model_id == '2d':
        from models.QuantumHydraSSM import QuantumMambaHydraSSM
        model = QuantumMambaHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,  # Map n_channels -> feature_dim
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
            feature_dim=n_channels,  # Map n_channels -> feature_dim
            d_model=d_model,
            d_state=d_state,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str
        )

    # NEW: 2h and 2i - QSVT+LCU SSM (true quantum cross-timestep mixing)
    elif model_id == '2h':
        from models.QuantumQSVTSSM import QuantumQSVTMambaSSM
        model = QuantumQSVTMambaSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str,
            degree=2,
        )

    elif model_id == '2i':
        from models.QuantumQSVTSSM import QuantumQSVTHydraSSM
        model = QuantumQSVTHydraSSM(
            n_qubits=n_qubits,
            n_timesteps=n_timesteps,
            qlcu_layers=n_layers,
            feature_dim=n_channels,
            d_model=d_model,
            d_state=d_state,
            output_dim=output_dim,
            dropout=dropout,
            device=device_str,
            degree=2,
        )

    # ========================================
    # Group 3: Classical Features → Classical Mixing (Baselines)
    # These models use 'n_channels'
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
    # These models use 'n_channels' and separate layer params
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

    elif model_id == '4b':
        from models.QuantumE2E import QuantumMambaE2E
        model = QuantumMambaE2E(
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

    elif model_id == '4c':
        from models.QuantumE2E import QuantumHydraE2E
        model = QuantumHydraE2E(
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

    # NEW: 4d and 4e - E2E + True Superposition
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
        raise ValueError(f"Model {model_id} not implemented yet")

    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        # Ensure data is on correct device
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Ensure outputs are on same device as labels
        if outputs.device != batch_y.device:
            outputs = outputs.to(batch_y.device)

        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
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
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # Ensure data is on correct device
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)

            # Ensure outputs are on same device as labels
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

    # Calculate additional metrics
    try:
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def train_model(
    model_id,
    n_qubits,
    n_layers,
    d_model,
    d_state,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    sample_size,
    sampling_freq,
    seed,
    output_dir,
    device,
    early_stopping_patience=10,
    resume=False,
):
    """
    Train a single model and save results.

    Features:
        - Checkpoint saving every epoch for resume capability
        - Early stopping when validation performance plateaus
        - Proper device handling for hybrid quantum-classical models

    Returns:
        dict with training history and final test metrics
    """
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

    print("=" * 80)
    print(f"ABLATION STUDY - EEG CLASSIFICATION")
    print("=" * 80)
    print(f"Model ID: {model_id} ({model_name})")
    print(f"  Group {model_info['group']}: {model_info['feat']} feat → {model_info['mix']} mix ({model_info['type']})")
    print(f"Seed: {seed}")
    print(f"Device: {device_obj}")
    print(f"Sampling Frequency: {sampling_freq} Hz")
    print("-" * 80)
    print(f"Hyperparameters:")
    print(f"  n_qubits={n_qubits}, n_layers={n_layers}")
    print(f"  d_model={d_model}, d_state={d_state}")
    print(f"  epochs={n_epochs}, batch_size={batch_size}")
    print(f"  lr={learning_rate}, weight_decay={weight_decay}")
    print(f"  early_stopping={early_stopping_patience}")
    print("=" * 80)

    # Load data
    print("\nLoading PhysioNet EEG Data...", flush=True)
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=seed,
        device=device_obj,
        batch_size=batch_size,
        sampling_freq=sampling_freq,
        sample_size=sample_size
    )

    n_channels = input_dim[1]
    n_timesteps = input_dim[2]
    output_dim = 2  # Binary classification (left/right)

    print(f"Data loaded!", flush=True)
    print(f"  Input: ({n_channels} channels, {n_timesteps} timesteps)", flush=True)
    print(f"  Output: {output_dim} classes", flush=True)

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
        device=device_obj
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # Verify all model parameters are on correct device
    for name, param in model.named_parameters():
        if param.device != device_obj:
            print(f"  Warning: {name} on {param.device}, moving to {device_obj}", flush=True)
            param.data = param.data.to(device_obj)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'epochs': [], 'lr': []
    }

    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    start_epoch = 0

    # Resume from checkpoint if requested
    checkpoint_path = get_checkpoint_path(output_dir, model_id, sampling_freq, seed)
    if resume:
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device_obj)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_val_acc = checkpoint['best_val_acc']
            best_model_state = checkpoint['best_model_state']
            epochs_without_improvement = checkpoint['epochs_without_improvement']

    start_time = time.time()

    print(f"\n{'=' * 80}", flush=True)
    print(f"Training Started (epochs {start_epoch + 1} to {n_epochs})", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device_obj)

        # Validate
        val_loss, val_acc, val_auc, val_f1, _ = evaluate(model, val_loader, criterion, device_obj)

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Check for improvement
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print progress (with flush for visibility)
        if (epoch + 1) % 5 == 0 or improved:
            improve_str = " *" if improved else ""
            print(f"Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s){improve_str} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}", flush=True)

        # Save checkpoint every 5 epochs (for resume capability)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch,
                history, best_val_acc, best_model_state, epochs_without_improvement
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

    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(model, test_loader, criterion, device_obj)

    print(f"\n{'=' * 80}")
    print(f"Training Complete!")
    print(f"{'=' * 80}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Prepare results
    results = {
        'model_id': model_id,
        'model_name': model_name,
        'model_info': model_info,
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
            'sample_size': sample_size,
            'sampling_freq': sampling_freq
        },
        'data_info': {
            'n_channels': n_channels,
            'n_timesteps': n_timesteps,
            'output_dim': output_dim
        },
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'training_time': total_time,
        'epochs_trained': len(history['epochs']),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    freq_str = f"{sampling_freq}Hz"
    results_file = output_path / f"ablation_{model_id}_{freq_str}_seed{seed}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save final model checkpoint
    final_checkpoint_file = output_path / f"ablation_{model_id}_{freq_str}_seed{seed}_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'model_id': model_id,
        'model_name': model_name,
        'n_params': n_params,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'hyperparameters': results['hyperparameters'],
        'data_info': results['data_info']
    }, final_checkpoint_file)
    print(f"Model saved to: {final_checkpoint_file}")

    # Clean up training checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Cleaned up training checkpoint: {checkpoint_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study Runner for EEG Classification (12 models × 3 frequencies × 5 seeds)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model IDs:
  Group 1 (Quantum Features → Classical Mixing):
    1a: QuantumTransformer, 1b: QuantumMambaSSM, 1c: QuantumHydraSSM

  Group 2 (Classical Features → Quantum Mixing):
    2a: ClassicalQuantumAttention, 2b: ClassicalMambaQuantumSSM, 2c: ClassicalHydraQuantumSSM

  Group 3 (Classical Features → Classical Mixing - Baseline):
    3a: ClassicalTransformer, 3b: TrueClassicalMamba, 3c: TrueClassicalHydra

  Group 4 (Quantum Features → Quantum Mixing - E2E):
    4a: QuantumTransformerE2E, 4b: QuantumMambaE2E, 4c: QuantumHydraE2E

Examples:
  python run_ablation_eeg.py --model-id 2b --sampling-freq 80 --seed 2024
  python run_ablation_eeg.py --model-id 3a --sampling-freq 160 --seed 2025 --resume
        """
    )

    # Model selection
    parser.add_argument("--model-id", type=str, required=True,
                        choices=['1a', '1b', '1c', '2a', '2b', '2c', '2d', '2e', '2h', '2i',
                                 '3a', '3b', '3c', '4a', '4b', '4c', '4d', '4e'],
                        help="Model ID (1a-4e, including 2d/2e superposition, 2h/2i QSVT+LCU, 4d/4e E2E superposition)")

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
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience (0 to disable)")

    # Data parameters
    parser.add_argument("--sample-size", type=int, default=109,
                        help="Number of PhysioNet subjects (max 109)")
    parser.add_argument("--sampling-freq", type=int, default=80,
                        choices=[40, 80, 160],
                        help="EEG sampling frequency: 40, 80, or 160 Hz")

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./results/ablation_eeg",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint if available")

    args = parser.parse_args()

    results = train_model(
        model_id=args.model_id,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        sample_size=args.sample_size,
        sampling_freq=args.sampling_freq,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        early_stopping_patience=args.early_stopping,
        resume=args.resume,
    )

    print("\n" + "=" * 80)
    print("ABLATION STUDY RUN COMPLETE")
    print("=" * 80)
    print(f"Model: {args.model_id} ({results['model_name']})")
    print(f"Sampling Freq: {args.sampling_freq} Hz ({results['data_info']['n_timesteps']} timesteps)")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print("=" * 80 + "\n")
