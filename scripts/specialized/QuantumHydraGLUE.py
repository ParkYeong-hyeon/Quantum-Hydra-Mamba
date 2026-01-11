#!/usr/bin/env python3
"""
Quantum Hydra Models for GLUE Benchmark

Wraps QuantumHydraGated and QuantumMambaGated for NLP tasks with:
- Token embedding layer
- Sentence pair handling
- Task-specific output heads
- Support for both classification and regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Quantum-Hydra-Mamba/

# Import base quantum models
from models.QuantumGatedRecurrence import (
    QuantumHydraGated,
    QuantumMambaGated,
    QuantumSuperpositionBranches,
    ChunkedGatedSuperposition
)


# ================================================================================
# Token Embedding Module
# ================================================================================

class TokenEmbedding(nn.Module):
    """
    Token embedding with positional encoding for quantum models.
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # BERT vocab size
        embedding_dim: int = 128,
        max_length: int = 128,
        dropout: float = 0.1,
        use_positional: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional embeddings
        self.use_positional = use_positional
        if use_positional:
            self.positional_embedding = nn.Embedding(max_length, embedding_dim)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        if self.use_positional:
            nn.init.normal_(self.positional_embedding.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - optional

        Returns:
            embeddings: (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        embeddings = self.token_embedding(input_ids)

        # Add positional embeddings
        if self.use_positional:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
            embeddings = embeddings + self.positional_embedding(positions)

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply attention mask (set padded positions to zero)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings


# ================================================================================
# Quantum Hydra for GLUE
# ================================================================================

class QuantumHydraGLUE(nn.Module):
    """
    Quantum Hydra model adapted for GLUE benchmark tasks.

    Architecture:
        Input IDs → Token Embedding → QuantumHydraGated → Task Head → Output

    Features:
        - Three-branch bidirectional quantum processing
        - LSTM-style gating for long sequences
        - Task-specific output heads (classification/regression)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 128,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        hidden_dim: int = 64,
        num_labels: int = 2,
        max_length: int = 128,
        chunk_size: int = 16,
        dropout: float = 0.1,
        is_regression: bool = False,
        device: str = "cpu"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.max_length = max_length
        self.is_regression = is_regression
        self.device = torch.device(device) if isinstance(device, str) else device

        # Token embedding
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout
        )

        # Quantum Hydra backbone
        self.quantum_hydra = QuantumHydraGated(
            n_qubits=n_qubits,
            n_timesteps=max_length,
            qlcu_layers=qlcu_layers,
            feature_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Intermediate output
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )

        # Task-specific output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels if not is_regression else 1)
        )

        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) or (batch_size, 1) for regression

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Embed tokens
        embeddings = self.embedding(input_ids, attention_mask)
        # embeddings: (batch_size, seq_len, embedding_dim)

        # Process through quantum backbone
        # QuantumHydraGated expects (batch, seq_len, feature_dim)
        quantum_output = self.quantum_hydra(embeddings)
        # quantum_output: (batch_size, hidden_dim)

        # Classification head
        logits = self.classifier(quantum_output)

        output = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            if self.is_regression:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            output['loss'] = loss

        return output

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================================
# Quantum Mamba for GLUE
# ================================================================================

class QuantumMambaGLUE(nn.Module):
    """
    Quantum Mamba model adapted for GLUE benchmark tasks.

    Note: QuantumMambaGated may have stability issues on very long sequences (>1000).
    For GLUE tasks (typically <512 tokens), this should be fine.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 128,
        n_qubits: int = 6,
        qlcu_layers: int = 2,
        hidden_dim: int = 64,
        num_labels: int = 2,
        max_length: int = 128,
        chunk_size: int = 16,
        dropout: float = 0.1,
        is_regression: bool = False,
        device: str = "cpu"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.max_length = max_length
        self.is_regression = is_regression
        self.device = torch.device(device) if isinstance(device, str) else device

        # Token embedding
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout
        )

        # Quantum Mamba backbone
        self.quantum_mamba = QuantumMambaGated(
            n_qubits=n_qubits,
            n_timesteps=max_length,
            qlcu_layers=qlcu_layers,
            feature_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            chunk_size=chunk_size,
            dropout=dropout,
            device=device
        )

        # Task-specific output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels if not is_regression else 1)
        )

        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        embeddings = self.embedding(input_ids, attention_mask)
        quantum_output = self.quantum_mamba(embeddings)
        logits = self.classifier(quantum_output)

        output = {'logits': logits}

        if labels is not None:
            if self.is_regression:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            output['loss'] = loss

        return output

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================================
# Classical Baseline: LSTM for GLUE
# ================================================================================

class LSTMBaseline(nn.Module):
    """
    Classical LSTM baseline for fair comparison.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_labels: int = 2,
        max_length: int = 128,
        dropout: float = 0.1,
        bidirectional: bool = True,
        is_regression: bool = False,
        device: str = "cpu"
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.is_regression = is_regression
        self.bidirectional = bidirectional
        self.device = torch.device(device) if isinstance(device, str) else device

        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels if not is_regression else 1)
        )

        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        embeddings = self.embedding(input_ids, attention_mask)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(embeddings)

        # Use final hidden state
        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]

        logits = self.classifier(hidden)

        output = {'logits': logits}

        if labels is not None:
            if self.is_regression:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            output['loss'] = loss

        return output

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================================
# Model Factory
# ================================================================================

def create_glue_model(
    model_name: str,
    vocab_size: int = 30522,
    embedding_dim: int = 128,
    hidden_dim: int = 64,
    num_labels: int = 2,
    max_length: int = 128,
    n_qubits: int = 6,
    qlcu_layers: int = 2,
    chunk_size: int = 16,
    dropout: float = 0.1,
    is_regression: bool = False,
    device: str = "cpu"
) -> nn.Module:
    """
    Factory function to create GLUE models.

    Args:
        model_name: 'quantum_hydra', 'quantum_mamba', or 'lstm_baseline'
        vocab_size: Vocabulary size
        embedding_dim: Token embedding dimension
        hidden_dim: Hidden layer dimension
        num_labels: Number of output labels
        max_length: Maximum sequence length
        n_qubits: Number of qubits (quantum models only)
        qlcu_layers: Quantum circuit layers (quantum models only)
        chunk_size: Chunk size for processing (quantum models only)
        dropout: Dropout rate
        is_regression: Whether task is regression
        device: Device to use

    Returns:
        Model instance
    """
    model_name = model_name.lower()

    if model_name == 'quantum_hydra':
        return QuantumHydraGLUE(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            max_length=max_length,
            chunk_size=chunk_size,
            dropout=dropout,
            is_regression=is_regression,
            device=device
        )

    elif model_name == 'quantum_mamba':
        return QuantumMambaGLUE(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_qubits=n_qubits,
            qlcu_layers=qlcu_layers,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            max_length=max_length,
            chunk_size=chunk_size,
            dropout=dropout,
            is_regression=is_regression,
            device=device
        )

    elif model_name in ['lstm', 'lstm_baseline']:
        return LSTMBaseline(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_labels=num_labels,
            max_length=max_length,
            dropout=dropout,
            bidirectional=True,
            is_regression=is_regression,
            device=device
        )

    else:
        raise ValueError(f"Unknown model: {model_name}. Available: quantum_hydra, quantum_mamba, lstm_baseline")


# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Quantum GLUE Models - Testing")
    print("=" * 80)

    device = "cpu"
    batch_size = 4
    seq_len = 64
    vocab_size = 30522
    num_labels = 2

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, num_labels, (batch_size,))

    print("\n[1] Testing QuantumHydraGLUE...")
    model_hydra = QuantumHydraGLUE(
        vocab_size=vocab_size,
        embedding_dim=64,
        n_qubits=4,
        qlcu_layers=2,
        hidden_dim=32,
        num_labels=num_labels,
        max_length=seq_len,
        chunk_size=8,
        device=device
    )
    print(f"  Parameters: {model_hydra.get_num_params():,}")

    output = model_hydra(input_ids, attention_mask, labels)
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

    print("\n[2] Testing QuantumMambaGLUE...")
    model_mamba = QuantumMambaGLUE(
        vocab_size=vocab_size,
        embedding_dim=64,
        n_qubits=4,
        qlcu_layers=2,
        hidden_dim=32,
        num_labels=num_labels,
        max_length=seq_len,
        chunk_size=8,
        device=device
    )
    print(f"  Parameters: {model_mamba.get_num_params():,}")

    output = model_mamba(input_ids, attention_mask, labels)
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

    print("\n[3] Testing LSTMBaseline...")
    model_lstm = LSTMBaseline(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=32,
        num_layers=2,
        num_labels=num_labels,
        max_length=seq_len,
        device=device
    )
    print(f"  Parameters: {model_lstm.get_num_params():,}")

    output = model_lstm(input_ids, attention_mask, labels)
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

    print("\n[4] Testing gradient flow...")
    model_hydra.train()
    optimizer = torch.optim.Adam(model_hydra.parameters(), lr=1e-3)

    output = model_hydra(input_ids, attention_mask, labels)
    loss = output['loss']
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flow: OK")

    print("\n[5] Testing model factory...")
    for model_name in ['quantum_hydra', 'quantum_mamba', 'lstm_baseline']:
        model = create_glue_model(
            model_name=model_name,
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=32,
            num_labels=2,
            max_length=64,
            n_qubits=4,
            device=device
        )
        print(f"  {model_name}: {model.get_num_params():,} params")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
