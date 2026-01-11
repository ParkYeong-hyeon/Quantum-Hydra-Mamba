"""
BACKUP: QuantumMambaGated with Conv1d (depthwise) architecture.

This is the version that was used for DNA and genomic experiments.
The DNA experiments completed with this code and showed good results (86.7% avg).
The genomic experiments are currently running with this code.

NOTE: This version has groups=feature_dim in Conv1d, which causes issues for EEG
(64 channels) because it processes each channel independently without mixing.
For DNA/genomic (4 channels), this is less problematic.

The current QuantumGatedRecurrence.py uses nn.Linear instead for proper channel mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumMambaGated_Conv1d(nn.Module):
    """
    BACKUP VERSION with Conv1d (depthwise convolution).

    This version uses groups=feature_dim which processes each channel independently.
    Works OK for DNA/genomic (4 channels) but fails for EEG (64 channels).
    """

    def __init__(
        self,
        n_qubits=4,
        n_timesteps=160,
        qlcu_layers=2,
        feature_dim=64,
        hidden_dim=64,
        output_dim=2,
        chunk_size=16,
        dropout=0.1,
        device="cpu"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Temporal convolution with DEPTHWISE (groups=feature_dim)
        # This processes each channel independently - NO cross-channel mixing
        # Works for DNA (4 channels) but fails for EEG (64 channels)
        self.temporal_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim  # DEPTHWISE - each channel processed independently
        )

        # Gated recurrence with superposition
        # (ChunkedGatedSuperposition would be imported from QuantumGatedRecurrence)
        # self.gated_recurrence = ChunkedGatedSuperposition(...)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.to(self.device)

    def forward(self, x):
        """
        Args:
            x: (batch, feature_dim) for 2D or
               (batch, feature_dim, n_timesteps) or (batch, n_timesteps, feature_dim) for 3D

        Returns:
            output: (batch, output_dim)
        """
        # Handle 2D input (batch, features) - add temporal dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)

        # Handle different input formats - ensure (batch_size, n_timesteps, feature_dim)
        if x.dim() == 3 and x.shape[1] != self.n_timesteps and x.shape[2] == self.n_timesteps:
            # Input is (batch_size, feature_dim, n_timesteps)
            x = x.permute(0, 2, 1)  # Convert to (batch_size, n_timesteps, feature_dim)

        # Now x is (batch, n_timesteps, feature_dim)
        # Conv1d expects (batch, channels, length), so permute
        x_conv_input = x.permute(0, 2, 1)  # -> (batch, feature_dim, n_timesteps)
        x_conv = self.temporal_conv(x_conv_input)
        x_conv = F.silu(x_conv)
        x_seq = x_conv.permute(0, 2, 1)  # -> (batch, n_timesteps, feature_dim)

        # h_final, _ = self.gated_recurrence(x_seq)
        # output = self.output_layer(h_final)

        # return output
        pass  # Incomplete - just for reference


# Summary of changes:
#
# OLD (this file - Conv1d with groups=feature_dim):
#   - self.temporal_conv = nn.Conv1d(..., groups=feature_dim)
#   - Depthwise convolution - each channel processed independently
#   - Works for DNA (4 channels), fails for EEG (64 channels)
#
# NEW (QuantumGatedRecurrence.py - nn.Linear):
#   - self.feature_proj = nn.Linear(feature_dim, feature_dim)
#   - Full channel mixing - all channels interact
#   - Works for both DNA and EEG
