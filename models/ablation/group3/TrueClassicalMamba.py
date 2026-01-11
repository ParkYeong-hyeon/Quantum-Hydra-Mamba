import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
True Classical Mamba Implementation

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Gu and Dao (arXiv:2312.00752)

Key components:
1. State Space Model (SSM) foundation with A, B, C, D matrices
2. Selective state space mechanism (input-dependent B, C, and dt)
3. Unidirectional processing using an efficient parallel scan (here simulated with a loop)
4. Gated MLP-like block structure

Reference: https://github.com/state-spaces/mamba
"""


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core (Unidirectional).

    Implements the discrete-time SSM:
        x[t] = A * x[t-1] + B * u[t]
        y[t] = C * x[t]

    With selective (input-dependent) B, C, and dt parameters.

    Args:
        d_model: Model dimension
        d_state: State dimension
        dt_rank: Rank for dt projection
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank='auto',
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # A parameter: diagonal state matrix (negative for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter: skip connection (Mamba's D is usually a simple pass-through)
        self.D = nn.Parameter(torch.ones(d_model))

        # dt projection parameters
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Initialize dt projection
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x, dt, B, C):
        """
        Unidirectional selective SSM scan.

        Args:
            x: Input (batch, seq_len, d_model)
            dt: Time steps (batch, seq_len, d_model)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)

        Returns:
            y: Output (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Get A matrix and discretize
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq_len, d_model, 1)
        
        # Discretization using Zero-Order Hold (ZOH)
        # A_discrete = exp(A * dt)
        # B_discrete = (A_discrete - 1) / A * B
        # For simplicity, we use a first-order approximation here.
        # A_discrete ≈ 1 + A * dt
        # B_discrete ≈ dt * B
        A_discrete = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt_expanded)
        B_discrete = dt_expanded * B.unsqueeze(2) # (B, L, d_inner, d_state)

        # Recurrent scan (simulating the efficient parallel scan)
        state = torch.zeros(batch, d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            u_t = x[:, t, :]
            B_t = B_discrete[:, t, :, :]
            A_t = A_discrete[:, t, :, :]
            C_t = C[:, t, :]

            # State update: x[t] = A_d * x[t-1] + B_d * u[t]
            state = A_t * state + u_t.unsqueeze(-1) * B_t
            
            # Output: y[t] = C * x[t]
            y_t = torch.sum(C_t.unsqueeze(1) * state, dim=-1)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        
        # Add skip connection D * x
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """
    Mamba block implementing a unidirectional selective state space model.

    Architecture:
    1. Input projection to expand dimension
    2. Split into gate (z) and value (x) branches
    3. 1D convolution for local context
    4. Unidirectional Selective SSM
    5. Gated output with RMSNorm
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank='auto',
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection: projects to expanded dimension for x and z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Projection for dt, B, C parameters of the SSM
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # 1D convolution for local context (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM core
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=self.dt_rank,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = RMSNorm(d_model)

    def forward(self, u):
        """
        Unidirectional Mamba forward pass.

        Args:
            u: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = u.shape
        
        # Apply normalization before the block
        u_norm = self.norm(u)

        # 1. Input projection
        xz = self.in_proj(u_norm)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # 2. Apply 1D convolution with SiLU activation
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x = F.silu(x_conv.transpose(1, 2))

        # 3. Project x to get dt, B, C
        x_projected = self.x_proj(x)
        dt, B, C = torch.split(x_projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 4. Compute dt (time steps)
        dt = self.ssm.dt_proj(dt)
        dt = F.softplus(dt)

        # 5. UNIDIRECTIONAL PROCESSING
        y = self.ssm(x, dt, B, C)

        # 6. Apply gating
        y = y * F.silu(z)

        # 7. Output projection and add residual connection
        output = self.out_proj(y) + u

        return output


class TrueClassicalMamba(nn.Module):
    """
    Complete Classical Mamba model for sequence classification.
    """

    def __init__(
        self,
        n_channels,
        n_timesteps,
        d_model=128,
        d_state=16,
        n_layers=2,
        d_conv=4,
        expand=2,
        output_dim=2,
        dropout=0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.embedding = nn.Linear(n_channels, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_dim)

        # Move all parameters to specified device
        self.to(device)

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_timesteps) for 3D inputs (e.g., EEG)
               OR (batch, features) for 2D inputs (e.g., DNA, MNIST)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # 2D input: (B, features) -> (B, 1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D tensor")

        # Now x is 3D: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        output = self.classifier(x)

        return output

# ================================================================================
# Testing
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("True Classical Mamba - Testing")
    print("=" * 80)

    batch_size = 4
    n_channels = 64
    n_timesteps = 160
    d_model = 128
    d_state = 16
    output_dim = 2

    print("\n[1] Testing MambaBlock...")
    block = MambaBlock(d_model=d_model, d_state=d_state)
    u = torch.randn(batch_size, n_timesteps, d_model)
    output = block(u)
    print(f"  Input shape: {u.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == u.shape, "MambaBlock output shape mismatch!"

    print("\n[2] Testing TrueClassicalMamba (full model)...")
    model = TrueClassicalMamba(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        d_state=d_state,
        n_layers=2,
        output_dim=output_dim,
    )
    x_input = torch.randn(batch_size, n_channels, n_timesteps)
    output = model(x_input)
    print(f"  Input shape: {x_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, output_dim), "Model output shape mismatch!"

    print("\n[3] Checking trainable parameters...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")

    print("\n[4] Testing gradient flow...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    outputs = model(x_input)
    loss = criterion(outputs, torch.randint(0, output_dim, (batch_size,)))
    loss.backward()
    optimizer.step()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm (embedding): {model.embedding.weight.grad.norm().item():.4f}")

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
    print("\nKey Mamba Features Implemented:")
    print("  ✓ Selective State Space Model (SSM) Core")
    print("  ✓ UNIDIRECTIONAL processing via forward scan")
    print("  ✓ Gated block structure with residual connections")
    print("  ✓ 1D convolution for local context")
    print("=" * 80)
