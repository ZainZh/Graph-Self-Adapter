from __future__ import annotations

import torch
from torch import Tensor, nn


class HistoryConditionedAdapter(nn.Module):
    """RNN/MLP adapter baseline without z-gradient inner adaptation.

    The context is updated only during support using observed transitions.
    During query, callers reuse the frozen context and must not update it from
    query history.
    """

    def __init__(
        self,
        u_dim: int = 256,
        action_dim: int = 2,
        context_dim: int = 32,
        max_residual: float | Tensor = 0.15,
    ):
        super().__init__()
        residual = torch.as_tensor(max_residual, dtype=torch.float32)
        if residual.ndim == 0:
            residual = residual.repeat(action_dim)
        self.register_buffer("max_residual", residual)
        self.context_dim = context_dim
        self.transition_proj = nn.Sequential(
            nn.Linear(u_dim + action_dim + u_dim, 128),
            nn.GELU(),
            nn.Linear(128, context_dim),
        )
        self.rnn = nn.GRUCell(context_dim, context_dim)
        self.policy = nn.Sequential(
            nn.Linear(u_dim + context_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def initial_context(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.context_dim, device=device)

    def update_context(self, context: Tensor, u: Tensor, action: Tensor, u_next: Tensor) -> Tensor:
        transition = torch.cat([u, action.detach(), u_next - u], dim=-1)
        return self.rnn(self.transition_proj(transition), context)

    def forward(self, u: Tensor, context: Tensor, a_base: Tensor) -> Tensor:
        raw = self.policy(torch.cat([u, context, a_base], dim=-1))
        return self.max_residual.to(raw.device) * torch.tanh(raw)
