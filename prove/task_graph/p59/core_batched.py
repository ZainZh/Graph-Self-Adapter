"""Numerically equivalent batched-history execution backend for P59.

The scientific model, losses, update order, and state dict are inherited
unchanged.  Only the repeated per-transition feedback-head calls inside one
last-W inner loss are fused into one tensor call.
"""
from __future__ import annotations

import torch

from .core import P59AdaptationCore


class BatchedP59AdaptationCore(P59AdaptationCore):
    """P59 core with a vectorized prepared-history feedback evaluation."""

    @staticmethod
    def _stack_prepared(prepared_history: list[tuple]) -> tuple:
        if not prepared_history:
            raise ValueError("empty prepared history")
        width = len(prepared_history[0])
        if width != 5 or any(len(item) != width for item in prepared_history):
            raise ValueError("prepared feedback tuple schema changed")
        return tuple(torch.stack([item[index] for item in prepared_history], 0)
                     for index in range(width))

    def _adapt_prepared_history(self, prepared_history: list[tuple], z: torch.Tensor,
                                z0: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        window = prepared_history[-self.contract.history:]
        stacked = self._stack_prepared(window)
        losses = []
        for _ in range(self.contract.inner_steps):
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            # _prepared_feedback_loss broadcasts z [B,16] across W and
            # returns [W,B]. This is the original mean(W).sum(B) reduction.
            loss = self._prepared_feedback_loss(stacked, z, z0).mean(0).sum()
            gradient, = torch.autograd.grad(loss, z, create_graph=create_graph)
            gradient = torch.cat((gradient[..., :9], torch.zeros_like(gradient[..., 9:])), -1)
            z = z - self.contract.inner_lr * gradient
            z = torch.cat((z[..., :9], torch.zeros_like(z[..., 9:])), -1)
            losses.append(loss)
        return z, torch.stack(losses)
