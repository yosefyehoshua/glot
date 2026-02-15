# glot/baselines.py
import torch
import torch.nn as nn


class MeanPooler(nn.Module):
    """Average hidden states over valid (non-padding) tokens."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts


class MaxPooler(nn.Module):
    """Element-wise max over valid tokens."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
        return hidden_states.max(dim=1).values


class CLSPooler(nn.Module):
    """CLS token (encoder) or last valid token / EOS (decoder)."""

    def __init__(self, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_decoder:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths]
        return hidden_states[:, 0]


class AdaPool(nn.Module):
    """Learned scoring MLP with softmax-weighted average (Brothers, 2025)."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
