import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool


class AttentionReadout(nn.Module):
    """Learned attention-weighted aggregation over graph nodes.

    Computes a scalar importance score per token via an MLP,
    applies per-graph softmax, then weighted-sums to produce
    one vector per sentence.

    Scorer hidden dim scales with input: max(128, input_dim // 2).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        scorer_hidden = max(128, input_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, scorer_hidden),
            nn.Tanh(),
            nn.Linear(scorer_hidden, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)  # (N_total,)
        weights = pyg_softmax(scores, batch)  # (N_total,)
        weighted = x * weights.unsqueeze(-1)  # (N_total, D)
        return global_add_pool(weighted, batch)  # (B, D)
