import torch
import torch.nn as nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool


class AttentionReadout(nn.Module):
    """Learned attention-weighted aggregation over graph nodes.

    Computes a scalar importance score per token via an MLP,
    applies per-graph softmax, then weighted-sums to produce
    one vector per sentence.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_total, D) all token representations across batch.
            batch: (N_total,) graph assignment vector from PyG.

        Returns:
            (B, D) sentence-level representations.
        """
        scores = self.attention(x).squeeze(-1)  # (N_total,)
        weights = pyg_softmax(scores, batch)  # (N_total,)
        weighted = x * weights.unsqueeze(-1)  # (N_total, D)
        return global_add_pool(weighted, batch)  # (B, D)
