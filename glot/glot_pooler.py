# glot/glot_pooler.py
import torch
import torch.nn as nn
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout


class GLOTPooler(nn.Module):
    """GLOT pooling module: graph construction + GNN + readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        jk_mode: str = "cat",
        threshold: float = 0.6,
    ):
        super().__init__()
        self.threshold = threshold
        self.token_gnn = TokenGNN(input_dim, hidden_dim, num_gnn_layers, jk_mode)
        self.readout = AttentionReadout(self.token_gnn.output_dim)
        self.output_dim = self.token_gnn.output_dim

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_data = build_token_graph(hidden_states, attention_mask, self.threshold)
        refined = self.token_gnn(batch_data.x, batch_data.edge_index)
        return self.readout(refined, batch_data.batch)
