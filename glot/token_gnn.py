# glot/token_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, JumpingKnowledge


class TokenGNN(nn.Module):
    """GNN that refines token representations via message passing.

    Architecture: input projection (d -> p), K GATConv layers with ReLU,
    Jumping Knowledge aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        jk_mode: str = "cat",
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim))

        self.activation = nn.ReLU()

        self.jk = JumpingKnowledge(
            mode=jk_mode, channels=hidden_dim, num_layers=num_layers
        )

        if jk_mode == "cat":
            self.output_dim = hidden_dim * (num_layers + 1)
        else:
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        layer_outputs = [h]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)
            layer_outputs.append(h)

        return self.jk(layer_outputs)
