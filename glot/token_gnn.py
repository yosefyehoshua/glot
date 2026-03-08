import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, JumpingKnowledge


class TokenGNN(nn.Module):
    """GNN that refines token representations via message passing.

    Original architecture: no input projection, first GNN layer takes raw
    d-dim input. JK cat output = input_dim + num_layers * hidden_dim.
    Supports GAT, GCN, GIN, GINE backends.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        jk_mode: str = "cat",
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(self._make_conv(gnn_type, in_channels, hidden_dim))

        self.activation = nn.ReLU()
        self.jk = JumpingKnowledge(mode=jk_mode)

        if jk_mode == "cat":
            self.output_dim = input_dim + num_layers * hidden_dim
        else:
            self.output_dim = hidden_dim

    @staticmethod
    def _make_conv(gnn_type: str, in_channels: int, out_channels: int) -> nn.Module:
        if gnn_type == "GAT":
            return GATConv(in_channels, out_channels, edge_dim=1)
        elif gnn_type == "GCN":
            return GCNConv(in_channels, out_channels)
        elif gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp)
        elif gnn_type == "GINE":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINEConv(mlp, edge_dim=1)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_outputs = [x]  # Include raw input (dim d) in JK
        h = x
        for conv in self.convs:
            if self.gnn_type == "GAT" and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            elif self.gnn_type == "GCN" and edge_attr is not None:
                h = conv(h, edge_index, edge_weight=edge_attr.squeeze(-1))
            elif self.gnn_type == "GINE" and edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = self.activation(h)
            layer_outputs.append(h)

        return self.jk(layer_outputs)
