import torch.nn as nn
from glot.graph_construction import build_token_graph
from glot.token_gnn import TokenGNN
from glot.readout import AttentionReadout
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler


class GLOTPooler(nn.Module):
    """GLOT pooling module: graph construction + GNN + readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        jk_mode: str = "cat",
        threshold: float = 0.6,
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.threshold = threshold
        self.token_gnn = TokenGNN(
            input_dim, hidden_dim, num_gnn_layers, jk_mode, gnn_type=gnn_type,
        )
        self.readout = AttentionReadout(self.token_gnn.output_dim)
        self.output_dim = self.token_gnn.output_dim

    def forward(self, hidden_states, attention_mask):
        batch_data = build_token_graph(hidden_states, attention_mask, self.threshold)
        refined = self.token_gnn(
            batch_data.x, batch_data.edge_index, edge_attr=batch_data.edge_attr,
        )
        return self.readout(refined, batch_data.batch)


def create_pooler_and_head(
    pooler_type: str,
    input_dim: int,
    num_classes: int,
    task_type: str = "classification",
    glot_config: dict | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Create a pooler and task head.

    Returns:
        (pooler, head) tuple. The pooler maps (B, L, d) -> (B, D).
        The head maps (B, D) -> (B, num_classes) for classification,
        or (B, 2*D) -> (B, num_classes) for pair classification.
    """
    if pooler_type == "glot":
        pooler = GLOTPooler(input_dim=input_dim, **(glot_config or {}))
        pool_dim = pooler.output_dim
    elif pooler_type == "mean":
        pooler = MeanPooler()
        pool_dim = input_dim
    elif pooler_type == "max":
        pooler = MaxPooler()
        pool_dim = input_dim
    elif pooler_type == "cls":
        pooler = CLSPooler(is_decoder=False)
        pool_dim = input_dim
    elif pooler_type == "eos":
        pooler = EOSPooler()
        pool_dim = input_dim
    elif pooler_type == "adapool":
        pooler = AdaPool(input_dim)
        pool_dim = input_dim
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")

    if task_type == "pair_classification":
        head = nn.Linear(pool_dim * 2, num_classes)
    else:
        head = nn.Linear(pool_dim, num_classes)

    return pooler, head
