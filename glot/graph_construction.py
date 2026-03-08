import torch
from torch_geometric.data import Data, Batch


def build_token_graph(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.3,
) -> Batch:
    """Construct token-similarity graphs from hidden states.

    Matches the original GLOT code: binary threshold on cosine similarity,
    self-loops included, edge_attr stores binary 1.0 values (not raw similarity).

    Args:
        hidden_states: (B, L, d) token hidden states from frozen LLM.
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding.
        threshold: cosine similarity threshold for edge creation (default 0.3).

    Returns:
        PyG Batch containing B graphs with binary edge_attr.
    """
    graphs = []
    for i in range(hidden_states.size(0)):
        mask = attention_mask[i].bool()
        h = hidden_states[i][mask]  # (L', d)

        # Pairwise cosine similarity
        sim = torch.nn.functional.cosine_similarity(
            h.unsqueeze(1), h.unsqueeze(0), dim=-1,
        )  # (L', L')

        # Binary adjacency (self-loops kept, matching original)
        adj = (sim > threshold).float()

        # COO edge list via dense_to_sparse
        edge_index = adj.nonzero(as_tuple=False).T.contiguous()  # (2, |E|)

        # Edge attr = binary 1.0 for all surviving edges (matching original)
        edge_attr = torch.ones(edge_index.shape[1], 1, device=h.device)

        graphs.append(Data(x=h, edge_index=edge_index, edge_attr=edge_attr))

    return Batch.from_data_list(graphs)
