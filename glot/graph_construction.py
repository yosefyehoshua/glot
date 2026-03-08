import torch
from torch_geometric.data import Data, Batch


def build_token_graph(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.6,
) -> Batch:
    """Construct edge-weighted token-similarity graphs from hidden states.

    Args:
        hidden_states: (B, L, d) token hidden states from frozen LLM.
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding.
        threshold: cosine similarity threshold for edge creation.

    Returns:
        PyG Batch containing B graphs with edge_attr (cosine similarity).
    """
    graphs = []
    for i in range(hidden_states.size(0)):
        mask = attention_mask[i].bool()
        h = hidden_states[i][mask]  # (L', d)

        # Pairwise cosine similarity
        h_norm = torch.nn.functional.normalize(h, p=2, dim=-1)
        sim = h_norm @ h_norm.T  # (L', L')

        # Threshold to binary adjacency, remove self-loops
        adj = (sim > threshold).long()
        adj.fill_diagonal_(0)

        # COO edge list
        edge_index = adj.nonzero(as_tuple=False).T.contiguous()  # (2, |E|)

        # Edge weights = cosine similarity for surviving edges
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            edge_attr = sim[src, dst].unsqueeze(-1)  # (|E|, 1)
        else:
            edge_attr = torch.zeros(0, 1, device=h.device)

        graphs.append(Data(x=h, edge_index=edge_index, edge_attr=edge_attr))

    return Batch.from_data_list(graphs)
