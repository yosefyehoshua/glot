# tests/test_graph_construction.py
import torch
import pytest
from glot.graph_construction import build_token_graph


class TestBuildTokenGraph:
    def test_output_is_pyg_batch(self):
        """build_token_graph returns a PyG Batch object."""
        from torch_geometric.data import Batch
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert isinstance(batch, Batch)

    def test_batch_has_correct_num_graphs(self):
        """Batch contains one graph per sentence."""
        hidden = torch.randn(3, 4, 8)
        mask = torch.ones(3, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.num_graphs == 3

    def test_node_features_match_valid_tokens(self):
        """Node features are the hidden states of valid (non-padding) tokens."""
        hidden = torch.randn(1, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.x.shape == (3, 8)
        assert torch.allclose(batch.x, hidden[0, :3])

    def test_padding_tokens_excluded(self):
        """Padding tokens (mask=0) are not included as nodes."""
        hidden = torch.randn(2, 6, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.x.shape[0] == 5

    def test_high_threshold_produces_fewer_edges(self):
        """Higher threshold creates sparser graphs."""
        hidden = torch.randn(1, 10, 16)
        mask = torch.ones(1, 10, dtype=torch.long)
        batch_low = build_token_graph(hidden, mask, threshold=0.1)
        batch_high = build_token_graph(hidden, mask, threshold=0.9)
        assert batch_high.edge_index.shape[1] <= batch_low.edge_index.shape[1]

    def test_no_self_loops(self):
        """Diagonal of adjacency is zeroed, so no self-loops."""
        hidden = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        src, dst = batch.edge_index
        assert (src != dst).all()

    def test_identical_vectors_fully_connected(self):
        """Identical token vectors should all be connected at any threshold < 1."""
        hidden = torch.ones(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.5)
        assert batch.edge_index.shape[1] == 12

    def test_batch_vector_correct(self):
        """The batch vector assigns nodes to correct graphs."""
        hidden = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        batch = build_token_graph(hidden, mask, threshold=0.0)
        expected_batch = torch.tensor([0, 0, 0, 1, 1])
        assert torch.equal(batch.batch, expected_batch)

    def test_edge_attr_present(self):
        """Graph edges should have cosine similarity as edge_attr."""
        hidden = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.0)
        assert batch.edge_attr is not None
        assert batch.edge_attr.shape == (batch.edge_index.shape[1], 1)

    def test_edge_attr_values_are_similarities(self):
        """edge_attr values should be cosine similarities > threshold."""
        hidden = torch.randn(1, 4, 16)
        mask = torch.ones(1, 4, dtype=torch.long)
        threshold = 0.3
        batch = build_token_graph(hidden, mask, threshold=threshold)
        if batch.edge_attr.numel() > 0:
            assert (batch.edge_attr > threshold).all()
            assert (batch.edge_attr <= 1.0).all()

    def test_edge_attr_matches_edge_count(self):
        """Number of edge_attr values equals number of edges."""
        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        batch = build_token_graph(hidden, mask, threshold=0.5)
        assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
