# tests/test_readout.py
import torch
import pytest
from glot.readout import AttentionReadout


class TestAttentionReadout:
    def test_output_shape(self):
        """Output is (B, D) where B = number of graphs."""
        readout = AttentionReadout(input_dim=384)
        x = torch.randn(7, 384)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        z = readout(x, batch)
        assert z.shape == (2, 384)

    def test_single_node_graph(self):
        """Graph with one node should return that node's representation (scaled)."""
        readout = AttentionReadout(input_dim=8)
        x = torch.randn(1, 8)
        batch = torch.tensor([0])
        z = readout(x, batch)
        assert z.shape == (1, 8)
        assert torch.allclose(z, x, atol=1e-6)

    def test_weights_sum_to_one_per_graph(self):
        """Attention weights within each graph should sum to 1."""
        readout = AttentionReadout(input_dim=16)
        x = torch.randn(5, 16)
        batch = torch.tensor([0, 0, 0, 1, 1])
        scores = readout.attention(x).squeeze(-1)
        from torch_geometric.utils import softmax as pyg_softmax
        weights = pyg_softmax(scores, batch)
        from torch_geometric.nn import global_add_pool
        weight_sums = global_add_pool(weights.unsqueeze(-1), batch).squeeze(-1)
        assert torch.allclose(weight_sums, torch.ones(2), atol=1e-5)

    def test_gradients_flow(self):
        """Gradients should flow through readout parameters."""
        readout = AttentionReadout(input_dim=16)
        x = torch.randn(5, 16, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 1, 1])
        z = readout(x, batch)
        loss = z.sum()
        loss.backward()
        for name, param in readout.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_batch_of_one(self):
        """Single graph in batch should work correctly."""
        readout = AttentionReadout(input_dim=32)
        x = torch.randn(4, 32)
        batch = torch.zeros(4, dtype=torch.long)
        z = readout(x, batch)
        assert z.shape == (1, 32)
