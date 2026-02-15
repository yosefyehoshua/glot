# tests/test_token_gnn.py
import torch
import pytest
from glot.token_gnn import TokenGNN


class TestTokenGNN:
    def test_output_shape_jk_cat(self):
        """With JK='cat' and 2 layers, output dim = hidden * 3."""
        gnn = TokenGNN(input_dim=768, hidden_dim=128, num_layers=2, jk_mode='cat')
        x = torch.randn(10, 768)
        edge_index = torch.tensor([list(range(9)) + list(range(1, 10)),
                                   list(range(1, 10)) + list(range(9))], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (10, 384)

    def test_output_dim_attribute(self):
        """output_dim attribute matches actual output dimension."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=2, jk_mode='cat')
        assert gnn.output_dim == 64 * 3

    def test_output_dim_4_layers(self):
        """With 4 layers and JK='cat', output dim = hidden * 5."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=4, jk_mode='cat')
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 64 * 5)
        assert gnn.output_dim == 64 * 5

    def test_no_edges_still_works(self):
        """GNN should handle isolated nodes (no edges) gracefully."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode='cat')
        x = torch.randn(5, 16)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 24)

    def test_gradients_flow(self):
        """Gradients should flow through all parameters."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode='cat')
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        loss = out.sum()
        loss.backward()
        for name, param in gnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_input_dims(self):
        """Should work with various LLM hidden dimensions."""
        for input_dim in [768, 2048, 4096]:
            gnn = TokenGNN(input_dim=input_dim, hidden_dim=128, num_layers=2, jk_mode='cat')
            x = torch.randn(3, input_dim)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = gnn(x, edge_index)
            assert out.shape == (3, 384)
