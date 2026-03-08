import torch
import pytest
from glot.token_gnn import TokenGNN


class TestTokenGNN:
    def test_output_shape_jk_cat_bert(self):
        """With BERT (d=768), K=2, p=128, JK=cat: output = 768 + 2*128 = 1024."""
        gnn = TokenGNN(input_dim=768, hidden_dim=128, num_layers=2, jk_mode="cat")
        x = torch.randn(10, 768)
        edge_index = torch.tensor(
            [list(range(9)) + list(range(1, 10)),
             list(range(1, 10)) + list(range(9))],
            dtype=torch.long,
        )
        out = gnn(x, edge_index)
        assert out.shape == (10, 1024)

    def test_output_dim_attribute(self):
        """output_dim attribute matches d + K*p."""
        gnn = TokenGNN(input_dim=32, hidden_dim=64, num_layers=2, jk_mode="cat")
        assert gnn.output_dim == 32 + 2 * 64  # 160

    def test_output_dim_4_layers(self):
        """With K=4, p=256: output = d + 4*256."""
        gnn = TokenGNN(input_dim=768, hidden_dim=256, num_layers=4, jk_mode="cat")
        x = torch.randn(5, 768)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 768 + 4 * 256)  # 1792
        assert gnn.output_dim == 1792

    def test_no_edges_still_works(self):
        """GNN should handle isolated nodes (no edges) gracefully."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat")
        x = torch.randn(5, 16)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 16 + 2 * 8)  # 32

    def test_gradients_flow(self):
        """Gradients should flow through all parameters."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat")
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 1)
        out = gnn(x, edge_index, edge_attr=edge_attr)
        loss = out.sum()
        loss.backward()
        for name, param in gnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_input_dims(self):
        """Should work with various LLM hidden dimensions."""
        for input_dim in [768, 2048, 4096]:
            gnn = TokenGNN(input_dim=input_dim, hidden_dim=128, num_layers=2, jk_mode="cat")
            x = torch.randn(3, input_dim)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            out = gnn(x, edge_index)
            assert out.shape == (3, input_dim + 2 * 128)

    def test_gnn_type_gcn(self):
        """GCN backend should work."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GCN")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 32 + 2 * 16)

    def test_gnn_type_gin(self):
        """GIN backend should work."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GIN")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        out = gnn(x, edge_index)
        assert out.shape == (5, 32 + 2 * 16)

    def test_gnn_type_gine(self):
        """GINE backend should work with edge_attr."""
        gnn = TokenGNN(input_dim=32, hidden_dim=16, num_layers=2, jk_mode="cat", gnn_type="GINE")
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 1)
        out = gnn(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (5, 32 + 2 * 16)

    def test_edge_attr_passed_to_gat(self):
        """GAT should accept edge_attr without error."""
        gnn = TokenGNN(input_dim=16, hidden_dim=8, num_layers=2, jk_mode="cat", gnn_type="GAT")
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_attr = torch.randn(3, 1)
        out = gnn(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (5, 16 + 2 * 8)
