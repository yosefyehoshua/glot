import torch
import pytest
from glot.model import create_pooler_and_head, GLOTPooler


class TestCreatePoolerAndHead:
    def test_glot_classification(self):
        pooler, head = create_pooler_and_head(
            pooler_type="glot", input_dim=768, num_classes=2,
            task_type="classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hs, mask)
        logits = head(z)
        assert logits.shape == (2, 2)

    def test_mean_classification(self):
        pooler, head = create_pooler_and_head(
            pooler_type="mean", input_dim=768, num_classes=2,
            task_type="classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hs, mask)
        logits = head(z)
        assert logits.shape == (2, 2)

    def test_pair_classification_head(self):
        pooler, head = create_pooler_and_head(
            pooler_type="glot", input_dim=768, num_classes=2,
            task_type="pair_classification",
        )
        hs = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z_a = pooler(hs, mask)
        z_b = pooler(hs, mask)
        combined = torch.cat([z_a, z_b], dim=-1)
        logits = head(combined)
        assert logits.shape == (2, 2)

    def test_all_pooler_types(self):
        for ptype in ["glot", "mean", "max", "cls", "adapool"]:
            pooler, head = create_pooler_and_head(
                pooler_type=ptype, input_dim=64, num_classes=3,
                task_type="classification",
            )
            hs = torch.randn(2, 5, 64)
            mask = torch.ones(2, 5, dtype=torch.long)
            z = pooler(hs, mask)
            logits = head(z)
            assert logits.shape == (2, 3), f"Failed for {ptype}"

    def test_eos_pooler_creation(self):
        pooler, head = create_pooler_and_head(
            "eos", input_dim=768, num_classes=2, task_type="classification"
        )
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)
        logits = head(z)
        assert logits.shape == (2, 2)

    def test_unknown_pooler_raises(self):
        with pytest.raises(ValueError):
            create_pooler_and_head(
                pooler_type="unknown", input_dim=64, num_classes=2,
                task_type="classification",
            )


class TestGLOTPooler:
    def test_output_shape_default_config(self):
        """Default config: d=768, hidden=128, layers=2, jk=cat -> 768 + 2*128 = 1024."""
        pooler = GLOTPooler(input_dim=768)
        hidden = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 1024)

    def test_output_dim_attribute(self):
        """output_dim = input_dim + num_layers * hidden_dim."""
        pooler = GLOTPooler(input_dim=768, hidden_dim=128, num_gnn_layers=2, jk_mode="cat")
        assert pooler.output_dim == 768 + 2 * 128

    def test_handles_padding(self):
        """Padding tokens should not affect the output shape."""
        pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2)
        hidden = torch.randn(2, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (2, 32 + 2 * 16)  # 64

    def test_single_token_sentence(self):
        """A sentence with only 1 valid token should still produce output."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(1, 5, 16)
        mask = torch.tensor([[1, 0, 0, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 16 + 2 * 8)  # 32

    def test_gradients_flow_through_pooler(self):
        """Gradients should flow through the entire pooler pipeline."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        loss = z.sum()
        loss.backward()
        for name, param in pooler.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_threshold_parameter_used(self):
        """Different thresholds should produce different outputs."""
        torch.manual_seed(42)
        hidden = torch.randn(1, 10, 32)
        mask = torch.ones(1, 10, dtype=torch.long)

        pooler_low = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, threshold=0.1)
        pooler_high = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, threshold=0.9)

        pooler_high.load_state_dict(pooler_low.state_dict())

        z_low = pooler_low(hidden, mask)
        z_high = pooler_high(hidden, mask)
        assert not torch.allclose(z_low, z_high, atol=1e-4)

    def test_gnn_type_parameter(self):
        """gnn_type parameter should be passed to TokenGNN."""
        for gnn_type in ["GAT", "GCN", "GIN"]:
            pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2, gnn_type=gnn_type)
            hidden = torch.randn(2, 5, 32)
            mask = torch.ones(2, 5, dtype=torch.long)
            z = pooler(hidden, mask)
            assert z.shape == (2, 32 + 2 * 16)
