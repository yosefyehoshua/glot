# tests/test_glot_pooler.py
import torch
import pytest
from glot.glot_pooler import GLOTPooler


class TestGLOTPooler:
    def test_output_shape_default_config(self):
        """Default config: hidden=128, layers=2, jk=cat -> output 384."""
        pooler = GLOTPooler(input_dim=768)
        hidden = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 384)

    def test_output_dim_attribute(self):
        """output_dim attribute matches actual output."""
        pooler = GLOTPooler(input_dim=768, hidden_dim=128, num_gnn_layers=2, jk_mode='cat')
        assert pooler.output_dim == 384

    def test_handles_padding(self):
        """Padding tokens should not affect the output shape."""
        pooler = GLOTPooler(input_dim=32, hidden_dim=16, num_gnn_layers=2)
        hidden = torch.randn(2, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (2, 48)

    def test_single_token_sentence(self):
        """A sentence with only 1 valid token should still produce output."""
        pooler = GLOTPooler(input_dim=16, hidden_dim=8, num_gnn_layers=2)
        hidden = torch.randn(1, 5, 16)
        mask = torch.tensor([[1, 0, 0, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 24)

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
