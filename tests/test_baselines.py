# tests/test_baselines.py
import torch
import pytest
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool


class TestMeanPooler:
    def test_output_shape(self):
        pooler = MeanPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_ignores_padding(self):
        pooler = MeanPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[2.0, 3.0]])
        assert torch.allclose(z, expected)


class TestMaxPooler:
    def test_output_shape(self):
        pooler = MaxPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_ignores_padding(self):
        pooler = MaxPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
        mask = torch.tensor([[1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(z, expected)


class TestCLSPooler:
    def test_encoder_returns_first_token(self):
        pooler = CLSPooler(is_decoder=False)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.ones(1, 3, dtype=torch.long)
        z = pooler(hidden, mask)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(z, expected)

    def test_decoder_returns_last_valid_token(self):
        pooler = CLSPooler(is_decoder=True)
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(z, expected)


class TestAdaPool:
    def test_output_shape(self):
        pooler = AdaPool(input_dim=768)
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_has_trainable_params(self):
        pooler = AdaPool(input_dim=64)
        params = list(pooler.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_default_hidden_dim_is_128(self):
        """AdaPool default hidden_dim should be 128 (matching original code)."""
        pooler = AdaPool(input_dim=768)
        first_linear = pooler.scorer[0]
        assert first_linear.out_features == 128

    def test_ignores_padding(self):
        pooler = AdaPool(input_dim=4)
        hidden = torch.randn(1, 5, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        z = pooler(hidden, mask)
        assert z.shape == (1, 4)
        hidden2 = hidden.clone()
        hidden2[0, 3:] = 999.0
        z2 = pooler(hidden2, mask)
        assert torch.allclose(z, z2)


class TestEOSPooler:
    def test_output_shape(self):
        from glot.baselines import EOSPooler

        pooler = EOSPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_returns_last_valid_token(self):
        from glot.baselines import EOSPooler

        pooler = EOSPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(z, expected)

    def test_different_lengths_in_batch(self):
        from glot.baselines import EOSPooler

        pooler = EOSPooler()
        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[3.0, 4.0], [9.0, 10.0]])
        assert torch.allclose(z, expected)
