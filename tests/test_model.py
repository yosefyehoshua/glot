import torch
import pytest
from glot.model import create_pooler_and_head


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
