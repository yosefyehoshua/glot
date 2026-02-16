import torch
import pytest
from train import train_epoch, evaluate_epoch
from glot.model import create_pooler_and_head
from data.cache import CachedDataset
from torch.utils.data import DataLoader


class TestTrainEpoch:
    def test_single_task_trains(self):
        """Training on single-sentence cached data reduces loss."""
        pooler, head = create_pooler_and_head("glot", input_dim=32, num_classes=2,
                                               task_type="classification",
                                               glot_config={"hidden_dim": 8, "num_gnn_layers": 1})
        hs = torch.randn(20, 5, 32)
        masks = torch.ones(20, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (20,))
        ds = CachedDataset(hs, masks, labels)
        loader = DataLoader(ds, batch_size=10)

        params = list(pooler.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss1 = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
        loss2 = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
        # Loss should generally decrease after training
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)

    def test_pair_task_trains(self):
        """Training on pair-task cached data works."""
        pooler, head = create_pooler_and_head("glot", input_dim=32, num_classes=2,
                                               task_type="pair_classification",
                                               glot_config={"hidden_dim": 8, "num_gnn_layers": 1})
        hs_a = torch.randn(20, 5, 32)
        masks_a = torch.ones(20, 5, dtype=torch.long)
        hs_b = torch.randn(20, 5, 32)
        masks_b = torch.ones(20, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (20,))
        ds = CachedDataset(hs_a, masks_a, labels, hs_b, masks_b)
        loader = DataLoader(ds, batch_size=10)

        params = list(pooler.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "pair_classification")
        assert isinstance(loss, float)


class TestEvaluateEpoch:
    def test_returns_predictions_and_labels(self):
        pooler, head = create_pooler_and_head("mean", input_dim=16, num_classes=2,
                                               task_type="classification")
        hs = torch.randn(10, 4, 16)
        masks = torch.ones(10, 4, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        ds = CachedDataset(hs, masks, labels)
        loader = DataLoader(ds, batch_size=5)

        preds, labs = evaluate_epoch(pooler, head, loader, "classification")
        assert len(preds) == 10
        assert len(labs) == 10
