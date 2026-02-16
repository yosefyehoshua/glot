import torch
import pytest
from torch.utils.data import DataLoader
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics
from data.cache import CachedDataset
from train import train_epoch, evaluate_epoch


class TestEndToEnd:
    def test_glot_trains_and_evaluates_single_task(self):
        """Full pipeline: create model, train on synthetic cached data, evaluate."""
        torch.manual_seed(42)
        input_dim = 64
        n_train, n_val = 40, 10
        seq_len = 8

        # Synthetic cached hidden states
        train_hs = torch.randn(n_train, seq_len, input_dim)
        train_masks = torch.ones(n_train, seq_len, dtype=torch.long)
        train_labels = torch.randint(0, 2, (n_train,))

        val_hs = torch.randn(n_val, seq_len, input_dim)
        val_masks = torch.ones(n_val, seq_len, dtype=torch.long)
        val_labels = torch.randint(0, 2, (n_val,))

        train_ds = CachedDataset(train_hs, train_masks, train_labels)
        val_ds = CachedDataset(val_hs, val_masks, val_labels)
        train_loader = DataLoader(train_ds, batch_size=10)
        val_loader = DataLoader(val_ds, batch_size=10)

        pooler, head = create_pooler_and_head(
            "glot", input_dim=input_dim, num_classes=2,
            task_type="classification",
            glot_config={"hidden_dim": 16, "num_gnn_layers": 1},
        )

        optimizer = torch.optim.Adam(
            list(pooler.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train 3 epochs
        for _ in range(3):
            train_epoch(pooler, head, train_loader, optimizer, loss_fn, "classification")

        # Evaluate
        preds, labels = evaluate_epoch(pooler, head, val_loader, "classification")
        score = compute_metrics(preds, labels, "accuracy")
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_glot_pair_task_e2e(self):
        """Full pipeline for a pair classification task."""
        torch.manual_seed(42)
        input_dim = 32
        n = 30
        seq_len = 6

        ds = CachedDataset(
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
            torch.randint(0, 2, (n,)),
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=10)

        pooler, head = create_pooler_and_head(
            "glot", input_dim=input_dim, num_classes=2,
            task_type="pair_classification",
            glot_config={"hidden_dim": 8, "num_gnn_layers": 1},
        )

        optimizer = torch.optim.Adam(
            list(pooler.parameters()) + list(head.parameters()), lr=1e-3
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "pair_classification")
        assert isinstance(loss, float)
        assert loss > 0

        preds, labels = evaluate_epoch(pooler, head, loader, "pair_classification")
        assert len(preds) == n

    def test_all_baselines_train(self):
        """All baseline poolers can train on synthetic data without error."""
        torch.manual_seed(42)
        input_dim = 32
        n = 20
        seq_len = 4

        train_ds = CachedDataset(
            torch.randn(n, seq_len, input_dim),
            torch.ones(n, seq_len, dtype=torch.long),
            torch.randint(0, 2, (n,)),
        )
        loader = DataLoader(train_ds, batch_size=10)

        for pooler_type in ["glot", "mean", "max", "cls", "adapool"]:
            glot_cfg = {"hidden_dim": 8, "num_gnn_layers": 1} if pooler_type == "glot" else None
            pooler, head = create_pooler_and_head(
                pooler_type, input_dim=input_dim, num_classes=2,
                task_type="classification", glot_config=glot_cfg,
            )
            optimizer = torch.optim.Adam(
                list(pooler.parameters()) + list(head.parameters()), lr=1e-3
            )
            loss_fn = torch.nn.CrossEntropyLoss()

            loss = train_epoch(pooler, head, loader, optimizer, loss_fn, "classification")
            assert isinstance(loss, float), f"Failed for {pooler_type}"
