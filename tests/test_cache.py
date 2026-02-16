import torch
import pytest
import os
from data.cache import CachedDataset, save_cache, load_cache


class TestCachedDataset:
    def test_single_task_getitem(self):
        hs = torch.randn(100, 10, 32)
        masks = torch.ones(100, 10, dtype=torch.long)
        labels = torch.randint(0, 2, (100,))
        ds = CachedDataset(hs, masks, labels)
        assert len(ds) == 100
        h, m, l = ds[0]
        assert h.shape == (10, 32)
        assert m.shape == (10,)

    def test_pair_task_getitem(self):
        hs_a = torch.randn(50, 10, 32)
        masks_a = torch.ones(50, 10, dtype=torch.long)
        hs_b = torch.randn(50, 10, 32)
        masks_b = torch.ones(50, 10, dtype=torch.long)
        labels = torch.randint(0, 2, (50,))
        ds = CachedDataset(hs_a, masks_a, labels, hs_b, masks_b)
        assert len(ds) == 50
        result = ds[0]
        assert len(result) == 5  # hs_a, mask_a, hs_b, mask_b, label


class TestSaveLoadCache:
    def test_roundtrip_single(self, tmp_path):
        hs = torch.randn(10, 5, 16)
        masks = torch.ones(10, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        path = str(tmp_path / "cache.pt")
        save_cache(path, hs, masks, labels)
        loaded = load_cache(path)
        assert torch.equal(loaded["hidden_states"], hs)
        assert torch.equal(loaded["attention_masks"], masks)
        assert torch.equal(loaded["labels"], labels)

    def test_roundtrip_pair(self, tmp_path):
        hs_a = torch.randn(10, 5, 16)
        masks_a = torch.ones(10, 5, dtype=torch.long)
        hs_b = torch.randn(10, 5, 16)
        masks_b = torch.ones(10, 5, dtype=torch.long)
        labels = torch.randint(0, 2, (10,))
        path = str(tmp_path / "cache.pt")
        save_cache(path, hs_a, masks_a, labels, hs_b, masks_b)
        loaded = load_cache(path)
        assert "hidden_states_b" in loaded
        assert torch.equal(loaded["hidden_states_b"], hs_b)
