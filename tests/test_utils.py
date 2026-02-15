import torch
import pytest
import yaml
import os
from glot.utils import compute_metrics, load_config, GLUE_TASKS


class TestComputeMetrics:
    def test_accuracy(self):
        preds = [1, 0, 1, 1]
        labels = [1, 0, 1, 0]
        score = compute_metrics(preds, labels, "accuracy")
        assert score == pytest.approx(75.0)

    def test_mcc_perfect(self):
        preds = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        score = compute_metrics(preds, labels, "mcc")
        assert score == pytest.approx(100.0)

    def test_f1(self):
        preds = [1, 1, 0, 0]
        labels = [1, 0, 1, 0]
        score = compute_metrics(preds, labels, "f1")
        assert score == pytest.approx(50.0)

    def test_spearman(self):
        preds = [1.0, 2.0, 3.0, 4.0]
        labels = [1.0, 2.0, 3.0, 4.0]
        score = compute_metrics(preds, labels, "spearman")
        assert score == pytest.approx(100.0)


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        config = {"backbone": {"name": "bert-base-uncased"}, "training": {"lr": 0.001}}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        loaded = load_config(str(path))
        assert loaded["backbone"]["name"] == "bert-base-uncased"
        assert loaded["training"]["lr"] == 0.001


class TestGlueTasks:
    def test_sst2_config(self):
        cfg = GLUE_TASKS["sst2"]
        assert cfg["type"] == "single"
        assert cfg["num_classes"] == 2
        assert cfg["metric"] == "accuracy"

    def test_mrpc_config(self):
        cfg = GLUE_TASKS["mrpc"]
        assert cfg["type"] == "pair"
        assert cfg["num_classes"] == 2
        assert cfg["metric"] == "f1"

    def test_cola_config(self):
        cfg = GLUE_TASKS["cola"]
        assert cfg["type"] == "single"
        assert cfg["metric"] == "mcc"
