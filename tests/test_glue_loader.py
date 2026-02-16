import pytest
from data.glue_loader import load_glue_task, get_task_config


class TestGetTaskConfig:
    def test_sst2_single_sentence(self):
        cfg = get_task_config("sst2")
        assert cfg["type"] == "single"
        assert cfg["sentence_keys"] == ("sentence",)

    def test_mrpc_pair_sentence(self):
        cfg = get_task_config("mrpc")
        assert cfg["type"] == "pair"
        assert cfg["sentence_keys"] == ("sentence1", "sentence2")

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            get_task_config("nonexistent_task")


class TestLoadGlueTask:
    """These tests download data so they are marked slow.
    Run with: pytest -m 'not slow' to skip, or pytest to include."""

    @pytest.mark.slow
    def test_sst2_loads(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_glue_task("sst2", tokenizer, max_length=32)
        assert "train" in dataset
        assert "validation" in dataset
        assert "input_ids" in dataset["train"].column_names

    @pytest.mark.slow
    def test_mrpc_pair_loads_separate(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_glue_task("mrpc", tokenizer, max_length=32)
        assert "input_ids_a" in dataset["train"].column_names
        assert "input_ids_b" in dataset["train"].column_names
        assert "attention_mask_a" in dataset["train"].column_names
