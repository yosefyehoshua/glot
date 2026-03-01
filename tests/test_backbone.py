import pytest
from glot.backbone import BACKBONE_REGISTRY, get_backbone_config


class TestBackboneRegistry:
    def test_bert_is_encoder(self):
        cfg = get_backbone_config("bert-base-uncased")
        assert cfg["type"] == "encoder"
        assert cfg["hidden_dim"] == 768
        assert cfg["pooling_token"] == "cls"

    def test_roberta_is_encoder(self):
        cfg = get_backbone_config("roberta-base")
        assert cfg["type"] == "encoder"
        assert cfg["hidden_dim"] == 768

    def test_tinyllama_is_decoder(self):
        cfg = get_backbone_config("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert cfg["type"] == "decoder"
        assert cfg["hidden_dim"] == 2048
        assert cfg["pooling_token"] == "eos"

    def test_smollm2_is_decoder(self):
        cfg = get_backbone_config("HuggingFaceTB/SmolLM2-1.7B")
        assert cfg["type"] == "decoder"
        assert cfg["hidden_dim"] == 2048

    def test_llama3b_is_decoder(self):
        cfg = get_backbone_config("meta-llama/Llama-3.2-3B")
        assert cfg["type"] == "decoder"
        assert cfg["hidden_dim"] == 3072

    def test_mistral7b_is_decoder(self):
        cfg = get_backbone_config("mistralai/Mistral-7B-v0.1")
        assert cfg["type"] == "decoder"
        assert cfg["hidden_dim"] == 4096

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            get_backbone_config("nonexistent-model")

    def test_all_backbones_have_params(self):
        for name, cfg in BACKBONE_REGISTRY.items():
            assert "params" in cfg
            assert isinstance(cfg["params"], (int, float))
