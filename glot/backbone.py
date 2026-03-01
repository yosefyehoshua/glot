"""Backbone model loading and configuration registry."""
import torch

BACKBONE_REGISTRY = {
    "bert-base-uncased": {
        "type": "encoder",
        "hidden_dim": 768,
        "pooling_token": "cls",
        "params": 110e6,
    },
    "roberta-base": {
        "type": "encoder",
        "hidden_dim": 768,
        "pooling_token": "cls",
        "params": 125e6,
    },
    "HuggingFaceTB/SmolLM2-1.7B": {
        "type": "decoder",
        "hidden_dim": 2048,
        "pooling_token": "eos",
        "params": 1.7e9,
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "type": "decoder",
        "hidden_dim": 2048,
        "pooling_token": "eos",
        "params": 1.1e9,
    },
    "meta-llama/Llama-3.2-3B": {
        "type": "decoder",
        "hidden_dim": 3072,
        "pooling_token": "eos",
        "params": 3.2e9,
    },
    "mistralai/Mistral-7B-v0.1": {
        "type": "decoder",
        "hidden_dim": 4096,
        "pooling_token": "eos",
        "params": 7.2e9,
    },
}


def get_backbone_config(name: str) -> dict:
    """Return config dict for a backbone by name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[name]


def load_backbone(name: str, device: str = "cpu", dtype=None):
    """Load a frozen backbone model and tokenizer.

    Args:
        name: HuggingFace model name (must be in BACKBONE_REGISTRY).
        device: Device to load model on.
        dtype: Optional torch dtype (e.g. torch.float16 for large models).

    Returns:
        (model, tokenizer, config) tuple. Model has requires_grad=False.
    """
    from transformers import AutoModel, AutoTokenizer

    cfg = get_backbone_config(name)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(name, **model_kwargs).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)
    if cfg["type"] == "decoder":
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, cfg
