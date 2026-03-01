import torch.nn as nn
from glot.glot_pooler import GLOTPooler
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler


def create_pooler_and_head(
    pooler_type: str,
    input_dim: int,
    num_classes: int,
    task_type: str = "classification",
    glot_config: dict | None = None,
) -> tuple[nn.Module, nn.Module]:
    """Create a pooler and task head.

    Returns:
        (pooler, head) tuple. The pooler maps (B, L, d) -> (B, D).
        The head maps (B, D) -> (B, num_classes) for classification,
        or (B, 2*D) -> (B, num_classes) for pair classification.
    """
    if pooler_type == "glot":
        pooler = GLOTPooler(input_dim=input_dim, **(glot_config or {}))
        pool_dim = pooler.output_dim
    elif pooler_type == "mean":
        pooler = MeanPooler()
        pool_dim = input_dim
    elif pooler_type == "max":
        pooler = MaxPooler()
        pool_dim = input_dim
    elif pooler_type == "cls":
        pooler = CLSPooler(is_decoder=False)
        pool_dim = input_dim
    elif pooler_type == "eos":
        pooler = EOSPooler()
        pool_dim = input_dim
    elif pooler_type == "adapool":
        pooler = AdaPool(input_dim)
        pool_dim = input_dim
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")

    if task_type == "pair_classification":
        head = nn.Linear(pool_dim * 2, num_classes)
    else:
        head = nn.Linear(pool_dim, num_classes)

    return pooler, head
