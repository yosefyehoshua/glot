import os
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Dataset wrapping precomputed hidden states.

    Supports both single-sentence and pair tasks.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
        hidden_states_b: torch.Tensor | None = None,
        attention_masks_b: torch.Tensor | None = None,
    ):
        self.hidden_states = hidden_states
        self.attention_masks = attention_masks
        self.labels = labels
        self.hidden_states_b = hidden_states_b
        self.attention_masks_b = attention_masks_b
        self.is_pair = hidden_states_b is not None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_pair:
            return (
                self.hidden_states[idx],
                self.attention_masks[idx],
                self.hidden_states_b[idx],
                self.attention_masks_b[idx],
                self.labels[idx],
            )
        return (
            self.hidden_states[idx],
            self.attention_masks[idx],
            self.labels[idx],
        )


def save_cache(
    path: str,
    hidden_states: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: torch.Tensor,
    hidden_states_b: torch.Tensor | None = None,
    attention_masks_b: torch.Tensor | None = None,
) -> None:
    """Save cached hidden states to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "hidden_states": hidden_states,
        "attention_masks": attention_masks,
        "labels": labels,
    }
    if hidden_states_b is not None:
        data["hidden_states_b"] = hidden_states_b
        data["attention_masks_b"] = attention_masks_b
    torch.save(data, path)


def load_cache(path: str) -> dict:
    """Load cached hidden states from disk."""
    return torch.load(path, weights_only=True)


def make_cached_dataset(path: str) -> CachedDataset:
    """Load a cache file and return a CachedDataset."""
    data = load_cache(path)
    return CachedDataset(
        data["hidden_states"],
        data["attention_masks"],
        data["labels"],
        data.get("hidden_states_b"),
        data.get("attention_masks_b"),
    )
