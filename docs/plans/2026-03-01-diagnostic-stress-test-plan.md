# Diagnostic Stress Test + Decoder Backbone Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the signal dilution diagnostic experiment (Table 7 / Figure 3 from the paper) with all 6 backbones, 5 poolers, and 4 distractor ratios, plus decoder backbone support.

**Architecture:** Synthetic data generation produces `(text, label)` pairs. A backbone loader handles encoder/decoder differences (CLS vs EOS, padding side). The diagnostic script tokenizes text, runs it through the frozen backbone, trains pooler + head on the hidden states, and evaluates accuracy. Results are logged to both JSON and W&B.

**Tech Stack:** PyTorch, PyG, Transformers, NLTK (distractor vocab), wandb, matplotlib

---

### Task 1: Add EOSPooler to baselines

**Files:**
- Modify: `glot/baselines.py`
- Modify: `glot/__init__.py`
- Test: `tests/test_baselines.py`

**Step 1: Write the failing tests**

Add to the bottom of `tests/test_baselines.py`:

```python
from glot.baselines import EOSPooler


class TestEOSPooler:
    def test_output_shape(self):
        pooler = EOSPooler()
        hidden = torch.randn(2, 5, 768)
        mask = torch.ones(2, 5, dtype=torch.long)
        z = pooler(hidden, mask)
        assert z.shape == (2, 768)

    def test_returns_last_valid_token(self):
        pooler = EOSPooler()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 1, 0]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(z, expected)

    def test_different_lengths_in_batch(self):
        pooler = EOSPooler()
        hidden = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        z = pooler(hidden, mask)
        expected = torch.tensor([[3.0, 4.0], [9.0, 10.0]])
        assert torch.allclose(z, expected)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_baselines.py::TestEOSPooler -v`
Expected: FAIL with `ImportError: cannot import name 'EOSPooler'`

**Step 3: Write minimal implementation**

Add to `glot/baselines.py` after the `CLSPooler` class:

```python
class EOSPooler(nn.Module):
    """Extract the last non-padding token (EOS position for decoder models)."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lengths]
```

Update `glot/__init__.py` — add `EOSPooler` to the baselines import:

```python
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_baselines.py::TestEOSPooler -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add glot/baselines.py glot/__init__.py tests/test_baselines.py
git commit -m "feat: add EOSPooler for decoder backbone support"
```

---

### Task 2: Add `eos` pooler type to model factory

**Files:**
- Modify: `glot/model.py`
- Test: `tests/test_model.py`

**Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::TestModelFactory::test_eos_pooler_creation -v`
Expected: FAIL with `ValueError: Unknown pooler type: eos`

**Step 3: Write minimal implementation**

In `glot/model.py`, add the import and the elif branch:

Add `EOSPooler` to the import line:
```python
from glot.baselines import MeanPooler, MaxPooler, CLSPooler, AdaPool, EOSPooler
```

Add this elif block after the `"cls"` case (before `"adapool"`):

```python
    elif pooler_type == "eos":
        pooler = EOSPooler()
        pool_dim = input_dim
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add glot/model.py tests/test_model.py
git commit -m "feat: add eos pooler type to model factory"
```

---

### Task 3: Create backbone loader with registry

**Files:**
- Create: `glot/backbone.py`
- Test: `tests/test_backbone.py`

**Step 1: Write the failing tests**

Create `tests/test_backbone.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_backbone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'glot.backbone'`

**Step 3: Write minimal implementation**

Create `glot/backbone.py`:

```python
"""Backbone model loading and configuration registry."""
import torch
from transformers import AutoModel, AutoTokenizer

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
```

Update `glot/__init__.py` — add backbone imports:

```python
from glot.backbone import BACKBONE_REGISTRY, get_backbone_config, load_backbone
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_backbone.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add glot/backbone.py glot/__init__.py tests/test_backbone.py
git commit -m "feat: add backbone registry and loader for encoder/decoder models"
```

---

### Task 4: Create diagnostic data generator

**Files:**
- Create: `data/diagnostic.py`
- Test: `tests/test_diagnostic.py`

**Step 1: Write the failing tests**

Create `tests/test_diagnostic.py`:

```python
import pytest
from data.diagnostic import generate_diagnostic_dataset, SIGNAL_TEMPLATES, CONTENT_WORDS


class TestDiagnosticDataGeneration:
    def test_returns_correct_count(self):
        data = generate_diagnostic_dataset(num_samples=100, seq_length=50, distractor_ratio=0.5, seed=42)
        assert len(data) == 100

    def test_returns_text_label_tuples(self):
        data = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=42)
        for text, label in data:
            assert isinstance(text, str)
            assert label in (0, 1)

    def test_labels_are_balanced(self):
        data = generate_diagnostic_dataset(num_samples=1000, seq_length=50, distractor_ratio=0.5, seed=42)
        labels = [label for _, label in data]
        ratio = sum(labels) / len(labels)
        assert 0.35 < ratio < 0.65, f"Label balance is {ratio}, expected ~0.5"

    def test_word_count_matches_seq_length(self):
        seq_length = 100
        data = generate_diagnostic_dataset(num_samples=50, seq_length=seq_length, distractor_ratio=0.5, seed=42)
        for text, _ in data:
            words = text.split()
            assert len(words) == seq_length, f"Expected {seq_length} words, got {len(words)}"

    def test_signal_phrase_present(self):
        """At low distractor ratio, signal keywords should appear in text."""
        data = generate_diagnostic_dataset(num_samples=100, seq_length=50, distractor_ratio=0.2, seed=42)
        signal_keywords = {"but not", "but lacks", "without", "but excludes",
                          "and also", "and includes", "with"}
        found = 0
        for text, _ in data:
            if any(kw in text for kw in signal_keywords):
                found += 1
        assert found == 100, f"Signal phrase missing in {100 - found} samples"

    def test_different_seeds_produce_different_data(self):
        data1 = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=1)
        data2 = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=2)
        texts1 = [t for t, _ in data1]
        texts2 = [t for t, _ in data2]
        assert texts1 != texts2

    def test_high_distractor_ratio(self):
        data = generate_diagnostic_dataset(num_samples=10, seq_length=256, distractor_ratio=0.9, seed=42)
        assert len(data) == 10
        for text, label in data:
            assert len(text.split()) == 256
            assert label in (0, 1)

    def test_templates_exist(self):
        assert len(SIGNAL_TEMPLATES) == 10
        negation = [t for t, l in SIGNAL_TEMPLATES if l == 0]
        affirm = [t for t, l in SIGNAL_TEMPLATES if l == 1]
        assert len(negation) == 5
        assert len(affirm) == 5

    def test_content_words_exist(self):
        assert len(CONTENT_WORDS) >= 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.diagnostic'`

**Step 3: Write minimal implementation**

Create `data/diagnostic.py`:

```python
"""Synthetic signal dilution dataset for diagnostic stress test (Algorithm 2)."""
import random

SIGNAL_TEMPLATES = [
    # Negation patterns (label 0)
    ("the file has {X} but not {Y}", 0),
    ("the system includes {X} but lacks {Y}", 0),
    ("the report mentions {X} without {Y}", 0),
    ("the package contains {X} but excludes {Y}", 0),
    ("the plan covers {X} but not {Y}", 0),
    # Affirmation patterns (label 1)
    ("the file has {X} and also {Y}", 1),
    ("the system includes {X} and {Y}", 1),
    ("the report mentions {X} with {Y}", 1),
    ("the package contains {X} and includes {Y}", 1),
    ("the plan covers {X} and {Y}", 1),
]

CONTENT_WORDS = [
    "keys", "data", "images", "tables", "links",
    "charts", "graphs", "notes", "files", "records",
    "entries", "values", "codes", "tags", "labels",
]

# Distractor vocabulary — common English words (derived from Wikipedia per paper).
# Using a built-in list avoids the NLTK dependency for reproducibility.
_DISTRACTOR_VOCAB = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "was", "on", "are", "as", "with", "his", "they", "be", "at", "one",
    "have", "this", "from", "or", "had", "by", "not", "but", "what", "all",
    "were", "when", "we", "there", "can", "an", "your", "which", "their", "said",
    "each", "she", "do", "how", "if", "will", "up", "other", "about", "out",
    "many", "then", "them", "these", "so", "some", "her", "would", "make", "like",
    "him", "into", "time", "has", "look", "two", "more", "go", "see", "way",
    "could", "no", "than", "first", "been", "call", "who", "its", "now", "find",
    "long", "down", "day", "did", "get", "come", "made", "after", "back", "only",
    "me", "our", "under", "know", "last", "also", "use", "just", "over", "such",
    "great", "think", "say", "help", "low", "line", "before", "turn", "move", "right",
    "too", "old", "still", "same", "tell", "need", "house", "world", "head", "own",
    "every", "city", "tree", "cross", "farm", "hard", "start", "might", "story", "far",
    "sea", "late", "run", "left", "here", "school", "close", "night", "real", "life",
    "few", "north", "open", "seem", "next", "walk", "ease", "both", "mark", "mile",
    "river", "car", "feet", "care", "second", "group", "carry", "took", "rain", "eat",
    "room", "friend", "began", "idea", "fish", "mountain", "stop", "once", "base", "hear",
    "horse", "cut", "sure", "watch", "color", "face", "wood", "main", "enough", "plain",
    "girl", "usual", "young", "ready", "above", "ever", "red", "list", "though", "feel",
    "side", "keep", "land", "song", "door", "wind", "upon", "shall", "rock", "black",
    "short", "space", "while", "human", "during", "glass", "plant", "round", "change", "sun",
    "fire", "stand", "point", "page", "order", "place", "play", "end", "area", "water",
    "hand", "high", "small", "large", "given", "much", "may", "set", "part", "new",
    "number", "people", "state", "very", "take", "year", "most", "well", "those", "show",
    "form", "work", "must", "home", "even", "being", "where", "field", "good", "three",
    "kind", "name", "used", "men", "light", "road", "food", "book", "war", "between",
    "country", "never", "system", "best", "body", "paper", "power", "air", "done", "until",
    "white", "children", "put", "against", "should", "often", "important", "man", "big", "near",
    "why", "went", "family", "hands", "given", "along", "half", "nothing", "away", "surface",
    "political", "music", "possible", "woman", "feet", "fact", "class", "taken", "always", "words",
    "early", "eye", "true", "center", "less", "table", "rest", "already", "church", "five",
]


def generate_diagnostic_dataset(
    num_samples: int = 10000,
    seq_length: int = 256,
    distractor_ratio: float = 0.5,
    seed: int = 42,
):
    """Generate synthetic signal dilution dataset.

    Args:
        num_samples: Number of (text, label) pairs to generate.
        seq_length: Total word count per sequence.
        distractor_ratio: Fraction of words that are distractors (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        List of (text, label) tuples. label is 0 (negation) or 1 (affirmation).
    """
    rng = random.Random(seed)

    num_distractor_tokens = int(seq_length * distractor_ratio)
    num_signal_tokens = seq_length - num_distractor_tokens

    dataset = []
    for _ in range(num_samples):
        template, label = rng.choice(SIGNAL_TEMPLATES)
        x, y = rng.sample(CONTENT_WORDS, 2)
        signal_text = template.format(X=x, Y=y)
        signal_tokens = signal_text.split()

        # Adjust signal to fit allocated length
        if len(signal_tokens) > num_signal_tokens:
            signal_tokens = signal_tokens[:num_signal_tokens]
        elif len(signal_tokens) < num_signal_tokens:
            padding = rng.choices(_DISTRACTOR_VOCAB, k=num_signal_tokens - len(signal_tokens))
            signal_tokens = signal_tokens + padding

        # Generate distractor tokens and inject signal at random position
        distractor_tokens = rng.choices(_DISTRACTOR_VOCAB, k=num_distractor_tokens)
        inject_pos = rng.randint(0, num_distractor_tokens)
        sequence = distractor_tokens[:inject_pos] + signal_tokens + distractor_tokens[inject_pos:]

        dataset.append((" ".join(sequence), label))

    return dataset
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_diagnostic.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add data/diagnostic.py tests/test_diagnostic.py
git commit -m "feat: add synthetic signal dilution data generator"
```

---

### Task 5: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new dependencies**

Add these lines to `requirements.txt`:

```
wandb
matplotlib
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add wandb and matplotlib dependencies"
```

---

### Task 6: Create diagnostic training script

**Files:**
- Create: `run_diagnostic.py`

**Step 1: Write the script**

Create `run_diagnostic.py`:

```python
"""Diagnostic stress test: signal dilution experiment (Table 7 / Figure 3).

Usage:
    python run_diagnostic.py --backbone bert-base-uncased --pooler glot --ratio 0.9
    python run_diagnostic.py --all
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.diagnostic import generate_diagnostic_dataset
from glot.backbone import BACKBONE_REGISTRY, load_backbone
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics

ALL_BACKBONES = list(BACKBONE_REGISTRY.keys())
ALL_POOLERS = ["cls", "eos", "mean", "max", "adapool", "glot"]
ALL_RATIOS = [0.2, 0.5, 0.8, 0.9]

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 2000
SEQ_LENGTH = 256
MAX_TOKEN_LENGTH = 512
EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 32
SEED = 42


def _select_pooler_type(pooler_name, backbone_type):
    """Map pooler name to the correct type for the backbone.

    For 'cls': use 'cls' for encoders, 'eos' for decoders.
    For 'eos': always 'eos'.
    Others pass through unchanged.
    """
    if pooler_name == "cls" and backbone_type == "decoder":
        return "eos"
    if pooler_name == "eos" and backbone_type == "encoder":
        return "cls"
    return pooler_name


def _get_dtype(backbone_name):
    """Use float16 for large decoder models to fit in GPU memory."""
    cfg = BACKBONE_REGISTRY[backbone_name]
    if cfg["params"] >= 3e9:
        return torch.float16
    return None


def tokenize_and_encode(texts, backbone, tokenizer, device, max_length=MAX_TOKEN_LENGTH):
    """Tokenize texts and run through frozen backbone to get hidden states."""
    all_hs = []
    all_masks = []

    batch_size = 16  # small batches for backbone inference
    backbone.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            out = backbone(**encoded)
            all_hs.append(out.last_hidden_state.cpu().float())
            all_masks.append(encoded["attention_mask"].cpu())

    return torch.cat(all_hs), torch.cat(all_masks)


def run_single_experiment(backbone_name, pooler_name, ratio, device, results_dict, use_wandb=False):
    """Run one (backbone, pooler, ratio) experiment and return accuracy."""
    cfg = BACKBONE_REGISTRY[backbone_name]
    pooler_type = _select_pooler_type(pooler_name, cfg["type"])

    print(f"\n{'='*60}")
    print(f"Backbone: {backbone_name} | Pooler: {pooler_name} | Ratio: {ratio}")
    print(f"{'='*60}")

    # Load backbone
    dtype = _get_dtype(backbone_name)
    backbone, tokenizer, bcfg = load_backbone(backbone_name, device=device, dtype=dtype)

    # Generate data
    train_data = generate_diagnostic_dataset(
        num_samples=TRAIN_SAMPLES, seq_length=SEQ_LENGTH,
        distractor_ratio=ratio, seed=SEED,
    )
    test_data = generate_diagnostic_dataset(
        num_samples=TEST_SAMPLES, seq_length=SEQ_LENGTH,
        distractor_ratio=ratio, seed=SEED + 1,
    )

    train_texts = [t for t, _ in train_data]
    train_labels = torch.tensor([l for _, l in train_data])
    test_texts = [t for t, _ in test_data]
    test_labels = torch.tensor([l for _, l in test_data])

    # Tokenize and encode
    print("Encoding train set...")
    train_hs, train_masks = tokenize_and_encode(train_texts, backbone, tokenizer, device)
    print("Encoding test set...")
    test_hs, test_masks = tokenize_and_encode(test_texts, backbone, tokenizer, device)

    # Free backbone memory
    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create pooler + head
    input_dim = bcfg["hidden_dim"]
    glot_config = {"hidden_dim": 128, "num_gnn_layers": 2, "jk_mode": "cat", "threshold": 0.6}
    pooler, head = create_pooler_and_head(
        pooler_type=pooler_type,
        input_dim=input_dim,
        num_classes=2,
        task_type="classification",
        glot_config=glot_config if pooler_type == "glot" else None,
    )
    pooler = pooler.to(device)
    head = head.to(device)

    # Data loaders
    train_ds = TensorDataset(train_hs, train_masks, train_labels)
    test_ds = TensorDataset(test_hs, test_masks, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Train
    params = list(pooler.parameters()) + list(head.parameters())
    optimizer = Adam(params, lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(SEED)

    for epoch in range(EPOCHS):
        pooler.train()
        head.train()
        total_loss = 0.0
        n_batches = 0
        for hs, masks, labels in train_loader:
            hs, masks, labels = hs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            z = pooler(hs, masks)
            logits = head(z)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Evaluate
    pooler.eval()
    head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for hs, masks, labels in test_loader:
            hs, masks, labels = hs.to(device), masks.to(device), labels.to(device)
            z = pooler(hs, masks)
            logits = head(z)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = compute_metrics(all_preds, all_labels, "accuracy")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Store result
    key = f"{backbone_name}|{pooler_name}|{ratio}"
    results_dict[key] = {
        "backbone": backbone_name,
        "pooler": pooler_name,
        "ratio": ratio,
        "accuracy": accuracy,
    }

    # W&B logging
    if use_wandb:
        try:
            import wandb
            wandb.log({
                "backbone": backbone_name,
                "pooler": pooler_name,
                "distractor_ratio": ratio,
                "accuracy": accuracy,
            })
        except ImportError:
            pass

    # Cleanup
    del pooler, head, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Diagnostic stress test experiment")
    parser.add_argument("--backbone", default=None, help="Backbone model name")
    parser.add_argument("--pooler", default=None, help="Pooler type")
    parser.add_argument("--ratio", type=float, default=None, help="Distractor ratio")
    parser.add_argument("--all", action="store_true", help="Run all 120 combinations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/diagnostic_results.json")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Init W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(project="glot-diagnostic", config=vars(args))
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
            args.wandb = False

    # Determine what to run
    if args.all:
        backbones = ALL_BACKBONES
        poolers = ["cls", "mean", "max", "adapool", "glot"]
        ratios = ALL_RATIOS
    else:
        if not args.backbone or not args.pooler or args.ratio is None:
            parser.error("Provide --backbone, --pooler, and --ratio, or use --all")
        backbones = [args.backbone]
        poolers = [args.pooler]
        ratios = [args.ratio]

    # Load existing results if any
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    # Run experiments
    for backbone_name in backbones:
        for ratio in ratios:
            for pooler_name in poolers:
                key = f"{backbone_name}|{pooler_name}|{ratio}"
                if key in results:
                    print(f"Skipping {key} (already computed)")
                    continue
                run_single_experiment(
                    backbone_name, pooler_name, ratio,
                    args.device, results, use_wandb=args.wandb,
                )
                # Save after each experiment for resume capability
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

    # Print summary table
    print("\n\n=== RESULTS SUMMARY ===\n")
    for ratio in ALL_RATIOS:
        print(f"\n--- {int(ratio*100)}% Distractors ---")
        print(f"{'Backbone':<45} {'CLS/EOS':>8} {'Mean':>8} {'Max':>8} {'AdaPool':>8} {'GLOT':>8}")
        for backbone_name in ALL_BACKBONES:
            row = []
            for p in ["cls", "mean", "max", "adapool", "glot"]:
                key = f"{backbone_name}|{p}|{ratio}"
                if key in results:
                    row.append(f"{results[key]['accuracy']:.1f}")
                else:
                    row.append("--")
            print(f"{backbone_name:<45} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
```

**Step 2: Verify script parses without errors**

Run: `python -c "import run_diagnostic; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add run_diagnostic.py
git commit -m "feat: add diagnostic stress test training script"
```

---

### Task 7: Create visualization script

**Files:**
- Create: `scripts/plot_diagnostic.py`

**Step 1: Write the script**

Create `scripts/plot_diagnostic.py`:

```python
"""Plot diagnostic stress test results (Figure 3 / Table 7 from paper).

Usage:
    python scripts/plot_diagnostic.py --input results/diagnostic_results.json
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from glot.backbone import BACKBONE_REGISTRY

# Plotting style
POOLER_STYLES = {
    "cls": {"color": "#1f77b4", "marker": "s", "label": "CLS/EOS"},
    "mean": {"color": "#ff7f0e", "marker": "^", "label": "Mean"},
    "max": {"color": "#2ca02c", "marker": "v", "label": "Max"},
    "adapool": {"color": "#d62728", "marker": "D", "label": "AdaPool"},
    "glot": {"color": "#9467bd", "marker": "o", "label": "GLOT", "linewidth": 2.5},
}

RATIOS = [0.2, 0.5, 0.8, 0.9]
BACKBONES = list(BACKBONE_REGISTRY.keys())


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_figure3(results, output_prefix="results/diagnostic_figure"):
    """Generate 2x2 grid plot (Figure 3 from paper)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    model_sizes = [BACKBONE_REGISTRY[b]["params"] for b in BACKBONES]
    model_labels = [b.split("/")[-1] for b in BACKBONES]

    for ax, ratio in zip(axes.flat, RATIOS):
        for pooler_name, style in POOLER_STYLES.items():
            accs = []
            valid_sizes = []
            for backbone_name, size in zip(BACKBONES, model_sizes):
                key = f"{backbone_name}|{pooler_name}|{ratio}"
                if key in results:
                    accs.append(results[key]["accuracy"])
                    valid_sizes.append(size)

            if accs:
                ax.plot(
                    valid_sizes, accs,
                    color=style["color"],
                    marker=style["marker"],
                    label=style["label"],
                    linewidth=style.get("linewidth", 1.5),
                    markersize=7,
                )

        ax.set_title(f"{int(ratio * 100)}% Distractors", fontsize=13)
        ax.set_xlabel("Parameters", fontsize=11)
        ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(40, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Custom x-tick labels
        ax.set_xticks(model_sizes)
        ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)

    plt.suptitle("Diagnostic Stress Test: Signal Dilution", fontsize=15, fontweight="bold")
    plt.tight_layout()

    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved: {output_prefix}.png and {output_prefix}.pdf")
    plt.close()


def print_table7(results):
    """Print Table 7 formatted results to stdout."""
    for ratio in RATIOS:
        print(f"\n### {int(ratio*100)}% Distractors\n")
        print(f"| {'Model':<30} | {'CLS/EOS':>8} | {'Mean':>8} | {'Max':>8} | {'AdaPool':>8} | {'GLOT':>8} |")
        print(f"|{'-'*32}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
        for backbone_name in BACKBONES:
            short = backbone_name.split("/")[-1]
            row = []
            for p in ["cls", "mean", "max", "adapool", "glot"]:
                key = f"{backbone_name}|{p}|{ratio}"
                if key in results:
                    val = results[key]["accuracy"]
                    row.append(f"{val:.1f}")
                else:
                    row.append("--")
            print(f"| {short:<30} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} |")


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic results")
    parser.add_argument("--input", default="results/diagnostic_results.json")
    parser.add_argument("--output", default="results/diagnostic_figure")
    args = parser.parse_args()

    results = load_results(args.input)
    print_table7(results)
    plot_figure3(results, output_prefix=args.output)


if __name__ == "__main__":
    main()
```

**Step 2: Verify script parses without errors**

Run: `python -c "import scripts.plot_diagnostic; print('OK')"` — this will fail because `scripts/` is not a package. That's fine; this script is run directly.

Run instead: `python -c "exec(open('scripts/plot_diagnostic.py').read().split('def main')[0]); print('OK')"`

Or just: `python scripts/plot_diagnostic.py --help`
Expected: prints help text without error

**Step 3: Commit**

```bash
mkdir -p scripts
git add scripts/plot_diagnostic.py
git commit -m "feat: add diagnostic results visualization script"
```

---

### Task 8: Add .gitignore entries for results

**Files:**
- Modify: `.gitignore`

**Step 1: Add results directory pattern**

Add to `.gitignore`:

```
# Experiment results (large files, regenerated)
results/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add results/ to gitignore"
```

---

### Task 9: Run full test suite to verify nothing is broken

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass. New tests (TestEOSPooler, TestBackboneRegistry, TestDiagnosticDataGeneration) should all be green. Existing tests should remain unchanged.

**Step 2: If any failures, fix and recommit**

---

### Task 10: Smoke-test the diagnostic script with tiny data

This is a manual verification task to confirm the full pipeline works end-to-end before running the real experiments.

**Step 1: Create a quick smoke test**

Run (on a machine with GPU and BERT downloaded):

```bash
python run_diagnostic.py --backbone bert-base-uncased --pooler mean --ratio 0.5 --device cpu
```

This should:
1. Generate 10,000 train + 2,000 test samples
2. Tokenize with BERT tokenizer
3. Run through BERT (slow on CPU but works)
4. Train MeanPooler + head for 2 epochs
5. Print accuracy
6. Save to `results/diagnostic_results.json`

Expected: completes without error, accuracy printed, JSON file created.

**Step 2: Verify JSON output**

Run: `cat results/diagnostic_results.json`
Expected: JSON with one entry containing backbone, pooler, ratio, accuracy fields.

**Step 3: No commit needed — this is a verification step**

---

## Summary of all files

| File | Action | Task |
|------|--------|------|
| `glot/baselines.py` | Modify (add EOSPooler) | 1 |
| `glot/__init__.py` | Modify (add exports) | 1, 3 |
| `tests/test_baselines.py` | Modify (add EOSPooler tests) | 1 |
| `glot/model.py` | Modify (add eos type) | 2 |
| `tests/test_model.py` | Modify (add eos test) | 2 |
| `glot/backbone.py` | Create | 3 |
| `tests/test_backbone.py` | Create | 3 |
| `data/diagnostic.py` | Create | 4 |
| `tests/test_diagnostic.py` | Create | 4 |
| `requirements.txt` | Modify | 5 |
| `run_diagnostic.py` | Create | 6 |
| `scripts/plot_diagnostic.py` | Create | 7 |
| `.gitignore` | Modify | 8 |

Total: 7 new files, 6 modified files, 10 tasks, ~10 commits.
