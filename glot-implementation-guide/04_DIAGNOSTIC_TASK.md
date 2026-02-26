# 04: Diagnostic Stress Test (Signal Dilution)

## Overview
A synthetic task that tests robustness to signal dilution. A short "signal phrase" containing a logical dependency (e.g., negation) is injected into a long sequence of random distractor words. The model must classify based on the logic of the signal phrase.

This is the key experiment that demonstrates GLOT's advantage — at 90% distractors, GLOT maintains >97% accuracy while baselines collapse to 50-60%.

---

## Data Generation (`data/diagnostic.py`)

### Algorithm (from Algorithm 2 in Appendix B)

```python
import random
import numpy as np

def generate_diagnostic_dataset(
    num_samples: int = 10000,
    seq_length: int = 256,
    distractor_ratio: float = 0.5,
    seed: int = 42
):
    """
    Generate synthetic diagnostic dataset.
    
    Args:
        num_samples: Number of samples to generate
        seq_length: Total sequence length (L = 256 in paper)
        distractor_ratio: Fraction of tokens that are distractors (0.2, 0.5, 0.8, 0.9)
        seed: Random seed
    
    Returns:
        List of (token_sequence, label) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Signal phrase templates with logical dependencies
    # Label 0 = negation present, Label 1 = affirmation
    # The key is that the phrase contains a RELATIONAL dependency
    templates = [
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
    
    # Vocabulary for placeholders
    content_words = ["keys", "data", "images", "tables", "links", 
                     "charts", "graphs", "notes", "files", "records",
                     "entries", "values", "codes", "tags", "labels"]
    
    # Large distractor vocabulary (from English Wikipedia - use a word list)
    # In practice, load a large vocabulary file
    distractor_vocab = load_distractor_vocabulary()  # See below
    
    dataset = []
    num_distractor_tokens = int(seq_length * distractor_ratio)
    num_signal_tokens = seq_length - num_distractor_tokens
    
    for _ in range(num_samples):
        # Choose random template
        template, label = random.choice(templates)
        
        # Fill placeholders
        x, y = random.sample(content_words, 2)
        signal_text = template.format(X=x, Y=y)
        signal_tokens = signal_text.split()
        
        # Adjust signal to fit allocated length
        if len(signal_tokens) > num_signal_tokens:
            signal_tokens = signal_tokens[:num_signal_tokens]
        elif len(signal_tokens) < num_signal_tokens:
            padding = random.choices(distractor_vocab, k=num_signal_tokens - len(signal_tokens))
            signal_tokens = signal_tokens + padding
        
        # Generate distractor tokens
        distractor_tokens = random.choices(distractor_vocab, k=num_distractor_tokens)
        
        # Inject signal at random position within distractors
        inject_pos = random.randint(0, num_distractor_tokens)
        sequence = (distractor_tokens[:inject_pos] + 
                   signal_tokens + 
                   distractor_tokens[inject_pos:])
        
        dataset.append((' '.join(sequence), label))
    
    return dataset


def load_distractor_vocabulary():
    """
    Load a large vocabulary of common English words for distractors.
    The paper says "derived from English Wikipedia."
    
    Options:
    1. Use nltk.corpus.words
    2. Download a word frequency list
    3. Use a subset of common English words
    """
    # Simple approach: use a large set of common words
    # You could also load from a file
    try:
        import nltk
        nltk.download('words', quiet=True)
        from nltk.corpus import words
        vocab = [w.lower() for w in words.words() if len(w) > 2 and w.isalpha()]
        return vocab[:10000]  # Use top 10K words
    except:
        # Fallback: generate a reasonable vocabulary
        # In practice, use a proper word list file
        return ["the", "a", "an", "is", "was", "are", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "could", "should", "may", "might", "shall", "can",
                "this", "that", "these", "those", "it", "they", "we",
                "where", "when", "how", "what", "which", "who", "whom",
                # ... extend with thousands more common words
                ]
```

### Dataset Configuration
```python
DIAGNOSTIC_CONFIG = {
    'train_samples': 10000,
    'test_samples': 2000,
    'seq_length': 256,
    'distractor_ratios': [0.2, 0.5, 0.8, 0.9],
    'seed': 42,
    'task': 'binary_classification',
    'loss': 'CrossEntropy',
    'metric': 'accuracy',
}
```

---

## Training the Diagnostic Task

```python
def run_diagnostic_experiment(backbone_name, pooler_type, distractor_ratio):
    """Run one diagnostic experiment."""
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    # Generate data
    train_data = generate_diagnostic_dataset(
        num_samples=10000, distractor_ratio=distractor_ratio
    )
    test_data = generate_diagnostic_dataset(
        num_samples=2000, distractor_ratio=distractor_ratio, 
        seed=123  # Different seed for test
    )
    
    # Tokenize
    train_encodings = tokenizer(
        [text for text, _ in train_data],
        truncation=True, max_length=512, padding='max_length',
        return_tensors='pt'
    )
    train_labels = torch.tensor([label for _, label in train_data])
    
    # Build model
    model = GLOTModel(
        backbone_name=backbone_name,
        pooler_type=pooler_type,
        num_classes=2,
        task_type='classification'
    )
    
    # Train and evaluate
    train(model, train_loader, test_loader, TRAINING_CONFIG, 
          {'type': 'single', 'metric': 'accuracy'})
```

---

## Expected Results (Table 7)

At 90% distractor ratio:

| Model | CLS/EOS | Mean | Max | AdaPool | GLOT |
|-------|---------|------|-----|---------|------|
| BERT | 67.6 | 53.4 | 50.4 | 61.6 | **98.8** |
| RoBERTa | 48.6 | 57.2 | 50.2 | 59.2 | **98.2** |
| SmolLM2 | 51.4 | 51.4 | 51.4 | 55.2 | **92.2** |
| TinyLlama | 56.6 | 56.4 | 51.4 | 53.0 | **94.0** |
| LLaMA-3B | 68.4 | 61.8 | 54.6 | 51.0 | **93.2** |
| Mistral-7B | 70.6 | 63.8 | 55.6 | 78.4 | **97.2** |

The massive gap between GLOT and baselines at high noise levels is the core finding that validates the "relational learning before compression" hypothesis.

---

## Visualization (Figure 3)

Create a 2x2 grid of plots:
- X-axis: Backbone parameter count (log scale): 110M, 360M, 1.1B, 3B, 7B
- Y-axis: Classification accuracy (%)
- One panel per distractor ratio: 20%, 50%, 80%, 90%
- Lines for each pooling method

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ratio in zip(axes.flat, [0.2, 0.5, 0.8, 0.9]):
    for method in ['cls', 'mean', 'max', 'adapool', 'glot']:
        ax.plot(model_sizes, results[ratio][method], label=method, marker='o')
    ax.set_title(f'{int(ratio*100)}% Distractors')
    ax.set_xlabel('Parameters (log scale)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_xscale('log')
    ax.legend()
plt.tight_layout()
```
