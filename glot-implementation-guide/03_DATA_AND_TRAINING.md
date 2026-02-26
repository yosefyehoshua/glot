# 03: Data Loading, Training, and Evaluation

## Overview
This file covers data loading for all benchmarks, the training loop, and evaluation protocols.

---

## GLUE Benchmark Data Loading (`data/glue_loader.py`)

### Task Specifications

| Task | Type | # Classes | Loss | Metric | Max Len | Train Size |
|------|------|-----------|------|--------|---------|------------|
| CoLA | Single-sentence | 2 | CrossEntropy | MCC (Matthews Correlation) | 128 | Full |
| SST-2 | Single-sentence | 2 | CrossEntropy | Accuracy | 128 | Full |
| STS-B | Sentence-pair | Regression | MSE | Spearman correlation | 128 | Full |
| MRPC | Sentence-pair | 2 | CrossEntropy | F1 | 128 | Full |
| QQP | Sentence-pair | 2 | CrossEntropy | F1 | 128 | 20,000 subsample |
| MNLI | Sentence-pair | 3 | CrossEntropy | Accuracy (matched & mismatched) | 128 | 20,000 subsample |
| QNLI | Sentence-pair | 2 | CrossEntropy | Accuracy | 128 | 20,000 subsample |
| RTE | Sentence-pair | 2 | CrossEntropy | Accuracy | 128 | Full |
| WNLI | Sentence-pair | 2 | CrossEntropy | Accuracy | 128 | Full |

### Loading Pattern
```python
from datasets import load_dataset

# GLUE tasks
GLUE_TASKS = {
    'cola': {'type': 'single', 'num_classes': 2, 'metric': 'mcc',
             'sentence_keys': ('sentence',)},
    'sst2': {'type': 'single', 'num_classes': 2, 'metric': 'accuracy',
             'sentence_keys': ('sentence',)},
    'stsb': {'type': 'pair', 'num_classes': 1, 'metric': 'spearman',
             'sentence_keys': ('sentence1', 'sentence2')},
    'mrpc': {'type': 'pair', 'num_classes': 2, 'metric': 'f1',
             'sentence_keys': ('sentence1', 'sentence2')},
    'qqp': {'type': 'pair', 'num_classes': 2, 'metric': 'f1',
            'sentence_keys': ('question1', 'question2'), 'subsample': 20000},
    'mnli': {'type': 'pair', 'num_classes': 3, 'metric': 'accuracy',
             'sentence_keys': ('premise', 'hypothesis'), 'subsample': 20000},
    'qnli': {'type': 'pair', 'num_classes': 2, 'metric': 'accuracy',
             'sentence_keys': ('question', 'sentence'), 'subsample': 20000},
    'rte': {'type': 'pair', 'num_classes': 2, 'metric': 'accuracy',
            'sentence_keys': ('sentence1', 'sentence2')},
    'wnli': {'type': 'pair', 'num_classes': 2, 'metric': 'accuracy',
             'sentence_keys': ('sentence1', 'sentence2')},
}

def load_glue_task(task_name, tokenizer, max_length=128, seed=42):
    config = GLUE_TASKS[task_name]
    dataset = load_dataset('glue', task_name)
    
    # Subsample large datasets
    if 'subsample' in config:
        dataset['train'] = dataset['train'].shuffle(seed=seed).select(
            range(min(config['subsample'], len(dataset['train'])))
        )
    
    def tokenize(examples):
        keys = config['sentence_keys']
        if len(keys) == 1:
            return tokenizer(examples[keys[0]], truncation=True, 
                           max_length=max_length, padding='max_length')
        else:
            return tokenizer(examples[keys[0]], examples[keys[1]], 
                           truncation=True, max_length=max_length, 
                           padding='max_length')
    
    dataset = dataset.map(tokenize, batched=True)
    return dataset
```

### IMPORTANT: For sentence-pair tasks with GLOT
For sentence-pair tasks, you need to encode each sentence **separately** to get individual embeddings, then concatenate. This means you need separate tokenization:

```python
def tokenize_pair_separate(examples, tokenizer, max_length=128):
    """Tokenize sentence pairs separately for GLOT pair classification."""
    keys = config['sentence_keys']
    tok_a = tokenizer(examples[keys[0]], truncation=True, 
                      max_length=max_length, padding='max_length')
    tok_b = tokenizer(examples[keys[1]], truncation=True, 
                      max_length=max_length, padding='max_length')
    return {
        'input_ids_a': tok_a['input_ids'],
        'attention_mask_a': tok_a['attention_mask'],
        'input_ids_b': tok_b['input_ids'],
        'attention_mask_b': tok_b['attention_mask'],
        'label': examples['label']
    }
```

**Exception: STS-B** — For regression/similarity tasks, compute cosine similarity between z_a and z_b and use MSE loss against the (scaled) similarity score.

---

## IMDB Data Loading (`data/imdb_loader.py`)

```python
def load_imdb(tokenizer, max_length=512):
    dataset = load_dataset('imdb')
    
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True,
                        max_length=max_length, padding='max_length')
    
    dataset = dataset.map(tokenize, batched=True)
    return dataset
```

- Task: Binary sentiment classification
- Loss: CrossEntropy
- Metric: Accuracy
- Max length: **512** tokens (longer than GLUE)

---

## Training Loop (`train.py`)

### Core Training Configuration
```python
# From Section B.1
TRAINING_CONFIG = {
    'epochs': 2,
    'optimizer': 'Adam',
    'learning_rate': 2e-4,
    'weight_decay': 0.0,
    'batch_size_train': 32,
    'batch_size_eval': 64,
    'seed': 42,
}
```

### Training Loop Pattern
```python
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

def train(model, train_loader, val_loader, config, task_config):
    # Only optimize pooler + classifier parameters (backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=config['learning_rate'], 
                     weight_decay=config['weight_decay'])
    
    loss_fn = get_loss_fn(task_config['metric'])
    
    for epoch in range(config['epochs']):
        model.train()
        # NOTE: Keep backbone in eval mode even during training
        model.backbone.eval()
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if task_config['type'] == 'single':
                logits = model(batch['input_ids'], batch['attention_mask'])
                loss = loss_fn(logits, batch['labels'])
            elif task_config['type'] == 'pair':
                logits = model(
                    batch['input_ids_a'], batch['attention_mask_a'],
                    batch['input_ids_b'], batch['attention_mask_b']
                )
                loss = loss_fn(logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        metrics = evaluate(model, val_loader, task_config)
        print(f"Epoch {epoch+1}: {metrics}")
```

### With Cached Hidden States (Recommended)
```python
def train_with_cache(pooler, classifier, cached_dataset, config):
    """Train only the pooler + classifier on pre-computed hidden states."""
    train_loader = DataLoader(cached_dataset, batch_size=32, shuffle=True)
    
    params = list(pooler.parameters()) + list(classifier.parameters())
    optimizer = Adam(params, lr=config['learning_rate'])
    
    for epoch in range(config['epochs']):
        for hidden_states, masks, labels in train_loader:
            optimizer.zero_grad()
            z = pooler(hidden_states, masks)
            logits = classifier(z)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
```

---

## Evaluation (`evaluate.py`)

### Metric Implementations
```python
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import spearmanr

def compute_metrics(predictions, labels, metric_name):
    if metric_name == 'mcc':
        return matthews_corrcoef(labels, predictions)
    elif metric_name == 'f1':
        return f1_score(labels, predictions)
    elif metric_name == 'accuracy':
        return accuracy_score(labels, predictions)
    elif metric_name == 'spearman':
        return spearmanr(predictions, labels).correlation
```

### GLUE Evaluation Protocol
- Train on training split
- Evaluate on validation split (paper uses validation since test labels are hidden)
- Report: MCC for CoLA, Spearman for STS-B, F1 for MRPC/QQP, Accuracy for rest
- All scores multiplied by 100

### MNLI Note
MNLI has two validation sets: `validation_matched` and `validation_mismatched`. Report both (MNLI-m and MNLI-mm columns in Table 1).

---

## Full Fine-Tuning & LoRA Baselines (for Table 5 comparison)

### Full Fine-Tuning
```python
# Unfreeze backbone
for param in model.backbone.parameters():
    param.requires_grad = True

# Config from paper:
# - Learning rate: 2e-5
# - Weight decay: 0.01
# - Optimizer: AdamW
# - Epochs: 3
```

### LoRA Baseline
```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",  # attention
                     "gate_proj", "up_proj", "down_proj"],     # FFN
    lora_alpha=16,
    lora_dropout=0.1,
)
model = get_peft_model(backbone, lora_config)

# Config:
# - Learning rate: 2e-4
# - Weight decay: 0.01
# - Epochs: 3
```
