# 02: Model Wrapper and Baselines

## Overview
This file covers:
1. Wrapping frozen LLM backbones with GLOT
2. Task-specific heads (classification, similarity, retrieval)
3. Baseline pooling implementations (Mean, Max, CLS/EOS, AdaPool)

---

## Frozen Backbone Wrapper (`model.py`)

### Supported Backbones

| Model | Type | HuggingFace ID | Hidden Dim (d) | Max Context |
|-------|------|----------------|-----------------|-------------|
| BERT | Encoder | `bert-base-uncased` | 768 | 512 |
| RoBERTa | Encoder | `roberta-base` | 768 | 512 |
| SmolLM2 | Decoder | `HuggingFaceTB/SmolLM2-1.7B` | 2048 | 8192 |
| TinyLlama | Decoder | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 | 2048 |
| LLaMA-3.2-3B | Decoder | `meta-llama/Llama-3.2-3B` | 3072 | 8192 |
| Mistral-7B | Decoder | `mistralai/Mistral-7B-v0.1` | 4096 | 32768 |

### Implementation Pattern

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class GLOTModel(nn.Module):
    def __init__(self, backbone_name: str, pooler_type: str = 'glot', 
                 num_classes: int = 2, task_type: str = 'classification',
                 glot_config: dict = None):
        super().__init__()
        
        # Load frozen backbone
        self.backbone = AutoModel.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False  # CRITICAL: Freeze backbone
        self.backbone.eval()
        
        hidden_dim = self.backbone.config.hidden_size
        
        # Pooler
        if pooler_type == 'glot':
            self.pooler = GLOTPooler(input_dim=hidden_dim, **(glot_config or {}))
            pool_output_dim = self.pooler.output_dim
        elif pooler_type == 'mean':
            self.pooler = MeanPooler()
            pool_output_dim = hidden_dim
        elif pooler_type == 'max':
            self.pooler = MaxPooler()
            pool_output_dim = hidden_dim
        elif pooler_type == 'cls':  # or 'eos' for decoder models
            self.pooler = CLSPooler()
            pool_output_dim = hidden_dim
        elif pooler_type == 'adapool':
            self.pooler = AdaPool(hidden_dim)
            pool_output_dim = hidden_dim
        
        # Task head
        self.task_type = task_type
        if task_type == 'classification':
            self.classifier = nn.Linear(pool_output_dim, num_classes)
        elif task_type == 'pair_classification':
            self.classifier = nn.Linear(pool_output_dim * 2, num_classes)
        elif task_type == 'regression':
            self.classifier = nn.Linear(pool_output_dim, 1)
        elif task_type == 'similarity':
            self.classifier = None  # Use cosine similarity directly
    
    def encode(self, input_ids, attention_mask):
        """Get sentence embedding from input tokens."""
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, L, d)
        
        z = self.pooler(hidden_states, attention_mask)  # (B, D)
        return z
    
    def forward(self, input_ids, attention_mask, 
                input_ids_b=None, attention_mask_b=None):
        """Forward pass for different task types."""
        z_a = self.encode(input_ids, attention_mask)
        
        if self.task_type == 'classification':
            return self.classifier(z_a)
        
        elif self.task_type == 'pair_classification':
            z_b = self.encode(input_ids_b, attention_mask_b)
            combined = torch.cat([z_a, z_b], dim=-1)
            return self.classifier(combined)
        
        elif self.task_type == 'regression':
            if input_ids_b is not None:
                z_b = self.encode(input_ids_b, attention_mask_b)
                # For STS-B: predict similarity score
                return torch.cosine_similarity(z_a, z_b, dim=-1)
            return self.classifier(z_a).squeeze(-1)
        
        elif self.task_type == 'similarity':
            z_b = self.encode(input_ids_b, attention_mask_b)
            return torch.cosine_similarity(z_a, z_b, dim=-1)
```

### Important: Decoder Model Tokenizer Setup
```python
tokenizer = AutoTokenizer.from_pretrained(backbone_name)
# For decoder-only models:
tokenizer.padding_side = 'right'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Hidden State Caching (Critical for Efficiency)
The paper caches frozen backbone outputs before training:
```python
def precompute_hidden_states(backbone, dataloader, device):
    """Cache all hidden states to avoid recomputing backbone forward pass."""
    all_hidden_states = []
    all_masks = []
    all_labels = []
    
    backbone.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = backbone(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            all_hidden_states.append(outputs.last_hidden_state.cpu())
            all_masks.append(batch['attention_mask'].cpu())
            all_labels.append(batch['labels'])
    
    return TensorDataset(
        torch.cat(all_hidden_states),
        torch.cat(all_masks),
        torch.cat(all_labels)
    )
```

---

## Baseline Pooling Methods (`baselines.py`)

### Mean Pooling
```python
class MeanPooler(nn.Module):
    def forward(self, hidden_states, attention_mask):
        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, d)
        counts = mask.sum(dim=1).clamp(min=1)        # (B, 1)
        return summed / counts
```

### Max Pooling
```python
class MaxPooler(nn.Module):
    def forward(self, hidden_states, attention_mask):
        # Set padding positions to -inf before max
        mask = attention_mask.unsqueeze(-1).bool()
        hidden_states = hidden_states.masked_fill(~mask, float('-inf'))
        return hidden_states.max(dim=1).values  # (B, d)
```

### CLS/EOS Pooling
```python
class CLSPooler(nn.Module):
    """For encoder models: first token. For decoder models: last non-padded token."""
    def __init__(self, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
    
    def forward(self, hidden_states, attention_mask):
        if self.is_decoder:
            # Last non-padded token (EOS)
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths]
        else:
            # First token (CLS)
            return hidden_states[:, 0]
```

### AdaPool (Brothers, 2025)
```python
class AdaPool(nn.Module):
    """Two-layer MLP scoring function with softmax-weighted average."""
    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    
    def forward(self, hidden_states, attention_mask):
        scores = self.scorer(hidden_states).squeeze(-1)  # (B, L)
        
        # Mask padding positions
        scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=1)  # (B, L)
        
        # Weighted sum
        z = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, d)
        return z
```

---

## Task Heads

### Single-Sentence Classification (CoLA, SST-2)
```python
# y = softmax(W·z + b)
classifier = nn.Linear(pool_output_dim, num_classes)
loss_fn = nn.CrossEntropyLoss()
```

### Sentence-Pair Classification (MRPC, QQP, MNLI, QNLI, RTE, WNLI)
```python
# y = softmax(W·[z_a || z_b] + b)
classifier = nn.Linear(pool_output_dim * 2, num_classes)
loss_fn = nn.CrossEntropyLoss()
```

### Regression / Similarity (STS-B)
```python
# sim = cosine_similarity(z_a, z_b)
loss_fn = nn.MSELoss()
# Note: STS-B scores are 1-5, normalize to 0-1 or use raw with MSE
```

### Contrastive Learning (MTEB / MS MARCO)
```python
# Symmetric in-batch contrastive loss with temperature 0.07
def contrastive_loss(z_queries, z_passages, temperature=0.07):
    z_q = F.normalize(z_queries, dim=-1)
    z_p = F.normalize(z_passages, dim=-1)
    sim_matrix = z_q @ z_p.T / temperature  # (B, B)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = (F.cross_entropy(sim_matrix, labels) + 
            F.cross_entropy(sim_matrix.T, labels)) / 2
    return loss
```
