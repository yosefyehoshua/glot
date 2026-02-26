# 06: Hyperparameters and Configuration Reference

## Complete Hyperparameter Reference

All values are extracted directly from the paper (Tables 5, 6, and Appendix B).

---

## GLOT Architecture Hyperparameters

### Search Space (Table 6)

| Hyperparameter | Search Space | Default/Best |
|----------------|-------------|--------------|
| Learning Rate | {1e-3, 2e-4, 2e-5} | 2e-4 |
| Weight Decay | {0.0, 1e-5, 5e-5} | 0.0 |
| GNN Layers (K) | {2, 4} | 2 |
| GNN Hidden Dimension | {64, 128, 256} | 128 |
| Jumping Knowledge | {cat, max, mean, none} | cat |
| Input Projection Dimension | {128, 256, 512} | — (matches GNN hidden) |
| Similarity Threshold (τ) | {0.1, 0.3, 0.6} | 0.6 |

### Final Configuration Used in Paper
```yaml
# GLOT Configuration
glot:
  gnn_type: GATConv          # Graph Attention Network
  num_layers: 2               # K = 2 GNN layers  
  hidden_dim: 128             # GNN hidden dimension
  activation: ReLU            # Non-linearity
  jk_mode: cat                # Jumping Knowledge concatenation
  threshold: 0.6              # Cosine similarity threshold τ
  
  # Derived values:
  # output_dim = hidden_dim * (num_layers + 1) = 128 * 3 = 384 (with JK='cat')
  # Total trainable params ≈ 8.92M
```

---

## Training Hyperparameters

### General Training (GLUE, IMDB, Diagnostic)
```yaml
training:
  epochs: 2
  optimizer: Adam
  learning_rate: 2e-4
  weight_decay: 0.0
  batch_size_train: 32
  batch_size_eval: 64
  seed: 42
  max_length_glue: 128
  max_length_imdb: 512
```

### MTEB / MS MARCO Contrastive Training
```yaml
mteb_training:
  epochs: 2
  optimizer: Adam
  learning_rate: 2e-4
  weight_decay: 0.0
  batch_size: 32
  loss: symmetric_contrastive
  temperature: 0.07
```

### Full Fine-Tuning Baseline (Table 5)
```yaml
full_ft:
  epochs: 3
  optimizer: AdamW
  learning_rate: 2e-5
  weight_decay: 0.01
  pooling: EOS  # For decoder models
```

### LoRA Baseline (Table 5)
```yaml
lora:
  rank: 64
  target_modules: [attention, feed-forward]  # Both attention and FFN blocks
  epochs: 3
  optimizer: AdamW  # Implied
  learning_rate: 2e-4
  weight_decay: 0.01
```

---

## Backbone Model Configurations

### Encoder Models
```yaml
bert:
  name: bert-base-uncased
  hidden_size: 768
  max_length: 512
  pooling_token: CLS  # First token
  type: encoder
  params: 110M

roberta:
  name: roberta-base
  hidden_size: 768
  max_length: 512
  pooling_token: CLS  # First token
  type: encoder
  params: 360M
```

### Decoder Models
```yaml
smollm2:
  name: HuggingFaceTB/SmolLM2-1.7B  # Verify exact HF ID
  hidden_size: 2048
  max_length: 8192
  pooling_token: EOS  # Last non-padded token
  type: decoder
  params: 1.7B
  tokenizer_config:
    padding_side: right
    pad_token: eos_token

tinyllama:
  name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  hidden_size: 2048
  max_length: 2048
  pooling_token: EOS
  type: decoder
  params: 1.1B
  tokenizer_config:
    padding_side: right
    pad_token: eos_token

llama3b:
  name: meta-llama/Llama-3.2-3B
  hidden_size: 3072
  max_length: 8192
  pooling_token: EOS
  type: decoder
  params: 3.2B
  tokenizer_config:
    padding_side: right
    pad_token: eos_token

mistral7b:
  name: mistralai/Mistral-7B-v0.1
  hidden_size: 4096
  max_length: 32768
  pooling_token: EOS
  type: decoder
  params: 7.2B
  tokenizer_config:
    padding_side: right
    pad_token: eos_token
```

---

## Resource Requirements (Table 5, Table 10)

### GLOT (Our Method)
| Metric | Value |
|--------|-------|
| Trainable Parameters | 8.92M |
| GPU Memory (training pooler only) | 0.42 GB |
| Batch Runtime (training) | 13.4 ± 3.0 ms |
| Batch Runtime (inference, Mistral) | ~3.25 s |

### Graph Construction Overhead (Table 10)
| Backbone | Max Context | Graph Time | Total Time | Overhead |
|----------|-------------|-----------|------------|---------|
| BERT | 512 | 0.043 ms | 5.36 ms | 0.8% |
| TinyLlama | 2048 | 0.672 ms | 143.15 ms | 0.5% |
| SmolLM2 | 8192 | 4.46 ms | 772.30 ms | 0.6% |
| LLaMA-3B | 8192 | 15.05 ms | 2041.77 ms | 0.7% |
| Mistral-7B | 32768 | 303.47 ms | 23460.29 ms | 1.3% |

### Hardware
- All experiments: Single NVIDIA A6000 GPU
- For Mistral-7B with GLOT: Only 0.42 GB needed for pooler training (backbone runs in inference mode)

---

## Graph Sparsity Ablation (Table 4, Mistral-7B)

| τ | CoLA (MCC) | SST-2 (ACC) | QQP (F1) | RTE (ACC) | WNLI (ACC) |
|---|-----------|-------------|----------|----------|------------|
| 0.0 (fully connected) | 50.19 | 93.69 | 62.79 | 49.81 | 38.03 |
| 0.2 | 53.40 | **94.38** | 62.53 | 49.45 | 36.62 |
| 0.4 | 51.73 | 93.46 | **64.07** | 50.54 | 40.84 |
| 0.6 | **54.30** | 93.23 | 63.49 | **54.15** | **56.34** |
| 0.8 | 52.48 | 92.66 | 63.22 | 52.70 | 56.34 |

**Takeaway**: τ=0.6 is generally the best. Fully connected (τ=0.0) is worst — sparse graphs outperform dense ones.

---

## GNN Architecture Comparison (Table 11)

| GNN | CoLA (MCC) | STS-B (Spearman) | RTE (ACC) |
|-----|-----------|------------------|----------|
| **Mistral-7B** | | | |
| AdaPool (no GNN) | 48.00 | 79.55 | 54.87 |
| GLOT (GCN) | 52.65 | 79.74 | 57.04 |
| GLOT (GAT) ✓ | 54.30 | 80.51 | 59.21 |
| GLOT (GIN) | **59.30** | 79.73 | **59.30** |
| **BERT** | | | |
| AdaPool (no GNN) | 29.20 | 80.01 | 51.62 |
| GLOT (GCN) | 45.19 | 80.17 | 58.12 |
| GLOT (GAT) ✓ | 47.49 | **83.86** | **59.21** |
| GLOT (GIN) | **47.78** | 80.71 | 57.04 |

---

## Hyperparameter Tuning Process
- Tool: Weights & Biases
- Strategy: Grid search over the search space in Table 6
- Applied consistently across all backbone models and datasets
- Each configuration is trained for 2 epochs and evaluated on validation set

## Config File Template (`configs/default.yaml`)
```yaml
# Backbone
backbone:
  name: "mistralai/Mistral-7B-v0.1"
  freeze: true

# GLOT
glot:
  gnn_type: "GAT"
  hidden_dim: 128
  num_layers: 2
  jk_mode: "cat"
  threshold: 0.6

# Training  
training:
  epochs: 2
  lr: 2e-4
  weight_decay: 0.0
  batch_size: 32
  eval_batch_size: 64
  seed: 42

# Task
task:
  name: "cola"
  type: "classification"
  num_classes: 2
  max_length: 128
  metric: "mcc"
```
