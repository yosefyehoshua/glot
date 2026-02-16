# GLOT: Graph-based Learning Over Token Graphs

Research prototype implementing GLOT for sentence representations. GLOT is a lightweight pooling module (~8.92M params) that sits on frozen LLM outputs, constructs token-similarity graphs, refines representations with a GNN, and aggregates via learned attention readout.

## Architecture

```
Frozen BERT-base (768-dim)
        |
        v
  Token Hidden States (B, L, 768)
        |
        v
  1. Graph Construction  -- pairwise cosine similarity, threshold at tau=0.6, sparse COO
        |
        v
  2. TokenGNN            -- input projection (768->128) + 2 GATConv layers + ReLU + JK(cat)
        |
        v
  3. Attention Readout   -- MLP scorer + per-graph softmax + weighted sum
        |
        v
  Sentence Embedding (B, 384)
        |
        v
  Task Head (Linear classifier)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v -m "not slow"

# Cache hidden states (one-time, requires GPU recommended)
python cache_hidden_states.py --backbone bert-base-uncased --tasks sst2 cola mrpc

# Train GLOT on SST-2
python train.py --task sst2 --pooler glot

# Train baselines for comparison
python train.py --task sst2 --pooler mean
python train.py --task sst2 --pooler max
python train.py --task sst2 --pooler cls
python train.py --task sst2 --pooler adapool

# Evaluate a checkpoint
python evaluate.py --task sst2 --pooler glot --checkpoint path/to/checkpoint.pt
```

## Project Structure

```
glot/
├── configs/
│   └── default.yaml              # All hyperparameters
├── glot/
│   ├── graph_construction.py     # build_token_graph(): (B,L,d) -> PyG Batch
│   ├── token_gnn.py              # TokenGNN: projection + GATConv + JK
│   ├── readout.py                # AttentionReadout: per-graph attention aggregation
│   ├── glot_pooler.py            # GLOTPooler: wires steps 1-3
│   ├── baselines.py              # MeanPooler, MaxPooler, CLSPooler, AdaPool
│   ├── model.py                  # create_pooler_and_head() factory
│   └── utils.py                  # Metrics, config loading, GLUE task configs
├── data/
│   ├── glue_loader.py            # GLUE task loading, separate pair tokenization
│   └── cache.py                  # CachedDataset, save/load cache
├── train.py                      # Training on cached hidden states
├── cache_hidden_states.py        # Precompute backbone outputs
├── evaluate.py                   # Evaluation on validation set
├── requirements.txt
└── tests/                        # 60 tests across 10 test files
```

## Training Workflow

**Phase 1: Cache Hidden States** -- Run the frozen backbone once and save outputs to disk:
```bash
python cache_hidden_states.py --backbone bert-base-uncased --tasks sst2 cola mrpc
# Saves to cached_hidden_states/bert-base-uncased/{task}/{split}.pt
```

**Phase 2: Train Pooler** -- Train only the pooler (~8.92M params) + task head on cached data:
```bash
python train.py --task sst2 --pooler glot --config configs/default.yaml
```

This two-phase approach means the expensive backbone forward pass happens once, and you can iterate quickly on the pooler.

## Supported Tasks

| Task | Type | Metric | Classes |
|------|------|--------|---------|
| SST-2 | Single sentence | Accuracy | 2 |
| CoLA | Single sentence | MCC | 2 |
| MRPC | Sentence pair | F1 | 2 |

Pair tasks tokenize each sentence separately so GLOT builds individual graphs per sentence, then concatenates the two sentence embeddings before the classifier.

## Poolers

| Pooler | Description | Trainable Params |
|--------|-------------|-----------------|
| `glot` | Graph construction + GATConv GNN + attention readout | ~8.92M |
| `mean` | Average over valid tokens | 0 |
| `max` | Element-wise max over valid tokens | 0 |
| `cls` | CLS token (first token for encoders) | 0 |
| `adapool` | Learned MLP scoring + softmax-weighted average | ~590K (768-dim) |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| GNN type | GATConv |
| GNN layers | 2 |
| Hidden dim | 128 |
| JK mode | cat |
| Threshold (tau) | 0.6 |
| Learning rate | 2e-4 |
| Weight decay | 0.0 |
| Epochs | 2 |
| Batch size | 32 |
| Max sequence length | 128 |

All configurable via `configs/default.yaml`.

## Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4 (+ torch-scatter, torch-sparse)
- HuggingFace Transformers >= 4.36
- HuggingFace Datasets >= 2.16
- scikit-learn, scipy, PyYAML
