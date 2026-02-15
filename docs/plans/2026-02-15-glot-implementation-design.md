# GLOT Implementation Design

## Goal

Research prototype implementing the GLOT paper (Graph-based Learning Over Token Graphs) for sentence representations. Priority: working, clean code for experimentation over completeness.

## Scope

- **Backbone**: BERT-base-uncased (768-dim, encoder) to start
- **Experiments**: GLUE subset (SST-2, CoLA, MRPC) covering single-sentence and pair classification
- **Baselines**: Mean, Max, CLS, AdaPool poolers for comparison
- **Hardware**: NVIDIA datacenter GPU (A6000/A100)
- **Caching**: Precompute and cache backbone hidden states before training pooler

## Architecture

GLOT is a 3-step pooling module (8.92M trainable params) that sits on frozen LLM outputs:

1. **Graph Construction**: Pairwise cosine similarity among tokens, threshold at tau=0.6, sparse adjacency in COO format
2. **TOKEN-GNN**: Input projection (768->128) + 2 GATConv layers + ReLU + Jumping Knowledge (cat)
3. **Readout**: Learned attention weights (MLP scorer + per-graph softmax) + weighted sum

Output dim with default config: 128 x 3 = 384 (JK concatenates input + 2 GNN layer outputs).

## Project Structure

```
glot/
├── configs/
│   └── default.yaml              # All hyperparameters
├── glot/
│   ├── __init__.py
│   ├── graph_construction.py     # build_token_graph(): (B,L,d) -> PyG Batch
│   ├── token_gnn.py              # TokenGNN: projection + GATConv layers + JK
│   ├── readout.py                # AttentionReadout: per-graph attention aggregation
│   ├── glot_pooler.py            # GLOTPooler: wires steps 1-3
│   ├── baselines.py              # MeanPooler, MaxPooler, CLSPooler, AdaPool
│   ├── model.py                  # GLOTModel: frozen backbone + pooler + task head
│   └── utils.py                  # Metrics, config loading
├── data/
│   ├── glue_loader.py            # GLUE task loading, separate pair tokenization
│   └── cache.py                  # Hidden state precomputation and caching
├── train.py                      # Training on cached hidden states
├── cache_hidden_states.py        # Precompute backbone outputs
├── evaluate.py                   # Evaluation on validation set
├── requirements.txt
└── setup.py
```

## Module Interfaces

- `graph_construction.build_token_graph(hidden_states: (B,L,d), attention_mask: (B,L), threshold: float) -> PyG Batch`
- `TokenGNN(input_dim, hidden_dim=128, num_layers=2, jk_mode='cat').forward(x, edge_index) -> (N_total, output_dim)`
- `AttentionReadout(input_dim).forward(x, batch) -> (B, D)`
- `GLOTPooler(input_dim, ...).forward(hidden_states, attention_mask) -> (B, output_dim)`
- All baselines: `forward(hidden_states, attention_mask) -> (B, d)`

## Training Workflow

### Phase 1: Cache Hidden States
1. Load BERT-base frozen
2. Tokenize GLUE tasks (pair tasks tokenized separately per sentence)
3. Run backbone forward, save `{hidden_states, masks, labels}` as `.pt` files
4. Cache path: `cached_hidden_states/{backbone}/{task}/{split}.pt`

### Phase 2: Train Pooler
1. Load cached hidden states
2. Instantiate pooler (GLOT or baseline) + task head (Linear classifier)
3. Adam optimizer, lr=2e-4, weight_decay=0.0, 2 epochs, batch_size=32
4. Evaluate on validation split with task-specific metric

## Task Heads

- Single-sentence (CoLA, SST-2): `Linear(pool_dim, num_classes)`, CrossEntropy
- Pair classification (MRPC): `Linear(pool_dim * 2, num_classes)`, CrossEntropy, encode sentences separately then concatenate
- Metrics: MCC (CoLA), Accuracy (SST-2), F1 (MRPC), all x100

## Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| GNN type | GATConv |
| GNN layers (K) | 2 |
| Hidden dim | 128 |
| JK mode | cat |
| Threshold (tau) | 0.6 |
| Learning rate | 2e-4 |
| Weight decay | 0.0 |
| Epochs | 2 |
| Batch size (train) | 32 |
| Batch size (eval) | 64 |
| Max length (GLUE) | 128 |
| Seed | 42 |

## Edge Cases

- **Empty graphs** (zero edges after thresholding): GNN passes through input projection only; readout still works on node features. Graceful degradation to non-relational representation.
- **Variable sequence lengths**: Handled by PyG Batch which manages variable-size graphs natively.
- **Pair tasks**: Each sentence gets its own graph, encoded independently through the same pooler.

## Dependencies

- torch>=2.0, torch-geometric>=2.4, torch-scatter, torch-sparse
- transformers>=4.36, datasets>=2.16
- scikit-learn, scipy
- pyyaml, tqdm, wandb (optional)

## Future Extensions (out of scope for now)

- Decoder backbones (TinyLlama, LLaMA-3B, Mistral-7B)
- Diagnostic stress test (signal dilution)
- MTEB contrastive training + evaluation
- Full GLUE (QQP, MNLI, QNLI, RTE, WNLI, STS-B)
- GNN architecture ablations (GCN, GIN)
- Threshold ablation (tau in {0.1, 0.3, 0.6})
