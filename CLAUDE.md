# CLAUDE.md — GLOT

## Project Overview

GLOT (Graph-based Learning Over Token Graphs) is a research prototype implementing a lightweight pooling module for sentence representations on top of frozen LLMs. It constructs edge-weighted token-similarity graphs, refines representations with a GNN (GAT/GCN/GIN/GINE), and aggregates via learned attention-based readout.

## Tech Stack

- Python 3.10+, PyTorch >= 2.0, PyTorch Geometric >= 2.4, HuggingFace Transformers/Datasets
- scikit-learn, scipy, PyYAML, numpy, tqdm
- Dev: pytest, wandb (optional), matplotlib (optional)

## Project Structure

```
glot/                    # Core library
  graph_construction.py  # Edge-weighted token graph via cosine similarity + threshold
  token_gnn.py           # Multi-GNN (GAT/GCN/GIN/GINE) with Jumping Knowledge
  readout.py             # Attention-based aggregation with scaled scorer
  baselines.py           # Baseline poolers: mean, max, cls, eos, adapool
  backbone.py            # Backbone registry + loader (encoders/decoders)
  model.py               # GLOTPooler + factory: create_pooler_and_head()
  utils.py               # Metrics, config loading, GLUE task definitions
data/                    # Data loading
  glue_loader.py         # GLUE task loading & tokenization
  cache.py               # CachedDataset & save/load utilities
  diagnostic.py          # Synthetic signal dilution dataset generator
tests/                   # 12 test files, 93 test cases
configs/default.yaml     # Hyperparameter configuration
train.py                 # Training loop on cached hidden states
cache_hidden_states.py   # Backbone precomputation script
evaluate.py              # Evaluation on cached validation data
run_diagnostic.py        # Diagnostic stress test runner
scripts/plot_diagnostic.py  # Visualization script
notebooks/demo.ipynb     # Colab demo notebook
docs/plans/              # Historical design & implementation plans
```

## Common Commands

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v
pytest tests/ -v -m "not slow"    # Skip slow tests

# Two-phase workflow
python cache_hidden_states.py --backbone bert-base-uncased --tasks sst2 cola mrpc
python train.py --task sst2 --pooler glot --config configs/default.yaml

# Evaluate
python evaluate.py --task sst2 --pooler glot --checkpoint path/to/checkpoint.pt

# Diagnostic stress test
python run_diagnostic.py --backbone bert-base-uncased --pooler glot --ratio 0.9
```

## Coding Conventions

- **Pooler types**: lowercase strings (`glot`, `mean`, `max`, `cls`, `eos`, `adapool`)
- **Backbone names**: HuggingFace format (`bert-base-uncased`, `HuggingFaceTB/SmolLM2-1.7B`)
- **Variable naming**: `hidden_states`/`hs` for (B,L,d) tensors, `attention_mask`/`mask` for (B,L) masks, `batch_data` for PyG Batch objects
- **Task types**: `"classification"` (single-sentence) or `"pair_classification"` (pair-sentence) — derived from `GLUE_TASKS[task]["type"]`
- **Graph batching**: Use PyG `Batch.from_data_list()` for variable-length sequences
- **Masking**: Always respect attention_mask for variable-length sequences
- **Device handling**: Explicit `.to(device)` calls; frozen models use `.eval()`
- **Pair tasks**: Tokenize sentences separately, concatenate embeddings before classifier
- **Metrics**: `compute_metrics()` returns scores × 100 (percentage format)
- **Testing**: pytest with class-based organization, `torch.manual_seed(42)` for determinism, shape assertions + gradient checks
- **GNN types**: uppercase strings (`GAT`, `GCN`, `GIN`, `GINE`) — config key `gnn_type`
- **Factory pattern**: `create_pooler_and_head()` in `model.py` for instantiation
- **Config param mapping**: YAML `num_layers` → code `num_gnn_layers`, YAML `gnn_type` → code `gnn_type` (mapped in `train.py`)
- **Cache paths**: `cached_hidden_states/{backbone.replace("/","_")}/{task}/{split}.pt`
- **Subsampling**: QQP, MNLI, QNLI are subsampled to 20K training samples for efficiency

## Architecture (3-Stage Pipeline)

1. **Graph Construction**: Cosine similarity → threshold (τ=0.6) → edge-weighted adjacency (edge_attr = cosine sim) → COO edge lists → PyG Data
2. **TokenGNN**: No input projection — first GNN layer takes raw d-dim input → K GNN layers (GAT/GCN/GIN/GINE) + ReLU → Jumping Knowledge (cat) → output dim = d + K×hidden_dim (e.g., 768 + 2×128 = 1024 for BERT)
3. **Attention Readout**: MLP scorer (Linear(d_in, max(128, d_in//2))→Tanh→Linear(1)) → per-graph softmax weights → weighted sum → sentence embedding (B, output_dim)

Edge cases: Empty graphs (zero edges after thresholding) degrade gracefully — GNN layers receive no messages, JK cat still includes raw input. Variable sequence lengths handled natively by PyG Batch.

## Key Module APIs

```
build_token_graph(hidden_states: (B,L,d), attention_mask: (B,L), threshold=0.6) -> PyG Batch (with edge_attr)
TokenGNN(input_dim, hidden_dim=128, num_layers=2, jk_mode='cat', gnn_type='GAT').forward(x, edge_index, edge_attr=None) -> (N, output_dim)
AttentionReadout(input_dim).forward(x, batch) -> (B, D)
GLOTPooler(input_dim, hidden_dim=128, num_gnn_layers=2, jk_mode='cat', threshold=0.6, gnn_type='GAT').forward(hs, mask) -> (B, output_dim)
create_pooler_and_head(pooler_type, input_dim, num_classes, task_type, glot_config=None) -> (pooler, head)
train_epoch(pooler, head, loader, optimizer, loss_fn, task_type, device) -> avg_loss
evaluate_epoch(pooler, head, loader, task_type, device) -> (preds, labels)
```

`glot_config` dict keys: `hidden_dim`, `num_gnn_layers`, `jk_mode`, `threshold`, `gnn_type` — passed to GLOTPooler via `**kwargs`.
GLOTPooler lives in `model.py` (not a separate file).
Checkpoint format (evaluate.py): `{"pooler": state_dict, "head": state_dict}`.

## Supported Backbones

Encoders: `bert-base-uncased` (768-dim), `roberta-base` (768-dim)
Decoders: `HuggingFaceTB/SmolLM2-1.7B` (2048-dim), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2048-dim), `meta-llama/Llama-3.2-3B` (3072-dim), `mistralai/Mistral-7B-v0.1` (4096-dim)

Decoders use right padding, `pad_token = eos_token`, and EOSPooler (last non-padding token) instead of CLSPooler.

## Supported GLUE Tasks

Single-sentence: SST-2 (accuracy, 2 classes), CoLA (MCC, 2 classes)
Pair-sentence: MRPC (F1, 2), QQP (F1, 2, subsampled 20K), STS-B (Spearman, regression num_classes=1), MNLI (accuracy, 3 classes, subsampled 20K), QNLI (accuracy, 2, subsampled 20K), RTE (accuracy, 2), WNLI (accuracy, 2)

All defined in `GLUE_TASKS` dict in `glot/utils.py` with `sentence_keys` for tokenization.

## Diagnostic Stress Test

Signal dilution experiment testing pooler robustness: synthetic sequences with signal phrases buried in distractor words. 120 combinations: 6 backbones × 5 poolers × 4 distractor ratios (0.2, 0.5, 0.8, 0.9). Results saved to `results/diagnostic_results.json`.

## Design Documents

Historical design and implementation plans in `docs/plans/`:
- `2026-02-15-glot-implementation-design.md` — Architecture decisions, module interfaces, hyperparameters, edge cases
- `2026-02-15-glot-implementation-plan.md` — 14-task step-by-step TDD build plan (the main implementation guide)
- `2026-02-16-colab-demo-notebook.md` — Colab demo notebook plan
- `2026-03-01-diagnostic-stress-test-design.md` — Diagnostic experiment + decoder backbone design
- `2026-03-01-diagnostic-stress-test-plan.md` — 10-task diagnostic implementation plan

Note: README.md is outdated (only covers initial 3 GLUE tasks, no decoder/diagnostic support).
