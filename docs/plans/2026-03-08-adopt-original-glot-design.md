# Design: Adopt Original GLOT Code

**Date:** 2026-03-08
**Goal:** Replace the current reimplementation with the original GLOT code from [ipsitmantri/GLOT](https://github.com/ipsitmantri/GLOT), extracted into the existing modular project structure.

## Motivation

- Reproduction accuracy: reimplementation diverges from paper results due to architectural differences
- Feature completeness: original has MTEB support, 4 GNN types, edge-weighted graphs
- Trust: build on author-validated code rather than maintaining a reimplementation

## Critical Differences (Current vs Original)

| Aspect | Current | Original |
|--------|---------|----------|
| JK output dim | `p × (K+1)` = 384 | `d + K×p` = 1024 (BERT) |
| Graph edges | Binary (unweighted) | Weighted (cosine sim as `edge_attr`) |
| Default τ | 0.6 | 0.3 |
| GNN types | GAT only | GAT, GCN, GIN, GINE |
| Scorer hidden | Fixed 128 | `max(128, out_dim // 2)` |
| MTEB | Not supported | Contrastive training + 56-task eval |

## Module Mapping

### Rewritten (original logic replaces current)

- **`glot/graph_construction.py`** — Edge-weighted graph building with cosine similarity stored as `edge_attr`. Default τ=0.3.
- **`glot/token_gnn.py`** — Multi-GNN support (GAT/GCN/GIN/GINE). JK concatenation includes input dim: `output_dim = d + K×p`. GATConv and GINE receive `edge_dim=1`.
- **`glot/readout.py`** — Scaled scorer: `hidden = max(128, output_dim // 2)`.
- **`data/diagnostic.py`** — Port Algorithm 2 from `diagnostic_stress_test.py`.

### Updated

- **`glot/model.py`** — Absorb `glot_pooler.py` (GLOTPooler class moves here). Update output dim formula. Add `gnn_type` parameter to factory.
- **`train.py`** — Add MTEB contrastive training mode (symmetric in-batch loss, T=0.07).
- **`run_diagnostic.py`** — Match original's experiment runner.
- **`configs/default.yaml`** — τ=0.3, add `gnn_type: GAT`, `scorer_hidden: auto`.

### Kept as-is

- `glot/baselines.py` — Functionally identical to original.
- `glot/backbone.py` — Already supports all 6 backbones.
- `glot/utils.py` — Metrics and GLUE task defs match.
- `data/glue_loader.py`, `data/cache.py` — Same approach.
- `notebooks/`, `docs/plans/` — Preserved.

### New

- **`data/mteb_loader.py`** — MS MARCO passage loading for contrastive training.

### Deleted

- **`glot/glot_pooler.py`** — Merged into `model.py`.

## Structure After Migration

```
glot/
  graph_construction.py   # edge-weighted graph building
  token_gnn.py            # GAT/GCN/GIN/GINE + JK (d + K×p)
  readout.py              # scaled attention aggregation
  model.py                # GLOTPooler + factory + task heads
  baselines.py            # unchanged
  backbone.py             # unchanged
  utils.py                # unchanged
data/
  glue_loader.py          # unchanged
  cache.py                # unchanged
  diagnostic.py           # rewritten from original
  mteb_loader.py          # new
```

## Test Impact

~6 of 13 test files need updates:
- `test_graph_construction.py` — edge weights, τ=0.3
- `test_token_gnn.py` — new output dims, 4 GNN types, edge_attr
- `test_readout.py` — scaled scorer hidden dim
- `test_glot_pooler.py` → `test_model.py` — merged, new output dims
- `test_diagnostic.py` — new Algorithm 2 data generator
- `test_e2e.py` — updated dims cascade through

Unaffected: `test_baselines.py`, `test_backbone.py`, `test_glue_loader.py`, `test_cache.py`, `test_train.py`, `test_utils.py`.

## Implementation Order

1. `glot/graph_construction.py` + tests (edge weights, τ=0.3)
2. `glot/token_gnn.py` + tests (4 GNN types, correct JK formula)
3. `glot/readout.py` + tests (scaled scorer)
4. `glot/model.py` + tests (absorb glot_pooler, update factory)
5. Delete `glot/glot_pooler.py`, update imports
6. `data/diagnostic.py` + tests (Algorithm 2)
7. `data/mteb_loader.py` + tests
8. `train.py` updates (contrastive mode)
9. `run_diagnostic.py` updates
10. `configs/default.yaml` updates
11. Update notebooks for API changes
12. Validate against original code output (smoke test)
