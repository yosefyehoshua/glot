# Reproduce Original GLOT Paper Results — Design Doc

**Date:** 2026-03-03
**Goal:** Run the original GLOT implementation (ipsitmantri/GLOT) on Google Colab to reproduce the paper's reported results and compare against our re-implementation.

## Context

The GLOT paper authors published their official code at https://github.com/ipsitmantri/GLOT. Their repo is minimal (7 files), with a monolithic `main.py` (~1200 lines) that handles all poolers, data loading, training, MTEB evaluation, and a separate `diagnostic_stress_test.py`.

Our re-implementation is modular (13+ source files, 83 tests) but may differ from the original in hyperparameters or implementation details. Reproducing results with the original code validates the paper's claims independently.

## Key Differences: Original vs Our Implementation

| Aspect | Original | Ours |
|--------|----------|------|
| Structure | Monolithic `main.py` | Modular `glot/` package |
| Default threshold τ | 0.3 | 0.6 |
| GNN types | GAT, GCN, GIN, GINE | GAT only |
| GATConv edge_dim | 1 (edge-weighted) | None |
| JK cat output | in_dim + num_layers × hidden_dim | hidden_dim × (num_layers + 1) |
| Score layer hidden | max(128, out_dim // 2) | Fixed 128 |
| Readout | torch_scatter.scatter_add | PyG scatter via AttentionReadout |
| MTEB support | Yes | No |
| Contrastive training | Yes | No |

## Approach

**Separate clone**: The original repo will be cloned inside Google Colab at runtime. No changes to our repo's git structure.

**Colab notebook**: A new notebook `notebooks/reproduce_org_glot.ipynb` in our repo orchestrates everything.

## Notebook Structure

### 1. Setup & Environment
- Check GPU availability (T4/A100)
- Clone `ipsitmantri/GLOT`
- Install their dependencies from `requirements.txt`
- Handle CUDA/PyG wheel compatibility for Colab's CUDA version
- Mount Google Drive for persistent result storage
- Set HF token from Colab secrets

### 2. GLUE Benchmark Experiments
Run `main.py` for each combination:
- **Backbones**: bert-base-uncased, roberta-base (encoder-only first)
- **Poolers**: mean, max, cls, adapool, glot
- **Tasks**: sst2, cola, mrpc, qqp, stsb, mnli, qnli, rte, wnli
- Precompute hidden states first, then train pooler heads

### 3. Diagnostic Stress Test
Run `diagnostic_stress_test.py`:
- **Backbones**: bert-base-uncased, roberta-base
- **Poolers**: mean, max, cls, adapool, glot
- **Distractor ratios**: 0.2, 0.5, 0.8, 0.9

### 4. Decoder Experiments (Optional)
If GPU allows (Colab Pro / A100):
- SmolLM2-1.7B, TinyLlama-1.1B
- Same GLUE tasks and poolers

### 5. Results Collection & Comparison
- Parse W&B logs or stdout for metrics
- Build comparison tables: original results vs paper Table 1/2
- Save structured JSON results to Google Drive
- Display summary with pass/fail against paper numbers (within reasonable tolerance)

## Colab Considerations

- **Session limits**: Free tier ~12h. Encoder experiments should complete well within this. Save intermediate results to Drive.
- **CUDA version**: Colab may have CUDA 12.1/12.4, not 12.9. Need to adjust PyG wheel URLs in their requirements.
- **Memory**: BERT/RoBERTa fit easily on T4 (16GB). Mistral-7B needs A100.
- **W&B**: Their code uses wandb heavily. Can either set up a W&B account or capture results from stdout.
- **Reproducibility**: Set seed=0 (their default) for all runs.

## Success Criteria

- All GLUE experiments complete without errors
- Results are within ±2% of paper-reported numbers (accounting for hardware/seed variance)
- Results are saved persistently and formatted for easy comparison
