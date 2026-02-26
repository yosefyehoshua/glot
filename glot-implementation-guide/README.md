# GLOT: Graph-based Learning Over Token Graphs

## Paper Reference
"Towards Improved Sentence Representations Using Token Graphs" (Under review at ICLR 2026)

## Project Overview

GLOT is a lightweight, structure-aware pooling module that produces sentence-level representations from frozen LLM token hidden states. Instead of treating tokens as an independent set (like mean/max pooling), GLOT constructs a latent token-similarity graph, refines representations with a GNN, and aggregates them via a learned readout.

## Key Innovation
Standard pooling (mean, max, CLS/EOS) treats tokens independently → discards relational structure → susceptible to signal dilution. GLOT reframes pooling as **relational learning followed by aggregation**.

## Architecture (3 Steps)

```
Input Tokens → Frozen LLM → Token Hidden States X ∈ R^{L×d}
                                    ↓
                        ┌──────────────────────┐
                        │       GLOT           │
                        │                      │
                        │  Step 1: Build Graph │  Cosine similarity → threshold τ → sparse adjacency
                        │  Step 2: TOKEN-GNN   │  K layers of GATConv message passing
                        │  Step 3: Readout     │  Learned attention-weighted aggregation
                        │                      │
                        └──────────────────────┘
                                    ↓
                          Sentence Vector z ∈ R^D
                                    ↓
                        Task-Specific Head (linear classifier / cosine sim)
```

## Key Results
- **GLUE benchmark**: Consistently outperforms all baselines across 6 frozen backbones
- **Signal dilution stress test**: Maintains >97% accuracy at 90% distractor ratio (baselines collapse to ~50-60%)
- **Efficiency**: 8.92M trainable params (vs 167.8M LoRA, 7.11B Full FT), 0.42 GB memory, 100x faster training
- **Works on both encoder (BERT, RoBERTa) and decoder (LLaMA, Mistral) models**

## File Structure to Implement

```
glot/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml              # Hyperparameter configs
├── glot/
│   ├── __init__.py
│   ├── graph_construction.py     # Step 1: Token graph building
│   ├── token_gnn.py              # Step 2: GNN message passing
│   ├── readout.py                # Step 3: Attention readout
│   ├── glot_pooler.py            # Main GLOT module combining steps 1-3
│   ├── model.py                  # Full model: frozen backbone + GLOT + task head
│   ├── baselines.py              # Mean, Max, CLS/EOS, AdaPool baselines
│   └── utils.py                  # Helpers
├── data/
│   ├── glue_loader.py            # GLUE benchmark data loading
│   ├── imdb_loader.py            # IMDB long-text data
│   ├── mteb_loader.py            # MTEB benchmark data
│   └── diagnostic.py             # Synthetic stress test data generation
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── train_mteb.py                 # MTEB contrastive training on MS MARCO
└── scripts/
    ├── run_glue.sh
    ├── run_diagnostic.sh
    └── run_mteb.sh
```

## Implementation Order
1. Start with `01_CORE_MODULE.md` — implements the GLOT module itself
2. Then `02_MODEL_AND_BASELINES.md` — wraps GLOT with frozen backbones and task heads
3. Then `03_DATA_AND_TRAINING.md` — data loading, training loop, evaluation
4. Then `04_DIAGNOSTIC_TASK.md` — synthetic stress test
5. Then `05_MTEB_EVALUATION.md` — large-scale MTEB benchmarking
6. Reference `06_HYPERPARAMETERS.md` throughout for exact values
