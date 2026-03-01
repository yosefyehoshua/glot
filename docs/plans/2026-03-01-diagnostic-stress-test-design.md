# Diagnostic Stress Test + Decoder Backbone Support

## Goal

Implement the signal dilution diagnostic experiment (Step 4 from the implementation guide) — the paper's most compelling result showing GLOT maintains >97% accuracy at 90% distractor noise while baselines collapse to ~50-60%. Includes decoder backbone support for all 6 backbones and experiment tracking via both JSON files and W&B.

## Scope

- Synthetic data generation for signal dilution experiment
- Decoder backbone support (EOS pooling, tokenizer config, backbone registry)
- Training script for all 120 experiment combinations (6 backbones x 5 poolers x 4 ratios)
- Visualization reproducing Figure 3 and Table 7 from the paper
- Unit tests for all new components

---

## Section 1: Data Generation (`data/diagnostic.py`)

Generates synthetic sequences where a short signal phrase (with logical dependency like negation vs affirmation) is injected into a long sequence of random distractor words.

**Key parameters:**
- `seq_length=256` (total tokens)
- `distractor_ratios`: [0.2, 0.5, 0.8, 0.9]
- 10,000 train / 2,000 test samples per ratio
- Binary classification: negation (0) vs affirmation (1)

**Signal templates:** 10 templates (5 per class) with `{X}` and `{Y}` placeholders filled from a content word list. Distractors drawn from NLTK's word corpus (fallback to a built-in list).

**Output:** Returns list of `(text, label)` tuples. No tokenization at this stage — the training script handles that per-backbone.

---

## Section 2: Decoder Backbone Support

### New: `glot/backbone.py`

Helper that loads a frozen backbone + tokenizer with correct config for encoder vs decoder. Single function: `load_backbone(name) -> (model, tokenizer, config)`.

### Backbone Config Registry

| Backbone | Type | Hidden Dim | Pooling Token |
|----------|------|-----------|---------------|
| bert-base-uncased | encoder | 768 | CLS |
| roberta-base | encoder | 768 | CLS |
| HuggingFaceTB/SmolLM2-1.7B | decoder | 2048 | EOS |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | decoder | 2048 | EOS |
| meta-llama/Llama-3.2-3B | decoder | 3072 | EOS |
| mistralai/Mistral-7B-v0.1 | decoder | 4096 | EOS |

### New: `EOSPooler` in `glot/baselines.py`

For decoder models, extracts the last non-padding token using the attention mask. Equivalent of CLSPooler for decoders.

### Updates to `glot/model.py`

`create_pooler_and_head` accepts backbone config and auto-selects CLS vs EOS baseline based on model type.

### Tokenizer Setup for Decoders

Right padding, `pad_token = eos_token` (decoders don't have a native pad token).

---

## Section 3: Diagnostic Training Script (`run_diagnostic.py`)

Runs the full diagnostic experiment — generates data, tokenizes with each backbone, trains all 5 poolers, evaluates across all 4 distractor ratios.

### Workflow per (backbone, pooler, ratio) combination:

1. Load frozen backbone + tokenizer via `load_backbone()`
2. Generate diagnostic data at the given distractor ratio
3. Tokenize (max_length=512)
4. Forward through frozen backbone to get hidden states
5. Train pooler + binary classifier head for 2 epochs (Adam, lr=2e-4)
6. Evaluate on test set, record accuracy

### Total runs: 120

6 backbones x 5 poolers x 4 ratios = 120 experiments

### Logging

- Each run logged to W&B (project: `glot-diagnostic`, grouped by backbone)
- Results also accumulated into `results/diagnostic_results.json`

### CLI Interface

```bash
python run_diagnostic.py --backbone bert-base-uncased --pooler glot --ratio 0.9
python run_diagnostic.py --all  # runs all 120 combinations
```

### Memory Optimization

For large decoders (LLaMA-3B, Mistral-7B), use `torch.float16` and `torch.no_grad()` for backbone inference. The pooler trains in float32.

---

## Section 4: Visualization (`scripts/plot_diagnostic.py`)

Reads `results/diagnostic_results.json` and generates Figure 3 from the paper.

### Layout

- 2x2 grid: 20%, 50%, 80%, 90% distractor ratios
- X-axis: Backbone parameter count (log scale): 110M, 360M, 1.1B, 1.7B, 3.2B, 7.2B
- Y-axis: Classification accuracy (%)
- 5 lines per panel: CLS/EOS, Mean, Max, AdaPool, GLOT
- Color-coded with markers, legend

### Output

- `results/diagnostic_figure.png` and `results/diagnostic_figure.pdf`
- Formatted markdown table (Table 7 format) printed to stdout

---

## Section 5: Test Coverage

All fast, lightweight unit tests — no GPU, no model downloads. Follow existing test suite patterns.

- **`tests/test_diagnostic.py`** — Tests `generate_diagnostic_dataset()`: correct sample count, label balance (~50/50), sequence length, distractor ratio respected, signal phrase present in text.
- **`tests/test_backbone.py`** — Tests `load_backbone()`: encoder returns 768-dim + CLS config, decoder returns correct dim + EOS config, tokenizer has correct padding side and pad token.
- **`tests/test_eos_pooler.py`** — Tests `EOSPooler` correctly picks the last non-padding token from a sequence.

---

## Files Created/Modified

### New Files
- `data/diagnostic.py` — Synthetic data generation
- `glot/backbone.py` — Backbone loading + registry
- `run_diagnostic.py` — Diagnostic experiment script
- `scripts/plot_diagnostic.py` — Visualization
- `tests/test_diagnostic.py`
- `tests/test_backbone.py`
- `tests/test_eos_pooler.py`

### Modified Files
- `glot/baselines.py` — Add EOSPooler
- `glot/model.py` — Update create_pooler_and_head for encoder/decoder
- `requirements.txt` — Add wandb, nltk (if not present)
