# 07: Dependencies and Environment Setup

## requirements.txt
```
# Core
torch>=2.0.0
torch-geometric>=2.4.0

# For GATConv, GCNConv, GINConv
torch-scatter
torch-sparse

# Transformers & datasets
transformers>=4.36.0
datasets>=2.16.0
tokenizers>=0.15.0

# PEFT (for LoRA baseline comparison)
peft>=0.7.0

# Evaluation
scikit-learn>=1.3.0
scipy>=1.11.0
mteb>=1.1.0   # For MTEB benchmarking
ranx           # For retrieval metrics

# Experiment tracking
wandb

# Utilities
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm
pyyaml

# Optional: for diagnostic task vocabulary
nltk
```

## Installation Steps

```bash
# 1. Create environment
conda create -n glot python=3.10
conda activate glot

# 2. Install PyTorch (adjust CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 4. Install remaining deps
pip install transformers datasets peft scikit-learn scipy mteb ranx
pip install wandb matplotlib tqdm pyyaml nltk

# 5. (Optional) Login to HuggingFace for gated models like LLaMA
huggingface-cli login
```

## Hardware Requirements

| Experiment | Min GPU Memory | Recommended |
|-----------|---------------|-------------|
| GLOT training (any backbone) | 0.42 GB (pooler only) | Any GPU |
| Backbone inference (BERT) | ~1 GB | Any GPU |
| Backbone inference (Mistral-7B) | ~14 GB (fp16) | A6000 / A100 |
| Full FT baseline (Mistral-7B) | ~32 GB | A100 40GB |
| LoRA baseline (Mistral-7B) | ~33.5 GB | A100 40GB |

**Key insight**: With cached hidden states, GLOT pooler training only needs 0.42 GB GPU memory regardless of backbone size. The backbone forward pass for caching can be done in batches on any GPU that fits the model.

## Recommended Workflow
1. Load backbone, run forward pass on all training data, cache hidden states to disk
2. Train GLOT pooler on cached hidden states (very fast, minimal GPU memory)
3. Evaluate by running backbone forward pass + GLOT pooler on test data
