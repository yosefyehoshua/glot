# Colab Demo Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Google Colab notebook that demos the full GLOT pipeline — cache hidden states, train all 5 poolers (GLOT, mean, max, cls, adapool) on GLUE tasks, and compare results with tables and charts.

**Architecture:** Single self-contained `.ipynb` notebook. Clones the repo, installs deps, runs caching once, trains each pooler in a loop, collects metrics, and renders a comparison table + bar chart. Uses Colab's free T4 GPU.

**Tech Stack:** PyTorch, PyTorch Geometric, HuggingFace Transformers/Datasets, matplotlib, Google Colab

---

### Task 1: Create notebook skeleton with setup cells

**Files:**
- Create: `notebooks/demo.ipynb`

**Step 1: Create the notebook file with the first 4 cells**

Cell 1 — Markdown title and overview:

```markdown
# GLOT: Graph-based Learning Over Token Graphs — Demo

This notebook demonstrates the full GLOT pipeline:
1. **Cache** frozen BERT-base hidden states for GLUE tasks
2. **Train** 5 pooling strategies on cached representations
3. **Compare** results across tasks and poolers

**Hardware:** Requires GPU runtime. Go to `Runtime → Change runtime type → T4 GPU`.

> **Paper concept:** Instead of using simple CLS/mean/max pooling over transformer outputs, GLOT builds a token-similarity graph and refines representations with a Graph Attention Network before aggregating via learned attention.
```

Cell 2 — GPU check:

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. Go to Runtime → Change runtime type → T4 GPU"
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
device = "cuda"
```

Cell 3 — Install dependencies:

```python
%%capture
!pip install torch-geometric torch-scatter torch-sparse \
    transformers datasets scikit-learn scipy pyyaml tqdm
```

Cell 4 — Clone repo and add to path:

```python
import os, sys

if not os.path.exists("glot"):
    !git clone https://github.com/yosefyehoshua/glot.git

sys.path.insert(0, "glot")
print("Setup complete ✓")
```

**Step 2: Verify the notebook is valid JSON and opens**

Run: `python -c "import json; json.load(open('notebooks/demo.ipynb'))"`
Expected: No error

**Step 3: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "feat: add Colab demo notebook skeleton with setup cells"
```

---

### Task 2: Add caching cell

**Files:**
- Modify: `notebooks/demo.ipynb` (add cells 5-6)

**Step 1: Add markdown + caching cell**

Cell 5 — Markdown:

```markdown
## Phase 1: Cache Hidden States

Run the frozen BERT-base backbone once on each task and save the hidden states to disk. This is the expensive step (~2-3 min on T4). Subsequent training runs are fast since they only train the small pooler.
```

Cell 6 — Caching code:

```python
TASKS = ["sst2", "cola", "mrpc"]
BACKBONE = "bert-base-uncased"
CACHE_DIR = "cached_hidden_states"

# Only cache if not already done
if not os.path.exists(f"{CACHE_DIR}/{BACKBONE}/sst2/train.pt"):
    os.chdir("glot")
    !python cache_hidden_states.py \
        --backbone {BACKBONE} \
        --tasks {" ".join(TASKS)} \
        --batch_size 64 \
        --device cuda
    os.chdir("..")
else:
    print("Cache already exists, skipping ✓")

# Show cached files
for task in TASKS:
    path = f"glot/{CACHE_DIR}/{BACKBONE}/{task}"
    files = os.listdir(path)
    sizes = {f: os.path.getsize(f"{path}/{f}") / 1e6 for f in files}
    print(f"  {task}: {sizes}")
```

**Step 2: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "feat: add hidden state caching cell to demo notebook"
```

---

### Task 3: Add training loop cell

**Files:**
- Modify: `notebooks/demo.ipynb` (add cells 7-8)

**Step 1: Add markdown + training cell**

Cell 7 — Markdown:

```markdown
## Phase 2: Train All Poolers

Train each of the 5 pooling strategies on cached hidden states. Each trains for 2 epochs (takes ~30s per pooler per task on T4).

| Pooler | Params | Description |
|--------|--------|-------------|
| **GLOT** | ~8.9M | Graph-based: builds token similarity graph → GATConv → attention readout |
| **AdaPool** | ~590K | Learned MLP attention weights → weighted average |
| **Mean** | 0 | Average over valid tokens |
| **Max** | 0 | Element-wise max over valid tokens |
| **CLS** | 0 | Use [CLS] token representation |
```

Cell 8 — Training loop:

```python
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from data.cache import make_cached_dataset
from train import train_epoch, evaluate_epoch

cfg = load_config("glot/configs/default.yaml")

POOLERS = ["glot", "adapool", "mean", "max", "cls"]
results = {}  # {(task, pooler): score}

for task_name in TASKS:
    task_cfg = GLUE_TASKS[task_name]
    task_type = "pair_classification" if task_cfg["type"] == "pair" else "classification"
    backbone_short = BACKBONE.replace("/", "_")

    train_ds = make_cached_dataset(f"glot/{CACHE_DIR}/{backbone_short}/{task_name}/train.pt")
    val_ds = make_cached_dataset(f"glot/{CACHE_DIR}/{backbone_short}/{task_name}/validation.pt")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    for pooler_name in POOLERS:
        print(f"\n{'='*50}")
        print(f"Training {pooler_name.upper()} on {task_name}")
        print(f"{'='*50}")

        glot_cfg = cfg["glot"] if pooler_name == "glot" else None
        pooler, head = create_pooler_and_head(
            pooler_type=pooler_name,
            input_dim=768,
            num_classes=task_cfg["num_classes"],
            task_type=task_type,
            glot_config=glot_cfg,
        )
        pooler.to(device)
        head.to(device)

        params = list(pooler.parameters()) + list(head.parameters())
        n_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")

        optimizer = Adam(params, lr=cfg["training"]["lr"])

        if task_cfg["num_classes"] == 1:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        best_score = -float("inf")
        for epoch in range(cfg["training"]["epochs"]):
            avg_loss = train_epoch(pooler, head, train_loader, optimizer, loss_fn, task_type, device)
            preds, labels = evaluate_epoch(pooler, head, val_loader, task_type, device)
            score = compute_metrics(preds, labels, task_cfg["metric"])
            best_score = max(best_score, score)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  {task_cfg['metric']}={score:.2f}")

        results[(task_name, pooler_name)] = best_score
        print(f"  Best: {best_score:.2f}")

print("\n\nAll training complete ✓")
```

**Step 2: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "feat: add multi-pooler training loop to demo notebook"
```

---

### Task 4: Add results comparison cells

**Files:**
- Modify: `notebooks/demo.ipynb` (add cells 9-12)

**Step 1: Add markdown + results table + chart cells**

Cell 9 — Markdown:

```markdown
## Phase 3: Results Comparison

Compare all pooling strategies across GLUE tasks.
```

Cell 10 — Results table:

```python
import pandas as pd

metrics = {task: GLUE_TASKS[task]["metric"] for task in TASKS}

rows = []
for pooler_name in POOLERS:
    row = {"Pooler": pooler_name.upper()}
    for task_name in TASKS:
        metric = metrics[task_name]
        score = results.get((task_name, pooler_name), float("nan"))
        row[f"{task_name}\n({metric})"] = f"{score:.2f}"
    rows.append(row)

df = pd.DataFrame(rows)
df.set_index("Pooler", inplace=True)
print(df.to_string())
df
```

Cell 11 — Bar chart:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 5), sharey=False)
if len(TASKS) == 1:
    axes = [axes]

colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

for ax, task_name in zip(axes, TASKS):
    metric = GLUE_TASKS[task_name]["metric"]
    scores = [results.get((task_name, p), 0) for p in POOLERS]
    labels = [p.upper() for p in POOLERS]

    bars = ax.bar(labels, scores, color=colors)
    ax.set_title(f"{task_name.upper()}\n({metric})", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(min(scores) - 5, max(scores) + 5)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{score:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

fig.suptitle("GLOT vs Baseline Poolers on GLUE", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("glot_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

Cell 12 — Summary markdown:

```markdown
## Summary

**Key takeaways:**
- **GLOT** uses graph neural networks to capture token-token relationships that simple pooling misses
- The graph is constructed from cosine similarity between token hidden states (threshold τ=0.6)
- Only **~9M parameters** are trained — the BERT backbone stays frozen
- The two-phase design (cache once → train many) makes experimentation fast

**Next steps:**
- Try different backbones (RoBERTa, DeBERTa)
- Tune the graph threshold τ
- Run on more GLUE tasks (QQP, MNLI, QNLI)
- Experiment with deeper GNN layers or different GNN architectures (GCN, GraphSAGE)
```

**Step 2: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "feat: add results table and comparison chart to demo notebook"
```

---

### Task 5: Test the notebook end-to-end locally

**Files:**
- Read: `notebooks/demo.ipynb`

**Step 1: Validate notebook JSON structure**

Run: `python -c "import json; nb = json.load(open('notebooks/demo.ipynb')); print(f'{len(nb[\"cells\"])} cells, format v{nb[\"nbformat\"]}')" `
Expected: `12 cells, format v4`

**Step 2: Verify all imports resolve**

Run: `cd /Users/Yosef.Yehoshua/PycharmProjects/glot && python -c "from glot.model import create_pooler_and_head; from glot.utils import compute_metrics, GLUE_TASKS; from data.cache import make_cached_dataset; from train import train_epoch, evaluate_epoch; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Final commit and push**

```bash
git add -A
git commit -m "feat: complete Colab demo notebook for GLOT pooler comparison"
git push origin main
```
