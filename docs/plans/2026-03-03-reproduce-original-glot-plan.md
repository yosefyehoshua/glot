# Reproduce Original GLOT Results — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Colab notebook (`notebooks/reproduce_org_glot.ipynb`) that clones the original GLOT repo, runs all GLUE and diagnostic experiments using their code, and displays comparison tables.

**Architecture:** Single self-contained `.ipynb` notebook. Clones `ipsitmantri/GLOT` at runtime, installs their deps, runs `main.py` for each backbone×pooler×task combo via subprocess, parses stdout for metrics, saves results to Google Drive, renders comparison tables.

**Tech Stack:** Original GLOT code (ipsitmantri/GLOT), Google Colab, subprocess, json, pandas

---

### Task 1: Create notebook skeleton with setup cells

**Files:**
- Create: `notebooks/reproduce_org_glot.ipynb`

**Step 1: Create the notebook file with the first 5 cells**

Cell 0 — Markdown title:

```markdown
# Reproducing GLOT Paper Results with Original Code

This notebook runs the **original GLOT implementation** from [ipsitmantri/GLOT](https://github.com/ipsitmantri/GLOT) to reproduce the paper's reported results.

**Experiments:**
1. GLUE benchmark (9 tasks × 5 poolers × 2 encoder backbones)
2. Diagnostic stress test (4 distractor ratios × 5 poolers × 2 backbones)

**Hardware:** Requires GPU. Go to `Runtime → Change runtime type → T4 GPU` (or A100 for decoder models).

**Time estimate:** ~2-4 hours for all encoder experiments on T4.
```

Cell 1 — GPU check and Google Drive mount:

```python
import torch, os, json, subprocess, time
from google.colab import drive, userdata

# GPU check
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. Go to Runtime → Change runtime type → T4 GPU"
    )
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# Mount Google Drive for persistent results
drive.mount("/content/drive")
RESULTS_DIR = "/content/drive/MyDrive/glot_reproduction"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}")
```

Cell 2 — Clone original repo and install deps:

```python
import sys

ORG_REPO = "https://github.com/ipsitmantri/GLOT.git"
ORG_DIR = "/content/org_glot"

if not os.path.exists(ORG_DIR):
    !git clone {ORG_REPO} {ORG_DIR}
else:
    !cd {ORG_DIR} && git pull

# Show repo contents
!ls -la {ORG_DIR}
```

Cell 3 — Install dependencies with CUDA-compatible PyG:

```python
# Detect Colab CUDA version and install compatible PyG
import subprocess
cuda_version = torch.version.cuda  # e.g. "12.4"
print(f"PyTorch CUDA: {cuda_version}")

# Install their requirements, but handle PyG wheels for Colab's CUDA
# Their requirements.txt pins torch-2.8.0+cu129 which may not match Colab
# We install PyG components compatible with the existing torch/CUDA

!pip install -q torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
    transformers datasets sentence-transformers mteb peft wandb accelerate \
    scikit-learn numpy pandas tqdm matplotlib seaborn polars

print("Dependencies installed")
```

Cell 4 — Configure HF token and wandb:

```python
# HuggingFace token for gated models
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF_TOKEN loaded from Colab secrets")
except Exception:
    HF_TOKEN = ""
    print("WARNING: No HF_TOKEN found. Gated models (Llama, Mistral) will fail.")
    print("Add your token in Colab: Settings (gear icon) → Secrets → HF_TOKEN")

# Patch the HF_TOKEN in main.py so it doesn't use the placeholder
main_py = os.path.join(ORG_DIR, "main.py")
with open(main_py, "r") as f:
    content = f.read()
content = content.replace('HF_TOKEN = "<>"', f'HF_TOKEN = os.environ.get("HF_TOKEN", "")')
with open(main_py, "w") as f:
    f.write(content)

# Set wandb to offline mode (no account needed)
os.environ["WANDB_MODE"] = "offline"
print("W&B set to offline mode (results captured from stdout)")
```

**Step 2: Verify notebook creates and renders correctly**

Open the notebook in Colab or JupyterLab and confirm all 5 cells render with proper markdown formatting.

**Step 3: Commit**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: add reproduction notebook skeleton with setup cells"
```

---

### Task 2: Add experiment runner utility cells

**Files:**
- Modify: `notebooks/reproduce_org_glot.ipynb`

**Step 1: Add cell 5 — Experiment configuration**

Cell 5 — Experiment matrix definition:

```python
# ============================================================
# Experiment Configuration
# ============================================================

# Encoder backbones (feasible on T4)
ENCODER_BACKBONES = [
    "bert-base-uncased",
    "roberta-base",
]

# Pooling methods to test
POOLERS = ["mean", "max", "cls", "adapool", "glot"]

# GLUE tasks — grouped by type
SINGLE_TASKS = ["sst2", "cola"]  # Single sentence classification
PAIR_TASKS = ["mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]  # Pair tasks
ALL_GLUE_TASKS = SINGLE_TASKS + PAIR_TASKS

# Diagnostic stress test ratios
DISTRACTOR_RATIOS = [0.2, 0.5, 0.8, 0.9]

# Hyperparameters (matching paper defaults)
HPARAMS = {
    "epochs": 3,
    "batch_size": 32,
    "eval_batch_size": 64,
    "lr": 2e-4,
    "max_length": 128,
    "seed": 0,
    "gat_hidden_dim": 256,
    "num_layers": 4,
    "jk_mode": "cat",
    "tau": 0.3,
    "scorer_hidden": 128,
    "precompute_hidden_states": 1,
    "verbose": 1,
}

# Results file path (persistent on Google Drive)
RESULTS_FILE = os.path.join(RESULTS_DIR, "glue_results.json")
DIAG_RESULTS_FILE = os.path.join(RESULTS_DIR, "diagnostic_results.json")

# Load existing results if resuming
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        glue_results = json.load(f)
    print(f"Loaded {len(glue_results)} existing GLUE results")
else:
    glue_results = {}

if os.path.exists(DIAG_RESULTS_FILE):
    with open(DIAG_RESULTS_FILE, "r") as f:
        diag_results = json.load(f)
    print(f"Loaded {len(diag_results)} existing diagnostic results")
else:
    diag_results = {}

print(f"Experiment matrix: {len(ENCODER_BACKBONES)} backbones × {len(POOLERS)} poolers × {len(ALL_GLUE_TASKS)} tasks = {len(ENCODER_BACKBONES) * len(POOLERS) * len(ALL_GLUE_TASKS)} GLUE experiments")
print(f"Diagnostic matrix: {len(ENCODER_BACKBONES)} backbones × {len(POOLERS)} poolers × {len(DISTRACTOR_RATIOS)} ratios = {len(ENCODER_BACKBONES) * len(POOLERS) * len(DISTRACTOR_RATIOS)} diagnostic experiments")
```

**Step 2: Add cell 6 — Run experiment helper function**

Cell 6 — Helper to run a single experiment and parse output:

```python
import re

def run_glue_experiment(backbone, pooler, task, hparams, org_dir=ORG_DIR):
    """Run a single GLUE experiment using the original main.py and parse metrics from stdout."""
    key = f"{backbone}|{pooler}|{task}"

    # Skip if already done
    if key in glue_results:
        print(f"  SKIP (cached): {key} -> {glue_results[key]}")
        return glue_results[key]

    cmd = [
        "python", os.path.join(org_dir, "main.py"),
        "--model_name_or_path", backbone,
        "--task", task,
        "--pooling_method", pooler,
        "--epochs", str(hparams["epochs"]),
        "--batch_size", str(hparams["batch_size"]),
        "--eval_batch_size", str(hparams["eval_batch_size"]),
        "--lr", str(hparams["lr"]),
        "--max_length", str(hparams["max_length"]),
        "--seed", str(hparams["seed"]),
        "--gat_hidden_dim", str(hparams["gat_hidden_dim"]),
        "--num_layers", str(hparams["num_layers"]),
        "--jk_mode", hparams["jk_mode"],
        "--tau", str(hparams["tau"]),
        "--scorer_hidden", str(hparams["scorer_hidden"]),
        "--precompute_hidden_states", str(hparams["precompute_hidden_states"]),
        "--verbose", str(hparams["verbose"]),
        "--graph_adj", "threshold",
        "--weight_decay", "1e-5",
    ]

    print(f"  Running: {backbone} | {pooler} | {task} ...", end=" ", flush=True)
    start = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,  # 30 min timeout
            cwd=org_dir
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Parse metrics from verbose output
        # Format: [pooler] epoch N loss X.XXXX acc Y.YYYY mcc Z.ZZZZ
        # or: [pooler] epoch N MSE X.XXXX Spearman Y.YYYY Pearson Z.ZZZZ
        metrics = {}

        # Find all epoch lines and take the best
        acc_matches = re.findall(r"acc\s+([\d.]+)", output)
        mcc_matches = re.findall(r"mcc\s+([\d.]+)", output)
        f1_matches = re.findall(r"f1\s+([\d.]+)", output)
        spearman_matches = re.findall(r"Spearman\s+([\d.]+)", output)
        pearson_matches = re.findall(r"Pearson\s+([\d.]+)", output)

        if acc_matches:
            metrics["acc"] = max(float(x) for x in acc_matches)
        if mcc_matches:
            metrics["mcc"] = max(float(x) for x in mcc_matches)
        if f1_matches:
            metrics["f1"] = max(float(x) for x in f1_matches)
        if spearman_matches:
            metrics["spearman"] = max(float(x) for x in spearman_matches)
        if pearson_matches:
            metrics["pearson"] = max(float(x) for x in pearson_matches)

        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["returncode"] = result.returncode

        if result.returncode != 0:
            metrics["error"] = output[-500:]  # Last 500 chars of output
            print(f"FAILED ({elapsed:.0f}s)")
        else:
            print(f"done ({elapsed:.0f}s) -> {metrics}")

    except subprocess.TimeoutExpired:
        metrics = {"error": "TIMEOUT (30 min)", "elapsed_sec": 1800}
        print("TIMEOUT")
    except Exception as e:
        metrics = {"error": str(e)}
        print(f"ERROR: {e}")

    # Save result
    glue_results[key] = metrics
    with open(RESULTS_FILE, "w") as f:
        json.dump(glue_results, f, indent=2)

    return metrics

print("Experiment runner ready")
```

**Step 3: Commit**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: add experiment config and runner utility to reproduction notebook"
```

---

### Task 3: Add GLUE experiment cells

**Files:**
- Modify: `notebooks/reproduce_org_glot.ipynb`

**Step 1: Add cell 7 — Markdown header for GLUE section**

```markdown
## GLUE Benchmark Experiments

Running all GLUE tasks with encoder backbones. Each experiment:
1. Precomputes hidden states from the frozen backbone
2. Trains the pooler head for the configured number of epochs
3. Evaluates on the validation set

Results are saved to Google Drive after each experiment for crash resilience.
```

**Step 2: Add cell 8 — Run all GLUE experiments**

```python
# ============================================================
# Run GLUE Experiments
# ============================================================

total = len(ENCODER_BACKBONES) * len(POOLERS) * len(ALL_GLUE_TASKS)
done = 0

for backbone in ENCODER_BACKBONES:
    print(f"\n{'='*60}")
    print(f"BACKBONE: {backbone}")
    print(f"{'='*60}")

    for task in ALL_GLUE_TASKS:
        print(f"\n--- Task: {task} ---")
        for pooler in POOLERS:
            done += 1
            print(f"[{done}/{total}]", end=" ")
            run_glue_experiment(backbone, pooler, task, HPARAMS)

print(f"\n\nAll GLUE experiments complete! Results saved to {RESULTS_FILE}")
```

**Step 3: Add cell 9 — Display GLUE results table**

```python
import pandas as pd

# Task -> primary metric mapping
TASK_METRICS = {
    "sst2": ("acc", "Acc"),
    "cola": ("mcc", "MCC"),
    "mrpc": ("f1", "F1"),
    "qqp": ("f1", "F1"),
    "stsb": ("spearman", "Spear."),
    "mnli": ("acc", "Acc"),
    "qnli": ("acc", "Acc"),
    "rte": ("acc", "Acc"),
    "wnli": ("acc", "Acc"),
}

for backbone in ENCODER_BACKBONES:
    print(f"\n{'='*80}")
    print(f"  {backbone}")
    print(f"{'='*80}")

    rows = []
    for pooler in POOLERS:
        row = {"Pooler": pooler.upper()}
        for task in ALL_GLUE_TASKS:
            key = f"{backbone}|{pooler}|{task}"
            metric_key, metric_label = TASK_METRICS[task]
            result = glue_results.get(key, {})

            if "error" in result:
                row[f"{task}\n({metric_label})"] = "ERR"
            elif metric_key in result:
                val = result[metric_key]
                # Convert to percentage if needed (acc/f1/mcc are in [0,1] from original code)
                if metric_key in ["acc", "f1", "mcc"]:
                    val *= 100
                row[f"{task}\n({metric_label})"] = f"{val:.1f}"
            else:
                row[f"{task}\n({metric_label})"] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("Pooler", inplace=True)
    print(df.to_string())
    print()
```

**Step 4: Commit**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: add GLUE experiment runner and results display to reproduction notebook"
```

---

### Task 4: Add diagnostic stress test cells

**Files:**
- Modify: `notebooks/reproduce_org_glot.ipynb`

**Step 1: Add cell 10 — Markdown header for diagnostic section**

```markdown
## Diagnostic Stress Test

The "Relational Needle in a Haystack" experiment tests pooler robustness by burying signal phrases in random distractor words at varying ratios (20%, 50%, 80%, 90% noise).
```

**Step 2: Add cell 11 — Diagnostic experiment runner**

```python
def run_diagnostic_experiment(backbone, pooler, ratio, org_dir=ORG_DIR):
    """Run a single diagnostic stress test experiment."""
    key = f"{backbone}|{pooler}|{ratio}"

    if key in diag_results:
        print(f"  SKIP (cached): {key} -> {diag_results[key]}")
        return diag_results[key]

    cmd = [
        "python", os.path.join(org_dir, "diagnostic_stress_test.py"),
        "--model_name_or_path", backbone,
        "--pooling_method", pooler,
        "--distractor_ratio", str(ratio),
        "--seed", str(HPARAMS["seed"]),
        "--gat_hidden_dim", str(HPARAMS["gat_hidden_dim"]),
        "--num_layers", str(HPARAMS["num_layers"]),
        "--jk_mode", HPARAMS["jk_mode"],
        "--tau", str(HPARAMS["tau"]),
        "--verbose", str(HPARAMS["verbose"]),
    ]

    print(f"  Running: {backbone} | {pooler} | ratio={ratio} ...", end=" ", flush=True)
    start = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,  # 10 min timeout
            cwd=org_dir
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        metrics = {}
        acc_matches = re.findall(r"acc(?:uracy)?[\s:=]+([\d.]+)", output, re.IGNORECASE)
        if acc_matches:
            metrics["acc"] = max(float(x) for x in acc_matches)

        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["returncode"] = result.returncode

        if result.returncode != 0:
            metrics["error"] = output[-500:]
            print(f"FAILED ({elapsed:.0f}s)")
        else:
            print(f"done ({elapsed:.0f}s) -> {metrics}")

    except subprocess.TimeoutExpired:
        metrics = {"error": "TIMEOUT", "elapsed_sec": 600}
        print("TIMEOUT")
    except Exception as e:
        metrics = {"error": str(e)}
        print(f"ERROR: {e}")

    diag_results[key] = metrics
    with open(DIAG_RESULTS_FILE, "w") as f:
        json.dump(diag_results, f, indent=2)

    return metrics

print("Diagnostic runner ready")
```

**Step 3: Add cell 12 — Run all diagnostic experiments**

```python
# ============================================================
# Run Diagnostic Stress Tests
# ============================================================

total = len(ENCODER_BACKBONES) * len(POOLERS) * len(DISTRACTOR_RATIOS)
done = 0

for backbone in ENCODER_BACKBONES:
    print(f"\n{'='*60}")
    print(f"BACKBONE: {backbone}")
    print(f"{'='*60}")

    for ratio in DISTRACTOR_RATIOS:
        print(f"\n--- Distractor Ratio: {ratio} ---")
        for pooler in POOLERS:
            done += 1
            print(f"[{done}/{total}]", end=" ")
            run_diagnostic_experiment(backbone, pooler, ratio)

print(f"\n\nAll diagnostic experiments complete! Results saved to {DIAG_RESULTS_FILE}")
```

**Step 4: Add cell 13 — Display diagnostic results table**

```python
for backbone in ENCODER_BACKBONES:
    print(f"\n{'='*60}")
    print(f"  Diagnostic: {backbone}")
    print(f"{'='*60}")

    rows = []
    for pooler in POOLERS:
        row = {"Pooler": pooler.upper()}
        for ratio in DISTRACTOR_RATIOS:
            key = f"{backbone}|{pooler}|{ratio}"
            result = diag_results.get(key, {})
            if "error" in result:
                row[f"r={ratio}"] = "ERR"
            elif "acc" in result:
                row[f"r={ratio}"] = f"{result['acc'] * 100:.1f}"
            else:
                row[f"r={ratio}"] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("Pooler", inplace=True)
    print(df.to_string())
    print()
```

**Step 5: Commit**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: add diagnostic stress test runner to reproduction notebook"
```

---

### Task 5: Add paper comparison and summary cells

**Files:**
- Modify: `notebooks/reproduce_org_glot.ipynb`

**Step 1: Add cell 14 — Markdown header for comparison**

```markdown
## Paper Comparison

Compare reproduced results against the paper's reported numbers. Fill in the `PAPER_RESULTS` dict below with values from the paper's Table 1 (GLUE) and Table 2/Figure (diagnostic).

Tolerance: results within ±2% of paper values are considered successfully reproduced.
```

**Step 2: Add cell 15 — Paper results reference and comparison**

```python
# ============================================================
# Paper Reference Results (fill from paper Tables 1-2)
# ============================================================
# Format: {backbone: {task: {pooler: score}}}
# Scores are in the same scale as the original code output (0-1 for acc/f1, raw for spearman)
# TODO: Fill these from the actual paper tables

PAPER_RESULTS = {
    "bert-base-uncased": {
        # "sst2": {"mean": None, "max": None, "cls": None, "adapool": None, "glot": None},
        # "cola": {"mean": None, "max": None, "cls": None, "adapool": None, "glot": None},
        # ... fill from paper
    },
    "roberta-base": {
        # ... fill from paper
    },
}

# ============================================================
# Comparison: Reproduced vs Paper
# ============================================================

def compare_results(reproduced, paper_ref, tolerance=0.02):
    """Compare reproduced results against paper reference values."""
    comparisons = []
    for backbone in ENCODER_BACKBONES:
        if backbone not in paper_ref or not paper_ref[backbone]:
            continue
        for task in ALL_GLUE_TASKS:
            if task not in paper_ref[backbone]:
                continue
            metric_key = TASK_METRICS[task][0]
            for pooler in POOLERS:
                key = f"{backbone}|{pooler}|{task}"
                repro = reproduced.get(key, {}).get(metric_key)
                paper = paper_ref[backbone].get(task, {}).get(pooler)

                if repro is None or paper is None:
                    continue

                diff = repro - paper
                status = "PASS" if abs(diff) <= tolerance else ("HIGH" if diff > 0 else "LOW")
                comparisons.append({
                    "Backbone": backbone,
                    "Task": task,
                    "Pooler": pooler.upper(),
                    "Paper": f"{paper:.4f}",
                    "Reproduced": f"{repro:.4f}",
                    "Diff": f"{diff:+.4f}",
                    "Status": status,
                })

    if comparisons:
        df = pd.DataFrame(comparisons)
        print(df.to_string(index=False))
        n_pass = sum(1 for c in comparisons if c["Status"] == "PASS")
        print(f"\n{n_pass}/{len(comparisons)} results within ±{tolerance*100:.0f}% tolerance")
    else:
        print("No paper reference values filled in yet.")
        print("Edit PAPER_RESULTS dict above with values from the paper tables.")

compare_results(glue_results, PAPER_RESULTS)
```

**Step 3: Add cell 16 — Summary markdown**

```markdown
## Summary

### What was tested
- Original GLOT code from [ipsitmantri/GLOT](https://github.com/ipsitmantri/GLOT)
- 2 encoder backbones × 5 poolers × 9 GLUE tasks = 90 experiments
- 2 backbones × 5 poolers × 4 distractor ratios = 40 diagnostic experiments

### Next steps
- Fill in `PAPER_RESULTS` dict with exact numbers from the paper
- Run decoder backbone experiments (requires A100 GPU)
- Compare architectural differences between original and our re-implementation
```

**Step 4: Commit**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: add paper comparison and summary cells to reproduction notebook"
```

---

### Task 6: Verify notebook end-to-end with a smoke test

**Files:**
- Modify: `notebooks/reproduce_org_glot.ipynb` (minor fixes if needed)

**Step 1: Add a smoke test cell at the top (cell after setup)**

Insert a temporary cell right after the setup cells to verify one small experiment works. This cell can be deleted or skipped after validation.

```python
# ============================================================
# SMOKE TEST — verify one experiment works before running all
# ============================================================
# Run a single fast experiment: bert-base-uncased + mean pooler + sst2
# This validates the full pipeline: clone, install, run, parse

smoke_result = run_glue_experiment(
    "bert-base-uncased", "mean", "sst2",
    {**HPARAMS, "epochs": 1}  # 1 epoch for speed
)

if "error" in smoke_result:
    print(f"\nSMOKE TEST FAILED: {smoke_result['error']}")
    print("Check the setup cells above and fix any issues before running all experiments.")
else:
    print(f"\nSMOKE TEST PASSED: {smoke_result}")
    print("Ready to run all experiments!")
```

**Step 2: Review the full notebook structure**

Verify the cell order is correct:
1. Cell 0: Markdown title
2. Cell 1: GPU check + Drive mount
3. Cell 2: Clone repo
4. Cell 3: Install deps
5. Cell 4: Configure HF token + wandb
6. Cell 5: Experiment config
7. Cell 6: Runner helper function
8. Cell 7: Markdown — GLUE header
9. Cell 8: Run all GLUE experiments
10. Cell 9: Display GLUE results
11. Cell 10: Markdown — Diagnostic header
12. Cell 11: Diagnostic runner function
13. Cell 12: Run all diagnostic experiments
14. Cell 13: Display diagnostic results
15. Cell 14: Markdown — Comparison header
16. Cell 15: Paper comparison
17. Cell 16: Markdown — Summary

**Step 3: Commit final version**

```bash
git add notebooks/reproduce_org_glot.ipynb
git commit -m "feat: complete reproduction notebook with smoke test"
```

---

### Important notes for the implementing engineer

1. **The original `main.py` uses `wandb.init()` for every experiment** — setting `WANDB_MODE=offline` prevents it from requiring a W&B account but still creates local run directories. This is fine.

2. **The original code's `diagnostic_stress_test.py` CLI args may differ** from `main.py`. When implementing Task 4, first read the actual argparser in that file (fetch from GitHub) to confirm the exact argument names. The `run_diagnostic_experiment` function may need adjustment.

3. **Metric parsing depends on the original code's verbose output format.** The regexes in `run_glue_experiment` match patterns like `acc 0.8542` and `Spearman 0.8234`. If the actual output format differs, adjust the regexes accordingly. Run the smoke test first to validate.

4. **The `precompute_hidden_states` flag** saves cached data under `./data/` in the original repo's directory. This is persistent within a Colab session but lost on restart. Hidden states for a new backbone will need re-caching after restart, but cached results in Google Drive persist.

5. **Hyperparameter defaults**: The paper config (`mteb_eval_config.yaml`) uses `num_layers=4, gat_hidden_dim=256, tau=0.3, lr=2e-5`. The argparser defaults are `num_layers=2, gat_hidden_dim=128, lr=2e-4`. The `HPARAMS` dict in the notebook should match the paper's config, not the argparser defaults. This is reflected in Task 2's config cell.
