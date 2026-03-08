"""Diagnostic stress test: signal dilution experiment (Table 7 / Figure 3).

Usage:
    python run_diagnostic.py --backbone bert-base-uncased --pooler glot --ratio 0.9
    python run_diagnostic.py --all
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.diagnostic import generate_diagnostic_dataset
from glot.backbone import BACKBONE_REGISTRY, load_backbone
from glot.model import create_pooler_and_head
from glot.utils import compute_metrics

ALL_BACKBONES = list(BACKBONE_REGISTRY.keys())
ALL_POOLERS = ["cls", "eos", "mean", "max", "adapool", "glot"]
ALL_RATIOS = [0.2, 0.5, 0.8, 0.9]

TRAIN_SAMPLES = 2000
TEST_SAMPLES = 500
SEQ_LENGTH = 128
MAX_TOKEN_LENGTH = 512
SIGNAL_POSITION = "random"
RELATIONAL_DISTANCE = 10
EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 32
SEED = 42


def _select_pooler_type(pooler_name, backbone_type):
    """Map pooler name to the correct type for the backbone.

    For 'cls': use 'cls' for encoders, 'eos' for decoders.
    For 'eos': always 'eos'.
    Others pass through unchanged.
    """
    if pooler_name == "cls" and backbone_type == "decoder":
        return "eos"
    if pooler_name == "eos" and backbone_type == "encoder":
        return "cls"
    return pooler_name


def _get_dtype(backbone_name):
    """Use float16 for large decoder models to fit in GPU memory."""
    cfg = BACKBONE_REGISTRY[backbone_name]
    if cfg["params"] >= 3e9:
        return torch.float16
    return None


def tokenize_and_encode(texts, backbone, tokenizer, device, max_length=MAX_TOKEN_LENGTH):
    """Tokenize texts and run through frozen backbone to get hidden states."""
    all_hs = []
    all_masks = []

    batch_size = 16  # small batches for backbone inference
    backbone.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            out = backbone(**encoded)
            all_hs.append(out.last_hidden_state.cpu().float())
            all_masks.append(encoded["attention_mask"].cpu())

    return torch.cat(all_hs), torch.cat(all_masks)


def run_single_experiment(backbone_name, pooler_name, ratio, device, results_dict, use_wandb=False):
    """Run one (backbone, pooler, ratio) experiment and return accuracy."""
    cfg = BACKBONE_REGISTRY[backbone_name]
    pooler_type = _select_pooler_type(pooler_name, cfg["type"])

    print(f"\n{'='*60}")
    print(f"Backbone: {backbone_name} | Pooler: {pooler_name} | Ratio: {ratio}")
    print(f"{'='*60}")

    # Load backbone
    dtype = _get_dtype(backbone_name)
    backbone, tokenizer, bcfg = load_backbone(backbone_name, device=device, dtype=dtype)

    # Generate data
    train_data = generate_diagnostic_dataset(
        num_samples=TRAIN_SAMPLES, seq_length=SEQ_LENGTH,
        distractor_ratio=ratio, signal_position=SIGNAL_POSITION,
        relational_distance=RELATIONAL_DISTANCE, seed=SEED,
    )
    test_data = generate_diagnostic_dataset(
        num_samples=TEST_SAMPLES, seq_length=SEQ_LENGTH,
        distractor_ratio=ratio, signal_position=SIGNAL_POSITION,
        relational_distance=RELATIONAL_DISTANCE, seed=SEED + 1,
    )

    train_texts = [t for t, _ in train_data]
    train_labels = torch.tensor([l for _, l in train_data])
    test_texts = [t for t, _ in test_data]
    test_labels = torch.tensor([l for _, l in test_data])

    # Tokenize and encode
    print("Encoding train set...")
    train_hs, train_masks = tokenize_and_encode(train_texts, backbone, tokenizer, device)
    print("Encoding test set...")
    test_hs, test_masks = tokenize_and_encode(test_texts, backbone, tokenizer, device)

    # Free backbone memory
    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create pooler + head
    input_dim = bcfg["hidden_dim"]
    glot_config = {"hidden_dim": 128, "num_gnn_layers": 2, "jk_mode": "cat", "threshold": 0.3}
    pooler, head = create_pooler_and_head(
        pooler_type=pooler_type,
        input_dim=input_dim,
        num_classes=2,
        task_type="classification",
        glot_config=glot_config if pooler_type == "glot" else None,
    )
    pooler = pooler.to(device)
    head = head.to(device)

    # Data loaders
    train_ds = TensorDataset(train_hs, train_masks, train_labels)
    test_ds = TensorDataset(test_hs, test_masks, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # Train
    params = list(pooler.parameters()) + list(head.parameters())
    optimizer = Adam(params, lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(SEED)

    for epoch in range(EPOCHS):
        pooler.train()
        head.train()
        total_loss = 0.0
        n_batches = 0
        for hs, masks, labels in train_loader:
            hs, masks, labels = hs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            z = pooler(hs, masks)
            logits = head(z)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Evaluate
    pooler.eval()
    head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for hs, masks, labels in test_loader:
            hs, masks, labels = hs.to(device), masks.to(device), labels.to(device)
            z = pooler(hs, masks)
            logits = head(z)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = compute_metrics(all_preds, all_labels, "accuracy")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Store result
    key = f"{backbone_name}|{pooler_name}|{ratio}"
    results_dict[key] = {
        "backbone": backbone_name,
        "pooler": pooler_name,
        "ratio": ratio,
        "accuracy": accuracy,
    }

    # W&B logging
    if use_wandb:
        try:
            import wandb
            wandb.log({
                "backbone": backbone_name,
                "pooler": pooler_name,
                "distractor_ratio": ratio,
                "accuracy": accuracy,
            })
        except ImportError:
            pass

    # Cleanup
    del pooler, head, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Diagnostic stress test experiment")
    parser.add_argument("--backbone", default=None, help="Backbone model name")
    parser.add_argument("--pooler", default=None, help="Pooler type")
    parser.add_argument("--ratio", type=float, default=None, help="Distractor ratio")
    parser.add_argument("--all", action="store_true", help="Run all 120 combinations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/diagnostic_results.json")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Init W&B
    if args.wandb:
        try:
            import wandb
            wandb.init(project="glot-diagnostic", config=vars(args))
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
            args.wandb = False

    # Determine what to run
    if args.all:
        backbones = ALL_BACKBONES
        poolers = ["cls", "mean", "max", "adapool", "glot"]
        ratios = ALL_RATIOS
    else:
        if not args.backbone or not args.pooler or args.ratio is None:
            parser.error("Provide --backbone, --pooler, and --ratio, or use --all")
        backbones = [args.backbone]
        poolers = [args.pooler]
        ratios = [args.ratio]

    # Load existing results if any
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    # Run experiments
    for backbone_name in backbones:
        for ratio in ratios:
            for pooler_name in poolers:
                key = f"{backbone_name}|{pooler_name}|{ratio}"
                if key in results:
                    print(f"Skipping {key} (already computed)")
                    continue
                run_single_experiment(
                    backbone_name, pooler_name, ratio,
                    args.device, results, use_wandb=args.wandb,
                )
                # Save after each experiment for resume capability
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

    # Print summary table
    print("\n\n=== RESULTS SUMMARY ===\n")
    for ratio in ALL_RATIOS:
        print(f"\n--- {int(ratio*100)}% Distractors ---")
        print(f"{'Backbone':<45} {'CLS/EOS':>8} {'Mean':>8} {'Max':>8} {'AdaPool':>8} {'GLOT':>8}")
        for backbone_name in ALL_BACKBONES:
            row = []
            for p in ["cls", "mean", "max", "adapool", "glot"]:
                key = f"{backbone_name}|{p}|{ratio}"
                if key in results:
                    row.append(f"{results[key]['accuracy']:.1f}")
                else:
                    row.append("--")
            print(f"{backbone_name:<45} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
