"""Training script for GLOT on cached hidden states."""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from data.cache import make_cached_dataset


def train_epoch(pooler, head, loader, optimizer, loss_fn, task_type, device="cpu"):
    """Train for one epoch. Returns average loss."""
    pooler.train()
    head.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        if task_type == "pair_classification":
            hs_a, mask_a, hs_b, mask_b, labels = [b.to(device) for b in batch]
            z_a = pooler(hs_a, mask_a)
            z_b = pooler(hs_b, mask_b)
            logits = head(torch.cat([z_a, z_b], dim=-1))
        else:
            hs, mask, labels = [b.to(device) for b in batch]
            z = pooler(hs, mask)
            logits = head(z)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(pooler, head, loader, task_type, device="cpu"):
    """Evaluate and return (predictions, labels) lists."""
    pooler.eval()
    head.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        if task_type == "pair_classification":
            hs_a, mask_a, hs_b, mask_b, labels = [b.to(device) for b in batch]
            z_a = pooler(hs_a, mask_a)
            z_b = pooler(hs_b, mask_b)
            logits = head(torch.cat([z_a, z_b], dim=-1))
        else:
            hs, mask, labels = [b.to(device) for b in batch]
            z = pooler(hs, mask)
            logits = head(z)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default=None, help="Override task name")
    parser.add_argument("--pooler", default=None, help="Override pooler type")
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--backbone", default=None, help="Override backbone name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = args.task or cfg["task"]["name"]
    pooler_type = args.pooler or "glot"
    backbone_name = args.backbone or cfg["backbone"]["name"]
    backbone_short = backbone_name.replace("/", "_")

    task_cfg = GLUE_TASKS[task_name]
    task_type = "pair_classification" if task_cfg["type"] == "pair" else "classification"
    hidden_dim_map = {"bert-base-uncased": 768, "roberta-base": 768}
    input_dim = hidden_dim_map.get(backbone_name, 768)

    # Load cached data
    train_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{task_name}/train.pt")
    val_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{task_name}/validation.pt")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["eval_batch_size"],
    )

    # Create model
    glot_config = {
        "hidden_dim": cfg["glot"]["hidden_dim"],
        "num_gnn_layers": cfg["glot"]["num_layers"],
        "jk_mode": cfg["glot"]["jk_mode"],
        "threshold": cfg["glot"]["threshold"],
    }
    pooler, head = create_pooler_and_head(
        pooler_type=pooler_type,
        input_dim=input_dim,
        num_classes=task_cfg["num_classes"],
        task_type=task_type,
        glot_config=glot_config if pooler_type == "glot" else None,
    )
    pooler = pooler.to(args.device)
    head = head.to(args.device)

    params = list(pooler.parameters()) + list(head.parameters())
    optimizer = Adam(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    # Seed
    torch.manual_seed(cfg["training"]["seed"])

    # Train
    for epoch in range(cfg["training"]["epochs"]):
        avg_loss = train_epoch(pooler, head, train_loader, optimizer, loss_fn, task_type, args.device)
        preds, labels = evaluate_epoch(pooler, head, val_loader, task_type, args.device)
        score = compute_metrics(preds, labels, task_cfg["metric"])
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  {task_cfg['metric']}={score:.2f}")

    print(f"\nFinal {task_cfg['metric']}: {score:.2f}")


if __name__ == "__main__":
    main()
