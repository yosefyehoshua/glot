"""Evaluate a trained pooler on cached validation data."""
import argparse
import torch
from torch.utils.data import DataLoader

from glot.model import create_pooler_and_head
from glot.utils import compute_metrics, load_config, GLUE_TASKS
from data.cache import make_cached_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", required=True)
    parser.add_argument("--pooler", default="glot")
    parser.add_argument("--checkpoint", required=True, help="Path to saved model checkpoint")
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = GLUE_TASKS[args.task]
    if task_cfg.get("num_classes") == 1:
        task_type = "regression"
    elif task_cfg["type"] == "pair":
        task_type = "pair_classification"
    else:
        task_type = "classification"
    backbone_short = args.backbone.replace("/", "_")

    hidden_dim_map = {"bert-base-uncased": 768, "roberta-base": 768}
    input_dim = hidden_dim_map.get(args.backbone, 768)

    glot_config = {
        "hidden_dim": cfg["glot"]["hidden_dim"],
        "num_gnn_layers": cfg["glot"]["num_layers"],
        "jk_mode": cfg["glot"]["jk_mode"],
        "threshold": cfg["glot"]["threshold"],
        "gnn_type": cfg["glot"].get("gnn_type", "GAT"),
    }
    pooler, head = create_pooler_and_head(
        pooler_type=args.pooler,
        input_dim=input_dim,
        num_classes=task_cfg["num_classes"],
        task_type=task_type,
        glot_config=glot_config if args.pooler == "glot" else None,
    )

    checkpoint = torch.load(args.checkpoint, weights_only=True)
    pooler.load_state_dict(checkpoint["pooler"])
    head.load_state_dict(checkpoint["head"])

    pooler = pooler.to(args.device).eval()
    head = head.to(args.device).eval()

    val_ds = make_cached_dataset(f"{args.cache_dir}/{backbone_short}/{args.task}/validation.pt")
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["eval_batch_size"])

    from train import evaluate_epoch
    preds, labels = evaluate_epoch(pooler, head, val_loader, task_type, args.device)
    score = compute_metrics(preds, labels, task_cfg["metric"])
    print(f"{args.task} ({args.pooler}): {task_cfg['metric']} = {score:.2f}")


if __name__ == "__main__":
    main()
