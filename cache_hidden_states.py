"""Precompute and cache frozen backbone hidden states for all GLUE tasks."""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from data.glue_loader import load_glue_task, get_task_config
from data.cache import save_cache


def precompute(backbone, dataloader, device, is_pair=False):
    """Run frozen backbone on all data and collect hidden states."""
    all_hs, all_masks, all_labels = [], [], []
    all_hs_b, all_masks_b = [], []

    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching"):
            if is_pair:
                ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)

                out_a = backbone(input_ids=ids_a, attention_mask=mask_a)
                out_b = backbone(input_ids=ids_b, attention_mask=mask_b)

                all_hs.append(out_a.last_hidden_state.cpu())
                all_masks.append(mask_a.cpu())
                all_hs_b.append(out_b.last_hidden_state.cpu())
                all_masks_b.append(mask_b.cpu())
            else:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out = backbone(input_ids=ids, attention_mask=mask)
                all_hs.append(out.last_hidden_state.cpu())
                all_masks.append(mask.cpu())

            all_labels.append(batch["label"])

    result = {
        "hs": torch.cat(all_hs),
        "masks": torch.cat(all_masks),
        "labels": torch.cat(all_labels),
    }
    if is_pair:
        result["hs_b"] = torch.cat(all_hs_b)
        result["masks_b"] = torch.cat(all_masks_b)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "cola", "mrpc"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", default="cached_hidden_states")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModel.from_pretrained(args.backbone).to(args.device)
    for p in backbone.parameters():
        p.requires_grad = False

    backbone_short = args.backbone.replace("/", "_")

    for task_name in args.tasks:
        print(f"\n=== Caching {task_name} ===")
        task_cfg = get_task_config(task_name)
        dataset = load_glue_task(task_name, tokenizer, args.max_length)
        is_pair = task_cfg["type"] == "pair"

        for split in ["train", "validation"]:
            if split not in dataset:
                continue

            ds = dataset[split]
            ds.set_format("torch")

            if is_pair:
                cols = ["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "label"]
            else:
                cols = ["input_ids", "attention_mask", "label"]
            ds = ds.select_columns(cols)

            loader = DataLoader(ds, batch_size=args.batch_size)
            result = precompute(backbone, loader, args.device, is_pair)

            cache_path = f"{args.cache_dir}/{backbone_short}/{task_name}/{split}.pt"
            if is_pair:
                save_cache(
                    cache_path, result["hs"], result["masks"], result["labels"],
                    result["hs_b"], result["masks_b"],
                )
            else:
                save_cache(cache_path, result["hs"], result["masks"], result["labels"])

            print(f"  Saved {split}: {result['hs'].shape} -> {cache_path}")


if __name__ == "__main__":
    main()
