from datasets import load_dataset
from glot.utils import GLUE_TASKS


def get_task_config(task_name: str) -> dict:
    """Return the GLUE task configuration dict."""
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown GLUE task: {task_name}. Choose from {list(GLUE_TASKS.keys())}")
    return GLUE_TASKS[task_name]


def load_glue_task(task_name: str, tokenizer, max_length: int = 128, seed: int = 42):
    """Load and tokenize a GLUE task.

    For pair tasks, tokenizes each sentence separately so that GLOT
    can build individual graphs per sentence.
    """
    config = get_task_config(task_name)
    dataset = load_dataset("glue", task_name)

    # Subsample large datasets
    if "subsample" in config:
        n = min(config["subsample"], len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(n))

    keys = config["sentence_keys"]

    if config["type"] == "single":
        def tokenize(examples):
            return tokenizer(
                examples[keys[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        dataset = dataset.map(tokenize, batched=True)
    else:
        # Pair tasks: tokenize each sentence separately
        def tokenize_pair(examples):
            tok_a = tokenizer(
                examples[keys[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tok_b = tokenizer(
                examples[keys[1]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            return {
                "input_ids_a": tok_a["input_ids"],
                "attention_mask_a": tok_a["attention_mask"],
                "input_ids_b": tok_b["input_ids"],
                "attention_mask_b": tok_b["attention_mask"],
            }

        dataset = dataset.map(tokenize_pair, batched=True)

    return dataset
