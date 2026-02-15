import yaml
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import spearmanr


GLUE_TASKS = {
    "cola": {
        "type": "single",
        "num_classes": 2,
        "metric": "mcc",
        "sentence_keys": ("sentence",),
    },
    "sst2": {
        "type": "single",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence",),
    },
    "mrpc": {
        "type": "pair",
        "num_classes": 2,
        "metric": "f1",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "qqp": {
        "type": "pair",
        "num_classes": 2,
        "metric": "f1",
        "sentence_keys": ("question1", "question2"),
        "subsample": 20000,
    },
    "stsb": {
        "type": "pair",
        "num_classes": 1,
        "metric": "spearman",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "mnli": {
        "type": "pair",
        "num_classes": 3,
        "metric": "accuracy",
        "sentence_keys": ("premise", "hypothesis"),
        "subsample": 20000,
    },
    "qnli": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("question", "sentence"),
        "subsample": 20000,
    },
    "rte": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence1", "sentence2"),
    },
    "wnli": {
        "type": "pair",
        "num_classes": 2,
        "metric": "accuracy",
        "sentence_keys": ("sentence1", "sentence2"),
    },
}


def compute_metrics(predictions: list, labels: list, metric_name: str) -> float:
    """Compute task-specific metric. Returns score * 100."""
    if metric_name == "accuracy":
        return accuracy_score(labels, predictions) * 100
    elif metric_name == "mcc":
        return matthews_corrcoef(labels, predictions) * 100
    elif metric_name == "f1":
        return f1_score(labels, predictions) * 100
    elif metric_name == "spearman":
        return spearmanr(predictions, labels).correlation * 100
    raise ValueError(f"Unknown metric: {metric_name}")


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)
