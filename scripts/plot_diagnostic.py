"""Plot diagnostic stress test results (Figure 3 / Table 7 from paper).

Usage:
    python scripts/plot_diagnostic.py --input results/diagnostic_results.json
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from glot.backbone import BACKBONE_REGISTRY

# Plotting style
POOLER_STYLES = {
    "cls": {"color": "#1f77b4", "marker": "s", "label": "CLS/EOS"},
    "mean": {"color": "#ff7f0e", "marker": "^", "label": "Mean"},
    "max": {"color": "#2ca02c", "marker": "v", "label": "Max"},
    "adapool": {"color": "#d62728", "marker": "D", "label": "AdaPool"},
    "glot": {"color": "#9467bd", "marker": "o", "label": "GLOT", "linewidth": 2.5},
}

RATIOS = [0.2, 0.5, 0.8, 0.9]
BACKBONES = list(BACKBONE_REGISTRY.keys())


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_figure3(results, output_prefix="results/diagnostic_figure"):
    """Generate 2x2 grid plot (Figure 3 from paper)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    model_sizes = [BACKBONE_REGISTRY[b]["params"] for b in BACKBONES]
    model_labels = [b.split("/")[-1] for b in BACKBONES]

    for ax, ratio in zip(axes.flat, RATIOS):
        for pooler_name, style in POOLER_STYLES.items():
            accs = []
            valid_sizes = []
            for backbone_name, size in zip(BACKBONES, model_sizes):
                key = f"{backbone_name}|{pooler_name}|{ratio}"
                if key in results:
                    accs.append(results[key]["accuracy"])
                    valid_sizes.append(size)

            if accs:
                ax.plot(
                    valid_sizes, accs,
                    color=style["color"],
                    marker=style["marker"],
                    label=style["label"],
                    linewidth=style.get("linewidth", 1.5),
                    markersize=7,
                )

        ax.set_title(f"{int(ratio * 100)}% Distractors", fontsize=13)
        ax.set_xlabel("Parameters", fontsize=11)
        ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(40, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Custom x-tick labels
        ax.set_xticks(model_sizes)
        ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)

    plt.suptitle("Diagnostic Stress Test: Signal Dilution", fontsize=15, fontweight="bold")
    plt.tight_layout()

    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    print(f"Saved: {output_prefix}.png and {output_prefix}.pdf")
    plt.close()


def print_table7(results):
    """Print Table 7 formatted results to stdout."""
    for ratio in RATIOS:
        print(f"\n### {int(ratio*100)}% Distractors\n")
        print(f"| {'Model':<30} | {'CLS/EOS':>8} | {'Mean':>8} | {'Max':>8} | {'AdaPool':>8} | {'GLOT':>8} |")
        print(f"|{'-'*32}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
        for backbone_name in BACKBONES:
            short = backbone_name.split("/")[-1]
            row = []
            for p in ["cls", "mean", "max", "adapool", "glot"]:
                key = f"{backbone_name}|{p}|{ratio}"
                if key in results:
                    val = results[key]["accuracy"]
                    row.append(f"{val:.1f}")
                else:
                    row.append("--")
            print(f"| {short:<30} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} |")


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic results")
    parser.add_argument("--input", default="results/diagnostic_results.json")
    parser.add_argument("--output", default="results/diagnostic_figure")
    args = parser.parse_args()

    results = load_results(args.input)
    print_table7(results)
    plot_figure3(results, output_prefix=args.output)


if __name__ == "__main__":
    main()
