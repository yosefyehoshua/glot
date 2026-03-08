"""Synthetic signal dilution dataset for diagnostic stress test (Algorithm 2).

Faithful port of generate_dataset() from the original GLOT code:
https://github.com/ipsitmantri/GLOT/blob/main/diagnostic_stress_test.py

Task: "Are {target_noun} present?" — binary classification.
Positive: context phrase + target noun (label 1).
Negative: context phrase + modifier + relational_distance noise words + target noun (label 0).
The relational distance forces long-range dependency understanding.
"""
import random
from typing import List, Tuple

# Exact constants from the original GLOT code
NOISE_WORDS = (
    "the of and to a in for is on that by this with i you it not or be are "
    "from at as your all have new more an was we will home can us about if "
    "page my has search free but our one other do no information time they "
    "site he up may what which their news out use any there see only so his "
    "when contact here business who web also now help get pm view online "
    "first am been would how were me services some these click its like "
    "service x than find date top yet"
).split()

TARGET_NOUNS = ["keys", "reports", "files", "tickets", "documents", "alerts"]

MODIFIERS = ["not", "never", "without", "excluding"]

CONTEXTS = ["the delivery contains", "the folder includes", "in the box are"]


def generate_diagnostic_dataset(
    num_samples: int = 2000,
    seq_length: int = 128,
    distractor_ratio: float = 0.5,
    signal_position: str = "random",
    relational_distance: int = 10,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """Generate synthetic signal dilution dataset (Algorithm 2).

    Constructs a "Relational Needle in a Haystack" task where the model must
    determine whether a target noun is present (label 1) or negated (label 0).
    For negated samples, ``relational_distance`` noise words are inserted
    between the modifier ("not", "never", ...) and the target noun, forcing
    long-range dependency understanding.

    Args:
        num_samples: Number of (text, label) pairs to generate.
        seq_length: Total word count per sequence.
        distractor_ratio: Fraction of sequence that is noise words (0.0–1.0).
        signal_position: Where to inject the signal phrase in the noise.
            One of "start", "middle", "end", "random".
        relational_distance: Number of noise words inserted between the
            modifier and the target noun in negative samples.
        seed: Random seed for reproducibility.

    Returns:
        List of (text, label) tuples. label is 0 (negated) or 1 (present).
    """
    rng = random.Random(seed)

    dataset: List[Tuple[str, int]] = []

    for _ in range(num_samples):
        # 1. Choose signal and label
        is_positive = rng.choice([True, False])
        target_noun = rng.choice(TARGET_NOUNS)
        context = rng.choice(CONTEXTS)

        # 2. Construct the signal phrase (the "needle")
        if is_positive:
            signal_phrase_words = context.split() + [target_noun]
            label = 1
        else:
            modifier = rng.choice(MODIFIERS)
            distractor_words_between = rng.choices(NOISE_WORDS, k=relational_distance)
            signal_phrase_words = (
                context.split() + [modifier] + distractor_words_between + [target_noun]
            )
            label = 0

        # 3. Construct the noise (the "haystack")
        num_signal_words = len(signal_phrase_words)
        num_noise_words = int(seq_length * distractor_ratio)
        num_total_words = num_noise_words + num_signal_words
        if num_total_words > seq_length:
            num_noise_words -= num_total_words - seq_length

        noise = rng.choices(NOISE_WORDS, k=num_noise_words)

        # 4. Inject needle into haystack
        if signal_position == "start":
            signal_start_idx = 0
        elif signal_position == "middle":
            signal_start_idx = len(noise) // 2
        elif signal_position == "end":
            signal_start_idx = len(noise)
        elif signal_position == "random":
            signal_start_idx = rng.randint(0, len(noise))
        else:
            raise ValueError(
                f"Invalid signal_position '{signal_position}', "
                "expected one of: start, middle, end, random"
            )

        final_words = noise[:signal_start_idx] + signal_phrase_words + noise[signal_start_idx:]
        final_words = final_words[:seq_length]  # Ensure exact length

        dataset.append((" ".join(final_words), label))

    return dataset
