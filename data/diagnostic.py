"""Synthetic signal dilution dataset for diagnostic stress test (Algorithm 2)."""
import random

SIGNAL_TEMPLATES = [
    # Negation patterns (label 0)
    ("the file has {X} but not {Y}", 0),
    ("the system includes {X} but lacks {Y}", 0),
    ("the report mentions {X} without {Y}", 0),
    ("the package contains {X} but excludes {Y}", 0),
    ("the plan covers {X} but not {Y}", 0),
    # Affirmation patterns (label 1)
    ("the file has {X} and also {Y}", 1),
    ("the system includes {X} and {Y}", 1),
    ("the report mentions {X} with {Y}", 1),
    ("the package contains {X} and includes {Y}", 1),
    ("the plan covers {X} and {Y}", 1),
]

CONTENT_WORDS = [
    "keys", "data", "images", "tables", "links",
    "charts", "graphs", "notes", "files", "records",
    "entries", "values", "codes", "tags", "labels",
]

# Distractor vocabulary — common English words (derived from Wikipedia per paper).
# Using a built-in list avoids the NLTK dependency for reproducibility.
_DISTRACTOR_VOCAB = [
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "was", "on", "are", "as", "with", "his", "they", "be", "at", "one",
    "have", "this", "from", "or", "had", "by", "not", "but", "what", "all",
    "were", "when", "we", "there", "can", "an", "your", "which", "their", "said",
    "each", "she", "do", "how", "if", "will", "up", "other", "about", "out",
    "many", "then", "them", "these", "so", "some", "her", "would", "make", "like",
    "him", "into", "time", "has", "look", "two", "more", "go", "see", "way",
    "could", "no", "than", "first", "been", "call", "who", "its", "now", "find",
    "long", "down", "day", "did", "get", "come", "made", "after", "back", "only",
    "me", "our", "under", "know", "last", "also", "use", "just", "over", "such",
    "great", "think", "say", "help", "low", "line", "before", "turn", "move", "right",
    "too", "old", "still", "same", "tell", "need", "house", "world", "head", "own",
    "every", "city", "tree", "cross", "farm", "hard", "start", "might", "story", "far",
    "sea", "late", "run", "left", "here", "school", "close", "night", "real", "life",
    "few", "north", "open", "seem", "next", "walk", "ease", "both", "mark", "mile",
    "river", "car", "feet", "care", "second", "group", "carry", "took", "rain", "eat",
    "room", "friend", "began", "idea", "fish", "mountain", "stop", "once", "base", "hear",
    "horse", "cut", "sure", "watch", "color", "face", "wood", "main", "enough", "plain",
    "girl", "usual", "young", "ready", "above", "ever", "red", "list", "though", "feel",
    "side", "keep", "land", "song", "door", "wind", "upon", "shall", "rock", "black",
    "short", "space", "while", "human", "during", "glass", "plant", "round", "change", "sun",
    "fire", "stand", "point", "page", "order", "place", "play", "end", "area", "water",
    "hand", "high", "small", "large", "given", "much", "may", "set", "part", "new",
    "number", "people", "state", "very", "take", "year", "most", "well", "those", "show",
    "form", "work", "must", "home", "even", "being", "where", "field", "good", "three",
    "kind", "name", "used", "men", "light", "road", "food", "book", "war", "between",
    "country", "never", "system", "best", "body", "paper", "power", "air", "done", "until",
    "white", "children", "put", "against", "should", "often", "important", "man", "big", "near",
    "why", "went", "family", "hands", "given", "along", "half", "nothing", "away", "surface",
    "political", "music", "possible", "woman", "feet", "fact", "class", "taken", "always", "words",
    "early", "eye", "true", "center", "less", "table", "rest", "already", "church", "five",
]


def generate_diagnostic_dataset(
    num_samples: int = 10000,
    seq_length: int = 256,
    distractor_ratio: float = 0.5,
    seed: int = 42,
):
    """Generate synthetic signal dilution dataset.

    Args:
        num_samples: Number of (text, label) pairs to generate.
        seq_length: Total word count per sequence.
        distractor_ratio: Fraction of words that are distractors (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        List of (text, label) tuples. label is 0 (negation) or 1 (affirmation).
    """
    rng = random.Random(seed)

    num_distractor_tokens = int(seq_length * distractor_ratio)
    num_signal_tokens = seq_length - num_distractor_tokens

    dataset = []
    for _ in range(num_samples):
        template, label = rng.choice(SIGNAL_TEMPLATES)
        x, y = rng.sample(CONTENT_WORDS, 2)
        signal_text = template.format(X=x, Y=y)
        signal_tokens = signal_text.split()

        # Adjust signal to fit allocated length
        if len(signal_tokens) > num_signal_tokens:
            signal_tokens = signal_tokens[:num_signal_tokens]
        elif len(signal_tokens) < num_signal_tokens:
            padding = rng.choices(_DISTRACTOR_VOCAB, k=num_signal_tokens - len(signal_tokens))
            signal_tokens = signal_tokens + padding

        # Generate distractor tokens and inject signal at random position
        distractor_tokens = rng.choices(_DISTRACTOR_VOCAB, k=num_distractor_tokens)
        inject_pos = rng.randint(0, num_distractor_tokens)
        sequence = distractor_tokens[:inject_pos] + signal_tokens + distractor_tokens[inject_pos:]

        dataset.append((" ".join(sequence), label))

    return dataset
