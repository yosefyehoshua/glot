import pytest
from data.diagnostic import generate_diagnostic_dataset, SIGNAL_TEMPLATES, CONTENT_WORDS


class TestDiagnosticDataGeneration:
    def test_returns_correct_count(self):
        data = generate_diagnostic_dataset(num_samples=100, seq_length=50, distractor_ratio=0.5, seed=42)
        assert len(data) == 100

    def test_returns_text_label_tuples(self):
        data = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=42)
        for text, label in data:
            assert isinstance(text, str)
            assert label in (0, 1)

    def test_labels_are_balanced(self):
        data = generate_diagnostic_dataset(num_samples=1000, seq_length=50, distractor_ratio=0.5, seed=42)
        labels = [label for _, label in data]
        ratio = sum(labels) / len(labels)
        assert 0.35 < ratio < 0.65, f"Label balance is {ratio}, expected ~0.5"

    def test_word_count_matches_seq_length(self):
        seq_length = 100
        data = generate_diagnostic_dataset(num_samples=50, seq_length=seq_length, distractor_ratio=0.5, seed=42)
        for text, _ in data:
            words = text.split()
            assert len(words) == seq_length, f"Expected {seq_length} words, got {len(words)}"

    def test_signal_phrase_present(self):
        """At low distractor ratio, signal keywords should appear in text."""
        data = generate_diagnostic_dataset(num_samples=100, seq_length=50, distractor_ratio=0.2, seed=42)
        signal_keywords = {"but not", "but lacks", "without", "but excludes",
                          "and also", "and includes", "with"}
        found = 0
        for text, _ in data:
            if any(kw in text for kw in signal_keywords):
                found += 1
        # 8/10 templates contain distinctive keywords; 2 use plain "and"
        assert found >= 80, f"Signal phrase found in only {found}/100 samples"

    def test_different_seeds_produce_different_data(self):
        data1 = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=1)
        data2 = generate_diagnostic_dataset(num_samples=10, seq_length=50, distractor_ratio=0.5, seed=2)
        texts1 = [t for t, _ in data1]
        texts2 = [t for t, _ in data2]
        assert texts1 != texts2

    def test_high_distractor_ratio(self):
        data = generate_diagnostic_dataset(num_samples=10, seq_length=256, distractor_ratio=0.9, seed=42)
        assert len(data) == 10
        for text, label in data:
            assert len(text.split()) == 256
            assert label in (0, 1)

    def test_templates_exist(self):
        assert len(SIGNAL_TEMPLATES) == 10
        negation = [t for t, l in SIGNAL_TEMPLATES if l == 0]
        affirm = [t for t, l in SIGNAL_TEMPLATES if l == 1]
        assert len(negation) == 5
        assert len(affirm) == 5

    def test_content_words_exist(self):
        assert len(CONTENT_WORDS) >= 10
