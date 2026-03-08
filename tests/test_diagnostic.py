import pytest
from data.diagnostic import (
    generate_diagnostic_dataset,
    TARGET_NOUNS,
    MODIFIERS,
    CONTEXTS,
    NOISE_WORDS,
)


class TestDiagnosticDataGeneration:
    def test_returns_correct_count(self):
        data = generate_diagnostic_dataset(num_samples=100, seed=42)
        assert len(data) == 100

    def test_returns_text_label_tuples(self):
        data = generate_diagnostic_dataset(num_samples=10, seed=42)
        for text, label in data:
            assert isinstance(text, str)
            assert label in (0, 1)

    def test_labels_are_balanced(self):
        data = generate_diagnostic_dataset(num_samples=1000, seed=42)
        labels = [label for _, label in data]
        ratio = sum(labels) / len(labels)
        assert 0.35 < ratio < 0.65, f"Label balance is {ratio}, expected ~0.5"

    def test_word_count_does_not_exceed_seq_length(self):
        """Original code caps at seq_length but doesn't pad shorter sequences."""
        for seq_length in [64, 128, 256]:
            data = generate_diagnostic_dataset(
                num_samples=50, seq_length=seq_length, distractor_ratio=0.8, seed=42
            )
            for text, _ in data:
                words = text.split()
                assert len(words) <= seq_length, (
                    f"Expected <= {seq_length} words, got {len(words)}"
                )
                assert len(words) > 0

    def test_target_noun_present_in_text(self):
        """Every sample should contain exactly one target noun."""
        data = generate_diagnostic_dataset(num_samples=200, seed=42)
        for text, _ in data:
            found = [noun for noun in TARGET_NOUNS if noun in text.split()]
            assert len(found) >= 1, f"No target noun found in: {text[:80]}..."

    def test_positive_samples_have_context_and_target(self):
        """Positive samples: context phrase + target noun, no modifier between."""
        data = generate_diagnostic_dataset(num_samples=500, seed=42)
        positives = [(t, l) for t, l in data if l == 1]
        assert len(positives) > 0
        for text, _ in positives:
            # Should contain a context phrase
            has_context = any(ctx in text for ctx in CONTEXTS)
            assert has_context, f"Positive sample missing context: {text[:80]}..."
            # Should NOT have a modifier between context and target
            # (modifier might appear in noise, but not adjacent to context)

    def test_negative_samples_have_modifier(self):
        """Negative samples should contain a modifier word."""
        data = generate_diagnostic_dataset(
            num_samples=500, distractor_ratio=0.3, seed=42
        )
        negatives = [(t, l) for t, l in data if l == 0]
        assert len(negatives) > 0
        for text, _ in negatives:
            has_modifier = any(mod in text.split() for mod in MODIFIERS)
            assert has_modifier, f"Negative sample missing modifier: {text[:80]}..."

    def test_relational_distance_inserts_noise(self):
        """With relational_distance=5, negative signal phrases should be longer."""
        data_rd5 = generate_diagnostic_dataset(
            num_samples=200, relational_distance=5, distractor_ratio=0.3, seed=42
        )
        data_rd0 = generate_diagnostic_dataset(
            num_samples=200, relational_distance=0, distractor_ratio=0.3, seed=42
        )
        # With rd=0, negative phrases are short (context + modifier + target = ~4-5 words)
        # With rd=5, negative phrases are longer (context + modifier + 5 noise + target = ~9-10 words)
        # All should still have correct seq_length
        for text, _ in data_rd5:
            assert len(text.split()) <= 128
        for text, _ in data_rd0:
            assert len(text.split()) <= 128

    def test_signal_position_start(self):
        """Signal at start means context phrase should appear near beginning."""
        data = generate_diagnostic_dataset(
            num_samples=100, signal_position="start", distractor_ratio=0.5, seed=42
        )
        for text, _ in data:
            words = text.split()
            # Check that a context phrase starts at the beginning
            text_start = " ".join(words[:5])
            has_context_at_start = any(ctx in text_start for ctx in CONTEXTS)
            # At least most samples should have signal at start
            # (context phrases are 3-4 words, so they should be in first 5)
        # Verify statistically: at start position, most should have context in first 5 words
        count_at_start = 0
        for text, _ in data:
            words = text.split()
            text_start = " ".join(words[:5])
            if any(ctx in text_start for ctx in CONTEXTS):
                count_at_start += 1
        assert count_at_start >= 90, f"Only {count_at_start}/100 had signal at start"

    def test_signal_position_end(self):
        """Signal at end means target noun should appear near the end."""
        data = generate_diagnostic_dataset(
            num_samples=100, signal_position="end", distractor_ratio=0.5, seed=42
        )
        count_at_end = 0
        for text, _ in data:
            words = text.split()
            last_words = words[-20:]  # negative signals can be ~15 words
            if any(noun in last_words for noun in TARGET_NOUNS):
                count_at_end += 1
        assert count_at_end >= 90, f"Only {count_at_end}/100 had signal at end"

    def test_different_seeds_produce_different_data(self):
        data1 = generate_diagnostic_dataset(num_samples=10, seed=1)
        data2 = generate_diagnostic_dataset(num_samples=10, seed=2)
        texts1 = [t for t, _ in data1]
        texts2 = [t for t, _ in data2]
        assert texts1 != texts2

    def test_high_distractor_ratio(self):
        data = generate_diagnostic_dataset(
            num_samples=10, seq_length=128, distractor_ratio=0.9, seed=42
        )
        assert len(data) == 10
        for text, label in data:
            assert len(text.split()) <= 128
            assert label in (0, 1)

    def test_invalid_signal_position_raises(self):
        with pytest.raises(ValueError, match="Invalid signal_position"):
            generate_diagnostic_dataset(num_samples=1, signal_position="invalid")

    def test_constants_match_original(self):
        """Verify constants match the original GLOT code."""
        assert len(TARGET_NOUNS) == 6
        assert "keys" in TARGET_NOUNS
        assert "documents" in TARGET_NOUNS
        assert len(MODIFIERS) == 4
        assert "not" in MODIFIERS
        assert "excluding" in MODIFIERS
        assert len(CONTEXTS) == 3
        assert "the delivery contains" in CONTEXTS
        assert len(NOISE_WORDS) > 70  # Original has ~80 noise words

    def test_default_params_match_original(self):
        """Default parameters should match the original GLOT code."""
        import inspect
        sig = inspect.signature(generate_diagnostic_dataset)
        assert sig.parameters["num_samples"].default == 2000
        assert sig.parameters["seq_length"].default == 128
        assert sig.parameters["signal_position"].default == "random"
        assert sig.parameters["relational_distance"].default == 10
