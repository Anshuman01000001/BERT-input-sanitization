"""Tests for the text normalizer."""

from __future__ import annotations

from app.normalize import message_hash, normalize, similarity


class TestLeetspeak:
    def test_maps_leet_characters_inside_words(self):
        assert normalize("n00b f@gg0t sk1ll").normalized == "noob faggot skill"

    def test_leaves_pure_numbers_alone(self):
        assert normalize("1000").normalized == "1000"

    def test_maps_at_sign_to_a(self):
        assert normalize("@sshole").normalized == "asshole"


class TestHomoglyphs:
    def test_maps_cyrillic_lookalikes(self):
        assert normalize("Соlin fаggot").normalized == "colin faggot"

    def test_preserves_capitalisation_when_mapping(self):
        assert normalize("YOU АRE TRASH").folded == "YOU ARE TRASH"

    def test_maps_greek_lookalikes(self):
        assert normalize("αccident").normalized == "accident"


class TestUnicode:
    def test_folds_fullwidth_characters(self):
        assert normalize("Ｉ Ａ Ｍ Ａ Ｄ").normalized == "i a m a d"

    def test_strips_zero_width_and_bidi_characters(self):
        assert normalize("hello​‮world").normalized == "helloworld"

    def test_removes_control_characters_without_splitting_words(self):
        assert normalize("hel\x00lo").normalized == "hello"

    def test_turns_newlines_into_spaces(self):
        assert normalize("hello\nworld").normalized == "hello world"


class TestRepeats:
    def test_collapses_long_runs(self):
        assert normalize("fuuuuck you").normalized == "fuuck you"

    def test_leaves_short_runs_alone(self):
        assert normalize("good").normalized == "good"


class TestWhitespace:
    def test_collapses_runs_of_whitespace(self):
        assert normalize("he  llo \t world").normalized == "he llo world"


class TestHash:
    def test_is_stable_for_equal_text(self):
        assert normalize("gg").message_hash == normalize("gg").message_hash

    def test_differs_for_different_text(self):
        assert normalize("gg").message_hash != normalize("wp").message_hash

    def test_ignores_evasion_differences(self):
        assert normalize("n00b").message_hash == normalize("noob").message_hash

    def test_is_a_short_hex_digest(self):
        digest = message_hash("hello")
        assert len(digest) == 16
        assert all(character in "0123456789abcdef" for character in digest)


class TestOriginalIsPreserved:
    def test_keeps_the_message_as_it_was_sent(self):
        raw = "n00b  f@gg0t"
        assert normalize(raw).original == raw


class TestSimilarity:
    def test_identical_strings_are_fully_similar(self):
        assert similarity("gg wp", "gg wp") == 1.0

    def test_near_duplicate_stays_above_the_threshold(self):
        assert similarity("gg noob", "gg noob!") >= 0.90

    def test_unrelated_strings_are_not_similar(self):
        assert similarity("gg noob", "hello there friend") < 0.5

    def test_empty_strings_are_not_similar(self):
        assert similarity("", "hello") == 0.0
