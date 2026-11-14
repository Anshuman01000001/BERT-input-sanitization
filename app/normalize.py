"""Evasion-resistant text normalization.

The original message is always preserved for logs, moderator review and
appeals. Only the normalized copy is fed to the classifier and used for
duplicate detection.

Pipeline (the order matters):

1. NFKC normalization     - folds fullwidth, styled and compatibility forms
2. invisible char strip  - zero-width, bidi overrides, tag/format chars
3. homoglyph mapping      - Cyrillic / Greek / other lookalikes -> Latin
4. leetspeak mapping      - ``@``->a, ``0``->o, ... (scoped to real words)
5. repeated-char collapse - "fuuuuck" -> "fuuck"
6. whitespace collapse    - runs of whitespace -> a single space

``folded`` keeps the original casing (spam heuristics need to see ALL CAPS).
``normalized`` is lowercased and is what the model and the duplicate detector
see.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
from dataclasses import dataclass

# --- Cyrillic lookalikes (U+0400 block) -------------------------------------
_CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "b", "г": "r", "д": "d", "е": "e", "ё": "e",
    "ж": "x", "з": "z", "и": "u", "й": "u", "к": "k", "л": "l", "м": "m",
    "н": "h", "о": "o", "п": "n", "р": "p", "с": "c", "т": "t", "у": "y",
    "ф": "f", "х": "x", "ц": "c", "ч": "4", "ш": "w", "щ": "w", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "io", "я": "ia",
}

# --- Greek lookalikes --------------------------------------------------------
_GREEK_TO_LATIN = {
    "α": "a", "β": "b", "γ": "y", "δ": "d", "ε": "e", "ζ": "z", "η": "n",
    "θ": "0", "ι": "i", "κ": "k", "λ": "l", "μ": "u", "ν": "v", "ξ": "x",
    "ο": "o", "π": "n", "ρ": "p", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
    "φ": "f", "χ": "x", "ψ": "y", "ω": "w",
}

# --- Other lookalikes seen in gaming chats ----------------------------------
_OTHER_LOOKALIKES = {
    "ı": "i", "İ": "i", "ø": "o", "œ": "oe", "æ": "ae", "ł": "l", "đ": "d",
    "ð": "d", "þ": "th", "ß": "ss", "ŋ": "n", "ə": "e", "ƒ": "f",
    "ɡ": "g", "ʀ": "r", "ǀ": "l", "ƚ": "l", "Ʌ": "t", "ʜ": "h", "ʟ": "l",
    "ᴀ": "a", "ᴏ": "o", "ᴛ": "t", "ɪ": "i", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e",
    "ᴊ": "j", "ᴋ": "k", "ᴍ": "m", "ᴎ": "n", "ᴘ": "p", "ᴠ": "v", "ᴡ": "w",
    "ᴢ": "x", "ᴤ": "u", "ᴦ": "v", "ᶜ": "c", "ᶻ": "z", "ᶿ": "s", "ᶮ": "n",
    "ᵏ": "k", "ᵼ": "m", "ᵽ": "p", "ᵾ": "u", "ᵿ": "v",
}

# --- Leetspeak ---------------------------------------------------------------
# Digit substitutions are safe to apply to any word. The punctuation ones are
# far more often punctuation than leet ("nice shot!" is not "nice shoti"), so
# they are only applied to tokens that also contain a digit.
_LEET_DIGITS = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
}
_LEET_SYMBOLS = {
    "@": "a", "$": "s",
}
_LEET_MAP = {**_LEET_DIGITS, **_LEET_SYMBOLS}

_LOOKALIKES = {**_CYRILLIC_TO_LATIN, **_GREEK_TO_LATIN, **_OTHER_LOOKALIKES}

# Zero-width, bidi override, tag and variation-selector characters.
# The escapes are written out in the source on purpose: literal invisible
# characters in source are invisible to the next reader and easy to mangle
# in a diff. Python interprets them into the characters they name.
_INVISIBLE_RE = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff"
    "\ufe00-\ufe0f\U000e0000-\U000e007f]"
)
# C0/C1 control characters, keeping tab/newline/carriage-return (collapsed later).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEAT_RE = re.compile(r"([a-zA-Z])\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_HAS_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """A message and the forms derived from it.

    Attributes:
        original: The message exactly as the player sent it.
        folded: Case-preserving, homoglyph-corrected text used by the spam
            heuristics that care about casing and character content.
        normalized: Lowercased, leet-mapped, de-duplicated text. This is what
            the classifier sees and what the duplicate detector compares.
        tokens: Word tokens of ``normalized``.
        message_hash: Short, stable digest of ``normalized``.
    """

    original: str
    folded: str
    normalized: str
    tokens: tuple[str, ...]
    message_hash: str


def _strip_invisible(text: str) -> str:
    return _INVISIBLE_RE.sub("", text)


def _map_lookalikes(text: str) -> str:
    """Replace Cyrillic/Greek/other lookalike characters with Latin ones.

    The lookup tables are written in lowercase, so uppercase lookalikes are
    resolved through their lowercase form and the result keeps the original
    casing (ALL-CAPS spam heuristics still see caps).
    """
    out: list[str] = []
    for ch in text:
        mapped = _LOOKALIKES.get(ch)
        if mapped is None:
            mapped = _LOOKALIKES.get(ch.lower())
            if mapped is not None and ch.isupper():
                mapped = mapped.upper()
        out.append(mapped if mapped is not None else ch)
    return "".join(out)


def _apply_leet(text: str) -> str:
    """Map leetspeak characters, but only inside tokens that contain letters.

    Scoping the mapping this way avoids turning pure numbers into nonsense
    ("1000" stays "1000") while still catching "n00b", "f@gg0t" and "sk1ll".
    """
    out: list[str] = []
    for token in text.split(" "):
        if _HAS_ASCII_LETTER_RE.search(token):
            out.append("".join(_LEET_MAP.get(ch, ch) for ch in token))
        else:
            out.append(token)
    return " ".join(out)


def _collapse_repeats(text: str) -> str:
    """Collapse runs of 3+ identical letters down to 2 ("fuuuuck" -> "fuuck")."""
    return _REPEAT_RE.sub(r"\1\1", text)


def message_hash(normalized: str) -> str:
    """Stable short digest of a normalized message."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalize(text: str) -> NormalizedMessage:
    """Normalize a raw chat message into its comparable forms."""
    original = text
    folded = unicodedata.normalize("NFKC", text)
    folded = _strip_invisible(_CONTROL_RE.sub("", folded))
    folded = _map_lookalikes(folded)
    folded = _WHITESPACE_RE.sub(" ", folded).strip()

    normalized = _apply_leet(folded)
    normalized = _collapse_repeats(normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip().lower()

    return NormalizedMessage(
        original=original,
        folded=folded,
        normalized=normalized,
        tokens=tuple(_TOKEN_RE.findall(normalized)),
        message_hash=message_hash(normalized),
    )


def similarity(left: str, right: str) -> float:
    """Similarity of two normalized strings in ``[0.0, 1.0]``.

    Uses ``difflib``'s ratio, which is 2*M/T for M matching characters over
    the combined length. A message padded with four extra characters still
    scores ~0.91 against its original, so the 0.90 near-duplicate threshold
    catches the realistic variants without flagging merely similar messages.
    """
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    shorter, longer = sorted((len(left), len(right)))
    if shorter / longer < 0.5:
        # Too different in shape to be worth the comparison.
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()
