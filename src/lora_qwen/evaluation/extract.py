"""Answer extraction for math-task evaluation.

Three cascading strategies, in priority order:

1. ``#### N`` — the GSM8K ground-truth marker.
2. ``The answer is[:]? N`` — the MetaMathQA-style response our FT model
   is trained to emit.
3. Last number in the string — robust fallback.

All strategies normalize commas and dollar signs; matching is done on floats
with a tight tolerance so "72", "72.0", "$72", and "72,000" vs "72000" are
handled consistently.
"""

from __future__ import annotations

import re

# Number patterns: comma-grouped form requires >=1 comma (so plain digit runs
# don't get truncated to the first 3 digits); plain form is the fallback.
_NUM = r"(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
_NUMBER_RE = re.compile(_NUM)
_HASH_RE = re.compile(rf"####\s*\$?({_NUM})")
_ANSWER_IS_RE = re.compile(rf"[Tt]he\s+answer\s+is[:\s]*\$?({_NUM})")


def _to_float(raw: str) -> float | None:
    cleaned = raw.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """Return the canonical answer number from ``text``, or ``None`` if nothing parseable."""
    if (m := _HASH_RE.search(text)):
        return _to_float(m.group(1))
    if (m := _ANSWER_IS_RE.search(text)):
        return _to_float(m.group(1))
    # Fallback: last number anywhere in the text.
    matches = _NUMBER_RE.findall(text)
    if matches:
        return _to_float(matches[-1])
    return None


def numbers_match(a: float | None, b: float | None, *, tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))
