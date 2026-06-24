"""Unit tests for answer extraction (pure string logic, no model/data download).

Pins the cascade priority that the whole GSM8K pipeline relies on:
``#### N`` wins over ``The answer is: N`` wins over last-number, and numeric
normalization (commas, ``$``, decimals) is consistent so ground-truth and
model output compare on value, not surface form.
"""

from __future__ import annotations

from lora_qwen.evaluation.extract import extract_number, numbers_match


def test_hash_marker_has_top_priority() -> None:
    # GSM8K ground-truth format; also a distractor number earlier in the text.
    text = "First 12, then 7 more, so 12 + 7 = 19.\n#### 19"
    assert extract_number(text) == 19.0


def test_hash_wins_over_answer_is_when_both_present() -> None:
    text = "The answer is: 5 ... but the final is\n#### 42"
    assert extract_number(text) == 42.0


def test_answer_is_marker_used_when_no_hash() -> None:
    text = "Work it out: 6 * 7 = 42. The answer is: 42"
    assert extract_number(text) == 42.0


def test_last_number_fallback() -> None:
    text = "He has 3 apples and buys 4 more for a total of 7 apples."
    assert extract_number(text) == 7.0


def test_comma_grouped_numbers_normalized() -> None:
    assert extract_number("#### 70,000") == 70000.0
    assert numbers_match(extract_number("#### 70,000"), 70000.0)


def test_dollar_and_decimal_normalized() -> None:
    assert extract_number("#### $18.50") == 18.5


def test_gsm8k_ground_truth_shape_parses() -> None:
    # The exact shape GSM8K's `answer` field has (multi-line CoT + #### N).
    gt = (
        "Janet sells 16 - 3 - 4 = 9 eggs.\n"
        "She makes 9 * 2 = 18 dollars.\n"
        "#### 18"
    )
    assert extract_number(gt) == 18.0


def test_none_when_no_number() -> None:
    assert extract_number("no digits here") is None
    assert not numbers_match(None, 5.0)
