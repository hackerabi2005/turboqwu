"""Answer scorers for math, multiple-choice, and retrieval tasks."""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract contents of the last \\boxed{...} in text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize_math_answer(ans: str) -> str:
    """Normalize common math answer formatting differences."""
    ans = ans.strip().strip("$").strip()
    # Remove \text{...}, \mathrm{...}
    ans = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", ans)
    # Remove spaces
    ans = re.sub(r"\s+", "", ans)
    # Simplify fractions if direct integers, e.g. \frac{a}{b}
    ans = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", ans)
    return ans


def score_math(prediction: str, ground_truth: str) -> bool:
    """Score a mathematical prediction against ground truth."""
    pred_boxed = extract_boxed_answer(prediction)
    if pred_boxed is None:
        # Fallback: look for "The answer is X" or the last number/token
        m = re.findall(r"(?:answer is|equals?|=)\s*([+-]?\d+(?:\.\d+)?)", prediction, re.IGNORECASE)
        if m:
            pred_boxed = m[-1]
        else:
            return False

    norm_pred = normalize_math_answer(pred_boxed)
    norm_gold = normalize_math_answer(ground_truth)

    if norm_pred == norm_gold:
        return True

    # Check numerical float equivalence
    try:
        val_pred = float(eval(norm_pred.replace("^", "**")))
        val_gold = float(eval(norm_gold.replace("^", "**")))
        if abs(val_pred - val_gold) < 1e-5:
            return True
    except Exception:
        pass

    return False


def score_mcq(prediction: str, ground_truth: str) -> bool:
    """Score a multiple choice prediction against ground truth letter (A, B, C, D)."""
    gold = ground_truth.strip().upper()
    # Find boxed letter or 'answer is (X)' or 'X'
    boxed = extract_boxed_answer(prediction)
    if boxed and boxed.strip().upper() == gold:
        return True

    m = re.findall(r"(?:answer is|option)\s*[:(]?\s*([A-D])\b", prediction, re.IGNORECASE)
    if m and m[-1].upper() == gold:
        return True

    # If prediction itself is just the letter
    if prediction.strip().upper() == gold:
        return True

    return False


def score_retrieval(prediction: str, needle_key: str) -> bool:
    """Score a needle retrieval response by exact substring presence."""
    return needle_key.strip().lower() in prediction.lower()
