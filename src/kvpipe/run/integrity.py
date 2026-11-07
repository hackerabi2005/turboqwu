"""Integrity checks on model generations and artifacts/review report generation (PLAN.md P3.4)."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from kvpipe.run.scoring import extract_boxed_answer


def calculate_repetition_ratio(text: str, n: int = 4) -> float:
    """Calculate the ratio of duplicate n-grams to total n-grams in text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    return 1.0 - (unique_count / len(ngrams))


def check_integrity(text: str, finish_reason: str = "stop", task_type: str = "math") -> dict[str, Any]:
    """Perform integrity checks on a model generation."""
    reasons: list[str] = []

    # 1. Empty output check
    is_empty = not text or not text.strip()
    if is_empty:
        reasons.append("empty_output")

    # 2. Finish reason check
    finish_ok = finish_reason == "stop"
    if not finish_ok and finish_reason == "length":
        reasons.append("truncated_length")

    # 3. Corrupt characters: replacement chars (\ufffd) or runs of question marks (>= 4)
    has_replacement = "\ufffd" in text
    has_question_runs = bool(re.search(r"\?{4,}", text))
    if has_replacement or has_question_runs:
        reasons.append("corrupt_characters")

    # 4. N-gram repetition ratio
    rep_ratio = calculate_repetition_ratio(text, n=4)
    if rep_ratio > 0.35 and len(text.split()) > 50:
        reasons.append(f"high_repetition_ratio_{round(rep_ratio, 2)}")

    # 5. Answer extraction success
    extraction_ok = True
    if task_type == "math":
        extraction_ok = extract_boxed_answer(text) is not None
        if not extraction_ok:
            reasons.append("no_boxed_answer")
    elif task_type == "mcq":
        extraction_ok = bool(re.search(r"\b[A-D]\b", text))
        if not extraction_ok:
            reasons.append("no_mcq_option")

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "is_empty": is_empty,
        "finish_reason_ok": finish_ok,
        "corrupt_chars": has_replacement or has_question_runs,
        "repetition_ratio": round(rep_ratio, 3),
        "extraction_ok": extraction_ok,
        "failure_reasons": reasons,
    }


def write_review_file(
    arm_id: str,
    samples: list[dict[str, Any]],
    output_dir: str | Path = "artifacts/review",
    num_samples: int = 20,
) -> Path:
    """Write 20 sample generations to artifacts/review/ARM.md for manual review."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{arm_id}.md"

    selected = random.sample(samples, min(len(samples), num_samples)) if samples else []

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# Review Generations for Arm: {arm_id}\n\n")
        f.write(f"Total evaluated samples: {len(samples)} | Displayed: {len(selected)}\n\n")
        for i, s in enumerate(selected, 1):
            f.write(f"## Sample {i} (Suite: {s.get('suite')}, Problem ID: {s.get('item_id')})\n")
            f.write(f"**Integrity Passed:** {s.get('integrity', {}).get('passed', True)}\n")
            f.write(f"**Score Correct:** {s.get('correct', False)}\n\n")
            f.write(f"**Prompt:**\n```\n{s.get('prompt', '')[:400]}...\n```\n\n")
            f.write(f"**Response:**\n```\n{s.get('response', '')}\n```\n\n")
            f.write("---\n\n")

    return file_path
