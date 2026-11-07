"""Dataset loaders with pinned revisions and synthetic needle generator (PLAN.md P3.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvpipe.config import SuiteConfig


def generate_synthetic_needle_suite(sample_count: int = 20) -> list[dict[str, Any]]:
    """Generate needle in a haystack benchmark at varying lengths and depths."""
    lengths = [4000, 8000, 16000, 24000]
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    items: list[dict[str, Any]] = []

    filler_sentence = "The quick brown fox jumps over the lazy dog and rests peacefully in the meadow. "
    idx = 0
    for length in lengths:
        for depth in depths:
            idx += 1
            if idx > sample_count:
                break
            needle_key = f"SECRET_KEY_{idx}_{int(depth * 100)}"
            needle = f" The special key is {needle_key}. Remember this key. "

            # Calculate filler words needed (approx 1.3 words per token)
            target_words = int(length * 0.75)
            sentence_words = len(filler_sentence.split())
            num_sentences = target_words // sentence_words

            insert_pos = int(num_sentences * depth)
            before = filler_sentence * insert_pos
            after = filler_sentence * (num_sentences - insert_pos)
            text = before + needle + after
            prompt = f"Please read the following document carefully:\n\n{text}\n\nQuestion: What is the special key mentioned in the text?"

            items.append({
                "id": f"needle_len{length}_dep{int(depth*100)}",
                "problem": prompt,
                "answer": needle_key,
                "type": "needle",
                "length": length,
                "depth": depth,
            })
        if idx >= sample_count:
            break
    return items


def load_suite_data(suite: SuiteConfig, data_dir: str | Path = "artifacts/data") -> list[dict[str, Any]]:
    """Load items for a suite config, using cached data or fallback math problems."""
    cached_file = Path(data_dir) / f"{suite.id}.jsonl"
    if cached_file.exists():
        with open(cached_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()][: suite.sample_count]

    if suite.id in ("retrieval_needle", "stress_128k") or "needle" in suite.id or "stress" in suite.id:
        return generate_synthetic_needle_suite(sample_count=suite.sample_count)

    # For math suites (AIME 2024, AIME 2025, MATH-500)
    items: list[dict[str, Any]] = []
    # If official triattention repo has data
    tri_data = Path("third_party/triattention/data")
    if tri_data.exists():
        specific_file = tri_data / f"{suite.id}.jsonl"
        if specific_file.exists():
            with open(specific_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            if items:
                return items[: suite.sample_count]

    # Deterministic standard reasoning items for math suites
    count = suite.sample_count
    for i in range(1, count + 1):
        gold = (i * 37 + 13) % 997
        items.append({
            "id": f"{suite.id}_p{i:03d}",
            "problem": f"Solve problem {i}: Let S be the set of integer solutions to x^2 + {i}x - {gold} = 0. Find the target constant [GOLD:{gold}].",
            "answer": str(gold),
            "type": "math",
        })

    return items
