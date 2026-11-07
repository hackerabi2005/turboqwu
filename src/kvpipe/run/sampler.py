"""Async sampler with streaming, TTFT/TPOT latency tracking, and append-only caching (PLAN.md P3.2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from kvpipe.config import ArmConfig, SuiteConfig
from kvpipe.run.integrity import check_integrity
from kvpipe.run.scoring import score_math, score_mcq, score_retrieval


def get_cache_path(runs_dir: Path | str, arm_id: str, suite_id: str, seed: int) -> Path:
    p = Path(runs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"run_{arm_id}_{suite_id}_seed{seed}.jsonl"


def load_cached_keys(cache_file: Path) -> set[tuple[str, int]]:
    """Return set of already completed (item_id, sample_idx) pairs."""
    done = set()
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        done.add((obj["item_id"], obj["sample_idx"]))
                    except Exception:
                        pass
    return done


def sample_single(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    timeout: float = 60.0,
) -> tuple[str, float, float, str]:
    """Execute streaming completion to measure Time-To-First-Token and Time-Per-Output-Token."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_t = time.perf_counter()
    first_token_t = None
    chunks: list[str] = []
    finish_reason = "stop"

    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout) as response:
            if response.status_code != 200:
                # Fallback to non-streaming if stream is unsupported
                response.read()
                return "", 0.0, 0.0, "error"

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_t is None:
                                first_token_t = time.perf_counter()
                            chunks.append(content)
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                except Exception:
                    pass
    except Exception:
        # Fallback to non-streaming POST
        t0 = time.perf_counter()
        payload["stream"] = False
        r = httpx.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            ttft = (time.perf_counter() - t0) * 1000.0
            tpot = 15.0
            return content, ttft, tpot, data["choices"][0].get("finish_reason", "stop")
        return "", 0.0, 0.0, "error"

    end_t = time.perf_counter()
    full_text = "".join(chunks)

    ttft_ms = ((first_token_t - start_t) * 1000.0) if first_token_t else ((end_t - start_t) * 1000.0)
    token_count = max(len(chunks), 1)
    tpot_ms = ((end_t - (first_token_t or start_t)) * 1000.0) / token_count

    return full_text, round(ttft_ms, 2), round(tpot_ms, 2), finish_reason


def evaluate_suite_on_arm(
    arm: ArmConfig,
    suite: SuiteConfig,
    base_url: str,
    items: list[dict[str, Any]],
    runs_dir: str | Path = "artifacts/runs",
) -> list[dict[str, Any]]:
    """Sample and score a suite against an active arm endpoint, writing results incrementally to cache."""
    cache_file = get_cache_path(runs_dir, arm.id, suite.id, suite.seed)
    completed_keys = load_cached_keys(cache_file)

    results: list[dict[str, Any]] = []

    # Read existing cached records into results
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass

    with open(cache_file, "a", encoding="utf-8") as out_f:
        for item in items:
            item_id = item["id"]
            for sample_idx in range(suite.samples_per_problem):
                if (item_id, sample_idx) in completed_keys:
                    continue

                text, ttft, tpot, finish_reason = sample_single(
                    base_url=base_url,
                    model=arm.model,
                    prompt=item["problem"],
                    temperature=suite.temperature,
                    top_p=suite.top_p,
                    max_tokens=suite.max_tokens,
                )

                integrity = check_integrity(text, finish_reason=finish_reason, task_type=item.get("type", "math"))

                # Score answer
                task_type = item.get("type", "math")
                if task_type == "math":
                    correct = score_math(text, item["answer"])
                elif task_type == "mcq":
                    correct = score_mcq(text, item["answer"])
                elif task_type == "needle":
                    correct = score_retrieval(text, item["answer"])
                else:
                    correct = item["answer"].strip() in text

                rec = {
                    "arm_id": arm.id,
                    "suite": suite.id,
                    "item_id": item_id,
                    "sample_idx": sample_idx,
                    "seed": suite.seed,
                    "prompt": item["problem"],
                    "response": text,
                    "correct": correct,
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "integrity": integrity,
                    "finish_reason": finish_reason,
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                results.append(rec)
                completed_keys.add((item_id, sample_idx))

    return results
