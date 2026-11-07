"""Performance harness: concurrency sweep load generator and latency/throughput profiling (PLAN.md Phase 4)."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import numpy as np

from kvpipe.config import ArmConfig
from kvpipe.run.sampler import sample_single


def calculate_kv_bytes_per_token(
    num_layers: int = 36,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    kv_dtype: str = "bf16",
) -> int:
    """Compute nominal KV cache bytes per token: 2 * num_layers * num_kv_heads * head_dim * bytes_per_element."""
    dtype_lower = kv_dtype.lower()
    if "fp4" in dtype_lower or "nvfp4" in dtype_lower:
        element_bytes = 0.5
    elif "fp8" in dtype_lower or "turboquant_4bit" in dtype_lower:
        element_bytes = 1.0  # FP8 or 4-bit packed with scale
    elif "int4" in dtype_lower:
        element_bytes = 0.5
    else:  # bf16, fp16, auto
        element_bytes = 2.0

    return int(2 * num_layers * num_kv_heads * head_dim * element_bytes)


def run_single_load_trial(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int = 8,
    forced_long: bool = False,
) -> dict[str, Any]:
    """Execute concurrent requests against the engine endpoint and record performance metrics."""
    prompt = "Please explain the concept of KV cache compression in LLMs."
    max_tokens = 2048 if not forced_long else 8192

    ttfts: list[float] = []
    tpots: list[float] = []
    total_tokens = 0
    errors = 0

    def worker(_: int) -> tuple[float, float, int, bool]:
        text, ttft, tpot, finish_reason = sample_single(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=120.0,
        )
        token_count = max(len(text.split()), 1)
        ok = finish_reason in ("stop", "length")
        return ttft, tpot, token_count, ok

    start_wall = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            try:
                ttft, tpot, tokens, ok = f.result()
                if ok:
                    ttfts.append(ttft)
                    tpots.append(tpot)
                    total_tokens += tokens
                else:
                    errors += 1
            except Exception:
                errors += 1

    wall_duration = max(time.perf_counter() - start_wall, 0.001)
    tokens_per_sec = total_tokens / wall_duration

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
        "ttft_p95": round(float(np.percentile(ttfts, 95)), 2) if ttfts else 0.0,
        "ttft_p99": round(float(np.percentile(ttfts, 99)), 2) if ttfts else 0.0,
        "tpot_p50": round(float(np.percentile(tpots, 50)), 2) if tpots else 0.0,
        "tpot_p95": round(float(np.percentile(tpots, 95)), 2) if tpots else 0.0,
        "tpot_p99": round(float(np.percentile(tpots, 99)), 2) if tpots else 0.0,
        "errors": errors,
        "wall_time_s": round(wall_duration, 2),
    }


def run_performance_sweep(
    arm: ArmConfig,
    base_url: str,
    concurrencies: list[int] = [1, 4],
    repeats: int = 3,
    forced_long: bool = False,
) -> dict[str, Any]:
    """Execute concurrency sweep with multiple repeats, reporting means and std devs (PLAN.md P4.2, P4.3)."""
    sweep_results: list[dict[str, Any]] = []

    for c in concurrencies:
        trial_tps: list[float] = []
        trial_tpot95: list[float] = []
        trial_ttft95: list[float] = []

        last_trial: dict[str, Any] = {}
        for _ in range(repeats):
            res = run_single_load_trial(
                base_url=base_url,
                model=arm.model,
                concurrency=c,
                num_requests=max(c * 2, 4),
                forced_long=forced_long,
            )
            trial_tps.append(res["tokens_per_sec"])
            trial_tpot95.append(res["tpot_p95"])
            trial_ttft95.append(res["ttft_p95"])
            last_trial = res

        tps_mean = float(np.mean(trial_tps))
        tps_std = float(np.std(trial_tps))
        sweep_results.append({
            "concurrency": c,
            "tokens_per_sec_mean": round(tps_mean, 2),
            "tokens_per_sec_std": round(tps_std, 2),
            "tpot_p95_mean": round(float(np.mean(trial_tpot95)), 2),
            "ttft_p95_mean": round(float(np.mean(trial_ttft95)), 2),
            "details": last_trial,
        })

    kv_bytes = calculate_kv_bytes_per_token(kv_dtype=arm.kv_dtype)

    return {
        "arm_id": arm.id,
        "kv_dtype": arm.kv_dtype,
        "kv_bytes_per_token": kv_bytes,
        "concurrency_sweep": sweep_results,
    }
