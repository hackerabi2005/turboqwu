"""Regression testing and production canary monitoring (PLAN.md P8.1 - P8.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvpipe.config import ArmConfig, SuiteConfig
from kvpipe.data.loaders import load_suite_data
from kvpipe.run.sampler import evaluate_suite_on_arm, sample_single


def run_regression_check(
    arm: ArmConfig,
    base_url: str,
    golden_file: str | Path = "tests/golden.jsonl",
    previous_report_path: str | Path = "artifacts/report.json",
    runs_dir: str | Path = "artifacts/runs",
) -> dict[str, Any]:
    """Re-check golden prompts, canary, and reduced suite, comparing with stored report."""
    golden_path = Path(golden_file)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden prompts not found at {golden_path}")

    # 1. Evaluate golden prompts
    golden_passed = 0
    golden_total = 0
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            golden_total += 1
            resp, _, _, finish = sample_single(
                base_url=base_url,
                model=arm.model,
                prompt=item["prompt"],
                temperature=0.0,
                max_tokens=256,
            )
            exp = item["expected"].strip().lower()
            if exp in resp.strip().lower() and finish in ("stop", "length"):
                golden_passed += 1

    golden_all_pass = golden_passed == golden_total

    # 2. Reduced suite run
    suite = SuiteConfig(
        id="aime25",
        name="AIME 2025 Regression",
        dataset_id="test",
        sample_count=5,
        samples_per_problem=1,
    )
    items = load_suite_data(suite)
    results = evaluate_suite_on_arm(arm, suite, base_url, items[:5], runs_dir=runs_dir)
    reduced_accuracy = (sum(1 for r in results if r["correct"]) / len(results)) * 100.0 if results else 0.0

    # 3. Canary / eviction check if eviction arm
    canary_ok = True
    if arm.compression.method == "triattention":
        canary_ok = (arm.compression.budget or 0) > 0

    # 4. Compare with previous report if available
    baseline_match = True
    prev_report = Path(previous_report_path)
    if prev_report.exists():
        with open(prev_report, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
            # Find arm in previous report
            for pa in prev_data.get("arms", []):
                if pa.get("arm_id") == arm.id:
                    # Compare accuracy within standard margin
                    prev_acc = pa.get("cand_mean", pa.get("accuracy", 100.0))
                    if abs(reduced_accuracy - prev_acc) > 30.0:  # Large discrepancy flag
                        baseline_match = False
                    break

    overall_pass = golden_all_pass and canary_ok and baseline_match

    return {
        "arm_id": arm.id,
        "overall_pass": overall_pass,
        "golden_passed": golden_passed,
        "golden_total": golden_total,
        "reduced_accuracy_pct": round(reduced_accuracy, 1),
        "canary_ok": canary_ok,
        "baseline_match": baseline_match,
    }


def evaluate_canary_rollout_metrics(
    baseline_integrity_fail_rate: float,
    canary_integrity_fail_rate: float,
    baseline_p95_tpot: float,
    canary_p95_tpot: float,
    tpot_threshold_ms: float = 30.0,
) -> dict[str, Any]:
    """Evaluate 5% canary rollout health and trigger rollback if thresholds breached (PLAN.md P8.3).

    Rollback triggered if:
    - Integrity failure rate exceeds baseline by 0.5 points
    - p95 TPOT breaches target threshold
    """
    integrity_delta = canary_integrity_fail_rate - baseline_integrity_fail_rate
    integrity_breach = integrity_delta > 0.5
    tpot_breach = canary_p95_tpot > tpot_threshold_ms

    rollback_needed = integrity_breach or tpot_breach
    status = "ROLLBACK_TRIGGERED" if rollback_needed else "HEALTHY"

    return {
        "status": status,
        "rollback_needed": rollback_needed,
        "integrity_delta": round(integrity_delta, 2),
        "integrity_breach": integrity_breach,
        "canary_p95_tpot": canary_p95_tpot,
        "tpot_breach": tpot_breach,
    }
