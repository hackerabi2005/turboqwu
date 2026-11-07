"""Gate evaluation and recommendation decision engine (PLAN.md Section 7, P7.1)."""

from __future__ import annotations

from typing import Any

from kvpipe.config import GateThresholds


def evaluate_arm_gates(
    arm_id: str,
    arm_role: str,
    is_eviction: bool,
    golden_pass_count: int,
    corrupt_rate_delta: float,
    extraction_rate_delta: float,
    reasoning_ci_lower: float,
    retrieval_ci_lower: float | None,
    eviction_applied_events: int | None,
    canary_drop: float | None,
    capacity_ratio: float,
    throughput_ratio: float,
    repeat_std: float,
    thresholds: GateThresholds,
    protect_prompt: bool = True,
) -> dict[str, Any]:
    """Evaluate gates G1 through G7 for an individual arm."""
    gate_results: dict[str, dict[str, Any]] = {}

    # G1: Integrity
    g1_pass = (
        golden_pass_count >= thresholds.g1_golden_count
        and corrupt_rate_delta <= thresholds.g1_corrupt_delta_max
        and extraction_rate_delta >= thresholds.g1_extraction_delta_min
    )
    gate_results["G1_Integrity"] = {
        "passed": g1_pass,
        "golden_passed": golden_pass_count,
        "corrupt_delta": corrupt_rate_delta,
        "extraction_delta": extraction_rate_delta,
    }

    # G2: Reasoning Accuracy
    g2_pass = reasoning_ci_lower >= -abs(thresholds.g2_reasoning_margin)
    gate_results["G2_Reasoning"] = {
        "passed": g2_pass,
        "ci_lower": reasoning_ci_lower,
        "margin": thresholds.g2_reasoning_margin,
    }

    # G3: Retrieval Accuracy (if applicable)
    g3_pass = True
    if is_eviction and not protect_prompt:
        g3_pass = False
    elif retrieval_ci_lower is not None:
        g3_pass = retrieval_ci_lower >= -abs(thresholds.g3_retrieval_margin)
    gate_results["G3_Retrieval"] = {
        "passed": g3_pass,
        "ci_lower": retrieval_ci_lower,
        "margin": thresholds.g3_retrieval_margin,
        "protect_prompt": protect_prompt,
    }

    # G4: Eviction Active (eviction arms only)
    g4_pass = True
    if is_eviction:
        applied_ok = (eviction_applied_events or 0) > 0
        canary_ok = (canary_drop is not None) and (canary_drop >= thresholds.g4_canary_min_drop)
        g4_pass = applied_ok and canary_ok
        gate_results["G4_EvictionActive"] = {
            "passed": g4_pass,
            "applied_events": eviction_applied_events,
            "canary_drop": canary_drop,
        }

    # G5: Capacity Gain vs A0
    # Reference A0 itself doesn't need 1.8x, but compression arms must be >= 1.8x
    if arm_role in ("reference", "paper_reproduction"):
        g5_pass = True
    else:
        g5_pass = capacity_ratio >= thresholds.g5_compression_min
    gate_results["G5_Capacity"] = {
        "passed": g5_pass,
        "capacity_ratio": capacity_ratio,
        "min_required": thresholds.g5_compression_min,
    }

    # G6: Performance (Throughput >= 70% of A1)
    g6_pass = throughput_ratio >= thresholds.g6_throughput_ratio_min
    gate_results["G6_Performance"] = {
        "passed": g6_pass,
        "throughput_ratio": throughput_ratio,
        "min_required": thresholds.g6_throughput_ratio_min,
    }

    # G7: Stability
    g7_pass = repeat_std <= thresholds.g7_repeat_std_max
    gate_results["G7_Stability"] = {
        "passed": g7_pass,
        "repeat_std": repeat_std,
        "max_allowed": thresholds.g7_repeat_std_max,
    }

    overall_pass = all(g["passed"] for g in gate_results.values())

    return {
        "arm_id": arm_id,
        "overall_pass": overall_pass,
        "gates": gate_results,
    }


def select_best_arm(
    arm_evaluations: list[dict[str, Any]],
    fallback_id: str = "A1",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select the highest-compression passing arm, breaking ties on throughput.

    Also identifies the fallback arm (default A1).
    """
    passing_candidates = [
        a for a in arm_evaluations
        if a.get("overall_pass")
        and a.get("arm_id") not in ("A0", "R1", "C1", "A6", "A8")
        and a.get("role") not in ("reference", "paper_reproduction", "canary", "negative_control", "unverified_combination", "offline_docqa")
    ]

    if not passing_candidates:
        # If no compression candidate passed all gates, fall back to A1 if passing
        a1_candidates = [a for a in arm_evaluations if a.get("arm_id") == fallback_id]
        fallback = a1_candidates[0] if a1_candidates else None
        return fallback, fallback

    # Sort descending by capacity_ratio, then by throughput (tokens/sec)
    passing_candidates.sort(
        key=lambda x: (x.get("capacity_ratio", 1.0), x.get("throughput", 0.0)),
        reverse=True,
    )

    recommended = passing_candidates[0]

    # Find fallback arm (e.g. A1 or next highest safe arm)
    fallback = None
    for a in arm_evaluations:
        if a.get("arm_id") == fallback_id and a.get("overall_pass"):
            fallback = a
            break
    if not fallback:
        fallback = passing_candidates[-1]

    return recommended, fallback
