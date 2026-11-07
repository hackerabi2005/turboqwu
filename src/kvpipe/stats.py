"""Statistical engine: paired bootstrap, 95% confidence intervals, verdicts, and power planning."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_problem_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute per-problem accuracy (mean across samples for that problem)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in records:
        pid = r["item_id"]
        val = 100.0 if r.get("correct", False) else 0.0
        sums[pid] = sums.get(pid, 0.0) + val
        counts[pid] = counts.get(pid, 0) + 1

    return {pid: sums[pid] / counts[pid] for pid in sums}


def paired_bootstrap_ci(
    cand_records: list[dict[str, Any]],
    ref_records: list[dict[str, Any]],
    num_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate 95% paired bootstrap CI of (candidate - reference) over problems."""
    cand_scores = compute_problem_scores(cand_records)
    ref_scores = compute_problem_scores(ref_records)

    common_pids = sorted(set(cand_scores.keys()) & set(ref_scores.keys()))
    if not common_pids:
        return {
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "cand_mean": 0.0,
            "ref_mean": 0.0,
            "n_problems": 0,
        }

    c_vals = np.array([cand_scores[p] for p in common_pids])
    r_vals = np.array([ref_scores[p] for p in common_pids])
    n = len(common_pids)

    cand_mean = float(np.mean(c_vals))
    ref_mean = float(np.mean(r_vals))
    mean_delta = cand_mean - ref_mean

    rng = np.random.default_rng(seed)
    deltas = np.empty(num_resamples)

    for i in range(num_resamples):
        idx = rng.choice(n, size=n, replace=True)
        deltas[i] = np.mean(c_vals[idx]) - np.mean(r_vals[idx])

    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100
    ci_lower = float(np.percentile(deltas, lower_p))
    ci_upper = float(np.percentile(deltas, upper_p))

    return {
        "mean_delta": round(mean_delta, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "cand_mean": round(cand_mean, 2),
        "ref_mean": round(ref_mean, 2),
        "n_problems": n,
        "delta_std": round(float(np.std(deltas)), 3),
    }


def evaluate_gate_verdict(ci_lower: float, ci_upper: float, margin: float = 2.0) -> str:
    """Evaluate PASS, FAIL, or INCONCLUSIVE based on non-inferiority margin (Section 7).

    Pass: CI lower bound >= -margin
    Fail: CI upper bound < -margin
    Inconclusive: CI lower bound < -margin and CI upper bound >= -margin
    """
    threshold = -abs(margin)
    if ci_lower >= threshold:
        return "PASS"
    elif ci_upper < threshold:
        return "FAIL"
    else:
        return "INCONCLUSIVE"


def power_planning(
    delta_std: float,
    target_margin: float = 2.0,
    z_score: float = 1.96,
    max_budget_samples: int = 128,
) -> dict[str, Any]:
    """Calculate required sample size per problem so CI half-width is at most target_margin."""
    if delta_std <= 0:
        return {"recommended_samples_per_problem": 8, "budget_sufficient": True}

    # CI half-width H = z * (std / sqrt(k)) -> k = (z * std / H)^2
    needed_k = int(np.ceil((z_score * delta_std / target_margin) ** 2))
    needed_k = max(needed_k, 8)

    budget_sufficient = needed_k <= max_budget_samples
    recommended = min(needed_k, max_budget_samples) if not budget_sufficient else needed_k

    return {
        "recommended_samples_per_problem": recommended,
        "needed_samples": needed_k,
        "budget_sufficient": budget_sufficient,
        "status": "sufficient" if budget_sufficient else "budget_binds_inconclusive",
    }
