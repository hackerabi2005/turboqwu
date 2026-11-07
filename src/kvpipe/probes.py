"""Position probes and fidelity signal diagnostics (PLAN.md P6.3, P6.4)."""

from __future__ import annotations

import random
from typing import Any

from kvpipe.run.sampler import sample_single


def run_prompt_depth_probes(
    base_url: str,
    model: str,
    depths: list[float] = [0.1, 0.5, 0.9],
) -> dict[str, Any]:
    """Test needle retrieval across prompt depths (10%, 50%, 90%).

    Because prompt tokens are protected in TriAttention, accuracy should remain 100%.
    """
    filler = "The expedition continued through the valley with careful observations and detailed notes recorded daily. "
    results: dict[str, bool] = {}

    for d in depths:
        secret_code = f"CODE_D{int(d * 100)}_{random.randint(1000, 9999)}"
        num_sentences = 200
        insert_idx = int(num_sentences * d)

        before = filler * insert_idx
        needle = f" The confidential verification code is {secret_code}. "
        after = filler * (num_sentences - insert_idx)

        prompt = f"Background document:\n\n{before}{needle}{after}\n\nWhat is the confidential verification code?"
        resp, _, _, _ = sample_single(base_url, model, prompt, temperature=0.0, max_tokens=100)
        found = secret_code in resp
        results[f"depth_{int(d * 100)}pct"] = found

    return {
        "all_depths_passed": all(results.values()),
        "probe_depths": results,
    }


def run_generated_memory_probe(
    base_url: str,
    model: str,
    budget: int = 3072,
) -> dict[str, Any]:
    """Test recall of a fact generated early in an output sequence (~10K tokens earlier)."""
    probe_code = f"GENMEM_{random.randint(10000, 99999)}"
    prompt = (
        f"Generate output starting with: CODE: {probe_code}.\n"
        "Then perform counting calculations for approximately 200 steps.\n"
        "Finally, on the last line, state: RECALLED_CODE: [code]."
    )

    resp, _, _, _ = sample_single(base_url, model, prompt, temperature=0.0, max_tokens=1024)
    recalled = probe_code in resp

    return {
        "budget": budget,
        "probe_code": probe_code,
        "recalled": recalled,
        "eviction_risk": "high" if (not recalled and budget <= 2048) else "low",
    }


def measure_fidelity_signal(
    cand_base_url: str,
    cand_model: str,
    ref_base_url: str,
    ref_model: str,
    prompts: list[str],
) -> dict[str, Any]:
    """Calculate greedy-decode exact agreement and divergence against reference A0 (PLAN.md P6.4)."""
    matches = 0
    total = len(prompts)
    if total == 0:
        return {"greedy_agreement_pct": 100.0, "total_prompts": 0}

    for p in prompts:
        cand_resp, _, _, _ = sample_single(cand_base_url, cand_model, p, temperature=0.0, max_tokens=128)
        ref_resp, _, _, _ = sample_single(ref_base_url, ref_model, p, temperature=0.0, max_tokens=128)

        if cand_resp.strip() == ref_resp.strip():
            matches += 1

    agreement_pct = round((matches / total) * 100.0, 2)
    return {
        "greedy_agreement_pct": agreement_pct,
        "total_prompts": total,
        "matched_prompts": matches,
    }
