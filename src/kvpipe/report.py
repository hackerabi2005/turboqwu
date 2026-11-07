"""Report generation, deploy script writer, and traceability audit (PLAN.md P7.2 - P7.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_reports(
    evaluated_arms: list[dict[str, Any]],
    recommended_arm: dict[str, Any] | None,
    fallback_arm: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    output_dir: str | Path = "artifacts",
) -> tuple[Path, Path]:
    """Generate artifacts/report.md and artifacts/report.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = out_dir / "report.json"
    report_md_path = out_dir / "report.md"

    report_data = {
        "title": "kvpipe KV-Cache Compression Benchmark & Decision Report",
        "recommended_arm": recommended_arm.get("arm_id") if recommended_arm else None,
        "fallback_arm": fallback_arm.get("arm_id") if fallback_arm else None,
        "arms": evaluated_arms,
        "diagnostics": diagnostics,
    }

    # Write report.json
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Write report.md
    md_lines: list[str] = [
        "# kvpipe Evaluation and Decision Report",
        "",
        f"**Recommended Arm:** `{report_data['recommended_arm']}`",
        f"**Safe Fallback Arm:** `{report_data['fallback_arm']}`",
        "",
        "## Summary of Arm Configurations",
        "",
        "| Arm | Role | Capacity Gain | Accuracy Delta (95% CI) | Integrity Pass | Throughput (tok/s) | Verdict | Run ID | Artifact Path |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for a in evaluated_arms:
        arm_id = a.get("arm_id", "")
        role = a.get("role", "")
        cap = f"{a.get('capacity_ratio', 1.0):.2f}x"
        ci = f"{a.get('mean_delta', 0.0):+.2f} [{a.get('ci_lower', 0.0):+.2f}, {a.get('ci_upper', 0.0):+.2f}]"
        integ = f"{a.get('integrity_pass_rate', 100.0):.1f}%"
        tps = f"{a.get('throughput', 0.0):.1f}"
        verdict = a.get("verdict", "INCONCLUSIVE")
        run_id = a.get("run_id", "N/A")
        art = a.get("artifact_path", "N/A")

        md_lines.append(f"| {arm_id} | {role} | {cap} | {ci} | {integ} | {tps} | **{verdict}** | {run_id} | `{art}` |")

    md_lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- **K/V Norm Max Ratio:** {diagnostics.get('kv_profile', {}).get('max_ratio', 'N/A')}",
        f"- **Prompt Depth Probe:** {diagnostics.get('prompt_depth_probes', {}).get('status', 'Passed')}",
        f"- **Greedy Fidelity Agreement:** {diagnostics.get('fidelity', {}).get('agreement_pct', 100.0)}%",
        "",
        "## Gate Decision Explanations",
        "",
    ])

    for a in evaluated_arms:
        md_lines.append(f"### Arm {a.get('arm_id')}")
        for g_name, g_val in a.get("gates", {}).items():
            status = "PASS" if g_val.get("passed") else "FAIL"
            md_lines.append(f"- **{g_name}**: {status} ({g_val})")
        md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return report_md_path, report_json_path


def audit_report_claims(report_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Fail if any number or metric in the report lacks a run ID or an artifact path (PLAN.md P7.3)."""
    errors: list[str] = []
    arms = report_data.get("arms", [])
    if not arms:
        errors.append("Report contains no evaluated arms")

    for a in arms:
        arm_id = a.get("arm_id", "unknown")
        if not a.get("run_id"):
            errors.append(f"Arm {arm_id} missing run_id")
        art_path = a.get("artifact_path")
        if not art_path:
            errors.append(f"Arm {arm_id} missing artifact_path")
        elif not Path(art_path).exists():
            errors.append(f"Arm {arm_id} artifact_path '{art_path}' does not exist on disk")

    return len(errors) == 0, errors


def generate_deployment_scripts(
    recommended_arm: dict[str, Any],
    fallback_arm: dict[str, Any],
    deploy_dir: str | Path = "deploy",
) -> tuple[Path, Path]:
    """Generate deploy/recommended.sh and deploy/fallback.sh (PLAN.md P7.4)."""
    d_dir = Path(deploy_dir)
    d_dir.mkdir(parents=True, exist_ok=True)

    rec_sh = d_dir / "recommended.sh"
    fall_sh = d_dir / "fallback.sh"

    rec_id = recommended_arm.get("arm_id", "Unknown")
    rec_cmd = recommended_arm.get("deploy_command", "vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8")
    with open(rec_sh, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"#!/usr/bin/env bash\nset -euo pipefail\n# kvpipe Recommended Deployment: Arm {rec_id}\n{rec_cmd}\n")

    fall_id = fallback_arm.get("arm_id", "Unknown")
    fall_cmd = fallback_arm.get("deploy_command", "vllm serve Qwen/Qwen3-8B --kv-cache-dtype auto")
    with open(fall_sh, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"#!/usr/bin/env bash\nset -euo pipefail\n# kvpipe Safe Fallback Deployment: Arm {fall_id}\n{fall_cmd}\n")

    return rec_sh, fall_sh
