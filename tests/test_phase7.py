"""Unit tests for Phase 7: Gate evaluation, decision selection, reports, and claims audit."""

from kvpipe.config import GateThresholds
from kvpipe.decide import evaluate_arm_gates, select_best_arm
from kvpipe.report import audit_report_claims, generate_deployment_scripts, generate_reports


def test_evaluate_arm_gates():
    thresholds = GateThresholds()

    # Successful arm A4 evaluation
    eval_res = evaluate_arm_gates(
        arm_id="A4",
        arm_role="reasoning_candidate_target",
        is_eviction=True,
        golden_pass_count=20,
        corrupt_rate_delta=0.0,
        extraction_rate_delta=0.0,
        reasoning_ci_lower=-1.2,  # >= -2.0
        retrieval_ci_lower=-0.5,
        eviction_applied_events=12,
        canary_drop=15.0,  # >= 10.0
        capacity_ratio=2.0,  # >= 1.8
        throughput_ratio=0.85,  # >= 0.70
        repeat_std=0.8,  # <= 1.5
        thresholds=thresholds,
    )
    assert eval_res["overall_pass"] is True
    assert eval_res["gates"]["G1_Integrity"]["passed"] is True
    assert eval_res["gates"]["G2_Reasoning"]["passed"] is True
    assert eval_res["gates"]["G4_EvictionActive"]["passed"] is True
    assert eval_res["gates"]["G5_Capacity"]["passed"] is True


def test_select_best_arm():
    candidates = [
        {"arm_id": "A1", "overall_pass": True, "capacity_ratio": 2.0, "throughput": 100.0},
        {"arm_id": "A4", "overall_pass": True, "capacity_ratio": 2.8, "throughput": 120.0},
        {"arm_id": "A5", "overall_pass": False, "capacity_ratio": 3.2, "throughput": 130.0},
    ]
    recommended, fallback = select_best_arm(candidates, fallback_id="A1")
    assert recommended is not None
    assert recommended["arm_id"] == "A4"
    assert fallback is not None
    assert fallback["arm_id"] == "A1"


def test_generate_reports_and_audit(tmp_path):
    # Create fake artifact file to satisfy audit
    art_file = tmp_path / "run_A4.jsonl"
    art_file.write_text("{}", encoding="utf-8")

    evaluated = [
        {
            "arm_id": "A4",
            "role": "candidate",
            "capacity_ratio": 2.8,
            "mean_delta": -0.5,
            "ci_lower": -1.2,
            "ci_upper": 0.2,
            "integrity_pass_rate": 100.0,
            "throughput": 120.0,
            "verdict": "PASS",
            "run_id": "RUN_A4_001",
            "artifact_path": str(art_file),
            "gates": {"G1": {"passed": True}},
        }
    ]

    diagnostics = {
        "kv_profile": {"max_ratio": 1.35},
        "prompt_depth_probes": {"status": "Passed"},
        "fidelity": {"agreement_pct": 98.5},
    }

    rep_md, rep_json = generate_reports(
        evaluated_arms=evaluated,
        recommended_arm=evaluated[0],
        fallback_arm=evaluated[0],
        diagnostics=diagnostics,
        output_dir=tmp_path,
    )
    assert rep_md.exists()
    assert rep_json.exists()

    # Audit check
    with open(rep_json, "r", encoding="utf-8") as f:
        import json
        data = json.load(f)

    ok, errors = audit_report_claims(data)
    assert ok is True
    assert len(errors) == 0

    # Test audit failure when artifact path missing
    data["arms"][0]["artifact_path"] = "non_existent.jsonl"
    ok_fail, errors_fail = audit_report_claims(data)
    assert ok_fail is False
    assert len(errors_fail) > 0


def test_generate_deployment_scripts(tmp_path):
    arm_rec = {"arm_id": "A4", "deploy_command": "vllm serve Qwen3-8B --kv-cache-dtype fp8"}
    arm_fall = {"arm_id": "A1", "deploy_command": "vllm serve Qwen3-8B --kv-cache-dtype auto"}

    rec_sh, fall_sh = generate_deployment_scripts(arm_rec, arm_fall, deploy_dir=tmp_path)
    assert rec_sh.exists()
    assert fall_sh.exists()
    assert "A4" in rec_sh.read_text(encoding="utf-8")
    assert "A1" in fall_sh.read_text(encoding="utf-8")
