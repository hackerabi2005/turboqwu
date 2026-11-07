"""Unit tests for Phase 8: Regression verification and production canary rollout monitoring."""

from kvpipe.config import ArmConfig
from kvpipe.engines.stub import StubEngine
from kvpipe.regress import evaluate_canary_rollout_metrics, run_regression_check


def test_regression_check_with_stub(tmp_path):
    stub = StubEngine()
    arm_a1 = ArmConfig(id="A1", engine="stub", model="Qwen/Qwen3-8B", kv_dtype="fp8")
    handle = stub.launch(arm_a1, port=8997)

    try:
        res = run_regression_check(
            arm=arm_a1,
            base_url=handle.base_url,
            golden_file="tests/golden.jsonl",
            previous_report_path="artifacts/report.json",
            runs_dir=tmp_path / "runs",
        )
        assert res["overall_pass"] is True
        assert res["golden_passed"] == 20
        assert res["canary_ok"] is True
    finally:
        handle.stop()


def test_canary_rollout_monitoring():
    # 1. Healthy rollout case
    healthy = evaluate_canary_rollout_metrics(
        baseline_integrity_fail_rate=0.0,
        canary_integrity_fail_rate=0.2,  # delta = +0.2 <= 0.5
        baseline_p95_tpot=15.0,
        canary_p95_tpot=16.5,
        tpot_threshold_ms=30.0,
    )
    assert healthy["status"] == "HEALTHY"
    assert healthy["rollback_needed"] is False

    # 2. Rollback triggered by integrity degradation (> 0.5 points)
    unhealthy_integrity = evaluate_canary_rollout_metrics(
        baseline_integrity_fail_rate=0.0,
        canary_integrity_fail_rate=0.8,  # delta = +0.8 > 0.5
        baseline_p95_tpot=15.0,
        canary_p95_tpot=16.0,
        tpot_threshold_ms=30.0,
    )
    assert unhealthy_integrity["status"] == "ROLLBACK_TRIGGERED"
    assert unhealthy_integrity["rollback_needed"] is True
    assert unhealthy_integrity["integrity_breach"] is True

    # 3. Rollback triggered by p95 TPOT breach
    unhealthy_latency = evaluate_canary_rollout_metrics(
        baseline_integrity_fail_rate=0.0,
        canary_integrity_fail_rate=0.1,
        baseline_p95_tpot=15.0,
        canary_p95_tpot=35.0,  # > 30.0 ms target
        tpot_threshold_ms=30.0,
    )
    assert unhealthy_latency["status"] == "ROLLBACK_TRIGGERED"
    assert unhealthy_latency["rollback_needed"] is True
    assert unhealthy_latency["tpot_breach"] is True
