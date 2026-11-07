"""Unit tests for Phase 5: TriAttention calibration, eviction events, canary C1, and R1 reproduction."""

from kvpipe.calib import resolve_calibration_stats
from kvpipe.config import ArmConfig, CompressionConfig, SuiteConfig
from kvpipe.data.loaders import load_suite_data
from kvpipe.engines.stub import StubEngine
from kvpipe.run.sampler import evaluate_suite_on_arm
from kvpipe.stats import paired_bootstrap_ci


def test_calibration_stats_resolution():
    stats_path = resolve_calibration_stats("Qwen/Qwen3-8B")
    assert stats_path.exists()
    assert stats_path.name == "qwen3_8b.pt"


def test_eviction_events_and_canary_c1(tmp_path):
    stub = StubEngine()

    arm_c1 = ArmConfig(
        id="C1",
        engine="stub",
        model="Qwen/Qwen3-8B",
        kv_dtype="fp8",
        compression=CompressionConfig(method="triattention", budget=256),
    )
    arm_a4 = ArmConfig(
        id="A4",
        engine="stub",
        model="Qwen/Qwen3-8B",
        kv_dtype="fp8",
        compression=CompressionConfig(method="triattention", budget=3072),
    )

    suite = SuiteConfig(
        id="aime25",
        name="AIME 2025",
        dataset_id="test",
        sample_count=10,
        samples_per_problem=1,
    )
    items = load_suite_data(suite)

    # Launch Canary C1
    h_c1 = stub.launch(arm_c1, port=8991)
    try:
        results_c1 = evaluate_suite_on_arm(arm_c1, suite, h_c1.base_url, items[:10], runs_dir=tmp_path / "runs")
        ev_c1 = h_c1.eviction_events()
        assert ev_c1 is not None
        # Gate G4 check: applied events > 0
        assert ev_c1[1] > 0, "Eviction applied events must be > 0"
    finally:
        h_c1.stop()

    # Launch Target Arm A4
    h_a4 = stub.launch(arm_a4, port=8992)
    try:
        results_a4 = evaluate_suite_on_arm(arm_a4, suite, h_a4.base_url, items[:10], runs_dir=tmp_path / "runs")
    finally:
        h_a4.stop()

    # P5.5: Canary C1 must score clearly below budget 3072/4096 (CI excludes zero, drop > 10 points)
    ci = paired_bootstrap_ci(results_c1, results_a4, num_resamples=1000)
    assert ci["ci_upper"] < -10.0, f"Canary did not show clear degradation: CI upper {ci['ci_upper']}"


def test_paper_reproduction_r1_and_a5_comparison(tmp_path):
    stub = StubEngine()

    arm_a0 = ArmConfig(id="A0", engine="stub", model="Qwen/Qwen3-8B", kv_dtype="auto")
    arm_r1 = ArmConfig(
        id="R1",
        engine="stub",
        model="Qwen/Qwen3-8B",
        kv_dtype="auto",
        compression=CompressionConfig(method="triattention", budget=3072),
    )
    arm_a5 = ArmConfig(
        id="A5",
        engine="stub",
        model="Qwen/Qwen3-8B",
        kv_dtype="fp8",
        compression=CompressionConfig(method="triattention", budget=2048),
    )

    suite = SuiteConfig(
        id="aime25",
        name="AIME 2025",
        dataset_id="test",
        sample_count=10,
        samples_per_problem=1,
    )
    items = load_suite_data(suite)

    h_a0 = stub.launch(arm_a0, port=8993)
    try:
        res_a0 = evaluate_suite_on_arm(arm_a0, suite, h_a0.base_url, items[:10], runs_dir=tmp_path / "runs")
    finally:
        h_a0.stop()

    h_r1 = stub.launch(arm_r1, port=8994)
    try:
        res_r1 = evaluate_suite_on_arm(arm_r1, suite, h_r1.base_url, items[:10], runs_dir=tmp_path / "runs")
    finally:
        h_r1.stop()

    h_a5 = stub.launch(arm_a5, port=8995)
    try:
        res_a5 = evaluate_suite_on_arm(arm_a5, suite, h_a5.base_url, items[:10], runs_dir=tmp_path / "runs")
    finally:
        h_a5.stop()

    # P5.6 Reproduction check: R1 vs A0 agrees within 5 points
    ci_r1 = paired_bootstrap_ci(res_r1, res_a0, num_resamples=1000)
    assert abs(ci_r1["mean_delta"]) <= 5.0, f"R1 vs A0 delta {ci_r1['mean_delta']} exceeds 5 points"

    # Negative control A5 (budget 2048) scores lower than A0/R1 (E4 expects ~8 points drop)
    ci_a5 = paired_bootstrap_ci(res_a5, res_a0, num_resamples=1000)
    assert ci_a5["mean_delta"] < 0, "A5 budget 2048 did not score below baseline"
