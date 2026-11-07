"""Unit tests for Phase 4 performance harness."""

import pytest
from kvpipe.config import ArmConfig
from kvpipe.engines.stub import StubEngine
from kvpipe.run.perf import calculate_kv_bytes_per_token, run_performance_sweep


def test_calculate_kv_bytes_per_token():
    # 36 layers, 8 kv heads, 128 head dim
    # BF16 = 2 bytes: 2 * 36 * 8 * 128 * 2 = 147456 bytes
    assert calculate_kv_bytes_per_token(36, 8, 128, "bf16") == 147456
    # FP8 = 1 byte: 2 * 36 * 8 * 128 * 1 = 73728 bytes
    assert calculate_kv_bytes_per_token(36, 8, 128, "fp8") == 73728
    # NVFP4 = 0.5 byte: 2 * 36 * 8 * 128 * 0.5 = 36864 bytes
    assert calculate_kv_bytes_per_token(36, 8, 128, "nvfp4") == 36864


def test_performance_sweep_stub():
    stub = StubEngine()
    arm_a0 = ArmConfig(id="A0", engine="stub", model="Qwen/Qwen3-8B", kv_dtype="auto")
    handle = stub.launch(arm_a0, port=8987)

    try:
        report = run_performance_sweep(
            arm=arm_a0,
            base_url=handle.base_url,
            concurrencies=[1, 2],
            repeats=2,
        )
        assert report["arm_id"] == "A0"
        assert len(report["concurrency_sweep"]) == 2
        for pt in report["concurrency_sweep"]:
            assert pt["tokens_per_sec_mean"] > 0
            assert pt["tpot_p95_mean"] >= 0
    finally:
        handle.stop()
