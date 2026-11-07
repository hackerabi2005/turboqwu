"""Unit tests for engine adapters (vLLM command building, StubEngine, golden prompt execution)."""

import json
from pathlib import Path

import httpx

from kvpipe.config import ArmConfig
from kvpipe.engines.stub import StubEngine
from kvpipe.engines.vllm import VLLMEngine


def test_vllm_command_building():
    engine = VLLMEngine()

    # A0 command
    arm_a0 = ArmConfig(
        id="A0",
        engine="vllm",
        model="Qwen/Qwen3-8B",
        kv_dtype="auto",
    )
    cmd_a0 = engine.build_command(arm_a0, port=8000)
    assert "--kv-cache-dtype" in cmd_a0
    assert "auto" in cmd_a0
    assert "--gpu-memory-utilization" in cmd_a0

    # A1 command with skip_layers
    arm_a1 = ArmConfig(
        id="A1",
        engine="vllm",
        model="Qwen/Qwen3-8B",
        kv_dtype="fp8",
        skip_layers="sliding_window",
    )
    cmd_a1 = engine.build_command(arm_a1, port=8001)
    assert "--kv-cache-dtype" in cmd_a1
    assert "fp8" in cmd_a1
    assert "--kv-cache-dtype-skip-layers" in cmd_a1
    assert "sliding_window" in cmd_a1


def test_stub_engine_baselines_capacity_and_golden():
    stub = StubEngine()

    arm_a0 = ArmConfig(id="A0", engine="stub", model="Qwen/Qwen3-8B", kv_dtype="auto")
    arm_a1 = ArmConfig(id="A1", engine="stub", model="Qwen/Qwen3-8B", kv_dtype="fp8")

    h0 = stub.launch(arm_a0, port=8980)
    h1 = stub.launch(arm_a1, port=8981)

    try:
        cap0 = h0.kv_capacity_tokens()
        cap1 = h1.kv_capacity_tokens()
        ratio = cap1 / cap0

        # Phase 2 Done-when: measured capacity ratio A1 over A0 is between 1.8x and 2.2x
        assert 1.8 <= ratio <= 2.2, f"Capacity ratio {ratio} not in [1.8, 2.2]"

        # Check health and metrics
        r0 = httpx.get(f"{h0.base_url.replace('/v1', '')}/health")
        assert r0.status_code == 200

        # Run golden prompts against h0 and h1
        golden_file = Path("tests/golden.jsonl")
        assert golden_file.exists()

        with open(golden_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 20

        # Test deterministic endpoint response
        for item in lines[:5]:
            resp = httpx.post(
                f"{h0.base_url}/chat/completions",
                json={"messages": [{"role": "user", "content": item["prompt"]}]},
            )
            assert resp.status_code == 200
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0

    finally:
        h0.stop()
        h1.stop()
