"""Unit tests for Phase 6: K/V profiling, position probes, fidelity signal, and SGLang adapter."""

from kvpipe.config import ArmConfig
from kvpipe.engines.sglang import SGLangEngine
from kvpipe.engines.stub import StubEngine
from kvpipe.probes import (
    measure_fidelity_signal,
    run_generated_memory_probe,
    run_prompt_depth_probes,
)
from kvpipe.profile import profile_kv_norms


def test_profile_kv_norms(tmp_path):
    report = profile_kv_norms("Qwen/Qwen3-8B", num_layers=36, output_dir=tmp_path)
    assert report["num_layers"] == 36
    assert len(report["layers"]) == 36
    assert (tmp_path / "Qwen_Qwen3-8B.json").exists()
    assert report["summary"]["max_ratio"] >= 1.0


def test_sglang_command_building():
    engine = SGLangEngine()
    arm_a2 = ArmConfig(
        id="A2",
        engine="sglang",
        model="Qwen/Qwen3-8B-FP8",
        kv_dtype="nvfp4",
        serve={"extra_args": ["--enable-metrics"]},
    )
    cmd = engine.build_command(arm_a2, port=8100)
    assert "--kv-cache-dtype" in cmd
    assert "nvfp4" in cmd
    assert "--enable-metrics" in cmd


def test_probes_and_fidelity_with_stub(tmp_path):
    stub = StubEngine()
    arm_a0 = ArmConfig(id="A0", engine="stub", model="Qwen/Qwen3-8B")
    h = stub.launch(arm_a0, port=8989)

    try:
        # Prompt depth probes
        probe_res = run_prompt_depth_probes(h.base_url, arm_a0.model)
        assert probe_res["all_depths_passed"] is True

        # Generated memory probe
        mem_res = run_generated_memory_probe(h.base_url, arm_a0.model, budget=3072)
        assert "eviction_risk" in mem_res

        # Fidelity signal against itself (100% agreement expected)
        prompts = ["What is 2 + 2?", "Capital of France?"]
        fid_res = measure_fidelity_signal(h.base_url, arm_a0.model, h.base_url, arm_a0.model, prompts)
        assert fid_res["greedy_agreement_pct"] == 100.0
    finally:
        h.stop()
