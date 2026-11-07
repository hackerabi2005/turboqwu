"""Unit tests for kvpipe doctor: tier mapping, arm support, and fixtures."""

import pytest

from kvpipe.config import ArmConfig, CompressionConfig
from kvpipe.doctor import check_arm_support, classify_tier


@pytest.fixture
def b200_env():
    return {
        "detected_tier": "T1",
        "gpu_available": True,
        "gpus": [{"name": "NVIDIA B200", "compute_cap": "10.0", "total_memory_mb": 180000}],
        "packages": {"vllm": "0.7.0", "sglang": "0.4.0", "tensorrt_llm": "0.18.0", "torch": "2.6.0"},
    }


@pytest.fixture
def rtx5090_env():
    return {
        "detected_tier": "T2",
        "gpu_available": True,
        "gpus": [{"name": "NVIDIA GeForce RTX 5090", "compute_cap": "12.0", "total_memory_mb": 32000}],
        "packages": {"vllm": "0.7.0", "sglang": "0.4.0", "tensorrt_llm": "0.18.0", "torch": "2.6.0"},
    }


@pytest.fixture
def h100_env():
    return {
        "detected_tier": "T3",
        "gpu_available": True,
        "gpus": [{"name": "NVIDIA H100 80GB HBM3", "compute_cap": "9.0", "total_memory_mb": 80000}],
        "packages": {"vllm": "0.7.0", "sglang": "0.4.0", "tensorrt_llm": "0.18.0", "torch": "2.6.0"},
    }


@pytest.fixture
def rtx4090_env():
    return {
        "detected_tier": "T4",
        "gpu_available": True,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "compute_cap": "8.9", "total_memory_mb": 24000}],
        "packages": {"vllm": "0.7.0", "sglang": "0.4.0", "tensorrt_llm": "0.18.0", "torch": "2.6.0"},
    }


@pytest.fixture
def no_gpu_env():
    return {
        "detected_tier": "None",
        "gpu_available": False,
        "gpus": [],
        "packages": {"torch": "2.13.0"},
    }


def test_tier_classification():
    assert classify_tier({"gpus": [{"compute_cap": "10.0"}]}) == "T1"
    assert classify_tier({"gpus": [{"compute_cap": "10.3"}]}) == "T1"
    assert classify_tier({"gpus": [{"compute_cap": "12.0"}]}) == "T2"
    assert classify_tier({"gpus": [{"compute_cap": "12.1"}]}) == "T2"
    assert classify_tier({"gpus": [{"compute_cap": "9.0"}]}) == "T3"
    assert classify_tier({"gpus": [{"compute_cap": "8.9"}]}) == "T4"
    assert classify_tier({"gpus": [{"compute_cap": "8.0"}]}) == "T4"
    assert classify_tier({"gpus": []}) == "None"


def test_trtllm_support_rules(b200_env, rtx5090_env):
    arm_trt = ArmConfig(
        id="T1_TRT",
        engine="trtllm",
        model="Qwen/Qwen3-8B",
        compression=CompressionConfig(method="triattention", budget=3072),
    )
    # T1 supported
    ok, msg = check_arm_support(arm_trt, b200_env)
    assert ok is True
    assert "Supported" in msg

    # T2 not supported for TRT TriAttention
    ok, msg = check_arm_support(arm_trt, rtx5090_env)
    assert ok is False
    assert "TRT-LLM TriAttention is supported only on T1" in msg


def test_nvfp4_support_rules(b200_env, rtx5090_env, h100_env, rtx4090_env):
    arm_nvfp4 = ArmConfig(
        id="A2",
        engine="sglang",
        model="Qwen/Qwen3-8B-FP8",
        kv_dtype="nvfp4",
    )
    # T1 & T2 supported
    ok, _ = check_arm_support(arm_nvfp4, b200_env)
    assert ok is True

    ok, _ = check_arm_support(arm_nvfp4, rtx5090_env)
    assert ok is True

    # T3 not supported for NVFP4
    ok, msg = check_arm_support(arm_nvfp4, h100_env)
    assert ok is False
    assert "NVFP4 KV requires Blackwell" in msg

    # T4 not supported for NVFP4
    ok, msg = check_arm_support(arm_nvfp4, rtx4090_env)
    assert ok is False
    assert "NVFP4 KV requires Blackwell" in msg


def test_fp8_experimental_on_t4(rtx4090_env):
    arm_fp8 = ArmConfig(
        id="A1",
        engine="vllm",
        model="Qwen/Qwen3-8B",
        kv_dtype="fp8",
    )
    ok, msg = check_arm_support(arm_fp8, rtx4090_env)
    assert ok is True
    assert "experimental" in msg.lower()


def test_turboquant_hybrid_rejection(rtx4090_env):
    arm_gemma = ArmConfig(
        id="A7_gemma",
        engine="vllm",
        model="google/gemma-7b",
        kv_dtype="turboquant_4bit_nc",
    )
    ok, msg = check_arm_support(arm_gemma, rtx4090_env)
    assert ok is False
    assert "hybrid or sliding-window" in msg


def test_stub_engine_always_supported(no_gpu_env):
    arm_stub = ArmConfig(
        id="A0_stub",
        engine="stub",
        model="Qwen/Qwen3-8B",
        kv_dtype="auto",
    )
    ok, msg = check_arm_support(arm_stub, no_gpu_env)
    assert ok is True
    assert "stub" in msg
