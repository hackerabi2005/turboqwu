"""Pydantic schemas and loaders for arms, suites, gates, and compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field


class CompressionConfig(BaseModel):
    method: str = "none"  # "triattention", "kvzap", "turboquant", "none"
    budget: Optional[int] = None
    protect_prompt: bool = True
    stats_path: Optional[str] = None
    beta: int = 128
    eviction_mode: str = "union"  # union, per_head, per_layer_perhead


class ServeConfig(BaseModel):
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = 32768
    extra_args: List[str] = Field(default_factory=list)


class ArmConfig(BaseModel):
    id: str
    role: str = "candidate"  # "reference", "default_recommendation", "reasoning_candidate", "negative_control", "canary", "fallback", etc.
    engine: str = "stub"  # "vllm", "sglang", "trtllm", "stub", "hf"
    model: str = "Qwen/Qwen3-8B"
    model_revision: Optional[str] = None
    weights_dtype: str = "bf16"
    kv_dtype: str = "auto"  # "auto", "fp8", "nvfp4", "turboquant_4bit_nc", etc.
    skip_layers: Optional[str] = None  # "sliding_window", "first2_last2", etc.
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    expected_notes: Optional[str] = None


class SuiteConfig(BaseModel):
    id: str
    name: str
    dataset_id: str
    revision: Optional[str] = "main"
    split: str = "test"
    sample_count: int = 30
    samples_per_problem: int = 8
    seed: int = 42
    reasoning_mode: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 32768


class GateThresholds(BaseModel):
    margin_standard: float = 2.0
    margin_strict: float = 1.0
    g1_golden_count: int = 20
    g1_corrupt_delta_max: float = 0.5
    g1_extraction_delta_min: float = -1.0
    g2_reasoning_margin: float = 2.0
    g3_retrieval_margin: float = 1.0
    g4_canary_min_drop: float = 10.0
    g5_compression_min: float = 1.8
    g6_throughput_ratio_min: float = 0.70
    g7_repeat_std_max: float = 1.5


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_arm_config(path: Path | str) -> ArmConfig:
    data = load_yaml(path)
    return ArmConfig.model_validate(data)


def load_suite_config(path: Path | str) -> SuiteConfig:
    data = load_yaml(path)
    return SuiteConfig.model_validate(data)


def load_gates_config(path: Path | str) -> GateThresholds:
    data = load_yaml(path)
    return GateThresholds.model_validate(data)
