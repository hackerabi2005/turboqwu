"""kvpipe doctor: Hardware & inference engine environment discovery and tier selection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from kvpipe.config import ArmConfig


def get_installed_package_version(pkg_name: str) -> Optional[str]:
    """Return installed package version using importlib.metadata if present."""
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


def detect_environment(env_file: Optional[Path | str] = None) -> dict[str, Any]:
    """Detect local GPU, CUDA, and package environment, or load from cached env_file."""
    if env_file and Path(env_file).exists():
        with open(env_file, "r", encoding="utf-8") as f:
            return json.load(f)

    env: dict[str, Any] = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "gpu_available": False,
        "gpus": [],
        "cuda_available_torch": False,
        "nvcc_version": None,
        "packages": {},
        "disk_free_gb": None,
        "detected_tier": "None",
    }

    # Python packages
    target_pkgs = ["torch", "torchvision", "transformers", "vllm", "sglang", "tensorrt_llm", "flash-attn", "kvpress"]
    for pkg in target_pkgs:
        env["packages"][pkg] = get_installed_package_version(pkg)

    # PyTorch CUDA
    try:
        import torch
        env["cuda_available_torch"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_available"] = True
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                env["gpus"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_cap": f"{cap[0]}.{cap[1]}",
                    "total_memory_mb": round(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024), 2),
                })
    except Exception:
        pass

    # nvidia-smi fallback if torch has no cuda
    if not env["gpus"] and shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for idx, line in enumerate(res.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        env["gpus"].append({
                            "id": idx,
                            "name": parts[0],
                            "compute_cap": parts[1],
                            "total_memory_mb": float(parts[2]),
                            "driver_version": parts[3],
                        })
                if env["gpus"]:
                    env["gpu_available"] = True
        except Exception:
            pass

    # nvcc
    if shutil.which("nvcc"):
        try:
            res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            for line in res.stdout.splitlines():
                if "release" in line:
                    env["nvcc_version"] = line.strip()
                    break
        except Exception:
            pass

    # Disk free
    try:
        usage = shutil.disk_usage(os.getcwd())
        env["disk_free_gb"] = round(usage.free / (1024**3), 2)
    except Exception:
        pass

    env["detected_tier"] = classify_tier(env)
    return env


def classify_tier(env: dict[str, Any]) -> str:
    """Classify environment into T1, T2, T3, T4, or None (Section 4)."""
    gpus = env.get("gpus", [])
    if not gpus:
        return "None"

    # Evaluate best GPU compute capability
    best_tier = "None"
    for g in gpus:
        cap_str = str(g.get("compute_cap", "0.0"))
        try:
            major, minor = map(int, cap_str.split("."))
        except Exception:
            continue

        # T1: Datacenter Blackwell (SM100/103, e.g. B200)
        if major == 10 and minor in (0, 3, 7):
            return "T1"
        # T2: Workstation or consumer Blackwell (SM120/121, e.g. RTX 5090, RTX PRO 6000, GB10)
        elif major == 12 or (major == 10 and minor not in (0, 3, 7)):
            return "T2"
        # T3: Hopper (SM90, e.g. H100, H200)
        elif major == 9:
            best_tier = "T3"
        # T4: Ada or Ampere (SM80, SM86, SM89, e.g. RTX 4090, RTX 3090, A100)
        elif major == 8:
            if best_tier not in ("T1", "T2", "T3"):
                best_tier = "T4"
        # T5: Turing or Volta (SM70, SM75, e.g. NVIDIA T4, V100)
        elif major == 7:
            if best_tier not in ("T1", "T2", "T3", "T4"):
                best_tier = "T5"
    return best_tier


def check_arm_support(arm: ArmConfig, env: dict[str, Any]) -> Tuple[bool, str]:
    """Determine whether an arm can run in the current environment with rationale (Section 3.2, 4, P1.4)."""
    # Stub engine is always supported (used for CI/CD, local testing, verification without GPU)
    if arm.engine == "stub":
        return True, "Supported (stub engine)"

    tier = env.get("detected_tier", "None")
    pkgs = env.get("packages", {})

    # Check engine installation
    if arm.engine == "llamacpp":
        if not shutil.which("llama-server") and not os.environ.get("LLAMA_SERVER_BIN"):
            return False, "Engine 'llama-server' binary not found in PATH"
        return True, "Supported (llama.cpp engine)"
    if arm.engine == "vllm" and not pkgs.get("vllm"):
        return False, "Engine 'vllm' is not installed"
    if arm.engine == "sglang" and not pkgs.get("sglang"):
        return False, "Engine 'sglang' is not installed"
    if arm.engine == "trtllm" and not pkgs.get("tensorrt_llm"):
        return False, "Engine 'tensorrt_llm' is not installed"
    if arm.engine == "hf" and arm.compression.method == "kvzap" and not pkgs.get("kvpress"):
        return False, "Package 'kvpress' is not installed for KVzap arm"

    # Hardware tier check
    if tier == "None":
        return False, "No compatible NVIDIA GPU found in environment"

    # NVFP4 check: needs SM12x or SM100/103 and a GQA or sparse-MLA model (no hybrid)
    if arm.kv_dtype == "nvfp4":
        if tier not in ("T1", "T2"):
            return False, f"NVFP4 KV requires Blackwell (T1 or T2, SM100/103/120/121), detected tier is {tier}"
        model_lower = arm.model.lower()
        if any(h in model_lower for h in ("jamba", "mamba", "zamba", "qwen3.5", "qwen3_5")):
            return False, "NVFP4 KV requires a GQA or sparse-MLA model (hybrid architectures unsupported)"

    # TensorRT-LLM TriAttention check: SM100/103/107 and GQA/MHA/MQA only (no MLA, no hybrid)
    if arm.engine == "trtllm" and arm.compression.method == "triattention":
        if tier != "T1":
            return False, f"TRT-LLM TriAttention is supported only on T1 (SM100/103/107), detected tier is {tier}"
        model_lower = arm.model.lower()
        if any(arch in model_lower for arch in ("deepseek-v2", "deepseek-v3", "deepseek-r1", "kimi", "jamba", "mamba", "qwen3.5", "qwen3_5")):
            if "distill" not in model_lower:
                return False, "TRT-LLM TriAttention requires GQA, MHA, or MQA architecture (MLA and hybrid models are unsupported)"

    # FP8 KV check
    if arm.kv_dtype == "fp8":
        if tier == "T4":
            return True, "Supported (T4 FP8 KV is experimental; test with care)"
        if tier == "T5":
            return False, "Hardware FP8 KV cache is not supported on Turing/Volta (SM70/75; requires SM89+)"

    # TurboQuant check (sliding window / hybrid architecture check)
    if "turboquant" in arm.kv_dtype or arm.compression.method == "turboquant":
        model_lower = arm.model.lower()
        if any(h in model_lower for h in ("gemma", "jamba", "mamba", "zamba", "qwen3.5", "qwen3_5")):
            return False, "TurboQuant does not support hybrid or sliding-window architectures"

    return True, "Supported"


def run_doctor(env_file: Optional[Path | str] = None) -> dict[str, Any]:
    """Execute kvpipe doctor check and report environment and arm support status."""
    env = detect_environment(env_file)
    from kvpipe.config import load_arm_config

    arms_dir = Path("configs/arms")
    arm_reports: list[dict[str, Any]] = []
    if arms_dir.exists():
        for f in sorted(arms_dir.glob("*.yaml")):
            arm = load_arm_config(f)
            supported, reason = check_arm_support(arm, env)
            arm_reports.append({
                "id": arm.id,
                "role": arm.role,
                "engine": arm.engine,
                "kv_dtype": arm.kv_dtype,
                "supported": supported,
                "reason": reason,
            })

    return {
        "tier": env.get("detected_tier", "None"),
        "gpu_available": env.get("gpu_available", False),
        "gpus": env.get("gpus", []),
        "python": env.get("python_version"),
        "packages": env.get("packages", {}),
        "arms": arm_reports,
    }
