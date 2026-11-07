"""vLLM engine adapter: process management, health polling, log & metrics parsing."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from kvpipe.config import ArmConfig
from kvpipe.engines.base import Engine, EngineHandle


class VLLMHandle(EngineHandle):
    def __init__(
        self,
        arm: ArmConfig,
        proc: subprocess.Popen,
        port: int,
        log_file: Path,
        capacity_tokens: int,
        effective_dtype: str,
    ):
        self._arm = arm
        self._proc = proc
        self._port = port
        self._log_file = log_file
        self._capacity_tokens = capacity_tokens
        self._effective_dtype = effective_dtype

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def kv_capacity_tokens(self) -> int:
        return self._capacity_tokens

    def metrics(self) -> dict[str, Any]:
        metrics_dict: dict[str, Any] = {}
        try:
            resp = httpx.get(f"http://127.0.0.1:{self._port}/metrics", timeout=3.0)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            metrics_dict[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
        except Exception:
            pass
        return metrics_dict

    def eviction_events(self) -> tuple[int, int] | None:
        """Parse TriAttention log for received and applied eviction events (PLAN.md P5.3)."""
        if self._arm.compression.method != "triattention":
            return None

        received_total = 0
        applied_total = 0
        if self._log_file.exists():
            with open(self._log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Look for pattern: "[TriAttention] received X events, applied Y events"
                    m = re.search(r"received\s+(\d+)\s+events?,\s+applied\s+(\d+)\s+events?", line, re.IGNORECASE)
                    if m:
                        received_total += int(m.group(1))
                        applied_total += int(m.group(2))
        return (received_total, applied_total)

    def effective_kv_dtype(self) -> str:
        return self._effective_dtype

    def stop(self) -> None:
        if self._proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(self._proc.pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                except Exception:
                    self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class VLLMEngine(Engine):
    """vLLM engine launcher."""

    def supports(self, arm: ArmConfig, env: dict[str, Any]) -> tuple[bool, str]:
        from kvpipe.doctor import check_arm_support
        return check_arm_support(arm, env)

    def build_command(self, arm: ArmConfig, port: int) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            arm.model,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(arm.serve.gpu_memory_utilization),
        ]
        if arm.serve.max_model_len:
            cmd.extend(["--max-model-len", str(arm.serve.max_model_len)])

        # KV cache dtype
        cmd.extend(["--kv-cache-dtype", arm.kv_dtype])

        # Skip layers
        if arm.skip_layers:
            cmd.extend(["--kv-cache-dtype-skip-layers", arm.skip_layers])

        # Extra args
        cmd.extend(arm.serve.extra_args)
        return cmd

    def launch(self, arm: ArmConfig, port: int = 8000, log_dir: str = "artifacts/logs") -> EngineHandle:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"vllm_{arm.id}_{int(time.time())}.log"

        cmd = self.build_command(arm, port)
        env = os.environ.copy()

        # TriAttention plugin environment configuration (PLAN.md P5.2)
        if arm.compression.method == "triattention":
            if arm.compression.stats_path:
                env["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = str(Path(arm.compression.stats_path).resolve())
            if arm.compression.budget:
                env["TRIATTN_RUNTIME_KV_BUDGET"] = str(arm.compression.budget)
            if arm.compression.protect_prompt:
                env["TRIATTN_PROTECT_PROMPT"] = "1"

        log_fd = open(log_file, "w", encoding="utf-8")
        kwargs: dict[str, Any] = {
            "stdout": log_fd,
            "stderr": subprocess.STDOUT,
            "env": env,
        }
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid

        proc = subprocess.Popen(cmd, **kwargs)

        # Poll /health until server is ready
        health_url = f"http://127.0.0.1:{port}/health"
        start_time = time.time()
        timeout = 180  # 3 minutes startup budget
        healthy = False
        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                log_fd.close()
                raise RuntimeError(f"vLLM server exited prematurely with code {proc.returncode}. See {log_file}")
            try:
                r = httpx.get(health_url, timeout=1.0)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(1.0)

        log_fd.flush()
        if not healthy:
            proc.kill()
            log_fd.close()
            raise TimeoutError(f"vLLM server failed to become healthy within {timeout}s. See {log_file}")

        # Parse log for KV cache capacity and effective KV dtype (P2.1)
        capacity_tokens, effective_dtype = self._parse_startup_log(log_file, arm)

        return VLLMHandle(
            arm=arm,
            proc=proc,
            port=port,
            log_file=log_file,
            capacity_tokens=capacity_tokens,
            effective_dtype=effective_dtype,
        )

    def _parse_startup_log(self, log_path: Path, arm: ArmConfig) -> tuple[int, str]:
        capacity = 32768  # Fallback
        effective_dtype = arm.kv_dtype

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Parse KV capacity tokens: "# GPU blocks: X ... KV cache size: Y tokens"
                m_cap = re.search(r"KV cache size:\s*(\d+)\s*tokens", content, re.IGNORECASE)
                if m_cap:
                    capacity = int(m_cap.group(1))

                # Parse effective KV dtype
                m_dt = re.search(r"kv_cache_dtype:\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
                if m_dt:
                    effective_dtype = m_dt.group(1).lower()

        return capacity, effective_dtype
