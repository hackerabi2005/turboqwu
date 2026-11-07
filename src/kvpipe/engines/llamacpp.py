"""llama.cpp (llama-server) engine adapter supporting TurboQuant (-ctk/-ctv) and TriAttention flags."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from kvpipe.config import ArmConfig
from kvpipe.engines.base import Engine, EngineHandle


class LlamaCppHandle(EngineHandle):
    def __init__(
        self,
        arm: ArmConfig,
        proc: subprocess.Popen,
        port: int,
        log_file: Path,
        capacity_tokens: int,
        effective_dtype: str,
        kv_buffer_mb: float = 0.0,
    ):
        self._arm = arm
        self._proc = proc
        self._port = port
        self._log_file = log_file
        self._capacity_tokens = capacity_tokens
        self._effective_dtype = effective_dtype
        self._kv_buffer_mb = kv_buffer_mb

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def kv_capacity_tokens(self) -> int:
        return self._capacity_tokens

    def metrics(self) -> dict[str, Any]:
        return {"kv_buffer_mb": self._kv_buffer_mb}

    def eviction_events(self) -> tuple[int, int] | None:
        if self._arm.compression.method != "triattention":
            return None
        received_total = 0
        applied_total = 0
        if self._log_file.exists():
            with open(self._log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"triattention.*evicted\s+(\d+)\s+blocks", line, re.IGNORECASE)
                    if m:
                        received_total += 1
                        applied_total += int(m.group(1))
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


class LlamaCppEngine(Engine):
    """llama-server adapter with native -ctk/-ctv TurboQuant & TriAttention CLI support."""

    def __init__(self, binary: str = "llama-server"):
        self.binary = binary

    def supports(self, arm: ArmConfig, env: dict[str, Any]) -> tuple[bool, str]:
        if not shutil.which(self.binary) and not os.environ.get("LLAMA_SERVER_BIN"):
            return False, f"Binary '{self.binary}' not found in PATH"
        return True, "Supported (llama.cpp engine)"

    def _resolve_kv_types(self, kv_dtype: str) -> tuple[str, str]:
        dt = kv_dtype.lower()
        mapping = {
            "auto": ("f16", "f16"),
            "fp16": ("f16", "f16"),
            "fp8": ("q8_0", "q8_0"),
            "q8_0": ("q8_0", "q8_0"),
            "q4_0": ("q4_0", "q4_0"),
            "turboquant_k8v4": ("q8_0", "turbo4"),
            "turboquant_4bit_nc": ("turbo4", "turbo4"),
            "turboquant_3bit_nc": ("turbo3", "turbo3"),
            "turbo2": ("turbo2", "turbo2"),
            "turbo4": ("turbo4", "turbo4"),
        }
        return mapping.get(dt, (dt, dt))

    def build_command(self, arm: ArmConfig, port: int) -> list[str]:
        bin_path = os.environ.get("LLAMA_SERVER_BIN", self.binary)
        ctk, ctv = self._resolve_kv_types(arm.kv_dtype)
        ctx_len = arm.serve.max_model_len or 32768
        cmd = [
            bin_path,
            "-m",
            arm.model,
            "--port",
            str(port),
            "-c",
            str(ctx_len),
            "-fa",
            "on",
            "-ctk",
            ctk,
            "-ctv",
            ctv,
        ]
        if arm.compression.method == "triattention":
            if arm.compression.stats_path:
                cmd.extend(["--triattention-stats", str(arm.compression.stats_path)])
            if arm.compression.budget:
                cmd.extend(["--triattention-budget", str(arm.compression.budget)])
            if arm.compression.protect_prompt:
                cmd.extend(["--triattention-prefix-length", "2048"])
            cmd.extend(["--triattention-segments", "8"])

        cmd.extend(arm.serve.extra_args)
        return cmd

    def _parse_startup_log(self, log_path: Path, arm: ArmConfig) -> tuple[int, str, float]:
        capacity = arm.serve.max_model_len or 32768
        ctk, ctv = self._resolve_kv_types(arm.kv_dtype)
        effective_dtype = f"{ctk}/{ctv}" if ctk != ctv else ctk
        kv_buffer_mb = 0.0

        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                m_ctx = re.search(r"n_ctx\s*=\s*(\d+)", content)
                if m_ctx:
                    capacity = int(m_ctx.group(1))
                m_buf = re.search(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB", content, re.IGNORECASE)
                if m_buf:
                    kv_buffer_mb = float(m_buf.group(1))
        return capacity, effective_dtype, kv_buffer_mb

    def launch(self, arm: ArmConfig, port: int = 8080, log_dir: str = "artifacts/logs") -> EngineHandle:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"llamacpp_{arm.id}_{int(time.time())}.log"
        cmd = self.build_command(arm, port)
        log_fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_fd, stderr=subprocess.STDOUT)

        health_url = f"http://127.0.0.1:{port}/health"
        start_time = time.time()
        healthy = False
        while time.time() - start_time < 120:
            if proc.poll() is not None:
                log_fd.close()
                raise RuntimeError(f"llama-server exited prematurely with code {proc.returncode}")
            try:
                r = httpx.get(health_url, timeout=1.0)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(0.5)

        log_fd.flush()
        if not healthy:
            proc.kill()
            log_fd.close()
            raise TimeoutError(f"llama-server failed to become healthy within 120s. See {log_file}")

        cap, eff_dt, kv_mb = self._parse_startup_log(log_file, arm)
        return LlamaCppHandle(
            arm=arm,
            proc=proc,
            port=port,
            log_file=log_file,
            capacity_tokens=cap,
            effective_dtype=eff_dt,
            kv_buffer_mb=kv_mb,
        )
