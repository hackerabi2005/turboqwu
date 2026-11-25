"""SGLang engine adapter for Blackwell (SM100/103/120/121) NVFP4 KV and TriAttention (PLAN.md P6.5)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from kvpipe.config import ArmConfig
from kvpipe.engines.base import Engine, EngineHandle


class SGLangHandle(EngineHandle):
    def __init__(
        self,
        arm: ArmConfig,
        proc: subprocess.Popen,
        port: int,
        log_file: Path,
        capacity_tokens: int = 116508,
    ):
        self._arm = arm
        self._proc = proc
        self._port = port
        self._log_file = log_file
        self._capacity_tokens = capacity_tokens

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def kv_capacity_tokens(self) -> int:
        return self._capacity_tokens

    def metrics(self) -> dict[str, Any]:
        metrics_dict: dict[str, Any] = {}
        try:
            r = httpx.get(f"http://127.0.0.1:{self._port}/metrics", timeout=2.0)
            if r.status_code == 200:
                for line in r.text.splitlines():
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
        if self._arm.compression.method != "triattention":
            return None
        received_total = 0
        applied_total = 0
        if self._log_file.exists():
            import re
            with open(self._log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"received\s+(\d+)\s+events?,\s+applied\s+(\d+)\s+events?", line, re.IGNORECASE)
                    if m:
                        received_total += int(m.group(1))
                        applied_total += int(m.group(2))
        return (received_total, applied_total)

    def effective_kv_dtype(self) -> str:
        return self._arm.kv_dtype

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


class SGLangEngine(Engine):
    """SGLang engine launcher."""

    def supports(self, arm: ArmConfig, env: dict[str, Any]) -> tuple[bool, str]:
        from kvpipe.doctor import check_arm_support
        return check_arm_support(arm, env)

    def build_command(self, arm: ArmConfig, port: int) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            arm.model,
            "--port",
            str(port),
            "--mem-fraction-static",
            str(arm.serve.gpu_memory_utilization),
            "--kv-cache-dtype",
            arm.kv_dtype,
        ]
        cmd.extend(arm.serve.extra_args)
        return cmd

    def launch(self, arm: ArmConfig, port: int = 8000, log_dir: str = "artifacts/logs") -> EngineHandle:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"sglang_{arm.id}_{int(time.time())}.log"
        cmd = self.build_command(arm, port)

        log_fd = open(log_file, "w", encoding="utf-8")
        env = os.environ.copy()
        if arm.compression.method == "triattention":
            if arm.compression.stats_path:
                env["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = str(Path(arm.compression.stats_path).resolve())
            if arm.compression.budget:
                env["TRIATTN_RUNTIME_KV_BUDGET"] = str(arm.compression.budget)
            env["TRIATTN_RUNTIME_PROTECT_PREFILL"] = "true" if arm.compression.protect_prompt else "false"
        kwargs: dict[str, Any] = {"stdout": log_fd, "stderr": subprocess.STDOUT, "env": env}
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid

        proc = subprocess.Popen(cmd, **kwargs)

        health_url = f"http://127.0.0.1:{port}/health"
        start_t = time.time()
        timeout = 180
        healthy = False
        while time.time() - start_t < timeout:
            if proc.poll() is not None:
                log_fd.close()
                raise RuntimeError(f"SGLang process died with exit code {proc.returncode}. See {log_file}")
            try:
                if httpx.get(health_url, timeout=1.0).status_code == 200:
                    healthy = True
                    break
            except Exception:
                time.sleep(1.0)

        log_fd.flush()
        if not healthy:
            proc.kill()
            log_fd.close()
            raise TimeoutError(f"SGLang server failed to become healthy within {timeout}s.")

        return SGLangHandle(arm=arm, proc=proc, port=port, log_file=log_file)
