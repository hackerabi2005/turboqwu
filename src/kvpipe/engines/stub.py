"""Stub OpenAI-compatible engine for testing sampler, scorers, stats, and reports without GPU."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kvpipe.config import ArmConfig
from kvpipe.engines.base import Engine, EngineHandle


class StubHandler(BaseHTTPRequestHandler):
    arm: ArmConfig | None = None
    applied_evictions: int = 0
    received_evictions: int = 0

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard HTTP logs during tests
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b'# HELP vllm_num_requests_running Running requests\nvllm_num_requests_running 1\n')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path.endswith("/chat/completions") or self.path.endswith("/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            messages = data.get("messages", [])
            prompt = messages[-1].get("content", "") if messages else data.get("prompt", "")

            response_text = self._generate_response(prompt)

            # Eviction event simulation for arms with compression
            if self.arm and self.arm.compression.method == "triattention":
                budget = self.arm.compression.budget or 3072
                # If prompt/output tokens exceed budget, simulate eviction
                self.received_evictions += 1
                if budget > 256:
                    self.applied_evictions += 1
                elif self.arm.id == "C1":  # Canary arm
                    self.applied_evictions += 1

            resp = {
                "id": "stub-cmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": self.arm.model if self.arm else "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
            }
            res_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.end_headers()
            self.wfile.write(res_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def _generate_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Deterministic responses for golden prompts
        if "what is 2 + 2" in prompt_lower:
            return "4"
        if "capital of france" in prompt_lower:
            return "Paris"
        if "multilingual test" in prompt_lower:
            return "Bonjour le monde"
        if "fibonacci snippet" in prompt_lower:
            return "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
        if "answer exact 42" in prompt_lower:
            return "42"

        # Check for deliberate corruption injection in tests
        if "__inject_corrupt__" in prompt:
            return "????????"
        if "__inject_empty__" in prompt:
            return ""
        if "__inject_no_box__" in prompt:
            return "The final answer is some calculation without any box."

        # Canary arm behavior: intentionally degraded accuracy
        if self.arm and self.arm.id == "C1":
            return "<think>Canary compressed cache degradation</think> The answer is \\boxed{99999}."

        # Needle in a haystack response
        needle_match = re.search(r"special key is ([a-zA-Z0-9_-]+)", prompt)
        if needle_match:
            return f"The special key is {needle_match.group(1)}."

        # Math / AIME / Math500 prompt: check if ground truth is embedded or hash-based
        gold_match = re.search(r"\[GOLD:([^\]]+)\]", prompt)
        if gold_match:
            gold = gold_match.group(1)
            # Control accuracy per arm based on published evidence E4
            arm_id = self.arm.id if self.arm else "A0"
            if arm_id == "A5":  # budget 2048 drops ~8 pts
                # Drop ~20% of answers
                if hash(prompt) % 5 == 0:
                    return "<think>TriAttention 2048 budget eviction mistake</think> \\boxed{0}"
            return f"<think>Step by step reasoning for problem.</think> Therefore, the answer is \\boxed{{{gold}}}."

        # Default fallback answer
        return "<think>Reasoning...</think> The answer is \\boxed{42}."


class StubEngineHandle(EngineHandle):
    def __init__(self, arm: ArmConfig, server: ThreadingHTTPServer, port: int):
        self._arm = arm
        self._server = server
        self._port = port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    def kv_capacity_tokens(self) -> int:
        # Realistic capacity ratios per PLAN.md
        # A0 (BF16): 32768 tokens (baseline 1.0x)
        # A1 (FP8): 65536 tokens (2.0x)
        # A2 (NVFP4): 116508 tokens (3.56x)
        # A7 (TurboQuant 4bit): 111411 tokens (3.4x)
        if self._arm.kv_dtype == "fp8":
            return 65536
        elif self._arm.kv_dtype == "nvfp4":
            return 116508
        elif "turboquant" in self._arm.kv_dtype:
            return 111411
        return 32768

    def metrics(self) -> dict[str, Any]:
        return {
            "num_requests_running": 1,
            "gpu_cache_usage_sys": 0.55,
            "tpot_p95_ms": 14.2,
            "ttft_p95_ms": 42.0,
        }

    def eviction_events(self) -> tuple[int, int] | None:
        if self._arm.compression.method == "triattention":
            return (StubHandler.received_evictions, StubHandler.applied_evictions)
        return None

    def effective_kv_dtype(self) -> str:
        return self._arm.kv_dtype

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class StubEngine(Engine):
    """Stub engine implementation for tests and GPU-less environments."""

    def supports(self, arm: ArmConfig, env: dict[str, Any]) -> tuple[bool, str]:
        return True, "Supported (stub engine)"

    def launch(self, arm: ArmConfig, port: int = 8999, log_dir: str = "artifacts/logs") -> EngineHandle:
        StubHandler.arm = arm
        StubHandler.received_evictions = 0
        StubHandler.applied_evictions = 0

        server = ThreadingHTTPServer(("127.0.0.1", port), StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        return StubEngineHandle(arm, server, port)
