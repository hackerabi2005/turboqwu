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

            # Eviction event simulation on class attributes for arms with compression
            if StubHandler.arm and StubHandler.arm.compression.method == "triattention":
                budget = StubHandler.arm.compression.budget or 3072
                StubHandler.received_evictions += 1
                if budget > 256 or StubHandler.arm.id == "C1":
                    StubHandler.applied_evictions += 1

            response_text = self._generate_response(prompt)

            # Support streaming SSE if requested
            if data.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                words = response_text.split(" ")
                for i, word in enumerate(words):
                    space = " " if i > 0 else ""
                    chunk = {
                        "id": "stub-cmpl-123",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": space + word}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                final_chunk = {
                    "id": "stub-cmpl-123",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
                return

            resp = {
                "id": "stub-cmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": StubHandler.arm.model if StubHandler.arm else "stub-model",
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
        if StubHandler.arm and StubHandler.arm.id == "C1":
            return "<think>Canary compressed cache degradation</think> The answer is \\boxed{99999}."

        # Needle in a haystack response
        needle_match = re.search(r"special key is ([a-zA-Z0-9_-]+)", prompt)
        if needle_match:
            return f"The special key is {needle_match.group(1)}."

        # Math / AIME / Math500 prompt: check if ground truth is embedded or hash-based
        gold_match = re.search(r"\[GOLD:([^\]]+)\]", prompt)
        if gold_match:
            gold = gold_match.group(1)
            arm_id = StubHandler.arm.id if StubHandler.arm else "A0"
            if arm_id == "A5":  # budget 2048 drops ~8 pts
                # Intentionally drop ~20% of problems (e.g. problem 1 and 5)
                if "problem 1:" in prompt.lower() or "problem 5:" in prompt.lower():
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
