"""Base classes and interfaces for inference engine adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kvpipe.config import ArmConfig


class EngineHandle(ABC):
    """Active instance handle for a launched inference server."""

    @property
    @abstractmethod
    def base_url(self) -> str:
        """OpenAI-compatible base URL (e.g., http://localhost:8000/v1)."""

    @abstractmethod
    def kv_capacity_tokens(self) -> int:
        """KV cache capacity in total tokens reported by the engine at startup."""

    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Scraped engine performance & runtime metrics."""

    @abstractmethod
    def eviction_events(self) -> tuple[int, int] | None:
        """Return (received_events, applied_events) or None if non-eviction arm."""

    @abstractmethod
    def effective_kv_dtype(self) -> str:
        """Effective KV cache dtype reported by the engine at runtime."""

    @abstractmethod
    def stop(self) -> None:
        """Cleanly terminate the server and release GPU memory."""


class Engine(ABC):
    """Abstract engine launcher and validator."""

    @abstractmethod
    def supports(self, arm: ArmConfig, env: dict[str, Any]) -> tuple[bool, str]:
        """Check if this engine can execute the requested arm configuration in this environment."""

    @abstractmethod
    def launch(self, arm: ArmConfig, port: int = 8000, log_dir: str = "artifacts/logs") -> EngineHandle:
        """Launch the engine server and wait until healthy."""
