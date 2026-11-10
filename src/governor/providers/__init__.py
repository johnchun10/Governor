"""Model providers. ``get_provider(name)`` builds (and caches) one per process."""

from __future__ import annotations

import threading

from agent_core import Provider, ProviderError

__all__ = ["PROVIDERS", "Provider", "ProviderError", "get_provider", "reset_providers"]

PROVIDERS = ("bedrock", "mantle", "anthropic", "openai")

_cache: dict[str, Provider] = {}
_lock = threading.Lock()


def get_provider(name: str) -> Provider:
    with _lock:
        if name not in _cache:
            if name == "bedrock":
                from .bedrock import BedrockProvider

                _cache[name] = BedrockProvider()
            elif name == "mantle":
                from .mantle import MantleProvider

                _cache[name] = MantleProvider()
            elif name == "anthropic":
                from .anthropic_api import AnthropicProvider

                _cache[name] = AnthropicProvider()
            elif name == "openai":
                from .openai_api import OpenAIProvider

                _cache[name] = OpenAIProvider()
            else:
                raise ProviderError(f"unknown provider {name!r}", hint=f"use one of: {', '.join(PROVIDERS)}")
        return _cache[name]


def reset_providers() -> None:
    """Drop cached clients, so the next call picks up new credentials (after a login)."""
    with _lock:
        _cache.clear()
