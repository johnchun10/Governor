"""What the loop needs to know about a model: where to send it and what it costs.

Which models exist, their ids and prices, live in Governor's model registry;
core only defines the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    provider: str
    """Name of the provider that serves this model (e.g. ``bedrock``)."""
    model_id: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    supports_cache: bool = True

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_tokens * self.cache_read_per_mtok
            + usage.cache_write_tokens * self.cache_write_per_mtok
        ) / 1_000_000
