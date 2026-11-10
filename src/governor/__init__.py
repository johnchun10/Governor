"""governor: the concrete agent harness built on ``agent_core``.

Providers (Bedrock, Mantle, Anthropic, OpenAI), the per-provider model catalog,
settings and login, built-in tools, agent profiles, context providers,
permissions, sinks, the session and its commands and modes, the terminal UI and
the CLI. Feature work happens here; the primitives it builds on live in
``agent_core``.
"""

from .config import Settings
from .context import AgentsMdContext, EnvironmentContext, default_context_providers
from .models import Selection, catalog, resolve
from .permissions import Permissions
from .profiles import load_profiles, parse_profile
from .runtime import build_runtime, build_session
from .session import Session
from .tools import default_registry

__all__ = [
    "AgentsMdContext",
    "EnvironmentContext",
    "Permissions",
    "Selection",
    "Session",
    "Settings",
    "build_runtime",
    "build_session",
    "catalog",
    "default_context_providers",
    "default_registry",
    "load_profiles",
    "parse_profile",
    "resolve",
]
