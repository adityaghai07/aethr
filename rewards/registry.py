"""
Reward plugin system for Aethr.

To add a custom reward function:
  1. Create a class that extends RewardPlugin in rewards/plugins/
  2. Implement async score() — return a RewardResult
  3. Add it to ACTIVE_PLUGINS in config.py

To remove or disable a plugin:
  - Set enabled=False or remove it from ACTIVE_PLUGINS

The composite reward is a weighted average of all enabled plugins.
Hard violations (violated=True) immediately flag the response for regeneration.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RewardResult:
    score: float                    # −1.0 to 1.0
    details: dict = field(default_factory=dict)   # breakdown for logging/wandb
    violated: bool = False          # True → response must be blocked/regenerated


class RewardPlugin(ABC):
    """
    Base class for all reward functions.
    Subclass this and implement score().
    """
    name: str = "unnamed"
    weight: float = 1.0     # relative contribution to composite reward
    enabled: bool = True

    @abstractmethod
    async def score(
        self,
        prompt: str,
        response: str,
        history: list[dict],
    ) -> RewardResult:
        """
        Score the assistant's response given the prompt and conversation history.

        prompt   — the most recent user message
        response — the assistant's response to score
        history  — full conversation history as [{role, content}]
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight}, enabled={self.enabled})"


# ── Registry ──────────────────────────────────────────────────────────────────
# Populated by each plugin module at import time via register()

_REGISTRY: dict[str, RewardPlugin] = {}


def register(plugin: RewardPlugin) -> RewardPlugin:
    """Register a plugin instance. Called at module level in each plugin file."""
    _REGISTRY[plugin.name] = plugin
    return plugin


def get_plugin(name: str) -> RewardPlugin:
    if name not in _REGISTRY:
        raise KeyError(f"Reward plugin '{name}' not found. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_plugins() -> list[RewardPlugin]:
    return list(_REGISTRY.values())
