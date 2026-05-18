"""
Auto-discovers and loads reward plugins from rewards/plugins/.

Importing a plugin module triggers its register() call at module level,
which adds it to the registry. This file replaces the manual import list
in config.get_active_plugins().

Usage:
    from rewards.loader import get_active_plugins
    plugins = get_active_plugins(["rule_based", "llm_judge"])
"""
import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rewards.registry import RewardPlugin

import rewards.plugins as _plugins_pkg
import rewards.models as _models_pkg

logger = logging.getLogger(__name__)

_discovered = False


def _discover_package(pkg, prefix: str) -> None:
    """Import all non-private modules in a package to trigger register() calls."""
    for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
        if module_name.startswith("_") or module_name == "server":
            continue  # skip __init__, private, and the HTTP server
        full = f"{prefix}.{module_name}"
        try:
            importlib.import_module(full)
            logger.debug(f"Loaded reward plugin module: {full}")
        except Exception as e:
            logger.error(f"Failed to import {full}: {e}")


def autodiscover() -> None:
    """
    Import every module under rewards/plugins/ and rewards/models/ once.
    Each module calls register() at import time, which populates the registry.
    Idempotent — safe to call multiple times.
    """
    global _discovered
    if _discovered:
        return
    _discover_package(_plugins_pkg, "rewards.plugins")
    _discover_package(_models_pkg, "rewards.models")
    _discovered = True


def get_active_plugins(active_names: list[str]) -> list["RewardPlugin"]:
    """
    Auto-discover all plugins, then return instances for the requested names.
    Unknown names are skipped with a warning rather than raising, so a typo
    in config doesn't crash the whole bot.
    """
    autodiscover()
    from rewards.registry import _REGISTRY

    plugins = []
    for name in active_names:
        plugin = _REGISTRY.get(name)
        if plugin is None:
            logger.warning(
                f"Reward plugin '{name}' listed in ACTIVE_PLUGIN_NAMES but not found "
                f"after auto-discovery. Registered plugins: {list(_REGISTRY)}"
            )
        else:
            plugins.append(plugin)
    return plugins
