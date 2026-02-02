"""Plugin system for RAG OS."""

from rag_os.plugins.base import (
    Plugin,
    PluginInfo,
    PluginManager,
    PluginRegistry,
)
from rag_os.plugins.hooks import (
    HookType,
    Hook,
    HookManager,
)
from rag_os.plugins.loader import (
    PluginLoader,
    discover_plugins,
)

__all__ = [
    "Plugin",
    "PluginInfo",
    "PluginManager",
    "PluginRegistry",
    "HookType",
    "Hook",
    "HookManager",
    "PluginLoader",
    "discover_plugins",
]
