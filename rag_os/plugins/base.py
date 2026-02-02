"""Plugin base classes for RAG OS."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class PluginState(Enum):
    """State of a plugin."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class PluginInfo:
    """Information about a plugin.

    Attributes:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        dependencies: Required plugin dependencies
        tags: Plugin tags for categorization
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }


class Plugin(ABC):
    """Abstract base class for RAG OS plugins.

    Plugins can extend RAG OS functionality by:
    - Adding new step types
    - Registering hooks for pipeline events
    - Providing custom storage backends
    - Adding evaluation metrics

    Usage:
        class MyPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(name="my-plugin", version="1.0.0")

            def activate(self, context: PluginContext) -> None:
                # Register steps, hooks, etc.
                pass
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin information."""
        pass

    @abstractmethod
    def activate(self, context: "PluginContext") -> None:
        """Activate the plugin.

        Args:
            context: Plugin context for registration
        """
        pass

    def deactivate(self) -> None:
        """Deactivate the plugin. Override if cleanup is needed."""
        pass

    def on_load(self) -> None:
        """Called when plugin is loaded. Override if needed."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded. Override if needed."""
        pass


@dataclass
class PluginContext:
    """Context passed to plugins for registration.

    Provides access to RAG OS systems for extending functionality.
    """
    step_registry: Any = None  # StepRegistry
    hook_manager: Any = None   # HookManager
    storage: Any = None        # Storage
    config: dict[str, Any] = field(default_factory=dict)

    def register_step(
        self,
        name: str,
        step_class: type,
        **metadata: Any,
    ) -> None:
        """Register a new step type.

        Args:
            name: Step name
            step_class: Step class
            **metadata: Additional metadata
        """
        if self.step_registry:
            self.step_registry.register(name, step_class, **metadata)

    def register_hook(
        self,
        hook_type: str,
        callback: Callable,
        priority: int = 100,
    ) -> None:
        """Register a hook callback.

        Args:
            hook_type: Type of hook
            callback: Callback function
            priority: Hook priority (lower = earlier)
        """
        if self.hook_manager:
            self.hook_manager.register(hook_type, callback, priority)


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._states: dict[str, PluginState] = {}

    def register(self, plugin: Plugin) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance
        """
        info = plugin.info
        self._plugins[info.name] = plugin
        self._states[info.name] = PluginState.UNLOADED

    def unregister(self, name: str) -> bool:
        """Unregister a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unregistered
        """
        if name in self._plugins:
            del self._plugins[name]
            del self._states[name]
            return True
        return False

    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin or None
        """
        return self._plugins.get(name)

    def get_state(self, name: str) -> PluginState | None:
        """Get plugin state.

        Args:
            name: Plugin name

        Returns:
            Plugin state or None
        """
        return self._states.get(name)

    def set_state(self, name: str, state: PluginState) -> None:
        """Set plugin state.

        Args:
            name: Plugin name
            state: New state
        """
        if name in self._states:
            self._states[name] = state

    def list_plugins(self) -> list[PluginInfo]:
        """Get list of all plugins."""
        return [p.info for p in self._plugins.values()]

    def list_by_state(self, state: PluginState) -> list[str]:
        """List plugins in a specific state.

        Args:
            state: Target state

        Returns:
            List of plugin names
        """
        return [name for name, s in self._states.items() if s == state]


class PluginManager:
    """Manager for loading and managing plugins."""

    def __init__(self, context: PluginContext | None = None):
        """Initialize plugin manager.

        Args:
            context: Plugin context for activation
        """
        self._registry = PluginRegistry()
        self._context = context or PluginContext()

    @property
    def registry(self) -> PluginRegistry:
        """Get the plugin registry."""
        return self._registry

    def install(self, plugin: Plugin) -> bool:
        """Install a plugin.

        Args:
            plugin: Plugin to install

        Returns:
            True if installed successfully
        """
        info = plugin.info

        # Check dependencies
        for dep in info.dependencies:
            dep_state = self._registry.get_state(dep)
            if dep_state is None or dep_state != PluginState.ACTIVE:
                return False

        self._registry.register(plugin)
        return True

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin.

        Args:
            name: Plugin name

        Returns:
            True if uninstalled
        """
        plugin = self._registry.get(name)
        if plugin is None:
            return False

        # Deactivate first if active
        if self._registry.get_state(name) == PluginState.ACTIVE:
            self.deactivate(name)

        plugin.on_unload()
        return self._registry.unregister(name)

    def activate(self, name: str) -> bool:
        """Activate a plugin.

        Args:
            name: Plugin name

        Returns:
            True if activated
        """
        plugin = self._registry.get(name)
        if plugin is None:
            return False

        state = self._registry.get_state(name)
        if state == PluginState.ACTIVE:
            return True

        try:
            plugin.on_load()
            plugin.activate(self._context)
            self._registry.set_state(name, PluginState.ACTIVE)
            return True
        except Exception:
            self._registry.set_state(name, PluginState.ERROR)
            return False

    def deactivate(self, name: str) -> bool:
        """Deactivate a plugin.

        Args:
            name: Plugin name

        Returns:
            True if deactivated
        """
        plugin = self._registry.get(name)
        if plugin is None:
            return False

        try:
            plugin.deactivate()
            self._registry.set_state(name, PluginState.LOADED)
            return True
        except Exception:
            self._registry.set_state(name, PluginState.ERROR)
            return False

    def get_active_plugins(self) -> list[str]:
        """Get list of active plugin names."""
        return self._registry.list_by_state(PluginState.ACTIVE)

    def list_plugins(self) -> list[dict[str, Any]]:
        """Get list of all plugins with their states."""
        result = []
        for info in self._registry.list_plugins():
            state = self._registry.get_state(info.name)
            result.append({
                **info.to_dict(),
                "state": state.value if state else "unknown",
            })
        return result
