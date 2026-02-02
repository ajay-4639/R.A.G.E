"""Plugin loader for RAG OS."""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from rag_os.plugins.base import Plugin, PluginInfo


class PluginLoader:
    """Loader for discovering and loading plugins.

    Supports loading plugins from:
    - Python modules
    - Entry points (pkg_resources)
    - File paths
    """

    def __init__(self):
        self._loaded_modules: dict[str, Any] = {}

    def load_from_module(self, module_name: str) -> Plugin | None:
        """Load a plugin from a Python module.

        The module must define a `create_plugin()` function or
        have a class that inherits from Plugin.

        Args:
            module_name: Fully qualified module name

        Returns:
            Plugin instance or None
        """
        try:
            module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module

            # Try factory function first
            if hasattr(module, "create_plugin"):
                return module.create_plugin()

            # Look for Plugin subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    return attr()

            return None

        except ImportError:
            return None

    def load_from_path(self, path: str | Path) -> Plugin | None:
        """Load a plugin from a file path.

        Args:
            path: Path to Python file

        Returns:
            Plugin instance or None
        """
        path = Path(path)

        if not path.exists() or not path.suffix == ".py":
            return None

        module_name = f"rag_os_plugin_{path.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules[module_name] = module

            # Try factory function first
            if hasattr(module, "create_plugin"):
                return module.create_plugin()

            # Look for Plugin subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    return attr()

            return None

        except Exception:
            return None

    def load_from_entry_points(self, group: str = "rag_os.plugins") -> list[Plugin]:
        """Load plugins from setuptools entry points.

        Args:
            group: Entry point group name

        Returns:
            List of loaded plugins
        """
        plugins = []

        try:
            # Python 3.10+ has importlib.metadata.entry_points
            from importlib.metadata import entry_points

            eps = entry_points(group=group)
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if isinstance(plugin_class, type) and issubclass(plugin_class, Plugin):
                        plugins.append(plugin_class())
                    elif callable(plugin_class):
                        plugins.append(plugin_class())
                except Exception:
                    continue

        except Exception:
            pass

        return plugins

    def unload(self, module_name: str) -> bool:
        """Unload a loaded module.

        Args:
            module_name: Module name

        Returns:
            True if unloaded
        """
        if module_name in self._loaded_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
            del self._loaded_modules[module_name]
            return True
        return False

    def get_loaded_modules(self) -> list[str]:
        """Get list of loaded module names."""
        return list(self._loaded_modules.keys())


def discover_plugins(
    paths: list[str | Path] | None = None,
    modules: list[str] | None = None,
    entry_point_group: str = "rag_os.plugins",
) -> list[Plugin]:
    """Discover and load plugins from various sources.

    Args:
        paths: List of file paths to search
        modules: List of module names to load
        entry_point_group: Entry point group name

    Returns:
        List of discovered plugins
    """
    loader = PluginLoader()
    plugins = []

    # Load from modules
    if modules:
        for module_name in modules:
            plugin = loader.load_from_module(module_name)
            if plugin:
                plugins.append(plugin)

    # Load from paths
    if paths:
        for path in paths:
            path = Path(path)
            if path.is_file():
                plugin = loader.load_from_path(path)
                if plugin:
                    plugins.append(plugin)
            elif path.is_dir():
                for py_file in path.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    plugin = loader.load_from_path(py_file)
                    if plugin:
                        plugins.append(plugin)

    # Load from entry points
    plugins.extend(loader.load_from_entry_points(entry_point_group))

    return plugins


class PluginPackage:
    """Helper for creating plugin packages.

    Simplifies creating plugins that package multiple components.

    Usage:
        package = PluginPackage("my-plugin", "1.0.0")
        package.add_step("MyStep", MyStepClass)
        package.add_hook(HookType.PIPELINE_START, my_hook)

        plugin = package.build()
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
    ):
        self._info = PluginInfo(
            name=name,
            version=version,
            description=description,
            author=author,
        )
        self._steps: list[tuple[str, type]] = []
        self._hooks: list[tuple[str, Any, int]] = []
        self._on_activate: list[Any] = []

    def add_step(self, name: str, step_class: type) -> "PluginPackage":
        """Add a step to the package.

        Args:
            name: Step name
            step_class: Step class

        Returns:
            Self for chaining
        """
        self._steps.append((name, step_class))
        return self

    def add_hook(
        self,
        hook_type: str,
        callback: Any,
        priority: int = 100,
    ) -> "PluginPackage":
        """Add a hook to the package.

        Args:
            hook_type: Hook type
            callback: Hook callback
            priority: Hook priority

        Returns:
            Self for chaining
        """
        self._hooks.append((hook_type, callback, priority))
        return self

    def on_activate(self, callback: Any) -> "PluginPackage":
        """Add an activation callback.

        Args:
            callback: Callback function

        Returns:
            Self for chaining
        """
        self._on_activate.append(callback)
        return self

    def build(self) -> Plugin:
        """Build the plugin.

        Returns:
            Plugin instance
        """
        info = self._info
        steps = self._steps.copy()
        hooks = self._hooks.copy()
        activate_callbacks = self._on_activate.copy()

        class PackagedPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return info

            def activate(self, context) -> None:
                # Register steps
                for name, step_class in steps:
                    context.register_step(name, step_class)

                # Register hooks
                for hook_type, callback, priority in hooks:
                    context.register_hook(hook_type, callback, priority)

                # Run custom callbacks
                for callback in activate_callbacks:
                    callback(context)

        return PackagedPlugin()
