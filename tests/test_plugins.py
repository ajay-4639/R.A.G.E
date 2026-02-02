"""Tests for RAG OS plugin system."""

import pytest
import tempfile
from pathlib import Path

from rag_os.plugins import (
    Plugin,
    PluginInfo,
    PluginManager,
    PluginRegistry,
    HookType,
    Hook,
    HookManager,
    PluginLoader,
    discover_plugins,
)
from rag_os.plugins.base import PluginState, PluginContext
from rag_os.plugins.hooks import HookResult
from rag_os.plugins.loader import PluginPackage


# =============================================================================
# PluginInfo Tests
# =============================================================================

class TestPluginInfo:
    """Tests for PluginInfo."""

    def test_create_info(self):
        """Test creating plugin info."""
        info = PluginInfo(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin",
        )

        assert info.name == "test-plugin"
        assert info.version == "1.0.0"
        assert info.description == "A test plugin"

    def test_to_dict(self):
        """Test serializing to dict."""
        info = PluginInfo(
            name="test",
            version="2.0.0",
            author="Test Author",
            tags=["test", "demo"],
        )

        data = info.to_dict()

        assert data["name"] == "test"
        assert data["version"] == "2.0.0"
        assert data["author"] == "Test Author"
        assert data["tags"] == ["test", "demo"]


# =============================================================================
# Plugin Tests
# =============================================================================

class MockPlugin(Plugin):
    """Mock plugin for testing."""

    def __init__(self, name: str = "mock-plugin"):
        self._name = name
        self.activated = False
        self.deactivated = False

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name=self._name, version="1.0.0")

    def activate(self, context: PluginContext) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True


class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_info(self):
        """Test getting plugin info."""
        plugin = MockPlugin("my-plugin")

        assert plugin.info.name == "my-plugin"
        assert plugin.info.version == "1.0.0"

    def test_activate(self):
        """Test activating plugin."""
        plugin = MockPlugin()
        context = PluginContext()

        plugin.activate(context)

        assert plugin.activated is True

    def test_deactivate(self):
        """Test deactivating plugin."""
        plugin = MockPlugin()

        plugin.deactivate()

        assert plugin.deactivated is True


# =============================================================================
# PluginRegistry Tests
# =============================================================================

class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin("test")

        registry.register(plugin)

        assert registry.get("test") is plugin
        assert registry.get_state("test") == PluginState.UNLOADED

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin("test")
        registry.register(plugin)

        result = registry.unregister("test")

        assert result is True
        assert registry.get("test") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent plugin."""
        registry = PluginRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_list_plugins(self):
        """Test listing plugins."""
        registry = PluginRegistry()
        registry.register(MockPlugin("plugin-1"))
        registry.register(MockPlugin("plugin-2"))

        plugins = registry.list_plugins()

        assert len(plugins) == 2
        names = [p.name for p in plugins]
        assert "plugin-1" in names
        assert "plugin-2" in names

    def test_list_by_state(self):
        """Test listing plugins by state."""
        registry = PluginRegistry()
        registry.register(MockPlugin("p1"))
        registry.register(MockPlugin("p2"))
        registry.set_state("p1", PluginState.ACTIVE)

        active = registry.list_by_state(PluginState.ACTIVE)
        unloaded = registry.list_by_state(PluginState.UNLOADED)

        assert active == ["p1"]
        assert unloaded == ["p2"]


# =============================================================================
# PluginManager Tests
# =============================================================================

class TestPluginManager:
    """Tests for PluginManager."""

    def test_install_plugin(self):
        """Test installing a plugin."""
        manager = PluginManager()
        plugin = MockPlugin("test")

        result = manager.install(plugin)

        assert result is True
        assert manager.registry.get("test") is plugin

    def test_activate_plugin(self):
        """Test activating a plugin."""
        manager = PluginManager()
        plugin = MockPlugin("test")
        manager.install(plugin)

        result = manager.activate("test")

        assert result is True
        assert plugin.activated is True
        assert manager.registry.get_state("test") == PluginState.ACTIVE

    def test_deactivate_plugin(self):
        """Test deactivating a plugin."""
        manager = PluginManager()
        plugin = MockPlugin("test")
        manager.install(plugin)
        manager.activate("test")

        result = manager.deactivate("test")

        assert result is True
        assert plugin.deactivated is True
        assert manager.registry.get_state("test") == PluginState.LOADED

    def test_uninstall_plugin(self):
        """Test uninstalling a plugin."""
        manager = PluginManager()
        plugin = MockPlugin("test")
        manager.install(plugin)
        manager.activate("test")

        result = manager.uninstall("test")

        assert result is True
        assert manager.registry.get("test") is None

    def test_get_active_plugins(self):
        """Test getting active plugins."""
        manager = PluginManager()
        manager.install(MockPlugin("p1"))
        manager.install(MockPlugin("p2"))
        manager.install(MockPlugin("p3"))
        manager.activate("p1")
        manager.activate("p3")

        active = manager.get_active_plugins()

        assert set(active) == {"p1", "p3"}

    def test_list_plugins(self):
        """Test listing all plugins."""
        manager = PluginManager()
        manager.install(MockPlugin("test"))
        manager.activate("test")

        plugins = manager.list_plugins()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "test"
        assert plugins[0]["state"] == "active"


# =============================================================================
# Hook Tests
# =============================================================================

class TestHook:
    """Tests for Hook dataclass."""

    def test_create_hook(self):
        """Test creating a hook."""
        def my_callback():
            pass

        hook = Hook(
            hook_type=HookType.PIPELINE_START,
            callback=my_callback,
            priority=50,
            name="my-hook",
        )

        assert hook.hook_type == HookType.PIPELINE_START
        assert hook.callback == my_callback
        assert hook.priority == 50
        assert hook.name == "my-hook"
        assert hook.enabled is True


# =============================================================================
# HookManager Tests
# =============================================================================

class TestHookManager:
    """Tests for HookManager."""

    def test_register_hook(self):
        """Test registering a hook."""
        manager = HookManager()

        hook = manager.register(HookType.PIPELINE_START, lambda: "result")

        assert hook is not None
        hooks = manager.get_hooks(HookType.PIPELINE_START)
        assert len(hooks) == 1

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        manager = HookManager()
        hook = manager.register(HookType.PIPELINE_START, lambda: None)

        result = manager.unregister(hook)

        assert result is True
        assert len(manager.get_hooks(HookType.PIPELINE_START)) == 0

    def test_execute_hooks(self):
        """Test executing hooks."""
        manager = HookManager()
        results_list = []

        manager.register(HookType.PIPELINE_START, lambda: results_list.append(1))
        manager.register(HookType.PIPELINE_START, lambda: results_list.append(2))

        manager.execute(HookType.PIPELINE_START)

        assert results_list == [1, 2]

    def test_execute_with_args(self):
        """Test executing hooks with arguments."""
        manager = HookManager()
        captured = []

        def capture_hook(data, extra):
            captured.append((data, extra))

        manager.register(HookType.STEP_START, capture_hook)
        manager.execute(HookType.STEP_START, "data1", extra="extra1")

        assert captured == [("data1", "extra1")]

    def test_execute_returns_results(self):
        """Test that execute returns results."""
        manager = HookManager()
        manager.register(HookType.QUERY_START, lambda: "result1", name="h1")
        manager.register(HookType.QUERY_START, lambda: "result2", name="h2")

        results = manager.execute(HookType.QUERY_START)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].data == "result1"

    def test_hook_priority(self):
        """Test hook execution priority."""
        manager = HookManager()
        order = []

        manager.register(HookType.PIPELINE_END, lambda: order.append(3), priority=300)
        manager.register(HookType.PIPELINE_END, lambda: order.append(1), priority=100)
        manager.register(HookType.PIPELINE_END, lambda: order.append(2), priority=200)

        manager.execute(HookType.PIPELINE_END)

        assert order == [1, 2, 3]

    def test_execute_chain(self):
        """Test chained execution."""
        manager = HookManager()

        manager.register(HookType.DATA_TRANSFORM, lambda x: x * 2)
        manager.register(HookType.DATA_TRANSFORM, lambda x: x + 10)

        data, results = manager.execute_chain(HookType.DATA_TRANSFORM, 5)

        assert data == 20  # (5 * 2) + 10

    def test_disabled_hook(self):
        """Test disabled hooks are skipped."""
        manager = HookManager()
        results = []

        hook = manager.register(HookType.STEP_END, lambda: results.append(1), name="h1")
        hook.enabled = False
        manager.register(HookType.STEP_END, lambda: results.append(2))

        manager.execute(HookType.STEP_END)

        assert results == [2]

    def test_enable_disable_by_name(self):
        """Test enabling/disabling hooks by name."""
        manager = HookManager()
        manager.register(HookType.PIPELINE_ERROR, lambda: None, name="test-hook")

        count = manager.disable("test-hook")
        assert count == 1

        hooks = manager.get_hooks(HookType.PIPELINE_ERROR)
        assert hooks[0].enabled is False

        manager.enable("test-hook")
        assert hooks[0].enabled is True

    def test_clear_hooks(self):
        """Test clearing hooks."""
        manager = HookManager()
        manager.register(HookType.STEP_START, lambda: None)
        manager.register(HookType.STEP_END, lambda: None)

        count = manager.clear(HookType.STEP_START)

        assert count == 1
        assert len(manager.get_hooks(HookType.STEP_START)) == 0
        assert len(manager.get_hooks(HookType.STEP_END)) == 1

    def test_clear_all_hooks(self):
        """Test clearing all hooks."""
        manager = HookManager()
        manager.register(HookType.STEP_START, lambda: None)
        manager.register(HookType.STEP_END, lambda: None)

        count = manager.clear()

        assert count == 2


# =============================================================================
# PluginLoader Tests
# =============================================================================

class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_load_from_path(self):
        """Test loading plugin from file path."""
        # Create a temporary plugin file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from rag_os.plugins import Plugin, PluginInfo

class TestFilePlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(name="file-plugin", version="1.0.0")

    def activate(self, context):
        pass
""")
            path = f.name

        try:
            loader = PluginLoader()
            plugin = loader.load_from_path(path)

            assert plugin is not None
            assert plugin.info.name == "file-plugin"
        finally:
            Path(path).unlink()

    def test_load_nonexistent_path(self):
        """Test loading from nonexistent path."""
        loader = PluginLoader()

        plugin = loader.load_from_path("/nonexistent/path.py")

        assert plugin is None

    def test_unload_module(self):
        """Test unloading a module."""
        loader = PluginLoader()

        # After loading something
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
from rag_os.plugins import Plugin, PluginInfo

class TempPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(name="temp", version="1.0.0")
    def activate(self, context):
        pass
""")
            path = f.name

        try:
            loader.load_from_path(path)
            modules = loader.get_loaded_modules()

            assert len(modules) == 1

            loader.unload(modules[0])
            assert len(loader.get_loaded_modules()) == 0
        finally:
            Path(path).unlink()


# =============================================================================
# PluginPackage Tests
# =============================================================================

class TestPluginPackage:
    """Tests for PluginPackage helper."""

    def test_build_package(self):
        """Test building a plugin package."""
        package = PluginPackage(
            name="packaged-plugin",
            version="2.0.0",
            description="A packaged plugin",
        )

        plugin = package.build()

        assert plugin.info.name == "packaged-plugin"
        assert plugin.info.version == "2.0.0"

    def test_package_with_hooks(self):
        """Test package with hooks."""
        calls = []

        package = (
            PluginPackage("hook-plugin")
            .add_hook("custom_event", lambda: calls.append("called"), priority=50)
        )

        plugin = package.build()

        # Create a mock context that tracks registrations
        class MockContext(PluginContext):
            def __init__(self):
                super().__init__()
                self.hooks = []

            def register_hook(self, hook_type, callback, priority):
                self.hooks.append((hook_type, callback, priority))
                callback()  # Execute to verify

        context = MockContext()
        plugin.activate(context)

        assert len(context.hooks) == 1
        assert calls == ["called"]

    def test_package_chaining(self):
        """Test fluent chaining."""
        class MockStep:
            pass

        package = (
            PluginPackage("chained")
            .add_step("step1", MockStep)
            .add_step("step2", MockStep)
            .add_hook("event", lambda: None)
        )

        plugin = package.build()
        assert plugin.info.name == "chained"
