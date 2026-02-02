"""Hook system for RAG OS plugins."""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import threading


class HookType(Enum):
    """Types of hooks in the pipeline lifecycle."""

    # Pipeline hooks
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PIPELINE_ERROR = "pipeline_error"

    # Step hooks
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_ERROR = "step_error"
    STEP_SKIP = "step_skip"

    # Data hooks
    DATA_TRANSFORM = "data_transform"
    DATA_VALIDATE = "data_validate"

    # Query hooks
    QUERY_START = "query_start"
    QUERY_END = "query_end"

    # Custom hooks
    CUSTOM = "custom"


@dataclass
class Hook:
    """A registered hook.

    Attributes:
        hook_type: Type of hook
        callback: Callback function
        priority: Execution priority (lower = earlier)
        name: Optional hook name
        enabled: Whether hook is enabled
    """
    hook_type: HookType | str
    callback: Callable[..., Any]
    priority: int = 100
    name: str = ""
    enabled: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = f"hook_{id(self.callback)}"


@dataclass
class HookResult:
    """Result of executing a hook.

    Attributes:
        hook_name: Name of the hook
        success: Whether execution succeeded
        data: Data returned by the hook
        error: Error message if failed
    """
    hook_name: str
    success: bool = True
    data: Any = None
    error: str | None = None


class HookManager:
    """Manager for hook registration and execution.

    Handles registration and execution of hooks at various
    points in the pipeline lifecycle.

    Usage:
        manager = HookManager()

        # Register a hook
        manager.register(HookType.PIPELINE_START, my_callback)

        # Execute hooks
        results = manager.execute(HookType.PIPELINE_START, pipeline_data)
    """

    def __init__(self):
        self._hooks: dict[str, list[Hook]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        hook_type: HookType | str,
        callback: Callable[..., Any],
        priority: int = 100,
        name: str = "",
    ) -> Hook:
        """Register a hook.

        Args:
            hook_type: Type of hook
            callback: Callback function
            priority: Execution priority
            name: Optional hook name

        Returns:
            Registered Hook
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        hook = Hook(
            hook_type=hook_type,
            callback=callback,
            priority=priority,
            name=name,
        )

        with self._lock:
            if hook_key not in self._hooks:
                self._hooks[hook_key] = []
            self._hooks[hook_key].append(hook)
            # Sort by priority
            self._hooks[hook_key].sort(key=lambda h: h.priority)

        return hook

    def unregister(self, hook: Hook) -> bool:
        """Unregister a hook.

        Args:
            hook: Hook to unregister

        Returns:
            True if unregistered
        """
        hook_key = hook.hook_type.value if isinstance(hook.hook_type, HookType) else hook.hook_type

        with self._lock:
            if hook_key not in self._hooks:
                return False
            try:
                self._hooks[hook_key].remove(hook)
                return True
            except ValueError:
                return False

    def unregister_by_name(self, name: str) -> int:
        """Unregister hooks by name.

        Args:
            name: Hook name

        Returns:
            Number of hooks removed
        """
        count = 0
        with self._lock:
            for hook_key in self._hooks:
                original_len = len(self._hooks[hook_key])
                self._hooks[hook_key] = [
                    h for h in self._hooks[hook_key]
                    if h.name != name
                ]
                count += original_len - len(self._hooks[hook_key])
        return count

    def execute(
        self,
        hook_type: HookType | str,
        *args: Any,
        **kwargs: Any,
    ) -> list[HookResult]:
        """Execute all hooks of a type.

        Args:
            hook_type: Type of hook
            *args: Arguments to pass to callbacks
            **kwargs: Keyword arguments to pass

        Returns:
            List of HookResults
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        with self._lock:
            hooks = self._hooks.get(hook_key, []).copy()

        results = []
        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                data = hook.callback(*args, **kwargs)
                results.append(HookResult(
                    hook_name=hook.name,
                    success=True,
                    data=data,
                ))
            except Exception as e:
                results.append(HookResult(
                    hook_name=hook.name,
                    success=False,
                    error=str(e),
                ))

        return results

    def execute_chain(
        self,
        hook_type: HookType | str,
        data: Any,
    ) -> tuple[Any, list[HookResult]]:
        """Execute hooks as a chain, passing data through.

        Each hook receives the output of the previous hook.

        Args:
            hook_type: Type of hook
            data: Initial data

        Returns:
            Tuple of (final_data, results)
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        with self._lock:
            hooks = self._hooks.get(hook_key, []).copy()

        results = []
        current_data = data

        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                current_data = hook.callback(current_data)
                results.append(HookResult(
                    hook_name=hook.name,
                    success=True,
                    data=current_data,
                ))
            except Exception as e:
                results.append(HookResult(
                    hook_name=hook.name,
                    success=False,
                    error=str(e),
                ))
                # Stop chain on error
                break

        return current_data, results

    def get_hooks(self, hook_type: HookType | str) -> list[Hook]:
        """Get all hooks for a type.

        Args:
            hook_type: Type of hook

        Returns:
            List of hooks
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        with self._lock:
            return self._hooks.get(hook_key, []).copy()

    def enable(self, name: str) -> int:
        """Enable hooks by name.

        Args:
            name: Hook name

        Returns:
            Number of hooks enabled
        """
        return self._set_enabled(name, True)

    def disable(self, name: str) -> int:
        """Disable hooks by name.

        Args:
            name: Hook name

        Returns:
            Number of hooks disabled
        """
        return self._set_enabled(name, False)

    def _set_enabled(self, name: str, enabled: bool) -> int:
        """Set enabled state for hooks by name."""
        count = 0
        with self._lock:
            for hooks in self._hooks.values():
                for hook in hooks:
                    if hook.name == name:
                        hook.enabled = enabled
                        count += 1
        return count

    def clear(self, hook_type: HookType | str | None = None) -> int:
        """Clear hooks.

        Args:
            hook_type: Type to clear (None = all)

        Returns:
            Number of hooks cleared
        """
        with self._lock:
            if hook_type is None:
                count = sum(len(h) for h in self._hooks.values())
                self._hooks.clear()
                return count
            else:
                hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type
                if hook_key in self._hooks:
                    count = len(self._hooks[hook_key])
                    del self._hooks[hook_key]
                    return count
                return 0
