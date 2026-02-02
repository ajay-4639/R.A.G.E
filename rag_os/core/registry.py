"""Step registry for discovering and instantiating pipeline steps."""

from typing import Type, Callable, Any
from functools import wraps

from rag_os.core.types import StepType
from rag_os.core.step import Step


class StepMetadata:
    """Metadata about a registered step."""

    def __init__(
        self,
        step_class: Type[Step],
        name: str,
        step_type: StepType,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self.step_class = step_class
        self.name = name
        self.step_type = step_type
        self.description = description
        self.version = version
        self.author = author
        self.tags = tags or []

    def __repr__(self) -> str:
        return f"StepMetadata(name='{self.name}', step_type={self.step_type}, version='{self.version}')"


class StepRegistry:
    """
    Singleton registry for all available pipeline steps.

    Steps can be registered using the @register_step decorator or
    by calling registry.register() directly.

    Example:
        @register_step(
            name="my_chunker",
            step_type=StepType.CHUNKING,
            description="My custom chunking step"
        )
        class MyChunkingStep(Step):
            ...

        # Later, to instantiate:
        registry = StepRegistry()
        step_class = registry.get("my_chunker")
        step = step_class(step_id="s1", step_type=StepType.CHUNKING, config={})
    """

    _instance: "StepRegistry | None" = None
    _initialized: bool = False

    def __new__(cls) -> "StepRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only once)."""
        if not StepRegistry._initialized:
            self._steps: dict[str, StepMetadata] = {}
            self._by_type: dict[StepType, list[str]] = {st: [] for st in StepType}
            StepRegistry._initialized = True

    def register(
        self,
        step_class: Type[Step],
        name: str,
        step_type: StepType,
        description: str = "",
        version: str = "1.0.0",
        author: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """
        Register a step class with the registry.

        Args:
            step_class: The Step subclass to register
            name: Unique name for this step
            step_type: The type/category of this step
            description: Human-readable description
            version: Version string for this step
            author: Author of this step
            tags: Optional tags for categorization

        Raises:
            ValueError: If name is already registered or step_class is not a Step subclass
        """
        if not isinstance(step_class, type) or not issubclass(step_class, Step):
            raise ValueError(f"step_class must be a subclass of Step, got {step_class}")

        if name in self._steps:
            raise ValueError(f"Step '{name}' is already registered")

        metadata = StepMetadata(
            step_class=step_class,
            name=name,
            step_type=step_type,
            description=description,
            version=version,
            author=author,
            tags=tags,
        )

        self._steps[name] = metadata
        self._by_type[step_type].append(name)

    def unregister(self, name: str) -> bool:
        """
        Remove a step from the registry.

        Args:
            name: Name of the step to remove

        Returns:
            True if step was removed, False if it wasn't registered
        """
        if name not in self._steps:
            return False

        metadata = self._steps[name]
        self._by_type[metadata.step_type].remove(name)
        del self._steps[name]
        return True

    def get(self, name: str) -> Type[Step] | None:
        """
        Get a step class by its registered name.

        Args:
            name: The registered name of the step

        Returns:
            The Step subclass, or None if not found
        """
        metadata = self._steps.get(name)
        return metadata.step_class if metadata else None

    def get_metadata(self, name: str) -> StepMetadata | None:
        """
        Get full metadata for a registered step.

        Args:
            name: The registered name of the step

        Returns:
            StepMetadata instance, or None if not found
        """
        return self._steps.get(name)

    def get_by_type(self, step_type: StepType) -> list[str]:
        """
        Get all step names of a specific type.

        Args:
            step_type: The StepType to filter by

        Returns:
            List of registered step names of that type
        """
        return self._by_type.get(step_type, []).copy()

    def list_steps(self) -> list[str]:
        """
        Get all registered step names.

        Returns:
            List of all registered step names
        """
        return list(self._steps.keys())

    def list_metadata(self) -> list[StepMetadata]:
        """
        Get metadata for all registered steps.

        Returns:
            List of StepMetadata instances
        """
        return list(self._steps.values())

    def has(self, name: str) -> bool:
        """
        Check if a step is registered.

        Args:
            name: Name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._steps

    def clear(self) -> None:
        """Clear all registered steps. Useful for testing."""
        self._steps.clear()
        self._by_type = {st: [] for st in StepType}

    def create_step(
        self,
        name: str,
        step_id: str,
        config: dict[str, Any] | None = None,
    ) -> Step:
        """
        Create an instance of a registered step.

        Args:
            name: Registered name of the step
            step_id: Unique ID for this step instance
            config: Configuration for the step

        Returns:
            Instantiated Step

        Raises:
            KeyError: If step is not registered
        """
        metadata = self._steps.get(name)
        if metadata is None:
            raise KeyError(f"Step '{name}' is not registered")

        return metadata.step_class(
            step_id=step_id,
            step_type=metadata.step_type,
            config=config or {},
        )

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, name: str) -> bool:
        return name in self._steps


def register_step(
    name: str,
    step_type: StepType,
    description: str = "",
    version: str = "1.0.0",
    author: str = "",
    tags: list[str] | None = None,
) -> Callable[[Type[Step]], Type[Step]]:
    """
    Decorator to register a step class with the global registry.

    Args:
        name: Unique name for this step
        step_type: The type/category of this step
        description: Human-readable description
        version: Version string for this step
        author: Author of this step
        tags: Optional tags for categorization

    Returns:
        Decorator function

    Example:
        @register_step(
            name="token_chunker",
            step_type=StepType.CHUNKING,
            description="Chunks text by token count"
        )
        class TokenChunkingStep(Step):
            ...
    """

    def decorator(cls: Type[Step]) -> Type[Step]:
        registry = StepRegistry()
        registry.register(
            step_class=cls,
            name=name,
            step_type=step_type,
            description=description,
            version=version,
            author=author,
            tags=tags,
        )
        return cls

    return decorator


# Convenience function to get the singleton registry
def get_registry() -> StepRegistry:
    """Get the global step registry singleton."""
    return StepRegistry()
