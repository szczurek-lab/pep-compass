"""PepCompass package without eager initialization of compute dependencies."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pep_compass.optimization.pipeline import PepCompassPipeline

__all__ = ["PepCompassPipeline"]


def __getattr__(name: str) -> Any:
    """Load the compute API only when it is explicitly requested.

    This keeps configuration and analysis imports usable after a lightweight
    installation. Executing a pipeline still requires one device extra, such
    as ``pep-compass[cpu]`` or ``pep-compass[cu126]``.
    """
    if name == "PepCompassPipeline":
        from pep_compass.optimization.pipeline import PepCompassPipeline

        return PepCompassPipeline
    raise AttributeError(name)
