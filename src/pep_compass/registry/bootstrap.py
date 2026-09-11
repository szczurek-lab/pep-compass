"""Deterministic registration of PepCompass built-in component declarations."""

from __future__ import annotations

from threading import Lock


_lock = Lock()
_loaded = False


def load_builtin_registrations() -> None:
    """Load lightweight built-in registration modules exactly once.

    The imported modules register factories, not instantiated models. Heavy
    model implementations remain imported by their factory only on ``build``.
    """
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        import pep_compass.autoencoder.strategies  # noqa: F401
        import pep_compass.optimization.components.filters.registry  # noqa: F401
        import pep_compass.optimization.components.mutation_generators.strategies  # noqa: F401
        import pep_compass.optimization.components.oracles.strategies  # noqa: F401
        import pep_compass.optimization.components.walkers.strategies  # noqa: F401

        _loaded = True
