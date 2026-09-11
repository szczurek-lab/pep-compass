"""Dependency injection helper for registered strategy factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from typing import Any


def build_with_services(
    factory: Callable[..., Any],
    parameters: Mapping[str, Any],
    services: Mapping[str, Any] | None,
) -> Any:
    """Invoke a factory with only the services it explicitly declares.

    :param factory: Registered strategy factory or class.
    :type factory: Callable[..., Any]
    :param parameters: User-provided strategy parameters.
    :type parameters: Mapping[str, Any]
    :param services: Developer-owned services supplied by the composition root.
    :type services: Mapping[str, Any] | None
    :return: Constructed strategy.
    :rtype: Any
    :raises ValueError: If a user parameter attempts to replace a core service.
    """
    arguments = dict(parameters)
    for name, service in (services or {}).items():
        if name not in signature(factory).parameters:
            continue
        if name in arguments:
            raise ValueError(f"Strategy parameter cannot override core service: {name}")
        arguments[name] = service
    return factory(**arguments)


def parameter_contract(
    *,
    accepted: set[str] | None = None,
    required: set[str] | None = None,
    source: Callable[..., Any] | None = None,
):
    """Attach a model-free parameter contract to a strategy factory."""

    def decorator(factory):
        factory.__strategy_accepted__ = accepted
        factory.__strategy_required__ = required or set()
        factory.__strategy_source__ = source
        return factory

    return decorator


def validate_factory_parameters(
    factory: Callable[..., Any],
    parameters: Mapping[str, Any],
    *,
    service_names: set[str] | None = None,
) -> None:
    """Validate parameter names and required values without invoking a factory."""
    source = getattr(factory, "__strategy_source__", None) or factory
    declared = signature(source).parameters
    accepted = getattr(factory, "__strategy_accepted__", None)
    if accepted is None:
        accepted = {
            name
            for name, parameter in declared.items()
            if name != "self" and parameter.kind.name not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
        }
    accepted -= service_names or set()
    required = set(getattr(factory, "__strategy_required__", set()))
    if not required:
        required = {
            name
            for name, parameter in declared.items()
            if name != "self"
            and name not in (service_names or set())
            and parameter.default is parameter.empty
            and parameter.kind.name not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
        }
    unknown = set(parameters) - accepted
    missing = required - set(parameters)
    if unknown:
        raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing required strategy parameters: {sorted(missing)}")
