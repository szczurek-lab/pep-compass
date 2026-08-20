"""Typed, validated registries for factories selected from configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RegistryEntry(Generic[T]):
    """Describe one named factory owned by a :class:`Registry`.

    :param name: Stable configuration name.
    :param factory: Callable constructing the registered implementation.
    :param service_names: Composition-root services accepted by the factory.
    :param accepted_parameters: Explicit public parameter names. ``None``
        derives names from the factory signature.
    :param required_parameters: Explicit required public parameter names.
    """

    name: str
    factory: Callable[..., T]
    service_names: frozenset[str]
    accepted_parameters: frozenset[str] | None = None
    required_parameters: frozenset[str] = frozenset()


class Registry(Generic[T]):
    """Register, validate, and build one namespaced family of factories.

    Registration is strict: a duplicate name is always an error. This avoids
    import-order-dependent implementation replacement.

    :param namespace: Human-readable namespace included in diagnostics.
    :param expected_type: Optional runtime type required from each factory.
    """

    def __init__(self, namespace: str, *, expected_type: type[T] | None = None) -> None:
        if not namespace:
            raise ValueError("Registry namespace cannot be empty.")
        self.namespace = namespace
        self.expected_type = expected_type
        self._entries: dict[str, RegistryEntry[T]] = {}

    @property
    def entries(self) -> Mapping[str, Callable[..., T]]:
        """Return a read-only-compatible view of registered factories."""
        return {name: entry.factory for name, entry in self._entries.items()}

    def register(
        self,
        name: str,
        *,
        services: set[str] | frozenset[str] | None = None,
        accepted_parameters: set[str] | frozenset[str] | None = None,
        required_parameters: set[str] | frozenset[str] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator registering one factory.

        :param name: Stable configuration name.
        :param services: Names injected only by the composition root.
        :param accepted_parameters: Explicit YAML parameter allowlist.
        :param required_parameters: Explicit required YAML parameters.
        :return: Decorator preserving the original callable.
        :raises ValueError: If the name is invalid or already registered.
        """
        normalized = self._normalize_name(name)
        if normalized in self._entries:
            raise ValueError(
                f"Duplicate registration in {self.namespace}: {normalized!r}."
            )

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            legacy_accepted = getattr(factory, "__strategy_accepted__", None)
            legacy_required = getattr(factory, "__strategy_required__", None)
            self._entries[normalized] = RegistryEntry(
                name=normalized,
                factory=factory,
                service_names=frozenset(services or ()),
                accepted_parameters=(
                    frozenset(
                        accepted_parameters
                        if accepted_parameters is not None
                        else legacy_accepted
                    )
                    if accepted_parameters is not None or legacy_accepted is not None
                    else None
                ),
                required_parameters=frozenset(
                    required_parameters
                    if required_parameters is not None
                    else legacy_required or ()
                ),
            )
            return factory

        return decorator

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._entries))

    def factory(self, name: str) -> Callable[..., T]:
        """Return a registered factory.

        :raises ValueError: If ``name`` is unknown.
        """
        return self._entry(name).factory

    def validate(self, name: str, parameters: Mapping[str, Any]) -> None:
        """Validate public parameters without constructing an implementation.

        :param name: Registered strategy name.
        :param parameters: YAML-provided strategy parameters.
        :raises ValueError: If parameters are unknown, missing, or override a
            composition-root service.
        """
        entry = self._entry(name)
        supplied = set(parameters)
        forbidden = supplied & entry.service_names
        if forbidden:
            raise ValueError(
                f"{self.namespace} parameters cannot override services: "
                f"{sorted(forbidden)}."
            )
        accepted, required = self._parameter_contract(entry)
        unknown = supplied - accepted
        missing = required - supplied
        if unknown:
            raise ValueError(
                f"Unknown {self.namespace} parameters: {sorted(unknown)}."
            )
        if missing:
            raise ValueError(
                f"Missing {self.namespace} parameters: {sorted(missing)}."
            )

    def build(
        self,
        name: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        services: Mapping[str, Any] | None = None,
    ) -> T:
        """Validate and construct one registered implementation.

        :param name: Registered strategy name.
        :param parameters: YAML-provided parameters.
        :param services: Composition-root-only dependencies.
        :return: Constructed implementation.
        :raises TypeError: If the factory returns an unexpected type.
        """
        resolved_parameters = dict(parameters or {})
        entry = self._entry(name)
        self.validate(name, resolved_parameters)
        available_services = dict(services or {})
        factory_parameters = signature(entry.factory).parameters
        required_services = {
            service
            for service in entry.service_names
            if service in factory_parameters
        }
        missing_services = required_services - set(available_services)
        if missing_services:
            raise ValueError(
                f"{self.namespace} strategy {name!r} requires services: "
                f"{sorted(missing_services)}."
            )
        arguments = {
            **resolved_parameters,
            **{
                service: available_services[service]
                for service in required_services
            },
        }
        value = entry.factory(**arguments)
        if self.expected_type is not None and not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.namespace} strategy {name!r} returned "
                f"{type(value).__name__}, expected {self.expected_type.__name__}."
            )
        return value

    def _entry(self, name: str) -> RegistryEntry[T]:
        normalized = self._normalize_name(name)
        try:
            return self._entries[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unknown {self.namespace} strategy: {normalized!r}. "
                f"Available: {list(self.names())}."
            ) from error

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry name must be a non-empty string.")
        return name.strip()

    @staticmethod
    def _parameter_contract(entry: RegistryEntry[T]) -> tuple[set[str], set[str]]:
        parameters = signature(entry.factory).parameters
        if entry.accepted_parameters is None:
            accepted = {
                name
                for name, parameter in parameters.items()
                if parameter.kind not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
                and name not in entry.service_names
            }
        else:
            accepted = set(entry.accepted_parameters)
        if entry.required_parameters:
            required = set(entry.required_parameters)
        else:
            required = {
                name
                for name, parameter in parameters.items()
                if name not in entry.service_names
                and parameter.default is Parameter.empty
                and parameter.kind
                not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
            }
        return accepted, required
