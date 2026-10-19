"""Explicit data-flow and resource requirements of a configured component."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepContract:
    """Describe guarantees consumed by graph validation and estimation.

    :param required_fields: Fields that must exist at this node.
    :param optional_fields: Subset of required_fields accepted as OptionalField with missing rows.
    :param produced_fields: Fields guaranteed after successful execution.
    :param invalidated_prefixes: Input field prefixes removed by the component.
    :param resources: Shared resources required for initialization/execution.
    """

    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    produced_fields: tuple[str, ...] = ()
    invalidated_prefixes: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require partial-input permissions to refer to required input fields.

        :rtype: None
        """
        if set(self.optional_fields) - set(self.required_fields):
            raise ValueError("Partial-input permissions must name required fields.")
