"""Traverse executable objects without changing their lifecycle or RNG state."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pep_compass.steps.base import Step


def iter_steps(root: Step) -> Iterator[Step]:
    """Yield composite operations and leaf components in structural order.

    :param root: Root resource directory or root graph node, as declared by its type.
    :type root: Step
    :return: Result of this operation.
    :rtype: Iterator[Step]
    """
    yield root
    children = list(getattr(root, "steps", ()))
    if hasattr(root, "body"):
        children.append(root.body)
    children.extend(getattr(root, "branches", {}).values())
    if hasattr(root, "walker"):
        children.extend((root.walker, root.mutation_generator, root.filters))
    for child in children:
        yield from iter_steps(child)


def find_step(root: Step, node_id: str) -> Step | None:
    """Find an executable by its stable declaration identity.

    :param root: Root resource directory or root graph node, as declared by its type.
    :type root: Step
    :param node_id: Stable graph placement identity.
    :type node_id: str
    :return: Result of this operation.
    :rtype: Step | None
    """
    return next(
        (
            step
            for step in iter_steps(root)
            if getattr(step, "node_id", None) == node_id
        ),
        None,
    )
