"""Immutable graph placement snapshots, independent of builders and execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

from pep_compass.utils.configuration import (
    construct,
    freeze,
    locate,
    plain,
    target_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Self

    from pep_compass.steps.base import Step


def new_id() -> str:
    """Allocate a placement identity without consuming computational RNG state.

    :return: Result of this operation.
    :rtype: str
    """
    return "node_" + uuid4().hex


def _children(value: object, transform: Callable[[object], object]) -> object:
    """Children.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :param transform: Transform.
    :type transform: Callable[[object], object]
    :return: Result of this operation.
    :rtype: object
    """
    if isinstance(value, NodeSpec):
        return transform(value)
    if isinstance(value, Mapping):
        return {key: _children(item, transform) for key, item in value.items()}
    return [_children(item, transform) for item in value]


def _freeze_children(value: object) -> object:
    """Freeze children.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: object
    """
    if isinstance(value, NodeSpec):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_children(item) for key, item in value.items()}
        )
    return tuple(_freeze_children(item) for item in value)


@dataclass(frozen=True)
class NodeSpec:
    """One graph placement: constructor parameters, children and stable identity."""

    target: str
    parameters: Mapping
    children: Mapping
    node_id: str
    name: str | None = None

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        object.__setattr__(self, "parameters", freeze(self.parameters))
        object.__setattr__(self, "children", _freeze_children(self.children))

    @classmethod
    def capture(
        cls,
        component: Step,
        *,
        name: str | None = None,
        node_id: str | None = None,
        _active: set[int] | None = None,
    ) -> Self:
        """Snapshot a lightweight component and each owned child placement.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :param node_id: Stable graph placement identity.
        :type node_id: str | None
        :param _active:  active.
        :type _active: set[int] | None
        :return: Result of this operation.
        :rtype: Self
        """
        from pep_compass.steps.base import Step

        if not isinstance(component, Step):
            raise TypeError("Graph nodes must implement Step.")
        active = set() if _active is None else _active
        if id(component) in active:
            raise ValueError("Computation ownership must be acyclic.")
        active.add(id(component))
        values = component.configuration
        children = component.graph_children()

        def capture(value: object) -> object:
            """Capture.

            :param value: Value consumed by this conversion or transformation.
            :type value: object
            :return: Result of this operation.
            :rtype: object
            """
            if isinstance(value, Step):
                return cls.capture(value, _active=active)
            if isinstance(value, Mapping):
                return {key: capture(item) for key, item in value.items()}
            if not isinstance(value, (tuple, list)):
                raise TypeError(
                    "Graph children must be Step objects or named ordered collections."
                )
            return tuple(capture(item) for item in value)

        result = cls(
            target_path(component),
            {key: value for key, value in values.items() if key not in children},
            {key: capture(value) for key, value in children.items()},
            node_id or new_id(),
            name,
        )
        active.remove(id(component))
        return result

    @property
    def kind(self) -> str:
        """Return the control-flow role declared by the selected class.

        :return: Result of this operation.
        :rtype: str
        """
        return locate(self.target).kind

    @property
    def category(self) -> str:
        """Return the component's declared category for presentation and validation.

        :return: Result of this operation.
        :rtype: str
        """
        return locate(self.target).category

    @property
    def label(self) -> str:
        """Label.

        :return: Result of this operation.
        :rtype: str
        """
        return self.name or self.target.rsplit(".", 1)[1]

    def to_config(self) -> dict[str, object]:
        """Return a direct constructor tree without placement metadata.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return {
            "_target_": self.target,
            **plain(self.parameters),
            **{
                key: _children(value, lambda node: node.to_config())
                for key, value in self.children.items()
            },
        }

    def instantiate(self) -> Step:
        """Construct fresh components and bind the saved identities recursively.

        :return: Result of this operation.
        :rtype: Step
        """
        component = construct(self.to_config())

        def bind(node: NodeSpec, step: Step) -> Step:
            """Bind.

            :param node: Node.
            :type node: NodeSpec
            :param step: Step.
            :type step: Step
            :return: Result of this operation.
            :rtype: Step
            """
            step.node_id, step.display_name = node.node_id, node.name
            slots = step.graph_children()

            def pair(nodes: object, objects: object) -> None:
                """Pair.

                :param nodes: Nodes.
                :type nodes: object
                :param objects: Objects.
                :type objects: object
                :rtype: None
                """
                if isinstance(nodes, NodeSpec):
                    bind(nodes, objects)
                elif isinstance(nodes, Mapping):
                    for key in nodes:
                        pair(nodes[key], objects[key])
                else:
                    for child, obj in zip(nodes, objects, strict=True):
                        pair(child, obj)

            for key in node.children:
                pair(node.children[key], slots[key])

        bind(self, component)
        return component

    def to_dict(self) -> dict[str, object]:
        """To dict.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return {
            "target": self.target,
            "parameters": plain(self.parameters),
            "node_id": self.node_id,
            "name": self.name,
            "children": {
                key: _children(value, lambda node: node.to_dict())
                for key, value in self.children.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """From dict.

        :param value: Value consumed by this conversion or transformation.
        :type value: Mapping[str, object]
        :return: Result of this operation.
        :rtype: Self
        """

        def children(item: object) -> object:
            """Children.

            :param item: Item.
            :type item: object
            :return: Result of this operation.
            :rtype: object
            """
            if isinstance(item, dict) and "node_id" in item:
                return cls.from_dict(item)
            if isinstance(item, dict):
                return {key: children(child) for key, child in item.items()}
            return tuple(children(child) for child in item)

        return cls(
            value["target"],
            value["parameters"],
            children(value.get("children", {})),
            value["node_id"],
            value.get("name"),
        )


@dataclass(frozen=True)
class PipelineSchema:
    """Independent constructor snapshots, graph identities and shared resource declaration."""

    steps: tuple[NodeSpec, ...] = ()
    autoencoder: Mapping | None = None

    def __post_init__(self) -> None:
        """Post init.

        :rtype: None
        """
        object.__setattr__(self, "steps", tuple(self.steps))
        if any(not isinstance(node, NodeSpec) for node in self.steps):
            raise TypeError("Schema steps must be NodeSpec snapshots.")
        if self.autoencoder is not None:
            object.__setattr__(self, "autoencoder", freeze(self.autoencoder))

    def walk(self) -> Iterator[tuple[NodeSpec, str | None, str]]:
        """Yield every placement with parent identity and a readable structural position.

        :return: Result of this operation.
        :rtype: Iterator[tuple[NodeSpec, str | None, str]]
        """

        def nested(
            value: object, parent: str | None, path: str
        ) -> Iterator[tuple[NodeSpec, str | None, str]]:
            """Nested.

            :param value: Value consumed by this conversion or transformation.
            :type value: object
            :param parent: Parent.
            :type parent: str | None
            :param path: Input or output path used by this operation.
            :type path: str
            :return: Result of this operation.
            :rtype: Iterator[tuple[NodeSpec, str | None, str]]
            """
            if isinstance(value, NodeSpec):
                yield value, parent, path
                for key, child in value.children.items():
                    yield from nested(child, value.node_id, path + "." + key)
            elif isinstance(value, Mapping):
                for key, child in value.items():
                    yield from nested(child, parent, path + "." + key)
            else:
                for index, child in enumerate(value):
                    yield from nested(child, parent, path + "." + str(index))

        for index, node in enumerate(self.steps):
            yield from nested(node, None, str(index))

    def find(self, node_id: str) -> NodeSpec:
        """Return a placed node by stable identity; positions are not identifiers.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :return: Result of this operation.
        :rtype: NodeSpec
        """
        for node, _, _ in self.walk():
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, object]:
        """To dict.

        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        return {
            "steps": [node.to_dict() for node in self.steps],
            "autoencoder": plain(self.autoencoder),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """From dict.

        :param value: Value consumed by this conversion or transformation.
        :type value: Mapping[str, object]
        :return: Result of this operation.
        :rtype: Self
        """
        if set(value) - {"steps", "autoencoder"}:
            raise ValueError("Unknown pipeline schema fields.")
        return cls(
            tuple(NodeSpec.from_dict(node) for node in value["steps"]),
            value.get("autoencoder"),
        )

    def edit(
        self,
        node_id: str,
        *,
        replacement: Step | None = None,
        insert: Step | None = None,
        before: bool = True,
    ) -> Self:
        """Return an edited snapshot, preserving unaffected placement identities.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :param replacement: Replacement.
        :type replacement: Step | None
        :param insert: Insert.
        :type insert: Step | None
        :param before: Placement identity before which insertion occurs.
        :type before: bool
        :return: Result of this operation.
        :rtype: Self
        """
        data = self.to_dict()

        def locate_node(
            container: list[object] | dict[str, object],
        ) -> tuple[list[object] | dict[str, object], int | str] | None:
            """Locate node.

            :param container: Container.
            :type container: list[object] | dict[str, object]
            :return: Result of this operation.
            :rtype: tuple[list[object] | dict[str, object], int | str] | None
            """
            if isinstance(container, list):
                for index, item in enumerate(container):
                    if isinstance(item, dict) and item.get("node_id") == node_id:
                        return container, index
                    found = locate_node(item)
                    if found:
                        return found
            elif isinstance(container, dict):
                for key, item in container.items():
                    if isinstance(item, dict) and item.get("node_id") == node_id:
                        return container, key
                    if key in {"steps", "children"} or "node_id" not in container:
                        found = locate_node(item)
                        if found:
                            return found
            return None

        found = locate_node(data["steps"])
        if found is None:
            raise KeyError(node_id)
        owner, key = found
        if insert is not None:
            if not isinstance(owner, list):
                raise ValueError("Insertion requires an ordered step sequence.")
            owner.insert(key + (not before), NodeSpec.capture(insert).to_dict())
        elif replacement is not None:
            owner[key] = NodeSpec.capture(replacement, node_id=node_id).to_dict()
        elif isinstance(owner, list):
            del owner[key]
        else:
            raise ValueError(
                "A required role cannot be removed; replace its component."
            )
        return PipelineSchema.from_dict(data)
