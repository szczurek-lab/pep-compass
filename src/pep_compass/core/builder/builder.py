"""Own graph edits, immutable declarations and compilation into executable objects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pep_compass.autoencoder.resource import AutoencoderResource
from pep_compass.core.builder.validation import validate
from pep_compass.data_schemas.execution import ExecutionSettings, RunnerSettings
from pep_compass.data_schemas.graph import NodeSpec, PipelineSchema
from pep_compass.steps.flow.strategies.sequence import Flow
from pep_compass.utils.configuration import construct
from pep_compass.utils.logger import get_custom_logger
from pep_compass.utils.randomness import random_scope

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType
    from typing import Self

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.data_schemas.candidates import BatchField
    from pep_compass.optimization.engine.pipeline import ExecutablePipeline
    from pep_compass.steps.base import Step


logger = get_custom_logger(__name__)


class PipelineBuilder:
    """Collect component placements; every build constructs an independent graph.

    :param steps: Ordered lightweight Step instances.
    :param autoencoder: Lightweight resource declaration or an initialized resource.
    :param schema: Alternative immutable graph declaration.
    """

    def __init__(
        self,
        steps: Sequence[Step] = (),
        *,
        autoencoder: AutoencoderResource | Autoencoder | None = None,
        schema: PipelineSchema | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param steps: Ordered child step instances.
        :type steps: Sequence[Step]
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: AutoencoderResource | Autoencoder | None
        :param schema: Immutable declaration or its explicitly supported serialized representation.
        :type schema: PipelineSchema | None
        :rtype: None
        """
        if schema is not None and steps:
            raise ValueError("Select steps or schema, not both.")
        self._schema = schema or PipelineSchema(
            tuple(NodeSpec.capture(step) for step in steps)
        )
        self._provided_autoencoder = None
        self._open_scopes = 0
        if autoencoder is not None:
            self.add_autoencoder(autoencoder)

    @property
    def schema(self) -> PipelineSchema:
        """Return the current immutable declaration snapshot.

        :return: Result of this operation.
        :rtype: PipelineSchema
        """
        return self._schema

    def add(self, component: Step, *, name: str | None = None) -> str:
        """Append a component; return its stable placement identity.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :return: Result of this operation.
        :rtype: str
        """
        node = NodeSpec.capture(component, name=name)
        return self._append_node(node)

    def _append_node(self, node: NodeSpec) -> str:
        """Append node.

        :param node: Node.
        :type node: NodeSpec
        :return: Result of this operation.
        :rtype: str
        """
        self._schema = replace(self._schema, steps=(*self._schema.steps, node))
        return node.node_id

    def add_autoencoder(self, component: AutoencoderResource | Autoencoder) -> Self:
        """Attach one shared resource declaration or initialized Python resource.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: AutoencoderResource | Autoencoder
        :return: Result of this operation.
        :rtype: Self
        """
        from pep_compass.autoencoder.base import Autoencoder

        if not isinstance(component, (AutoencoderResource, Autoencoder)):
            raise TypeError(
                "Expected an AutoencoderResource declaration or an initialized Autoencoder."
            )
        self._provided_autoencoder = (
            None if isinstance(component, AutoencoderResource) else component
        )
        self._schema = replace(
            self._schema,
            autoencoder=component.to_config()
            if isinstance(component, AutoencoderResource)
            else None,
        )
        return self

    def insert(
        self, component: Step, *, before: str | None = None, after: str | None = None
    ) -> str:
        """Insert in an ordered sequence; return the inserted placement identity.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :param before: Placement identity before which insertion occurs.
        :type before: str | None
        :param after: Placement identity after which insertion occurs.
        :type after: str | None
        :return: Result of this operation.
        :rtype: str
        """
        if (before is None) == (after is None):
            raise ValueError("Select exactly one of before or after.")
        previous = {node.node_id for node, _, _ in self._schema.walk()}
        self._schema = self._schema.edit(
            before or after, insert=component, before=before is not None
        )
        return next(
            node.node_id
            for node, _, _ in self._schema.walk()
            if node.node_id not in previous
        )

    def replace(self, node_id: str, component: Step) -> str:
        """Replace a declaration while retaining its identity.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :return: Result of this operation.
        :rtype: str
        """
        self._schema = self._schema.edit(node_id, replacement=component)
        return node_id

    def remove(self, node_id: str) -> None:
        """Remove a placement and its owned descendants.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :rtype: None
        """
        self._schema = self._schema.edit(node_id)

    def loop(self, *, iterations: int, name: str | None = None) -> _Scope:
        """Collect a loop body through the same builder API in a context manager.

        :param iterations: Number of complete logical iterations.
        :type iterations: int
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :return: Result of this operation.
        :rtype: _Scope
        """
        from pep_compass.steps.flow import Loop

        return _Scope(
            self, lambda steps: Loop(steps=steps, iterations=iterations), name
        )

    def _ensure_closed(self) -> None:
        """Ensure closed.

        :rtype: None
        """
        if self._open_scopes:
            raise RuntimeError(
                "Close all construction scopes before validation or build."
            )

    def validate(
        self, *, initial_fields: Sequence[str] | Mapping[str, BatchField] = ()
    ) -> list[dict[str, str]]:
        """Check structure and field flow before initializing any resources.

        :param initial_fields: Fields available on incoming candidates before the graph starts.
        :type initial_fields: Sequence[str] | Mapping[str, BatchField]
        :return: Result of this operation.
        :rtype: list[dict[str, str]]
        """
        self._ensure_closed()
        return validate(self._schema, initial_fields=initial_fields)

    def build(
        self,
        *,
        execution: ExecutionSettings | None = None,
        initial_fields: Sequence[str] | Mapping[str, BatchField] = (),
    ) -> ExecutablePipeline:
        """Validate and return a fresh ExecutablePipeline for this snapshot.

        :param execution: Resource initialization seed and device settings.
        :param initial_fields: Additional fields available to graph inputs.
        :return: Independent prepared computation graph.
        :raises ValueError: If declarations or required resource settings are invalid.

        :type execution: ExecutionSettings | None
        :type initial_fields: Sequence[str] | Mapping[str, BatchField]
        :rtype: ExecutablePipeline
        """
        from pep_compass.autoencoder import HydrAMP
        from pep_compass.optimization.engine.pipeline import ExecutablePipeline
        from pep_compass.optimization.saving.artifacts import capture_model_artifacts

        self.validate(initial_fields=initial_fields)
        execution = execution or ExecutionSettings()
        schema = self._schema
        if schema.autoencoder is None and self._provided_autoencoder is None:
            schema = replace(schema, autoencoder=HydrAMP().to_config())
        with capture_model_artifacts() as artifacts, random_scope(execution.seed):
            resource = (
                construct(schema.autoencoder)
                if schema.autoencoder is not None
                else self._provided_autoencoder
            )
            autoencoder = (
                resource.build(device=execution.device)
                if isinstance(resource, AutoencoderResource)
                else resource
            )
            root = Flow([node.instantiate() for node in schema.steps])
            root.initialize(autoencoder)
        logger.info("Compiled graph nodes=%s.", sum(1 for _ in schema.walk()))
        return ExecutablePipeline(
            autoencoder=autoencoder,
            root=root,
            schema=schema,
            model_manifest=tuple(artifacts),
            settings=RunnerSettings(execution=execution),
        )


class _Scope:
    """Commit an isolated child builder only after a successful scope exit."""

    def __init__(
        self,
        owner: PipelineBuilder,
        factory: Callable[[Sequence[Step]], Step],
        name: str | None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param owner: Owner.
        :type owner: PipelineBuilder
        :param factory: Factory.
        :type factory: Callable[[Sequence[Step]], Step]
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :rtype: None
        """
        self.owner, self.factory, self.name = owner, factory, name
        self.builder = PipelineBuilder()

    def __enter__(self) -> PipelineBuilder:
        """Enter the managed resource scope.

        :return: Result of this operation.
        :rtype: PipelineBuilder
        """
        self.owner._open_scopes += 1
        return self.builder

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Finalize the managed scope and propagate exceptions.

        :param kind: Kind of recorded artifact or exception type, as declared by this operation.
        :type kind: type[BaseException] | None
        :param value: Value consumed by this conversion or transformation.
        :type value: BaseException | None
        :param traceback: Original exception traceback, if present.
        :type traceback: TracebackType | None
        :return: Result of this operation.
        :rtype: bool
        """
        self.owner._open_scopes -= 1
        if kind is None:
            node = NodeSpec.capture(
                self.factory(
                    [child.instantiate() for child in self.builder.schema.steps]
                ),
                name=self.name,
            )
            self.owner._append_node(
                replace(node, children={"steps": self.builder.schema.steps})
            )
        return False
