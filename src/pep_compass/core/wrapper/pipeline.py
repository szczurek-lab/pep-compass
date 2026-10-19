"""Convenient Python facade over graph construction and independent execution."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, ParamSpec, TypeVar, overload

from pep_compass.core.builder import PipelineBuilder
from pep_compass.data_schemas.execution import RunnerSettings
from pep_compass.data_schemas.graph import PipelineSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Literal, Self

    import pandas as pd

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.resource import AutoencoderResource
    from pep_compass.core.builder.builder import _Scope
    from pep_compass.core.wrapper.visualization import GraphView
    from pep_compass.data_schemas.candidates import BatchField
    from pep_compass.data_schemas.execution import (
        ExecutionSettings,
        ResultsSettings,
        StorageSettings,
    )
    from pep_compass.data_schemas.results import RunResult
    from pep_compass.optimization.engine.pipeline import ExecutablePipeline
    from pep_compass.optimization.saving.reader import AnalysisReader
    from pep_compass.steps.base import Step
    from pep_compass.utils.configuration import Configurable


_P = ParamSpec("_P")
_R = TypeVar("_R")


class Pipeline:
    """Combine an editable builder, run defaults and the most recent result.

    :param steps: Ordered concrete Step objects, without initialized model weights.
    :param autoencoder: Shared resource declaration or initialized Python resource.
    :param schema: Alternative immutable declaration snapshot.
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
        self.builder = PipelineBuilder(steps, autoencoder=autoencoder, schema=schema)
        self._compiled: ExecutablePipeline | None = None
        self._compiled_schema: PipelineSchema | None = None
        self._compiled_device: str | None = None
        self._settings = RunnerSettings()
        self._lock = RLock()
        self._running = False
        self.last_result: RunResult | None = None

    @classmethod
    def from_schema(
        cls,
        schema: PipelineSchema | str | Mapping[str, object],
        *,
        autoencoder: AutoencoderResource | Autoencoder | None = None,
    ) -> Self:
        """Restore a portable snapshot without loading model weights.

        :param schema: Immutable declaration or its explicitly supported serialized representation.
        :type schema: PipelineSchema | str | Mapping[str, object]
        :param autoencoder: Shared numerical model or deferred resource, as declared by the parameter type.
        :type autoencoder: AutoencoderResource | Autoencoder | None
        :return: Result of this operation.
        :rtype: Self
        """
        if isinstance(schema, str):
            schema = json.loads(schema)
        if isinstance(schema, dict):
            schema = PipelineSchema.from_dict(schema)
        if not isinstance(schema, PipelineSchema):
            raise TypeError("Expected PipelineSchema, JSON or a schema mapping.")
        return cls(schema=schema, autoencoder=autoencoder)

    def _edit(
        self, action: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        """Edit.

        :param action: Action.
        :type action: Callable[_P, _R]
        :param args: Additional positional arguments forwarded to the underlying interface.
        :type args: _P.args
        :param kwargs: Additional named arguments forwarded to the underlying interface.
        :type kwargs: _P.kwargs
        :return: Result of this operation.
        :rtype: _R
        """
        with self._lock:
            if self._running:
                raise RuntimeError("An executing pipeline cannot be edited.")
            result = action(*args, **kwargs)
            self._compiled = None
            return result

    def add_component(self, component: Step, *, name: str | None = None) -> str:
        """Append a concrete component and return its stable placement identity.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :return: Result of this operation.
        :rtype: str
        """
        return self._edit(self.builder.add, component, name=name)

    def add_autoencoder(self, component: AutoencoderResource | Autoencoder) -> Self:
        """Attach a shared autoencoder declaration and return this wrapper.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: AutoencoderResource | Autoencoder
        :return: Result of this operation.
        :rtype: Self
        """
        self._edit(self.builder.add_autoencoder, component)
        return self

    def insert_component(
        self, component: Step, *, before: str | None = None, after: str | None = None
    ) -> str:
        """Insert next to one placement and return the new stable identity.

        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :param before: Placement identity before which insertion occurs.
        :type before: str | None
        :param after: Placement identity after which insertion occurs.
        :type after: str | None
        :return: Result of this operation.
        :rtype: str
        """
        return self._edit(self.builder.insert, component, before=before, after=after)

    def replace_component(self, node_id: str, component: Step) -> str:
        """Replace configuration while keeping the placement identity.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :param component: Concrete component or shared resource accepted by this operation.
        :type component: Step
        :return: Result of this operation.
        :rtype: str
        """
        return self._edit(self.builder.replace, node_id, component)

    def remove_component(self, node_id: str) -> None:
        """Remove a placement and its owned descendants.

        :param node_id: Stable graph placement identity.
        :type node_id: str
        :rtype: None
        """
        return self._edit(self.builder.remove, node_id)

    def loop(self, *, iterations: int, name: str | None = None) -> _Scope:
        """Collect a loop using a child builder context manager.

        :param iterations: Number of complete logical iterations.
        :type iterations: int
        :param name: Name identifying this object or recorded artifact.
        :type name: str | None
        :return: Result of this operation.
        :rtype: _Scope
        """
        return self._edit(self.builder.loop, iterations=iterations, name=name)

    def validate(
        self, *, initial_fields: Sequence[str] | Mapping[str, BatchField] = ()
    ) -> list[dict[str, str]]:
        """Validate graph structure and known data dependencies without weights.

        :param initial_fields: Fields available on incoming candidates before the graph starts.
        :type initial_fields: Sequence[str] | Mapping[str, BatchField]
        :return: Result of this operation.
        :rtype: list[dict[str, str]]
        """
        return self.builder.validate(initial_fields=initial_fields)

    def build(
        self, *, execution: ExecutionSettings | Mapping[str, object] | None = None
    ) -> ExecutablePipeline:
        """Return an independent ExecutablePipeline built from the current schema.

        :param execution: Execution settings or a mapping overriding their defaults.
        :type execution: ExecutionSettings | Mapping[str, object] | None
        :return: Result of this operation.
        :rtype: ExecutablePipeline
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot build during execution.")
            settings = self._settings.update(execution=execution)
            self._compiled = self.builder.build(execution=settings.execution)
            self._compiled_schema = self.builder.schema
            self._compiled_device = settings.execution.device
            return self._compiled

    def set_runner_settings(
        self,
        *,
        execution: ExecutionSettings | Mapping[str, object] | None = None,
        storage: StorageSettings | Mapping[str, object] | None = None,
        results: ResultsSettings | Mapping[str, object] | None = None,
    ) -> Self:
        """Set validated run defaults; per-run overrides leave these unchanged.

        :param execution: Execution settings or a mapping overriding their defaults.
        :type execution: ExecutionSettings | Mapping[str, object] | None
        :param storage: Storage settings or a mapping overriding their defaults.
        :type storage: StorageSettings | Mapping[str, object] | None
        :param results: Result-view settings or a mapping overriding their defaults.
        :type results: ResultsSettings | Mapping[str, object] | None
        :return: Result of this operation.
        :rtype: Self
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot change settings during execution.")
            self._settings = self._settings.update(
                execution=execution, storage=storage, results=results
            )
        return self

    @property
    def runner_settings(self) -> RunnerSettings:
        """Return immutable execution, storage and result defaults.

        :return: Result of this operation.
        :rtype: RunnerSettings
        """
        return self._settings

    def run(
        self,
        sequences: Sequence[str],
        *,
        execution: Mapping[str, object] | None = None,
        storage: Mapping[str, object] | None = None,
        results: Mapping[str, object] | None = None,
        seed: int | None = None,
    ) -> RunResult | AnalysisReader | Path:
        """Run with fresh state; build again after graph or device changes.

        :param sequences: Non-empty collection of peptide strings.
        :param execution: Overrides for seed, device and complete-batch limits.
        :param storage: Overrides for memory/SQLite and retained observations.
        :param results: Overrides for the public return mode.
        :param seed: Convenient deterministic seed override.
        :return: RunResult, AnalysisReader or persistent output path.

        :type sequences: Sequence[str]
        :type execution: Mapping[str, object] | None
        :type storage: Mapping[str, object] | None
        :type results: Mapping[str, object] | None
        :type seed: int | None
        :rtype: RunResult | AnalysisReader | Path
        """
        from pep_compass.utils.validation import validate_sequences

        sequences = validate_sequences(sequences)
        with self._lock:
            if self._running:
                raise RuntimeError(
                    "Concurrent runs require independent pipeline instances."
                )
            self.builder._ensure_closed()
            options = dict(execution or {})
            if seed is not None:
                options["seed"] = seed
            settings = self._settings.update(
                execution=options, storage=storage, results=results
            )
            if settings.storage.reconstruction == "none" and any(
                node.category == "save" for node, _, _ in self.builder.schema.walk()
            ):
                raise ValueError(
                    "Snapshots require reconstruction=boundaries or trajectories."
                )
            if (
                self._compiled is None
                or self._compiled_schema is not self.builder.schema
                or self._compiled_device != settings.execution.device
            ):
                self.build(execution=settings.execution)
            self._running = True
        try:
            result = self._compiled.run(sequences, settings=settings)
            self.last_result = result
            return {"result": result, "reader": result.reader, "path": result.path}[
                settings.results.return_mode
            ]
        finally:
            with self._lock:
                self._running = False

    @overload
    def describe(
        self, category: str | None = None, *, display: Literal["raw"]
    ) -> list[dict[str, object]]: ...

    @overload
    def describe(
        self, category: str | None = None, *, display: Literal["pd"]
    ) -> pd.DataFrame: ...

    @overload
    def describe(
        self, category: str | None = None, *, display: Literal["display"] = "display"
    ) -> None: ...

    def describe(
        self,
        category: str | None = None,
        *,
        display: Literal["raw", "pd", "display"] = "display",
    ) -> list[dict[str, object]] | pd.DataFrame | None:
        """List public importable components without constructing their models.

        :param category: Optional public steps namespace or autoencoder.
        :param display: ``raw``, ``pd`` or automatic formatted ``display``.
        :return: Records, a pandas DataFrame, or None after presentation.

        :type category: str | None
        :type display: Literal['raw', 'pd', 'display']
        :rtype: list[dict[str, object]] | pd.DataFrame | None
        """
        from pep_compass.core.wrapper.descriptions import available, present_description

        return present_description(available(category), display)

    @overload
    def describe_component(
        self,
        component: Configurable
        | type[Configurable]
        | Callable[..., Configurable]
        | None = None,
        *,
        node_id: str | None = None,
        display: Literal["raw"],
        technical: bool = False,
    ) -> dict[str, object]: ...

    @overload
    def describe_component(
        self,
        component: Configurable
        | type[Configurable]
        | Callable[..., Configurable]
        | None = None,
        *,
        node_id: str | None = None,
        display: Literal["pd"],
        technical: bool = False,
    ) -> pd.DataFrame: ...

    @overload
    def describe_component(
        self,
        component: Configurable
        | type[Configurable]
        | Callable[..., Configurable]
        | None = None,
        *,
        node_id: str | None = None,
        display: Literal["display"] = "display",
        technical: bool = False,
    ) -> None: ...

    def describe_component(
        self,
        component: Configurable
        | type[Configurable]
        | Callable[..., Configurable]
        | None = None,
        *,
        node_id: str | None = None,
        display: Literal["raw", "pd", "display"] = "display",
        technical: bool = False,
    ) -> dict[str, object] | pd.DataFrame | None:
        """Describe a concrete class, configured object or existing graph placement.

        :param component: Public component class, function or configured instance.
        :param node_id: Alternative stable identity of a placed graph node.
        :param display: ``raw``, ``pd`` or formatted ``display``.
        :param technical: Include resources and CandidateBatch field contracts.
        :return: Definition records, DataFrame or None after presentation.

        :type component: Configurable | type[Configurable] | Callable[..., Configurable] | None
        :type node_id: str | None
        :type display: Literal['raw', 'pd', 'display']
        :type technical: bool
        :rtype: dict[str, object] | pd.DataFrame | None
        """
        from pep_compass.core.wrapper.descriptions import (
            definition,
            present_description,
        )

        if (component is None) == (node_id is None):
            raise ValueError("Select a component or a node_id.")
        if node_id is not None:
            component = self.builder.schema.find(node_id).instantiate()
        return present_description(definition(component), display, technical=technical)

    @overload
    def to_schema(self, format: Literal["object"] = "object") -> PipelineSchema: ...

    @overload
    def to_schema(self, format: Literal["json", "text", "mermaid", "svg"]) -> str: ...

    def to_schema(
        self, format: Literal["object", "json", "text", "mermaid", "svg"] = "object"
    ) -> PipelineSchema | str:
        """Return a schema snapshot, JSON or its text, Mermaid or SVG rendering.

        :param format: Format.
        :type format: Literal['object', 'json', 'text', 'mermaid', 'svg']
        :return: Result of this operation.
        :rtype: PipelineSchema | str
        """
        from pep_compass.core.wrapper.visualization import render

        return render(self.builder.schema, format=format)

    def show(self, format: Literal["text", "mermaid", "svg"] = "svg") -> GraphView:
        """Return a portable graph string with inline SVG support in notebooks.

        :param format: Format.
        :type format: Literal['text', 'mermaid', 'svg']
        :return: Result of this operation.
        :rtype: GraphView
        """
        from pep_compass.core.wrapper.visualization import graph_view

        return graph_view(self.builder.schema, format)

    def show_pipeline_graph(
        self, format: Literal["text", "mermaid", "svg"] = "svg"
    ) -> GraphView:
        """Display the current computational graph using the selected representation.

        :param format: Format.
        :type format: Literal['text', 'mermaid', 'svg']
        :return: Result of this operation.
        :rtype: GraphView
        """
        return self.show(format)

    def save_schema(self, path: str | Path) -> Path:
        """Persist the constructor graph independently of run results.

        :param path: Input or output path used by this operation.
        :type path: str | Path
        :return: Result of this operation.
        :rtype: Path
        """
        path = Path(path)
        path.write_text(self.to_schema("json"), encoding="utf-8")
        return path

    def save_result(self, path: str | Path) -> Path:
        """Export the retained last result through the common storage backend.

        :param path: Input or output path used by this operation.
        :type path: str | Path
        :return: Result of this operation.
        :rtype: Path
        """
        if self.last_result is None:
            raise RuntimeError("No completed result is available.")
        return self.last_result.save(path)
