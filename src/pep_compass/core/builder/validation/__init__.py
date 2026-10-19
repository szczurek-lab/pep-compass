"""Validate ownership and symbolic field flow before loading model weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pep_compass.steps.base import Step
from pep_compass.steps.flow.strategies.sequence import Flow
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    from pep_compass.data_schemas.candidates import BatchField
    from pep_compass.data_schemas.contracts import StepContract
    from pep_compass.data_schemas.graph import PipelineSchema


logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class _Transfer:
    """Summarize fields required, propagated and unconditionally produced."""

    required: frozenset
    propagated: frozenset
    produced: frozenset

    def then(self, following: _Transfer) -> _Transfer:
        """Compose two field transfers without executing either component.

        :param following: Following.
        :type following: _Transfer
        :return: Result of this operation.
        :rtype: _Transfer
        """
        external = following.required - self.produced
        unavailable = external - self.propagated
        if unavailable:
            raise ValueError(
                f"Graph requires unavailable fields: {sorted(unavailable)}."
            )
        return _Transfer(
            self.required | external,
            self.propagated & following.propagated,
            (self.produced & following.propagated) | following.produced,
        )


@dataclass(frozen=True)
class _FieldFlow:
    """Track fully populated fields separately from fields allowing partial rows."""

    full: _Transfer
    present: _Transfer

    def then(self, following: _FieldFlow) -> _FieldFlow:
        """Then.

        :param following: Following.
        :type following: _FieldFlow
        :return: Result of this operation.
        :rtype: _FieldFlow
        """
        return _FieldFlow(
            self.full.then(following.full), self.present.then(following.present)
        )

    @classmethod
    def join(cls, branches: Sequence[_FieldFlow]) -> Self:
        """Concatenation preserves full fields in every branch and partial fields in any branch.

        :param branches: Branches.
        :type branches: Sequence[_FieldFlow]
        :return: Result of this operation.
        :rtype: Self
        """

        def combine(flows: Sequence[_Transfer], all_branches: bool) -> _Transfer:
            """Combine.

            :param flows: Flows.
            :type flows: Sequence[_Transfer]
            :param all_branches: All branches.
            :type all_branches: bool
            :return: Result of this operation.
            :rtype: _Transfer
            """
            required = frozenset.union(*(flow.required for flow in flows))
            if all_branches:
                produced = frozenset.intersection(*(flow.produced for flow in flows))
                propagated = frozenset.intersection(
                    *(flow.propagated | flow.produced for flow in flows)
                )
            else:
                produced = frozenset.union(*(flow.produced for flow in flows))
                propagated = frozenset.union(*(flow.propagated for flow in flows))
            return _Transfer(required, propagated, produced)

        return cls(
            combine([branch.full for branch in branches], True),
            combine([branch.present for branch in branches], False),
        )


def _children(value: object) -> Iterator[Step]:
    """Yield the actual step objects from named graph ownership slots.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: Iterator[Step]
    """
    if isinstance(value, Step):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _children(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _children(item)
    else:
        raise TypeError("Graph ownership slots must contain Step objects.")


def validate(
    schema: PipelineSchema,
    *,
    initial_fields: Sequence[str] | Mapping[str, BatchField] = (),
) -> list[dict[str, str]]:
    """Validate graph structure and field availability without model initialization.

    Each node contributes one symbolic transfer. Sequential composition, branch
    joins and loop feedback use those summaries rather than revisiting subtrees.

    :param schema: Immutable PipelineSchema.
    :param initial_fields: Extra incoming CandidateBatch field names.
    :return: Diagnostic records for a structurally valid graph.
    :raises ValueError: On repeated identities or unavailable required fields.
    :raises TypeError: If a graph child does not implement Step.

    :type schema: PipelineSchema
    :type initial_fields: Sequence[str] | Mapping[str, BatchField]
    :rtype: list[dict[str, str]]
    """
    identities = [node.node_id for node, _, _ in schema.walk()]
    if any(not isinstance(value, str) or not value for value in identities) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("Graph identities must be unique, non-empty strings.")
    from pep_compass.data_schemas.candidates import OptionalField

    base_fields = {"lineage.candidate_id", "lineage.parent_candidate_id"}
    incoming = frozenset(base_fields | set(initial_fields))
    incomplete = (
        {
            name
            for name, value in initial_fields.items()
            if isinstance(value, OptionalField)
        }
        if isinstance(initial_fields, Mapping)
        else set()
    )
    full_incoming = incoming - incomplete
    root = Flow([node.instantiate() for node in schema.steps])
    contracts = {}
    objective_configurations = {}
    universe = set(incoming)

    # Resolve each declared contract once; unknown fields cannot affect a later requirement.
    def collect(step: Step) -> None:
        """Collect.

        :param step: Step.
        :type step: Step
        :rtype: None
        """
        if not isinstance(step, Step):
            raise TypeError("Graph components must implement Step.")
        from pep_compass.steps.oracles.strategies.black_box import BlackBoxOracle

        if isinstance(step, BlackBoxOracle):
            declared = step.to_config()
            previous = objective_configurations.setdefault(step.objective, declared)
            if previous != declared:
                raise ValueError(
                    f"Conflicting configurations for objective {step.objective!r}."
                )
        contract = step.contract()
        contracts[id(step)] = contract
        universe.update(contract.required_fields)
        universe.update(contract.produced_fields)
        for child in _children(step.graph_children()):
            collect(child)

    collect(root)
    universe.update(
        {
            "local_enumeration.center_sequence",
            "tracking.trajectory_id",
            "tracking.trajectory_index",
            "tracking.trajectory_step",
            "tracking.rng_stream_seed",
        }
    )
    universe = frozenset(universe)
    identity_transfer = _Transfer(frozenset(), universe, frozenset())
    identity = _FieldFlow(identity_transfer, identity_transfer)

    def leaf(contract: StepContract) -> _FieldFlow:
        """Leaf.

        :param contract: Contract.
        :type contract: StepContract
        :return: Result of this operation.
        :rtype: _FieldFlow
        """
        propagated = frozenset(
            field
            for field in universe
            if not any(
                field.startswith(prefix) for prefix in contract.invalidated_prefixes
            )
        )
        produced = frozenset(contract.produced_fields)
        return _FieldFlow(
            _Transfer(
                frozenset(contract.required_fields) - set(contract.optional_fields),
                propagated,
                produced,
            ),
            _Transfer(frozenset(contract.required_fields), propagated, produced),
        )

    def sequence(steps: Sequence[Step], local: bool) -> _FieldFlow:
        """Sequence.

        :param steps: Ordered child step instances.
        :type steps: Sequence[Step]
        :param local: Local.
        :type local: bool
        :return: Result of this operation.
        :rtype: _FieldFlow
        """
        result = identity
        for step in steps:
            result = result.then(summarize(step, local))
        return result

    def summarize(step: Step, local: bool = False) -> _FieldFlow:
        """Summarize.

        :param step: Step.
        :type step: Step
        :param local: Local.
        :type local: bool
        :return: Result of this operation.
        :rtype: _FieldFlow
        """
        if getattr(step, "requires_local_enumeration", False) and not local:
            raise ValueError(f"{step.name} requires a LocalEnumeration context.")
        own = leaf(contracts[id(step)])
        slots = step.graph_children()
        if step.kind in {"sequence", "loop"}:
            body = sequence(slots["steps"], local)
            if step.kind == "loop":
                if step.iterations == 0:
                    return own
                # f(I) = (I intersect K) union P is idempotent. Checking required
                # fields after one body proves availability on every feedback edge.
                if step.iterations > 1:
                    for transfer in (body.full, body.present):
                        missing = transfer.required - (
                            transfer.propagated | transfer.produced
                        )
                        if missing:
                            raise ValueError(
                                f"Loop feedback requires unavailable fields: {sorted(missing)}."
                            )
            return own.then(body)
        if step.kind == "parallel":
            branches = [
                sequence(branch, local) for branch in slots["branches"].values()
            ]
            return own.then(_FieldFlow.join(branches))
        if step.kind == "local_enumeration":
            provided = universe & {
                "local_enumeration.center_sequence",
                "tracking.trajectory_id",
                "tracking.trajectory_index",
                "tracking.trajectory_step",
                "tracking.rng_stream_seed",
            }
            initialize_transfer = _Transfer(frozenset(), universe, provided)
            initialize = _FieldFlow(initialize_transfer, initialize_transfer)
            walked = initialize.then(summarize(slots["walker"], True))
            generated = walked.then(summarize(slots["mutation_generator"], True))
            pooled = generated.then(sequence(slots["filters"], True))
            from pep_compass.steps.flow.strategies.local_enumeration import (
                _TRANSIENT_WALKER_FIELDS,
            )

            retained = universe - set(_TRANSIENT_WALKER_FIELDS)
            if step.include_walk_points:
                pooled = _FieldFlow.join([walked, pooled])
            drop = _Transfer(frozenset(), retained, frozenset())
            return own.then(pooled).then(_FieldFlow(drop, drop))
        return own

    flow = summarize(root)
    missing = (flow.full.required - full_incoming) | (flow.present.required - incoming)
    if missing:
        raise ValueError(f"Pipeline requires unavailable fields: {sorted(missing)}.")
    logger.debug(
        "Validated graph nodes=%s field_names=%s.", len(identities), len(universe)
    )
    return []
