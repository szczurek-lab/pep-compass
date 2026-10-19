"""Exhaustive finite-domain verification of symbolic graph field transfers."""

from itertools import product

import pytest

from pep_compass.core.builder.validation import _FieldFlow, _Transfer

FIELDS = frozenset({"a", "b"})
SUBSETS = tuple(
    frozenset(field for field, included in zip(sorted(FIELDS), mask) if included)
    for mask in product((False, True), repeat=2)
)


def apply(transfer, incoming):
    """Execute the declared field contract in an independent tiny set model."""
    if not transfer.required <= incoming:
        return None
    return frozenset(
        field
        for field in FIELDS
        if field in transfer.produced
        or (field in incoming and field in transfer.propagated)
    )


def test_transfer_composition_matches_all_inputs_in_a_two_field_domain():
    transfers = tuple(
        _Transfer(required, kept, produced)
        for required, kept, produced in product(SUBSETS, repeat=3)
    )
    for first, second in product(transfers, repeat=2):
        reference = {}
        for incoming in SUBSETS:
            intermediate = apply(first, incoming)
            result = apply(second, intermediate) if intermediate is not None else None
            reference[incoming] = result
        if all(value is None for value in reference.values()):
            with pytest.raises(ValueError):
                first.then(second)
        else:
            composed = first.then(second)
            assert {
                incoming: apply(composed, incoming) for incoming in SUBSETS
            } == reference


def test_branch_join_preserves_partial_presence_and_full_row_guarantees():
    leaves = tuple(
        _FieldFlow(
            _Transfer(frozenset(), kept, produced),
            _Transfer(frozenset(), kept, produced),
        )
        for kept, produced in product(SUBSETS, repeat=2)
    )
    for branches in product(leaves, repeat=2):
        joined = _FieldFlow.join(branches)
        for complete, present in product(SUBSETS, repeat=2):
            if not complete <= present:
                continue
            full_outputs = [apply(branch.full, complete) for branch in branches]
            present_outputs = [apply(branch.present, present) for branch in branches]
            assert apply(joined.full, complete) == frozenset.intersection(*full_outputs)
            assert apply(joined.present, present) == frozenset.union(*present_outputs)
