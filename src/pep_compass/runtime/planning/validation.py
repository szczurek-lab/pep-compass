"""Validation helpers for materialized execution plans."""

from pep_compass.runtime.planning.plan import ExecutionPlan


def validate_execution_plan(plan: ExecutionPlan) -> None:
    """Validate plan identity and seed invariants."""
    indices = [entry.index for entry in plan.entries]
    run_ids = [entry.run_id for entry in plan.entries]
    if len(indices) != len(set(indices)):
        raise ValueError("Execution plan contains duplicate indices.")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Execution plan contains duplicate run identifiers.")
