"""Base contract for objective oracles installed as pipeline leaf steps."""

from abc import ABC

from pep_compass.optimization.engine.execution.step import Step


class Oracle(Step, ABC):
    """Evaluate candidates and attach objective fields to the returned batch.

    ``PipelineBuilder`` resolves YAML ``oracle`` declarations through
    ``OracleManager``. Implementations must provide :meth:`Step._execute`,
    respect ``context.state.remaining_oracle_calls()``, and report evaluated
    sequences through ``context.state.record_observations``. Objective values
    should use the ``oracle.<name>.score`` field convention consumed by
    ``PepCompassPipeline._summarize``. Implementations must not override
    :meth:`Step.__call__`.
    """
