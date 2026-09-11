"""Base contracts for filters installed as pipeline leaf steps."""

from abc import ABC

from pep_compass.optimization.engine.execution.step import Step


class Filter(Step, ABC):
    """Retain, rank, or annotate candidates according to one policy.

    ``PipelineBuilder`` resolves YAML ``filter`` declarations through
    ``FilterManager``. Implementations must provide :meth:`Step._execute` and
    preserve alignment between selected sequences, latent origins, and fields.
    Batch-independent preparation belongs in :meth:`Step.precompute`, while
    incoming-batch preparation belongs in :meth:`Step.prepare_iteration`.
    Implementations must not override :meth:`Step.__call__`.
    """
