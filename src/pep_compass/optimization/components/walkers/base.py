"""Base contract for walker components installed as pipeline leaf steps."""

from abc import ABC

from pep_compass.optimization.engine.execution.step import Step


class Walker(Step, ABC):
    """Move candidates through latent space and return an aligned batch.

    ``PipelineBuilder`` resolves a YAML ``walker`` declaration through
    ``WalkerManager`` and places the resulting object in the executable step
    tree. Implementations must provide :meth:`Step._execute`; they may override
    :meth:`Step.precompute` for batch-independent geometry and
    :meth:`Step.prepare_iteration` for geometry derived from the current
    candidates. They must not override :meth:`Step.__call__`.

    The returned ``CandidateBatch`` must keep sequences, latent origins, and
    candidate fields aligned. Generation counters belong to mutation
    generators, not walkers.
    """
