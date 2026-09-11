"""Base contract for mutation generators installed as pipeline leaf steps."""

from abc import ABC

from pep_compass.optimization.engine.execution.step import Step


class MutationGenerator(Step, ABC):
    """Generate candidate sequences from latent origins or prepared geometry.

    ``PipelineBuilder`` resolves YAML ``mutation_generator`` declarations
    through ``MutationGeneratorManager``. Implementations must provide
    :meth:`Step._execute`, return a valid ``CandidateBatch``, and report newly
    generated candidates through
    ``context.state.record_generated_candidates``. Use
    :meth:`Step.precompute` only for run-static work and
    :meth:`Step.prepare_iteration` for work dependent on the incoming batch.
    Implementations must not override :meth:`Step.__call__`.
    """
