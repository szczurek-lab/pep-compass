"""Registry of normalized tracker data schemas."""

from pep_compass.analysis.reader.data_schemas.base import DataSchema


SCHEMAS = {
    "steps": DataSchema(
        "steps",
        "steps.csv",
        frozenset(
            {
                "execution_id",
                "step_name",
                "path",
                "input_size",
                "output_size",
                "duration_seconds",
            }
        ),
    ),
    "candidates": DataSchema(
        "candidates",
        "candidates.csv",
        frozenset({"execution_id", "candidate_index", "sequence"}),
    ),
    "trajectory_points": DataSchema(
        "trajectory_points",
        "trajectory_points.csv",
        frozenset(
            {
                "execution_id",
                "trajectory_id",
                "trajectory_step",
                "sequence",
                "latent_index",
            }
        ),
    ),
    "local_enumerations": DataSchema(
        "local_enumerations",
        "local_enumerations.csv",
        frozenset(
            {
                "execution_id",
                "input_count",
                "output_count",
                "output_sequences_sha256",
            }
        ),
    ),
    "stability": DataSchema(
        "stability",
        "stability.csv",
        frozenset({"label", "candidates", "rss_bytes"}),
    ),
    "evaluations": DataSchema(
        "evaluations",
        "evaluations.csv",
        frozenset({"run_id", "iteration_id", "candidate_id", "sequence"}),
    ),
    "optimizer_iterations": DataSchema(
        "optimizer_iterations",
        "optimizer_iterations.csv",
        frozenset(
            {
                "run_id",
                "iteration_id",
                "trajectory_count",
                "walker_step_count",
            }
        ),
    ),
    "walker_steps": DataSchema(
        "walker_steps",
        "walker_steps.csv",
        frozenset(
            {
                "run_id",
                "iteration_id",
                "trajectory_id",
                "step_id",
                "current_latent_position",
                "next_latent_position",
            }
        ),
    ),
    "unique_candidates": DataSchema(
        "unique_candidates",
        "unique_candidates.csv",
        frozenset({"run_id", "candidate_id", "sequence", "encoded_latent_mean"}),
    ),
    "candidate_occurrences": DataSchema(
        "candidate_occurrences",
        "candidate_occurrences.csv",
        frozenset(
            {
                "run_id",
                "iteration_id",
                "trajectory_id",
                "step_id",
                "candidate_id",
            }
        ),
    ),
    "evaluated_origins": DataSchema(
        "evaluated_origins",
        "evaluated_origins.csv",
        frozenset(
            {
                "run_id",
                "evaluation_iteration_id",
                "candidate_id",
                "trajectory_id",
                "step_id",
            }
        ),
    ),
}


def get_schema(name: str) -> DataSchema:
    """Return a registered schema by logical name."""
    try:
        return SCHEMAS[name]
    except KeyError as error:
        raise KeyError(f"Unknown experiment table: {name}") from error
