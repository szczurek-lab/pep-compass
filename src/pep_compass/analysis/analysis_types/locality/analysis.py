"""Public facade collecting locality experiment analyses by research question."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pep_compass.analysis.analysis_types.locality.latent_jump import (
    ambient_jump,
    latent_jump,
    sorbes_trajectory_profile,
)
from pep_compass.analysis.reader.selection import ExperimentSelection
from pep_compass.analysis.result import AnalysisResult


class LocalityAnalysis:
    """Expose the maintained SORBES and MUTANG locality analyses."""

    def __init__(self, selection: ExperimentSelection, use_cache: bool = True) -> None:
        """Initialize this object with the declared configuration.

        :param selection: Strategy selecting admissible substitutions or candidate rows.
        :type selection: ExperimentSelection
        :param use_cache: Use cache.
        :type use_cache: bool
        :rtype: None
        """
        self.selection = selection
        self.use_cache = use_cache

    def _run(
        self,
        name: str,
        operation: Callable[..., AnalysisResult],
        parameters: dict[str, Any],
    ) -> AnalysisResult:
        """Run.

        :param name: Name identifying this object or recorded artifact.
        :type name: str
        :param operation: Operation.
        :type operation: Callable[..., AnalysisResult]
        :param parameters: Named configuration values consumed by this operation.
        :type parameters: dict[str, Any]
        :return: Result of this operation.
        :rtype: AnalysisResult
        """
        store = self.selection.reader.metrics
        specification = self.selection.specification()
        analysis_version = "8"
        if self.use_cache:
            cached = store.get_analysis(
                name, specification, parameters, analysis_version
            )
            if cached is not None:
                frame, metadata = cached
                return AnalysisResult(frame, metadata, {"cache_hit": True})
        result = operation(self.selection, **parameters)
        store.put_analysis(
            name,
            result.data,
            result.metadata,
            specification,
            parameters,
            analysis_version,
        )
        return result

    def latent_jump(self, **parameters: Any) -> AnalysisResult:
        """Relate edit distance to genuine (re-encoded) latent displacement.

        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Any
        :return: Result of this operation.
        :rtype: AnalysisResult
        """
        return self._run("locality.latent_jump", latent_jump, parameters)

    def ambient_jump(self, **parameters: Any) -> AnalysisResult:
        """Compare latent- vs. ambient-space displacement across HydrAMP conditions.

        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Any
        :return: Result of this operation.
        :rtype: AnalysisResult
        """
        return self._run("locality.ambient_jump", ambient_jump, parameters)

    def sorbes_trajectory_profile(self, **parameters: Any) -> AnalysisResult:
        """Return per-iteration SORBES latent drift from each trajectory origin.

        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Any
        :return: Result of this operation.
        :rtype: AnalysisResult
        """
        return self._run(
            "locality.sorbes_trajectory_profile",
            sorbes_trajectory_profile,
            parameters,
        )
